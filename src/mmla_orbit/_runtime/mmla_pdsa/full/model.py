"""Full-vocabulary MiniCPM projection, with explicit memory and Phi arguments.

This implementation is a foundation candidate, not a qualified full-v4 release.
It intentionally does not reuse historical four-program scoring or reset code.
"""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F

from .neural import LearningResidual, CompletedSegmentEncoder, MemoryRead, ActionRisk
from .profile import ArchitectureProfile
from .frontend import ClosedFrontend,tokenize_view
from .constructor import SlotConstructor


class MMLAModel(nn.Module):
    def __init__(self, mixer, profile: ArchitectureProfile, *, seed: int, substrate_identity: dict):
        super().__init__()
        self.profile = profile
        self.substrate_identity = dict(substrate_identity)
        self.mixer = mixer.eval().requires_grad_(False)
        embeddings = mixer.get_input_embeddings()
        if embeddings.weight.shape[1] != profile.hidden_size:
            raise ValueError("mixer hidden width does not match architecture")
        if profile.memory_tokens and embeddings.weight.shape[0]!=profile.memory_vocab_size:
            raise ValueError('token capsule vocabulary differs from frozen substrate')
        device = embeddings.weight.device
        # Isolate all new-module initialization from the host and episode RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rho = LearningResidual(profile.hidden_size, seed=seed)
            self.consolidator = CompletedSegmentEncoder(profile.hidden_size, profile.neural_width)
            self.readout = MemoryRead(profile.hidden_size, profile.key_width, profile.neural_width)
            self.canonical_key = nn.Linear(profile.neural_width, profile.key_width)
            self.canonical_value = nn.Linear(profile.neural_width, profile.neural_width)
            self.risk = ActionRisk(profile.neural_width + profile.key_width + 8)
            self.frontend = ClosedFrontend(profile.hidden_size,width=profile.neural_width,max_events=profile.max_events)
            self.constructor = SlotConstructor(profile.neural_width)
        for module in (self.rho, self.consolidator, self.readout, self.canonical_key, self.canonical_value, self.risk, self.frontend, self.constructor):
            module.to(device=device, dtype=torch.float32)
        self.module_status = dict(rho="INITIALIZED", consolidation="INITIALIZED", readout="INITIALIZED",
                                  canonical_encoder="INITIALIZED", risk="UNTRAINED_UNCALIBRATED",
                                  frontend="INITIALIZED",constructor="INITIALIZED")

    def train(self, mode=True):
        super().train(mode)
        self.mixer.eval()
        return self

    def compile_closed(self, tokenizer, text):
        """Caller must provide a closed observed view; no gold/future argument."""
        view,ids,pool=tokenize_view(tokenizer,text,max_tokens=self.profile.local_tokens)
        device=self.rho.a.device
        with torch.no_grad():embedded=self.mixer.get_input_embeddings()(ids.to(device)).detach()
        result=self.frontend(embedded,pool.to(device))
        return view,result,self.frontend.decode(result,view)

    def outer_parameters(self):
        for name, value in self.named_parameters():
            if not name.startswith("mixer."):
                yield value

    def forward(self, input_ids, attention_mask, *, fast=None, keys=None, values=None, memory_tokens=None, first_order_outer=False):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise ValueError("one isolated episode per model call")
        if input_ids.shape[1] > self.profile.local_tokens or not attention_mask.any():
            raise ValueError("local-context capacity/empty mask")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("binary attention mask required")
        if (keys is None) != (values is None):
            raise ValueError("complete neural memory read required")
        if self.profile.memory_tokens:
            if len({keys is None,values is None,memory_tokens is None})!=1:raise ValueError('complete K/V/token read or empty memory required')
            from .batching import collate_memories,batch_forward
            item=None if keys is None else (keys,values,memory_tokens)
            if keys is not None and memory_tokens is None:raise ValueError('complete token read required')
            memory=collate_memories([item],key_width=self.profile.key_width,value_width=self.profile.neural_width,device=input_ids.device)
            return batch_forward(self,input_ids,attention_mask,fast_states=None if fast is None else [fast],
                memory=memory,first_order_outer=first_order_outer)
        if memory_tokens is not None:raise ValueError('undeclared token memory')
        with torch.no_grad():
            embedded = self.mixer.get_input_embeddings()(input_ids).detach()
        if keys is not None:
            if (keys.ndim != 2 or values.shape != (keys.shape[0], self.profile.neural_width)
                    or keys.shape[1] != self.profile.key_width or keys.shape[0] > self.profile.memory_slots
                    or not torch.isfinite(keys).all() or not torch.isfinite(values).all()):
                raise ValueError("finite bounded typed neural read required")
            embedded = self.readout(embedded, keys, values, top_k=self.profile.top_k)
        z = self.rho.inject(embedded, fast, first_order_outer=first_order_outer)
        positions = (attention_mask.cumsum(-1)-1).clamp_min(0)
        # Weights freeze, not the graph. Recompute explicitly invalidates all
        # input-Phi/M-dependent KV and hidden caches on every call.
        memory = None
        if keys is not None:
            from .batching import collate_memories
            memory = collate_memories([(keys, values)], key_width=self.profile.key_width,
                                     value_width=self.profile.neural_width, device=z.device)
        return self._mixer_logits(z, attention_mask, positions, memory=memory)

    def _mixer_logits(self, z, attention_mask, positions, *, memory=None, logits_to_keep=None):
        extra = {} if logits_to_keep is None else {'logits_to_keep': logits_to_keep}
        return self.mixer(inputs_embeds=z, attention_mask=attention_mask, position_ids=positions,
                          use_cache=False, return_dict=True, **extra).logits

    def forward_batch(self, input_ids, attention_mask, **state):
        from .batching import batch_forward
        return batch_forward(self, input_ids, attention_mask, **state)

    def losses_batch(self, input_ids, attention_mask, labels, **state):
        from .batching import batch_losses
        return batch_losses(self, input_ids, attention_mask, labels, **state)

    def loss(self, input_ids, attention_mask, labels, **state):
        if labels.shape != input_ids.shape or labels[:, 1:].ne(-100).sum() == 0:
            raise ValueError("nonempty causal answer loss required")
        if ((labels != -100) & (attention_mask == 0)).any():
            raise ValueError("padding cannot carry target supervision")
        logits = self(input_ids, attention_mask, **state)
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(self, input_ids, *, fast=None, keys=None, values=None, memory_tokens=None, eos_ids: tuple[int, ...], max_new_tokens=None, on_forward=None, on_token=None):
        limit = self.profile.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(limit) is not int or not 0 < limit <= self.profile.max_new_tokens:
            raise ValueError("generation budget exceeded")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("one complete prompt required")
        if not eos_ids or any(type(x) is not int or x < 0 for x in eos_ids):
            raise ValueError("complete declared termination set required")
        if input_ids.shape[1] + limit > self.profile.local_tokens:
            raise ValueError("prompt plus generation exceeds declared local window; no silent truncation")
        if self.profile.memory_tokens:
            if len({keys is None,values is None,memory_tokens is None})!=1:raise ValueError('complete K/V/token read or empty memory required')
            from .batched_generation import generate_batch
            item=None if keys is None else (keys,values,memory_tokens)
            if keys is not None and memory_tokens is None:raise ValueError('complete token memory')
            result=generate_batch(self,[input_ids[0].tolist()],fast_states=None if fast is None else [fast],memories=[item],
                eos_ids=eos_ids,max_new_tokens=limit,on_forward=None if on_forward is None else lambda *args:on_forward(),
                on_tokens=None if on_token is None else lambda *args:on_token())
            return torch.tensor(result,device=input_ids.device,dtype=torch.long)
        tokens = input_ids.clone()
        for _ in range(limit):
            if on_forward is not None: on_forward()
            logits = self(tokens, torch.ones_like(tokens), fast=fast, keys=keys, values=values,memory_tokens=memory_tokens)
            if not torch.isfinite(logits[:, -1]).all():raise FloatingPointError("nonfinite generation logits")
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if on_token is not None: on_token()
            if int(next_token.item()) in eos_ids:
                break
        return tokens[:, input_ids.shape[1]:]

    def validate_state_configuration(self):
        """Validate model-specific state guards without copying tensor values.

        Derived models with configuration guards extend this method alongside
        their serialization validation. The base model has no such buffers.
        """
        pass

    def new_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith("mixer.")}

    def restore_new_state(self, state):
        expected = self.new_state_dict()
        if set(state) != set(expected):
            raise ValueError("complete new-module state required")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                raise ValueError("invalid new-module tensor: " + name)
        own = self.state_dict()
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value.to(own[name].device))
        self.rho.invalidate()
