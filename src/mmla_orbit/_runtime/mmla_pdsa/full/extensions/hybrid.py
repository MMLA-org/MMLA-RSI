"""Explicit hybrid migration: retain every Llama layer and add internal branches.

This is a bounded-prefix recomputation profile, not the report's 12-block
from-scratch example. Every call rebuilds its causal running state from its
own masked prefix; no episode state or KV lives on a shared nn.Module.
"""
from dataclasses import asdict, dataclass
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F

from ..model import MMLAModel
from ..profile import digest


@dataclass(frozen=True)
class HybridConfig:
    schema: str = 'mmla-retained-llama-hybrid-v1'
    original_layers: int = 24
    after_layers: tuple = (3, 7, 11, 15, 19, 23)
    width: int = 128
    heads: int = 4
    local_window: int = 64
    state_mode: str = 'masked-prefix-recompute'

    def __post_init__(self):
        object.__setattr__(self, 'after_layers', tuple(self.after_layers))
        if self.schema != 'mmla-retained-llama-hybrid-v1' or self.state_mode != 'masked-prefix-recompute':
            raise ValueError('registered migration and state semantics required')
        for name, cap in [('original_layers', 128), ('width', 512), ('heads', 32), ('local_window', 2048)]:
            x = getattr(self, name)
            if type(x) is not int or not 1 <= x <= cap:
                raise ValueError('finite hybrid dimension: ' + name)
        if (self.width % self.heads or not self.after_layers
                or any(type(i) is not int or not 0 <= i < self.original_layers for i in self.after_layers)
                or tuple(sorted(set(self.after_layers))) != self.after_layers):
            raise ValueError('unique ordered retained-layer insertion points required')

    @property
    def identity(self):
        return digest(asdict(self))


class HybridBranches(nn.Module):
    """Three computed branches with per-token softmax routing and zero outputs.

    Local attention has a causal token-distance window. The linear branch is
    the recurrence N_t=N_(t-1)+sigmoid(k_t)*tanh(v_t),
    D_t=D_(t-1)+sigmoid(k_t), r_t=N_t/(1+D_t). It is evaluated by prefix sums,
    with 2*width scalar running state per episode. Padding advances neither
    accumulator. Its magnitude is bounded for the declared finite prefix.
    Memory reads only explicit, currently admitted K/V rows of that episode.
    """
    def __init__(self, hidden, key_width, value_width, config):
        super().__init__()
        self.width, self.heads, self.window = config.width, config.heads, config.local_window
        self.norm = nn.LayerNorm(hidden)
        self.local_qkv = nn.Linear(hidden, 3*config.width, bias=False)
        self.linear_kv = nn.Linear(hidden, 2*config.width, bias=False)
        self.memory_query = nn.Linear(hidden, key_width, bias=False)
        self.memory_value = nn.Linear(value_width, config.width, bias=False)
        self.local_output = nn.Linear(config.width, hidden, bias=False)
        self.linear_output = nn.Linear(config.width, hidden, bias=False)
        self.memory_output = nn.Linear(config.width, hidden, bias=False)
        self.route = nn.Linear(hidden, 3)
        for output in (self.local_output, self.linear_output, self.memory_output):
            nn.init.zeros_(output.weight)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)

    def forward(self, hidden, mask, positions, memory):
        b, t, _ = hidden.shape
        x = self.norm(hidden.float())
        valid = mask.bool()
        q, k, v = self.local_qkv(x).chunk(3, dim=-1)
        head = lambda a: a.reshape(b, t, self.heads, self.width//self.heads).transpose(1, 2)
        distance = positions[:, :, None] - positions[:, None, :]
        causal = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        allowed = ((distance >= 0) & (distance < self.window) & causal
                   & valid[:, None, :] & valid[:, :, None])
        local = F.scaled_dot_product_attention(head(q), head(k), head(v),
                    attn_mask=allowed[:, None], dropout_p=0.)
        local = local.transpose(1, 2).reshape(b, t, self.width)
        kk, vv = self.linear_kv(x).chunk(2, dim=-1)
        kk = torch.sigmoid(kk)*valid[..., None]
        numerator = (kk*torch.tanh(vv)).cumsum(dim=1)
        denominator = kk.cumsum(dim=1)
        linear = numerator/(1+denominator)
        # Compute an explicit zero graph for no memory, so every owned parameter
        # has a defined gradient even in an empty-memory training minibatch.
        query = self.memory_query(x)
        if memory is None or memory.keys.shape[1] == 0:
            zero_values = torch.zeros((b, 1, self.memory_value.in_features), device=x.device, dtype=x.dtype)
            retrieved = (self.memory_value(zero_values).expand(b, t, self.width)
                         + query.sum(-1, keepdim=True)*0)*0
        else:
            k, v, mv = memory.keys, memory.values, memory.valid
            if (k.ndim != 3 or k.shape[0] != b or k.shape[2] != query.shape[-1]
                    or v.shape != (b, k.shape[1], self.memory_value.in_features)
                    or mv.shape != k.shape[:2] or mv.dtype != torch.bool):
                raise ValueError('explicit per-episode bounded hybrid memory required')
            scores = torch.einsum('btk,bmk->btm', F.normalize(query, dim=-1), F.normalize(k.float(), dim=-1))
            scores = scores.masked_fill(~mv[:, None, :], -torch.finfo(scores.dtype).max)
            weights = scores.softmax(-1)*mv[:, None, :]
            weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-30)
            retrieved = weights @ self.memory_value(v.float())
        route = self.route(x).softmax(-1)
        delta = (route[..., 0:1]*self.local_output(local)
                 + route[..., 1:2]*self.linear_output(linear)
                 + route[..., 2:3]*self.memory_output(retrieved))
        return hidden + (delta*valid[..., None]).to(hidden.dtype)


class HybridBank(nn.Module):
    def __init__(self, profile, config):
        super().__init__()
        self.register_buffer('configuration', torch.tensor(list(bytes.fromhex(config.identity)), dtype=torch.uint8))
        self.blocks = nn.ModuleDict({str(i): HybridBranches(profile.hidden_size, profile.key_width,
                                   profile.neural_width, config) for i in config.after_layers})


class HybridMMLAModel(MMLAModel):
    """Exact original theta/omega with explicitly owned new hybrid parameters."""
    def __init__(self, mixer, profile, *, seed, substrate_identity, hybrid_config):
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        if type(mixer) is not LlamaForCausalLM or type(hybrid_config) is not HybridConfig:
            raise ValueError('explicit reviewed Llama migration only')
        if ('mmla_hybrid' in substrate_identity or len(mixer.model.layers) != hybrid_config.original_layers
                or mixer.config.num_hidden_layers != hybrid_config.original_layers
                or hybrid_config.local_window > profile.local_tokens
                or mixer.config._attn_implementation not in ('eager', 'sdpa')
                or mixer.model.gradient_checkpointing):
            raise ValueError('exact retained layers, backend and bounded recompute required')
        definition = copy.deepcopy(substrate_identity)
        definition['mmla_hybrid'] = json.loads(json.dumps(asdict(hybrid_config)))
        super().__init__(mixer, profile, seed=seed, substrate_identity=definition)
        self.hybrid_config = hybrid_config
        self._hybrid_identity = hybrid_config.identity
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hybrid = HybridBank(profile, hybrid_config)
        self.hybrid.to(device=self.rho.a.device)
        self.module_status['hybrid'] = 'INITIALIZED_ZERO_OUTPUT'
        self._bind_configuration_buffer()

    def _bind_configuration_buffer(self):
        b = self.hybrid.configuration
        if (b.dtype != torch.uint8 or tuple(b.shape) != (32,)
                or bytes(b.detach().cpu().tolist()).hex() != self._hybrid_identity):
            raise ValueError('configuration bytes differ from declared hybrid')
        self._buffer_binding = (id(b), b._version, b.dtype, tuple(b.shape), b.device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, 'hybrid'):
            self._bind_configuration_buffer()
        return result

    def validate_hybrid(self):
        b = self.hybrid.configuration
        if (type(self.hybrid_config) is not HybridConfig or self.hybrid_config.identity != self._hybrid_identity
                or digest(self.substrate_identity.get('mmla_hybrid')) != self._hybrid_identity
                or (id(b), b._version, b.dtype, tuple(b.shape), b.device) != self._buffer_binding
                or len(self.mixer.model.layers) != self.hybrid_config.original_layers
                or self.mixer.config.num_hidden_layers != self.hybrid_config.original_layers
                or self.mixer.model.gradient_checkpointing
                or self.mixer.config._attn_implementation not in ('eager', 'sdpa')
                or tuple(self.hybrid.blocks) != tuple(map(str, self.hybrid_config.after_layers))):
            raise ValueError('hybrid configuration/ownership changed')
        for block in self.hybrid.blocks.values():
            if (block.width, block.heads, block.window) != (self.hybrid_config.width, self.hybrid_config.heads, self.hybrid_config.local_window):
                raise ValueError('live hybrid branch configuration changed')

    def _mixer_logits(self, z, attention_mask, positions, *, memory=None, logits_to_keep=None):
        from transformers.masking_utils import create_causal_mask
        self.validate_hybrid()
        from ..work_events import record_explicit_mixer
        record_explicit_mixer(self.mixer,attention_mask)
        trunk = self.mixer.model
        causal = create_causal_mask(config=trunk.config, inputs_embeds=z, attention_mask=attention_mask,
                                   past_key_values=None, position_ids=positions)
        rotary = trunk.rotary_emb(z, position_ids=positions)
        h = z
        for i, layer in enumerate(trunk.layers):
            h = layer(h, attention_mask=causal, position_embeddings=rotary, position_ids=positions,
                      past_key_values=None, use_cache=False)
            if str(i) in self.hybrid.blocks:
                h = self.hybrid.blocks[str(i)](h, attention_mask, positions, memory)
        h = trunk.norm(h)
        if logits_to_keep is not None:
            h = h[:, logits_to_keep, :]
        return self.mixer.lm_head(h)

    def validate_state_configuration(self):
        self.validate_hybrid()
        super().validate_state_configuration()

    def new_state_dict(self):
        self.validate_hybrid()
        return super().new_state_dict()

    def restore_new_state(self, state):
        self.validate_hybrid()
        if ('hybrid.configuration' not in state
                or not torch.equal(state['hybrid.configuration'].cpu(), self.hybrid.configuration.cpu())):
            raise ValueError('saved hybrid configuration differs')
        result = super().restore_new_state(state)
        self._bind_configuration_buffer()
        return result

    def initialize_reference_modules(self, reference):
        """Explicit warm start; prior reader qualification does not transfer."""
        if (type(reference) is not MMLAModel or reference.profile != self.profile
                or reference.substrate_identity != {k:v for k,v in self.substrate_identity.items() if k != 'mmla_hybrid'}
                or reference.mixer is not self.mixer):
            raise ValueError('same immutable substrate instance and original profile required')
        source = reference.new_state_dict()
        target = self.new_state_dict()
        expected = {k for k in target if not k.startswith('hybrid.')}
        if set(source) != expected:
            raise ValueError('complete original module initialization required')
        target.update(source)
        self.restore_new_state(target)
        self.rho.initialization = copy.deepcopy(reference.rho.initialization)
        for key in self.module_status:
            if key != 'hybrid':
                self.module_status[key] = 'REFERENCE_INITIALIZATION_NEW_READER_UNQUALIFIED'
