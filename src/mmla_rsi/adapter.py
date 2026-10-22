"""A simple HF-compatible causal-model projection for the RSI interfaces.

This adapter uses full recomputation and one episode per call. The frozen mixer
still propagates gradients to its input LARC. No weights are downloaded here.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .neural import LearningResidual


@dataclass(frozen=True)
class AdapterProfile:
    hidden_size: int
    local_tokens: int = 2048
    max_new_tokens: int = 64

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in
               (self.hidden_size, self.local_tokens, self.max_new_tokens)):
            raise ValueError("positive integer model capacities required")
        if self.max_new_tokens >= self.local_tokens:
            raise ValueError("generation must leave room for input")


class CausalLarcAdapter(nn.Module):
    """Wrap a supplied causal LM; memory is read through the explicit prompt."""
    def __init__(self, mixer, profile, *, seed, substrate_identity):
        super().__init__()
        if not substrate_identity:
            raise ValueError("explicit model provenance required")
        embeddings = mixer.get_input_embeddings().weight
        if embeddings.ndim != 2 or embeddings.shape[1] != profile.hidden_size:
            raise ValueError("profile and model hidden size disagree")
        self.profile = profile
        self.substrate_identity = dict(substrate_identity)
        self.mixer = mixer.eval().requires_grad_(False)
        self.rho = LearningResidual(profile.hidden_size, seed=seed).to(
            device=embeddings.device, dtype=torch.float32)
        self.eval()

    def forward_batch(self, input_ids, attention_mask, *, fast_states=None,
                      memory=None, logit_positions=None):
        if (input_ids.ndim != 2 or input_ids.shape != attention_mask.shape
                or input_ids.shape[0] != 1 or not 0 < input_ids.shape[1] <= self.profile.local_tokens
                or not torch.all((attention_mask == 0) | (attention_mask == 1))
                or not attention_mask.any()):
            raise ValueError("one nonempty isolated episode and binary mask required")
        if memory is not None:
            raise ValueError("this adapter reads authoritative memory through the prompt")
        if fast_states is not None and len(fast_states) != 1:
            raise ValueError("one private fast state per episode")
        if logit_positions is not None and (logit_positions.ndim != 1
                or logit_positions.dtype != torch.long
                or (logit_positions < 0).any()
                or (logit_positions >= input_ids.shape[1]).any()):
            raise ValueError("valid one-dimensional logit positions required")
        embedded = self.mixer.get_input_embeddings()(input_ids).detach()
        z = self.rho.inject(embedded, None if fast_states is None else fast_states[0])
        positions = (attention_mask.long().cumsum(-1)-1).clamp_min(0)
        logits = self.mixer(inputs_embeds=z, attention_mask=attention_mask,
                            position_ids=positions, use_cache=False, return_dict=True).logits
        return logits if logit_positions is None else logits[:, logit_positions, :]
