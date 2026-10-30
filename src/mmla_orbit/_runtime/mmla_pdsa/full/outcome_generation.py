"""Uniform action-local full recomputation; no KV or hybrid prefix cache.

Wrap the byte-preserved original generate_batch with pinned reader/Phi/M
ownership checks. Discrete choices remain greedy; no gradients cross sampling.
"""
import json
import torch
from .batched_generation import generate_batch
from .extensions.hybrid import HybridMMLAModel
GENERATION_ID='outcome-greedy-full-prefix-recompute-v1'


def _tensor(t):
    return (id(t),t._version,str(t.device),str(t.dtype),tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.data_ptr())


def _mark(model,fast,memories):
    names=('eps','variance_epsilon','head_dim','num_key_value_groups','scaling','attention_dropout','is_causal','layer_idx')
    return (id(model),model.profile,model.hybrid_config,model.rho._incarnation,
        json.dumps(model.substrate_identity,sort_keys=True),json.dumps(model.mixer.config.to_dict(),sort_keys=True),
        model.mixer.config._attn_implementation,
        tuple((n,id(m),m.training,tuple((k,getattr(m,k)) for k in names if hasattr(m,k))) for n,m in model.named_modules()),
        tuple((n,_tensor(t)) for n,t in (*model.named_parameters(),*model.named_buffers())),
        None if fast is None else tuple((id(s),s.rho_version,s.steps,s.epoch,s.rho_identity,s.recovery_id,
            *(_tensor(t) for t in (s.a,s.b,s.momentum_a,s.momentum_b))) for s in fast),
        tuple(None if row is None else tuple(_tensor(t) for t in row) for row in memories))


@torch.no_grad()
def generate_action_batch(model,prompts,*,fast_states=None,memories=None,eos_ids,max_new_tokens=None,
                          read_binding,on_forward=None,on_tokens=None):
    if not isinstance(model,HybridMMLAModel) or type(read_binding) is not str or not read_binding:
        raise ValueError('bound complete MMLA reader and read scope required')
    if not prompts or len(prompts)>64:raise ValueError('finite distinct episode batch')
    fast=None if fast_states is None else tuple(fast_states)
    memories=tuple([None]*len(prompts) if memories is None else memories)
    prompts=tuple(tuple(row) for row in prompts);eos_ids=frozenset(eos_ids)
    vocab=model.mixer.config.vocab_size
    if (not eos_ids or any(type(t) is not int or not 0<=t<vocab for t in eos_ids)
            or any(not row or any(type(t) is not int or not 0<=t<vocab for t in row) for row in prompts)
            or len(memories)!=len(prompts) or fast is not None and
            (len(fast)!=len(prompts) or len({id(s) for s in fast})!=len(fast))):
        raise ValueError('one typed legal prompt, memory and private Phi per episode')
    if any(m is not None and (type(m) is not tuple or len(m) not in (2,3) or len(m[0])>model.profile.memory_slots) for m in memories):
        raise ValueError('bounded complete memory rows')
    if fast is not None:
        for s in fast:model.rho.factors(s)
    def mode():
        model.validate_state_configuration()
        if (torch.is_grad_enabled() or torch.is_inference_mode_enabled()
                or any(m.training for m in model.modules())
                or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')):
            raise ValueError('dense generation requires eval/no_grad without autocast or inference mode')
    mode();bound=_mark(model,fast,memories)
    def check():
        mode()
        if _mark(model,fast,memories)!=bound:raise ValueError('reader/Phi/M/configuration changed during dense action')
    def forwarded(*args):
        check()
        if on_forward is not None:on_forward(*args)
        check()
    def tokens(*args):
        check()
        if on_tokens is not None:on_tokens(*args)
        check()
    result=generate_batch(model,prompts,fast_states=fast,memories=memories,eos_ids=eos_ids,
        max_new_tokens=max_new_tokens,on_forward=forwarded,on_tokens=tokens)
    check();return result
