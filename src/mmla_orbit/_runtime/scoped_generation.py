"""Action-scoped validation for the source-bound scalar task executor.

The original generic API remains unchanged. Prefix construction, model calls,
shapes, masks, positions, reductions, greedy decisions and per-token accounting
are byte-identical to the pinned original generate_batch implementation.
"""
import torch
from mmla_pdsa.full.batched_generation import generate_batch
from mmla_pdsa.full.extensions.hybrid import HybridMMLAModel
from mmla_pdsa.full.outcome_generation import _mark
from task_accounting import TaskAccounting


@torch.no_grad()
def generate_action_batch(model,prompts,*,fast_states=None,memories=None,eos_ids,max_new_tokens=None,
                          read_binding,accounting):
    if type(accounting) is not TaskAccounting:
        raise TypeError('typed role-owned task accounting required')
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
    result=generate_batch(model,prompts,fast_states=fast,memories=memories,eos_ids=eos_ids,
        max_new_tokens=max_new_tokens,on_forward=accounting.forwarded,on_tokens=accounting.tokens)
    check();return result
