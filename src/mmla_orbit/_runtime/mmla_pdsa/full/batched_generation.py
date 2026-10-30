"""Greedy generation sharing a mixer launch while retaining episode ownership."""
import torch
from .batching import collate_memories,ReadBatch


@torch.no_grad()
def generate_batch(model,prompts,*,fast_states=None,memories=None,eos_ids,max_new_tokens=None,on_forward=None,on_tokens=None):
    limit=model.profile.max_new_tokens if max_new_tokens is None else max_new_tokens
    if (type(limit) is not int or not 0<limit<=model.profile.max_new_tokens or not prompts
            or not eos_ids or any(type(x) is not int or x<0 for x in eos_ids)):
        raise ValueError('finite generation and explicit EOS set required')
    for prompt in prompts:
        if not prompt or len(prompt)+limit>model.profile.local_tokens or any(type(x) is not int or x<0 for x in prompt):
            raise ValueError('one bounded complete prompt per episode')
    if fast_states is not None and len(fast_states)!=len(prompts):raise ValueError('one fast state per prompt')
    memories=[None]*len(prompts) if memories is None else memories
    if len(memories)!=len(prompts):raise ValueError('one memory per prompt')
    device=model.rho.a.device;sequences=[list(x) for x in prompts];outputs=[[] for _ in prompts]
    # M is pinned for this generation. Collate/validate once, then index active
    # rows; the tensor is not a cache of Phi-dependent mixer outputs.
    all_memory=collate_memories(memories,key_width=model.profile.key_width,
                               value_width=model.profile.neural_width,device=device)
    from .token_memory import prefix_rows
    prefixes=prefix_rows(model.profile,all_memory)
    if max(map(len,prefixes),default=0)+max(map(len,prompts))+limit>model.profile.local_tokens:
        raise ValueError('memory+prompt+generation exceeds common context')
    active=list(range(len(prompts)))
    for step in range(limit):
        width=max(len(sequences[i]) for i in active)
        # Build one rectangular host batch, avoiding a host->GPU transfer per
        # episode at every generated token. Positions retain left-pad semantics.
        rows=[[0]*(width-len(sequences[i]))+sequences[i] for i in active]
        masks=[[0]*(width-len(sequences[i]))+[1]*len(sequences[i]) for i in active]
        ids=torch.tensor(rows,device=device,dtype=torch.long)
        mask=torch.tensor(masks,device=device,dtype=torch.long)
        indices=torch.tensor(active,device=device,dtype=torch.long)
        memory=ReadBatch(*(x.index_select(0,indices) for x in (all_memory.keys,all_memory.values,all_memory.valid)),
            tokens=None if all_memory.tokens is None else all_memory.tokens[active])
        if on_forward is not None:on_forward(tuple(active),sum(len(sequences[i])+len(prefixes[i]) for i in active),len(active)*(width+max(len(prefixes[i]) for i in active)))
        logits=model.forward_batch(ids,mask,fast_states=None if fast_states is None else [fast_states[i] for i in active],memory=memory,logit_positions=torch.tensor([width-1],device=device))
        scores=logits[:,-1]
        decisions=torch.stack((scores.argmax(-1),torch.isfinite(scores).all(-1).long()),-1).tolist()
        if any(valid!=1 for _,valid in decisions):raise FloatingPointError('nonfinite generation logits')
        chosen=[token for token,_ in decisions];following=[]
        if on_tokens is not None:on_tokens(tuple(active),tuple(chosen))
        for i,token in zip(active,chosen):
            outputs[i].append(token);sequences[i].append(token)
            if token not in eos_ids:following.append(i)
        active=following
        if not active:break
    return tuple(tuple(row) for row in outputs)
