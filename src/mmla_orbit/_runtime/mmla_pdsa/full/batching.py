"""Actual batched mixer work with private episode states and task-mean losses.

No state is padded by duplicating tasks. Discrete trajectories remain separate;
only independent tensor operations share a GPU launch. This is not MCore DDP.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ReadBatch:
    keys: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    tokens: torch.Tensor | None = None


def collate_sequences(sequences, *, pad_id, device, padding_side='right'):
    """Each sequence is (ids[1,L], mask[1,L], labels[1,L]); retain valid tokens."""
    if not sequences or padding_side not in ('left','right'):
        raise ValueError('nonempty sequences and declared padding required')
    rows=[]
    for ids,mask,labels in sequences:
        if (ids.ndim!=2 or ids.shape[0]!=1 or mask.shape!=ids.shape or labels.shape!=ids.shape
                or not torch.all((mask==0)|(mask==1)) or not mask.any()):
            raise ValueError('one bounded sequence per episode')
        if ((labels!=-100)&(mask==0)).any():raise ValueError('padding carries labels')
        valid=mask[0].bool();positions=valid.nonzero().flatten()
        if len(positions)>1 and not torch.all(positions[1:]-positions[:-1]==1):
            raise ValueError('only contiguous observed sequence; no gap removal')
        x,y=ids[0,valid],labels[0,valid]
        if y[0]!=-100 or not y[1:].ne(-100).any():
            raise ValueError('answer needs a causal predecessor and at least one scored token')
        rows.append((x,y))
    width=max(len(x) for x,y in rows)
    assembly_device='cpu' if all(x.device.type==y.device.type=='cpu' for x,y in rows) else device
    ids=torch.full((len(rows),width),pad_id,device=assembly_device,dtype=torch.long)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for row,(x,y) in enumerate(rows):
        start=0 if padding_side=='right' else width-len(x);end=start+len(x)
        ids[row,start:end]=x.to(assembly_device);labels[row,start:end]=y.to(assembly_device);mask[row,start:end]=1
    return ids.to(device),mask.to(device),labels.to(device)


def collate_memories(memories, *, key_width, value_width, device):
    """List of None or (keys[n,K], values[n,V]); gradients are preserved."""
    if not memories:raise ValueError('one memory per episode required')
    width=max([0]+[len(x[0]) for x in memories if x is not None])
    keys=[];values=[];masks=[];token_rows=[]
    token_width=max([0]+[x[2].shape[1] for x in memories if x is not None and len(x)==3])
    for item in memories:
        if item is None:
            k=torch.empty((0,key_width),device=device);v=torch.empty((0,value_width),device=device)
        else:
            if type(item) is not tuple or len(item) not in (2,3):raise ValueError('explicit complete memory tensors')
            k,v=item[:2]
        if token_width:
            if item is None:tokens=torch.empty((0,token_width),dtype=torch.long)
            elif len(item)!=3:raise ValueError('mixed token access definitions')
            else:tokens=item[2]
            if tokens.dtype!=torch.long or tokens.device.type!='cpu' or tokens.shape!=(len(k),token_width):
                raise ValueError('integer CPU token capsule per row')
            token_rows.append(F.pad(tokens,(0,0,0,width-len(k)),value=-1))
        if (k.ndim!=2 or v.ndim!=2 or k.shape[1]!=key_width or v.shape!=(k.shape[0],value_width)
                or not torch.isfinite(k).all() or not torch.isfinite(v).all()):
            raise ValueError('finite typed memory for each episode')
        n=len(k);keys.append(F.pad(k.to(device),(0,0,0,width-n)))
        values.append(F.pad(v.to(device),(0,0,0,width-n)))
        masks.append(torch.arange(width,device=device)<n)
    return ReadBatch(torch.stack(keys),torch.stack(values),torch.stack(masks),torch.stack(token_rows) if token_width else None)


def read_memory(module, local, memory, *, top_k):
    """Per-episode top-k, including a defined zero delta for empty memories."""
    b,t,_=local.shape;k,v,valid=memory.keys,memory.values,memory.valid
    if (k.ndim!=3 or k.shape[0]!=b or v.shape[:2]!=k.shape[:2]
            or valid.shape!=k.shape[:2] or valid.dtype!=torch.bool
            or k.shape[2]!=module.query.out_features or v.shape[2]!=module.output.in_features
            or not torch.isfinite(k).all() or not torch.isfinite(v).all()):
        raise ValueError('batched memory shape, type or numeric failure')
    if not k.shape[1]:return local
    q=F.normalize(module.query(local.float()),dim=-1)
    scores=torch.einsum('btk,bmk->btm',q,F.normalize(k.float(),dim=-1))
    scores=scores.masked_fill(~valid[:,None,:],-torch.finfo(scores.dtype).max)
    top=scores.topk(min(top_k,k.shape[1]),dim=-1)
    selected=valid[:,None,:].expand(b,t,k.shape[1]).gather(-1,top.indices)
    weights=top.values.softmax(-1)*selected
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)
    rows=torch.arange(b,device=local.device)[:,None,None]
    vectors=v[rows,top.indices]
    retrieved=(vectors*weights[...,None]).sum(-2)
    return local+(torch.sigmoid(module.gate(local.float()))*module.output(retrieved)).to(local.dtype)


def inject_residual(rho, embedded, fast_states=None, *, first_order_outer=False):
    b=embedded.shape[0]
    if fast_states is None:
        if first_order_outer:raise ValueError('first-order transfer requires explicit fast states')
        return rho.inject(embedded)
    if len(fast_states)!=b or len({id(s) for s in fast_states})!=b:
        raise ValueError('one distinct fast state per batch row')
    factors=[rho.factors(s,first_order_outer=first_order_outer) for s in fast_states]
    a=torch.stack([p[0] for p in factors]);bb=torch.stack([p[1] for p in factors])
    hidden=torch.einsum('bth,brh->btr',embedded.float(),a)
    return embedded+torch.einsum('btr,bhr->bth',hidden,bb).to(embedded.dtype)


def batch_forward(model, input_ids, attention_mask, *, fast_states=None, memory=None,
                  first_order_outer=False, logit_positions=None):
    if (input_ids.ndim!=2 or input_ids.shape!=attention_mask.shape or input_ids.shape[0]==0
            or input_ids.shape[1]>model.profile.local_tokens or not torch.all(attention_mask.any(-1))
            or not torch.all((attention_mask==0)|(attention_mask==1))):
        raise ValueError('nonempty bounded independently masked rows required')
    original_width=input_ids.shape[1]
    prefix_width=0
    if memory is not None:
        from .token_memory import prefix_rows
        prefixes=prefix_rows(model.profile,memory);prefix_width=max(map(len,prefixes),default=0)
        if prefix_width:
            if prefix_width+input_ids.shape[1]>model.profile.local_tokens:raise ValueError('memory+prompt+answer capacity exceeded')
            prefix_ids=torch.tensor([[0]*(prefix_width-len(x))+list(x) for x in prefixes],device=input_ids.device,dtype=torch.long)
            prefix_mask=torch.tensor([[0]*(prefix_width-len(x))+[1]*len(x) for x in prefixes],device=input_ids.device,dtype=torch.long)
            input_ids=torch.cat((prefix_ids,input_ids),dim=1);attention_mask=torch.cat((prefix_mask,attention_mask),dim=1)
    with torch.no_grad():embedded=model.mixer.get_input_embeddings()(input_ids).detach()
    if memory is not None:
        if memory.keys.shape[1]>model.profile.memory_slots:raise ValueError('memory capacity exceeded')
        embedded=read_memory(model.readout,embedded,memory,top_k=model.profile.top_k)
    z=inject_residual(model.rho,embedded,fast_states,first_order_outer=first_order_outer)
    positions=(attention_mask.cumsum(-1)-1).clamp_min(0)
    # Restrict the full-vocabulary head to actually scored/generation positions;
    # this changes neither the causal mixer nor the targets. It removes unused
    # B*T*V logits, which otherwise dominate memory on the 130560-token head.
    extra={}
    if logit_positions is not None:
        if (logit_positions.ndim!=1 or not len(logit_positions) or logit_positions.dtype!=torch.long
                or (logit_positions<0).any() or (logit_positions>=original_width).any()
                or not torch.all(logit_positions[1:]>logit_positions[:-1])):
            raise ValueError('unique increasing valid logit positions required')
        extra['logits_to_keep']=logit_positions+prefix_width
    # Frozen mixer weights still propagate to per-row Phi and memory encoders.
    logits=model._mixer_logits(z,attention_mask,positions,memory=memory,**extra)
    return logits if logit_positions is not None else logits[:,prefix_width:]


def batch_losses(model, input_ids, attention_mask, labels, **state):
    if labels.shape!=input_ids.shape:raise ValueError('target shape')
    valid=labels.ne(-100)
    predecessors=torch.zeros_like(valid);predecessors[:,1:]=attention_mask[:,:-1].bool()
    if (valid&(~attention_mask.bool()|~predecessors)).any() or not valid[:,1:].any(-1).all():
        raise ValueError('each scored token needs its own observed causal predecessor')
    scored=valid[:,1:].any(0).nonzero().flatten()
    logits=batch_forward(model,input_ids,attention_mask,logit_positions=scored,**state)
    # Token means are computed per row, never over a mixture of task lengths.
    loss=F.cross_entropy(logits.float().transpose(1,2),labels[:,scored+1],ignore_index=-100,reduction='none')
    return loss.sum(-1)/valid[:,1:].sum(-1)


def propose_batch(rho, fast_states, losses, *, learning_rate, momentum, clip, max_steps):
    """Sum independent losses: averaging here would change each inner step."""
    if (losses.ndim!=1 or len(losses)!=len(fast_states) or not fast_states
            or not torch.isfinite(losses).all() or not 0<learning_rate<=1
            or not 0<=momentum<1 or not 0<clip<float('inf')):
        raise ValueError('finite per-episode losses and bounded update required')
    params=[]
    for s in fast_states:
        rho.factors(s)
        if s.steps>=max_steps:raise RuntimeError('fast-update budget exhausted')
        if not s.a.is_leaf or not s.b.is_leaf:raise ValueError('private fast leaf factors required')
        params.extend((s.a,s.b))
    if len({id(p) for p in params})!=len(params):raise ValueError('fast states alias')
    gradients=torch.autograd.grad(losses.sum(),params)
    successors=[];statistics=[]
    for i,s in enumerate(fast_states):
        ga,gb=gradients[2*i:2*i+2]
        if not torch.isfinite(ga).all() or not torch.isfinite(gb).all():raise FloatingPointError('nonfinite fast gradient')
        norm=torch.sqrt(ga.square().sum()+gb.square().sum())
        scale=(clip/(norm+1e-12)).clamp(max=1.)
        ma=momentum*s.momentum_a+ga*scale;mb=momentum*s.momentum_b+gb*scale
        a=s.a-learning_rate*ma;b=s.b-learning_rate*mb
        if not all(torch.isfinite(x).all() for x in (a,b,ma,mb)):raise FloatingPointError('nonfinite fast successor')
        new=s.fork();new.a=a.detach().requires_grad_();new.b=b.detach().requires_grad_()
        new.momentum_a=ma.detach();new.momentum_b=mb.detach();new.steps+=1
        successors.append(new)
        statistics.append(dict(a_gradient_norm=float(ga.norm()),b_gradient_norm=float(gb.norm()),
            parameter_changed=not(torch.equal(a,s.a) and torch.equal(b,s.b))))
    return successors,statistics
