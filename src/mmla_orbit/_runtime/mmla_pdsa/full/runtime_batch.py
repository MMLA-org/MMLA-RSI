"""Tensor batching of independent live tasks; state transitions stay per task.

The ordered frame driver is unchanged. Only tasks at the same proposal/feedback
boundary share a mixer launch. No task sees another task's M/Phi/feedback.
"""
import time
import torch
from .runtime import Problem,Attempt,Feedback
from .batched_generation import generate_batch
from .batching import collate_sequences,collate_memories,propose_batch


def _owners(problems):
    if not problems or len(problems)>64 or any(type(p) is not Problem for p in problems):
        raise ValueError('finite actual unresolved problems required')
    if len({id(p) for p in problems})!=len(problems) or len({p.problem_id for p in problems})!=len(problems):
        raise ValueError('distinct problem owners required')
    model=problems[0].model
    if any(p.model is not model for p in problems):raise ValueError('one exact common loaded model')
    if len({id(p.memory) for p in problems})!=len(problems) or len({id(p.fast) for p in problems})!=len(problems):
        raise ValueError('episode M and Phi must not alias')
    for p in problems:
        p._ready()
        if p.segment._pending is not None or p.segment._core or p.memory._pending:
            raise ValueError('proposal/feedback requires published quiescent M')
    return model


def _memories(problems):
    memories=[]
    for p in problems:
        prefix,keys,values,tokens=p._reads()
        if prefix:raise ValueError('this production batch uses neural M only')
        memories.append(None if keys is None else (keys,values) if tokens is None else (keys,values,tokens))
    return memories


def propose_many(problems,prompts,*,eos_ids,max_new_tokens=None):
    model=_owners(problems)
    if model.profile.memory_view!='neural' or len(prompts)!=len(problems):raise ValueError('one neural-view prompt per problem')
    limit=model.profile.max_new_tokens if max_new_tokens is None else max_new_tokens
    if type(limit) is not int or not 0<limit<=model.profile.max_new_tokens:raise ValueError('bounded common decoding')
    for p,prompt in zip(problems,prompts):
        if (p.pending is not None or p.attempts>=model.profile.max_attempts or not prompt
                or len(prompt)+limit>model.profile.local_tokens or any(type(x) is not int or x<0 for x in prompt)):
            raise ValueError('ready bounded own proposal required')
    before=time.monotonic();bound=[]
    for p in problems:p.attempts+=1;p.clock+=1
    def on_forward(active,nonpadding,padded):
        for i in active:problems[i].cost['forwards']+=1
    def on_tokens(active,tokens):
        for i in active:problems[i].cost['generated_tokens']+=1
    try:
        memories=_memories(problems)
        bound=[(p.memory.identity,p.fast.identity,p.world_clock,p._reader) for p in problems]
        outputs=generate_batch(model,prompts,fast_states=[p.fast for p in problems],memories=memories,
            eos_ids=eos_ids,max_new_tokens=limit,on_forward=on_forward,on_tokens=on_tokens)
        if bound!=[(p.memory.identity,p.fast.identity,p.world_clock,p._reader) for p in problems]:
            raise RuntimeError('state changed inside pinned proposal batch')
        result=[]
        for p,prompt,output,(memory,phi,world,reader) in zip(problems,prompts,outputs,bound):
            attempt=Attempt(p.attempts,p.problem_id,phi,memory,reader,tuple(prompt),tuple(output),p.clock,world)
            result.append(attempt)
        # Publish only once the whole shared numerical operation succeeded.
        for p,attempt in zip(problems,result):p.pending=attempt
        return tuple(result)
    except Exception as error:
        for p in problems:
            p.failed=True
            p.receipts.append(dict(phase='PROPOSAL',attempt=p.attempts,status='FAILED',error=type(error).__name__,batched=True))
        raise
    finally:
        # Shares sum to the actual common wall duration; forwards/tokens retain
        # exact per-episode attribution. Process/GPU accounting is external.
        elapsed=time.monotonic()-before
        for p in problems:p.cost['wall_seconds']+=elapsed/len(problems)


def feedback_many(problems,feedbacks,*,update):
    """One actual feedback step per task; no memory commit is folded into it."""
    model=_owners(problems)
    if model.profile.memory_view!='neural' or len(feedbacks)!=len(problems):raise ValueError('complete neural feedback batch')
    if set(update)!={'learning_rate','momentum','clip'}:raise ValueError('exact fixed numerical updater')
    for p,f in zip(problems,feedbacks):
        if (type(f) is not Feedback or p.pending is None or f.attempt!=p.pending.identity
                or type(f.arrived_at) is not int or f.arrived_at<=p.pending.clock
                or type(f.source) is not str or not 0<len(f.source.encode())<=256
                or not f.input_ids or len(f.input_ids)!=len(f.labels)
                or len(f.input_ids)>model.profile.local_tokens
                or any(type(x) is not int or x<0 for x in f.input_ids)
                or any(type(x) is not int or (x<0 and x!=-100) for x in f.labels)
                or (p.pending.phi,p.pending.memory,p.pending.world_clock,p.pending.reader)!=(p.fast.identity,p.memory.identity,p.world_clock,p._reader)):
            raise ValueError('own bounded arrived feedback required')
    before=time.monotonic();records=[]
    for p,f in zip(problems,feedbacks):
        records.append(dict(attempt=p.pending.identity,source=f.source,world_clock=p.world_clock,
            phi_before=p.fast.identity,memory_before=p.memory.identity,phi='NO_UPDATE',memory='NULL',
            status='PENDING',updater=dict(update),batched=True))
        p.clock=f.arrived_at
    try:
        memories=_memories(problems);sequences=[];device=model.rho.a.device
        for p,f in zip(problems,feedbacks):
            ids=torch.tensor([f.input_ids],device=device,dtype=torch.long)
            labels=torch.tensor([f.labels],device=device,dtype=torch.long)
            sequences.append((ids,torch.ones_like(ids),labels))
        ids,mask,labels=collate_sequences(sequences,pad_id=1,device=device)
        memory=collate_memories(memories,key_width=model.profile.key_width,value_width=model.profile.neural_width,device=device)
        for p,f in zip(problems,feedbacks):p.cost['forwards']+=1;p.cost['feedback_tokens']+=len(f.input_ids)
        losses=model.losses_batch(ids,mask,labels,fast_states=[p.fast for p in problems],memory=memory)
        successors,stats=propose_batch(model.rho,[p.fast for p in problems],losses,
                                      **update,max_steps=model.profile.max_fast_steps)
        for p,new,stat,record in zip(problems,successors,stats,records):
            p.fast=new;p.cost['backward_updates']+=1
            record.update(phi='COMMITTED',update=stat,status='COMPLETED')
        return tuple(records)
    except Exception as error:
        for p,record in zip(problems,records):
            p.failed=True;record.update(status='FAILED',error=type(error).__name__)
        raise
    finally:
        elapsed=time.monotonic()-before
        for p,record in zip(problems,records):
            record.update(phi_after=p.fast.identity,memory_after=p.memory.identity)
            p.receipts.append(record);p.pending=None;p.cost['wall_seconds']+=elapsed/len(problems)
