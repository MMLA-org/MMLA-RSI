"""One proposal learner: parent-owned updates, pinned reads and local rollback."""
from dataclasses import asdict
import copy
import uuid
import torch
from ..neural import FastState, tensor_digest
from .rules import OPS, parse_rule, identity
from .evidence import Record, EvidenceMemory
from .updater import apply_update


def components(language, rule):
    """Split a legal joint AST into its existing U and V trees, without evaluation."""
    parse_rule(language,rule.mode,rule.symbols)
    if rule.mode == 'rtt':return rule,None
    if rule.mode == 'rtma':return None,rule
    pos=0
    def tree():
        nonlocal pos
        start=pos;symbol=rule.symbols[pos];pos+=1
        if symbol in OPS:tree();tree()
        return rule.symbols[start:pos]
    numerical=tuple(s for _ in range(3) for s in tree())
    memory=tree()
    return parse_rule(language,'rtt',numerical),parse_rule(language,'rtma',memory)


def fast_export(fast):
    return {k:v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k,v in fast.__dict__.items()}


class ProposalLearner:
    """The archive and audit outcomes are not arguments to this interface."""
    def __init__(self, proposer, parent, memory, *, seed, opportunity_cap, update_bounds):
        if (proposer.parent_identity != parent.identity or proposer.mode != parent.mode
                or type(opportunity_cap) is not int or opportunity_cap<2):
            raise ValueError('bound parent/domain and proposal opportunities required')
        components(proposer.language,parent)
        self.proposer,self.parent=proposer,parent
        self.memory=memory;self._memory_contract=copy.deepcopy(memory.contract)
        self.initial_memory=memory.identity
        self.fast=proposer.model.rho.begin(seed)
        self.initial_factors=tensor_digest({'a':self.fast.a,'b':self.fast.b})
        self.limit,self.bounds=opportunity_cap,update_bounds
        self.attempt=0;self.previous_loss=0.;self.closed=False
        self.numerical_commits=0;self.memory_commits=0
        self.receipts=[];self.pending=None
        self.epoch=uuid.uuid4().hex
        self._owned_fast=self.fast.identity;self._owned_memory=self.memory.identity
        self.contract=identity(dict(sampling=proposer.sampling_identity,parent=parent.identity,
            initial_memory=self.initial_memory,seed=seed,opportunities=opportunity_cap,
            update_bounds=asdict(update_bounds),memory_contract=self._memory_contract))

    def _valid(self):
        if self.closed:raise ValueError('closed proposal problem')
        if self.fast.identity!=self._owned_fast or self.memory.identity!=self._owned_memory:
            raise ValueError('state changed outside its parent-owned transition')
        if self.memory.contract!=self._memory_contract:raise ValueError('memory contract changed')
        self.memory.validate();self.proposer._check_reader()
        self.proposer.model.rho.factors(self.fast)

    def draw(self):
        self._valid()
        if self.pending is not None or self.attempt>=self.limit:raise ValueError('no unconsumed proposal opportunity')
        handle=(self.epoch,self.attempt,self.parent.identity)
        # Only states published by this learner can be read. A commit with no
        # factor change is structurally eligible, not evidence of functional use.
        eligible=(self.numerical_commits>0 if self.parent.mode=='rtt' else
                  self.memory_commits>0 if self.parent.mode=='rtma' else
                  self.numerical_commits>0 and self.memory_commits>0)
        self.pending=dict(handle=handle,eligible=eligible,phi_before=self.fast.identity,
                          memory_before=self.memory.identity,rollout=None)
        try:rollout=self.proposer.sample(self.fast,self.memory)
        finally:self._owned_fast=self.fast.identity
        if rollout.memory_read['memory']!=self.pending['memory_before']:
            raise ValueError('proposal bypassed pinned memory')
        self.pending['rollout']=rollout
        return handle,rollout,eligible

    def feedback(self,handle,rollout,*,loss,cost,tag,on_backward=lambda n:None):
        self._valid()
        if (self.pending is None or tuple(handle)!=self.pending['handle']
                or self.pending['rollout'] is not rollout or rollout.digest()!=rollout.seal):
            raise ValueError('unissued, stale or already-consumed proposal feedback')
        # Validate the common scalar feedback before either adaptation. Memory
        # payload assembly belongs to V and therefore follows a successful U.
        Record('',loss,cost,tag,'feedback',self.attempt).validate()
        feedback=dict(loss=loss,cost=cost,tag=tag)
        if self.attempt+1==self.limit:
            self.receipts.append(dict(attempt=self.attempt,feedback=feedback,last=True))
            self.pending=None;self.attempt+=1;return
        u,v=components(self.proposer.language,self.parent)
        receipt=dict(attempt=self.attempt,parent=self.parent.identity,feedback=feedback,
            numerical=None,memory=None,pre_phi=self.fast.identity,pre_memory=self.memory.identity)
        try:
            if u is not None:
                gradient,replay=self.proposer.gradient(self.fast,rollout,loss,on_backward=on_backward)
                updated,numeric=apply_update(self.proposer.model.rho,self.fast,gradient,
                    language=self.proposer.language,rule=u,bounds=self.bounds,
                    loss=loss,previous_loss=self.previous_loss,opportunity=self.attempt,opportunity_cap=self.limit)
                self.fast=updated;self.numerical_commits+=1
                receipt['numerical']=dict(**numeric,parent_pair=self.parent.identity,replay=replay)
            if v is not None:
                source=self.memory.source_prefix+'proposal/'+str(self.attempt)
                record=Record(' '.join(rollout.symbols),loss,cost,tag,source,self.attempt)
                receipt['memory_payload']=asdict(record)
                record.validate()
                updated,managed=self.memory.manage(record,language=self.proposer.language,
                    rule=v,opportunity_cap=self.limit)
                self.memory=updated
                if managed['status']=='COMMIT':self.memory_commits+=1
                receipt['memory']=dict(**managed,parent_pair=self.parent.identity)
            self.previous_loss=loss
            receipt['status']='COMPLETED'
        except Exception as exc:
            self.closed=True;receipt['status']='FAILED';receipt['error']=type(exc).__name__
            raise
        finally:
            self._owned_fast=self.fast.identity;self._owned_memory=self.memory.identity
            receipt.update(post_phi=self.fast.identity,post_memory=self.memory.identity)
            self.receipts.append(receipt)
            self.pending=None;self.attempt+=1

    def abort_pending(self,reason):
        self.closed=True
        if self.pending is not None:
            self.receipts.append(dict(attempt=self.attempt,status='FAILED',error=reason,
                pre_phi=self.pending['phi_before'],pre_memory=self.pending['memory_before'],
                post_phi=self.fast.identity,post_memory=self.memory.identity))
            self.attempt+=1;self.pending=None

    def export(self):
        if self.pending is not None:raise ValueError('inflight proposal may not be replayed after recovery')
        return dict(schema='rsi-proposal-learner-v1',contract=self.contract,
            fast=fast_export(self.fast),fast_identity=self.fast.identity,
            memory=self.memory.export(),attempt=self.attempt,previous_loss=self.previous_loss,
            numerical_commits=self.numerical_commits,memory_commits=self.memory_commits,
            closed=self.closed,receipts=copy.deepcopy(self.receipts),epoch=self.epoch)

    def restore(self,state):
        """Called on a fresh factory instance bound to the same actual rho/M0."""
        if (state['schema']!='rsi-proposal-learner-v1' or state['contract']!=self.contract
                or self.attempt or self.pending is not None):raise ValueError('fresh exact role required for recovery')
        for key in ('attempt','numerical_commits','memory_commits'):
            if type(state[key]) is not int or not 0<=state[key]<=self.limit:raise ValueError('invalid recovery cursor')
        if (state['numerical_commits']>state['attempt'] or state['memory_commits']>state['attempt']
                or type(state['closed']) is not bool or not 0<=state['previous_loss']<=1
                or len(state['receipts'])!=state['attempt']):raise ValueError('inconsistent recovery history')
        data=state['fast'];rho=self.proposer.model.rho
        if set(data)!=set(self.fast.__dict__) or FastState(**data).identity!=state['fast_identity']:
            raise ValueError('fast snapshot integrity')
        if type(data['steps']) is not int or data['steps']!=state['numerical_commits']:
            raise ValueError('numerical receipt cursor mismatch')
        for key,ref in [('a',rho.a),('b',rho.b),('momentum_a',rho.a),('momentum_b',rho.b)]:
            x=data[key]
            if x.shape!=ref.shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():
                raise ValueError('invalid private fast tensor')
        if data['rho_version']!=int(rho.version):raise ValueError('slow version changed')
        # Factory/model content identity is bound by contract; incarnation may
        # intentionally change after loading the exact weights into a new object.
        for key in ('rng_state','initial_rng'):torch.Generator(device='cpu').set_state(data[key])
        if not torch.equal(data['initial_rng'],self.fast.initial_rng):raise ValueError('initial random tape changed')
        if type(data['epoch']) is not int or not 0<=data['epoch']<2**63-2:raise ValueError('invalid epoch')
        memory=EvidenceMemory.restore(state['memory'],count_tokens=self.memory._count_tokens,
                                      expected_contract=self._memory_contract)
        fresh={k:v.to(rho.a.device).detach().clone() if torch.is_tensor(v) and k not in ('rng_state','initial_rng')
               else copy.deepcopy(v) for k,v in data.items()}
        fresh.update(a=fresh['a'].requires_grad_(),b=fresh['b'].requires_grad_(),
                     rho_identity=rho.identity,epoch=data['epoch']+1,recovery_id=uuid.uuid4().hex)
        fast=FastState(**fresh);rho.factors(fast)
        self.fast,self.memory=fast,memory
        self._owned_fast=fast.identity;self._owned_memory=memory.identity
        for key in ('attempt','previous_loss','numerical_commits','memory_commits','closed','receipts'):
            setattr(self,key,copy.deepcopy(state[key]))
        self.epoch=uuid.uuid4().hex
