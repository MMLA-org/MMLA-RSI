"""Task-private RTT/RTMA state, with arrived feedback and separate commits."""
from dataclasses import dataclass,asdict
import copy,uuid
import torch
from .neural import FastState
from .rules import identity
from .roles import components,fast_export
from .evidence import Record,EvidenceMemory
from .updater import apply_update
from .task_policy import TaskSamplingFailure,TaskSamplingInterrupted


@dataclass(frozen=True)
class TaskFeedback:
    action: str
    opportunity: int
    source: str
    loss: float
    tag: str
    resolved: bool


class TaskLearner:
    def __init__(self,policy,language,rule,memory,*,header,source,seed,attempts,update_bounds):
        if type(attempts) is not int or not 1<=attempts<=1024:
            raise ValueError('finite task opportunities include T=1')
        components(language,rule)
        if type(source) is not str or not 0<len(source.encode())<=96:raise ValueError('bounded feedback authority')
        self.policy,self.language,self.rule=policy,language,rule
        self.header,self.source,self.limit,self.bounds=header,source,attempts,update_bounds
        self.memory=EvidenceMemory.restore(memory.export(),count_tokens=memory._count_tokens,expected_contract=memory.contract)
        self._memory_contract=copy.deepcopy(memory.contract)
        self.fast=policy.model.rho.begin(seed)
        self.seed=seed;self.initial_memory=memory.identity;self._bound_policy=policy
        self.contract=self._configuration()
        self._bound_contract=self.contract
        self.attempt=0;self.previous_loss=0.;self.closed=False;self.receipts=[];self.pending=None
        self.numerical_commits=self.memory_commits=0;self.epoch=uuid.uuid4().hex
        self._owned_fast=self.fast.identity;self._owned_memory=self.memory.identity
    def _configuration(self):
        return identity(dict(schema='rsi-task-learner-v1',policy=self.policy.identity,language=self.language.identity,
            rule=self.rule.identity,memory0=self.initial_memory,header=self.header,source=self.source,seed=self.seed,
            attempts=self.limit,bounds=asdict(self.bounds)))
    def _check_binding(self):
        self.policy.check()
        if (self.policy is not self._bound_policy or self.contract!=self._bound_contract
                or self._configuration()!=self._bound_contract):raise ValueError('bound task configuration changed')
    def _check(self):
        self._check_binding()
        self.policy.check();self.policy.model.rho.factors(self.fast)
        if self.closed or self.attempt>=self.limit:raise ValueError('closed task learning problem')
        if self.fast.identity!=self._owned_fast or self.memory.identity!=self._owned_memory:
            raise ValueError('task state changed outside its declared rule')
        if self.memory.contract!=self._memory_contract:raise ValueError('task memory permission changed')
    def draw(self,meter):
        self._check()
        if self.pending is not None:raise ValueError('unconsumed task action')
        self.pending=dict(epoch=self.epoch,attempt=self.attempt,phi=self.fast.identity,memory=self.memory.identity,rollout=None)
        try:
            rollout=self.policy.sample(self.fast,self.memory,self.header,meter=meter)
            self.pending['rollout']=rollout
        except (TaskSamplingFailure,TaskSamplingInterrupted) as exc:self.pending['rollout']=exc.rollout;raise
        finally:self._owned_fast=self.fast.identity
        if rollout.memory_read['memory']!=self.pending['memory']:raise ValueError('task bypassed pinned memory')
        return rollout
    def accept_feedback(self,feedback,*,attempt_cost,meter):
        self._check();p=self.pending
        if (type(feedback) is not TaskFeedback or p is None or p['epoch']!=self.epoch
                or p['rollout'] is None or feedback.action!=p['rollout'].seal
                or p['rollout'].digest()!=feedback.action or feedback.opportunity!=self.attempt
                or feedback.source!=self.source or type(feedback.resolved) is not bool):
            raise ValueError('feedback did not arrive for this issued action and authority')
        # Task failures have incurred action work. They use task-failure, never
        # the proposal-only invalid-or-incomplete tag with zero dev cost.
        if feedback.tag not in ('completed','task-failure'):raise ValueError('task feedback tag')
        Record('',feedback.loss,attempt_cost,feedback.tag,self.source,self.attempt).validate()
        if p['rollout'].stop!='EOS' and (feedback.tag!='task-failure' or feedback.loss!=1 or feedback.resolved):
            raise ValueError('capped task actions remain failed unresolved attempts')
        u,v=components(self.language,self.rule)
        receipt=dict(attempt=self.attempt,rule=self.rule.identity,feedback=asdict(feedback),attempt_cost=attempt_cost,
            pre_phi=self.fast.identity,pre_memory=self.memory.identity,numerical=None,memory=None)
        try:
            self.closed=feedback.resolved or self.attempt+1==self.limit
            if not self.closed:
                if u is not None:
                    g,replay=self.policy.gradient(self.fast,p['rollout'],feedback.loss,meter=meter)
                    meter.charge(1,'numerical_interpreter')
                    updated,numeric=apply_update(self.policy.model.rho,self.fast,g,language=self.language,
                        rule=u,bounds=self.bounds,loss=feedback.loss,previous_loss=self.previous_loss,
                        opportunity=self.attempt,opportunity_cap=self.limit)
                    self.fast=updated;self.numerical_commits+=1;receipt['numerical']=dict(**numeric,replay=replay)
                if v is not None:
                    meter.charge(1+self.memory.bounds.slots*len(v.symbols),'memory_validation_and_action_scores')
                    record=Record(p['rollout'].action,feedback.loss,attempt_cost,feedback.tag,self.source,self.attempt)
                    receipt['memory_payload']=asdict(record)
                    self.memory,managed=self.memory.manage(record,language=self.language,rule=v,opportunity_cap=self.limit)
                    if managed['status']=='COMMIT':self.memory_commits+=1
                    receipt['memory']=managed
            self.previous_loss=feedback.loss;receipt['status']='COMPLETED'
        except BaseException as exc:self.closed=True;receipt.update(status='FAILED',error=type(exc).__name__);raise
        finally:
            self._owned_fast=self.fast.identity;self._owned_memory=self.memory.identity
            receipt.update(post_phi=self.fast.identity,post_memory=self.memory.identity,closed=self.closed)
            self.receipts.append(receipt);self.pending=None;self.attempt+=1
    def abort(self,reason):
        self.closed=True
        if self.pending is not None:
            self.receipts.append(dict(attempt=self.attempt,status='FAILED',error=reason,
                pre_phi=self.pending['phi'],pre_memory=self.pending['memory'],
                post_phi=self.fast.identity,post_memory=self.memory.identity))
            self.attempt+=1;self.pending=None
    def export(self):
        self._check_binding()
        if self.pending is not None:raise ValueError('task action in flight; no automatic replay')
        return dict(schema='rsi-task-learner-v1',contract=self.contract,fast=fast_export(self.fast),
            fast_identity=self.fast.identity,memory=self.memory.export(),attempt=self.attempt,
            previous_loss=self.previous_loss,closed=self.closed,numerical_commits=self.numerical_commits,
            memory_commits=self.memory_commits,receipts=copy.deepcopy(self.receipts),epoch=self.epoch)
    def restore(self,state):
        self._check_binding()
        if (state['schema']!='rsi-task-learner-v1' or state['contract']!=self.contract
                or self.attempt or self.pending is not None):raise ValueError('fresh exact task role required')
        if (type(state['attempt']) is not int or not 0<=state['attempt']<=self.limit
                or type(state['closed']) is not bool
                or (state['attempt']==self.limit and not state['closed'])
                or type(state['previous_loss']) not in (int,float) or not 0<=state['previous_loss']<=1
                or len(state['receipts'])!=state['attempt']):raise ValueError('task recovery cursor/history')
        for name in ('numerical_commits','memory_commits'):
            if type(state[name]) is not int or not 0<=state[name]<=state['attempt']:raise ValueError('commit counter')
        data=state['fast'];rho=self.policy.model.rho
        if (set(data)!=set(self.fast.__dict__) or FastState(**data).identity!=state['fast_identity']
                or type(data['steps']) is not int or data['steps']!=state['numerical_commits'] or data['rho_version']!=int(rho.version)
                or type(data['epoch']) is not int or not 0<=data['epoch']<2**63-2):
            raise ValueError('task fast-state identity/version mismatch')
        for key,ref in [('a',rho.a),('b',rho.b),('momentum_a',rho.a),('momentum_b',rho.b)]:
            x=data[key]
            if x.shape!=ref.shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():
                raise ValueError('invalid private task tensor')
        for key in ('rng_state','initial_rng'):torch.Generator(device='cpu').set_state(data[key])
        if not torch.equal(data['initial_rng'],self.fast.initial_rng):raise ValueError('task seed changed')
        memory=EvidenceMemory.restore(state['memory'],count_tokens=self.memory._count_tokens,
                                     expected_contract=self._memory_contract)
        restored={k:v.to(rho.a.device).detach().clone() if torch.is_tensor(v) and k not in ('rng_state','initial_rng')
                  else copy.deepcopy(v) for k,v in data.items()}
        restored.update(a=restored['a'].requires_grad_(),b=restored['b'].requires_grad_(),
                        rho_identity=rho.identity,epoch=data['epoch']+1,recovery_id=uuid.uuid4().hex)
        self.fast=FastState(**restored);rho.factors(self.fast);self.memory=memory
        for k in ('attempt','previous_loss','closed','numerical_commits','memory_commits','receipts'):
            setattr(self,k,copy.deepcopy(state[k]))
        self._owned_fast=self.fast.identity;self._owned_memory=self.memory.identity;self.epoch=uuid.uuid4().hex
