"""Task-level slow initialization training, with explicit first-order semantics.

The outer batch shares one rho. Fast states, momentum and RNG are independent.
FOMAML-style identity-Jacobian transfer applies only to the residual factors;
there is no second-order gradient through SGD or through discrete generation.
This module does not start a run or authorize access to any dataset.
"""
from dataclasses import dataclass
import copy,random
import torch

from .neural import tensor_digest
from .distributed import EpisodeParallel
from .profile import digest
from .batching import collate_sequences,collate_memories,propose_batch


def optimizer_digest(state, *, tensor_values=True):
    """Full artifact identity by default; structure-only for single-rank agreement.

    The latter still validates/serializes optimizer metadata, but does not copy
    tensor values to the host for a comparison with no other participant.
    Finite-value checks and durable checkpoint identities are separate.
    """
    def encode(x):
        if torch.is_tensor(x):
            if tensor_values:return {'tensor':tensor_digest({'value':x})}
            return {'shape':list(x.shape),'dtype':str(x.dtype)}
        if isinstance(x,dict):return {str(k):encode(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [encode(v) for v in x]
        return x
    return digest(encode(state))


def memory_digest(memory):
    return 'EMPTY' if memory is None else tensor_digest(dict(keys=memory[0],values=memory[1],**({} if len(memory)==2 else {'tokens':memory[2]})))


@dataclass(frozen=True)
class Episode:
    seed: int
    support: tuple
    query: tuple
    group: str
    memory: tuple | None = None
    support_memories: tuple | None = None
    query_memories: tuple | None = None
    actual_adapted_identity: str | None = None
    actual_trace: str | None = None
    execution_receipt: dict | None = None


@dataclass(frozen=True)
class EpisodeFamily:
    """One independent structural family, with actual distinct presentations.

    Leaves retain private fast states. A family contributes the mean of leaf
    losses, regardless of its number of presentations; it is one DP task.
    """
    group: str
    episodes: tuple[Episode, ...]


class SlowTrainer:
    def __init__(self, model, *, learning_rate, objective, inner_steps,
                 inner_rate, inner_momentum=0., inner_clip=1., parallel=None, microbatch=1, pad_id=0, finite_contract=None):
        if objective not in ('static', 'adapted') or type(inner_steps) is not int or not 1 <= inner_steps <= model.profile.max_fast_steps:
            raise ValueError('fixed objective and bounded inner steps required')
        if not 0 < learning_rate <= 1:
            raise ValueError('bounded slow learning rate required')
        if type(microbatch) is not int or not 1 <= microbatch <= 64 or type(pad_id) is not int or pad_id < 0:
            raise ValueError('bounded resource microbatch and pad ID required')
        self.microbatch,self.pad_id=microbatch,pad_id
        self.parallel = parallel or EpisodeParallel()
        self.model, self.objective = model, objective
        self.inner_steps = inner_steps
        self.inner = dict(learning_rate=inner_rate, momentum=inner_momentum, clip=inner_clip,
                          max_steps=model.profile.max_fast_steps)
        # This task trains rho only. New encoder/readout/Q weights require their
        # own declared objectives and are not silently included or called trained.
        # Explicit stage ownership after frontend/memory/G training. Fast leaves
        # still carry their own gradients, while unrelated modules stay frozen.
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(name in ('rho.a','rho.b'));parameter.grad=None
        self.optimizer = torch.optim.AdamW((model.rho.a, model.rho.b), lr=learning_rate, weight_decay=0.)
        self.steps = 0
        self.nonzero_steps=0;self.finite_contract=None;self.contract_identity=None
        if finite_contract is not None:
            if (set(finite_contract)!={'max_steps','global_batch','schedule','data','features','outer_clip','family_sources','schedule_seed'}
                    or type(finite_contract['max_steps']) is not int or not 1<=finite_contract['max_steps']<=4096
                    or finite_contract['global_batch']!=24 or not 0<finite_contract['outer_clip']<=1
                    or objective!='adapted' or inner_steps!=1 or microbatch!=3 or pad_id!=1):
                raise ValueError('finite actual24-family adapted rho contract required')
            for key in ('schedule','data','features'):
                if type(finite_contract[key]) is not str or len(finite_contract[key])!=64:raise ValueError('exact rho data/schedule/features')
                int(finite_contract[key],16)
            members=finite_contract['family_sources'];seed=finite_contract['schedule_seed']
            if (type(members) is not dict or len(members)!=31 or type(seed) is not int
                    or any(type(g) is not str or not g or type(ids) is not list or len(ids)!=8
                        or len(set(ids))!=8 or ids!=sorted(ids) or any(type(x) is not str or not x for x in ids)
                        for g,ids in members.items()) or len({s for ids in members.values() for s in ids})!=248):
                raise ValueError('exact 31-family/248-source finite inventory')
            schedule=digest(dict(schema='mmla-24-family-schedule-v1',dataset=finite_contract['data'],
                members={g:members[g] for g in sorted(members)},seed=seed,global_batch=24,timings=('early','late'),
                timeline_rule='each consecutive pair of updates uses identical selected families, early then late'))
            if schedule!=finite_contract['schedule']:raise ValueError('rho schedule does not bind its source inventory')
            self.finite_contract=dict(copy.deepcopy(finite_contract),kind='mmla-finite-rho-v1',
                learning_rate=learning_rate,objective=objective,inner_steps=inner_steps,inner=dict(self.inner),
                microbatch=microbatch,pad_id=pad_id,profile=model.profile.identity,substrate=copy.deepcopy(model.substrate_identity),
                betas=[.9,.999],eps=1e-8,weight_decay=0.)
            self.contract_identity=digest(self.finite_contract)

    def verify_finite(self):
        c=self.finite_contract
        if c is None:return
        if (digest(c)!=self.contract_identity or self.objective!=c['objective'] or self.inner_steps!=c['inner_steps']
                or self.inner!=c['inner'] or self.microbatch!=c['microbatch'] or self.pad_id!=c['pad_id']
                or self.model.profile.identity!=c['profile'] or self.model.substrate_identity!=c['substrate']
                or type(self.steps) is not int or not 0<=self.steps<=c['max_steps']
                or type(self.nonzero_steps) is not int or not 0<=self.nonzero_steps<=self.steps):
            raise ValueError('finite rho contract changed')
        params=[p for g in self.optimizer.param_groups for p in g['params']]
        if len(params)!=2 or params[0] is not self.model.rho.a or params[1] is not self.model.rho.b:
            raise ValueError('finite rho must own exactly A/B')
        self._check_optimizer(self.optimizer,c)

    @staticmethod
    def _check_optimizer(optimizer,c):
        expected=dict(lr=c['learning_rate'],betas=tuple(c['betas']),eps=c['eps'],weight_decay=0.,amsgrad=False,maximize=False)
        if len(optimizer.param_groups)!=1 or any(optimizer.param_groups[0][k]!=v for k,v in expected.items()):
            raise ValueError('finite rho optimizer settings changed')
        from .module_training import finite_optimizer
        finite_optimizer(optimizer)

    def export_finite(self):
        self.verify_finite()
        if self.finite_contract is None:raise ValueError('not a finite recipe trainer')
        return dict(kind='mmla-finite-rho-v1',contract=copy.deepcopy(self.finite_contract),steps=self.steps,
            next_batch=self.steps,nonzero_steps=self.nonzero_steps,optimizer=self.optimizer.state_dict())

    def validate_finite_state(self,state):
        self.verify_finite()
        if (self.finite_contract is None or set(state)!={'kind','contract','steps','next_batch','nonzero_steps','optimizer'}
                or state['kind']!='mmla-finite-rho-v1' or state['contract']!=self.finite_contract
                or type(state['steps']) is not int or not 0<=state['steps']<=self.finite_contract['max_steps']
                or state['next_batch']!=state['steps'] or type(state['next_batch']) is not int
                or type(state['nonzero_steps']) is not int or not 0<=state['nonzero_steps']<=state['steps']):
            raise ValueError('finite rho resume contract/cursor mismatch')
        candidate=copy.deepcopy(self.optimizer);candidate.load_state_dict(state['optimizer'])
        self._check_optimizer(candidate,self.finite_contract)
        for p in candidate.param_groups[0]['params']:
            slots=candidate.state.get(p,{})
            if state['steps'] and set(slots)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('complete rho Adam slots required')
            if slots and (slots['step'].numel()!=1 or float(slots['step'])!=state['steps']
                    or any(slots[k].shape!=p.shape or slots[k].dtype!=p.dtype for k in ('exp_avg','exp_avg_sq'))):
                raise ValueError('rho Adam slot/cursor mismatch')

    def update(self, episodes):
        device=self.model.rho.a.device
        parameters=(self.model.rho.a,self.model.rho.b)
        error=None
        try:
            self.verify_finite()
            if self.finite_contract is not None and self.steps>=self.finite_contract['max_steps']:
                raise RuntimeError('finite rho steps exhausted')
            groups=[x.group for x in episodes]
            valid=(bool(episodes) or self.parallel.size>1) and len(set(groups))==len(groups)
            leaves=[]
            if self.finite_contract is not None and any(type(t) is not EpisodeFamily or len(t.episodes)!=8
                    or any(e.actual_adapted_identity is None or e.execution_receipt is None for e in t.episodes) for t in episodes):
                raise ValueError('finite rho uses complete eight-source actual feedback families')
            if self.finite_contract is not None:
                c=self.finite_contract
                selected=random.Random(c['schedule_seed']+self.steps//2).sample(sorted(c['family_sources']),24)
                if groups!=selected[self.parallel.rank::self.parallel.size]:
                    raise ValueError('actual rank families differ from this immutable paired schedule step')
                for family in episodes:
                    records=[e.execution_receipt for e in family.episodes]
                    if [r['source'] for r in records]!=c['family_sources'][family.group]:
                        raise ValueError('actual eight sources differ from the frozen family inventory')
                    if any(r['family']!=family.group or r.get('dataset')!=c['data'] or r.get('split')!='fit'
                            or r.get('features')!=c['features'] or r['timing']!=('early','late')[self.steps%2]
                            for r in records):raise ValueError('rho actual source/domain/timing binding mismatch')
            for task in episodes:
                if type(task) is EpisodeFamily:
                    if (type(task.episodes) is not tuple or not 1<=len(task.episodes)<=8
                            or any(type(e) is not Episode or e.group!=task.group for e in task.episodes)
                            or len({e.seed for e in task.episodes})!=len(task.episodes)):
                        raise ValueError('bounded independent presentation states within one semantic family')
                    leaves.extend(task.episodes)
                elif type(task) is Episode:leaves.append(task)
                else:raise ValueError('explicit episode or episode-family required')
            valid=valid and all(type(x.group) is str and x.group and len(x.support)==self.inner_steps and x.query for x in leaves)
            valid=valid and all(not hasattr(p,'main_grad') for p in parameters)
            for episode in leaves:
                tables=[episode.memory]
                for phase in ('support','query'):
                    states=getattr(episode,phase+'_memories')
                    if states is not None:
                        if type(states) is not tuple or len(states)!=len(getattr(episode,phase)):
                            raise ValueError('one causal memory per support/query sequence')
                        tables.extend(states)
                for memory in tables:
                    if memory is not None and (type(memory) is not tuple or len(memory) not in (2,3)
                            or any(not torch.is_tensor(t) or t.requires_grad for t in memory)):
                        raise ValueError('rho-only trainer requires explicitly frozen memory tensors')
            if not valid:raise ValueError('complete independently grouped support/query required')
            if self.parallel.size==1:self.model.validate_state_configuration()
            contract=dict(objective=self.objective,inner_steps=self.inner_steps,inner=self.inner,
                microbatch=self.microbatch,pad_id=self.pad_id,
                steps=self.steps,optimizer=optimizer_digest(self.optimizer.state_dict(),tensor_values=self.parallel.size>1),
                profile=self.model.profile.identity,substrate=self.model.substrate_identity,
                modules=tensor_digest(self.model.new_state_dict()) if self.parallel.size>1 else None,
                finite_contract=self.contract_identity)
            digest(contract)
        except Exception as exc:error=exc
        if not self.parallel.agree(error is None,device=device):
            raise ValueError('invalid outer batch/configuration on at least one rank') from error
        if self.parallel.size>1:self.parallel.verify_parameters(parameters+(self.model.rho.version,))
        self.parallel.verify_batch(groups,contract)
        if self.finite_contract is not None:
            if sum(self.parallel.gather(len(episodes)))!=self.finite_contract['global_batch']:
                raise ValueError('exact finite rho global family count required')
        version=self.model.rho.identity
        before={k:v.detach().clone() for k,v in self.model.rho.state_dict().items()}
        optim_before=copy.deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad(set_to_none=True)
        losses=[];error=None
        try:
            try:
                for start in range(0,len(episodes),self.microbatch):
                    tasks=episodes[start:start+self.microbatch]
                    batch=[];owners=[];family_sizes=[]
                    for owner,task in enumerate(tasks):
                        present=task.episodes if type(task) is EpisodeFamily else (task,)
                        batch.extend(present);owners.extend([owner]*len(present));family_sizes.extend([len(present)]*len(present))
                    actual=[e for e in batch if e.actual_adapted_identity is not None]
                    if actual:
                        if len(actual)!=len(batch) or self.pad_id!=1:raise ValueError('complete original feedback batch with pad1 required')
                        order=tuple(e.execution_receipt['task'] for e in batch)
                        reader_parameters=tensor_digest({k:v for k,v in self.model.state_dict().items()
                            if not k.startswith(('mixer.','rho.','risk.'))})
                        for e in batch:
                            receipt=e.execution_receipt
                            if (type(receipt) is not dict or digest(receipt)!=e.actual_trace
                                    or tuple(receipt['feedback_order'])!=order or receipt['feedback_pad_id']!=self.pad_id
                                    or receipt['reader_parameters']!=reader_parameters
                                    or receipt['profile']!=self.model.profile.identity
                                    or receipt['substrate']!=self.model.substrate_identity
                                    or receipt['reader']!=digest(dict(substrate=self.model.substrate_identity,
                                        profile=self.model.profile.identity,view_reader=receipt['view_reader'],parameters=reader_parameters))
                                    or receipt['support_update']['updater']!={k:self.inner[k] for k in ('learning_rate','momentum','clip')}
                                    or receipt['support_update']['phi_after']!=e.actual_adapted_identity
                                    or len(e.support)!=1 or len(e.query)!=1):
                                raise ValueError('actual feedback batching/reader/update receipt mismatch')
                            for phase in ('support','query'):
                                sequence=getattr(e,phase)[0]
                                state=getattr(e,phase+'_memories')[0]
                                if (receipt[phase+'_sequence']!=tensor_digest(dict(ids=sequence[0],mask=sequence[1],labels=sequence[2]))
                                        or receipt[phase+'_memory']!=memory_digest(state)):
                                    raise ValueError('actual source sequence/memory receipt mismatch')
                    # Static CE reads slow rho directly. Do not clone unused
                    # episode factors/RNG; adapted and provenance checks stay.
                    fast=[self.model.rho.begin(e.seed) for e in batch] if self.objective=='adapted' else [None]*len(batch)
                    def packed(selected,phase,index):
                        def memory(i):
                            states=getattr(batch[i],phase+'_memories')
                            return batch[i].memory if states is None else states[index]
                        return collate_memories([memory(i) for i in selected],
                            key_width=self.model.profile.key_width,value_width=self.model.profile.neural_width,device=device)
                    if self.objective=='adapted':
                        for step in range(self.inner_steps):
                            memory=packed(range(len(batch)),'support',step)
                            ids,mask,labels=collate_sequences([e.support[step] for e in batch],pad_id=self.pad_id,device=device)
                            loss=self.model.losses_batch(ids,mask,labels,fast_states=fast,memory=memory)
                            fast,_=propose_batch(self.model.rho,fast,loss,**self.inner)
                    for episode,state in zip(batch,fast):
                        if (episode.actual_adapted_identity is None)!=(episode.actual_trace is None):
                            raise ValueError('complete actual execution provenance required')
                        if episode.actual_adapted_identity is not None:
                            if self.objective!='adapted' or state.identity!=episode.actual_adapted_identity:
                                raise ValueError('recomputed Phi does not match the actual feedback trajectory')
                            if type(episode.actual_trace) is not str or len(episode.actual_trace)!=64:
                                raise ValueError('actual source trace identity required')
                    batch_losses=[0. for _ in batch]
                    for round_index in range(max(len(e.query) for e in batch)):
                        active=[i for i,e in enumerate(batch) if round_index<len(e.query)]
                        ids,mask,labels=collate_sequences([batch[i].query[round_index] for i in active],pad_id=self.pad_id,device=device)
                        terms=self.model.losses_batch(ids,mask,labels,
                            fast_states=[fast[i] for i in active] if self.objective=='adapted' else None,
                            memory=packed(active,'query',round_index),first_order_outer=self.objective=='adapted')
                        counts=terms.new_tensor([len(batch[i].query)*family_sizes[i] for i in active])
                        weighted=terms/counts
                        if not torch.isfinite(weighted).all():raise FloatingPointError('nonfinite query')
                        weighted.sum().backward()
                        for i,v in zip(active,weighted.detach().tolist()):batch_losses[i]+=v
                    family_losses=[0. for _ in tasks]
                    for owner,value in zip(owners,batch_losses):family_losses[owner]+=value
                    losses.extend(family_losses)
                    if self.model.rho.identity!=version:raise RuntimeError('rho changed within outer batch')
                if episodes and any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('missing/nonfinite slow task gradient')
            except Exception as exc:
                error=exc
            if not self.parallel.agree(error is None,device=device):
                raise RuntimeError('outer batch aborted on at least one rank') from error
            total_tasks=self.parallel.reduce_task_sums(parameters,local_tasks=len(episodes))
            if not all(torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('nonfinite reduced gradient')
            if int(self.model.rho.version)>=2**63-1:raise OverflowError('rho version exhausted')
            error=None
            try:
                norm=None
                if self.finite_contract is not None:
                    norm=torch.nn.utils.clip_grad_norm_(parameters,self.finite_contract['outer_clip'],error_if_nonfinite=True)
                    if float(norm)==0 and self.nonzero_steps==0:raise FloatingPointError('finite rho has no task signal')
                self.optimizer.step()
                if self.finite_contract is not None:self._check_optimizer(self.optimizer,self.finite_contract)
                if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError('nonfinite slow successor')
            except Exception as exc:
                error=exc
            if not self.parallel.agree(error is None,device=device):
                raise RuntimeError('optimizer commit aborted on at least one rank') from error
            if self.parallel.size>1:self.parallel.verify_parameters(parameters)
            error=None;optim_digest=None
            try:optim_digest=optimizer_digest(self.optimizer.state_dict(),tensor_values=self.parallel.size>1)
            except Exception as exc:error=exc
            if not self.parallel.agree(error is None,device=device):
                raise ValueError('invalid optimizer successor') from error
            after=self.parallel.gather(optim_digest)
            if len(set(after))!=1:raise ValueError('optimizer successors disagree')
            self.model.rho.version.add_(1)
            self.steps+=1
            if self.finite_contract is not None:self.nonzero_steps+=int(float(norm)>0)
            self.model.module_status['rho']='TRAINED_BY_DECLARED_TASK_OBJECTIVE'
        except Exception:
            self.model.rho.load_state_dict(before)
            self.optimizer.load_state_dict(optim_before)
            self.model.rho.invalidate()
            raise
        finally:
            self.optimizer.zero_grad(set_to_none=True)
        return dict(outer_step=self.steps,rho_version=int(self.model.rho.version),
                    objective=self.objective,task_losses=losses,global_tasks=total_tasks,parallel_backend=self.parallel.mode,
                    microbatch=self.microbatch,effective_max_microbatch=min(self.microbatch,len(episodes)),
                    local_presentations=len(leaves),local_families=len(episodes),
                    estimator='FOMAML_STYLE_IDENTITY_JACOBIAN' if self.objective=='adapted' else 'STATIC_CE',
                    rho_tensor_digest=tensor_digest(self.model.rho.state_dict()))
