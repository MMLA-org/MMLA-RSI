"""Complete original-source timelines with real, batched feedback adaptation.

Trusted task labels are used only after a completed support attempt or for loss.
This adapter never fills production memory from annotations. A failed batch is
retained and aborts the finite stage; no source is silently replaced or dropped.
"""
from dataclasses import dataclass,asdict
import copy,time
import torch
from .profile import digest
from .frames import FrameDriver
from .runtime import Feedback
from .runtime_batch import propose_many,feedback_many
from .episode_adapter import EngineeringTask,supervised_sequence
from .neural import tensor_digest
from .training import Episode,EpisodeFamily,memory_digest

BEHAVIOR=dict(schema='mmla-first-issued-continuation-v1',
    rule='first issued non-NULL action, otherwise NULL; original action order',
    access='current issued actions only; no answer, future, Truth or Phi-dependent scoring')
BEHAVIOR_ID=digest(BEHAVIOR)


def first_issued(problem,context,candidates):
    return (1 if len(candidates)>1 else 0),dict(mode='FIXED_FIRST_ISSUED',policy=BEHAVIOR_ID,predictive=False)


def actual_feature_identity(factory,model):
    return digest(dict(factory=factory.identity,constructor=tensor_digest(model.constructor.state_dict())))


def pinned_memory(problem):
    prefix,keys,values,tokens=problem._reads()
    if prefix:raise ValueError('actual task adapter requires neural M')
    if keys is None:return None
    result=(keys.detach().clone(),values.detach().clone())
    return result if tokens is None else result+(tokens.detach().clone(),)


class TaskExecutionFailure(RuntimeError):
    def __init__(self,batch,cause):
        super().__init__('actual task batch failed in '+batch.phase+': '+type(cause).__name__)
        self.batch,self.cause=batch,cause


@dataclass(frozen=True)
class ExecutedTask:
    task:EngineeringTask
    driver:FrameDriver
    support:tuple
    support_memory:object
    query:tuple
    query_memory:object
    adapted_identity:str
    seed:int
    receipt:dict

    def episode(self):
        return Episode(self.seed,(self.support,),(self.query,),self.task.group,
            support_memories=(self.support_memory,),query_memories=(self.query_memory,),
            actual_adapted_identity=self.adapted_identity,
            actual_trace=digest(self.receipt),execution_receipt=copy.deepcopy(self.receipt))


class ActualTaskBatch:
    """At most 24 source presentations, same timing and one current rho version.

    Compilation/commit is owned per task. Support proposals and numerical
    feedback share genuine mixer batches. All original future source frames are
    consumed after early feedback; late feedback occurs only at source end.
    """
    def __init__(self,tasks,make_driver,*,update,eos_ids=(1,130073),max_new_tokens=64):
        if (not tasks or len(tasks)>24 or any(type(t) is not EngineeringTask for t in tasks)
                or len({t.identity for t in tasks})!=len(tasks)
                or len({t.timing for t in tasks})!=1):raise ValueError('distinct bounded same-timing original tasks')
        if set(update)!={'learning_rate','momentum','clip'}:raise ValueError('fixed complete numerical update')
        if tuple(eos_ids)!=(1,130073):raise ValueError('original MiniCPM EOS identities required')
        self.tasks=tuple(tasks);self.make_driver=make_driver;self.update=dict(update)
        self.eos_ids=tuple(eos_ids);self.max_new_tokens=max_new_tokens
        self.drivers=[];self.executed=[];self.phase='NEW';self.receipt={};self.support_attempts=[]

    def run(self,*,selector=first_issued,observe_decision=None):
        if self.phase!='NEW':raise RuntimeError('a failed or consumed batch is never retried in place')
        started=time.monotonic();support=[];memories=[];seeds=[];rho=None
        try:
            self.phase='EMPTY_SESSIONS'
            for task in self.tasks:
                driver=self.make_driver(task)
                if (type(driver) is not FrameDriver or driver.problem.problem_id!=task.identity
                        or driver.problem.memory.version!=0 or any(driver.problem.memory._rows)):
                    raise ValueError('new empty actual production task required')
                self.drivers.append(driver)
                seeds.append(int(digest(dict(task=task.identity,factory=self.make_driver.identity))[:15],16))
                expected=driver.problem.model.rho.begin(seeds[-1])
                if expected.identity!=driver.problem.fast.identity:raise ValueError('factory independent seed binding changed')
            model=self.drivers[0].problem.model;rho=model.rho.identity
            if any(d.problem.model is not model for d in self.drivers):raise ValueError('one common model/rho batch')
            def frames(index,driver,task):
                observer=None if observe_decision is None else lambda own,point:observe_decision(task,index,own,point)
                driver.consume(driver.compile(task.commands(index)),selector,bootstrap=index==0,observe_decision=observer)
            self.phase='INITIAL_FRAMES'
            for driver,task in zip(self.drivers,self.tasks):frames(0,driver,task)
            if self.tasks[0].timing=='late':
                self.phase='LATE_SOURCE_FRAMES'
                for driver,task in zip(self.drivers,self.tasks):
                    for index in range(1,len(task.row['observations']['segments'])):frames(index,driver,task)
            problems=[d.problem for d in self.drivers]
            self.phase='SUPPORT_PROPOSAL'
            for p,t in zip(problems,self.tasks):p.advance_world_clock(t.support_clock)
            prompts=[t.dataset.formatter.prompt(t.question(t.support_query())) for t in self.tasks]
            attempts=propose_many(problems,prompts,eos_ids=self.eos_ids,max_new_tokens=self.max_new_tokens)
            self.support_attempts=[asdict(attempt) for attempt in attempts]
            self.phase='ARRIVED_FEEDBACK';feedbacks=[]
            for p,t in zip(problems,self.tasks):
                answer=t.release_feedback(p)
                sequence=supervised_sequence(t.dataset.formatter,t.question(t.support_query()),answer,
                    device=model.rho.a.device,max_tokens=model.profile.local_tokens)
                support.append(tuple(x.detach().clone() for x in sequence));memories.append(pinned_memory(p))
                ids,mask,labels=sequence
                feedbacks.append(Feedback(p.pending.identity,'registered-source:'+t.identity,p.clock+1,
                    tuple(ids[0].tolist()),tuple(labels[0].tolist())))
            updates=feedback_many(problems,feedbacks,update=self.update)
            adapted=[p.fast.identity for p in problems]
            if self.tasks[0].timing=='early':
                self.phase='EARLY_SOURCE_FRAMES'
                for driver,task in zip(self.drivers,self.tasks):
                    for index in range(1,len(task.row['observations']['segments'])):frames(index,driver,task)
            self.phase='QUERY_BOUNDARY'
            for i,(driver,task) in enumerate(zip(self.drivers,self.tasks)):
                p=driver.problem;p.advance_world_clock(4)
                if p.fast.identity!=adapted[i]:raise ValueError('continuation altered the registered one-step Phi')
                query=supervised_sequence(task.dataset.formatter,task.question(task.row['template']['query']),
                    task.query_target(),device=model.rho.a.device,max_tokens=model.profile.local_tokens)
                q_memory=pinned_memory(p)
                record=dict(task=task.identity,family=task.group,source=task.view.identity,timing=task.timing,
                    dataset=task.dataset.identity,split=task.view.split,features=actual_feature_identity(self.make_driver,model),
                    rho=rho,phi=p.fast.identity,memory=p.memory.identity,reader=p.reader_identity(),
                    profile=model.profile.identity,substrate=copy.deepcopy(model.substrate_identity),
                    view_reader=None if p.view_reader is None else p.view_reader.identity,
                    source_frames=copy.deepcopy(driver.frame_receipts),support_update=copy.deepcopy(updates[i]),
                    support_sequence=tensor_digest(dict(ids=support[i][0],mask=support[i][1],labels=support[i][2])),
                    query_sequence=tensor_digest(dict(ids=query[0],mask=query[1],labels=query[2])),
                    support_memory=memory_digest(memories[i]),query_memory=memory_digest(q_memory),
                    feedback_order=[t.identity for t in self.tasks],feedback_pad_id=1,
                    reader_parameters=tensor_digest({k:v for k,v in model.state_dict().items()
                        if not k.startswith(('mixer.','rho.','risk.'))}),
                    cost=copy.deepcopy(p.cost),world_clock=p.world_clock)
                self.executed.append(ExecutedTask(task,driver,support[i],memories[i],query,q_memory,adapted[i],seeds[i],record))
            if model.rho.identity!=rho:raise ValueError('rho changed inside actual outer batch')
            self.phase='COMPLETED'
            return tuple(self.executed)
        except Exception as error:
            raise TaskExecutionFailure(self,error) from error
        finally:
            self.receipt=dict(phase=self.phase,status='COMPLETED' if self.phase=='COMPLETED' else 'FAILED',
                tasks=[t.identity for t in self.tasks],rho=rho,update=self.update,
                wall_seconds=time.monotonic()-started,attempted_sources=len(self.drivers),
                completed_sources=len(self.executed),support_attempts=copy.deepcopy(self.support_attempts),costs=[copy.deepcopy(d.problem.cost) for d in self.drivers],
                frames=[copy.deepcopy(d.frame_receipts) for d in self.drivers])


def episode_families(executed,families):
    """Reassemble the same eight actual presentations, never average their state."""
    indexed={x.task.view.identity:x for x in executed}
    expected={v.identity for f in families for v in f.views}
    if len(indexed)!=len(executed) or set(indexed)!=expected:raise ValueError('complete original presentations required')
    return tuple(EpisodeFamily(f.group,tuple(indexed[v.identity].episode() for v in f.views)) for f in families)
