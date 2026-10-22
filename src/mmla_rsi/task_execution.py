"""Trusted task execution bridge; exact payloads, private states, full receipts.

The environment adapter is a reviewed dependency, not a prompt-provided tool.
Its feedback sees only the support payload; final targets are exposed only to
its terminal scorer after learning has closed. A production dataset/backend
binding remains a separate contract, never inferred from engineering fixtures.
"""
from dataclasses import dataclass,asdict
import copy,hashlib,json,math,time
from .rules import identity
from .evaluation import Allowance,Meter,ResourceExhausted,TaskOutcome,TaskRunFailure,TaskInterrupted
from .task_learner import TaskLearner,TaskFeedback
from .task_policy import TaskSamplingFailure,TaskSamplingInterrupted


@dataclass(frozen=True)
class TaskLimits:
    attempts:int
    setup:Allowance
    sample:Allowance
    feedback:Allowance
    adapt:Allowance
    final:Allowance
    def __post_init__(self):
        if type(self.attempts) is not int or not 1<=self.attempts<=1024:raise ValueError('bounded task attempts')
    @property
    def work(self):return self.setup.work+self.attempts*(self.sample.work+self.feedback.work)+(self.attempts-1)*self.adapt.work+self.final.work
    @property
    def seconds(self):return self.setup.seconds+self.attempts*(self.sample.seconds+self.feedback.seconds)+(self.attempts-1)*self.adapt.seconds+self.final.seconds


class TaskPayloads:
    """Sealed JSON input bytes; IDs and physical groups do not replace content."""
    def __init__(self,payloads,*,source_identity):
        if not source_identity or not payloads:raise ValueError('nonempty bound task source')
        self._payloads={};self._contents={};rows={};contents=set()
        for key,raw in payloads.items():
            if type(raw) is not bytes or len(raw)>1048576:raise ValueError('bounded exact payload bytes')
            obj=json.loads(raw)
            if (set(obj)!={'task_id','physical_group','request','support','final'}
                    or obj['task_id']!=key or type(obj['physical_group']) is not str or not obj['physical_group']
                    or type(obj['request']) is not str or not obj['request']):raise ValueError('complete task/group/request/support/final payload')
            # No NaN or Infinity in the declared trusted values.
            json.dumps(obj,allow_nan=False)
            content=identity({k:v for k,v in obj.items() if k not in ('task_id','physical_group')})
            if content in contents:raise ValueError('duplicate task content with changed ID/group')
            contents.add(content);self._contents[key]=content
            h=hashlib.sha256(raw).hexdigest();self._payloads[key]=raw;rows[key]=h
        self.source_identity=source_identity;self.identity=identity(dict(source=source_identity,payloads=rows,contents=self._contents))
    def case(self,task_id,seed):
        from .evaluation import Case
        raw=self._payloads[task_id];obj=json.loads(raw)
        return Case(task_id,obj['physical_group'],hashlib.sha256(raw).hexdigest(),seed,self._contents[task_id])
    def open(self,case):
        raw=self._payloads[case.task_id]
        if hashlib.sha256(raw).hexdigest()!=case.payload_sha256:raise ValueError('actual task bytes mismatch')
        obj=json.loads(raw)
        content=identity({k:v for k,v in obj.items() if k not in ('task_id','physical_group')})
        if case.content_sha256!=content:raise ValueError('task canonical content signature mismatch')
        if obj['task_id']!=case.task_id or obj['physical_group']!=case.physical_group:raise ValueError('task/group mismatch')
        return obj


class TaskExecutor:
    """One reviewed backend, one frozen policy, and fresh episode state per call.

    backend.feedback(action, support, opportunity=..., meter=...) returns
    (loss, resolved, tag). backend.score(action, final, meter=...) returns the
    terminal loss. Neither backend call receives the policy or task learner.
    The prototype is deep-copied once per task; its configuration is bound.
    Trusted implementations must be clone-local, without mutable shared globals
    or external state; that property requires source review. They must charge their own attempted work and have a
    fixed identity; external process timeout is still required for hard bounds.
    """
    def __init__(self,policy,language,*,payloads,backend,memory,limits,update_bounds):
        if not backend.identity or type(backend.configuration) is not dict or memory.reader_identity!=policy.tokenizer_identity:
            raise ValueError('bound trusted backend and task memory reader required')
        self.policy,self.language,self.payloads,self.backend=policy,language,payloads,backend
        self.memory,self.limits,self.bounds=memory,limits,update_bounds
        self._bound_policy=policy;self._bound_backend=backend;self._bound_payloads=payloads
        self._memory0=memory.identity;self._backend_identity=backend.identity;self._payload_identity=payloads.identity
        self.factor_layout=(tuple(policy.model.rho.a.shape),tuple(policy.model.rho.b.shape))
        self.identity=self._configuration();self._bound_configuration=self.identity
    def _configuration(self):
        return identity(dict(schema='rsi-task-executor-v1',policy=self.policy.identity,language=self.language.identity,
            payloads=self.payloads.identity,backend=self.backend.identity,backend_configuration=self.backend.configuration,
            memory0=self.memory.identity,limits=asdict(self.limits),update_bounds=asdict(self.bounds),
            factor_layout=self.factor_layout))
    def run(self,rule,case,meter):
        self.policy.check();cfg=self.limits
        if (self.policy is not self._bound_policy or self.backend is not self._bound_backend
                or self.payloads is not self._bound_payloads or self.identity!=self._bound_configuration
                or self._configuration()!=self._bound_configuration):raise ValueError('task executor configuration changed')
        if (self.memory.identity!=self._memory0 or self.backend.identity!=self._backend_identity
                or self.payloads.identity!=self._payload_identity):raise ValueError('task dependency changed')
        if meter.allowance.work-meter.used<cfg.work or meter.deadline-time.time()<cfg.seconds:
            raise ResourceExhausted('complete task envelope cannot be reserved')
        trace=dict(schema='rsi-task-run-v1',executor=self.identity,case=asdict(case),rule=rule.identity,
                   actions=[],operations=[],state=None,complete=False)
        learner=None;last=None
        def operation(name,allowance,fn):
            local=Meter(allowance,deadline=meter.deadline)
            try:
                result=fn(local);local.check();return result
            finally:
                meter.used+=local.used;meter.events.append((name,local.used))
                trace['operations'].append(dict(name=name,**local.receipt()))
        try:
            def setup(m):
                m.charge(1,'payload_and_private_state')
                obj=self.payloads.open(case)
                role=TaskLearner(self.policy,self.language,rule,self.memory,header=obj['request'],
                    source='task/'+case.payload_sha256,seed=case.seed,attempts=cfg.attempts,update_bounds=self.bounds)
                worker=copy.deepcopy(self.backend)
                if worker is self.backend or worker.identity!=self.backend.identity or worker.configuration!=self.backend.configuration:
                    raise ValueError('independent exact task-local backend required')
                return obj,role,worker
            obj,learner,worker=operation('setup',cfg.setup,setup)
            while not learner.closed:
                before=meter.used
                def draw(m):
                    nonlocal last
                    try:last=learner.draw(m)
                    except (TaskSamplingFailure,TaskSamplingInterrupted) as exc:last=exc.rollout;trace['actions'].append(asdict(last));raise
                    trace['actions'].append(asdict(last));return last
                operation('sample',cfg.sample,draw)
                def feedback(m):
                    m.charge(1,'trusted_feedback_dispatch')
                    if last.stop=='TOKEN_CAP':return (1.,False,'task-failure')
                    return worker.feedback(last.action,copy.deepcopy(obj['support']),opportunity=learner.attempt,meter=m)
                loss,resolved,tag=operation('feedback',cfg.feedback,feedback)
                cost=(meter.used-before)/(cfg.sample.work+cfg.feedback.work)
                f=TaskFeedback(last.seal,learner.attempt,learner.source,loss,tag,resolved)
                # Final feedback closes without U/V; its constant work is part
                # of feedback dispatch. Earlier adaptations have their own quota.
                if resolved or learner.attempt+1==cfg.attempts:
                    learner.accept_feedback(f,attempt_cost=cost,meter=meter)
                else:
                    operation('adapt',cfg.adapt,lambda m:learner.accept_feedback(f,attempt_cost=cost,meter=m))
            def final(m):
                m.charge(1,'terminal_score_dispatch')
                if last.stop!='EOS':return 1.
                return worker.score(last.action,copy.deepcopy(obj['final']),meter=m)
            loss=operation('final',cfg.final,final)
            if type(loss) not in (int,float) or not 0<=loss<=1:raise ValueError('finite terminal task endpoint')
            trace.update(complete=True,loss=float(loss),state=learner.export(),resources=meter.receipt())
            return TaskOutcome(float(loss),trace)
        except BaseException as exc:
            if learner is not None:
                learner.abort(type(exc).__name__);trace['state']=learner.export()
            trace.update(error=type(exc).__name__,resources=meter.receipt())
            if isinstance(exc,Exception):raise TaskRunFailure(trace) from exc
            cause=exc.cause if isinstance(exc,TaskSamplingInterrupted) else exc
            raise TaskInterrupted(trace,cause) from exc
