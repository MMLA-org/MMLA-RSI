"""Finite, group-aware development and independent locked-candidate audit.

The trusted task executor owns its actual task adapter and fresh task states.
This module exposes neither task payloads nor audit outcomes to the proposer.
"""
from dataclasses import dataclass,asdict
import math
import time
from .rules import identity


class ResourceExhausted(RuntimeError):pass


@dataclass(frozen=True)
class TaskOutcome:
    loss: float
    trace: dict


class TaskRunFailure(RuntimeError):
    def __init__(self,trace):
        super().__init__('task execution failed; bounded trace retained')
        self.trace=trace



class TaskInterrupted(BaseException):
    """Control interruption with retained evidence; never a task-loss result."""
    def __init__(self,trace,cause):
        super().__init__(type(cause).__name__)
        self.trace,self.cause=trace,cause


@dataclass(frozen=True)
class Allowance:
    work: int
    seconds: float
    def __post_init__(self):
        if type(self.work) is not int or self.work<1 or not 0<self.seconds<math.inf:
            raise ValueError('finite positive operation allowance')


class Meter:
    """Cooperative quotas; a production process watchdog must bound blocked calls."""
    def __init__(self,allowance,*,deadline=None):
        self.allowance=allowance;self.used=0;self.started=time.time()
        self.deadline=min(self.started+allowance.seconds,deadline or math.inf)
        self.events=[]
    def check(self):
        if time.time()>self.deadline:raise ResourceExhausted('operation time allowance exhausted')
    def charge(self,amount,kind):
        if type(amount) is not int or amount<0:raise ValueError('integral actual work required')
        self.check()
        if self.used+amount>self.allowance.work:raise ResourceExhausted('operation work allowance exhausted')
        self.used+=amount;self.events.append((kind,amount))
    def forward(self,tokens,positions):self.charge(tokens,'forward_input_tokens')
    def backward(self,tokens):self.charge(tokens,'backward_input_tokens')
    def receipt(self):
        return dict(work=self.used,wall_seconds=time.time()-self.started,events=list(self.events),
                    allowance=asdict(self.allowance))


@dataclass(frozen=True)
class Case:
    task_id: str
    physical_group: str
    payload_sha256: str
    seed: int
    content_sha256: str | None = None
    def __post_init__(self):
        if (not self.task_id or not self.physical_group or len(self.payload_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.payload_sha256)
                or type(self.seed) is not int or not 0<=self.seed<2**63):
            raise ValueError('fixed task/group/content and random tape required')
        if self.content_sha256 is not None and (type(self.content_sha256) is not str or len(self.content_sha256)!=64
                or any(c not in '0123456789abcdef' for c in self.content_sha256)):
            raise ValueError('canonical content signature required when declared')


def validate_partitions(fit_groups,fit_payloads,dev,audits,proposal_seeds,*,fit_contents=()):
    groups=set(fit_groups);payloads=set(fit_payloads);contents=set(fit_contents);ids=set();tapes=set()
    if not dev or not audits or any(not b for b in audits):raise ValueError('complete nonempty blocks required')
    for block in (dev,*audits):
        block_groups=set()
        for case in block:
            if (case.task_id in ids or case.physical_group in groups
                    or case.payload_sha256 in payloads or case.seed in proposal_seeds
                    or (case.content_sha256 is not None and case.content_sha256 in contents)):
                raise ValueError('partition leakage, duplicate content/ID or shared proposal tape')
            ids.add(case.task_id);payloads.add(case.payload_sha256);block_groups.add(case.physical_group)
            if case.content_sha256 is not None:contents.add(case.content_sha256)
            # Within a block each case gets its own tape, reused across rules.
            if case.seed in tapes:raise ValueError('task stream table must use distinct tapes')
            tapes.add(case.seed)
        groups.update(block_groups)
    if len(set(proposal_seeds))!=len(proposal_seeds):raise ValueError('independent generation proposal tapes')


class Evaluator:
    """Executor.run must create fresh private learner state for each call.

    A production executor is bound by a reviewed source/data identity. Returned
    loss is a legal [0,1] terminal endpoint; cost is derived from charged work,
    not an executor-supplied normalized score. Ordinary legal failure returns 1.
    Unexpected exceptions count as task failure, except global budget exhaustion.
    """
    def __init__(self,executor,*,task_allowance):
        if not executor.identity:raise ValueError('fixed task runner identity required')
        self.executor=executor;self.identity=identity(dict(executor=executor.identity,allowance=asdict(task_allowance)))
        self.task_allowance=task_allowance
    def block(self,rule,cases,*,meter,out=None):
        if out is not None and out:raise ValueError('fresh output record required')
        out={} if out is None else out
        out.update(rule=rule.identity,cases=[asdict(x) for x in cases],results=[],complete=False)
        results=out['results']
        for case in cases:
            # Reserve the whole call before entering it, while retaining audit
            # quota separately in the controller's generation envelope.
            if meter.allowance.work-meter.used<self.task_allowance.work:
                raise ResourceExhausted('cannot reserve complete task execution')
            meter.check()
            if meter.deadline-time.time()<self.task_allowance.seconds:
                raise ResourceExhausted('cannot reserve complete task time')
            local=Meter(self.task_allowance,deadline=meter.deadline)
            error=None;loss=1.;trace=None;interrupted=None
            try:
                outcome=self.executor.run(rule,case,local)
                if type(outcome) is TaskOutcome:loss,trace=outcome.loss,outcome.trace
                else:loss=outcome
                local.check()
                if type(loss) not in (int,float) or not 0<=loss<=1:
                    raise ValueError('invalid terminal task loss')
            except TaskInterrupted as exc:
                interrupted=exc;error=type(exc.cause).__name__;trace=exc.trace;loss=1.
            except Exception as exc:
                error=type(exc).__name__;loss=1.
                if type(exc) is TaskRunFailure:trace=exc.trace
            finally:
                # A task-local failure spends what was actually attempted.
                meter.used+=local.used;meter.events.append(('task',local.used))
            results.append(dict(task=case.task_id,group=case.physical_group,
                payload=case.payload_sha256,tape=case.seed,rule=rule.identity,loss=float(loss),
                cost=local.used/self.task_allowance.work,error=error,resources=local.receipt(),trace=trace))
            if interrupted is not None:
                out.update(interrupted=True,interruption=error)
                raise interrupted
            meter.check() # Keep measured partial results, but no global-expiry admission.
        out.update(complete=True,mean_loss=sum(x['loss'] for x in results)/len(results),
            mean_cost=sum(x['cost'] for x in results)/len(results))
        return out


def group_means(block):
    groups={}
    for row in block['results']:groups.setdefault(row['group'],[]).append(row['loss'])
    return {k:sum(v)/len(v) for k,v in sorted(groups.items())}


def admission(parent,child,*,alpha,generations,delta,cost_cap):
    """One-sided Hoeffding bounds: group means, union over two rules and K."""
    if (not parent.get('complete',False) or not child.get('complete',False)
            or parent['cases']!=child['cases'] or len(parent['results'])!=len(child['results'])
            or not 0<alpha<1 or not 0<delta<=1 or not 0<=cost_cap<=1
            or type(generations) is not int or generations<1):raise ValueError('bound complete paired audit required')
    p,c=group_means(parent),group_means(child)
    if not p or set(p)!=set(c):raise ValueError('same independent physical audit units required')
    n=len(p);radius=math.sqrt(math.log(2*generations/alpha)/(2*n))
    parent_mean=sum(p.values())/n;child_mean=sum(c.values())/n
    lower=max(0.,parent_mean-radius);upper=min(1.,child_mean+radius)
    paired=[p[k]-c[k] for k in p]
    resource_ok=max(parent['mean_cost'],child['mean_cost'])<=cost_cap
    return dict(accepted=upper<=lower-delta and resource_ok,
        parent_mean=parent_mean,child_mean=child_mean,parent_lower=lower,child_upper=upper,
        paired_mean_gain=sum(paired)/n,independent_groups=n,alpha=alpha,
        alpha_per_one_sided_bound=alpha/(2*generations),radius=radius,delta=delta,
        cost_cap=cost_cap,resource_ok=resource_ok,
        scope='risk of complete rule/pair; independent-group assumption required; no path-participation claim')
