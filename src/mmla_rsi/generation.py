"""Algorithm 1 control flow, with isolated proposal state and locked fresh audit."""
from contextlib import contextmanager
from dataclasses import dataclass,asdict
import copy,math,time
from .rules import identity,Rule,parse_rule
from .proposals import SamplingFailure
from .journal import StorageFull,SnapshotTooLarge,JournalIntegrityError,StaleController
from .roles import ProposalLearner
from .evaluation import Allowance,Meter,ResourceExhausted,validate_partitions,admission,TaskInterrupted


@dataclass(frozen=True)
class RunConfig:
    generations: int
    attempts: int
    proposal_seeds: tuple[int,...]
    cost_weight: float
    alpha: float
    minimum_improvement: float
    audit_cost_cap: float
    total: Allowance
    setup: Allowance
    draw: Allowance
    update: Allowance
    closure_seconds: float
    def __post_init__(self):
        if (type(self.generations) is not int or not 1<=self.generations<=32
                or type(self.attempts) is not int or not 2<=self.attempts<=1024
                or len(self.proposal_seeds)!=self.generations
                or any(type(x) is not int or not 0<=x<2**63 for x in self.proposal_seeds)
                or not 0<=self.cost_weight<math.inf or not 0<self.alpha<1
                or not 0<self.minimum_improvement<=1 or not 0<=self.audit_cost_cap<=1
                or not 0<self.closure_seconds<self.total.seconds):raise ValueError('finite run configuration')


def wire_rule(rule):return asdict(rule)
def load_rule(x):return Rule(x['mode'],tuple(x['symbols']),x['language'])


class Controller:
    def __init__(self,*,language,initial_rule,config,factory,evaluator,dev,audits,
                 fit_groups,fit_payloads,journal,fit_contents=()):
        parse_rule(language,initial_rule.mode,initial_rule.symbols)
        if len(audits)!=config.generations:raise ValueError('one fresh audit block per possible generation')
        if len(dev)>65536 or any(len(b)>65536 for b in audits):raise ValueError('finite task inventory')
        validate_partitions(fit_groups,fit_payloads,dev,audits,config.proposal_seeds,fit_contents=fit_contents)
        self.language,self.config,self.factory,self.evaluator=language,config,factory,evaluator
        self.dev,self.audits=tuple(dev),tuple(tuple(x) for x in audits)
        self.journal=journal;self.initial=initial_rule
        if not factory.identity:raise ValueError('fixed role factory identity')
        self.contract=identity(dict(language=language.identity,initial=wire_rule(initial_rule),
            config=asdict(config),factory=factory.identity,evaluator=evaluator.identity,
            dev=[asdict(x) for x in dev],audits=[[asdict(x) for x in b] for b in audits],
            fit_groups=sorted(fit_groups),fit_payloads=sorted(fit_payloads),fit_contents=sorted(fit_contents),
            storage=dict(max_bytes=journal.max_bytes,max_cuts=journal.max_cuts,max_cut_bytes=journal.max_cut_bytes)))
        self.parent=initial_rule;self.generation=0;self.phase='READY_GENERATION'
        self.started=time.time();self.deadline=self.started+config.total.seconds
        self.used=0;self.costs=[];self.edges=[];self.archives=[]
        self.archive=[];self.locked=None;self.audit=None;self.role=None;self.failure=None
        self.versions=[wire_rule(initial_rule)];self.envelopes=[]
        self._bindings=(factory.identity,evaluator.identity)
        self._needs_restore=journal.has_evidence()
        self.persistence_complete=True;self._last_cut=None

    def _check(self):
        if self.phase in ('DONE','STOPPED'):raise ValueError('run already closed')
        if time.time()>self.deadline-self.config.closure_seconds:
            raise ResourceExhausted('closure time is reserved')
        if self.factory.identity!=self._factory_identity or self.evaluator.identity!=self._evaluator_identity:
            raise ValueError('runner identity changed')

    @property
    def _factory_identity(self):return self._bindings[0] if hasattr(self,'_bindings') else self.factory.identity
    @property
    def _evaluator_identity(self):return self._bindings[1] if hasattr(self,'_bindings') else self.evaluator.identity

    @contextmanager
    def _operation(self,name,allowance):
        self._check()
        if (self.config.total.work-self.used<allowance.work or
                self.deadline-self.config.closure_seconds-time.time()<allowance.seconds):
            raise ResourceExhausted('cannot reserve full '+name)
        meter=Meter(allowance,deadline=self.deadline-self.config.closure_seconds)
        try:
            yield meter
            meter.check()
        finally:
            self.used+=meter.used
            self.costs.append(dict(operation=name,generation=self.generation,**meter.receipt()))

    def _block_allowance(self,n):
        a=self.evaluator.task_allowance
        # Assembly/loop overhead is explicit, not taken out of the last task.
        return Allowance(n*a.work,n*a.seconds+self.config.setup.seconds)

    def _reserve_generation(self):
        c=self.config;dev=self._block_allowance(len(self.dev));audit=self._block_allowance(2*len(self.audits[self.generation]))
        work=c.setup.work+c.attempts*(c.draw.work+dev.work)+c.attempts*c.update.work+audit.work
        seconds=c.setup.seconds+c.attempts*(c.draw.seconds+dev.seconds)+c.attempts*c.update.seconds+audit.seconds+c.closure_seconds
        if self.config.total.work-self.used<work or self.deadline-time.time()<seconds:
            raise ResourceExhausted('complete generation, paired audit and closure do not fit')
        self.envelopes.append(dict(generation=self.generation,work=work,seconds=seconds,
                                   audit=asdict(audit),closure_seconds=c.closure_seconds))

    def _make_role(self,meter):
        role=self.factory.create(self.parent,self.generation,self.config.proposal_seeds[self.generation],meter.forward)
        if (type(role) is not ProposalLearner or role.parent!=self.parent or role.limit!=self.config.attempts
                or role.proposer.on_forward!=meter.forward):raise ValueError('exact private proposal learner and metered forward required')
        return role

    def state(self):
        return dict(schema='rsi-controller-v1',contract=self.contract,parent=wire_rule(self.parent),
            generation=self.generation,phase=self.phase,started=self.started,deadline=self.deadline,
            used=self.used,costs=copy.deepcopy(self.costs),edges=copy.deepcopy(self.edges),
            versions=copy.deepcopy(self.versions),archives=copy.deepcopy(self.archives),
            archive=copy.deepcopy(self.archive),locked=copy.deepcopy(self.locked),audit=copy.deepcopy(self.audit),
            failure=copy.deepcopy(self.failure),envelopes=copy.deepcopy(self.envelopes),
            role=None if self.role is None else self.role.export(),
            resumable=not self.phase.startswith('INFLIGHT'),previous_cut=self._last_cut,
            persistence_complete=self.persistence_complete)

    def _save(self):
        self._last_cut=self.journal.save(self.state())
        return self._last_cut

    def _stop(self,exc):
        self.failure=dict(reason=type(exc).__name__,message=str(exc)[:512],failed_phase=self.phase)
        if isinstance(exc,SamplingFailure):self.failure['partial_rollout']=asdict(exc.rollout)
        if self.role is not None:
            # draw may have completed before the operation's final time check.
            # Preserve its probability, masks, read identities and spent RNG.
            if self.role.pending is not None and self.role.pending['rollout'] is not None:
                self.failure['partial_rollout']=asdict(self.role.pending['rollout'])
            self.role.abort_pending(self.failure['reason'])
        self.phase='STOPPED'

    def _incomplete(self,exc):
        prior_interruption=copy.deepcopy(self.failure) if self.failure and self.failure.get('reason')=='TaskInterrupted' else None
        self.persistence_complete=False
        self._stop(exc)
        if prior_interruption is not None:self.failure['control_interruption']=prior_interruption
        # A predictable schema/envelope violation retains a bounded cost/error
        # receipt. It is deliberately NOT a resumable full-state checkpoint.
        last=self.costs[-1] if self.costs else {}
        summary=dict(schema='rsi-incomplete-journal-v1',resumable=False,phase='STOPPED',
            contract=self.contract,previous_cut=self._last_cut,started=self.started,deadline=self.deadline,
            used=self.used,operation_count=len(self.costs),generation=self.generation,
            failure={k:v for k,v in self.failure.items() if k!='partial_rollout'},
            last_operation={k:last[k] for k in ('operation','work','wall_seconds') if k in last},
            full_snapshot_saved=False,persistence_complete=False)
        try:
            with self.journal.transaction(1,expected_cut=self._last_cut):self._last_cut=self.journal.save(summary)
        except (OSError,RuntimeError,StaleController):
            # Actual filesystem failure cannot be repaired by claiming a durable
            # receipt. The last INFLIGHT publication remains non-resumable.
            self.failure['receipt_saved']=False
        else:self.failure['receipt_saved']=True

    def _transaction(self,action,cuts):
        interrupted=None
        try:
            with self.journal.transaction(cuts,expected_cut=self._last_cut):
                try:action()
                except (StaleController,SnapshotTooLarge,StorageFull,OSError,JournalIntegrityError):raise
                except TaskInterrupted as exc:
                    interrupted=exc;self._stop(exc);self._save()
                except Exception as exc:self._stop(exc);self._save()
        except StorageFull as exc:
            # No costly action starts unless its reserved publication fits.
            self._stop(exc)
            try:
                with self.journal.transaction(1,expected_cut=self._last_cut):self._save()
            except (SnapshotTooLarge,StorageFull,OSError,JournalIntegrityError) as failure:self._incomplete(failure)
        except (SnapshotTooLarge,OSError,JournalIntegrityError) as exc:self._incomplete(exc)
        finally:
            if interrupted is not None:raise interrupted


    def _inflight(self,stage):
        self.phase='INFLIGHT_'+stage;self._save()

    def step(self):
        if self._needs_restore:raise ValueError('nonempty journal requires explicit latest-cut restore')
        if self.phase in ('STOPPED','DONE'):raise ValueError('terminal run is read-only')
        # Costly transitions need INFLIGHT + outcome + a final closure reserve;
        # pure transitions need their outcome plus the same closure reserve.
        cuts=3 if self.phase in ('READY_GENERATION','READY_ATTEMPT','READY_AUDIT') else 2
        self._transaction(self._advance,cuts)

    def _advance(self):
        self._check()
        if self.phase=='READY_GENERATION':
            self._reserve_generation();self._inflight('SETUP')
            with self._operation('setup',self.config.setup) as meter:self.role=self._make_role(meter)
            self.archive=[];self.locked=None;self.audit=None
            self.phase='READY_ATTEMPT';self._save();return
        if self.phase=='READY_ATTEMPT':
            self._inflight('ATTEMPT')
            with self._operation('sample',self.config.draw) as meter:
                self.role.proposer.on_forward=meter.forward
                handle,rollout,eligible=self.role.draw()
            item=dict(attempt=self.role.attempt,eligible=eligible,rollout=asdict(rollout),
                      candidate=None if rollout.rule is None else wire_rule(rollout.rule),evaluation=None)
            self.archive.append(item)
            if rollout.rule is None:
                loss,cost,tag=1.,0.,'invalid-or-incomplete'
            else:
                item['evaluation']={}
                with self._operation('development',self._block_allowance(len(self.dev))) as meter:
                    self.evaluator.block(rollout.rule,self.dev,meter=meter,out=item['evaluation'])
                ev=item['evaluation'];cost=ev['mean_cost'];loss=min(1.,ev['mean_loss']+self.config.cost_weight*cost);tag='completed'
            item.update(loss=loss,cost=cost,reward=-loss)
            with self._operation('proposal-feedback',self.config.update) as meter:
                self.role.proposer.on_forward=meter.forward
                self.role.feedback(handle,rollout,loss=loss,cost=cost,tag=tag,on_backward=meter.backward)
            item['adaptation']=copy.deepcopy(self.role.receipts[-1])
            self.phase='READY_SELECTION' if self.role.attempt==self.config.attempts else 'READY_ATTEMPT'
            self._save();return
        if self.phase=='READY_SELECTION':
            legal=[x for x in self.archive if x['eligible'] and x['candidate'] is not None]
            if not legal:self.phase='STOPPED';self.failure={'reason':'NO_ELIGIBLE_SUCCESSOR'};self.role.closed=True;self._save();return
            chosen=min(legal,key=lambda x:(-x['reward'],x['cost'],tuple(x['candidate']['symbols']),x['attempt']))
            # Lock excludes the audit outcomes. Selection only reads dev archive.
            self.locked=dict(candidate=chosen['candidate'],attempt=chosen['attempt'],
                generation=self.generation,parent=self.parent.identity,contract=self.contract,
                selection=[{k:x[k] for k in ('attempt','eligible','candidate','loss','cost','reward')} for x in self.archive])
            self.locked['identity']=identity(self.locked)
            self.phase='READY_AUDIT';self._save();return
        if self.phase=='READY_AUDIT':
            self._inflight('AUDIT')
            candidate=load_rule(self.locked['candidate']);cases=self.audits[self.generation]
            self.audit=dict(lock=self.locked['identity'],parent={},child={})
            with self._operation('paired-audit',self._block_allowance(2*len(cases))) as meter:
                self.evaluator.block(self.parent,cases,meter=meter,out=self.audit['parent'])
                self.evaluator.block(candidate,cases,meter=meter,out=self.audit['child'])
            self.phase='AUDITED';self._save();return
        if self.phase=='AUDITED':
            decision=admission(self.audit['parent'],self.audit['child'],alpha=self.config.alpha,
                generations=self.config.generations,delta=self.config.minimum_improvement,cost_cap=self.config.audit_cost_cap)
            self.audit['decision']=decision
            self.role.closed=True
            self.archives.append(dict(generation=self.generation,parent=wire_rule(self.parent),
                attempts=self.archive,locked=self.locked,audit=self.audit,proposal_state=self.role.export()))
            if not decision['accepted']:
                self.phase='STOPPED';self.failure=dict(reason='ADMISSION_FAILED');self._save();return
            child=load_rule(self.locked['candidate'])
            self.edges.append(dict(version=len(self.versions),parent=self.parent.identity,child=child.identity,
                lock=self.locked['identity'],decision=decision,functional_participation='NOT_MEASURED'))
            self.parent=child;self.versions.append(wire_rule(child));self.generation+=1
            self.role=None;self.archive=[];self.locked=None;self.audit=None
            self.phase='DONE' if self.generation==self.config.generations else 'READY_GENERATION'
            self._save();return
        raise ValueError('inflight/unknown state cannot be replayed')
    def run(self):
        while self.phase not in ('STOPPED','DONE'):self.step()
        return dict(status=self.phase if self.persistence_complete else 'INCOMPLETE',last_rule=wire_rule(self.parent),accepted_edges=len(self.edges),
            failure=self.failure,work=self.used,wall_seconds=time.time()-self.started,
            persistence_complete=self.persistence_complete,
            beneficial_recursive_chain='NOT_CLAIMED_FUNCTIONAL_PROBES_NOT_RUN')

    def restore(self):
        if self.phase!='READY_GENERATION' or self.costs:raise ValueError('fresh controller required')
        source_cut,saved=self.journal.load_bound()
        if (saved.get('schema')!='rsi-controller-v1' or saved['contract']!=self.contract
                or not saved['resumable'] or saved['phase'].startswith('INFLIGHT')):
            raise ValueError('incomplete or differently bound run cannot resume')
        if saved['phase'] in ('STOPPED','DONE'):raise ValueError('terminal archive is read-only; no automatic continuation')
        if saved['phase'] not in ('READY_GENERATION','READY_ATTEMPT','READY_SELECTION','READY_AUDIT','AUDITED'):
            raise ValueError('unknown recovery phase')
        if (type(saved['generation']) is not int or not 0<=saved['generation']<=self.config.generations
                or type(saved['used']) is not int or not 0<=saved['used']<=self.config.total.work
                or saved['deadline']!=saved['started']+self.config.total.seconds
                or len(saved['versions'])!=len(saved['edges'])+1):raise ValueError('invalid budget/version cursor')
        parent=load_rule(saved['parent']);parse_rule(self.language,parent.mode,parent.symbols)
        if wire_rule(parent)!=saved['versions'][-1] or saved['versions'][0]!=wire_rule(self.initial):
            raise ValueError('admitted lineage changed')
        # Preserve absolute start/deadline and all consumed work; recovery is charged.
        for key in ('generation','phase','started','deadline','used','costs','edges','versions','archives','archive','locked','audit','failure','envelopes'):
            setattr(self,key,copy.deepcopy(saved[key]))
        self.parent=parent;self._bindings=(self.factory.identity,self.evaluator.identity)
        self._needs_restore=False;self._last_cut=source_cut
        target_phase=self.phase
        def recover():
            self._inflight('RECOVERY')
            if saved['role'] is not None:
                with self._operation('recovery',self.config.setup) as meter:
                    self.role=self._make_role(meter);self.role.restore(saved['role'])
            self.phase=target_phase;self._save()
        self._transaction(recover,3)
