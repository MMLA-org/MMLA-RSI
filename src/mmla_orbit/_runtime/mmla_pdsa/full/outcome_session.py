"""One real task's RTT/RTMA transaction, with a single Phi/M committer.

The driver supplies actual model losses and chooses among issued actions. This
carrier never sees teacher labels, task success, future branch scores or logs.
No action-local KV cache survives an action/feedback boundary.
"""
from dataclasses import dataclass,asdict,replace
import math
import copy
import torch

from .alfworld_episode import ReplayEpisode,VisibleFrame,_digest,_sha
from .alfworld_formatting import ALFWorldFormatter
from .extensions.evidence import EvidenceMemory,Record
from .extensions.updater import apply_update,UpdateBounds
from .outcome_source import source_options,choose_source,source_read


def _clone_evidence(memory):
    return EvidenceMemory.restore(memory.export(),count_tokens=memory._count_tokens,expected_contract=memory.contract)


def _tensor_versions(item):
    if item is None:return None
    return tuple((id(t),t._version,tuple(t.shape),str(t.dtype),str(t.device)) for t in item)


@dataclass
class ActionInput:
    frame: VisibleFrame
    prompt: tuple
    fast: object
    memory: object
    evaluation: EvidenceMemory
    source_read: str
    step: int
    binding: str


class OutcomeSession:
    def __init__(self,model,*,formatter,codec,source,evaluation,reader_identity,seed,
                 tenant,principal,language,updater,manager,bounds,max_updates):
        _sha(reader_identity)
        if (type(formatter) is not ALFWorldFormatter or type(evaluation) is not EvidenceMemory
                or type(bounds) is not UpdateBounds or updater.mode!='rtt' or manager.mode!='rtma'
                or type(max_updates) is not int or not 0<=max_updates<=model.profile.max_fast_steps
                or source.profile.identity!=model.profile.identity or codec.profile.identity!=model.profile.identity
                or source.version or any(source._rows) or evaluation.version or any(evaluation.rows)
                or evaluation.reader_identity!=formatter.identity
                or type(tenant) is not str or not tenant or type(principal) is not str or not principal):
            raise ValueError('new isolated task with bound empty carriers and rules required')
        self.model,self.formatter,self.codec=model,formatter,codec
        self.source,self.evaluation=source,evaluation
        self.reader_identity,self.tenant,self.principal=reader_identity,tenant,principal
        self.language,self.updater,self.manager,self.bounds=language,updater,manager,bounds
        self.max_updates=max_updates;self.fast=model.rho.begin(seed)
        self.step=0;self.previous_loss=0.;self.pending=None;self.failed=False;self.closed=False
        self._versions=self._model_versions();self._profile=model.profile.identity
        self._configuration=_digest(dict(reader=reader_identity,formatter=formatter.identity,codec=codec.identity,
            source_encoder=source.encoder.identity,evaluation=evaluation.contract,language=language.identity,
            updater=updater.identity,manager=manager.identity,bounds=asdict(bounds),max_updates=max_updates,
            tenant=tenant,principal=principal,source_domain='orbit-complete-observation-cyclic-batch-v1'))
        self.receipts=[]

    def _model_versions(self):
        return tuple((name,id(p),p._version,p.requires_grad,str(p.device),str(p.dtype)) for name,p in tuple(self.model.named_parameters())+tuple(self.model.named_buffers()))

    def _check(self):
        if self.failed or self.closed or self._versions!=self._model_versions() or self.model.profile.identity!=self._profile:
            raise RuntimeError('closed/failed session or reader changed within task')
        self.model.rho.factors(self.fast)
        current=_digest(dict(reader=self.reader_identity,formatter=self.formatter.identity,codec=self.codec.identity,
            source_encoder=self.source.encoder.identity,evaluation=self.evaluation.contract,language=self.language.identity,
            updater=self.updater.identity,manager=self.manager.identity,bounds=asdict(self.bounds),max_updates=self.max_updates,
            tenant=self.tenant,principal=self.principal,source_domain='orbit-complete-observation-cyclic-batch-v1'))
        if current!=self._configuration:raise ValueError('task configuration changed')

    def initialize_observation(self,episode,*,write=True):
        """Common reset observation; no evaluator label and no numerical update.

        Fixed start-slot zero gives every enabled M arm the same initialization.
        M-disabled controls choose NULL. The resulting snapshot is its declared
        memory reset baseline; it does not pretend to remove all historical effects.
        """
        self._check()
        if self.step or self.pending or self.receipts or type(episode) is not ReplayEpisode or type(write) is not bool:
            raise ValueError('exactly one original reset episode required')
        checkpoint=episode.checkpoint()
        if checkpoint.steps or episode.terminal:raise ValueError('unresolved original reset required')
        self._origin=checkpoint;self._prefix=();self._visible=episode.visible
        batch=self.codec.split(self._visible.observation,step=0,source_receipt=_digest(asdict(checkpoint)))
        options,excluded=source_options(self.source,batch,self.codec,tenant=self.tenant,principal=self.principal)
        index=next((i for i,x in enumerate(options) if x.action=='WRITE_BATCH_0'),0) if write else 0
        self.source,record=choose_source(self.source,options,index)
        self.receipts.append(dict(kind='INITIAL_SOURCE',source=record,excluded=excluded))
        self.initial_source=self.source.snapshot();self.initial_evaluation=self.evaluation.export()

    def action_input(self,frame):
        self._check()
        if self.pending is not None or not self.receipts or type(frame) is not VisibleFrame or frame!=self._visible:
            raise ValueError('one new action after the preceding transaction finishes')
        evaluation=_clone_evidence(self.evaluation)
        memory,read=source_read(self.source,tenant=self.tenant,principal=self.principal,clock=self.step,
                                device=self.model.rho.a.device)
        fast=self.fast.fork()
        prompt=self.formatter.prompt(frame,evaluation_memory=evaluation)
        seal=_digest(dict(reader=self.reader_identity,step=self.step,frame=asdict(frame),prompt=prompt,
            phi=fast.identity,source=read,evaluation=evaluation.identity))
        result=ActionInput(frame,prompt,fast,memory,evaluation,read,self.step,seal)
        self.pending=dict(input=result,seal=seal,phase='ACTION',state=(self.fast.identity,self.source.identity,self.evaluation.identity),
            tensor_versions=_tensor_versions(memory))
        return result

    def _pending_check(self,phase):
        self._check();p=self.pending
        if p is None or p['phase']!=phase:raise ValueError('wrong action/feedback phase')
        x=p['input']
        actual=_digest(dict(reader=self.reader_identity,step=self.step,frame=asdict(x.frame),prompt=x.prompt,
            phi=x.fast.identity,source=x.source_read,evaluation=x.evaluation.identity))
        if (actual!=p['seal'] or x.binding!=p['seal'] or x.step!=self.step
                or _tensor_versions(x.memory)!=p['tensor_versions']
                or p['state']!=(self.fast.identity,self.source.identity,self.evaluation.identity)):
            raise ValueError('pre-arrival Phi/M/context changed')
        return p

    def execute(self,episode,action,*,generated_tokens):
        p=self._pending_check('ACTION')
        if (type(episode) is not ReplayEpisode or episode.visible!=p['input'].frame
                or episode.checkpoint()!=replace(self._origin,steps=self._prefix)
                or len(self._prefix)!=self.step
                or type(generated_tokens) is not int or not 0<=generated_tokens<=self.model.profile.max_new_tokens):
            raise ValueError('actual current environment and finite completed action required')
        try:
            feedback=episode.step(action)
            self._prefix+= (feedback.receipt,);self._visible=feedback.after
            if episode.checkpoint()!=replace(self._origin,steps=self._prefix):
                raise RuntimeError('actual transition receipt differs from bound episode')
            p.update(phase='FEEDBACK',feedback=feedback,generated_tokens=generated_tokens)
            return feedback
        except BaseException:
            self.failed=True
            raise

    def feedback_text(self):
        p=self._pending_check('FEEDBACK')
        return self.formatter.feedback_supervision(p['feedback'],evaluation_memory=p['input'].evaluation)

    def prepare(self,loss,*,opportunity_cap,_gradient=None,allow_update=True,update_loss=None):
        """Real observation CE is evaluated on pending input.fast and old M.

        The source observation is assembled only after this loss and its gradient
        have been evaluated. The selected successor alone becomes persistent Phi.
        A terminal observation needs no further online adaptation.
        """
        p=self._pending_check('FEEDBACK')
        eligible=allow_update and self.fast.steps<self.max_updates
        if (type(allow_update) is not bool or p['feedback'].terminal or not torch.is_tensor(loss) or loss.ndim!=0
                or not bool(torch.isfinite(loss)) or float(loss.detach())<0
                or type(opportunity_cap) is not int or not self.step<opportunity_cap<=512):
            raise ValueError('finite actual feedback loss for an unresolved bounded task required')
        try:
            bounded=-math.expm1(-float(loss.detach()))
            numerical_loss=loss if update_loss is None else update_loss
            if (not torch.is_tensor(numerical_loss) or numerical_loss.ndim!=0
                    or not bool(torch.isfinite(numerical_loss)) or float(numerical_loss.detach())<0
                    or (eligible and not numerical_loss.requires_grad)):
                raise ValueError('finite numerical feedback loss and eligible gradient required')
            numerical_bounded=-math.expm1(-float(numerical_loss.detach()))
            shadow=None;update_receipt=None
            if eligible:
                fast=p['input'].fast
                gradient=torch.autograd.grad(numerical_loss,(fast.a,fast.b)) if _gradient is None else _gradient
                shadow,update_receipt=apply_update(self.model.rho,fast,gradient,language=self.language,
                    rule=self.updater,bounds=self.bounds,loss=numerical_bounded,previous_loss=self.previous_loss,
                    opportunity=self.step,opportunity_cap=opportunity_cap)
            feedback=p['feedback']
            source_receipt=_digest(asdict(feedback.receipt))
            batch=self.codec.split(feedback.observation_target,step=self.step+1,source_receipt=source_receipt)
            options,excluded=source_options(self.source,batch,self.codec,tenant=self.tenant,principal=self.principal)
            record=Record(action=feedback.action,loss=bounded,cost=p['generated_tokens']/self.model.profile.max_new_tokens,
                tag='completed',source='environment-observation',opportunity=self.step)
            record.validate()
            p.update(phase='PREPARED',shadow=shadow,update_receipt=update_receipt,options=options,excluded=excluded,
                record=record,opportunity_cap=opportunity_cap,bounded_loss=bounded,
                numerical_loss=numerical_bounded,numerical_target='REAL' if update_loss is None else 'DECLARED_INTERVENTION',
                shadow_identity=None if shadow is None else shadow.identity)
            return options,shadow
        except BaseException:
            self.failed=True
            raise

    def commit(self,*,update,source_index,manage_evaluation=True):
        """Sole task-state writer. U prepares; the registered policy chooses.

        Phi and source M are committed first. V then handles its own immutable
        evaluation carrier. A V failure aborts the task without undoing already
        committed Phi/source M or pretending the action had never happened.
        """
        p=self._pending_check('PREPARED')
        if type(update) is not bool or type(manage_evaluation) is not bool or (update and p['shadow'] is None):
            raise ValueError('UPDATE requires an issued numerical successor')
        if p['shadow'] is not None and p['shadow'].identity!=p['shadow_identity']:
            raise ValueError('numerical candidate changed after preparation')
        try:
            source,source_receipt=choose_source(self.source,p['options'],source_index)
            old_phi=self.fast.identity
            if update:self.fast=p['shadow'].fork()
            self.source=source
            self.receipts.append(dict(kind='STATE_COMMIT',step=self.step,update=update,old_phi=old_phi,
                phi=self.fast.identity,source=source_receipt,preparation=p['update_receipt'],excluded=p['excluded']))
            if manage_evaluation:
                self.evaluation,v_receipt=self.evaluation.manage(p['record'],language=self.language,rule=self.manager,
                                                                opportunity_cap=p['opportunity_cap'])
            else:
                v_receipt=dict(action='FROZEN',memory=self.evaluation.identity)
            self.receipts[-1]['numerical_target']=p['numerical_target']
            self.receipts[-1]['evaluation']=v_receipt
            self.previous_loss=p['bounded_loss'];self.step+=1;self.pending=None
            return self.receipts[-1]
        except BaseException:
            self.failed=True
            raise

    def finish_frozen_feedback(self):
        """A declared frozen continuation has no feedback computation/commit."""
        p=self._pending_check('FEEDBACK')
        if p['feedback'].terminal:raise ValueError('use terminal closure')
        self.receipts.append(dict(kind='FROZEN_FEEDBACK',step=self.step,phi=self.fast.identity,
            source=self.source.identity,evaluation=self.evaluation.identity))
        self.step+=1;self.pending=None

    def branch_identity(self):
        self._check()
        p=self.pending
        if p is not None:self._pending_check(p['phase'])
        return _digest(dict(configuration=self._configuration,phi=self.fast.identity,
            initial_rng=self.fast.initial_rng.tolist(),source=self.source.identity,evaluation=self.evaluation.identity,
            step=self.step,previous_loss=self.previous_loss,origin=asdict(self._origin),
            prefix=[asdict(x) for x in self._prefix],visible=asdict(self._visible),receipts=self.receipts,
            initial_source=self.initial_source,initial_evaluation=self.initial_evaluation,
            pending=None if p is None else dict(phase=p['phase'],seal=p['seal'],
                feedback=None if 'feedback' not in p else asdict(p['feedback']),
                shadow=None if p.get('shadow') is None else p['shadow'].identity,
                options=[(x.action,x.predecessor,x.payload,x.successor,x.memory.identity) for x in p.get('options',())],
                record=None if 'record' not in p else asdict(p['record']),
                scalars={k:p.get(k) for k in ('generated_tokens','opportunity_cap','bounded_loss','numerical_loss','numerical_target','shadow_identity')},
                preparation=p.get('update_receipt'),excluded=p.get('excluded'))))

    def fork_feedback(self,episode,*,factory,on_work):
        """Runner-only branch with fresh environment and exact prefix replay.

        This is not a Python security boundary. Selectors receive only numeric
        features; only the trusted label/audit runner receives an episode.
        No pending autograd graph or mutable Phi/M/RNG is shared between forks.
        """
        if (self.pending is None or self.pending['phase'] not in ('FEEDBACK','PREPARED')
                or type(episode) is not ReplayEpisode or episode.terminal
                or episode.checkpoint()!=replace(self._origin,steps=self._prefix)):
            raise ValueError('current unresolved actual feedback anchor required')
        seal=self.branch_identity();new_episode=None
        try:
            new_episode=ReplayEpisode.replay(episode.checkpoint(),factory=factory,on_work=on_work)
            branch=copy.copy(self)
            branch.fast=self.fast.fork();branch.source=self.source.fork();branch.evaluation=_clone_evidence(self.evaluation)
            branch.receipts=copy.deepcopy(self.receipts);branch.initial_source=copy.deepcopy(self.initial_source)
            branch.initial_evaluation=copy.deepcopy(self.initial_evaluation)
            p=dict(self.pending);x=p['input']
            memory=None if x.memory is None else tuple(v.detach().clone() for v in x.memory)
            p['input']=replace(x,fast=x.fast.fork(),memory=memory,evaluation=_clone_evidence(x.evaluation))
            p['tensor_versions']=_tensor_versions(memory)
            if 'shadow' in p:p['shadow']=None if p['shadow'] is None else p['shadow'].fork()
            if 'options' in p:p['options']=tuple(replace(v,memory=v.memory.fork(),receipts=copy.deepcopy(v.receipts)) for v in p['options'])
            if 'update_receipt' in p:p['update_receipt']=copy.deepcopy(p['update_receipt'])
            branch.pending=p;branch._branch_parent=seal
            if branch.branch_identity()!=seal or self.branch_identity()!=seal:
                raise RuntimeError('fork or replay changed the common anchor')
            return branch,new_episode
        except BaseException as exc:
            if new_episode is not None:
                try:new_episode.close()
                except BaseException as cleanup:exc.add_note('branch cleanup failed: '+repr(cleanup))
            raise

    def retention_reset(self,*,phi=False,memory=False):
        """Branch-only declared local intervention, before frozen continuation."""
        self._check()
        if (not hasattr(self,'_branch_parent') or self.pending is not None or self.step<1
                or type(phi) is not bool or type(memory) is not bool):
            raise ValueError('committed branch checkpoint required for retention intervention')
        before=(self.fast.identity,self.source.identity,self.evaluation.identity)
        if phi:self.fast=self.model.rho.reset(self.fast)
        if memory:
            self.source.restore(copy.deepcopy(self.initial_source),rollback=True)
            self.evaluation=EvidenceMemory.restore(copy.deepcopy(self.initial_evaluation),
                count_tokens=self.evaluation._count_tokens,expected_contract=self.evaluation.contract)
        self.receipts.append(dict(kind='LOCAL_RETENTION_RESET',step=self.step,phi_reset=phi,memory_reset=memory,
            before=before,after=(self.fast.identity,self.source.identity,self.evaluation.identity),
            scope='declared checkpoint state retention; not removal of prior cross-state influence'))

    def finish_terminal(self):
        p=self._pending_check('FEEDBACK')
        if not p['feedback'].terminal:raise ValueError('actual terminal task required')
        self.receipts.append(dict(kind='TERMINAL_FEEDBACK',step=self.step,
            source=self.source.identity,phi=self.fast.identity,evaluation=self.evaluation.identity,
            update='NONE_NO_LATER_ACTION'))
        self.step+=1;self.pending=None;self.closed=True


def prepare_feedback_batch(sessions,losses,*,opportunity_cap):
    """Batched inner backward; each private gradient retains its own scale.

    No averaging across episodes: the sum produces each row's own CE gradient.
    This trusted helper receives the actual per-row losses from outcome_losses.
    """
    if (not sessions or len({id(s) for s in sessions})!=len(sessions) or losses.ndim!=1
            or len(losses)!=len(sessions) or not losses.requires_grad or not torch.isfinite(losses).all()):
        raise ValueError('independent real feedback rows required')
    params=[];active=[]
    for i,session in enumerate(sessions):
        p=session._pending_check('FEEDBACK')
        if p['feedback'].terminal:raise ValueError('terminal feedback has no online update')
        if session.fast.steps<session.max_updates:
            active.append(i);params.extend((p['input'].fast.a,p['input'].fast.b))
    gradients=() if not params else torch.autograd.grad(losses[active].sum(),tuple(params))
    mapping={row:gradients[2*j:2*j+2] for j,row in enumerate(active)}
    return tuple(session.prepare(losses[i],opportunity_cap=opportunity_cap,_gradient=mapping.get(i))
                 for i,session in enumerate(sessions))
