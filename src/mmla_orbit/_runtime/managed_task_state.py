"""Shared causal Phi and authoritative-observation M for the new Orbit recipe.

The runner owns environment/verified demonstration provenance. This carrier
receives only the current visible frame, then one arrived observation. It has
no teacher archive, hidden task state, candidate-evaluation memory or KV cache.
"""
from dataclasses import dataclass, asdict
import math
import torch

from mmla_pdsa.full.alfworld_episode import VisibleFrame, _digest, _sha
from mmla_pdsa.full.alfworld_formatting import ALFWorldFormatter
from mmla_pdsa.full.observation_memory import ObservationCodec
from mmla_pdsa.full.outcome_source import source_options, choose_source, source_read
from mmla_pdsa.full.extensions.updater import apply_update, UpdateBounds
from task_source_manager import initialize_values, manage_source, state_identity, DOMAIN


@dataclass(frozen=True)
class ManagedInput:
    frame: VisibleFrame
    prompt: tuple
    fast: object
    memory: object
    source_read: str
    step: int
    binding: str


def _tensor_versions(memory):
    return None if memory is None else tuple((id(x),x._version,tuple(x.shape),str(x.dtype),str(x.device)) for x in memory)


class ManagedTaskState:
    def __init__(self,model,*,formatter,codec,source,reader_identity,seed,
                 tenant,principal,language,updater,manager,bounds,max_updates,horizon):
        _sha(reader_identity)
        if (type(formatter) is not ALFWorldFormatter or type(codec) is not ObservationCodec
                or type(bounds) is not UpdateBounds or updater.mode!='rtt' or manager.mode!='rtma'
                or type(max_updates) is not int or not 0<=max_updates<=model.profile.max_fast_steps
                or type(horizon) is not int or not 1<=horizon<=512
                or source.profile.identity!=model.profile.identity or codec.profile.identity!=model.profile.identity
                or source.version or any(source._rows) or not tenant or not principal):
            raise ValueError('new isolated state with exact observed-memory carrier and bounded rules required')
        self.model,self.formatter,self.codec=model,formatter,codec
        self.source=source;self.values=initialize_values(source)
        self.reader_identity,self.tenant,self.principal=reader_identity,tenant,principal
        self.language,self.updater,self.manager,self.bounds=language,updater,manager,bounds
        self.max_updates,self.horizon=max_updates,horizon
        self.fast=model.rho.begin(seed);self.step=0;self.previous_loss=0.
        self.pending=None;self.frame=None;self.failed=False;self.closed=False;self.receipts=[]
        self._versions=self._model_versions();self._configuration=self._config()

    def _model_versions(self):
        return tuple((n,id(p),p._version,p.requires_grad,str(p.device),str(p.dtype))
            for n,p in tuple(self.model.named_parameters())+tuple(self.model.named_buffers()))

    def _config(self):
        return _digest(dict(reader=self.reader_identity,profile=self.model.profile.identity,
            formatter=self.formatter.identity,codec=self.codec.identity,encoder=self.source.encoder.identity,
            language=self.language.identity,updater=self.updater.identity,manager=self.manager.identity,
            bounds=asdict(self.bounds),max_updates=self.max_updates,horizon=self.horizon,
            tenant=self.tenant,principal=self.principal,domain=DOMAIN,second_evaluation_memory=False))

    def _check(self):
        if self.failed or self.closed or self._versions!=self._model_versions() or self._configuration!=self._config():
            raise RuntimeError('closed/failed state or reader/rules changed within task')
        self.model.rho.factors(self.fast)

    @staticmethod
    def _visible(frame):
        if type(frame) is not VisibleFrame or frame.admissible_commands:
            raise ValueError('current public frame only, no hidden fields/action oracle')

    def initialize(self,frame,*,source_receipt,write=True):
        self._check();self._visible(frame);_sha(source_receipt)
        if self.frame is not None or self.receipts or type(write) is not bool:
            raise ValueError('one common reset observation required')
        batch=self.codec.split(frame.observation,step=0,source_receipt=source_receipt)
        options,excluded=source_options(self.source,batch,self.codec,tenant=self.tenant,principal=self.principal)
        selected=next((i for i,x in enumerate(options) if x.action=='WRITE_BATCH_0'),0) if write else 0
        self.source,receipt=choose_source(self.source,options,selected)
        self.values=initialize_values(self.source);self.frame=frame
        self.initial_source=self.source.snapshot();self.initial_values=self.values
        self.receipts.append(dict(kind='COMMON_INITIAL_MEMORY',source=receipt,excluded=excluded,
            controlled_by_candidate_V=False,reset_receipt=source_receipt))

    def action_input(self):
        self._check()
        if self.frame is None or self.pending is not None or self.step>=self.horizon:
            raise ValueError('one action following the preceding completed transaction')
        memory,read=source_read(self.source,tenant=self.tenant,principal=self.principal,
            clock=self.step,device=self.model.rho.a.device)
        fast=self.fast.fork();prompt=self.formatter.prompt(self.frame)
        seal=_digest(dict(configuration=self._configuration,frame=asdict(self.frame),step=self.step,
            prompt=prompt,phi=fast.identity,source=read))
        context=ManagedInput(self.frame,prompt,fast,memory,read,self.step,seal)
        self.pending=dict(context=context,seal=seal,state=(self.fast.identity,state_identity(self.source,self.values)),
            tensor_versions=_tensor_versions(memory))
        return context

    def _pending(self,context):
        self._check();p=self.pending
        if p is None or context is not p['context']:
            raise ValueError('exact outstanding current-state input required')
        seal=_digest(dict(configuration=self._configuration,frame=asdict(context.frame),step=self.step,
            prompt=context.prompt,phi=context.fast.identity,source=context.source_read))
        if (seal!=p['seal'] or context.binding!=seal or context.step!=self.step
                or p['state']!=(self.fast.identity,state_identity(self.source,self.values))
                or p['tensor_versions']!=_tensor_versions(context.memory)):
            raise ValueError('Phi, authoritative M or input changed before arrived feedback')

    def feedback_text(self,context,*,action,after):
        self._pending(context);self._visible(after)
        if after.goal!=context.frame.goal:raise ValueError('arrived observation belongs to another goal')
        # The target is behind the supervised loss boundary, never in its prompt.
        # Caller separately binds teacher demonstrations or actual actor feedback.
        return self.formatter._supervised('feedback',context.frame,after.observation,action)

    def commit_feedback(self,context,*,action,after,source_receipt,loss,gradient,
                        allow_update,action_tokens):
        self._pending(context);self._visible(after);_sha(source_receipt)
        if (type(allow_update) is not bool or type(action_tokens) is not int
                or not 0<action_tokens<=self.model.profile.max_new_tokens
                or after.goal!=context.frame.goal or type(action) is not str or len(action.encode())>self.formatter.max_action_bytes
                or not torch.is_tensor(loss) or loss.ndim or not bool(torch.isfinite(loss))
                or float(loss.detach())<0):
            raise ValueError('finite arrived feedback and exact completed command required')
        eligible=allow_update and self.fast.steps<self.max_updates
        if eligible!=(gradient is not None):raise ValueError('gradient only for an eligible real update')
        bounded=-math.expm1(-float(loss.detach()));update_receipt=None
        try:
            old_phi=self.fast.identity
            if eligible:
                self.fast,update_receipt=apply_update(self.model.rho,context.fast,gradient,
                    language=self.language,rule=self.updater,bounds=self.bounds,loss=bounded,
                    previous_loss=self.previous_loss,opportunity=self.step,opportunity_cap=self.horizon)
            # Commit order is explicit: U then V. A V failure does not erase an
            # already committed numerical update or allow another action.
            event=dict(kind='ARRIVED_FEEDBACK',step=self.step,source_receipt=source_receipt,
                action=action,old_phi=old_phi,phi=self.fast.identity,update=update_receipt,
                bounded_loss=bounded,manager_status='NOT_COMMITTED')
            self.receipts.append(event)
            batch=self.codec.split(after.observation,step=self.step+1,source_receipt=source_receipt)
            options,excluded=source_options(self.source,batch,self.codec,tenant=self.tenant,principal=self.principal)
            memory,values,decision=manage_source(self.source,self.values,options,language=self.language,
                rule=self.manager,loss=bounded,cost=action_tokens/self.model.profile.max_new_tokens,
                step=self.step+1,horizon=self.horizon,batch=batch,codec=self.codec,goal=self.frame.goal)
            self.source,self.values=memory,values
            event.update(manager_status='COMMITTED',manager=decision,excluded=excluded)
            self.previous_loss=bounded;self.frame=after;self.step+=1;self.pending=None
            return event
        except BaseException:
            self.failed=True
            raise

    def finish_terminal(self,context,*,source_receipt,action):
        self._pending(context);_sha(source_receipt)
        self.receipts.append(dict(kind='TERMINAL',step=self.step,source_receipt=source_receipt,
            action=action,phi=self.fast.identity,source=state_identity(self.source,self.values)))
        self.step+=1;self.pending=None;self.closed=True

    def freeze_feedback(self,context,*,after,source_receipt):
        """Declared frozen control: observe current frame, retain both states."""
        self._pending(context);self._visible(after);_sha(source_receipt)
        if after.goal!=context.frame.goal:raise ValueError('same unresolved task required')
        self.receipts.append(dict(kind='FROZEN_FEEDBACK',step=self.step,source_receipt=source_receipt,
            phi=self.fast.identity,source=state_identity(self.source,self.values)))
        self.frame=after;self.step+=1;self.pending=None
