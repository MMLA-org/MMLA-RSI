"""Task-specific J: arrived-state features -> own complete-future failure risk.

This head is separate from the historical four-query J and its qualifications.
The maximum held-out floor residual is an empirical screen, not a coverage
claim. Changed reader/rule/domain, incomplete calibration or an out-of-range
feature vector cannot license a positive action.
"""
from dataclasses import dataclass,asdict
import copy,math
import torch
from .alfworld_episode import _digest,_sha,TaskPermit
from .neural import ActionRisk,tensor_digest
from .outcome_source import SOURCE_DOMAIN,source_read
from .outcome_generation import GENERATION_ID

DEFINITION=dict(schema='orbit-outcome-j-features-v1',
    text='goal, pre-action observation, actual command, arrived observation: separate frozen embedding/consolidator/canonical_value tanh summaries; empty command zero',
    memory='tanh(old legal mean value/key), tanh(candidate legal mean minus old); all complete surviving rows',
    scalars=('UPDATE','NULL','slot_fraction','step_fraction','fast_step_fraction','tanh_rms_A','tanh_rms_B',
        'tanh_rms_momentum_A','tanh_rms_momentum_B','tanh_delta_A','tanh_delta_B','tanh_delta_momentum_A',
        'tanh_delta_momentum_B','bounded_real_observation_CE','eval_occupancy','eval_mean_loss'),
    width='6*neural_width+2*key_width+16',ordering='all KEEP x source options, then all UPDATE x same options if issued',
    target='1-won on own complete fixed-coverage future',
    access='arrived public context, retained legal M and issued Phi/M only; no ID, teacher, hidden fact, future outcome or archive',
    limitation='compressed features; not asserted sufficient')
SCHEMA=_digest(DEFINITION)


def outcome_protocol(session,*,contract,runtime,horizon=64,anchor_step=7):
    session._check();_sha(contract);_sha(runtime)
    if (type(horizon) is not int or not 1<=horizon<=512 or type(anchor_step) is not int
            or not 0<=anchor_step<horizon):raise ValueError('finite prebound anchor and complete horizon')
    evaluation={k:v for k,v in session.evaluation.contract.items() if k!='namespace'}
    return dict(schema='orbit-outcome-j-protocol-v1',generation=GENERATION_ID,features=SCHEMA,contract=contract,runtime=runtime,
        reader=session.reader_identity,profile=session.model.profile.identity,encoder=session.source.encoder.identity,
        formatter=session.formatter.identity,codec=session.codec.identity,language=session.language.identity,
        updater=session.updater.identity,manager=session.manager.identity,bounds=asdict(session.bounds),
        max_updates=session.max_updates,evaluation=evaluation,source_domain=SOURCE_DOMAIN,
        horizon=horizon,anchor_step=anchor_step,continuation='fixed_joint: rotate source slot step%slots, U if eligible, real V; no further J',
        endpoint='environment won after actual terminal or horizon; risk=1-won',
        feature_width=6*session.model.profile.neural_width+2*session.model.profile.key_width+16)


def _match_protocol(session,protocol):
    if protocol!=outcome_protocol(session,contract=protocol['contract'],runtime=protocol['runtime'],
            horizon=protocol['horizon'],anchor_step=protocol['anchor_step']):
        raise ValueError('reader/rules/representation/protocol changed')


@dataclass(frozen=True)
class OutcomeFeatures:
    matrix:tuple
    pairs:tuple
    binding:dict
    def __post_init__(self):
        object.__setattr__(self,'matrix',tuple(tuple(x) for x in self.matrix))
        object.__setattr__(self,'pairs',tuple(tuple(x) for x in self.pairs))
        self.validate()
    def validate(self):
        p=self.binding.get('protocol',{});n=sum(x[0]==0 for x in self.pairs)
        base=self.pairs[:n];rest=self.pairs[n:]
        if (not n or base[0]!=(0,'NULL') or len(self.matrix)!=len(self.pairs)
                or any(type(x[0]) is not int or x[0]!=0 for x in base)
                or len({x[1] for x in base})!=n or (rest and rest!=tuple((1,x[1]) for x in base))
                or self.binding.get('schema')!=SCHEMA or p.get('features')!=SCHEMA
                or p.get('schema')!='orbit-outcome-j-protocol-v1'
                or any(len(row)!=p.get('feature_width') or any(type(v) not in (int,float) or not math.isfinite(v) for v in row) for row in self.matrix)):
            raise ValueError('complete finite task-specific KEEP/UPDATE domain required')
        _sha(self.binding['anchor'])
    @property
    def identity(self):self.validate();return _digest(asdict(self))
    def tensor(self,device):self.validate();return torch.tensor(self.matrix,dtype=torch.float32,device=device)


@torch.no_grad()
def outcome_features(session,*,protocol,on_work):
    _match_protocol(session,protocol);p=session._pending_check('PREPARED')
    if (session.step!=protocol['anchor_step'] or p['opportunity_cap']!=protocol['horizon']
            or p['numerical_target']!='REAL' or not callable(on_work)
            or (p['shadow'] is not None and p['shadow'].identity!=p['shadow_identity'])):
        raise ValueError('one real bound feedback anchor and actual issued successor required')
    anchor=session.branch_identity();model=session.model;profile=model.profile;device=model.rho.a.device
    def encode(text):
        if not text:return torch.zeros(profile.neural_width,device=device)
        ids=session.formatter.tokenizer.encode(text,add_special_tokens=False)
        if not ids or len(ids)>profile.local_tokens:raise ValueError('whole arrived feature text exceeds fixed context')
        on_work(dict(kind='j_frozen_text_encoding',tokens=len(ids)))
        embedded=model.mixer.get_input_embeddings()(torch.tensor([ids],device=device))
        return model.canonical_value(model.consolidator(embedded)).squeeze(0).tanh()
    f=p['feedback']
    texts=torch.cat([encode(t) for t in (f.before.goal,f.before.observation,f.action,f.observation_target)])
    def summary(memory,clock):
        read,_=source_read(memory,tenant=session.tenant,principal=session.principal,clock=clock,device=device)
        if read is None:return torch.zeros(profile.neural_width+profile.key_width,device=device)
        return torch.cat((read[1].mean(0),read[0].mean(0)))
    old=summary(session.source,session.step);fast=session.fast
    norms=[float(getattr(fast,k).detach().float().square().mean().sqrt().tanh()) for k in ('a','b','momentum_a','momentum_b')]
    occupied=[x.record.loss for x in session.evaluation.rows if x is not None]
    rows=[];pairs=[]
    for update in ((0,1) if p['shadow'] is not None else (0,)):
        delta=[float((getattr(p['shadow'],k)-getattr(fast,k)).float().square().mean().sqrt().tanh())
            for k in ('a','b','momentum_a','momentum_b')] if update else [0.]*4
        for option in p['options']:
            if option.predecessor!=session.source.identity or option.memory.identity!=option.successor:
                raise ValueError('stale issued memory candidate')
            current=summary(option.memory,session.step+1)
            slot=-1. if option.action=='NULL' else int(option.action.removeprefix('WRITE_BATCH_'))/max(1,profile.memory_slots-1)
            scalars=[float(update),float(option.action=='NULL'),slot,session.step/protocol['horizon'],
                fast.steps/max(1,session.max_updates),*norms,*delta,p['bounded_loss'],
                len(occupied)/session.evaluation.bounds.slots,sum(occupied)/len(occupied) if occupied else 0.]
            row=torch.cat((texts,old.tanh(),(current-old).tanh())).cpu().tolist()+scalars
            rows.append(tuple(row));pairs.append((update,option.action))
    if session.branch_identity()!=anchor:raise RuntimeError('feature computation changed common predecessor')
    return OutcomeFeatures(tuple(rows),tuple(pairs),dict(schema=SCHEMA,protocol=copy.deepcopy(protocol),anchor=anchor))


def validate_outcome_record(record,*,role,protocol,expected):
    """Expected maps ORIGINAL task IDs to exact permits, never renamed groups."""
    if role not in ('FIT','C_J0'):raise ValueError('J source role must be FIT or C_J0')
    f=record.get('features');permit=expected.get(record.get('task'))
    if (type(permit) is not TaskPermit or permit.role!=role or record.get('task')!=permit.task_id or record.get('permit')!=asdict(permit)
            or record.get('floor')!=permit.floor or record.get('role')!=role
            or record.get('schema')!='orbit-own-complete-futures-v1' or record.get('coverage')!='COMPLETE'
            or type(f) is not OutcomeFeatures or f.binding.get('protocol')!=protocol
            or f.binding.get('anchor')!=record.get('anchor')):
        raise ValueError('bound complete original-task record required')
    f.validate();cp=record.get('checkpoint',{})
    if (cp.get('permit')!=asdict(permit) or len(cp.get('steps',()))!=protocol['anchor_step']+1
            or cp.get('limits',{}).get('max_steps')!=protocol['horizon']):
        raise ValueError('actual bound task prefix at declared anchor required')
    actions=record.get('actions',());risks=record.get('risks',())
    if len(actions)!=len(f.pairs) or len(risks)!=len(f.pairs):raise ValueError('every issued pair must have its own complete future')
    for pair,action,risk in zip(f.pairs,actions,risks):
        out=action.get('outcome',{})
        if (tuple(action.get('pair',()))!=pair or out.get('status')!='COMPLETE'
                or type(out.get('won')) is not bool or type(out.get('truncated')) is not bool
                or type(out.get('steps')) is not int or not protocol['anchor_step']+1<out['steps']<=protocol['horizon']
                or (out['truncated'] and (out['won'] or out['steps']!=protocol['horizon']))
                or type(risk) not in (int,float) or risk!=1-int(out['won']) or action.get('risk')!=risk):
            raise ValueError('risk is 1-won from this pair own COMPLETE finite future')
        _sha(action['final_checkpoint'])
    return f


class OutcomeHead(ActionRisk):
    def __init__(self,protocol,*,seed):
        if protocol.get('schema')!='orbit-outcome-j-protocol-v1' or protocol.get('features')!=SCHEMA:
            raise ValueError('new outcome J domain required')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed);super().__init__(protocol['feature_width'],width=64)
        self.protocol=copy.deepcopy(protocol);self.protocol_id=_digest(protocol)
    def forward(self,features):
        if _digest(self.protocol)!=self.protocol_id or features.shape[-1]!=self.protocol['feature_width']:
            raise ValueError('J protocol or feature width changed')
        return super().forward(features)


def fit_outcome_head(head,records,*,expected,seed,on_work):
    if type(head) is not OutcomeHead or int(head.updates)!=0 or len(expected)!=16 or len({p.floor for p in expected.values()})!=16:
        raise ValueError('one fresh head and fixed sixteen original FIT floors')
    if len(records)!=len(expected) or {r['task'] for r in records}!=set(expected):
        raise ValueError('missing or duplicated FIT task; no replacement')
    by_task={r['task']:r for r in records};device=next(head.parameters()).device
    order=sorted(expected,key=lambda task:_digest(['orbit-outcome-j-fit-order-v1',seed,task]))
    tensors={}
    for task in order:
        r=by_task[task];f=validate_outcome_record(r,role='FIT',protocol=head.protocol,expected=expected)
        tensors[task]=(f.tensor(device),torch.tensor(r['risks'],device=device,dtype=torch.float32))
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    history=[]
    for step in range(128):
        ids=[order[(4*step+j)%len(order)] for j in range(4)]
        optimizer.zero_grad(set_to_none=True)
        loss=torch.stack([(head(tensors[i][0])-tensors[i][1]).square().mean() for i in ids]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite J fitting loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in head.parameters()):raise FloatingPointError('nonfinite J parameter')
        head.updates.add_(1);on_work(dict(kind='j_optimizer_update',step=step+1,tasks=4,pairs=sum(len(by_task[i]['risks']) for i in ids)))
        history.append(dict(step=step+1,tasks=ids,loss=float(loss.detach()),gradient_norm=float(norm)))
    return dict(history=history,optimizer=optimizer.state_dict(),order=order,head=tensor_digest(head.state_dict()))


@dataclass(frozen=True)
class OutcomeCalibration:
    protocol:dict
    head:str
    floor_errors:tuple
    radius:float
    minimum:tuple
    maximum:tuple
    evidence:str
    coverage:str='EMPIRICAL_FIXED_PROTOCOL_ONLY'
    def __post_init__(self):
        object.__setattr__(self,'floor_errors',tuple(tuple(x) for x in self.floor_errors))
        object.__setattr__(self,'minimum',tuple(self.minimum));object.__setattr__(self,'maximum',tuple(self.maximum))
        self.validate()
    def validate(self):
        if (len(self.floor_errors)!=8 or len({g for g,e in self.floor_errors})!=8
                or any(type(g) is not str or not g or not math.isfinite(e) or not 0<=e<=1 for g,e in self.floor_errors)
                or self.radius!=max(e for g,e in self.floor_errors) or not self.minimum
                or len(self.minimum)!=self.protocol.get('feature_width') or len(self.minimum)!=len(self.maximum)
                or any(not math.isfinite(a) or not math.isfinite(b) or a>b for a,b in zip(self.minimum,self.maximum))
                or self.coverage!='EMPIRICAL_FIXED_PROTOCOL_ONLY' or self.protocol.get('features')!=SCHEMA):
            raise ValueError('complete fixed eight-floor empirical artifact required')
        _sha(self.head);_sha(self.evidence)
    @property
    def identity(self):self.validate();return _digest(asdict(self))


@torch.no_grad()
def predict(head,features):
    if type(head) is not OutcomeHead or type(features) is not OutcomeFeatures or features.binding['protocol']!=head.protocol:
        raise ValueError('matching task-specific features and J required')
    values=head(features.tensor(next(head.parameters()).device))
    if values.shape!=(len(features.pairs),) or not torch.isfinite(values).all():raise FloatingPointError('nonfinite complete J predictions')
    return values.clamp(0,1).cpu().tolist()


def calibrate_outcome_head(head,records,*,expected):
    if (type(head) is not OutcomeHead or int(head.updates)!=128 or len(expected)!=8
            or len({p.floor for p in expected.values()})!=8 or len(records)!=8
            or {r['task'] for r in records}!=set(expected)):
        raise ValueError('complete unchanged head and eight fixed calibration tasks required; no replacement')
    rows=[];errors=[];evidence=[]
    for r in records:
        f=validate_outcome_record(r,role='C_J0',protocol=head.protocol,expected=expected)
        values=predict(head,f);errors.append((r['floor'],max(abs(a-b) for a,b in zip(values,r['risks']))))
        rows.extend(f.matrix);evidence.append(dict(task=r['task'],features=f.identity,risks=r['risks']))
    return OutcomeCalibration(copy.deepcopy(head.protocol),tensor_digest(head.state_dict()),tuple(sorted(errors)),
        max(e for _,e in errors),tuple(map(min,zip(*rows))),tuple(map(max,zip(*rows))),_digest(evidence))


class OutcomePredictor:
    def __init__(self,head,calibration,*,reader):
        if type(head) is not OutcomeHead or (calibration is not None and type(calibration) is not OutcomeCalibration):
            raise ValueError('actual outcome head and optional complete calibration')
        if calibration is not None and (int(head.updates)!=128 or calibration.protocol!=head.protocol
                or calibration.head!=tensor_digest(head.state_dict())):raise ValueError('trained calibrated head identity mismatch')
        self.head,self.reader=head,reader;self.protocol=copy.deepcopy(head.protocol)
        self.calibration=copy.deepcopy(calibration)
        self._head=tensor_digest(head.state_dict());self._guard=self._reader_guard()
        self._calibration=None if calibration is None else calibration.identity
        self.identity=_digest(dict(schema='orbit-outcome-j-selector-v1',protocol=self.protocol,head=self._head,
            calibration=self._calibration,minimum_gain=0.,tie='lowest pair index; KEEP NULL first'))
    def _reader_guard(self):
        return tuple((n,id(v),v._version,str(v.device),str(v.dtype),tuple(v.shape),v.data_ptr())
            for n,v in tuple(self.reader.named_parameters())+tuple(self.reader.named_buffers()))
    def __call__(self,features):
        if (self._reader_guard()!=self._guard or tensor_digest(self.head.state_dict())!=self._head
                or _digest(self.protocol)!=self.head.protocol_id or features.binding.get('protocol')!=self.protocol
                or (None if self.calibration is None else self.calibration.identity)!=self._calibration):
            raise ValueError('outcome reader/head/calibration/protocol changed')
        features.validate();c=self.calibration
        if c is None:return 0,dict(mode='PDSA_J_COMPLETE_TASK',reason='UNQUALIFIED_NO_COMPLETE_CALIBRATION')
        if any(any(v<lo or v>hi for v,lo,hi in zip(row,c.minimum,c.maximum)) for row in features.matrix):
            return 0,dict(mode='PDSA_J_COMPLETE_TASK',reason='OUTSIDE_CALIBRATION_FEATURE_RANGE',calibration=c.identity)
        values=predict(self.head,features);gains=[values[0]-v-2*c.radius for v in values]
        eligible=[i for i in range(1,len(values)) if gains[i]>0.]
        chosen=min(eligible,key=lambda i:(values[i],i)) if eligible else 0
        return chosen,dict(mode='PDSA_J_COMPLETE_TASK',calibration=c.identity,coverage=c.coverage,
            predicted_risks=values,empirical_gain_bounds=gains,selected=chosen)
