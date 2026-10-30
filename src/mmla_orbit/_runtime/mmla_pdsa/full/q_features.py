"""One causal Q feature path and explicitly empirical reference-policy screen.

The old exact-Phi Calibration/select API is not changed. This radius describes
fixed-behavior calibration families, not coverage on an adaptive deployed path.
"""
from dataclasses import dataclass,asdict,field
import copy,math,struct
import torch
from .construction import validate_context
from .frames import action_identity
from .profile import digest
from .neural import tensor_digest
from .actual_tasks import BEHAVIOR_ID

FEATURE_DEFINITION=dict(schema='mmla-current-action-features-v1',
    vector='tanh(event + host-visible mean M value + candidate-minus-actual-old value); tanh(candidate-minus-old key)',
    scalars=('is_null','slot_fraction','fast_steps_fraction','world_clock_fraction',
             'tanh_rms_A','tanh_rms_B','tanh_rms_momentum_A','tanh_rms_momentum_B'),
    absent_or_unreadable='zero vectors; NULL has zero candidate delta',
    access='current typed context, issued actions, actual Phi; no future query/answer/Truth/partition',
    limitations='compressed features are not asserted to be a sufficient statistic')
FEATURE_SCHEMA=digest(FEATURE_DEFINITION)
COVERAGE='UNQUALIFIED_FOR_ADAPTIVE_DEPLOYMENT'
MODE='EMPIRICAL_REFERENCE_Q_PI_SCREEN'


@dataclass(frozen=True)
class ActionFeatures:
    matrix:tuple
    actions:tuple
    binding:dict
    def __post_init__(self):
        if (type(self.matrix) is not tuple or len(self.matrix)!=len(self.actions) or not self.actions
                or self.actions[0]!='NULL' or len(set(self.actions))!=len(self.actions)
                or not self.matrix[0] or any(len(row)!=len(self.matrix[0]) for row in self.matrix)
                or any(not math.isfinite(v) for row in self.matrix for v in row)):
            raise ValueError('complete finite rectangular action matrix required')
    @property
    def identity(self):return digest(asdict(self))
    def tensor(self,device):return torch.tensor(self.matrix,device=device,dtype=torch.float32)


def numerical_protocol(model,*,update):
    if set(update)!={'learning_rate','momentum','clip'}:raise ValueError('complete fixed updater required')
    if (not all(math.isfinite(x) for x in update.values()) or not 0<update['learning_rate']<=1
            or not 0<=update['momentum']<1 or update['clip']<=0):raise ValueError('finite numerical protocol')
    return dict(schema='mmla-early-one-step-reference-q-v1',rho=tensor_digest(model.rho.state_dict()),
        profile=model.profile.identity,substrate=model.substrate_identity,update=dict(update),
        timing='early',steps=1,feedback_world_clock=0,source_world_clocks=(1,2,3),
        feature_schema=FEATURE_SCHEMA,continuation=BEHAVIOR_ID)


@torch.no_grad()
def action_features(problem,context,candidates):
    problem._ready();model=problem.model;memory=problem.memory
    validate_context(model,memory,context)
    if (not candidates or candidates[0] is not None or len(candidates)>memory.profile.memory_slots+1
            or context.envelope.tenant!=problem.tenant or context.envelope.world_clock!=problem.world_clock):
        raise ValueError('current same-scope complete issued action set required')
    ids=tuple(action_identity(c) for c in candidates)
    if len(set(ids))!=len(ids):raise ValueError('duplicate issued action')
    for c in candidates[1:]:
        if (c is None or c.predecessor!=memory.identity or c.epoch!=memory.epoch
                or c.slot not in context.slots or memory._pending.get((c.slot,c.row.receipt))!=c):
            raise ValueError('unissued or stale action cannot be scored')
        fields=c.row.fields;env=context.envelope
        if (fields['tenant'],fields['object_id'],fields['source_receipt'])!=(env.tenant,env.object_id,env.source_receipt):
            raise ValueError('candidate belongs to another source event')
    if len(memory._pending)!=len(candidates)-1:raise ValueError('every issued action must be scored')
    width=model.profile.neural_width;key_width=model.profile.key_width;device=model.rho.a.device
    fast=problem.fast
    norms=torch.stack([torch.sqrt(t.detach().float().square().mean()).tanh() for t in
        (fast.a,fast.b,fast.momentum_a,fast.momentum_b)]).cpu().tolist()
    def vector(value,n):return (0.,)*n if value is None else struct.unpack('<'+'f'*n,value)
    output=[]
    for c in candidates:
        delta=[0.]*width;keys=[0.]*key_width
        if c is not None:
            pos=context.slots.index(c.slot);old=memory._rows[c.slot]
            # old_features are explicitly host-visible. Never read hidden old
            # bytes through a feature shortcut; unreadable old key is zero too.
            old_value=context.old_features[pos]
            old_key=vector(old.key,key_width) if old is not None and context.metadata[pos][1] else (0.,)*key_width
            from .token_memory import unpack_value
            new_value=vector(unpack_value(model.profile,c.row.value)[0],width);new_key=vector(c.row.key,key_width)
            delta=[x-y for x,y in zip(new_value,old_value)];keys=[x-y for x,y in zip(new_key,old_key)]
        mixed=[math.tanh(e+m+d) for e,m,d in zip(context.event_features,context.summary,delta)]
        row=tuple(mixed+[math.tanh(v) for v in keys]+[
            float(c is None),-1. if c is None else c.slot/max(1,model.profile.memory_slots-1),
            fast.steps/model.profile.max_fast_steps,problem.world_clock/4.,*norms])
        if len(row)!=width+key_width+8 or not all(math.isfinite(v) for v in row):raise ValueError('finite exact-width current features')
        output.append(row)
    feedback=tuple({k:copy.deepcopy(record.get(k)) for k in
        ('attempt','source','world_clock','updater','phi_before','phi_after')}
        for record in problem.receipts if record.get('phi')=='COMMITTED' and record.get('status')=='COMPLETED')
    binding=dict(schema=FEATURE_SCHEMA,problem=problem.problem_id,tenant=problem.tenant,
        principal=problem.principal,memory=memory.identity,phi=fast.identity,
        rho=tensor_digest(model.rho.state_dict()),reader=problem.reader_identity(),context=context.identity,
        encoder=context.encoder,gate=context.gate,clock=problem.world_clock,
        event=context.event.identity,segment=problem.segment.number,cursor=problem.segment._cursor,
        fast_steps=fast.steps,attempts=problem.attempts,arrived_feedback=feedback)
    return ActionFeatures(tuple(output),ids,binding)


@dataclass(frozen=True)
class EmpiricalRadius:
    protocol:dict
    reader:str
    encoder:str
    gate:str
    q:str
    family_errors:tuple
    alpha:float
    radius:float
    minimum:tuple
    maximum:tuple
    evidence:str
    coverage:str=COVERAGE
    _identity:str=field(init=False,repr=False)
    def payload(self):return {k:v for k,v in asdict(self).items() if k!='_identity'}
    def __post_init__(self):
        object.__setattr__(self,'_identity','')
        self.validate(check_identity=False)
        object.__setattr__(self,'_identity',digest(self.payload()))
    def validate(self,*,check_identity=True):
        if (self.coverage!=COVERAGE or self.alpha!=.10 or len(self.family_errors)!=17
                or len({g for g,_ in self.family_errors})!=17
                or any(type(g) is not str or not g or not math.isfinite(e) or not 0<=e<=1 for g,e in self.family_errors)
                or not math.isfinite(self.radius) or self.radius!=max(e for _,e in self.family_errors)
                or not self.minimum or len(self.minimum)!=len(self.maximum)
                or any(not math.isfinite(lo) or not math.isfinite(hi) or lo>hi for lo,hi in zip(self.minimum,self.maximum))
                or any(type(x) is not str or len(x)!=64 for x in (self.reader,self.encoder,self.gate,self.q,self.evidence))):
            raise ValueError('complete finite fixed17 empirical artifact required')
        if check_identity and digest(self.payload())!=self._identity:raise ValueError('empirical artifact mutated')
    @property
    def identity(self):
        self.validate();return self._identity


def clipped_predictions(model,features):
    with torch.no_grad():raw=model.risk(features.tensor(model.rho.a.device))
    if raw.shape!=(len(features.actions),) or not bool(torch.isfinite(raw).all()):
        raise FloatingPointError('complete finite raw action predictions required before clipping')
    return raw.clamp(0,1).cpu().tolist()


def empirical_radius(model,records,*,protocol,expected_families):
    """Complete17-family fixed-behavior residuals; never a deployment guarantee.

    Each record is one complete actual decision with all own issued actions.
    Collection must retain failures/coverage separately; a failed family is not
    an admissible record subset. The caller supplies the immutable expected set.
    """
    if protocol!=numerical_protocol(model,update=protocol['update']):
        raise ValueError('calibration must bind the current actual rho/protocol')
    if len(expected_families)!=17 or len(set(expected_families))!=17:raise ValueError('all 17 calibration families')
    groups={g:[] for g in expected_families};vectors=[];readers=set();encoders=set();gates=set();evidence=[]
    for row in records:
        if row['split']!='calibration' or row['family'] not in groups or row['coverage']!='COMPLETE':
            raise ValueError('only complete held-out calibration trajectories')
        features=row['features'];targets=tuple(row['risks'])
        if type(features) is not ActionFeatures or len(targets)!=len(features.actions):raise ValueError('all exact issued-action risks')
        binding=features.binding;feedback=binding.get('arrived_feedback',())
        if (row.get('protocol')!=digest(protocol) or binding.get('rho')!=protocol['rho']
                or binding.get('schema')!=FEATURE_SCHEMA or binding.get('fast_steps')!=1
                or binding.get('attempts')!=1 or binding.get('clock') not in (1,2,3)
                or len(feedback)!=1 or feedback[0].get('world_clock')!=0
                or feedback[0].get('updater')!=protocol['update']
                or feedback[0].get('phi_after')!=binding.get('phi')):
            raise ValueError('record is not this actual post-feedback numerical domain')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in targets):raise ValueError('bounded completed future risk')
        pred=clipped_predictions(model,features)
        groups[row['family']].append(max(abs(x-y) for x,y in zip(pred,targets)))
        vectors.extend(features.matrix);readers.add(features.binding['reader'])
        encoders.add(features.binding['encoder']);gates.add(features.binding['gate'])
        evidence.append(dict(family=row['family'],source=row['source'],anchor=row['anchor'],
            features=features.identity,risks=targets,clipped_predictions=pred))
    if any(not x for x in groups.values()) or len(readers)!=1 or len(encoders)!=1 or len(gates)!=1:
        raise ValueError('missing family or mixed physical reader definition')
    errors=tuple((g,max(groups[g])) for g in sorted(groups));n=len(errors);alpha=.10
    order=math.ceil((n+1)*(1-alpha));radius=1. if order>n else sorted(x for _,x in errors)[order-1]
    return EmpiricalRadius(protocol,readers.pop(),encoders.pop(),gates.pop(),tensor_digest(model.risk.state_dict()),
        errors,alpha,radius,tuple(map(min,zip(*vectors))),tuple(map(max,zip(*vectors))),digest(evidence))


class ReferencePredictor:
    def __init__(self,model,calibration,*,update):
        self.model=model;self.calibration=calibration;self.update=dict(update)
    def __call__(self,problem,context,candidates):
        info=dict(mode=MODE,coverage=COVERAGE,continuation=BEHAVIOR_ID,predictive=False)
        c=self.calibration
        if c is None:return 0,dict(info,reason='NO_COMPLETE_EMPIRICAL_CALIBRATION')
        if type(c) is not EmpiricalRadius or c.coverage!=COVERAGE:raise ValueError('explicit empirical reference-policy artifact required')
        c.validate()
        if problem.model is not self.model:raise ValueError('wrong active model')
        if (c.protocol!=numerical_protocol(self.model,update=self.update)
                or c.q!=tensor_digest(self.model.risk.state_dict())):
            return 0,dict(info,reason='MODEL_OR_PROTOCOL_CHANGED')
        feedback=[r for r in problem.receipts if r.get('phi')=='COMMITTED' and r.get('status')=='COMPLETED']
        if (problem.fast.steps!=1 or problem.world_clock not in (1,2,3) or problem.attempts!=1
                or len(feedback)!=1 or feedback[0].get('world_clock')!=0 or feedback[0].get('updater')!=self.update):
            return 0,dict(info,reason='OUTSIDE_EARLY_ONE_STEP_PROTOCOL')
        features=action_features(problem,context,candidates)
        if (features.binding['reader'],features.binding['encoder'],features.binding['gate'])!=(c.reader,c.encoder,c.gate):
            return 0,dict(info,reason='READER_ENCODER_GATE_CHANGED')
        if (len(c.minimum)!=len(features.matrix[0]) or any(not lo<=v<=hi for row in features.matrix
                for v,lo,hi in zip(row,c.minimum,c.maximum))):
            return 0,dict(info,reason='OUTSIDE_OBSERVED_FEATURE_RANGE')
        pred=clipped_predictions(self.model,features)
        if not all(math.isfinite(v) for v in pred):raise FloatingPointError('nonfinite reference risks')
        lower=[max(0.,x-c.radius) for x in pred];upper=[min(1.,x+c.radius) for x in pred]
        feasible=[i for i in range(1,len(pred)) if upper[i]<lower[0]]
        selected=min(feasible,key=lambda i:(pred[i],i)) if feasible else 0
        return selected,dict(info,predictive=True,reason='REFERENCE_INTERVAL_SEPARATION' if selected else 'NO_SEPARATED_WRITE',
            features=features.identity,calibration=c.identity,predictions=pred,lower=lower,upper=upper,
            applicability='protocol and coordinate ranges only; adaptive distribution shift is not certified')
