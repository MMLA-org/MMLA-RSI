"""Root-bound G inputs and trusted engineering candidate assembly.

This is an in-process capability boundary, not caller authentication. The task
adapter supplies authenticated envelopes and already-closed predictions. Neural
G supplies semantics only; it cannot alter grants, slots, provenance or versions.
"""
from dataclasses import dataclass,asdict
import json,struct
import torch
from .profile import digest
from .memory import Memory,Authority,Proposal,canonical
from .lifecycle import Grant,Lifecycle
from .constructor import SlotConstructor
from .frontend import TypedIR
from .neural import tensor_digest
from .qualification import typed_payload

ROUTES={'create':'insert','update_value':'update','update_unit':'patch','delete':'delete',
 'quarantine':'quarantine','protect':'protect','unprotect':'unprotect','acl':'restrict',
 'validity':'expire','alias':'alias','repair':'repair','rollback':'rollback','purge':'purge','null':'null'}


@dataclass(frozen=True)
class Envelope:
 tenant:str
 principal:str
 object_id:str
 source_receipt:str
 world_clock:int
 allowed:tuple[str,...]
 acl:tuple[str,...]|None=None
 valid_until:int|None=None
 override_protection:bool=False
 allow_eviction:bool=False

 def __post_init__(self):
  if any(type(x) is not str or not 0<len(x.encode())<=256 for x in (self.tenant,self.principal,self.object_id,self.source_receipt)):
   raise ValueError('bounded authenticated envelope')
  if type(self.world_clock) is not int or not 0<=self.world_clock<2**63:raise ValueError('trusted world clock')
  if type(self.allowed) is not tuple or any(x not in ROUTES.values() for x in self.allowed):raise ValueError('finite authority routes')
  if self.acl is not None and (type(self.acl) is not tuple or not self.acl or any(type(x) is not str or not x for x in self.acl)):raise ValueError('trusted ACL')
  if self.valid_until is not None and (type(self.valid_until) is not int or not 0<=self.valid_until<2**64):raise ValueError('trusted validity')
  if type(self.override_protection) is not bool or type(self.allow_eviction) is not bool:raise ValueError('explicit host flags')


@dataclass(frozen=True)
class ConstructionContext:
 root:str
 epoch:int
 event:TypedIR
 envelope:Envelope
 encoder:str
 gate:str
 frontend:str
 slots:tuple[int,...]
 receipts:tuple[str,...]
 old_contents:tuple[bytes|None,...]
 event_features:tuple[float,...]
 old_features:tuple[tuple[float,...],...]
 summary:tuple[float,...]
 metadata:tuple[tuple[float,...],...]

 @property
 def identity(self):
  data=asdict(self);data['old_contents']=[None if x is None else x.hex() for x in self.old_contents]
  return digest(data)


def _feature_id(model):return tensor_digest(model.frontend.state_dict())


def pin_context(model,memory,event,event_features,envelope):
 if type(memory) is not Memory or type(event) is not TypedIR or type(envelope) is not Envelope:
  raise ValueError('explicit model, memory, typed prediction and trusted envelope')
 if memory.profile!=model.profile:raise ValueError('profile mismatch')
 if event.kind!='null' and event.object_id!=envelope.object_id:raise PermissionError('predicted object outside source authority')
 if not getattr(memory.semantic_gate,'identity',None):raise ValueError('production gate identity required')
 width=model.profile.neural_width
 if event_features.shape!=(width,) or not torch.isfinite(event_features).all():raise ValueError('bounded event features')
 root=memory.identity
 selectors=memory.selectors(tenant=envelope.tenant,principal=envelope.principal,clock=envelope.world_clock,view='neural')
 visible={s.slot:s for s in selectors};vectors={}
 for slot,s in visible.items():
  row=memory._rows[slot];kind,content=typed_payload(row.canonical)
  if kind=='value' and content['object_id']!=row.fields['object_id']:raise ValueError('typed object identity mismatch')
  from .token_memory import unpack_value
  vectors[slot]=struct.unpack('<'+'f'*width,unpack_value(model.profile,row.value)[0])
 summary=tuple(sum(v[i] for v in vectors.values())/len(vectors) if vectors else 0. for i in range(width))
 slots=[];receipts=[];contents=[];features=[];metadata=[]
 for slot,row in enumerate(memory._rows):
  # Foreign/inaccessible live row bytes cannot enter G or the M summary.
  if row is not None and (row.fields['tenant']!=envelope.tenant or envelope.principal not in row.fields['acl']):continue
  same=row is not None and row.fields['object_id']==envelope.object_id
  slots.append(slot);receipts.append('EMPTY' if row is None else row.receipt)
  readable=slot in visible
  contents.append(row.canonical if same and readable else None)
  features.append(tuple(vectors.get(slot,(0.,)*width)))
  metadata.append((float(row is not None),float(readable),float(same),
    float(row is not None and row.fields['protected']),slot/max(1,memory.profile.memory_slots-1),
    min(1.,(0 if row is None else row.fields['version'])/1024),
    float(row is not None and row.status=='quarantine'),float(row is not None and row.status=='purged')))
 if not slots:raise PermissionError('no host-visible candidate slots')
 if memory.identity!=root:raise RuntimeError('memory changed while binding G context')
 return ConstructionContext(root,memory.epoch,event,envelope,memory.encoder.identity,memory.semantic_gate.identity,
    _feature_id(model),tuple(slots),tuple(receipts),tuple(contents),tuple(event_features.detach().cpu().tolist()),
    tuple(features),summary,tuple(metadata))


def validate_context(model,memory,context):
 if (type(context) is not ConstructionContext or memory.identity!=context.root or memory.epoch!=context.epoch
     or memory.encoder.identity!=context.encoder or getattr(memory.semantic_gate,'identity',None)!=context.gate
     or _feature_id(model)!=context.frontend):raise ValueError('stale G context or feature provider')
 for slot,receipt in zip(context.slots,context.receipts):
  row=memory._rows[slot]
  if ('EMPTY' if row is None else row.receipt)!=receipt:raise ValueError('target row changed')


def predict(model,memory,context):
 validate_context(model,memory,context);device=model.rho.a.device
 tensor=lambda x:torch.tensor(x,device=device,dtype=torch.float32)
 result=model.constructor(tensor(context.event_features),tensor(context.old_features),tensor(context.summary),tensor(context.metadata))
 # Return the differentiable output for module training; assembly rechecks its
 # exact detached identity, bound context and actual constructor version.
 receipt=dict(context=context.identity,constructor=tensor_digest(model.constructor.state_dict()),prediction=tensor_digest(result))
 return result,receipt


def assemble_actions(model,memory,context,result,receipt,*,rollback_rows=()):
 validate_context(model,memory,context)
 if receipt!=dict(context=context.identity,constructor=tensor_digest(model.constructor.state_dict()),prediction=tensor_digest(result)):
  raise ValueError('G output/context/constructor mismatch')
 env=context.envelope;ir=context.event;prepared=[];rejected=[]
 for i,slot in enumerate(context.slots):
  old=memory._rows[slot]
  try:
   intent=SlotConstructor.semantic(result,i,ir,context.old_contents[i])
   if intent is None:continue
   if intent.object_id!=env.object_id or ROUTES[intent.kind] not in env.allowed:raise PermissionError('ungranted predicted operation')
   same=old is not None and old.fields['object_id']==env.object_id
   if intent.kind in ('create','update_value','update_unit'):
    if intent.kind!='create' and (not same or old.status!='live'):raise ValueError('field update requires its actual live object')
    if intent.kind=='create' and old is not None and not same and not env.allow_eviction:raise PermissionError('no eviction capability')
    if intent.kind=='create' and same:raise ValueError('create cannot overwrite an existing incarnation')
    value=json.loads(intent.content);content=canonical(dict(object_id=intent.object_id,**value));typed_payload(content)
    authority=Authority(env.tenant,tuple(old.fields['acl']) if same else env.acl,
       env.source_receipt,env.object_id,False,old.fields['valid_until'] if same else env.valid_until)
    # Constructor cannot OLD-copy an evictee: old_contents was None unless same.
    prepared.append((slot,'upsert',content,authority,None));continue
   if not same:raise ValueError('lifecycle operation must address actual object')
   operation=intent.kind
   grant=Grant(env.tenant,env.principal,(operation,),env.source_receipt,memory.identity,env.override_protection)
   args={}
   if operation=='repair':
    value=json.loads(intent.content);args['content']=canonical(dict(object_id=env.object_id,**value));typed_payload(args['content'])
   elif operation=='alias':args['content']=canonical(dict(alias_object=intent.reference))
   elif operation=='acl':
    if env.acl is None:raise ValueError('ACL action needs an authenticated replacement ACL')
    args['acl']=env.acl
   elif operation=='validity':
    if env.valid_until is None:raise ValueError('validity action needs an authenticated end')
    args['valid_until']=env.valid_until
   elif operation=='rollback':
    if not rollback_rows:raise ValueError('committed prior row proof required')
    args.update(reference=rollback_rows[-1],history=tuple(rollback_rows[:-1]))
   prepared.append((slot,operation,args,grant,None))
  except (ValueError,PermissionError,TypeError,KeyError) as error:
   rejected.append(dict(slot=slot,status='INFEASIBLE_CONSTRUCTION',reason=type(error).__name__,message=str(error)[:256]))
 # Prefill immutable qualification work as a batch before AMR's per-row checks.
 contents=[]
 for slot,op,body,authority,_ in prepared:
  if op=='upsert':contents.append(body)
  elif op in ('repair','alias'):contents.append(body['content'])
  elif op=='rollback':contents.append(body['reference'].canonical)
 if contents and hasattr(memory.semantic_gate,'prepare'):
  memory.semantic_gate.prepare(tuple(dict.fromkeys(contents)),microbatch=8)
 candidates=[None]
 for slot,op,body,authority,_ in prepared:
  try:
   # Validate/qualify on an isolated issuance table. A rejected row must not
   # consume the live store's pending-candidate capacity.
   scratch=memory.fork();scratch._pending.clear()
   candidate=scratch.assemble(slot,Proposal(body),authority) if op=='upsert' else Lifecycle(scratch).prepare(slot,op,authority,**body)
   row=candidate.row
   if row.status=='live':
    kind,decoded=typed_payload(row.canonical)
    if kind=='value' and decoded['object_id']!=row.fields['object_id']:raise ValueError('candidate canonical/authority mismatch')
    if row.fields['semantic_qualified'] is not True:raise ValueError('candidate failed dual-view qualification')
   candidates.append(memory._issue(slot,row))
  except (ValueError,PermissionError,TypeError,KeyError) as error:
   # No rejected candidate is returned for scoring or commit.
   rejected.append(dict(slot=slot,status='INFEASIBLE_ASSEMBLY',reason=type(error).__name__,message=str(error)[:256]))
 if memory.identity!=context.root:raise RuntimeError('candidate construction changed authoritative M')
 return tuple(candidates),tuple(rejected)


def commit_decision(memory,candidates,chosen):
 """Consume one issued action set, including retirement after a NULL choice."""
 if not candidates or candidates[0] is not None or chosen not in candidates:
  raise ValueError('selected action must belong to its complete issued set')
 for candidate in candidates[1:]:
  if candidate.predecessor!=memory.identity or memory._pending.get((candidate.slot,candidate.row.receipt))!=candidate:
   raise ValueError('stale or unissued action set')
 result=memory.commit(chosen)
 if chosen is None:
  for candidate in candidates[1:]:
   key=(candidate.slot,candidate.row.receipt)
   if memory._pending.get(key)==candidate:memory._pending.pop(key)
 return result
