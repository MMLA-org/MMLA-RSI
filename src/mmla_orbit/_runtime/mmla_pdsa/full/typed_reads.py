"""Bounded engineering typed read views, including current alias dependencies.

Resolved aliases are ephemeral read products. Their resident authority, bytes and
versions never change without a separate commit. No archive or query answer is
available to this resolver.
"""
from dataclasses import dataclass
from .memory import canonical
from .qualification import typed_payload
from .profile import digest


class BoundedReadFailure(ValueError):
    """A legally committed alias graph cannot produce a bounded readable view."""


@dataclass(frozen=True)
class ReadRow:
    object_id:str
    canonical:bytes
    key:bytes|None
    value:bytes|None
    dependencies:tuple[str,...]


@dataclass(frozen=True)
class ReadSet:
    predecessor:str
    tenant:str
    principal:str
    world_clock:int
    reader:str
    view:str
    rows:tuple[ReadRow,...]

    @property
    def identity(self):
        return digest(dict(root=self.predecessor,tenant=self.tenant,principal=self.principal,
            world_clock=self.world_clock,reader=self.reader,view=self.view,
            rows=[dict(object=r.object_id,canonical=r.canonical.hex(),key=None if r.key is None else r.key.hex(),value=None if r.value is None else r.value.hex(),dependencies=r.dependencies) for r in self.rows]))


class EngineeringViewReader:
    def __init__(self,encoder,gate):
        if not getattr(gate,'identity',None) or not callable(gate):raise ValueError('versioned qualification gate required')
        self.encoder,self.gate=encoder,gate
        self.identity=digest(dict(schema='mmla-engineering-typed-read-v1',encoder=encoder.identity,gate=gate.identity,
            alias='bounded-current-root-chain; each-hop-ACL-TTL; ephemeral-derived-row'))
        self._encoder_identity=encoder.identity;self._gate_identity=gate.identity

    def read(self,memory,*,tenant,principal,clock,view='neural'):
        if view not in ('canonical','neural'):raise ValueError('declared read view')
        if memory.encoder.identity!=self._encoder_identity or self.encoder.identity!=self._encoder_identity or self.gate.identity!=self._gate_identity:
            raise ValueError('read/encoder/gate identity changed')
        if memory.semantic_gate is not self.gate:raise ValueError('resident and derived rows require the same gate')
        root=memory.identity
        selectors=memory.selectors(tenant=tenant,principal=principal,clock=clock,view='canonical')
        by_object={s.object_id:s for s in selectors}
        if len(by_object)!=len(selectors):raise ValueError('duplicate visible object')
        rows=[]
        for first in selectors:
            selected=first;seen=set();chain=[];resolved=None
            for _ in range(memory.profile.memory_slots):
                if selected.object_id in seen:raise BoundedReadFailure('alias cycle')
                seen.add(selected.object_id)
                content=memory.read(selected,principal=principal,clock=clock)
                if content is None:break
                kind,value=typed_payload(content);chain.append(selected.receipt)
                if kind=='value':
                    if value['object_id']!=selected.object_id:raise ValueError('canonical object/authority mismatch')
                    resolved=dict(value);resolved['object_id']=first.object_id;break
                # The capsule's own pointer must be qualified for neural use.
                row=memory._rows[selected.slot]
                if view=='neural' and self.gate(content,row.key,row.value) is not True:break
                selected=by_object.get(value['alias_object'])
                if selected is None:break
            else:raise BoundedReadFailure('alias depth exhausted')
            if resolved is None:continue
            content=canonical(resolved);key,value=self.encoder.encode(content)
            if view=='neural' and self.gate(content,key,value) is not True:continue
            rows.append(ReadRow(first.object_id,content,key if view=='neural' else None,value if view=='neural' else None,tuple(chain)))
        if memory.identity!=root:raise RuntimeError('memory changed during alias resolution')
        return ReadSet(root,tenant,principal,clock,self.identity,view,tuple(rows))
