"""Privileged AMR lifecycle, separate from learned content proposals.

The embedding application authenticates Grant. This in-process capability is not
an authentication protocol. Purge deletes resident content; this profile retains
no content archive. Exported owner backups are outside the live store's erasure
scope and must be governed by their own retention policy.
"""
from dataclasses import dataclass,asdict
import json

from .memory import Memory,Row,Candidate,MAX_VERSION,canonical
from .profile import wire,digest


OPERATIONS=frozenset(('protect','unprotect','acl','validity','delete','quarantine','repair','rollback','alias','purge'))


@dataclass(frozen=True)
class Grant:
    tenant:str
    principal:str
    operations:tuple[str,...]
    source_receipt:str
    predecessor:str
    override_protection:bool=False


class Lifecycle:
    def __init__(self,memory:Memory):self.memory=memory

    def prepare(self,slot,operation,grant:Grant,*,content=None,acl=None,reference=None,history=(),valid_until=None):
        m=self.memory;m._slot(slot)
        if (type(grant) is not Grant or operation not in OPERATIONS or operation not in grant.operations
                or type(grant.override_protection) is not bool or type(grant.operations) is not tuple
                or grant.predecessor!=m.identity or not grant.principal or not grant.source_receipt):
            raise PermissionError('current trusted lifecycle capability required')
        old=m._rows[slot]
        if old is None:raise ValueError('lifecycle operation requires an existing row')
        m._validate(old)
        f=old.fields
        if f['tenant']!=grant.tenant:raise PermissionError('foreign tenant')
        if f['protected'] and not grant.override_protection:raise PermissionError('protected record requires explicit management authority')
        if old.status=='purged' and operation not in ('acl','protect','unprotect'):
            raise PermissionError('purged resident object cannot be resurrected')
        if f['version']==MAX_VERSION or f['object_version']==MAX_VERSION or m.version==MAX_VERSION:
            raise OverflowError('monotone version exhausted')
        status,data,key,value=old.status,old.canonical,old.key,old.value
        f=dict(f);f.update(version=f['version']+1,object_version=f['object_version']+1,
                         parent=old.receipt,source_receipt=grant.source_receipt)
        if operation in ('protect','unprotect'):
            f['protected']=operation=='protect'
        elif operation=='acl':
            if type(acl) is not tuple or not acl or any(type(x) is not str or not x for x in acl):
                raise ValueError('complete nonempty ACL required')
            f['acl']=list(acl)
        elif operation=='validity':
            if type(valid_until) is not int or not 0 <= valid_until <= MAX_VERSION:
                raise ValueError('trusted bounded validity end required')
            f['valid_until']=valid_until
        elif operation in ('delete','quarantine','purge'):
            status={'delete':'tombstone','quarantine':'quarantine','purge':'purged'}[operation]
            data=b'';key=bytes(m.profile.key_width*4);value=bytes(m.profile.neural_value_bytes)
            f['semantic_qualified']=False
        elif operation=='rollback':
            if type(reference) is not Row:raise ValueError('verified prior complete row required')
            m._validate(reference)
            rf=reference.fields
            if reference.status!='live' or any(rf[k]!=f[k] for k in ('tenant','object_id','incarnation')):
                raise ValueError('rollback must restore this object incarnation')
            if type(history) is not tuple or len(history)+1>m.profile.max_candidates:
                raise ValueError('bounded committed history proof required')
            child=old
            for prior in history+(reference,):
                if type(prior) is not Row:raise ValueError('complete history rows required')
                m._validate(prior)
                pf=prior.fields;cf=child.fields
                if (cf['parent']!=prior.receipt or pf['object_version']!=cf['object_version']-1
                        or any(pf[k]!=cf[k] for k in ('tenant','object_id','incarnation'))):
                    raise ValueError('reference is not on the current committed parent chain')
                child=prior
            # Restore content, retain current authorization/protection and advance
            # both counters. A past ACL is never restored as a side effect.
            status,data='live',reference.canonical
        elif operation=='repair':
            if old.status!='quarantine':raise ValueError('repair requires quarantine')
            if type(content) is not bytes:raise ValueError('new validated canonical content required')
            decoded=json.loads(content)
            if type(decoded) is dict and set(decoded)=={'alias_object'}:
                raise PermissionError('repair cannot create a privileged alias')
            status,data='live',content
        elif operation=='alias':
            if type(content) is not bytes:raise ValueError('canonical alias target required')
            target=json.loads(content)
            if type(target) is not dict or set(target)!={'alias_object'} or type(target['alias_object']) is not str or not target['alias_object']:
                raise ValueError('bounded typed alias target required')
            if target['alias_object']==f['object_id']:raise ValueError('self alias')
            live=[r for r in m._rows if r is not None and r.status=='live' and r.fields['tenant']==f['tenant'] and r.fields['object_id']==target['alias_object']]
            if len(live)!=1 or grant.principal not in live[0].fields['acl']:
                raise PermissionError('alias target must be uniquely visible to manager')
            status,data='live',canonical(target)
        if operation in ('repair','rollback','alias'):
            if len(data)>m.profile.canonical_bytes or canonical(json.loads(data))!=data:
                raise ValueError('invalid lifecycle payload')
            key,value=m.encoder.encode(data)
            f['semantic_qualified']=m.semantic_gate is not None and m.semantic_gate(data,key,value) is True
        row=Row(status,data,key,value,wire(f));m._validate(row)
        return m._issue(slot,row)

    def resolve_alias(self,selector,*,principal,clock,max_hops=None):
        if max_hops is None:max_hops=self.memory.profile.memory_slots
        if type(max_hops) is not int or not 1<=max_hops<=self.memory.profile.memory_slots:
            raise ValueError('bounded alias traversal required')
        if selector.view!='canonical':raise ValueError('alias resolution is an explicit canonical read')
        m=self.memory;seen=set()
        for _ in range(max_hops):
            if selector.object_id in seen:raise ValueError('alias cycle')
            seen.add(selector.object_id)
            payload=m.read(selector,principal=principal,clock=clock)
            if payload is None:return None
            value=json.loads(payload)
            if type(value) is not dict or set(value)!={'alias_object'}:return payload
            choices=[s for s in m.selectors(tenant=selector.tenant,principal=principal,clock=clock,view='canonical') if s.object_id==value['alias_object']]
            if len(choices)!=1:return None
            selector=choices[0]
        raise ValueError('alias depth exhausted')
