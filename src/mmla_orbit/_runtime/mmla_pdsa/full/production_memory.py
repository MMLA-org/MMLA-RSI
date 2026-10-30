"""Production AMR identity binds encoder, semantic gate and typed schema.

Generic Memory remains usable for historical engineering fixtures. This wrapper
adds the selected production profile's strict row validation without changing
its serialized metadata fields. It grants no authority to the neural model.
"""
from .profile import digest
from .memory import Memory
from .qualification import FrozenSemanticGate,typed_payload


class QualifiedCanonicalEncoder:
 def __init__(self,encoder,gate):
  if type(gate) is not FrozenSemanticGate or gate.encoder is not encoder:
   raise ValueError('the actual frozen gate and its base encoder are required')
  gate._check()
  self.encoder,self.gate=encoder,gate
  self.definition=dict(schema='mmla-qualified-engineering-encoder-v1',base_encoder=encoder.identity,
                       gate=gate.identity,canonical='object_id/value/unit or alias_object')
  self.identity=digest(self.definition)

 def _check(self):
  self.gate._check()
  if (self.encoder is not self.gate.encoder or self.encoder.identity!=self.definition['base_encoder']
      or self.gate.identity!=self.definition['gate'] or digest(self.definition)!=self.identity):
   raise ValueError('production encoder/gate/schema changed')

 def validate_export(self):self._check()

 def encode(self,content):
  self._check();typed_payload(content)
  return self.encoder.encode(content)

 def validate_row(self,row):
  self._check()
  if row.fields['encoder']!=self.identity:raise ValueError('row was not qualified by this production definition')
  if row.status=='live':
   kind,value=typed_payload(row.canonical)
   if kind=='value' and value['object_id']!=row.fields['object_id']:
    raise ValueError('canonical object identity differs from authoritative metadata')
   if row.fields['semantic_qualified'] is not True or self.gate(row.canonical,row.key,row.value) is not True:
    raise ValueError('production live row failed frozen dual-view qualification')

 def qualification_record_key(self,row):
  self.validate_row(row)
  if row.status!='live':return None
  # An index, not independent evidence: the successful check is also bound to
  # row.receipt and must be reconstructible by this frozen gate.
  return self.gate._key(row.canonical)


def production_memory(profile,encoder,gate):
 if profile!=gate.reader.profile or profile!=encoder.profile:raise ValueError('production profile mismatch')
 wrapper=QualifiedCanonicalEncoder(encoder,gate)
 return Memory(profile,wrapper,semantic_gate=gate)
