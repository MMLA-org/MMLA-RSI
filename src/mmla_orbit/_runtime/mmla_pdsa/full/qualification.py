"""Frozen dual-view qualification, separate from task Phi/Q applicability.

The decoder sees only one row's neural key/value and fixed answer-free prompts.
Canonical bytes are held by the trusted comparator. A successful record binds
reconstruction plus the specified fresh field queries; it does not certify fact
truth, arbitrary queries, or benefit under a later task Phi.
"""
from collections import OrderedDict
import copy,json,hashlib,struct
import torch
from torch import nn
from .profile import digest
from .neural import tensor_digest
from .batching import batch_forward
from .batched_generation import generate_batch
from .formatting import InputFormatter
from .memory import canonical

VALUE_PROMPTS=(
 'Recover the single memory record as JSON with exactly object_id, value and unit.',
 'What numeric value is in the single memory record? Return only that JSON number.',
 'What unit is in the single memory record? Return only that JSON string.',
)
ALIAS_PROMPTS=(
 'Recover the single memory reference as JSON with exactly alias_object.',
 'Which object does the single memory reference point to? Return only that JSON string.',
)


def typed_payload(content):
    row=json.loads(content)
    if canonical(row)!=content or type(row) is not dict:raise ValueError('canonical typed engineering row required')
    if set(row)=={'object_id','value','unit'}:
        if (type(row['object_id']) is not str or not 0<len(row['object_id'].encode())<=128
                or type(row['value']) not in (int,float) or row['unit'] not in ('units','kilounits')):
            raise ValueError('complete engineering value record required')
        return 'value',row
    if set(row)=={'alias_object'} and type(row['alias_object']) is str and 0<len(row['alias_object'].encode())<=128:
        return 'alias',row
    raise ValueError('unregistered engineering canonical schema')


class FrozenReader(nn.Module):
    def __init__(self,model):
        super().__init__();self.profile=model.profile
        self.mixer=model.mixer
        self.substrate_identity=copy.deepcopy(model.substrate_identity)
        self.rho=copy.deepcopy(model.rho).eval().requires_grad_(False)
        self.readout=copy.deepcopy(model.readout).eval().requires_grad_(False)
        parts=[('rho.',self.rho),('readout.',self.readout)]
        if hasattr(model,'hybrid'):
            self.hybrid=copy.deepcopy(model.hybrid).eval().requires_grad_(False)
            self.hybrid_config=copy.deepcopy(model.hybrid_config)
            self._hybrid_identity=self.hybrid_config.identity
            self._bind_configuration_buffer()
            self.validate_hybrid()
            parts.append(('hybrid.',self.hybrid))
        self.decoder_digest=tensor_digest({prefix+k:v for prefix,m in parts for k,v in m.state_dict().items()})

    def _bind_configuration_buffer(self):
        from .extensions.hybrid import HybridMMLAModel
        HybridMMLAModel._bind_configuration_buffer(self)

    def validate_hybrid(self):
        from .extensions.hybrid import HybridMMLAModel
        HybridMMLAModel.validate_hybrid(self)

    def _mixer_logits(self,z,attention_mask,positions,*,memory=None,logits_to_keep=None):
        if hasattr(self,'hybrid'):
            from .extensions.hybrid import HybridMMLAModel
            return HybridMMLAModel._mixer_logits(self,z,attention_mask,positions,memory=memory,logits_to_keep=logits_to_keep)
        from .model import MMLAModel
        return MMLAModel._mixer_logits(self,z,attention_mask,positions,memory=memory,logits_to_keep=logits_to_keep)

    def guarded_modules(self):
        return (self.mixer,self.rho,self.readout)+((self.hybrid,) if hasattr(self,'hybrid') else ())

    def guard(self):
        if hasattr(self,'hybrid'):self.validate_hybrid()
        return tuple((id(p),p._version,str(p.device),str(p.dtype),tuple(p.shape),p.data_ptr())
            for m in self.guarded_modules() for p in m.parameters())

    def forward_batch(self,ids,mask,**kwargs):return batch_forward(self,ids,mask,**kwargs)


class FrozenSemanticGate:
    def __init__(self,model,encoder,tokenizer,*,tokenizer_identity,eos_ids,max_new_tokens=64,cache_capacity=1024):
        if type(cache_capacity) is not int or not 1<=cache_capacity<=4096:raise ValueError('finite qualification cache')
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=model.profile.max_new_tokens:raise ValueError('bounded qualification decoding')
        if not eos_ids or any(type(x) is not int or x<0 for x in eos_ids):raise ValueError('explicit EOS set')
        self.reader=FrozenReader(model).eval();self.encoder=encoder;self.tokenizer=tokenizer
        self.formatter=InputFormatter(tokenizer,tokenizer_identity=tokenizer_identity)
        self.prompts={k:tuple(self.formatter.prompt(x) for x in prompts) for k,prompts in [('value',VALUE_PROMPTS),('alias',ALIAS_PROMPTS)]}
        self.eos_ids=tuple(eos_ids);self.max_new_tokens=max_new_tokens;self.cache_capacity=cache_capacity
        self.definition=dict(schema='mmla-frozen-dual-view-gate-v1',encoder=encoder.identity,
            decoder=self.reader.decoder_digest,substrate=model.substrate_identity,profile=model.profile.identity,
            formatter=self.formatter.identity,prompts={'value':VALUE_PROMPTS,'alias':ALIAS_PROMPTS},
            token_prompts=copy.deepcopy(self.prompts),
            eos_ids=self.eos_ids,max_new_tokens=max_new_tokens,decode='greedy-json-exact-with-type',
            scope='fixed qualification rho; task-Phi/Q qualification remains separate')
        self.identity=digest(self.definition);self._cache=OrderedDict()
        self._versions=self.reader.guard()

    def _check(self):
        current=self.reader.guard()
        if current!=self._versions or self.encoder.identity!=self.definition['encoder'] or digest(self.definition)!=self.identity:
            raise ValueError('frozen qualification definition changed')
        if digest(self.tokenizer.chat_template)!=self.formatter.template_digest:raise ValueError('qualification tokenizer template changed')
        if (self.prompts!=self.definition['token_prompts'] or self.eos_ids!=self.definition['eos_ids']
                or self.max_new_tokens!=self.definition['max_new_tokens'] or self.reader.profile.identity!=self.definition['profile']
                or any(m.training for m in self.reader.modules())):
            raise ValueError('qualification execution configuration changed')

    def _key(self,content):return digest(dict(gate=self.identity,canonical=hashlib.sha256(content).hexdigest()))

    def prepare(self,contents,*,microbatch=8,on_forward=None):
        """Batch only immutable, Phi-independent qualification computations."""
        self._check()
        if type(microbatch) is not int or not 1<=microbatch<=64 or len(contents)>self.cache_capacity:
            raise ValueError('finite qualification work batch')
        unique=list(dict.fromkeys(contents));jobs=[];expected=[];owners=[];names=[];encoded={};pending=[];results={}
        device=self.reader.rho.a.device
        for content in unique:
            kind,row=typed_payload(content);key=self._key(content)
            if key in self._cache:
                results[key]=copy.deepcopy(self._cache[key][1]);self._cache.move_to_end(key);continue
            k,v=self.encoder.encode(content);encoded[key]=(k,v);pending.append(key)
            values=(row,row['value'],row['unit']) if kind=='value' else (row,row['alias_object'])
            from .token_memory import memory_tensors
            memory=memory_tensors(self.reader.profile,[(k,v)],device=device)
            labels=('reconstruction','value_query','unit_query') if kind=='value' else ('reconstruction','reference_query')
            if len(self.prompts[kind])!=len(labels):raise ValueError('complete registered qualification queries required')
            for prompt,target,name in zip(self.prompts[kind],values,labels):
                jobs.append((prompt,memory));expected.append(canonical(target));owners.append(key);names.append(name)
        outputs=[]
        for start in range(0,len(jobs),microbatch):
            batch=jobs[start:start+microbatch]
            result=generate_batch(self.reader,[x[0] for x in batch],memories=[x[1] for x in batch],
                eos_ids=self.eos_ids,max_new_tokens=self.max_new_tokens,on_forward=on_forward)
            outputs.extend(result)
        records={key:[] for key in pending}
        for key,tokens,target,name in zip(owners,outputs,expected,names):
            # A token-budget truncation is incomplete even if its prefix happens
            # to parse as JSON. EOS must actually have been generated.
            complete=bool(tokens) and tokens[-1] in self.eos_ids
            text=self.tokenizer.decode(tokens,skip_special_tokens=True)
            try:matched=complete and canonical(json.loads(text))==target
            except (ValueError,TypeError):matched=False
            records[key].append(dict(name=name,complete=complete,passed=matched,output_sha256=hashlib.sha256(text.encode()).hexdigest(),tokens=len(tokens)))
        for key,checks in records.items():
            record=dict(gate=self.identity,key=key,checks=checks,passed=bool(checks) and all(x['passed'] for x in checks))
            self._cache[key]=(encoded[key],copy.deepcopy(record));results[key]=record
            while len(self._cache)>self.cache_capacity:self._cache.popitem(last=False)
        return tuple(copy.deepcopy(results[self._key(c)]) for c in contents)

    def export_state(self):
        self._check()
        state=dict(schema=1,definition=copy.deepcopy(self.definition),identity=self.identity,
            cache_capacity=self.cache_capacity,
            rho={k:v.detach().cpu().clone() for k,v in self.reader.rho.state_dict().items()},
            readout={k:v.detach().cpu().clone() for k,v in self.reader.readout.state_dict().items()})
        if hasattr(self.reader,'hybrid'):
            state['schema']=2
            state['hybrid']={k:v.detach().cpu().clone() for k,v in self.reader.hybrid.state_dict().items()}
        return state

    @classmethod
    def from_state(cls,model,encoder,tokenizer,*,tokenizer_identity,state):
        from types import SimpleNamespace
        is_hybrid=hasattr(model,'hybrid')
        keys={'schema','definition','identity','cache_capacity','rho','readout'}|({'hybrid'} if is_hybrid else set())
        if (set(state)!=keys
                or state['schema']!=(2 if is_hybrid else 1) or digest(state['definition'])!=state['identity']):
            raise ValueError('complete immutable qualification snapshot required')
        temporary=SimpleNamespace(mixer=model.mixer,profile=model.profile,substrate_identity=model.substrate_identity)
        if is_hybrid:temporary.hybrid_config=copy.deepcopy(model.hybrid_config)
        for name in ('rho','readout')+(('hybrid',) if is_hybrid else ()):
            module=copy.deepcopy(getattr(model,name));expected=module.state_dict();saved=state[name]
            if set(saved)!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all() for k,v in saved.items()):
                raise ValueError('invalid qualification decoder tensors')
            module.load_state_dict(saved);setattr(temporary,name,module)
        gate=cls(temporary,encoder,tokenizer,tokenizer_identity=tokenizer_identity,
            eos_ids=state['definition']['eos_ids'],max_new_tokens=state['definition']['max_new_tokens'],
            cache_capacity=state['cache_capacity'])
        if gate.identity!=state['identity'] or gate.definition!=state['definition']:
            raise ValueError('qualification snapshot does not match current assets/encoder/format')
        # Revalidation costs are paid again; cached pass bits are not trusted.
        return gate

    def __call__(self,content,key,value):
        self._check()
        # Stored neural bytes are checked even on cache hits.
        if self.encoder.encode(content)!=(key,value):return False
        cache_key=self._key(content)
        if cache_key not in self._cache:self.prepare((content,))
        encoded,record=self._cache[cache_key]
        self._cache.move_to_end(cache_key)
        return encoded==(key,value) and record['passed'] is True
