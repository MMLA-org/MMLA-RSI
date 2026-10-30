"""Historical observation capsules for the ALFWorld text projection.

Qualification here is exact token-channel reconstruction of a typed historical
observation, not a learned answer/query test or a claim that it remains true now.
A trusted runner supplies source provenance; this codec never reads task files.
"""
from dataclasses import dataclass
import json

from .alfworld_episode import _sha, _digest
from .alfworld_formatting import _controls
from .memory import Authority, Memory, Proposal, canonical
from .token_memory import unpack_value


@dataclass(frozen=True)
class ObservationBatch:
    step: int
    observation_sha256: str
    source_receipt: str
    chunks: tuple[bytes, ...]
    codec_identity: str

    @property
    def identity(self):
        return _digest(dict(step=self.step,observation=self.observation_sha256,
            source=self.source_receipt,chunks=[x.hex() for x in self.chunks],codec=self.codec_identity))


class ObservationCodec:
    """Complete bounded fragments, with exact character offsets and step time.

    Splitting is independent of labels, rewards and hidden state. Nothing is
    dropped to make a row fit: failure to represent the whole observation is an
    explicit operational failure before any write decisions are scored.
    """
    def __init__(self,tokenizer,profile,*,tokenizer_identity,max_observation_bytes=65536,max_chunks=512):
        _sha(tokenizer_identity)
        if (not profile.memory_tokens or not tokenizer.is_fast
                or type(max_observation_bytes) is not int or not 1<=max_observation_bytes<=65536
                or type(max_chunks) is not int or not 1<=max_chunks<=512):
            raise ValueError('finite token-preserving observation profile required')
        self.tokenizer,self.profile=tokenizer,profile
        self._controls=_controls(tokenizer)
        self.max_observation_bytes,self.max_chunks=max_observation_bytes,max_chunks
        self._binding=(profile.identity,max_observation_bytes,max_chunks,self._controls)
        self.identity=_digest(dict(schema='orbit-observed-fragment-v1',profile=profile.identity,
            tokenizer=tokenizer_identity,controls=self._controls,fields=['observed_step','char_offset','text'],
            max_observation_bytes=max_observation_bytes,max_chunks=max_chunks,
            split='256-codepoint-prefix; halve-until-fit; preserve-every-codepoint',
            meaning='historical-local-observation-fragment; no-current-world-truth-claim'))

    def _text(self,text):
        if (self.profile.identity,self.max_observation_bytes,self.max_chunks,_controls(self.tokenizer))!=self._binding:
            raise ValueError('observation codec configuration changed')
        if type(text) is not str or not text or len(text.encode())>self.max_observation_bytes:
            raise ValueError('whole bounded observed text required')
        if any(x in text for x in self._controls):raise ValueError('control delimiter in observation')

    def validate(self,content):
        if type(content) is not bytes or not 0<len(content)<=self.profile.canonical_bytes:
            raise ValueError('bounded canonical fragment required')
        row=json.loads(content)
        if canonical(row)!=content or type(row) is not dict or set(row)!={'observed_step','char_offset','text'}:
            raise ValueError('exact historical observation schema required')
        for k in ('observed_step','char_offset'):
            if type(row[k]) is not int or not 0<=row[k]<2**32:raise ValueError('finite time/offset required')
        self._text(row['text'])
        ids=tuple(self.tokenizer.encode(content.decode(),add_special_tokens=False))
        if (not 0<len(ids)<=self.profile.memory_tokens
                or any(t in self.profile.memory_forbidden_tokens or not 0<=t<self.profile.memory_vocab_size for t in ids)):
            raise ValueError('complete capsule token capacity exceeded')
        if self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).encode()!=content:
            raise ValueError('token channel does not reconstruct original canonical bytes')
        return row,ids

    def split(self,observation,*,step,source_receipt):
        import hashlib
        self._text(observation);_sha(source_receipt)
        if type(step) is not int or not 0<=step<2**32:raise ValueError('finite actual step required')
        offset,chunks=0,[]
        while offset<len(observation):
            if len(chunks)>=self.max_chunks:raise ValueError('whole observation exceeds fragment cap')
            size=min(256,len(observation)-offset)
            while True:
                content=canonical(dict(observed_step=step,char_offset=offset,text=observation[offset:offset+size]))
                try:self.validate(content);break
                except ValueError:
                    if size==1:raise
                    size=max(1,size//2)
            chunks.append(content);offset+=size
        result=ObservationBatch(step,hashlib.sha256(observation.encode()).hexdigest(),source_receipt,tuple(chunks),self.identity)
        self.verify_batch(result)
        return result

    def verify_batch(self,batch):
        import hashlib
        if (type(batch) is not ObservationBatch or batch.codec_identity!=self.identity
                or type(batch.chunks) is not tuple or not 0<len(batch.chunks)<=self.max_chunks):
            raise ValueError('complete bound observation batch required')
        _sha(batch.observation_sha256);_sha(batch.source_receipt)
        if type(batch.step) is not int or not 0<=batch.step<2**32:raise ValueError('finite actual step required')
        offset,parts=0,[]
        for content in batch.chunks:
            row,_=self.validate(content)
            if row['observed_step']!=batch.step or row['char_offset']!=offset:
                raise ValueError('noncontiguous or mixed-time observation fragments')
            parts.append(row['text']);offset+=len(row['text'])
        observation=''.join(parts);self._text(observation)
        if hashlib.sha256(observation.encode()).hexdigest()!=batch.observation_sha256:
            raise ValueError('partial or changed observation payload')
        return observation

    def proposal(self,batch,index,*,tenant,principal):
        """All payload validation precedes the per-slot Memory.assemble calls.

        Caller holds the actual source receipt; returning Authority here is a
        trusted runner helper, not authentication of untrusted model text.
        """
        self.verify_batch(batch)
        if type(index) is not int or not 0<=index<len(batch.chunks):raise ValueError('fragment index')
        authority=Authority(tenant,(principal,),batch.source_receipt,f'observation:{batch.step}:{index}')
        return Proposal(batch.chunks[index]),authority


class ObservationEncoder:
    """Typed adapter over the original immutable numerical/capsule encoder."""
    def __init__(self,codec,encoder):
        if codec.profile.identity!=encoder.profile.identity:raise ValueError('matching memory profiles required')
        _sha(encoder.identity)
        self.codec,self.encoder,self.profile=codec,encoder,codec.profile
        self._encoder_identity=encoder.identity;self._codec_identity=codec.identity
        self.identity=_digest(dict(schema='orbit-observation-encoder-v1',encoder=encoder.identity,codec=codec.identity,
            qualification='exact-token-reconstruction-only; no-neural-query-claim'))

    def encode(self,content):
        if self.encoder.identity!=self._encoder_identity or self.codec.identity!=self._codec_identity:
            raise ValueError('observation encoder binding changed')
        _,expected=self.codec.validate(content)
        key,value=self.encoder.encode(content)
        _,tokens=unpack_value(self.profile,value)
        if tokens!=expected:raise ValueError('stored capsule differs from the exact original text')
        return key,value

    def validate_row(self,row):
        if row.status!='live':return
        record,_=self.codec.validate(row.canonical)
        f=row.fields
        if (not f['semantic_qualified'] or f['encoder']!=self.identity
                or not f['object_id'].startswith(f"observation:{record['observed_step']}:")
                or len(f['object_id'].split(':'))!=3 or not f['object_id'].split(':')[-1].isdigit()):
            raise ValueError('live observation row lacks its typed capsule qualification')
        _sha(f['source_receipt'])


class CapsuleFidelityGate:
    """Versioned reconstruction predicate, deliberately not FrozenSemanticGate.

    Memory's legacy field name semantic_qualified stores this explicit typed
    predicate only. The encoder identity pins its scope; neural task usefulness
    must still be measured on complete tasks and the current reader/Phi.
    """
    def __init__(self,encoder):
        self.encoder=encoder;self._encoder_identity=encoder.identity
        self.identity=_digest(dict(schema='orbit-capsule-fidelity-v1',encoder=encoder.identity,
            checks=['typed-record','exact-stored-tokens','frozen-reencoding'],
            scope='token-channel reconstruction; not learned semantic/query qualification'))

    def __call__(self,content,key,value):
        if self.encoder.identity!=self._encoder_identity:raise ValueError('fidelity encoder changed')
        return self.encoder.encode(content)==(key,value)


def observation_memory(codec,encoder):
    typed=ObservationEncoder(codec,encoder)
    return Memory(codec.profile,typed,semantic_gate=CapsuleFidelityGate(typed))
