"""Frozen, versioned canonical projection; semantic qualification is separate."""
from collections import OrderedDict
import copy
import torch

from .neural import tensor_digest
from .profile import digest
from .canonical_batch import canonical_vectors,CANONICAL_ALGORITHM


class FrozenCanonicalEncoder:
    def __init__(self, model, tokenizer, *, tokenizer_identity, cache_capacity=1024):
        if type(cache_capacity) is not int or not 1<=cache_capacity<=4096:raise ValueError('bounded canonical cache required')
        self.cache_capacity=cache_capacity;self._cache=OrderedDict()
        self.embedding = model.mixer.get_input_embeddings()
        self.tokenizer = tokenizer
        self.profile = model.profile
        # Frozen copies retain the actual model device; CPU is only the wire transfer.
        device=self.embedding.weight.device
        self.consolidator = copy.deepcopy(model.consolidator).to(device).eval().requires_grad_(False)
        self.key = copy.deepcopy(model.canonical_key).to(device).eval().requires_grad_(False)
        self.value = copy.deepcopy(model.canonical_value).to(device).eval().requires_grad_(False)
        self._backend=self._backend_identity()
        self.identity = digest(dict(substrate=model.substrate_identity, tokenizer=tokenizer_identity,
            profile=model.profile.identity, algorithm=CANONICAL_ALGORITHM, token_codec=('fp32-semantic-plus-u32-length-i32-token-sequence-v1' if model.profile.memory_tokens else None),
            device_type=device.type,dtype='float32',torch_version=str(torch.__version__),
            cuda_capability=list(torch.cuda.get_device_capability(device)) if device.type=='cuda' else None,
            matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
            modules=tensor_digest({prefix+k:v for prefix,m in
                (('consolidator.',self.consolidator),('key.',self.key),('value.',self.value))
                for k,v in m.state_dict().items()})))
        self._versions=tuple((id(p),p._version) for m in (self.embedding,self.consolidator,self.key,self.value) for p in m.parameters())

    @staticmethod
    def _backend_identity():
        return (bool(torch.backends.cuda.matmul.allow_tf32),bool(torch.backends.cudnn.allow_tf32))

    @torch.no_grad()
    def encode(self, canonical_bytes):
        if self._backend_identity()!=self._backend:raise ValueError('canonical backend configuration changed')
        if self._versions!=tuple((id(p),p._version) for m in (self.embedding,self.consolidator,self.key,self.value) for p in m.parameters()):raise ValueError('frozen canonical encoder changed')
        if type(canonical_bytes) is not bytes or not 0 < len(canonical_bytes) <= self.profile.canonical_bytes:
            raise ValueError('canonical bytes outside profile')
        if canonical_bytes in self._cache:
            self._cache.move_to_end(canonical_bytes);return self._cache[canonical_bytes]
        ids=self.tokenizer.encode(canonical_bytes.decode('utf-8'), add_special_tokens=False)
        if not ids or len(ids)>self.profile.local_tokens:
            raise ValueError('canonical token capacity exceeded')
        ids=torch.tensor(ids,dtype=torch.long)
        keys,values=canonical_vectors(self.embedding,self.consolidator,self.key,self.value,[ids],pad_id=1)
        k,v=keys.reshape(-1).cpu(),values.reshape(-1).cpu()
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise FloatingPointError('nonfinite canonical encoding')
        # Explicit little-endian FP32 is the canonical wire representation.
        from .token_memory import pack_value
        result=k.numpy().astype('<f4').tobytes(),pack_value(self.profile,v.numpy().astype('<f4').tobytes(),ids.tolist())
        self._cache[canonical_bytes]=result
        while len(self._cache)>self.cache_capacity:self._cache.popitem(last=False)
        return result
