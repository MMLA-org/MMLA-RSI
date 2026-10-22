"""Bounded process-local digest reuse for single-writer tensor state.

This stores derived strings outside the owner's __dict__, so checkpoint fields
and identity bytes stay unchanged. Registered writers must use versioned tensor
operations (copy_, optimizer steps, load_state_dict); .data/external-storage
writes and concurrent mutation during a read are unsupported, as for slow rho.
No model activation or gradient is cached.
"""
from collections import OrderedDict
from threading import RLock
import weakref


def _metadata(tensors):
    return tuple((t._version, str(t.device), str(t.dtype), tuple(t.shape),
                  tuple(t.stride()), t.storage_offset(), t.data_ptr()) for t in tensors)


def _same_objects(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


class TensorIdentityMemo:
    def __init__(self, capacity=256):
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError('bounded positive identity cache capacity')
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = RLock()

    def _expired(self, key, ref):
        with self._lock:
            old = self._entries.get(key)
            if old is not None and old[0] is ref:
                del self._entries[key]

    def read(self, owner, read_tensors, digest):
        """Reuse only identical tensor objects and their complete version guard.

        Callbacks are used synchronously and never retained. Tensor references
        in an entry prevent Python object-ID/storage reuse until invalidation;
        the owner is weakly held and never gains a serialized cache field.
        """
        key = id(owner)
        with self._lock:
            tensors = tuple(read_tensors())
            guard = _metadata(tensors)
            old = self._entries.get(key)
            if (old is not None and old[0]() is owner
                    and _same_objects(old[1], tensors) and old[2] == guard):
                self._entries.move_to_end(key)
                return old[3]
            value = digest(tensors)
            after = tuple(read_tensors())
            if not _same_objects(tensors, after) or _metadata(after) != guard:
                self._entries.pop(key, None)
                raise RuntimeError('tensor state changed during identity digest')
            if type(value) is not str:
                raise TypeError('identity digest must return a string')
            ref = weakref.ref(owner, lambda ref, key=key: self._expired(key, ref))
            self._entries[key] = (ref, tensors, guard, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
            return value
