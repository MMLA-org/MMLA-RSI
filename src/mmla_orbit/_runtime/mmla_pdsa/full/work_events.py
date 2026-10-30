"""Ephemeral accounting for ordinary HF and explicit retained-layer execution.

The shared frozen mixer is the accounting owner, including qualification readers.
No weights, activation caches or work counters are serialized into the model.
Original forwards report through a pre-hook; explicit layer execution reports
once at its own entry. Each subscription receives one event, never both.
"""
from weakref import WeakKeyDictionary

_EXPLICIT=WeakKeyDictionary()


class Subscription:
    def __init__(self,mixer,callback):
        self.mixer=mixer;self.callback=callback;self.closed=False
        def original(module,args,kwargs):
            mask=kwargs.get('attention_mask')
            if mask is None or mask.ndim!=2:raise ValueError('work meter requires original binary sequence mask')
            callback(mask)
        self.hook=mixer.register_forward_pre_hook(original,with_kwargs=True)
        _EXPLICIT.setdefault(mixer,[]).append(callback)

    def remove(self):
        if self.closed:return
        self.hook.remove()
        callbacks=_EXPLICIT.get(self.mixer,[])
        callbacks.remove(self.callback)
        if not callbacks:_EXPLICIT.pop(self.mixer,None)
        self.closed=True


def record_explicit_mixer(mixer,attention_mask):
    """Actual hybrid execution entry, before its first retained decoder layer."""
    if attention_mask.ndim!=2:raise ValueError('original sequence mask required for explicit mixer work')
    for callback in tuple(_EXPLICIT.get(mixer,())):callback(attention_mask)
