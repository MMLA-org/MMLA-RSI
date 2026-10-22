"""Process-local identity reuse for one unchanged resident proposal model.

Only the full tensor digest is cached. No activations, gradients, KV, fast state,
memory, acceptance certificate or role state are shared through this binding.
Construct it from the real model once; rebind after a slow parameter change.
The owner must serialize writers and keep the verified tokenizer immutable.
Raw-storage/.data writes and concurrent mutation are unsupported, as in the
original proposal reader; this is not a cross-process trust certificate.
"""
from dataclasses import dataclass, asdict
import os
import weakref
import torch
from .neural import tensor_digest
from .rules import identity


def tensor_stamp(tensor):
    return (id(tensor), tensor._version, tensor.data_ptr(),
                  str(tensor.device), str(tensor.dtype), tuple(tensor.shape),
                  tuple(tensor.stride()), tensor.storage_offset(), tensor.requires_grad)


def model_guard(model):
    return tuple((name, tensor_stamp(tensor))
                 for name, tensor in tuple(model.named_parameters(remove_duplicate=False))
                 + tuple(model.named_buffers(remove_duplicate=False)))


def role_guard(model):
    # Historical initialization is deliberately outside state_dict. A rebind
    # changes incarnation even when every registered tensor byte is unchanged.
    rho = model.rho
    initial = rho.initialization
    if (type(initial) is not dict or set(initial) != {'a', 'b', 'seed'}
            or type(initial['seed']) is not int
            or any(not isinstance(initial[k], torch.Tensor) for k in ('a', 'b'))):
        raise ValueError('complete historical residual initialization required')
    return (id(rho), rho._incarnation, id(initial), initial['seed'],
            tensor_stamp(initial['a']), tensor_stamp(initial['b']))


def module_guard(model):
    # state_dict keeps registered aliases and omits nonpersistent buffers.
    # Named-tensor iterators alone cannot detect a persistence-only change.
    return tuple((name, id(module), type(module), module.training,
                  tuple((key, id(value)) for key, value in module._parameters.items()),
                  tuple((key, id(value)) for key, value in module._buffers.items()),
                  tuple((key, id(value)) for key, value in module._modules.items()),
                  tuple(sorted(module._non_persistent_buffers_set)),
                  tuple((key, id(value)) for key, value in module._state_dict_pre_hooks.items()),
                  tuple((key, id(value)) for key, value in module._state_dict_hooks.items()))
                 for name, module in model.named_modules(remove_duplicate=False))


def configuration(model, tokenizer_identity):
    return identity(dict(profile=asdict(model.profile), substrate=model.substrate_identity,
                         mixer=model.mixer.config.to_dict(),
                         hybrid=None if not hasattr(model, 'hybrid_config') else asdict(model.hybrid_config),
                         tokenizer=tokenizer_identity, torch=torch.__version__))


@dataclass(frozen=True, init=False, eq=False)
class FrozenReaderBinding:
    _pid: int
    _owner: object
    _tokenizer: object
    _tokenizer_identity: str
    _guard: tuple
    _role_guard: tuple
    _module_guard: tuple
    _configuration: str
    parameters_identity: str
    initialization_identity: str
    reader_identity: str

    def __init__(self, model, tokenizer, *, tokenizer_identity):
        pid = os.getpid()
        if (type(tokenizer_identity) is not str or not tokenizer_identity
                or any(module.training for module in model.modules())
                or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')):
            raise ValueError('unchanged eval model and explicit tokenizer identity required')
        guard = model_guard(model)
        role = role_guard(model)
        modules = module_guard(model)
        config = configuration(model, tokenizer_identity)
        parameters = tensor_digest(model.state_dict())
        initialization = tensor_digest({k: model.rho.initialization[k] for k in ('a', 'b')})
        reader = identity(dict(parameters=parameters, profile=asdict(model.profile),
                               substrate=model.substrate_identity,
                               tokenizer=tokenizer_identity, torch=torch.__version__))
        # A changing model must not publish a digest assembled across versions.
        if (pid != os.getpid() or guard != model_guard(model) or role != role_guard(model)
                or modules != module_guard(model)
                or config != configuration(model, tokenizer_identity)):
            raise ValueError('resident reader changed during identity construction')
        for key, value in dict(_pid=pid, _owner=weakref.ref(model), _tokenizer=tokenizer,
                _tokenizer_identity=tokenizer_identity, _guard=guard,
                _role_guard=role, _module_guard=modules,
                _configuration=config, parameters_identity=parameters,
                initialization_identity=initialization,
                reader_identity=reader).items():
            object.__setattr__(self, key, value)

    def validate(self, model, tokenizer, *, tokenizer_identity):
        if (self._pid != os.getpid() or self._owner() is not model or self._tokenizer is not tokenizer
                or self._tokenizer_identity != tokenizer_identity
                or self._guard != model_guard(model)
                or self._role_guard != role_guard(model)
                or self._module_guard != module_guard(model)
                or self._configuration != configuration(model, tokenizer_identity)
                or any(module.training for module in model.modules())
                or torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')):
            raise ValueError('resident reader binding no longer matches the actual model')
        return self.reader_identity

    def __reduce_ex__(self, protocol):
        raise TypeError('resident reader binding is process-local; reconstruct from verified model')
