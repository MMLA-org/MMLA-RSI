"""Immutable complete-task checkpoints at closed outer-update boundaries.

Reuses the existing plain-data artifact container. Frozen mixer bytes stay in
original hash-bound artifacts; every new module, learned rho, AdamW moment,
initialization, schedule position and owned RNG is saved. No live task, decoder
cache or partially accumulated gradient is resumed. Loading does not authorize
execution: a caller must reconcile the saved ledger with its external supervisor.
"""
from dataclasses import asdict, dataclass
import copy
import math
import random
import torch
from .artifacts import save_artifact, load_artifact
from .module_training import rng_snapshot, restore_rng
from .local_rng import validate_cuda
from .neural import tensor_digest
from .profile import digest, wire
from .outcome_trajectory import OutcomeTrajectoryCore

KIND = 'orbit-complete-task-outer-checkpoint-v1'

@dataclass(frozen=True)
class CheckpointBinding:
    contract: str
    schedule: str
    partition: str
    general: str
    original_model: str
    original_sft: str
    runtime: str
    source: str
    seed: int
    objective: str

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in ('seed', 'objective'): continue
            if type(value) is not str or len(value) != 64:
                raise ValueError('exact original content identity required')
            int(value, 16)
        if type(self.seed) is not int or self.seed not in (17, 29, 43):
            raise ValueError('one fixed paired initialization required')
        if self.objective not in ('static', 'adapted'):
            raise ValueError('fixed training objective required')


def _core(core, binding, active_episodes):
    if type(core) is not OutcomeTrajectoryCore or type(binding) is not CheckpointBinding:
        raise TypeError('exact complete-task core and binding required')
    core._check()
    if (type(active_episodes) is not int or active_episodes != 0 or core._batch is not None
            or core.config.binding != binding.contract or core.config.objective != binding.objective):
        raise ValueError('same contract and closed outer boundary required')
    return core.model


def _ledger(value):
    fields = {'run_id', 'started_unix', 'deadline_unix', 'wall_seconds', 'CPU_seconds',
              'GPU_seconds', 'events_count', 'events_sha256'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('complete cumulative external resource receipt required')
    if type(value['run_id']) is not str or not 0 < len(value['run_id']) <= 128:
        raise ValueError('original run id required')
    for key in ('started_unix', 'wall_seconds', 'CPU_seconds', 'GPU_seconds'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError('finite nonnegative cumulative cost required')
    deadline=value['deadline_unix']
    if deadline is not None and (type(deadline) not in (int,float) or not math.isfinite(deadline) or deadline<=value['started_unix']):
        raise ValueError('finite deadline or explicitly workload-bounded receipt')
    if type(value['events_count']) is not int or value['events_count'] < 0:
        raise ValueError('complete event count required')
    if type(value['events_sha256']) is not str or len(value['events_sha256']) != 64:
        raise ValueError('bound ledger byte hash required')
    int(value['events_sha256'], 16)
    wire(value)


def _rng(state):
    if type(state) is not dict or set(state) != {'torch', 'python', 'cuda'}:
        raise ValueError('complete owning-process RNG required')
    torch.Generator(device='cpu').set_state(state['torch'])
    random.Random().setstate(state['python'])
    validate_cuda(state['cuda'])


def _initialization(value, modules):
    if type(value) is not dict or set(value) != {'a', 'b', 'seed'} or type(value['seed']) is not int:
        raise ValueError('original nonzero-A/zero-B initialization record required')
    for name in ('a', 'b'):
        x, target = value[name], modules['rho.'+name]
        if (not torch.is_tensor(x) or x.shape != target.shape or x.dtype != target.dtype
                or not bool(torch.isfinite(x).all())):
            raise ValueError('complete finite initialization tensors required')


def save_checkpoint(core, destination, *, binding, ledger, active_episodes):
    model = _core(core, binding, active_episodes)
    _ledger(ledger)
    modules = model.new_state_dict()
    if any(not bool(torch.isfinite(x).all()) for x in modules.values()):
        raise FloatingPointError('nonfinite model checkpoint')
    trainer = core.export()
    _initialization(model.rho.initialization, modules)
    payload = dict(schema=KIND, binding=asdict(binding), next_batch=core.steps,
        profile=model.profile.identity, substrate=copy.deepcopy(model.substrate_identity),
        modules=modules, module_identity=tensor_digest(modules),
        module_status=copy.deepcopy(model.module_status), initialization=copy.deepcopy(model.rho.initialization),
        trainer=trainer, rng=rng_snapshot(), ledger=copy.deepcopy(ledger),
        active_episodes=0, boundary='all environments closed; no pending Phi/M or gradients')
    # save_artifact clones tensors to CPU, fsyncs and publishes without overwrite.
    receipt = save_artifact(destination, payload, kind=KIND)
    return dict(**receipt, binding=digest(asdict(binding)), next_batch=core.steps,
                model_identity=payload['module_identity'])


def load_checkpoint(core, directory, *, identity, binding, reconcile_ledger, active_episodes):
    model = _core(core, binding, active_episodes)
    if core.steps or core.optimizer.state or not callable(reconcile_ledger):
        raise ValueError('fresh core and external cumulative-budget reconciliation required')
    payload = load_artifact(directory, kind=KIND, identity=identity)
    fields = {'schema', 'binding', 'next_batch', 'profile', 'substrate', 'modules', 'module_identity',
              'module_status', 'initialization', 'trainer', 'rng', 'ledger', 'active_episodes', 'boundary'}
    if (type(payload) is not dict or set(payload) != fields or payload['schema'] != KIND
            or payload['binding'] != asdict(binding) or payload['profile'] != model.profile.identity
            or payload['substrate'] != model.substrate_identity or payload['active_episodes'] != 0
            or type(payload['next_batch']) is not int
            or not 0 <= payload['next_batch'] <= core.config.max_steps
            or payload['next_batch'] != payload['trainer']['steps']):
        raise ValueError('exact model/data/runtime/recipe and completed cursor required')
    current = model.new_state_dict(); modules = payload['modules']
    if (type(modules) is not dict or set(modules) != set(current)
            or tensor_digest(modules) != payload['module_identity']):
        raise ValueError('complete model modules required')
    for name, target in current.items():
        x = modules[name]
        if not torch.is_tensor(x) or x.shape != target.shape or x.dtype != target.dtype or not bool(torch.isfinite(x).all()):
            raise ValueError('model tensor layout/value mismatch')
    status = payload['module_status']
    if type(status) is not dict or set(status) != set(model.module_status) or any(type(x) is not str for x in status.values()):
        raise ValueError('complete module status required')
    _initialization(payload['initialization'], modules); _rng(payload['rng']); _ledger(payload['ledger'])
    # This mandatory caller verifies the original admission, start/deadline and
    # monotone external cost log, including work after this last valid boundary.
    # Returning a rewritten/discounted ledger is not accepted by this loader.
    if reconcile_ledger(copy.deepcopy(payload['ledger'])) is not True:
        raise ValueError('external budget history did not authorize this restoration')
    before = dict(modules=current, initialization=copy.deepcopy(model.rho.initialization),
                  status=copy.deepcopy(model.module_status), optimizer=copy.deepcopy(core.optimizer.state_dict()),
                  rng=rng_snapshot())
    try:
        model.restore_new_state(modules)
        model.module_status = copy.deepcopy(status)
        model.rho.initialization = copy.deepcopy(payload['initialization'])
        core.restore_optimizer(payload['trainer'])
        if tensor_digest(model.new_state_dict()) != payload['module_identity']:
            raise ValueError('restored model identity differs')
        restore_rng(payload['rng'])
    except BaseException:
        # Roll back numerical/RNG state, but poison this core on any restoration
        # failure. An old live session/cache must never continue after copy_.
        core.failed = True
        try:
            model.restore_new_state(before['modules'])
            model.module_status = before['status']; model.rho.initialization = before['initialization']
            core.optimizer.load_state_dict(before['optimizer']); core.steps = 0
            restore_rng(before['rng'])
        except BaseException as rollback_error:
            raise RuntimeError('checkpoint restoration and rollback failed; discard process') from rollback_error
        raise
    return dict(next_batch=core.steps, ledger=copy.deepcopy(payload['ledger']),
                binding=asdict(binding), module_identity=payload['module_identity'])
