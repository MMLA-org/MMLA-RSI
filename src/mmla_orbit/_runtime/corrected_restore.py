"""Consume completed correction checkpoints without resuming their training.

The admitted entry authenticates the original collection and this module's
source closure before use. The reviewed metadata manifest binds all six final
artifacts to the closed correction run. Stored optimizer/RNG records remain
provenance; this consumer neither constructs an optimizer nor installs them.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import hashlib
import json
import math

import torch
from mmla_pdsa.full.artifacts import load_artifact
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.profile import digest
from mmla_pdsa.full.extensions.successor_model import SuccessorMMLAModel
from mmla_pdsa.full.extensions.rsi_binding import isolated_model, rebind, check_role_pair
from demonstration_checkpoint import KIND, CheckpointBinding, _ledger, _initialization
from role_inheritance import inherit_proposal_role


COLLECTION = 'e4d8d3728712575af41684ffe76bbcb3a9e43fafd869b3b5484ac4a3184fa11f'
KEYS = {f'{o}-{s}' for o in ('static', 'adapted') for s in (17, 29, 43)}
FIELDS = {'schema', 'binding', 'next_batch', 'profile', 'substrate', 'modules',
          'module_identity', 'module_status', 'initialization', 'trainer', 'rng',
          'ledger', 'active_episodes', 'boundary'}


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('exact lowercase SHA256 identity required')


@dataclass(frozen=True)
class ReaderSpec:
    key: str
    checkpoint: str
    checkpoint_binding: str
    parent_modules: str
    final_modules: str
    profile: str
    rho_version: int
    steps: int

    def __post_init__(self):
        if self.key not in KEYS:
            raise ValueError('one of the six retained reader lineages required')
        for field in ('checkpoint', 'checkpoint_binding', 'parent_modules', 'final_modules', 'profile'):
            _sha(getattr(self, field))
        if (type(self.rho_version) is not int or not 0 <= self.rho_version < 2**63 - 1
                or type(self.steps) is not int or not 0 <= self.steps <= 4096):
            raise ValueError('finite completed reader cursor required')


def _closed_history(closure, saved=None):
    expected = {'schema', 'contract', 'source', 'result_sha256', 'terminal_sha256',
                'ledger', 'final_CPU_seconds', 'final_GPU_seconds', 'accounting_complete'}
    if (type(closure) is not dict or set(closure) != expected
            or closure['schema'] != 'orbit-closed-correction-history-v1'
            or closure['accounting_complete'] is not True):
        raise ValueError('authenticated closed correction history required')
    for name in ('contract', 'source', 'result_sha256', 'terminal_sha256'):
        _sha(closure[name])
    last = closure['ledger']; _ledger(last)
    for name, old in (('final_CPU_seconds', 'CPU_seconds'), ('final_GPU_seconds', 'GPU_seconds')):
        value = closure[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < last[old]:
            raise ValueError('closed accounting must retain all historical cost')
    if saved is not None:
        _ledger(saved)
        if (any(saved[k] != last[k] for k in ('run_id', 'started_unix', 'deadline_unix'))
                or any(saved[k] > last[k] for k in ('wall_seconds', 'CPU_seconds', 'GPU_seconds', 'events_count'))
                or (saved['events_count'] == last['events_count'] and saved['events_sha256'] != last['events_sha256'])):
            raise ValueError('checkpoint does not belong to the bound closed training history')


def read_manifest(path, *, expected_sha256):
    """Read metadata only. The scientific entry pins expected_sha256 in its contract."""
    _sha(expected_sha256); path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 256 * 1024:
        raise ValueError('bounded regular reader manifest required')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('reader manifest identity changed')
    value = json.loads(data)
    if (set(value) != {'schema', 'collection', 'closure', 'readers', 'provenance'}
            or value['schema'] != 'orbit-six-corrected-readers-v1'
            or value['collection'] != COLLECTION or set(value['readers']) != KEYS):
        raise ValueError('complete six-reader manifest required')
    _closed_history(value['closure'])
    for key, row in value['readers'].items():
        if ReaderSpec(**row).key != key:
            raise ValueError('reader key and lineage differ')
    return value


def _frozen(model):
    if (type(model) is not SuccessorMMLAModel or any(m.training for m in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise ValueError('closed frozen Successor reader with no pending gradients required')
    model.validate_state_configuration()


def restore_task(prototype, directory, *, spec, closure):
    """Return a private corrected reader; source reader and process RNG stay intact.

    spec/closure must come from read_manifest on the contract-pinned manifest.
    The lower-level interface is also used with authenticated synthetic fixtures.
    It is not an admission API for caller-invented training histories.
    """
    if type(spec) is not ReaderSpec:
        raise TypeError('explicit corrected-reader binding required')
    _closed_history(closure); _frozen(prototype)
    before = tensor_digest(prototype.new_state_dict())
    if before != spec.parent_modules or prototype.profile.identity != spec.profile:
        raise ValueError('original paired parent and actual profile required')
    payload = load_artifact(directory, kind=KIND, identity=spec.checkpoint)
    if (type(payload) is not dict or set(payload) != FIELDS or payload['schema'] != KIND
            or payload['profile'] != prototype.profile.identity
            or payload['substrate'] != prototype.substrate_identity
            or type(payload['active_episodes']) is not int or payload['active_episodes'] != 0
            or type(payload['next_batch']) is not int or payload['next_batch'] != spec.steps
            or type(payload['trainer']) is not dict or payload['trainer'].get('steps') != spec.steps
            or payload['boundary'] != 'all verified demonstration states closed; no pending Phi/M or gradients'):
        raise ValueError('exact closed correction payload required')
    binding = CheckpointBinding(**payload['binding'])
    if (digest(asdict(binding)) != spec.checkpoint_binding or binding.seed != int(spec.key.rsplit('-', 1)[1])
            or binding.objective != 'adapted' or binding.original_model != spec.parent_modules
            or binding.contract != closure['contract'] or binding.source != closure['source']):
        raise ValueError('correction phase and source binding differ')
    modules = payload['modules']; current = prototype.new_state_dict()
    if (type(modules) is not dict or set(modules) != set(current)
            or payload['module_identity'] != spec.final_modules or tensor_digest(modules) != spec.final_modules):
        raise ValueError('complete corrected module identity required')
    for name, target in current.items():
        x = modules[name]
        if not torch.is_tensor(x) or x.shape != target.shape or x.dtype != target.dtype or not bool(torch.isfinite(x).all()):
            raise ValueError('corrected tensor layout/value mismatch')
    if int(modules['rho.version']) != spec.rho_version:
        raise ValueError('actual trained rho version differs')
    status = payload['module_status']
    if type(status) is not dict or set(status) != set(prototype.module_status) or any(type(x) is not str for x in status.values()):
        raise ValueError('complete corrected status required')
    _initialization(payload['initialization'], modules)
    initial = payload['initialization']; prior = prototype.rho.initialization
    if (initial['seed'] != prior['seed']
            or any(not torch.equal(initial[k].cpu(), prior[k].cpu()) for k in ('a', 'b'))):
        raise ValueError('original factor initialization changed')
    _closed_history(closure, payload['ledger'])
    # Copy only new modules; no source mutation and no training optimizer/RNG.
    result = isolated_model(prototype)
    result.restore_new_state(modules)
    result.rho.initialization = copy.deepcopy(initial)
    result.module_status = copy.deepcopy(status)
    result.eval().requires_grad_(False); rebind(result); _frozen(result)
    if tensor_digest(result.new_state_dict()) != spec.final_modules:
        raise ValueError('restored corrected reader differs')
    if tensor_digest(prototype.new_state_dict()) != before:
        raise RuntimeError('source reader changed during restoration')
    source = dict(prototype.named_parameters())
    for name, tensor in result.named_parameters():
        if not name.startswith('mixer.') and tensor.data_ptr() == source[name].data_ptr():
            raise RuntimeError('restored modules alias the source reader')
    receipt = dict(schema='orbit-corrected-inference-restore-v1', reader=asdict(spec),
        historical_checkpoint_ledger=copy.deepcopy(payload['ledger']), closed_history=copy.deepcopy(closure),
        actual_modules=spec.final_modules, actual_rho_version=int(result.rho.version),
        original_initialization=tensor_digest({k: initial[k] for k in ('a', 'b')}),
        optimizer_restored=False, training_RNG_restored=False, training_resumed=False,
        new_fast_states='caller creates each episode from this trained rho',
        new_run_ledger='separate; historical cost retained, not a new available balance',
        behavior_qualified=False, J_qualification_inherited=False)
    return result, receipt


def restore_pair(original, collection, checkpoint_root, *, key, manifest_path, manifest_sha256,
                 verified_collection_identity):
    """Use after the entry's full original collection verification; no model forward."""
    from role_package import assert_frozen_runtime
    from mmla_pdsa.full.outcome_original import OriginalOrbit
    from mmla_pdsa.full.outcome_export import load_reader
    assert_frozen_runtime(collection)
    if verified_collection_identity != COLLECTION or type(original) is not OriginalOrbit:
        raise ValueError('verified exact original collection and roles required')
    metadata = read_manifest(manifest_path, expected_sha256=manifest_sha256)
    if key not in metadata['readers']:
        raise ValueError('registered corrected reader required')
    spec = ReaderSpec(**metadata['readers'][key]); lineage, seed = key.rsplit('-', 1)
    parent = load_reader(original, collection, lineage, int(seed))
    path = Path(checkpoint_root) / key / 'adapted' / 'step-0247'
    task, restoration = restore_task(parent, path, spec=spec, closure=metadata['closure'])
    proposal, inheritance = inherit_proposal_role(task, original.proposal,
        task_modules=spec.final_modules, proposal_modules=original.provenance['original_proposal_modules'],
        task_profile=task.profile.identity, proposal_profile=original.proposal.profile.identity,
        source_package=COLLECTION)
    check_role_pair(task, proposal); assert_frozen_runtime(collection)
    return dict(task=task, proposal=proposal, restoration=restoration, inheritance=inheritance,
                objective=lineage, seed=int(seed), manifest_sha256=manifest_sha256)
