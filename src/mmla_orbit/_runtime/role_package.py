"""Bind existing task/proposal weights inside the selected frozen runtime.

Called by the reviewed first-release entry, never an automatic launcher.
The old collection stays immutable. No forward, optimization, environment,
J qualification, rule admission or network request happens in this module.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

COLLECTION = 'e4d8d3728712575af41684ffe76bbcb3a9e43fafd869b3b5484ac4a3184fa11f'
KEYS = tuple((objective, seed) for seed in (17, 29, 43) for objective in ('static', 'adapted'))
MODULES = {
    ('static', 17): '77d13795eb504bc40cf9147462718de7cef23a694e5b6538236d58a2e1cfca64',
    ('static', 29): 'cd00d255569a26985391b652132da3b8f6668ecb2fba7e6f71b98843dfa58e6e',
    ('static', 43): 'fdde0e9d9c373078ac5c45fe5cd92aaab41b009ae20d83354afc8909e99a1fff',
    ('adapted', 17): '14919f1fe8205b2adde28dbc27937ec4cafe9090d1a596667ab3da7ecd5a112f',
    ('adapted', 29): 'a45278a9cb268f5ddc9cfc15b7c7a84529f0ef271ad543587cecbd18243b507b',
    ('adapted', 43): '93eb7bb61d1e580b9c75bf254a80fd339153fe0b333e1555682f51d67cff96af',
}


def assert_frozen_runtime(collection):
    """Reject mixed canonical/old/public MMLA class definitions in this process."""
    code = (Path(collection) / 'code').resolve(strict=True)
    seen = 0
    for name, module in tuple(sys.modules.items()):
        if name == 'mmla_pdsa' or name.startswith('mmla_pdsa.'):
            source = getattr(module, '__file__', None)
            if source is None or not Path(source).resolve(strict=True).is_relative_to(code):
                raise ValueError('MMLA definitions must all come from the exact selected collection')
            seen += 1
    if not seen:
        raise ValueError('verified frozen runtime must be imported by the admitted entry')


def _helper(helper_path, expected_sha256):
    """Use the additive reviewed helper without replacing retained runtime files."""
    path = Path(helper_path).resolve(strict=True)
    if len(expected_sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('exact reviewed role-inheritance helper required')
    spec = importlib.util.spec_from_file_location('orbit_verified_role_inheritance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inherit_proposal_role


def _full_identity(model):
    from mmla_pdsa.full.neural import tensor_digest
    return dict(modules=tensor_digest(model.new_state_dict()),
        initialization=tensor_digest({k: model.rho.initialization[k] for k in ('a', 'b')}),
        initialization_seed=model.rho.initialization['seed'],
        profile=model.profile.identity, substrate=model.substrate_identity,
        statuses=dict(model.module_status))


def bind_existing_pair(original, collection, *, objective, seed, helper_path,
                       helper_sha256, verified_collection_identity):
    """The caller first authenticates all collection bytes with the existing loader.

    The full collection verification receipt is an entry prerequisite, not
    something a supplied hash or this function can grant to arbitrary objects.
    """
    assert_frozen_runtime(collection)
    if verified_collection_identity != COLLECTION or (objective, seed) not in MODULES:
        raise ValueError('exact six-reader collection and explicit role selection required')
    from mmla_pdsa.full.outcome_original import OriginalOrbit
    from mmla_pdsa.full.outcome_export import load_reader
    from mmla_pdsa.full.neural import tensor_digest
    if type(original) is not OriginalOrbit:
        raise ValueError('actual verified original roles required')
    task = load_reader(original, collection, objective, seed)
    task_identity = tensor_digest(task.new_state_dict())
    if task_identity != MODULES[objective, seed]:
        raise ValueError('selected trained reader changed')
    inherit = _helper(helper_path, helper_sha256)
    proposal, migration = inherit(task, original.proposal,
        task_modules=task_identity,
        proposal_modules=original.provenance['original_proposal_modules'],
        task_profile=task.profile.identity,
        proposal_profile=original.proposal.profile.identity,
        source_package=COLLECTION)
    assert_frozen_runtime(collection)
    return dict(task=task, proposal=proposal, inheritance=migration,
                objective=objective, seed=seed)


def save_and_restore_pair(pair, destination, *, collection, contract):
    """Persist the real pair and return reloaded objects for subsequent tests.

    Uses the existing weights_only component-bundle loader. These internal
    bundles are not a claim that a public HF safetensors export is finished.
    """
    assert_frozen_runtime(collection)
    if not contract or Path(destination).exists():
        raise ValueError('new destination and explicit first-release contract required')
    from mmla_pdsa.full.extensions.rsi_binding import save_role_pair, load_role_pair, check_role_pair
    task, proposal = pair['task'], pair['proposal']
    check_role_pair(task, proposal)
    before = {name: _full_identity(pair[name]) for name in ('task', 'proposal')}
    manifest = save_role_pair(task, proposal, destination, contract=contract)
    restored = load_role_pair(task, destination, contract=contract)
    actual = restored['models']
    after = {name: _full_identity(actual[name]) for name in ('task', 'proposal')}
    if before != after:
        raise ValueError('actual paired model state changed after restore')
    if {name: _full_identity(pair[name]) for name in ('task', 'proposal')} != before:
        raise ValueError('role persistence mutated its source objects')
    check_role_pair(actual['task'], actual['proposal'])
    if any(p.requires_grad for model in actual.values() for p in model.parameters()):
        raise ValueError('restored slow roles must remain frozen')
    receipt = dict(schema='orbit-inherited-role-pair-restoration-v1',
        collection=COLLECTION, objective=pair['objective'], seed=pair['seed'],
        inheritance=pair['inheritance'], identities=after, bundle=manifest,
        status='ACTUAL_TENSORS_RESTORED_NO_FORWARD', numerical_probe_performed=False,
        proposal_behavior_qualified=False, source_models_unchanged=True,
        slow_training_updates=0, J_qualification_inherited=False, successor_admissions=0)
    path = Path(destination) / 'inheritance-and-restore.json'
    with path.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return dict(task=actual['task'], proposal=actual['proposal'], receipt=receipt,
                objective=pair['objective'], seed=pair['seed'])
