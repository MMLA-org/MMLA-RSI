"""Atomic new-component bundles, with exact external substrate references.

No MiniCPM or omega bytes are duplicated. Packages are created once; callers
choose a new directory for each version. Hashes detect damage, not hostile authors.
"""
from dataclasses import asdict, fields
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
from .local_rng import capture_cuda,validate_cuda,restore_cuda

from .neural import FastState, tensor_digest
from .profile import wire


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _trainer_binding(model, trainer):
    if trainer is None: return
    from .module_training import ModuleTrainer
    from .training import SlowTrainer
    if type(trainer) is ModuleTrainer:
        trainer.verify_binding(model)
        return
    if type(trainer) is not SlowTrainer:raise ValueError('unregistered trainer type')
    trainer.verify_finite()
    params=[p for group in trainer.optimizer.param_groups for p in group['params']]
    if trainer.model is not model or len(params)!=2 or params[0] is not model.rho.a or params[1] is not model.rho.b:
        raise ValueError('trainer/optimizer must own exactly this model rho')


def _trainer_state(trainer):
    if trainer is None:return None
    from .module_training import ModuleTrainer
    if type(trainer) is ModuleTrainer:return trainer.export_state()
    if trainer.finite_contract is not None:return trainer.export_finite()
    return dict(objective=trainer.objective,inner_steps=trainer.inner_steps,
        inner=trainer.inner,microbatch=trainer.microbatch,pad_id=trainer.pad_id,
        steps=trainer.steps,optimizer=trainer.optimizer.state_dict())


def _trainer_validate_state(trainer,state):
    from .module_training import ModuleTrainer
    if type(trainer) is ModuleTrainer:
        trainer.validate_state(state);return
    if trainer.finite_contract is not None:
        trainer.validate_finite_state(state);return
    if state is None or state.get('kind') is not None or (state['objective'],state['inner_steps'],state['inner'],state.get('microbatch',1),state.get('pad_id',0))!=(trainer.objective,trainer.inner_steps,trainer.inner,trainer.microbatch,trainer.pad_id):
        raise ValueError('trainer contract mismatch')
    optimizer=copy.deepcopy(trainer.optimizer);optimizer.load_state_dict(state['optimizer'])


def _trainer_load_state(trainer,state):
    from .module_training import ModuleTrainer
    if type(trainer) is ModuleTrainer:trainer.import_state(state)
    else:
        trainer.optimizer.load_state_dict(state['optimizer']);trainer.steps=state['steps']
        if trainer.finite_contract is not None:trainer.nonzero_steps=state['nonzero_steps']


def _fast_validate(data, model, modules):
    if set(data)!={f.name for f in fields(FastState)}:
        raise ValueError('exact fast-state schema required')
    recovery=data["recovery_id"]
    if type(recovery) is not str or (recovery and (len(recovery)!=32 or any(c not in "0123456789abcdef" for c in recovery))):
        raise ValueError("invalid fast recovery namespace")
    for k in ('epoch','steps','rho_version'):
        if type(data[k]) is not int or not 0<=data[k]<2**63-1:
            raise ValueError('bounded integral fast counter required')
    if data['rho_version']!=int(modules['rho.version']) or data['steps']>model.profile.max_fast_steps:
        raise ValueError('fast version/budget mismatch')
    if type(data['rho_identity']) is not str or not data['rho_identity']:
        raise ValueError('fast identity missing')
    for k,ref in (('a','rho.a'),('b','rho.b'),('momentum_a','rho.a'),('momentum_b','rho.b')):
        v=data[k]
        if v.shape!=modules[ref].shape or v.dtype!=modules[ref].dtype or not torch.isfinite(v).all():
            raise ValueError('invalid fast tensor')
    for k in ('rng_state','initial_rng'): torch.Generator(device='cpu').set_state(data[k].cpu())


def save_bundle(model, destination, *, trainer=None, memory=None, fast=None):
    _trainer_binding(model,trainer)
    if memory is not None and memory.profile!=model.profile: raise ValueError('memory/model profile mismatch')
    if fast is not None: _fast_validate(fast.__dict__,model,model.new_state_dict())
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError('versioned destination already exists')
    if fast is not None:
        model.rho.factors(fast)
    payload=dict(schema=1, profile=asdict(model.profile), substrate=model.substrate_identity,
        modules=model.new_state_dict(), module_status=dict(model.module_status),
        initialization={k:v.cpu() if torch.is_tensor(v) else v for k,v in model.rho.initialization.items()},
        memory=None if memory is None else memory.snapshot(),
        fast=None if fast is None else {k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in fast.__dict__.items()},
        torch_rng=torch.get_rng_state(),
        cuda_rng=capture_cuda(),
        trainer=_trainer_state(trainer))
    tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        state=tmp/'state.pt'
        torch.save(payload,state)
        with state.open('rb') as f: os.fsync(f.fileno())
        manifest=dict(schema=1,profile=model.profile.identity,substrate=model.substrate_identity,
            state={'name':'state.pt','bytes':state.stat().st_size,'sha256':_sha(state)},
            modules=tensor_digest(payload['modules']),module_status=payload['module_status'],
            qualification='ENGINEERING_COMPONENT_BUNDLE_NOT_SCIENTIFIC_PASS')
        (tmp/'manifest.json').write_bytes(wire(manifest)+b'\n')
        with (tmp/'manifest.json').open('rb') as f: os.fsync(f.fileno())
        fd=os.open(tmp,os.O_RDONLY);os.fsync(fd);os.close(fd)
        # No-overwrite under the caller's serialized single-writer contract.
        if dest.exists(): raise FileExistsError(dest)
        os.rename(tmp,dest)
        fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        return manifest
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True)
        raise


def load_bundle(model, directory, *, memory=None, trainer=None, max_bytes=512*1024*1024):
    _trainer_binding(model,trainer)
    if memory is not None and memory.profile!=model.profile: raise ValueError('memory/model profile mismatch')
    path=Path(directory)
    manifest=json.loads((path/'manifest.json').read_bytes())
    if manifest['schema']!=1 or manifest['profile']!=model.profile.identity or manifest['substrate']!=model.substrate_identity:
        raise ValueError('bundle profile/substrate identity mismatch')
    row=manifest['state']
    if row['name']!='state.pt' or not 0 < row['bytes'] <= max_bytes:
        raise ValueError('bounded local state filename required')
    state=path/'state.pt'
    if state.stat().st_size!=row['bytes'] or _sha(state)!=row['sha256']:
        raise ValueError('bundle bytes changed')
    payload=torch.load(state,map_location='cpu',weights_only=True)
    if (payload['schema']!=1 or payload['profile']!=asdict(model.profile)
            or payload['substrate']!=model.substrate_identity
            or tensor_digest(payload['modules'])!=manifest['modules']):
        raise ValueError('bundle content identity mismatch')
    expected=model.new_state_dict()
    if set(payload['modules'])!=set(expected): raise ValueError('incomplete module payload')
    for k,v in payload['modules'].items():
        if v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all():
            raise ValueError('invalid module tensor')
    saved_fast=payload['fast']
    staged_fast=None
    if saved_fast is not None:
        _fast_validate(saved_fast,model,payload['modules'])
        device=model.rho.a.device
        data={k:v.to(device) if torch.is_tensor(v) and k not in ('rng_state','initial_rng') else v for k,v in saved_fast.items()}
        data['epoch']+=1
        data['a']=data['a'].requires_grad_();data['b']=data['b'].requires_grad_()
        staged_fast=FastState(**data)
    initialization=payload['initialization']
    if set(initialization)!={'a','b','seed'} or type(initialization['seed']) is not int:
        raise ValueError('invalid initialization record')
    for k in ('a','b'):
        v=initialization[k]; ref=expected['rho.'+k]
        if v.shape!=ref.shape or v.dtype!=ref.dtype or not torch.isfinite(v).all():
            raise ValueError('invalid initialization tensor')
    if set(payload['module_status'])!=set(model.module_status) or any(type(x) is not str for x in payload['module_status'].values()):
        raise ValueError('invalid module status record')
    torch.Generator(device='cpu').set_state(payload['torch_rng'])
    if payload['memory'] is not None:
        if memory is None: raise ValueError('explicit compatible memory required')
        # Validate on an isolated branch; no published state changes on failure.
        shadow=memory.fork();shadow.restore(payload['memory'],rollback=True)
    if trainer is not None:
        _trainer_validate_state(trainer,payload['trainer'])
    backup=dict(modules=expected,incarnation=model.rho._incarnation,
                status=model.module_status,initialization=model.rho.initialization)
    memory_before=None if memory is None else (memory._rows,memory.epoch,memory.version,dict(memory._pending),memory.recovery_id)
    trainer_before=None if trainer is None else copy.deepcopy(_trainer_state(trainer))
    try:
        model.restore_new_state(payload['modules'])
        model.module_status=dict(payload['module_status'])
        model.rho.initialization=initialization
        if payload['memory'] is not None:
            # shadow has already validated every row and selected a new epoch.
            memory._rows,memory.epoch,memory.version,memory.recovery_id=shadow._rows,shadow.epoch,shadow.version,shadow.recovery_id
            memory._pending.clear()
        if trainer is not None:
            _trainer_load_state(trainer,payload['trainer'])
        if staged_fast is not None: staged_fast.rho_identity=model.rho.identity
    except Exception:
        model.restore_new_state(backup['modules'])
        model.rho._incarnation=backup['incarnation']
        model.module_status=backup['status'];model.rho.initialization=backup['initialization']
        if memory_before is not None:
            memory._rows,memory.epoch,memory.version,memory._pending,memory.recovery_id=memory_before
        if trainer_before is not None:
            _trainer_load_state(trainer,trainer_before)
        raise
    fast=staged_fast
    # RNG states are returned for explicit restore by the owning runtime. Loading
    # a model library must not silently replace the host's global random stream.
    return dict(fast=fast,torch_rng=payload['torch_rng'],cuda_rng=payload['cuda_rng'],manifest=manifest)
