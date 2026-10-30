"""Bind and export trained proposal rho without changing the fixed task reader.

This is a data-only production component, not a scientific entry. Its caller
authenticates the original collection, training source and receipt beforehand.
The same verified bytes are decoded; no optimizer is installed for deployment.
"""
from recipe import FINAL_RHO,INITIAL_RHO,MIDPOINT,PHYSICAL_PRESENTATIONS,PHYSICAL_UPDATES,PRESENTATIONS,UPDATES
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import os
import torch
from safetensors.torch import load
from mmla_pdsa.full.safe_state import save_state, decode, SCHEMA, exact_equal
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.extensions.rsi_binding import isolated_model, rebind, check_role_pair
from training_state import _decode, _write, _json, sha
from curriculum import SEEDS, digest

STATUS='TRAINED_TASK_OUTCOME_MANAGER_PROPOSAL'
FILES=('task.json','task.safetensors','proposal.json','proposal.safetensors')

def identity(model):
    values=model.new_state_dict()
    return dict(modules=tensor_digest(values),profile=model.profile.identity,
        substrate=model.substrate_identity,rho_version=int(model.rho.version),
        initialization=tensor_digest({k:model.rho.initialization[k] for k in ('a','b')}),
        initialization_seed=model.rho.initialization['seed'],statuses=dict(model.module_status))

def _closed(model):
    model.new_state_dict()  # Validate existing guards before any rebind.
    model.validate_state_configuration()
    if (any(m.training for m in model.modules())
            or any(p.requires_grad or p.grad is not None for p in model.parameters())):
        raise ValueError('closed frozen model without pending gradients required')

def _receipt(value):
    if (type(value) is not dict or value.get('schema')!='orbit-task-outcome-proposal-trained-v1'
            or value.get('status')!='COMPLETE' or value.get('initial_rho_version')!=INITIAL_RHO
            or value.get('final_rho_version')!=FINAL_RHO or value.get('effective_updates')!=UPDATES
            or value.get('physical_updates')!=PHYSICAL_UPDATES or value.get('effective_examples')!=PRESENTATIONS
            or value.get('physical_examples')!=PHYSICAL_PRESENTATIONS or value.get('final_probability_forwards')!=2
            or value.get('restored_model_used_for_deployment') is not True
            or value.get('numerical_online_updates')!=0
            or value.get('independent_RSI_benefit_claim') is not False):
        raise ValueError('actual complete bounded training receipt required')
    for k in ('source','data','schedule','original_modules','trained_modules'):sha(value[k])
    for k,step,finished in (('midpoint',MIDPOINT,False),('final',UPDATES,True)):
        r=value[k];sha(r['manifest_sha256'])
        if r['step']!=step or r['next_example']!=step*8 or r['finished'] is not finished:
            raise ValueError('actual checkpoint boundaries required')

def _share_equal(task,proposal):
    """Share only already equal, frozen modules; rho/initialization stay private."""
    _closed(task);_closed(proposal)
    if (type(task) is not type(proposal) or task.profile!=proposal.profile
            or task.substrate_identity!=proposal.substrate_identity
            or set(task._modules)!=set(proposal._modules)):
        raise ValueError('same actual role architecture required')
    before={k:identity(m) for k,m in (('task',task),('proposal',proposal))}
    a=task.new_state_dict();b=proposal.new_state_dict()
    if set(a)!=set(b) or any(not exact_equal(v,b[k]) for k,v in a.items() if not k.startswith('rho.')):
        raise ValueError('frozen module bytes differ; do not overwrite task with proposal')
    if any(task.module_status[k]!=proposal.module_status[k] for k in task.module_status if k!='rho'):
        raise ValueError('shared module status mismatch')
    for k,module in task._modules.items():
        if k!='rho':proposal._modules[k]=module
    rebind(proposal);check_role_pair(task,proposal)
    if before!={k:identity(m) for k,m in (('task',task),('proposal',proposal))}:
        raise ValueError('binding changed numerical identities')

def bind_trained_pair(original_pair,trained,receipt):
    _receipt(receipt)
    if original_pair['objective']!='static' or original_pair['seed'] not in SEEDS:
        raise ValueError('all three predetermined static lineages only')
    task,original=original_pair['task'],original_pair['proposal']
    _closed(task);_closed(original);_closed(trained);check_role_pair(task,original)
    if (int(task.rho.version)!=942 or int(original.rho.version)!=INITIAL_RHO
            or int(trained.rho.version)!=FINAL_RHO or trained.module_status['rho']!=STATUS
            or tensor_digest(original.new_state_dict())!=receipt['original_modules']
            or tensor_digest(trained.new_state_dict())!=receipt['trained_modules']
            or not exact_equal(original.rho.initialization,trained.rho.initialization)):
        raise ValueError('original task/proposal and trained checkpoint identity mismatch')
    if original is trained or any(getattr(original.rho,k).data_ptr()==getattr(trained.rho,k).data_ptr()
            for k in ('a','b','version')):
        raise ValueError('preserve original proposal storage')
    original_before=identity(original)
    _share_equal(task,trained)
    if identity(original)!=original_before:raise ValueError('retained original proposal changed')
    return dict(task=task,proposal=trained,objective='static',seed=original_pair['seed'],
        training=copy.deepcopy(receipt),training_identity=digest(receipt))

def _validate(pair,source,collection):
    sha(source);sha(collection);_receipt(pair['training'])
    if (pair['objective']!='static' or pair['seed'] not in SEEDS
            or pair['training_identity']!=digest(pair['training'])
            or pair['training']['source']!=source):raise ValueError('bound trained pair required')
    _closed(pair['task']);_closed(pair['proposal']);check_role_pair(pair['task'],pair['proposal'])
    if (int(pair['task'].rho.version)!=942 or int(pair['proposal'].rho.version)!=FINAL_RHO
            or pair['proposal'].module_status['rho']!=STATUS
            or tensor_digest(pair['proposal'].new_state_dict())!=pair['training']['trained_modules']):
        raise ValueError('actual frozen trained proposal required')

def save_trained_pair(pair,directory,*,source_identity,collection_identity):
    _validate(pair,source_identity,collection_identity)
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    ids={role:identity(pair[role]) for role in ('task','proposal')}
    for role in ('task','proposal'):
        model=pair[role]
        save_state(root/role,dict(schema='orbit-task-outcome-role-v1',role=role,
            profile=asdict(model.profile),substrate=model.substrate_identity,
            modules=model.new_state_dict(),initialization=copy.deepcopy(model.rho.initialization),
            module_status=dict(model.module_status)))
    files={}
    for leaf in FILES:
        with (root/leaf).open('rb') as f:os.fsync(f.fileno());raw=f.read()
        files[leaf]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    meta=dict(schema='orbit-task-outcome-trained-pair-v1',collection=collection_identity,
        source=source_identity,objective='static',seed=pair['seed'],identities=ids,files=files,
        training=copy.deepcopy(pair['training']),training_identity=pair['training_identity'],
        proposal_status=STATUS,deployed_Phi='fixed',optimizer_restored=False,
        J_qualification_inherited=False,independent_RSI_benefit_claim=False)
    _write(root/'PAIR.json',_json(meta))
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(metadata=meta,metadata_sha256=hashlib.sha256(_json(meta)).hexdigest())

def _verified_bytes(path,size,expected,cap):
    sha(expected)
    if (path.is_symlink() or not path.is_file() or type(size) is not int
            or not 0<size<=cap or path.stat().st_size!=size):raise ValueError('bounded regular payload required')
    raw=path.read_bytes()
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('payload bytes changed')
    return raw

def restore_trained_pair(prototype,directory,*,source_identity,collection_identity,
                         metadata_sha256,training_identity):
    for value in (source_identity,collection_identity,metadata_sha256,training_identity):sha(value)
    _closed(prototype);prototype_before=identity(prototype)
    root=Path(directory)
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()}!=set(FILES)|{'PAIR.json'}:
        raise ValueError('exact trained pair inventory required')
    p=root/'PAIR.json';meta=_decode(_verified_bytes(p,p.stat().st_size,metadata_sha256,65536))
    if (set(meta)!={'schema','collection','source','objective','seed','identities','files','training',
            'training_identity','proposal_status','deployed_Phi','optimizer_restored',
            'J_qualification_inherited','independent_RSI_benefit_claim'}
            or meta['schema']!='orbit-task-outcome-trained-pair-v1' or meta['source']!=source_identity
            or meta['collection']!=collection_identity or meta['objective']!='static' or meta['seed'] not in SEEDS
            or set(meta['files'])!=set(FILES) or set(meta['identities'])!={'task','proposal'}
            or meta['training_identity']!=training_identity or digest(meta['training'])!=training_identity
            or meta['proposal_status']!=STATUS or meta['deployed_Phi']!='fixed'
            or meta['optimizer_restored'] is not False or meta['J_qualification_inherited'] is not False
            or meta['independent_RSI_benefit_claim'] is not False):raise ValueError('exact trained pair metadata required')
    tensor_bytes=sum(v.numel()*v.element_size() for v in prototype.new_state_dict().values())
    tensor_bytes+=sum(prototype.rho.initialization[k].numel()*prototype.rho.initialization[k].element_size()
                      for k in ('a','b'))
    payload_cap=tensor_bytes+4*1024**2
    raw={}
    for leaf,item in meta['files'].items():
        if set(item)!={'bytes','sha256'}:raise ValueError('complete payload identity required')
        raw[leaf]=_verified_bytes(root/leaf,item['bytes'],item['sha256'],payload_cap)
    models={}
    for role in ('task','proposal'):
        tree=_decode(raw[role+'.json'])
        if set(tree)!={'schema','tree'} or tree['schema']!=SCHEMA:raise ValueError('typed safe state required')
        tensors=load(raw[role+'.safetensors']);used=set();payload=decode(tree['tree'],tensors,used)
        if used!=set(tensors):raise ValueError('unreferenced role tensor')
        model=isolated_model(prototype);refs=model.new_state_dict()
        if (set(payload)!={'schema','role','profile','substrate','modules','initialization','module_status'}
                or payload['schema']!='orbit-task-outcome-role-v1' or payload['role']!=role
                or payload['profile']!=asdict(model.profile) or payload['substrate']!=model.substrate_identity
                or set(payload['modules'])!=set(refs)):
            raise ValueError('same original role layout required')
        for name,x in payload['modules'].items():
            ref=refs[name]
            if not torch.is_tensor(x) or x.dtype!=ref.dtype or x.shape!=ref.shape or not torch.isfinite(x).all():
                raise ValueError('finite exact module tensor required')
        init=payload['initialization'];status=payload['module_status']
        if set(init)!={'a','b','seed'} or type(init['seed']) is not int:raise ValueError('exact initialization required')
        for k in ('a','b'):
            x=init[k];ref=getattr(model.rho,k)
            if not torch.is_tensor(x) or x.dtype!=ref.dtype or x.shape!=ref.shape or not torch.isfinite(x).all():
                raise ValueError('finite initialization tensor required')
        if (type(status) is not dict or set(status)!=set(model.module_status)
                or any(type(v) is not str for v in status.values())):raise ValueError('complete role status required')
        model.restore_new_state(payload['modules']);model.rho.initialization=copy.deepcopy(init)
        model.module_status=dict(status);model.eval().requires_grad_(False);rebind(model)
        if identity(model)!=meta['identities'][role]:raise ValueError('decoded role identity differs')
        models[role]=model
    _share_equal(models['task'],models['proposal'])
    result=dict(**models,objective=meta['objective'],seed=meta['seed'],
        training=meta['training'],training_identity=training_identity)
    _validate(result,source_identity,collection_identity)
    if identity(prototype)!=prototype_before:raise ValueError('source prototype changed')
    return result
