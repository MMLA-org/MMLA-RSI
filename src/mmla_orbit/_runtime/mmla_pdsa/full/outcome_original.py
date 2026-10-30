"""Data-only bridge from the exact Orbit release to the current outcome runtime.

Old release code is never imported or executed. Its 109 retained runtime source
files must match the current definitions byte for byte. All inherited tensors
are restored and verified before the explicit context-capacity migration.
No model forward, optimizer update or historical J qualification happens here.
"""
from dataclasses import asdict,dataclass,replace
from pathlib import Path
import copy,hashlib
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from .safe_state import read_json,load_state
from .substrate_helpers import lora_linear,tensors_hash
from .profile import ArchitectureProfile,digest
from .neural import tensor_digest
from .extensions.hybrid import HybridConfig
from .extensions.successor_model import SuccessorMMLAModel
from .extensions.rsi_binding import isolated_model,rebind

PUBLIC_MANIFEST='3350e3482d56701186e2dd17e89ab6d46de739aec0ee057d3f729fc9dc3ae6e1'
INTERNAL_MANIFEST='46fc30ae9541d8545f35897953feb612e36242cc106dfb2cd63c759eb5fc9e30'
SFT_REVISION='a60b37f1fc409c54e1e337b0723aaac6f92dfec0'


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def verify_release(root,expected_manifest):
    root=Path(root).resolve(strict=True)
    if sha_file(root/'MANIFEST.json')!=expected_manifest:raise ValueError('original release MANIFEST bytes changed')
    m=read_json(root/'MANIFEST.json');seen=set()
    if m['schema']!='mmla-successor-release-v1':raise ValueError('original release schema')
    for row in m['files']:
        name=row['path'];rel=Path(name)
        if type(name) is not str or rel.is_absolute() or '..' in rel.parts or name in seen:raise ValueError('release inventory path')
        p=(root/rel).resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:
            raise ValueError('original file identity: '+name)
        seen.add(name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
    if actual!=seen:raise ValueError('unbound original release file')
    return m


def retained_source_parity(manifest):
    current=Path(__file__).resolve().parents[2];rows=[]
    for row in manifest['files']:
        rel=Path(row['path'])
        if rel.parts[0]!='mmla_pdsa' or rel.suffix!='.py':continue
        p=current/rel
        if not p.is_file() or sha_file(p)!=row['sha256']:raise ValueError('retained runtime source changed: '+str(rel))
        rows.append(dict(path=str(rel),sha256=row['sha256']))
    if len(rows)!=109:raise ValueError('exact 109-file retained Orbit runtime required')
    return rows


def restore_original_role(model,payload,child,role):
    if (payload['schema']!=1 or payload['profile']!=asdict(model.profile)
            or payload['substrate']!=model.substrate_identity or payload['memory'] is not None or payload['fast'] is not None):
        raise ValueError('original closed role/profile identity')
    expected=model.new_state_dict();values=payload['modules']
    if (set(values)!=set(expected) or tensor_digest(values)!=child['modules']
            or payload['module_status']!=child['module_status']):raise ValueError('original role module identity')
    for key,value in values.items():
        ref=expected[key]
        if value.shape!=ref.shape or value.dtype!=ref.dtype or not torch.isfinite(value).all():raise ValueError('original tensor layout: '+key)
    initial=payload['initialization']
    if set(initial)!={'a','b','seed'} or type(initial['seed']) is not int:raise ValueError('historical initialization record')
    for key in ('a','b'):
        value=initial[key];ref=expected['rho.'+key]
        if value.shape!=ref.shape or value.dtype!=ref.dtype or not torch.isfinite(value).all():raise ValueError('historical factors')
    if set(payload['module_status'])!=set(model.module_status):raise ValueError('complete original module statuses')
    torch.Generator(device='cpu').set_state(payload['torch_rng'])
    model.restore_new_state(values);model.rho.initialization=copy.deepcopy(initial)
    model.module_status=dict(payload['module_status']);model.eval().requires_grad_(False);rebind(model)
    if tensor_digest(model.rho.state_dict())!=role['rho'] or initial['seed']!=role['initialization_seed']:
        raise ValueError('historical rho/version/initialization identity')
    if tensor_digest(model.new_state_dict())!=child['modules']:raise ValueError('actual inherited tensor parity failed')


@dataclass
class OriginalOrbit:
    task:object
    proposal:object
    teacher:object
    tokenizer:object
    config:dict
    provenance:dict


def load_original(public_root,internal_root,*,device):
    """Called only inside the admitted production entry, including qualification."""
    target=torch.device(device)
    if target.type not in ('cpu','cuda'):raise ValueError('registered device required')
    devices=list(range(torch.cuda.device_count())) if target.type=='cuda' or torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        public=verify_release(public_root,PUBLIC_MANIFEST)
        internal=verify_release(internal_root,INTERNAL_MANIFEST)
        parity=retained_source_parity(public)
        root=Path(internal_root).resolve(strict=True);config=read_json(root/'mmla_config.json')
        if (config['schema']!='mmla-successor-config-v1' or config['profile']['base_revision']!=SFT_REVISION
                or config['substrate']['base_revision']!=SFT_REVISION):raise ValueError('exact SFT lineage required')
        # Publication metadata differs; computational payloads must match.
        pub={x['path']:x['sha256'] for x in public['files']};original={x['path']:x['sha256'] for x in internal['files']}
        for name in ('task.json','task.safetensors','proposal.json','proposal.safetensors','omega.safetensors','roles.json'):
            if pub[name]!=original[name]:raise ValueError('public/internal computational payload mismatch: '+name)
        for name in pub:
            if name.startswith('base/') and pub[name]!=original.get(name):raise ValueError('original base files differ')
        tokenizer=AutoTokenizer.from_pretrained(root/'base',local_files_only=True,trust_remote_code=False)
        def base():
            m=AutoModelForCausalLM.from_pretrained(root/'base',local_files_only=True,trust_remote_code=False,
                dtype=torch.bfloat16,attn_implementation='eager').to(target).eval().requires_grad_(False)
            if tensors_hash(torch,dict(m.named_parameters()))!=config['substrate']['theta0']:raise ValueError('exact theta0 tensor identity')
            m.config._name_or_path=config['original_mixer_name_or_path']
            return m
        teacher=base();mixer=base()
        omega=load_file(str(root/'omega.safetensors'),device='cpu')
        if tensors_hash(torch,omega)!=config['substrate']['omega']:raise ValueError('omega original bytes')
        recipe=config['omega_recipe'];rng=torch.Generator(device=target).manual_seed(recipe['omega_seed']);actual={}
        for index in recipe['layers']:
            attention=mixer.model.layers[index].self_attn
            for name in recipe['targets']:
                layer=lora_linear(torch,getattr(attention,name),recipe['rank'],recipe['alpha'],rng)
                setattr(attention,name,layer)
                for leaf in ('down','up'):
                    key=f'layer.{index}.{name}.{leaf}';value=omega[key];dest=getattr(layer,leaf)
                    if value.shape!=dest.shape or value.dtype!=dest.dtype or not torch.isfinite(value).all():raise ValueError('omega layout')
                    with torch.no_grad():dest.copy_(value.to(dest.device))
                    actual[key]=dest
        mixer.requires_grad_(False)
        if set(actual)!=set(omega) or tensors_hash(torch,actual)!=config['substrate']['omega']:raise ValueError('omega active attachment')
        profile=ArchitectureProfile(**config['profile'])
        if profile.identity!=config['profile_identity']:raise ValueError('original profile identity')
        roles=read_json(root/'roles.json');models={}
        for name in ('task','proposal'):
            model=SuccessorMMLAModel(mixer,profile,seed=config['construction_seed'],
                substrate_identity={k:v for k,v in config['substrate'].items() if k!='mmla_hybrid'},
                hybrid_config=HybridConfig(**config['hybrid'])).eval().requires_grad_(False)
            restore_original_role(model,load_state(root/name),config['children'][name],roles['roles'][name])
            if tensor_digest({k:v for k,v in model.new_state_dict().items() if not k.startswith('rho.')})!=roles['shared_modules']:
                raise ValueError('original common module identity')
            models[name]=model
        return OriginalOrbit(models['task'],models['proposal'],teacher,tokenizer,config,
            dict(public_manifest=PUBLIC_MANIFEST,internal_manifest=INTERNAL_MANIFEST,retained_sources=parity,
                original_task_modules=config['children']['task']['modules'],original_proposal_modules=config['children']['proposal']['modules'],
                source_bridge='verified-data-only/current-identical-retained-definitions',
                old_J_and_proposal_runtime='preserved as original files; NOT outcome-qualified'))


def clone_outcome_model(original):
    """Preserve inherited tensors; bind explicit context and BF16 arithmetic migrations."""
    if type(original) is not SuccessorMMLAModel or any(p.requires_grad for p in original.parameters()):
        raise ValueError('frozen exact original task required')
    p=original.profile
    if (p.local_tokens,p.memory_slots,p.memory_tokens,p.max_fast_steps,p.max_new_tokens)!=(2048,8,128,8,64):
        raise ValueError('original Orbit execution capacity required')
    before=tensor_digest(original.new_state_dict());model=isolated_model(original)
    model.profile=replace(p,local_tokens=4096);rebind(model)
    if tensor_digest(model.new_state_dict())!=before:raise ValueError('profile migration changed inherited numerical state')
    if any(a.data_ptr()==b.data_ptr() for (n,a),(m,b) in zip(original.named_parameters(),model.named_parameters()) if not n.startswith('mixer.')):
        raise ValueError('new outcome modules alias pristine original')
    from .outcome_dense_arithmetic import bind
    arithmetic=bind(model)
    return model,dict(original_profile=p.identity,outcome_profile=model.profile.identity,
        tensor_identity=before,changed_fields={'local_tokens':[2048,4096]},evaluation_memory={'slots':2,'read_tokens':512},
        execution_arithmetic=arithmetic)
