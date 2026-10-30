"""Complete inference-role tensors for the task-return-trained Orbit candidate.

Common hash/decode/model-ownership functions are reused unchanged from the
prior public orbit_pair_io.py. New course and role formats are explicit; no
old trained-proposal receipt or private-path fallback is accepted.
"""
from dataclasses import asdict
from pathlib import Path
import copy, hashlib, json, math
import torch
from safetensors.torch import load
from mmla_pdsa.full.safe_state import save_state, decode, SCHEMA, exact_equal
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.extensions.rsi_binding import isolated_model, rebind, check_role_pair


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def sha(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('exact SHA256 identity required')

def _decode(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def finite(text):
        v=float(text)
        if not math.isfinite(v):raise ValueError('nonfinite JSON')
        return v
    def reject(_):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=reject)

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

def _verified_bytes(path,size,expected,cap):
    sha(expected)
    if (path.is_symlink() or not path.is_file() or type(size) is not int
            or not 0<size<=cap or path.stat().st_size!=size):raise ValueError('bounded regular payload required')
    raw=path.read_bytes()
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('payload bytes changed')
    return raw

COLLECTION = 'e4d8d3728712575af41684ffe76bbcb3a9e43fafd869b3b5484ac4a3184fa11f'
STATUS = 'TRAINED_PROCESS_CREDIT_EDIT_PROPOSAL'
ROLES = ('task', 'proposal', 'original_proposal')
FILES = tuple(role+ext for role in ROLES for ext in ('.json', '.safetensors'))
SUMMARY_FIELDS = ('schema', 'status', 'source', 'course', 'seed', 'selection',
    'initial_rho_version', 'final_rho_version', 'effective_updates', 'physical_updates',
    'sampled_candidates', 'candidate_task_endpoints', 'parent_task_endpoints',
    'learned_process_batches', 'frozen_process_batches', 'backward_candidates',
    'original_modules', 'trained_modules', 'final_probability_forwards',
    'exact_final_probability_equal', 'numerical_deployment_updates',
    'task_reader_changed', 'original_initialization_preserved',
    'independent_RSI_benefit_claim', 'publication', 'successful_update_backward_candidates',
    'interrupted_backward_candidates', 'physical_update_calls', 'failed_update_calls',
    'segment_replay_forwards', 'segment_site_replay_forwards', 'physical_backward_calls', 'retained_binding',
    'final_probe_draws', 'final_probe_task_calls', 'segment_physical_backward_calls',
    'cumulative_replay_forwards', 'cumulative_site_replay_forwards', 'retained_work')


RETAINED_WORK_FIELDS = ('edit_replay_forward', 'edit_replay_backward',
    'site_replay_forward', 'site_replay_backward', 'edit_reward_backward')
ORIGINAL_SOURCE = '21c9e94a899ff80f4b743811bc875e5f0f58e166e831692aa94e91cb956738f1'

def checked_training(summary, *, source, seed):
    """Authenticate the actual new course; numeric recovery is separate from RSI."""
    expected = dict(initial_rho_version=156, final_rho_version=184,
        effective_updates=28, physical_updates=29, sampled_candidates=224,
        candidate_task_endpoints=896, parent_task_endpoints=56,
        learned_process_batches=28, frozen_process_batches=28, backward_candidates=116,
        successful_update_backward_candidates=116, interrupted_backward_candidates=0,
        physical_update_calls=29, failed_update_calls=0, cumulative_site_replay_forwards=118,
        numerical_deployment_updates=0, final_probe_draws=0, final_probe_task_calls=0)
    if (set(summary) != set(SUMMARY_FIELDS) | {'midpoint_manifest_sha256', 'final_manifest_sha256'}
            or summary['schema'] != 'orbit-process-credit-proposal-trained-v1'
            or summary['status'] != 'COMPLETE_PENDING_NATIVE_SETTLEMENT'
            or summary['source'] != source or type(summary['seed']) is not int or summary['seed'] != seed
            or any(type(summary[k]) is not int or summary[k] != v for k, v in expected.items())
            or summary['exact_final_probability_equal'] is not True
            or summary['original_initialization_preserved'] is not True
            or summary['task_reader_changed'] is not False
            or summary['independent_RSI_benefit_claim'] is not False or summary['publication'] != 'HOLD'):
        raise ValueError('complete process-credit course required; not native or RSI admission')
    retained=summary['retained_work'];binding=summary['retained_binding']
    if (set(retained)!=set(RETAINED_WORK_FIELDS)
            or any(type(v) is not int or v<0 for v in retained.values())):
        raise ValueError('typed retained work counters required')
    if binding is None:
        if any(retained.values()):raise ValueError('fresh course cannot inherit work')
    else:
        if (type(binding) is not dict or set(binding)!={'source','identity'}
                or binding['source']!=ORIGINAL_SOURCE or seed!=17
                or retained!=dict(edit_replay_forward=210,edit_replay_backward=210,
                    site_replay_forward=88,site_replay_backward=88,edit_reward_backward=88)):
            raise ValueError('exact source-bound retained seed17 course required')
        sha(binding['identity'])
    final=summary['final_probability_forwards'];calls=summary['cumulative_replay_forwards']
    backward=summary['physical_backward_calls'];segment=summary['segment_replay_forwards']
    segment_site=summary['segment_site_replay_forwards'];segment_backward=summary['segment_physical_backward_calls']
    if (type(final) is not int or not 4<=final<=64 or final%2
            or any(type(x) is not int or x<0 for x in (calls,backward,segment,segment_site,segment_backward))
            or not 116<=calls-(final-2)<=116*31 or backward!=calls-(final-2)+116
            or calls!=segment+retained['edit_replay_forward']
            or segment_site+retained['site_replay_forward']!=118
            or segment_backward!=backward-retained['edit_replay_backward']-retained['site_replay_backward']):
        raise ValueError('segment and cumulative gradients/restore prefixes must reconcile separately')
    for field in ('source', 'course', 'selection', 'original_modules', 'trained_modules',
                  'midpoint_manifest_sha256', 'final_manifest_sha256'):
        sha(summary[field])


def export_role_bundle(original_pair, trained_pair, directory, *, source_identity):
    """Save all three role states; caller authenticates the closed run separately.

    This tensor exporter does not infer native completion or scientific success.
    The base/always-on omega and frozen runtime belong to the containing package.
    Original proposal weights are retained to reproduce the frozen comparison.
    """
    sha(source_identity)
    seed = trained_pair['seed']
    if (type(seed) is not int or seed not in (17, 29, 43) or original_pair['seed'] != seed
            or original_pair['objective'] != 'static' or trained_pair['objective'] != 'static'
            or identity(original_pair['task']) != identity(trained_pair['task'])):
        raise ValueError('same original task reader and explicit static lineage required')
    receipt = trained_pair['training']
    if trained_pair['training_identity'] != digest(receipt):
        raise ValueError('original training receipt identity changed')
    summary = {key: receipt[key] for key in SUMMARY_FIELDS if key not in ('retained_binding','retained_work')}
    binding=receipt['retained_binding']
    if binding is not None:
        if (type(binding) is not dict or binding.get('schema')!='orbit-retained-course-input-v1'
                or binding.get('source')!=ORIGINAL_SOURCE
                or digest({k:v for k,v in binding.items() if k!='identity'})!=binding.get('identity')):
            raise ValueError('authentic retained input identity required before removing private paths')
        summary['retained_binding']={key:binding[key] for key in ('source','identity')}
    else:summary['retained_binding']=None
    summary['retained_work']={key:receipt['retained_work'].get(key,0) for key in RETAINED_WORK_FIELDS}
    for key in ('midpoint', 'final'):
        summary[key+'_manifest_sha256'] = receipt[key]['manifest_sha256']
    checked_training(summary, source=source_identity, seed=seed)
    models = dict(task=trained_pair['task'], proposal=trained_pair['proposal'],
        original_proposal=original_pair['proposal'])
    identities = {key: identity(model) for key, model in models.items()}
    check_role_pair(models['task'], models['proposal'])
    check_role_pair(models['task'], models['original_proposal'])
    if ([identities[k]['rho_version'] for k in ROLES] != [942, 184, 156]
            or identities['proposal']['modules'] != summary['trained_modules']
            or identities['original_proposal']['modules'] != summary['original_modules']
            or models['proposal'].module_status['rho'] != STATUS
            or not exact_equal(models['proposal'].rho.initialization, models['original_proposal'].rho.initialization)
            or len({getattr(model.rho, k).data_ptr() for model in models.values()
                    for k in ('a', 'b', 'version')}) != 9):
        raise ValueError('original task942, learned proposal184 and original proposal156 required')
    payloads = {}
    for role, model in models.items():
        _closed(model)
        modules = model.new_state_dict()
        if any(not torch.isfinite(x).all() for x in modules.values()):
            raise ValueError('finite complete role modules required')
        payloads[role] = dict(schema='orbit-process-credit-public-role-v1', role=role,
            profile=asdict(model.profile), substrate=model.substrate_identity, modules=modules,
            initialization=copy.deepcopy(model.rho.initialization), module_status=dict(model.module_status))
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    for role, payload in payloads.items():
        save_state(root/role, payload)
    files = {}
    for name in FILES:
        raw = (root/name).read_bytes()
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    meta = dict(schema='orbit-process-credit-public-roles-v2', collection=COLLECTION, source=source_identity,
        seed=seed, objective='static', identities=identities, files=files,
        original_training_identity=trained_pair['training_identity'], training_summary=summary,
        training_summary_identity=digest(summary), deployment_Phi='fixed',
        restored_optimizer=False, RSI_qualification_inherited=False)
    raw = json.dumps(meta, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n'
    with (root/'ROLES.json').open('xb') as stream:
        stream.write(raw)
    if {key: identity(model) for key, model in models.items()} != identities:
        raise ValueError('source role changed during export')
    return dict(metadata=meta, metadata_sha256=hashlib.sha256(raw).hexdigest())


def restore_role_bundle(prototype, directory, *, source_identity, metadata_sha256,
                        original_training_identity):
    """Load complete role tensors using only the containing base/runtime package.

    The caller pins ROLES.json through the reviewed package manifest. No old
    private collection or original correction checkpoint is opened by this API.
    """
    for value in (source_identity, metadata_sha256, original_training_identity):
        sha(value)
    _closed(prototype)
    before = identity(prototype)
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != set(FILES) | {'ROLES.json'}:
        raise ValueError('exact complete role-bundle inventory required')
    p = root/'ROLES.json'
    meta = _decode(_verified_bytes(p, p.stat().st_size, metadata_sha256, 65536))
    if (set(meta) != {'schema', 'collection', 'source', 'seed', 'objective', 'identities', 'files',
            'original_training_identity', 'training_summary', 'training_summary_identity',
            'deployment_Phi', 'restored_optimizer', 'RSI_qualification_inherited'}
            or meta['schema'] != 'orbit-process-credit-public-roles-v2' or meta['collection'] != COLLECTION
            or meta['source'] != source_identity or meta['objective'] != 'static'
            or type(meta['seed']) is not int or meta['seed'] not in (17, 29, 43)
            or set(meta['identities']) != set(ROLES) or set(meta['files']) != set(FILES)
            or meta['original_training_identity'] != original_training_identity
            or meta['training_summary_identity'] != digest(meta['training_summary'])
            or meta['deployment_Phi'] != 'fixed' or meta['restored_optimizer'] is not False
            or meta['RSI_qualification_inherited'] is not False):
        raise ValueError('source-bound three-role metadata required')
    checked_training(meta['training_summary'], source=source_identity, seed=meta['seed'])
    refs = prototype.new_state_dict()
    cap = sum(v.numel()*v.element_size() for v in refs.values())
    cap += sum(prototype.rho.initialization[k].numel()*prototype.rho.initialization[k].element_size() for k in ('a','b')) + 4*1024**2
    raw = {}
    for name, row in meta['files'].items():
        if set(row) != {'bytes', 'sha256'}:
            raise ValueError('complete role file identity required')
        raw[name] = _verified_bytes(root/name, row['bytes'], row['sha256'], cap)
    models = {}
    for role in ROLES:
        tree = _decode(raw[role+'.json'])
        if set(tree) != {'schema', 'tree'} or tree['schema'] != SCHEMA:
            raise ValueError('typed data-only role state required')
        tensors = load(raw[role+'.safetensors']); used = set()
        payload = decode(tree['tree'], tensors, used)
        model = isolated_model(prototype)
        if (used != set(tensors) or set(payload) != {'schema','role','profile','substrate','modules','initialization','module_status'}
                or payload['schema'] != 'orbit-process-credit-public-role-v1' or payload['role'] != role
                or payload['profile'] != asdict(model.profile) or payload['substrate'] != model.substrate_identity
                or set(payload['modules']) != set(refs)):
            raise ValueError('same actual architecture/substrate and complete role state required')
        for name, value in payload['modules'].items():
            ref = refs[name]
            if not torch.is_tensor(value) or value.shape != ref.shape or value.dtype != ref.dtype or not torch.isfinite(value).all():
                raise ValueError('finite exact module tensor required')
        init = payload['initialization']; statuses = payload['module_status']
        if set(init) != {'a','b','seed'} or type(init['seed']) is not int:
            raise ValueError('original initialization required')
        for key in ('a','b'):
            value = init[key]; ref = getattr(model.rho, key)
            if not torch.is_tensor(value) or value.shape != ref.shape or value.dtype != ref.dtype or not torch.isfinite(value).all():
                raise ValueError('finite original factor initialization required')
        if set(statuses) != set(model.module_status) or any(type(v) is not str for v in statuses.values()):
            raise ValueError('complete role statuses required')
        model.restore_new_state(payload['modules'])
        model.rho.initialization = copy.deepcopy(init)
        model.module_status = dict(statuses)
        model.eval().requires_grad_(False); rebind(model)
        if identity(model) != meta['identities'][role]:
            raise ValueError('restored numerical role identity differs')
        models[role] = model
    summary = meta['training_summary']
    if ([int(models[k].rho.version) for k in ROLES] != [942,184,156]
            or identity(models['proposal'])['modules'] != summary['trained_modules']
            or identity(models['original_proposal'])['modules'] != summary['original_modules']
            or models['proposal'].module_status['rho'] != STATUS
            or not exact_equal(models['proposal'].rho.initialization, models['original_proposal'].rho.initialization)):
        raise ValueError('exact task and learned/original proposal lineage required')
    for key in ('proposal', 'original_proposal'):
        _share_equal(models['task'], models[key])
    if identity(prototype) != before:
        raise ValueError('source prototype changed during restore')
    return dict(**models, seed=meta['seed'], objective='static', metadata=meta)
