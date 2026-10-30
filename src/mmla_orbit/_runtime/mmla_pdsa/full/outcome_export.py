"""Export all six closed readers and selected J, with actual reload verification.

This creates a new local collection, never publishes or overwrites a release.
The original base/omega and proposal role remain byte-for-byte originals. Slow
checkpoints include optimizer/RNG for provenance; this loader does not resume.
"""
from dataclasses import asdict
from pathlib import Path
import copy,json,os,shutil
import torch
from .safe_state import save_state,load_state,exact_equal
from .outcome_original import (load_original,clone_outcome_model,sha_file,
    verify_release,PUBLIC_MANIFEST,INTERNAL_MANIFEST)
from .outcome_restore import course_receipts
from .outcome_train_stage import TRAINING_ORDER,CONTRACT,_write_new
from .extensions.rsi_binding import rebind
from .neural import tensor_digest
from .artifacts import load_artifact
from .outcome_j_artifact import KIND as J_KIND,load_j_package
from .outcome_j import outcome_protocol
from .outcome_train_stage import SessionFactory
from .textworld_backend import TextWorldRuntime
from .profile import digest
from .outcome_progress import advance


def _copy(src,dst):
    src,dst=Path(src),Path(dst)
    if src.is_symlink() or not src.is_file() or dst.exists():raise ValueError('new regular-file copy required')
    dst.parent.mkdir(parents=True,exist_ok=True)
    with src.open('rb') as a,dst.open('xb') as b:
        while True:
            chunk=a.read(4*1024**2)
            if not chunk:break
            if b.write(chunk)!=len(chunk):raise OSError('short artifact copy')
            advance('export_copy_bytes_written')
        b.flush();os.fsync(b.fileno())
    if sha_file(src)!=sha_file(dst):raise ValueError('export copy byte identity')
    advance('export_file_verified')


def _safe(stem,payload):
    save_state(stem,payload)
    for suffix in ('.json','.safetensors'):
        with Path(stem).with_suffix(suffix).open('rb') as f:os.fsync(f.fileno())
    restored=load_state(stem)
    if not exact_equal(restored,payload):raise ValueError('lossless safe-state reload failed')
    advance('export_safe_state_reloaded')


def verify_collection(root,identity):
    root=Path(root).resolve(strict=True);manifest=json.loads((root/'MANIFEST.json').read_bytes())
    if manifest.get('schema')!='orbit-outcome-collection-v1' or digest(manifest)!=identity:
        raise ValueError('exact complete collection identity')
    names=set()
    for row in manifest['files']:
        rel=Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in names:raise ValueError('unique confined collection path')
        p=root/rel
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(root) or not p.is_file():raise ValueError('actual collection file')
        if p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:raise ValueError('collection bytes changed')
        names.add(str(rel));advance('export_manifest_member_verified')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    # Nested original MANIFESTs are also bound members.
    actual.update(str(p.relative_to(root)) for p in root.rglob('MANIFEST.json') if p!=root/'MANIFEST.json')
    if actual!=names:raise ValueError('unbound collection file')
    return manifest


def load_reader(original,root,objective,seed):
    """Call after verify_collection; only inference state, never an optimizer step."""
    if (objective,seed) not in TRAINING_ORDER:raise ValueError('one of all six published readers')
    payload=load_state(Path(root)/'readers'/f'{objective}-{seed}'/'closed-checkpoint')
    if (payload['next_batch']!=64 or payload['binding']['contract']!=CONTRACT
            or payload['binding']['objective']!=objective or payload['binding']['seed']!=seed):
        raise ValueError('exact closed final reader required')
    model,_=clone_outcome_model(original.task)
    from .outcome_dense_arithmetic import identity
    configuration=json.loads((Path(root)/'configuration.json').read_bytes())
    from .outcome_generation import GENERATION_ID
    if configuration.get('generation')!=GENERATION_ID:raise ValueError('exported generation identity changed')
    if configuration.get('execution_arithmetic')!=identity(model):raise ValueError('exported arithmetic identity changed')
    state=payload['modules'];reference=model.new_state_dict()
    if set(state)!=set(reference) or any(v.shape!=reference[k].shape or v.dtype!=reference[k].dtype
            or not torch.isfinite(v).all() for k,v in state.items()):raise ValueError('complete finite original-layout reader')
    model.restore_new_state(state);model.rho.initialization=copy.deepcopy(payload['initialization'])
    model.module_status=dict(payload['module_status']);model.eval().requires_grad_(False);rebind(model)
    if tensor_digest(model.new_state_dict())!=payload['module_identity']:raise ValueError('restored reader identity')
    advance('export_reader_restored')
    return model


def export_collection(*,original,readers,training_root,j_packages,public_root,internal_root,
        code_root,source_files,output,probe_ids,results):
    """Full immutable originals plus six safe states and actual fixed-probe parity.

    probe_ids is the already saved actual preflight prompt, not a new task.
    Results are existing finite run receipts, not rescored/tuned populations.
    """
    if set(readers)!=set(TRAINING_ORDER) or len(j_packages)!=3:raise ValueError('all six readers and three J artifacts')
    if len({o for o,s in j_packages})!=1 or {s for o,s in j_packages}!={17,29,43}:raise ValueError('one DEV-selected objective, every seed')
    if not torch.is_tensor(probe_ids) or probe_ids.ndim!=2 or probe_ids.shape[0]!=1 or not 1<=probe_ids.shape[1]<=4032:
        raise ValueError('actual bounded preflight prompt')
    root=Path(output);root.mkdir(exist_ok=False);rows=course_receipts(training_root)
    for name,source,identity in (('original-public',public_root,PUBLIC_MANIFEST),('original-internal',internal_root,INTERNAL_MANIFEST)):
        manifest=verify_release(source,identity)
        for row in manifest['files']:_copy(Path(source)/row['path'],root/name/row['path'])
        _copy(Path(source)/'MANIFEST.json',root/name/'MANIFEST.json')
    # Only the reviewed import closure; no home config, credentials or traces.
    if not source_files or 'mmla_pdsa/full/outcome_export.py' not in source_files:raise ValueError('bound current loader sources')
    for name,identity in source_files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or rel.suffix!='.py' or rel.parts[0]!='mmla_pdsa':raise ValueError('package sources only')
        path=Path(code_root)/rel
        if sha_file(path)!=identity:raise ValueError('reviewed runtime source changed')
        _copy(path,root/'code'/rel)
    training_source=source_files
    if 'lineage' in results:
        provenance=results['lineage']
        if (provenance['schema']!='orbit-evaluation-continuation-lineage-v1'
                or provenance['evaluation_source_files']!=source_files
                or provenance['evaluation_source_identity']!=digest(source_files)
                or provenance['training_source_identity']!=digest(provenance['training_source_files'])
                or Path(provenance['training_root']).resolve()!=Path(training_root).resolve()
                or provenance['new_training_updates']!=0):raise ValueError('explicit export training/evaluation lineage')
        training_source=provenance['training_source_files']
        _write_new(root/'lineage.json',provenance)
    runtime=TextWorldRuntime(**results['runtime']);courses={}
    for key in TRAINING_ORDER:
        course=rows[key]['course'];factory=SessionFactory(readers[key],original,runtime=runtime,on_work=lambda event:None)
        if course['binding']['source']!=digest(training_source):raise ValueError('actual original training source identity')
        if digest(course['interface'])!=digest(factory.definition):raise ValueError('actual evaluated interface differs from saved training course')
        courses[f'{key[0]}-{key[1]}']=dict(course=course,profile=asdict(readers[key].profile))
        payload=rows[key]['payload']
        if tensor_digest(readers[key].new_state_dict())!=payload['module_identity']:raise ValueError('export differs from evaluated reader')
        _safe(root/'readers'/f'{key[0]}-{key[1]}'/'closed-checkpoint',payload)
        _copy(Path(training_root)/f'{key[0]}-{key[1]}'/'course.json',root/'readers'/f'{key[0]}-{key[1]}'/'course.json')
    j_bindings={}
    for (objective,seed),spec in j_packages.items():
        payload=load_artifact(spec['path'],kind=J_KIND,identity=spec['identity'])
        if payload['seed']!=seed:raise ValueError('paired J export')
        _safe(root/'J'/f'{objective}-{seed}'/'state',payload)
        # Keep its original loader-compatible bytes as well as safe public data.
        for name in ('manifest.json','payload.pt'):_copy(Path(spec['path'])/name,root/'J'/f'{objective}-{seed}'/'original-artifact'/name)
        j_bindings[f'{objective}-{seed}']=dict(identity=spec['identity'],protocol=payload['protocol'],status=payload['status'])
    from .outcome_dense_arithmetic import identity as arithmetic_identity
    arithmetic=arithmetic_identity(readers[TRAINING_ORDER[0]])
    if any(arithmetic_identity(model)!=arithmetic for model in readers.values()):raise ValueError('six-reader arithmetic binding differs')
    from .outcome_generation import GENERATION_ID
    _write_new(root/'configuration.json',dict(schema='orbit-outcome-collection-v1',contract=CONTRACT,generation=GENERATION_ID,execution_arithmetic=arithmetic,
        readers=[dict(objective=o,seed=s) for o,s in TRAINING_ORDER],J=j_bindings,
        courses=courses,original_provenance=original.provenance,source_files=source_files,
        training_source_files=training_source,training_source_identity=digest(training_source),
        evaluation_source_identity=digest(source_files),lineage=results.get('lineage'),
        limitation='J binds the exact evaluated runtime/read protocol including paths; moving runtime requires requalification, never silently rebind.',
        conditional_RSI=dict(proposal_calls=0,admissions=0,reason='original fixed support cannot satisfy original admission bound')))
    _safe(root/'results',results)
    # Reload the exported original bytes, not just the live training object.
    loaded=load_original(root/'original-public',root/'original-internal',device=probe_ids.device)
    advance('export_original_reloaded')
    checks=[];j_checks=[]
    for key in TRAINING_ORDER:
        restored=load_reader(loaded,root,*key)
        with torch.no_grad():
            positions=torch.tensor([probe_ids.shape[1]-1],device=probe_ids.device)
            mask=torch.ones_like(probe_ids)
            before=readers[key].forward_batch(probe_ids,mask,fast_states=None,memory=None,logit_positions=positions)
            after=restored.forward_batch(probe_ids,mask,fast_states=None,memory=None,logit_positions=positions)
        if not torch.isfinite(before).all() or not torch.equal(before,after):raise ValueError('actual exported reader numerical reload mismatch')
        advance('export_reader_probe_verified')
        checks.append(dict(objective=key[0],seed=key[1],modules=tensor_digest(restored.new_state_dict()),logits=tensor_digest({'logits':after}),exact_equal=True))
        if key in j_packages:
            factory=SessionFactory(restored,loaded,runtime=runtime,on_work=lambda event:None)
            session=factory.session(seed=key[1],tenant='artifact-protocol',reader_identity=factory.reader_identity())
            protocol=outcome_protocol(session,contract=CONTRACT,runtime=runtime.dependency_manifest_sha256)
            spec=j_packages[key]
            head,selector,payload=load_j_package(root/'J'/f'{key[0]}-{key[1]}'/'original-artifact',identity=spec['identity'],reader=restored,expected_protocol=protocol)
            prior=load_artifact(spec['path'],kind=J_KIND,identity=spec['identity'])
            if not exact_equal(payload,prior) or tensor_digest(head.state_dict())!=prior['head_identity']:raise ValueError('exported deployment J differs')
            advance('export_J_reloaded')
            j_checks.append(dict(objective=key[0],seed=key[1],status=payload['status'],head_identity=payload['head_identity'],protocol=digest(protocol),selector_type=type(selector).__name__,calibration=payload['calibration'],exact_equal=True))
            del head,selector,session,factory
        del restored,before,after
    _write_new(root/'reload.json',dict(status='PASS',probe=tensor_digest({'input_ids':probe_ids}),readers=checks,J=j_checks,
        scope='all six inference readers and three actual reader-bound J heads/selectors, with full course configs; no resumed training or changed-runtime J qualification'))
    (root/'README.md').write_text('''# MMLA-Orbit complete-task training collection

Six final readers: static/adapted objectives, seeds 17, 29, 43. Each readers/<objective>-<seed>/course.json records the actual complete task interface, U/V programs, limits, migration and selected microbatch. configuration.json also stores the full current 4096-token architecture profiles. Original public/internal releases remain byte-identical provenance; their old profiles do not describe these trained readers.

This collection uses ordinary PyTorch BF16 projections and eager attention, with full-prefix recomputation for every generated token. configuration.json binds both the arithmetic and generation identities; load_reader verifies them. Cache-specific fixed-schedule kernels and ActionDecoder are not used. The original FP32 omega, rho and hybrid tensors retain their stored precision. The frozen SFT teacher remains separate. Reproducing this collection requires the recorded runtime versions and the supplied loader.

J/<objective>-<seed>/original-artifact is loadable with outcome_j_artifact.load_j_package and the unchanged reader/runtime protocol constructed by SessionFactory and outcome_protocol. Preserve qualification status: unqualified J uses the declared fallback. Runtime paths are part of the protocol; moving it needs requalification. reload.json records actual six-reader numerical parity and three deployed J loader checks. results.json + .safetensors retain all finite evaluation results, including failures and negative findings. Final service accounting is supplied separately in terminal.json after the entire service stops. No recursive successor was proposed or admitted under the fixed support bound.

This is a local collection, not an assertion of publication or performance superiority. Original license and attribution files are retained in both original directories.
''')
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha_file(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    manifest=dict(schema='orbit-outcome-collection-v1',contract=CONTRACT,files=files)
    _write_new(root/'MANIFEST.json',manifest);identity=digest(manifest);verify_collection(root,identity)
    return dict(status='EXPORTED_RELOADED',path=str(root.resolve()),identity=identity,bytes=sum(r['bytes'] for r in files),readers=checks,
        resource_accounting='service-exit reconciliation still required',public_upload=False)
