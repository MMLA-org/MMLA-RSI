"""Add a fixed logical worker to a verified, separately copied source resolver.

The dependency site and native libraries are reused byte-for-byte. The original
bootstrap, manifest and source tree are never changed. The existing native
attestation continues to inspect the exact four-entry no-site resolver, every
loaded source and the retained native-library binding. Nothing is executed by
this preparation function; a production caller needs normal native admission.
"""
from pathlib import Path,PurePosixPath
import hashlib,json,os

LOGICAL_MODULES=('logical_episode.py','logical_worker.py','logical_backend.py')


def _bytes(path,expected,cap):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<=p.stat().st_size<=cap:
        raise ValueError('bounded regular source artifact required')
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=expected:raise ValueError('source artifact changed')
    return b


def _write(path,data,mode=0o600):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    path.chmod(mode)


def prepare_logical_runtime(*,original_manifest,original_sha256,logical_source,
                            logical_manifest,logical_sha256,output):
    """Create an additive runtime identity, with an exclusive new output root."""
    old_raw=_bytes(original_manifest,original_sha256,16*1024**2);old=json.loads(old_raw)
    source_raw=_bytes(logical_manifest,logical_sha256,64*1024);entries=json.loads(source_raw)
    if (type(entries) is not list or len(entries)!=len(LOGICAL_MODULES)
            or {e['path'] for e in entries}!=set(LOGICAL_MODULES)):
        raise ValueError('exact logical source closure required')
    if (old['schema']!='orbit-textworld-runtime-v1' or old['textworld_version']!='1.7.0'
            or old['alfworld_revision']!='aaba6870f86c5be6a08a491f32a50b906227bc3e'
            or not old['source_files'] or len(old['source_files'])>1024):
        raise ValueError('pinned original native runtime required')
    inventory={str(Path(r['path']).resolve(strict=True)):r for r in old['files']}
    if len(inventory)!=len(old['files']):raise ValueError('duplicate original runtime path')
    python=Path(old['python']).resolve(strict=True);bootstrap=Path(old['bootstrap']).resolve(strict=True)
    for p in (python,bootstrap):
        entry=inventory.get(str(p))
        if entry is None:raise ValueError('original interpreter and bootstrap must be in the inventory')
        b=_bytes(p,entry['sha256'],64*1024**2)
        if len(b)!=entry['bytes']:raise ValueError('runtime executable size changed')
    if any(c in str(python) for c in ('\n','\r',' ')):
        raise ValueError('fixed shebang interpreter path required')
    original_code=Path(old['code_root']).resolve(strict=True)
    supplement=Path(logical_source).resolve(strict=True)
    payload={};total=0
    for name,expected in old['source_files'].items():
        rel=PurePosixPath(name)
        if (rel.is_absolute() or '..' in rel.parts or str(rel)!=name
                or not name.startswith('mmla_pdsa/') or not name.endswith('.py')):
            raise ValueError('original source namespace must remain exact')
        p=original_code/name
        if not p.resolve(strict=True).is_relative_to(original_code):raise ValueError('original source outside root')
        payload[name]=_bytes(p,expected,4*1024**2);total+=len(payload[name])
    if total>32*1024**2:raise ValueError('finite original-source copy bound exceeded')
    for row in entries:
        payload[row['path']]=_bytes(supplement/row['path'],row['sha256'],256*1024)
        if len(payload[row['path']])!=row['bytes']:raise ValueError('logical source size changed')
    required={'mmla_pdsa/full/alfworld_worker.py','mmla_pdsa/full/runtime_attestation.py',
              'mmla_pdsa/full/native_sas_guard.py','mmla_pdsa/full/textworld_state.py'}
    if not required<=set(payload):raise ValueError('original native guards/attestation missing')
    root=Path(output).absolute();root.mkdir(parents=True,exist_ok=False);root=root.resolve(strict=True)
    code=root/'code';code.mkdir()
    for name,data in payload.items():_write(code/name,data)
    wrapper=root/'logical-python';site=str(Path(old['site']).resolve(strict=True))
    text=(f'#!{python} -S\nimport os,runpy,sys,sysconfig\nsys.dont_write_bytecode=True\n'
        f'sys.path=[{site!r},{str(code)!r},sysconfig.get_path("stdlib"),os.path.join(sysconfig.get_path("stdlib"),"lib-dynload")]\n'
        'if sys.argv[1:] != ["-u","-m","logical_worker"]:raise SystemExit("fixed logical worker only")\n'
        'sys.argv=["logical_worker"]\n'
        'runpy.run_module("logical_worker",run_name="__main__")\n')
    _write(wrapper,text.encode(),0o700)
    manifest=dict(old)
    manifest.update(code_root=str(code),bootstrap=str(wrapper),
        source_files={name:hashlib.sha256(b).hexdigest() for name,b in payload.items()},
        files=[*old['files'],dict(path=str(wrapper),bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())],
        parent_runtime_sha256=original_sha256,logical_source_sha256=logical_sha256,
        status='ADDITIVE_LOGICAL_SOURCE_ASSEMBLED_NOT_NATIVE_QUALIFIED',
        resolver='exact private dependency site, copied original and logical source, stdlib, lib-dynload; no-site',
        pending=['actual derived-world native reset/step/replay; no scientific execution granted by assembly'])
    data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    path=root/'runtime-manifest.json';_write(path,data)
    closure=root/'logical-source.json';_write(closure,source_raw)
    return dict(python=str(wrapper),code_root=str(code),dependency_manifest=str(path),
        dependency_manifest_sha256=hashlib.sha256(data).hexdigest(),
        logical_source_root=str(code),logical_source_manifest=str(closure),
        logical_source_sha256=logical_sha256,
        worker_sha256=hashlib.sha256(payload['logical_worker.py']).hexdigest(),
        original_runtime_sha256=original_sha256,native_qualified=False)
