"""Construct minimal read-only role views; never launch a process on import.

The trusted native supervisor verifies bytes and owns all mounts. Every role
starts with an empty filesystem and a private PID/network/mount namespace.
Host /proc aliases are therefore not a way to reopen withheld audit sources.
This is a namespace specification, not evidence it ran on this host.
"""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import os
import socket

from production_common import read,digest,sha,sha_identity

COLLECTION='e4d8d3728712575af41684ffe76bbcb3a9e43fafd869b3b5484ac4a3184fa11f'
COLLECTION_PREFIXES=('code/','original-public/','original-internal/','readers/')
COLLECTION_FILES=('configuration.json',)
SYSTEM_RO=('/usr','/lib','/lib64','/sys')
ETC_RO=('/etc/ld.so.cache','/etc/localtime','/etc/passwd','/etc/group','/etc/nsswitch.conf')
GPU_DEVICES=('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm')


@dataclass(frozen=True)
class Mount:
    source:str
    target:str
    writable:bool=False

    def __post_init__(self):
        for v in (self.source,self.target):
            if (type(v) is not str or not v.startswith('/') or str(PurePosixPath(v))!=v
                    or '..' in PurePosixPath(v).parts or any(c in v for c in ('\x00','\n','\r'))):
                raise ValueError('canonical absolute mount path required')
        if type(self.writable) is not bool:raise ValueError('explicit write permission required')


def collection_rows(manifest):
    if (manifest.get('schema')!='orbit-outcome-collection-v1' or digest(manifest)!=COLLECTION):
        raise ValueError('exact original full collection manifest required')
    rows=[];seen=set()
    for row in manifest['files']:
        name=row['path'];p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or str(p)!=name or name in seen:
            raise ValueError('unique canonical collection path required')
        seen.add(name)
        if name.startswith(COLLECTION_PREFIXES) or name in COLLECTION_FILES:rows.append(row)
    if not rows or any(not any(x['path'].startswith(pre) for x in rows) for pre in COLLECTION_PREFIXES):
        raise ValueError('all original role and reader computational artifacts required')
    return rows


def verify_actor_collection(root):
    """Authenticate an exact selected view without exposing old result tensors.

The supervisor separately verifies the original complete collection. This
consumer checks the selected files against that exact immutable full manifest,
and rejects any extra result, J artifact, symlink or hidden file in its view.
The original public/internal release subtrees remain complete for their loaders.
"""
    root=Path(root);manifest=read(root/'MANIFEST.json');rows=collection_rows(manifest)
    expected={r['path'] for r in rows}|{'MANIFEST.json'};actual=set()
    for p in root.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):raise ValueError('regular selected view required')
        if p.is_file():actual.add(str(p.relative_to(root)))
    if actual!=expected:raise ValueError('selected view contains withheld or missing files')
    for row in rows:
        p=root/row['path']
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('selected original bytes changed')
    return dict(schema='orbit-selected-inference-view-v1',collection=COLLECTION,
        files=len(rows),bytes=sum(r['bytes'] for r in rows),old_result_payloads_visible=False,
        old_training_data_visible=False,full_original_verification='supervisor prerequisite')


def actor_collection_mounts(source):
    source=str(Path(source).absolute())
    return tuple(Mount(source+'/'+p,'/orbit/collection/'+p) for p in
        ('code','original-public','original-internal','readers','configuration.json','MANIFEST.json'))


def namespace_command(*,bwrap,python_root,role,mounts,work,code,role_file,
                      parent_namespaces,gpu,argv,system_ro=SYSTEM_RO,etc_ro=ETC_RO):
    """Only the fixed host role builder supplies mounts; no role RPC accepts them."""
    if (role not in ('history','support','proposal_reward','actor','qualification') or type(gpu) is not bool or gpu!=(role in ('proposal_reward','actor','qualification'))
            or type(mounts) is not tuple or not all(type(m) is Mount for m in mounts)
            or set(parent_namespaces)!={'mnt','net','pid'} or not argv
            or any(type(a) is not str or '\x00' in a for a in argv)):
        raise ValueError('exact finite role and argument list required')
    writes=[m for m in mounts if m.writable]
    if writes:raise ValueError('extra payload mounts must all be readonly')
    targets=[m.target for m in mounts]
    if len(set(targets))!=len(targets):raise ValueError('duplicate payload target')
    protected=('/usr','/lib','/lib64','/sys','/proc','/dev','/etc','/tmp','/var/tmp',
               '/orbit/code','/orbit/work','/orbit/role.json',str(Path(python_root)))
    for m in mounts:
        if m.target=='/' or any(m.target==p or m.target.startswith(p+'/') or p.startswith(m.target+'/') for p in protected):
            raise ValueError('payload mount may not shadow system, code, scratch or role boundary')
    if any(m.target.startswith('/orbit/collection') for m in mounts) and role not in ('proposal_reward','actor','qualification'):
        raise ValueError('support and historical index cannot mount model artifacts')
    if role in ('proposal_reward','actor','qualification') and any(m.target.startswith('/orbit/private') or m.target.startswith('/orbit/raw') for m in mounts):
        raise ValueError('actor cannot mount raw source or private producer evidence')
    validate_phase_mounts(role,mounts)
    cmd=[bwrap,'--unshare-net','--unshare-pid','--die-with-parent','--new-session','--tmpfs','/']
    # These fixed OS paths have no experiment data or user home directory.
    for p in (*system_ro,*etc_ro):
        if Path(p).exists():cmd+=['--ro-bind',p,p]
    for alias in ('/bin','/sbin'):
        if Path(alias).is_symlink():cmd+=['--symlink',os.readlink(alias),alias]
        elif Path(alias).exists():cmd+=['--ro-bind',alias,alias]
    cmd+=['--ro-bind',python_root,python_root,'--proc','/proc']
    for p in ('/proc/sys','/proc/irq','/proc/bus','/proc/sysrq-trigger'):
        cmd+=['--ro-bind',p,p]
    cmd+=['--dev','/dev','--tmpfs','/dev/shm','--remount-ro','/dev/shm']
    if gpu:
        for p in GPU_DEVICES:cmd+=['--dev-bind',p,p]
    cmd+=['--remount-ro','/dev','--ro-bind',code,'/orbit/code',
          '--ro-bind',role_file,'/orbit/role.json','--bind',work,'/orbit/work',
          '--bind',work+'/tmp','/tmp','--bind',work+'/var-tmp','/var/tmp']
    for m in mounts:cmd+=['--ro-bind',m.source,m.target]
    # All writable storage is bound inside the supervisor's metered output.
    cmd+=['--remount-ro','/','--clearenv','--setenv','PATH','/usr/bin:/bin',
          '--setenv','HOME','/orbit/work/home','--setenv','TMPDIR','/tmp',
          '--setenv','LANG','C.UTF-8','--setenv','PYTHONHASHSEED','0',
          '--setenv','PYTHONDONTWRITEBYTECODE','1','--setenv','HF_HUB_OFFLINE','1',
          '--setenv','TRANSFORMERS_OFFLINE','1','--setenv','CUDA_VISIBLE_DEVICES','0' if gpu else '',
          '--setenv','OMP_NUM_THREADS','1','--setenv','MKL_NUM_THREADS','1',
          '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','TOKENIZERS_PARALLELISM','false',
          '--setenv','CUBLAS_WORKSPACE_CONFIG',':4096:8']
    for key,value in parent_namespaces.items():cmd+=['--setenv','ORBIT_PARENT_'+key.upper(),value]
    return cmd+['--chdir','/orbit/work','--',*argv]


def attest_role(expected_role,*,expected_source):
    """Run inside the admitted namespace before source/model/native imports."""
    spec=read('/orbit/role.json');sha_identity(expected_source)
    if spec['role']!=expected_role or spec['source_identity']!=expected_source or os.geteuid()!=1000:
        raise ValueError('bound ordinary-user role required')
    ns={k:os.readlink('/proc/self/ns/'+k) for k in ('mnt','net','pid')}
    if any(v==os.environ.get('ORBIT_PARENT_'+k.upper()) for k,v in ns.items()):
        raise RuntimeError('actual separate mount/network/PID namespaces required')
    if {n for _,n in socket.if_nameindex()}!={'lo'}:raise RuntimeError('network namespace is not isolated')
    for p in ('/','/orbit/code','/orbit/role.json','/dev','/dev/shm','/proc/sys','/proc/irq','/proc/bus','/proc/sysrq-trigger'):
        if not os.statvfs(p).f_flag & os.ST_RDONLY:raise RuntimeError('readonly boundary missing: '+p)
    if os.statvfs('/proc/self/task').f_flag & os.ST_RDONLY:
        raise RuntimeError('CUDA process metadata must remain writable')
    for p in (*spec['must_be_absent'],*PHASE_ABSENT.get(expected_role,())):
        if Path(p).exists() or Path(p).is_symlink():raise RuntimeError('withheld role path is visible: '+p)
    if expected_role not in ('proposal_reward','actor','qualification') and any(Path(p).exists() for p in GPU_DEVICES):
        raise RuntimeError('non-model role acquired GPU devices')
    for alias,leaf in (('/tmp','tmp'),('/var/tmp','var-tmp')):
        probe=Path(alias)/('mount-proof-'+expected_role+'-'+str(os.getpid()))
        with probe.open('x') as f:f.write(expected_source)
        backing=Path('/orbit/work')/leaf/probe.name
        if not backing.exists() or not os.path.samefile(probe,backing):
            raise RuntimeError('scratch must be inside metered output')
    return dict(role=expected_role,source=expected_source,namespaces=ns,
        private_PID_namespace=True,network_interfaces=['lo'],sudo=False)


# Only the reward learner receives the selected FIT world view. Deployment
# sees completed exports and released public blocks, never FIT or the archive.
PHASE_ABSENT={
    'qualification':('/orbit/public','/orbit/raw','/orbit/private','/orbit/trained'),
    'proposal_reward':('/orbit/public','/orbit/fit-labels','/orbit/trained','/orbit/raw','/orbit/private','/orbit/fit-reuse','/orbit/retained-fit'),
    'actor':('/orbit/fit','/orbit/fit-labels','/orbit/raw','/orbit/private','/orbit/training','/orbit/fit-reuse','/orbit/retained-fit'),
}
PHASE_ROOTS={
    'qualification':('/orbit/collection','/orbit/corrected','/orbit/fit','/orbit/initial','/orbit/retained','/orbit/bindings'),
    'proposal_reward':('/orbit/collection','/orbit/corrected','/orbit/runtime','/orbit/bindings','/orbit/fit','/orbit/initial','/orbit/retained'),
    'actor':('/orbit/collection','/orbit/corrected','/orbit/trained','/orbit/runtime','/orbit/bindings','/orbit/public','/orbit/initial'),
}
PHASE_BINDINGS={
    'qualification':{'/orbit/bindings/fit.json'},
    'proposal_reward':{'/orbit/bindings/fit.json','/orbit/bindings/fit-metadata.json'},
    'actor':{'/orbit/bindings/dev.json','/orbit/bindings/trained.json'},
}
# Published deployments supply their own task host. Historical machine-specific
# mount aliases are not a public runtime dependency or fallback.
PINNED_RUNTIME_ROOTS=()


def validate_phase_mounts(role,mounts):
    if role not in PHASE_ROOTS:return
    targets={m.target for m in mounts}
    for m in mounts:
        if m.target.startswith('/orbit/initial') and m.target not in {
                '/orbit/initial/static-17','/orbit/initial/static-29','/orbit/initial/static-43'}:
            raise ValueError('only the three exact initial-pair directories may be mounted')
        if m.target in PINNED_RUNTIME_ROOTS:
            if role not in ('proposal_reward','actor','qualification') or m.source!=m.target:
                raise ValueError('only native task roles may bind exact original runtime roots')
            continue
        if not any(m.target==p or m.target.startswith(p+'/') for p in PHASE_ROOTS[role]):
            raise ValueError('unregistered stage payload target')
        if m.target.startswith('/orbit/bindings') and m.target not in PHASE_BINDINGS[role]:
            raise ValueError('exact stage binding files required; no whole bindings directory')
        if any(m.target==p or m.target.startswith(p+'/') for p in PHASE_ABSENT[role]):
            raise ValueError('withheld stage payload would be visible')
    needed={'qualification':{'/orbit/fit','/orbit/corrected','/orbit/retained'},'proposal_reward':{'/orbit/fit','/orbit/corrected'},
            'actor':{'/orbit/trained','/orbit/public','/orbit/corrected'}}[role]|PHASE_BINDINGS[role]|{
                '/orbit/initial/static-17','/orbit/initial/static-29','/orbit/initial/static-43'}
    if not needed<=targets:raise ValueError('complete stage payload and binding required')
