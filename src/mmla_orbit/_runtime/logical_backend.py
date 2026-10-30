"""Process transport for explicit logical worlds; frozen RPC/cleanup semantics."""
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,time
from mmla_pdsa.full.alfworld_episode import EpisodeLimits,_digest,_sha
from mmla_pdsa.full.textworld_backend import TextWorldRuntime,TextWorldProcessBackend
from logical_episode import LogicalWorldPermit


@dataclass(frozen=True)
class LogicalRuntime(TextWorldRuntime):
    logical_source_root: str = ''
    logical_source_manifest: str = ''
    logical_source_sha256: str = ''

    def __post_init__(self):
        super().__post_init__();_sha(self.logical_source_sha256)
        if not self.logical_source_root or not self.logical_source_manifest:
            raise ValueError('reviewed derived-source closure required')


def verify_sources(runtime):
    base=Path(runtime.logical_source_root).resolve(strict=True)
    raw=Path(runtime.logical_source_manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=runtime.logical_source_sha256:raise ValueError('logical source closure changed')
    entries=json.loads(raw)
    if (type(entries) is not list or {x['path'] for x in entries}!=
            {'logical_episode.py','logical_worker.py','logical_backend.py'} or len(entries)!=3):
        raise ValueError('complete derived worker source closure required')
    for row in entries:
        p=base/row['path'];data=p.read_bytes()
        if p.is_symlink() or len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('derived worker source changed')
    if hashlib.sha256((base/'logical_worker.py').read_bytes()).hexdigest()!=runtime.worker_sha256:
        raise ValueError('exact derived worker required')


class LogicalProcessBackend(TextWorldProcessBackend):
    def __init__(self,permit,seed,limits,*,runtime,on_cost):
        if (type(permit) is not LogicalWorldPermit or type(runtime) is not LogicalRuntime
                or type(limits) is not EpisodeLimits or limits.max_steps!=64
                or limits.expose_admissible_commands or type(seed) is not int or not 0<=seed<2**32
                or not callable(on_cost)):
            raise ValueError('exact derived role, runtime and actual cost sink required')
        permit.verify_files(runtime.data_root);verify_sources(runtime)
        manifest=Path(runtime.dependency_manifest).read_bytes()
        if hashlib.sha256(manifest).hexdigest()!=runtime.dependency_manifest_sha256:
            raise ValueError('reviewed runtime manifest changed')
        declaration=json.loads(manifest)
        code_root=Path(runtime.code_root).resolve(strict=True)
        if (Path(runtime.logical_source_root).resolve(strict=True)!=code_root
                or Path(declaration['code_root']).resolve(strict=True)!=code_root
                or Path(declaration['bootstrap']).resolve(strict=True)!=Path(runtime.python).resolve(strict=True)
                or declaration.get('logical_source_sha256')!=runtime.logical_source_sha256
                or any(declaration['source_files'].get(n)!=hashlib.sha256((code_root/n).read_bytes()).hexdigest()
                       for n in ('logical_episode.py','logical_worker.py','logical_backend.py'))):
            raise ValueError('additive no-site logical runtime declaration required')
        wrapper=next((r for r in declaration['files'] if Path(r['path']).resolve()==Path(runtime.python).resolve()),None)
        if wrapper is None or hashlib.sha256(Path(runtime.python).read_bytes()).hexdigest()!=wrapper['sha256']:
            raise ValueError('fixed logical bootstrap changed')
        self.identity=_digest(dict(schema='orbit-derived-textworld-process-v1',
            dependency_manifest=runtime.dependency_manifest_sha256,worker=runtime.worker_sha256,
            logical_source_closure=runtime.logical_source_sha256,
            original_worker_sha256=hashlib.sha256((Path(runtime.code_root)/'mmla_pdsa/full/alfworld_worker.py').read_bytes()).hexdigest(),
            teacher='ABSENT',hash_seed=0,demangle_shuffle=False))
        self.has_teacher=False;self._runtime,self._cost,self._closed=runtime,on_cost,False
        self._reset,self._label,self._last_cpu=False,None,0.;self._pending=b''
        work=Path(runtime.work_root);work.mkdir(parents=True,exist_ok=True)
        self._directory=tempfile.TemporaryDirectory(prefix='logical-',dir=work)
        self._stderr=open(Path(self._directory.name)/'stderr.log','wb');self._started=time.monotonic()
        env=dict(PATH='/usr/bin:/bin',HOME=self._directory.name,TMPDIR=self._directory.name,
            PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',ALFWORLD_DATA=self._directory.name,LANG='C.UTF-8')
        # Explicit additive bootstrap retains the original strict four-entry
        # no-site resolver and attests every logical module under source_files.
        # The legacy fixed-worker bootstrap is never passed a -c program.
        try:
            self._process=subprocess.Popen([runtime.python,'-u','-m','logical_worker'],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=self._stderr,env=env,cwd=self._directory.name,
                bufsize=0,start_new_session=True)
        except BaseException:
            self._closed=True;self._stderr.close();self._directory.cleanup();raise
        os.set_blocking(self._process.stdin.fileno(),False)
        try:
            response=self._rpc(dict(operation='initialize',purpose='validated-logical-execution',
                permit=asdict(permit),seed=seed,limits=asdict(limits),data_root=str(Path(runtime.data_root).resolve()),
                dependency_manifest=str(Path(runtime.dependency_manifest).resolve()),
                dependency_manifest_sha256=runtime.dependency_manifest_sha256,
                identity=self.identity,cpu_seconds=runtime.worker_cpu_seconds))
            self.runtime_attestation=response['runtime_attestation']
            if self.runtime_attestation.get('schema')!='orbit-worker-runtime-attestation-v1':
                raise ValueError('actual native import attestation required')
            on_cost(dict(kind='environment_runtime_attestation',identity=_digest(self.runtime_attestation)))
        except BaseException:self.close();raise

    def _frame(self,response):
        if response.get('teacher_label') is not None or response.get('teacher_status')!='ABSENT':
            raise PermissionError('derived actor received a teacher field')
        return super()._frame(response)
