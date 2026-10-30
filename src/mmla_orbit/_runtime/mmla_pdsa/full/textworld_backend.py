"""Process-isolated pinned TextWorld backend; constructed only by the run entry.

The PDDL C API has no session argument. Every episode/replay therefore owns a
new process. This file does not install dependencies or execute at import.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import tempfile
import time

from .alfworld_episode import BackendFrame, EpisodeLimits, TaskPermit, _digest, _sha


@dataclass(frozen=True)
class TextWorldRuntime:
    python: str
    code_root: str
    data_root: str
    work_root: str
    dependency_manifest: str
    dependency_manifest_sha256: str
    worker_sha256: str
    worker_cpu_seconds: int
    request_timeout_seconds: float
    max_reply_bytes: int = 1048576

    def __post_init__(self):
        _sha(self.dependency_manifest_sha256); _sha(self.worker_sha256)
        if (type(self.worker_cpu_seconds) is not int or not 1 <= self.worker_cpu_seconds <= 600
                or type(self.request_timeout_seconds) not in (int, float)
                or not 0 < self.request_timeout_seconds <= 120
                or type(self.max_reply_bytes) is not int or not 1024 <= self.max_reply_bytes <= 1048576):
            raise ValueError('finite isolated backend bounds required')


class TextWorldProcessBackend:
    """Private runner handle; no actor is given process, paths or hidden hashes."""
    def __init__(self, permit, seed, limits, *, runtime, on_cost):
        if type(permit) is not TaskPermit or type(limits) is not EpisodeLimits or type(runtime) is not TextWorldRuntime:
            raise TypeError('exact role/limits/runtime required')
        if type(seed) is not int or not 0 <= seed < 2**32 or not callable(on_cost):
            raise ValueError('fixed seed and actual backend cost sink required')
        game, trajectory = permit.verify_files(runtime.data_root)
        worker = Path(runtime.code_root)/'mmla_pdsa/full/alfworld_worker.py'
        if hashlib.sha256(worker.read_bytes()).hexdigest() != runtime.worker_sha256:
            raise ValueError('worker source differs from the reviewed original')
        manifest = Path(runtime.dependency_manifest).read_bytes()
        if hashlib.sha256(manifest).hexdigest() != runtime.dependency_manifest_sha256:
            raise ValueError('dependency manifest differs from the reviewed runtime')
        self.identity = _digest(dict(schema='orbit-textworld-process-v1',
            textworld='1.7.0', alfworld='aaba6870f86c5be6a08a491f32a50b906227bc3e',
            dependency_manifest=runtime.dependency_manifest_sha256, worker=runtime.worker_sha256,
            hash_seed=0, demangle_shuffle=False, goal='original-reset-observation',
            teacher='FIT-only-cached-current-state-label-with-status'))
        self.has_teacher = permit.teacher_allowed
        self._runtime, self._cost, self._closed = runtime, on_cost, False
        self._reset, self._label, self._last_cpu = False, None, 0.
        self._pending = b''
        work_root = Path(runtime.work_root); work_root.mkdir(parents=True, exist_ok=True)
        self._directory = tempfile.TemporaryDirectory(prefix='textworld-', dir=work_root)
        self._stderr = open(Path(self._directory.name)/'stderr.log', 'wb')
        # No inherited network credentials or parent CUDA visibility. Runtime
        # Python/package identities are explicit, not selected by PATH.
        env = dict(PATH='/usr/bin:/bin', HOME=self._directory.name, TMPDIR=self._directory.name,
            PYTHONPATH=str(Path(runtime.code_root).resolve()), PYTHONHASHSEED='0',
            CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            ALFWORLD_DATA=self._directory.name, LANG='C.UTF-8')
        self._started = time.monotonic()
        try:
            self._process = subprocess.Popen([runtime.python, '-u', '-m', 'mmla_pdsa.full.alfworld_worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, env=env,
                cwd=self._directory.name, bufsize=0, start_new_session=True)
        except BaseException:
            self._closed = True
            self._stderr.close()
            self._directory.cleanup()
            raise
        os.set_blocking(self._process.stdin.fileno(), False)
        try:
            initialized=self._rpc(dict(operation='initialize', permit=asdict(permit), seed=seed,
                limits=asdict(limits), game=str(game), trajectory=str(trajectory),
                dependency_manifest=str(Path(runtime.dependency_manifest).resolve()),
                dependency_manifest_sha256=runtime.dependency_manifest_sha256,
                identity=self.identity, cpu_seconds=runtime.worker_cpu_seconds))
            self.runtime_attestation=initialized['runtime_attestation']
            if self.runtime_attestation.get('schema')!='orbit-worker-runtime-attestation-v1':raise ValueError('actual worker import attestation required')
            self._cost(dict(kind='environment_runtime_attestation',identity=_digest(self.runtime_attestation)))
        except BaseException:
            self.close()
            raise

    def _rpc(self, message):
        if self._closed:
            raise RuntimeError('backend process is closed')
        start = time.monotonic()
        wire = json.dumps(message, ensure_ascii=False, allow_nan=False).encode()+b'\n'
        if len(wire) > 65536:
            raise ValueError('bounded backend request required')
        try:
            deadline = start + self._runtime.request_timeout_seconds
            # Bound writes too: a live but stalled worker must not block its
            # commander when a request exceeds the pipe's available capacity.
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdin, selectors.EVENT_WRITE)
                sent = 0
                while sent < len(wire):
                    remain = deadline-time.monotonic()
                    if remain <= 0 or not selector.select(remain):
                        raise TimeoutError('isolated TextWorld request timed out')
                    try:
                        sent += os.write(self._process.stdin.fileno(), wire[sent:])
                    except BlockingIOError:
                        continue
                selector.unregister(self._process.stdin)
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while b'\n' not in self._pending:
                    remain = deadline-time.monotonic()
                    if remain <= 0 or not selector.select(remain):
                        raise TimeoutError('isolated TextWorld response timed out')
                    chunk = os.read(self._process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('isolated TextWorld exited without complete response')
                    self._pending += chunk
                    if len(self._pending) > self._runtime.max_reply_bytes:
                        raise ValueError('isolated TextWorld response exceeds byte bound')
            line, self._pending = self._pending.split(b'\n', 1)
            response = json.loads(line)
            cpu = response['cpu_seconds']
            if type(cpu) not in (float, int) or not self._last_cpu <= cpu < float('inf'):
                raise ValueError('invalid cumulative backend CPU account')
            self._cost(dict(kind='environment_rpc', operation=message['operation'],
                pid=self._process.pid, wall_seconds=time.monotonic()-start,
                cpu_seconds=cpu-self._last_cpu, cumulative_cpu_seconds=cpu,
                success=response.get('ok') is True, teacher_status=response.get('teacher_status')))
            self._last_cpu = cpu
            if response.get('ok') is not True or response.get('identity') != self.identity:
                raise RuntimeError('isolated TextWorld operation failed: '+str(response.get('error', 'identity mismatch')))
            return response
        except BaseException:
            self.close()
            raise

    def _frame(self, response):
        frame = dict(response['frame'])
        frame['admissible_commands'] = tuple(frame['admissible_commands'])
        self._label = response['teacher_label']
        return BackendFrame(**frame)

    def reset(self):
        if self._reset:
            raise RuntimeError('one reset per isolated process; construct a fresh replay')
        self._reset = True
        return self._frame(self._rpc(dict(operation='reset')))

    def step(self, action):
        if not self._reset:
            raise RuntimeError('reset required before action')
        return self._frame(self._rpc(dict(operation='step', action=action)))

    def teacher_action(self):
        if not self.has_teacher:
            raise PermissionError('no teacher exists in non-FIT process')
        if self._closed or not self._reset:
            raise RuntimeError('current state required')
        return self._label  # Pure local cached read, no environment/RNG advance.

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic()+2
            while True:
                pid, status, usage = os.wait4(self._process.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() >= deadline:
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    pid, status, usage = os.wait4(self._process.pid, 0)
                    break
                time.sleep(.01)
            self._process.returncode = os.waitstatus_to_exitcode(status)
            # Always kill any translator child retaining this process group.
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._stderr.close()
            log = Path(self._directory.name)/'stderr.log'
            evidence = log.read_bytes()[-8192:]
            self._cost(dict(kind='environment_close', pid=self._process.pid,
                wall_seconds=time.monotonic()-self._started, returncode=self._process.returncode,
                accounted_cpu_seconds=self._last_cpu,
                wait4_cpu_seconds=usage.ru_utime+usage.ru_stime,
                extra_reaped_cpu_seconds=max(0., usage.ru_utime+usage.ru_stime-self._last_cpu),
                max_rss_kib=usage.ru_maxrss,
                cpu_complete=False, cpu_note='wait4 includes reaped process usage; run supervisor must also meter any killed/unreaped translator descendants',
                stderr_tail=evidence.decode('utf-8', errors='replace')))
        finally:
            self._process.stdin.close(); self._process.stdout.close()
            self._directory.cleanup()
