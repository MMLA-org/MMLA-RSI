"""Finite whole-service accounting for the single outcome training contract.

The production launcher must put this supervisor and every descendant in one
new dedicated systemd user service (KillMode=control-group). cpu.stat measures
all descendants, including detached translators and reaped build processes.
This module never starts systemd or grants scientific execution permission.
"""
from dataclasses import dataclass,asdict
import hashlib,json,math,os,re,signal,socket,stat,subprocess,time
from pathlib import Path
from .pidfd import Pidfd

PHASE_LIMITS={
    'preflight':(1800.,3600.), 'train':(43200.,21600.),
    'dev':(18000.,28800.), 'final':(14400.,21600.), 'export':(3600.,3600.)}
# The 1.5 GPU / 2 CPU hours reserved for ineligible RSI are NOT redistributed.
SEGMENTS=(('runtime','preflight',False),('preflight','preflight',True),
          ('train','train',True),('dev','dev',True),('final','final',True),('export','export',True))

@dataclass(frozen=True)
class Limits:
    gpu:float=86400.
    cpu:float=86400.
    wall:float=172800.
    disk:int=64*1024**3
    reserve_seconds:float=120.
    reserve_bytes:int=512*1024**2
    reserve_cpu_seconds:float=120.
    def __post_init__(self):
        if any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in (self.gpu,self.cpu,self.wall,self.reserve_seconds,self.reserve_cpu_seconds)):
            raise ValueError('positive finite limits')
        if (type(self.disk) is not int or type(self.reserve_bytes) is not int
                or not 0<=self.reserve_bytes<self.disk or self.reserve_seconds>=min(self.gpu,self.cpu,self.wall) or self.reserve_cpu_seconds>=self.cpu):
            raise ValueError('finite storage and nonempty stop margin')


class ResourceLedger:
    """Monotone cumulative costs, with one immutable phase order and origin."""
    def __init__(self,run_id,started_unix,*,limits=Limits(),clock=time.monotonic,wall_clock=time.time,prior_preflight_gpu_seconds=0.):
        if (type(run_id) is not str or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',run_id)
                or type(started_unix) not in (int,float) or not math.isfinite(started_unix)
                or started_unix<=0 or type(limits) is not Limits):raise ValueError('original identity/origin required')
        self.run_id,self.started_unix,self.limits=run_id,started_unix,limits
        self._clock=clock;self._origin=clock();self._last=self._origin
        self._boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self._initial_wall=wall_clock()-started_unix
        if self._initial_wall<0:raise ValueError('origin cannot be in the future')
        self.cpu=0.;self.gpu=0.;self.disk=0;self.peak_disk=0;self.wall=self._initial_wall
        self.phases={p:dict(cpu=0.,gpu=0.) for p in PHASE_LIMITS}
        self.segment=-1;self.active=False;self.gpu_active=False;self.phase='preflight'
        self.events=0;self._hash=hashlib.sha256();self.failures=[]
        if (type(prior_preflight_gpu_seconds) not in (int,float) or not math.isfinite(prior_preflight_gpu_seconds)
                or not 0<=prior_preflight_gpu_seconds<PHASE_LIMITS['preflight'][0]):
            raise ValueError('bounded inherited preflight GPU cost required')
        self.gpu=float(prior_preflight_gpu_seconds);self.phases['preflight']['gpu']=self.gpu

    def sample(self,*,cpu_seconds,disk_bytes):
        now=self._clock();elapsed=now-self._last
        if (elapsed<0 or type(cpu_seconds) not in (float,int) or not math.isfinite(cpu_seconds)
                or cpu_seconds<self.cpu or type(disk_bytes) is not int or disk_bytes<0):
            raise ValueError('resource counters or clock moved backwards / became invalid')
        self.phases[self.phase]['cpu']+=cpu_seconds-self.cpu
        self.cpu=float(cpu_seconds);self.disk=disk_bytes;self.peak_disk=max(self.peak_disk,disk_bytes)
        if self.gpu_active:
            self.gpu+=elapsed;self.phases[self.phase]['gpu']+=elapsed
        self._last=now;self.wall=self._initial_wall+now-self._origin

    def begin(self,name):
        next_index=self.segment+1
        if self.failures or self.active or next_index>=len(SEGMENTS) or name!=SEGMENTS[next_index][0]:
            raise ValueError('one original finite segment order, no retry or skipped stage')
        # Caller samples immediately before begin and after each child exit.
        self.segment=next_index;_,self.phase,self.gpu_active=SEGMENTS[next_index];self.active=True

    def end(self,*,success):
        if not self.active or type(success) is not bool:raise ValueError('one active segment')
        self.active=False;self.gpu_active=False
        if not success:self.failures.append(SEGMENTS[self.segment][0])

    def violations(self,*,reserve=True):
        seconds=self.limits.reserve_seconds if reserve else 0.
        cpu_reserve=self.limits.reserve_cpu_seconds if reserve else 0.
        storage=self.limits.reserve_bytes if reserve else 0
        v=[]
        for name,value,cap in (('total_cpu',self.cpu,self.limits.cpu),('total_gpu',self.gpu,self.limits.gpu),('wall',self.wall,self.limits.wall)):
            if value>=cap-(cpu_reserve if name.endswith('cpu') else seconds):v.append(name)
        gpu,cpu=PHASE_LIMITS[self.phase]
        for name,value,cap in (('phase_cpu',self.phases[self.phase]['cpu'],cpu),('phase_gpu',self.phases[self.phase]['gpu'],gpu)):
            if value>=cap-(cpu_reserve if name.endswith('cpu') else seconds):v.append(name)
        if self.peak_disk>=self.limits.disk-storage:v.append('disk')
        return v

    def note_persisted_event(self,wire):
        # Called only after the complete event bytes have been written+fsynced.
        if type(wire) is not bytes or not wire.endswith(b'\n'):raise ValueError('complete durable event bytes')
        self._hash.update(wire);self.events+=1

    def checkpoint(self):
        return dict(run_id=self.run_id,started_unix=self.started_unix,
            deadline_unix=self.started_unix+self.limits.wall,wall_seconds=self.wall,
            CPU_seconds=self.cpu,GPU_seconds=self.gpu,events_count=self.events,events_sha256=self._hash.hexdigest())

    def snapshot(self):
        return dict(**self.checkpoint(),schema='outcome-service-resources-v1',limits=asdict(self.limits),
            sample_monotonic=self._last,boot_id=self._boot_id,phase=self.phase,segment=SEGMENTS[self.segment][0] if self.segment>=0 else None,
            segment_active=self.active,gpu_reserved=self.gpu_active,phase_costs={k:dict(v) for k,v in self.phases.items()},
            allocated_bytes=self.disk,peak_allocated_bytes=self.peak_disk,
            early_stop_reasons=self.violations(),hard_limit_exceeded=self.violations(reserve=False),
            failed_segments=list(self.failures))


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class DurableEvents:
    def __init__(self,path,ledger):
        self.file=Path(path).open('xb',buffering=0);self.ledger=ledger
        try:
            os.fsync(self.file.fileno());_sync_directory(Path(path).parent)
        except BaseException:
            self.file.close();raise
        self.failed=False;self.persisted_bytes=0
    def write(self,value):
        if self.failed:raise RuntimeError('event journal has an uncommitted tail')
        wire=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if getattr(self.ledger,'workload_bounded',False):
            # Match both independent restore and terminal readers. Keep room
            # for an honest stop event; refusal does not corrupt the prefix.
            terminal=value.get('kind')=='incomplete'
            byte_limit=512*1024**2-(0 if terminal else 1024**2)
            event_limit=400000-(0 if terminal else 16)
            if len(wire)>32768 or self.persisted_bytes+len(wire)>byte_limit or self.ledger.events+1>event_limit:
                raise RuntimeError('durable resource journal capacity reached')
        try:
            pending=memoryview(wire)
            while pending:
                count=self.file.write(pending)
                if count is None or count<=0:raise OSError('incomplete event write')
                pending=pending[count:]
            os.fsync(self.file.fileno())
        except BaseException:
            self.failed=True
            raise
        self.ledger.note_persisted_event(wire);self.persisted_bytes+=len(wire)
    def close(self):self.file.close()
    def receipt(self):
        return dict(journal_complete=not self.failed,persisted_prefix_bytes=self.persisted_bytes,
            observed_file_bytes=os.fstat(self.file.fileno()).st_size)


def allocated_bytes(root,member_pids=()):
    """Allocated run files plus regular unlinked open/mapped descendant files.

    All mutable caches/builds/downloads live below run_root. Directory errors
    fail closed. Unique inode keys avoid counting hardlinks or mappings twice.
    Native storage containment/quota remains an entry qualification obligation;
    this sampled meter alone is NOT a kernel storage quota.
    """
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('real confined run directory required')
    total=0;seen=set()
    def charge(st):
        nonlocal total
        key=(st.st_dev,st.st_ino)
        if key not in seen:total+=st.st_blocks*512;seen.add(key)
    def failed(error):raise error
    for parent,dirs,files in os.walk(root,followlinks=False,onerror=failed):
        for name in dirs+files:
            p=Path(parent)/name
            try:st=p.lstat()
            except FileNotFoundError:continue
            if stat.S_ISLNK(st.st_mode):raise ValueError('run storage contains a symlink: '+str(p))
            charge(st)
    for pid in member_pids:
        if type(pid) is not int or pid<=0:raise ValueError('actual service member PID required')
        proc=Path('/proc')/str(pid)
        try:descriptors=list((proc/'fd').iterdir())
        except FileNotFoundError:continue
        for fd in descriptors:
            try:st=fd.stat()
            except FileNotFoundError:continue
            if stat.S_ISREG(st.st_mode) and st.st_nlink==0:charge(st)
        try:maps=(proc/'maps').read_text().splitlines()
        except FileNotFoundError:continue
        for row in maps:
            if not row.endswith(' (deleted)'):continue
            address=row.split()[0]
            if not re.fullmatch(r'[0-9a-f]+-[0-9a-f]+',address):raise ValueError('invalid mapping address')
            try:st=(proc/'map_files'/address).stat()
            except FileNotFoundError:continue
            if stat.S_ISREG(st.st_mode):charge(st)
    return total


class CgroupMeter:
    def __init__(self,unit):
        if type(unit) is not str or not re.fullmatch(r'mmla-outcome-20261017-[0-9a-f]{12}\.service',unit):
            raise ValueError('exact dedicated systemd service required')
        raw=Path('/proc/self/cgroup').read_text().splitlines()
        if len(raw)!=1 or not raw[0].startswith('0::/'):raise RuntimeError('cgroup v2 required')
        self.relative=raw[0][3:]
        if Path(self.relative).name!=unit or '..' in Path(self.relative).parts:raise RuntimeError('not in the bound run service')
        self.path=Path('/sys/fs/cgroup')/self.relative.lstrip('/')
        self.identity=(self.path.stat().st_dev,self.path.stat().st_ino)
        self.pid=os.getpid();self.cpu_seconds();self.handles=Pidfd();self.handles.probe()
        if self.members()!={self.pid}:raise RuntimeError('service must start with this supervisor alone')

    def _check(self):
        if self.identity!=(self.path.stat().st_dev,self.path.stat().st_ino):raise RuntimeError('run cgroup replaced')
        if Path('/proc/self/cgroup').read_text().strip()!='0::'+self.relative:raise RuntimeError('supervisor escaped run cgroup')

    def cpu_seconds(self):
        self._check() if hasattr(self,'identity') else None
        values=dict(line.split() for line in (self.path/'cpu.stat').read_text().splitlines())
        count=int(values['usage_usec'])
        if count<0:raise RuntimeError('invalid cpu counter')
        return count/1e6

    def members(self):
        self._check();result=set()
        def failed(error):raise error
        for parent,dirs,files in os.walk(self.path,onerror=failed):
            if 'cgroup.procs' in files:
                result.update(int(x) for x in (Path(parent)/'cgroup.procs').read_text().split())
        return result

    def signal_descendants(self,sig):
        """PID handles prevent reuse races; detached sessions remain in cgroup."""
        for pid in sorted(self.members()-{self.pid}):
            try:
                fd=self.handles.open(pid)
            except ProcessLookupError:continue
            try:
                # Verify membership again AFTER opening the handle.
                if pid in self.members():self.handles.send(fd,sig)
            except ProcessLookupError:pass
            finally:os.close(fd)


def _json_atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.part')
    with temp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path);_sync_directory(path.parent)


def supervise_segments(*,commands,run_root,unit,run_id,started_unix,environment,
                       setup_cpu_seconds=0.,prior_preflight_gpu_seconds=0.,storage_meter=None,notify_systemd=False,
                       cpu_cut_binding=None,workload_policy=False,prior_phase_costs=None):
    """One normally admitted run; commands are fixed by the reviewed entry.

    Sample every 0.5 s, stop 120 aggregate CPU s / 120 GPU-wall s / 512 MiB early; preserve measured
    overshoot. Systemd KillMode=control-group is the final cleanup backstop.
    No failed segment is retried. This meter is not the native quota mechanism:
    CPU rate, storage containment and external deadline/cleanup must be qualified
    by the fixed launcher. Final counters require collection after service exit.
    The module is not a general launch CLI.
    """
    if (type(commands) is not dict or tuple(commands)!=tuple(x[0] for x in SEGMENTS)
            or any(type(v) not in (tuple,list) or not v or any(type(x) is not str or not x for x in v)
                   or not Path(v[0]).is_absolute() for v in commands.values())):
        raise ValueError('all six exact segment argv vectors required')
    root=Path(run_root).resolve(strict=True)
    if type(workload_policy) is not bool:raise TypeError('explicit workload policy flag')
    if prior_phase_costs is not None and not workload_policy:raise ValueError('phase inheritance requires reviewed work-count policy')
    if (not math.isfinite(setup_cpu_seconds) or setup_cpu_seconds<0 or (not workload_policy and setup_cpu_seconds>=3600)
            or storage_meter is not None and not callable(storage_meter)):
        raise ValueError('actual finite native setup cost and fixed storage meter')
    def notify(message):
        if not notify_systemd:return
        address=os.environ.get('NOTIFY_SOCKET','')
        if not address:raise RuntimeError('native watchdog socket missing')
        if address.startswith('@'):address='\0'+address[1:]
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as s:s.connect(address);s.sendall(message.encode())
    meter=CgroupMeter(unit)
    limits=Limits() if cpu_cut_binding is None else Limits(reserve_cpu_seconds=256.)
    if workload_policy:
        from .outcome_workload_resources import WorkloadLedger,STALL_SECONDS
        ledger=WorkloadLedger(run_id,started_unix,prior_preflight_gpu_seconds=prior_preflight_gpu_seconds,prior_phase_costs=prior_phase_costs)
        if ledger.cpu>setup_cpu_seconds:raise ValueError('setup cumulative CPU below inherited phase charge')
    else:ledger=ResourceLedger(run_id,started_unix,prior_preflight_gpu_seconds=prior_preflight_gpu_seconds,limits=limits)
    last_cpu_cut=None
    if cpu_cut_binding is not None:
        if (set(cpu_cut_binding)!={'allowed_cpus','boot_id'} or len(cpu_cut_binding['allowed_cpus'])!=8
                or len(set(cpu_cut_binding['allowed_cpus']))!=8):raise ValueError('exact eight-CPU native binding')
    def save_cpu_cut():
        nonlocal last_cpu_cut
        from .outcome_cpu_cut import live_sample,_cpus
        cut=live_sample(meter)
        if (cut['boot_id']!=cpu_cut_binding['boot_id']
                or _cpus((meter.path/'cpuset.cpus').read_text())!=cpu_cut_binding['allowed_cpus']
                or not set(cut['effective_cpus'])<=set(cpu_cut_binding['allowed_cpus'])
                or last_cpu_cut is not None and cut['service_CPU_seconds']<last_cpu_cut['service_CPU_seconds']):
            raise RuntimeError('native CPU cut identity/capacity/monotonicity changed')
        _json_atomic(root/'cpu-cut.json',dict(sample=cut,run_id=run_id,unit=unit,setup_CPU_seconds=setup_cpu_seconds,
            allowed_cpus=cpu_cut_binding['allowed_cpus'],capacity_cpus=8))
        last_cpu_cut=cut
        return cut['service_CPU_seconds']
    events=root/'resource-events.jsonl'
    # Exclusive event creation prevents a new start from overwriting old costs.
    journal=DurableEvents(events,ledger)
    try:
        last_persisted=[0.]
        def record(kind,*,enforce=True,**detail):
            # Persist a raw bracketed cut before any slow storage walk, then
            # take a fresh cut afterwards for the ledger/watchdog decision.
            if cpu_cut_binding is not None:save_cpu_cut()
            disk=allocated_bytes(root,meter.members()) if storage_meter is None else storage_meter()
            cpu=meter.cpu_seconds() if cpu_cut_binding is None else save_cpu_cut()
            ledger.sample(cpu_seconds=setup_cpu_seconds+cpu,disk_bytes=disk)
            # Renew only from a fresh safe sample, BEFORE any blocking durable I/O.
            # Repaired entry reserves256 CPU seconds for the complete chain:
            # 0.5s sampling + 8s watchdog + 5s abort + 5s ExecStopPost, at8CPUs,
            # with remaining CPU for launcher closeout. GPU/wall reserve stays120s.
            if enforce:
                if ledger.violations():raise RuntimeError('finite resource reserve reached')
                notify('WATCHDOG=1')
            # Fresh CPU/space/watchdog checks still occur each sample. Only
            # durable summary frequency changes, keeping full transition cuts.
            if not workload_policy or kind!='sample' or time.monotonic()-last_persisted[0]>=30.:
                journal.write(dict(kind=kind,resources=ledger.snapshot(),**detail))
                _json_atomic(root/'resource-latest.json',ledger.snapshot());last_persisted[0]=time.monotonic()
        def terminate():
            errors=[]
            try:
                meter.signal_descendants(signal.SIGTERM)
                end=time.monotonic()+5
                while meter.members()-{os.getpid()} and time.monotonic()<end:
                    time.sleep(.1)
                    # No durable I/O or watchdog renewal in the shutdown grace.
            except BaseException as error:errors.append(repr(error))
            finally:
                # Accounting may be the reason for shutdown. It must never
                # prevent escalation against every remaining service member.
                try:meter.signal_descendants(signal.SIGKILL)
                except BaseException as error:errors.append(repr(error))
            end=time.monotonic()+5
            try:
                while meter.members()-{os.getpid()} and time.monotonic()<end:
                    time.sleep(.1)
                    # Native watchdog continues towards its original expiry.
                remaining=sorted(meter.members()-{os.getpid()})
                if remaining:errors.append('live descendants after KILL: '+repr(remaining))
            except BaseException as error:errors.append(repr(error))
            return errors
        process=None
        try:
            notify('READY=1')
            record('start',unit=unit)
            for name,_,_ in SEGMENTS:
                record('before_segment',name=name)
                if ledger.violations():raise RuntimeError('resource reserve reached before '+name)
                if meter.members()!={os.getpid()}:raise RuntimeError('prior segment left live descendants')
                ledger.begin(name)
                record('segment_armed',name=name) # Publish active phase before the child can read it.
                with (root/(name+'.stdout.log')).open('xb') as stdout,(root/(name+'.stderr.log')).open('xb') as stderr:
                    process=subprocess.Popen(commands[name],cwd=root,env=environment,stdin=subprocess.DEVNULL,
                        stdout=stdout,stderr=stderr,start_new_session=True)
                    record('segment_started',name=name,pid=process.pid)
                    progress_started=time.monotonic();progress_count=-1;progress_seen=progress_started
                    while process.poll() is None:
                        if workload_policy:
                            beacon=root/'operation-progress.json'
                            if beacon.exists():
                                raw=beacon.read_bytes()
                                if len(raw)>2048:raise ValueError('bounded operation progress')
                                progress=json.loads(raw)
                                if progress['phase']==name:
                                    if progress['pid']!=process.pid or type(progress['counter']) is not int or progress['counter']<progress_count:
                                        raise ValueError('operation progress identity/counter changed')
                                    stamp=progress['monotonic']
                                    if not isinstance(stamp,(int,float)) or not math.isfinite(stamp) or not progress_started-5.<=stamp<=time.monotonic():
                                        raise ValueError('operation progress clock')
                                    if progress['counter']>progress_count:progress_count=progress['counter'];progress_seen=stamp
                            if time.monotonic()-progress_seen>STALL_SECONDS:raise RuntimeError('no completed operation for 1800s; preserve checkpoint and stop')
                        record('sample')
                        if ledger.violations():raise RuntimeError('finite resource reserve reached')
                        time.sleep(.5)
                    record('segment_exit',name=name,returncode=process.returncode)
                    if meter.members()!={os.getpid()}:raise RuntimeError('segment exited with live descendants')
                    ledger.end(success=process.returncode==0)
                    if process.returncode or ledger.violations():raise RuntimeError('segment failed or resource reserve reached')
                process=None
            record('completion_boundary')
            if ledger.violations():raise RuntimeError('resource reserve reached at completion boundary')
            result=dict(status='STAGES_COMPLETE_RESOURCE_RECONCILIATION_PENDING',resources=ledger.snapshot())
        except BaseException as exc:
            cleanup=None
            try:cleanup=terminate() or None
            except BaseException as error:cleanup=[repr(error)]
            if process is not None:
                try:process.wait(timeout=5)
                except BaseException as error:cleanup=(cleanup or [])+[repr(error)]
            try:record('incomplete',enforce=False,error=repr(exc),cleanup_error=cleanup)
            except BaseException as error:cleanup=(cleanup or [])+['accounting: '+repr(error)]
            try:
                closed=meter.members()=={os.getpid()}
                # Last accounting is independent of journal health; preserve
                # readable CPU even if storage enumeration is the failed part.
                try:disk=allocated_bytes(root,meter.members()) if storage_meter is None else storage_meter()
                except BaseException as error:
                    disk=ledger.disk;cleanup=(cleanup or [])+['storage meter: '+repr(error)]
                cpu=meter.cpu_seconds() if cpu_cut_binding is None else save_cpu_cut()
                ledger.sample(cpu_seconds=setup_cpu_seconds+cpu,disk_bytes=disk)
            except BaseException as error:
                closed=False;cleanup=(cleanup or [])+['final meter: '+repr(error)]
            if ledger.active and closed:ledger.end(success=False)
            result=dict(status='INCOMPLETE',error=repr(exc),cleanup_error=cleanup,resources=ledger.snapshot())
            result.update(accounting_complete=False,external_service_reconciliation_required=True,
                descendants_closed=closed,event_journal=journal.receipt())
            _json_atomic(root/'resource-result.json',result)
            raise
        result.update(accounting_complete=False,external_service_reconciliation_required=True,
            descendants_closed=True,event_journal=journal.receipt())
        _json_atomic(root/'resource-result.json',result)
        return result
    finally:
        journal.close()
