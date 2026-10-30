"""Work-count-bounded delivery: actual time accounting without hour windows.

This policy applies only to the separately reviewed work-count native entries. Old R1-R6
snapshots retain their original finite resource policy and outcomes.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,math,time
from .outcome_resources import ResourceLedger,PHASE_LIMITS

POLICY='orbit-finite-work-unlimited-local-time-v1'
DISK_BYTES=128*1024**3
STALL_SECONDS=1800.

@dataclass(frozen=True)
class WorkloadLimits:
    gpu:object=None
    cpu:object=None
    wall:object=None
    disk:int=DISK_BYTES
    reserve_seconds:float=0.
    reserve_bytes:int=512*1024**2
    reserve_cpu_seconds:float=0.

class WorkloadLedger(ResourceLedger):
    workload_bounded=True
    def __init__(self,run_id,started_unix,*,prior_preflight_gpu_seconds=0.,prior_phase_costs=None,clock=time.monotonic,wall_clock=time.time):
        # Reuse identifier and monotonic initialization validation without
        # passing the inherited costs through obsolete R6 phase allowances.
        super().__init__(run_id,started_unix,clock=clock,wall_clock=wall_clock)
        if type(prior_preflight_gpu_seconds) not in (float,int) or not math.isfinite(prior_preflight_gpu_seconds) or prior_preflight_gpu_seconds<0:
            raise ValueError('finite recorded prior GPU cost')
        self.limits=WorkloadLimits();self.gpu=float(prior_preflight_gpu_seconds)
        self.phases['preflight']['gpu']=self.gpu
        if prior_phase_costs is not None:
            if type(prior_phase_costs) is not dict or set(prior_phase_costs)!=set(self.phases):raise ValueError('complete inherited phase costs required')
            for value in prior_phase_costs.values():
                if type(value) is not dict or set(value)!={'cpu','gpu'} or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in value.values()):raise ValueError('finite nonnegative inherited phase cost')
            gpu=sum(value['gpu'] for value in prior_phase_costs.values())
            if not math.isclose(gpu,self.gpu,rel_tol=0.,abs_tol=1e-7):raise ValueError('inherited GPU total differs from phase sum')
            self.cpu=sum(value['cpu'] for value in prior_phase_costs.values())
            self.phases={k:dict(v) for k,v in prior_phase_costs.items()}
    def checkpoint(self):
        return dict(run_id=self.run_id,started_unix=self.started_unix,deadline_unix=None,wall_seconds=self.wall,
            CPU_seconds=self.cpu,GPU_seconds=self.gpu,events_count=self.events,events_sha256=self._hash.hexdigest())
    def violations(self,*,reserve=True):
        margin=self.limits.reserve_bytes if reserve else 0
        return ['disk'] if self.peak_disk>=self.limits.disk-margin else []
    def snapshot(self):
        return dict(super().snapshot(),time_policy=POLICY,progress_stall_seconds=STALL_SECONDS)
