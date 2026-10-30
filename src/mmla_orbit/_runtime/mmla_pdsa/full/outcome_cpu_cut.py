"""Conservative, monotone final CPU accounting when systemd drops a counter.

The running cgroup counter is measured; its bounded shutdown tail is charged,
not claimed measured. All clocks below are from one boot's monotonic clock.
"""
import math,time
from pathlib import Path


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def _cpus(text):
    values=set()
    for part in text.strip().split(','):
        fields=part.split('-')
        if len(fields) not in (1,2) or any(not x.isdecimal() for x in fields):raise ValueError('effective CPU set syntax')
        lo=int(fields[0]);hi=int(fields[-1])
        if not 0<=lo<=hi<65536:raise ValueError('bounded effective CPU set')
        values.update(range(lo,hi+1))
    if not 1<=len(values)<=8:raise ValueError('actual service capacity must be at most8CPUs')
    return sorted(values)


def live_sample(meter):
    """Timestamp BEFORE the raw read, before any potentially slow disk scan."""
    meter._check()
    cpus=_cpus((meter.path/'cpuset.cpus.effective').read_text())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before=time.monotonic();cpu=meter.cpu_seconds();after=time.monotonic()
    return dict(service_CPU_seconds=cpu,monotonic_before=before,monotonic_after=after,
        boot_id=boot,cgroup_identity=list(meter.identity),effective_cpus=cpus)


def final_cpu_charge(*,sample,stop_monotonic,boot_id,prior_CPU_seconds,launcher_CPU_seconds,raw_final_service_CPU_seconds,capacity_cpus=None):
    """Return a charge upper bound, preserving raw counters and reset evidence.

Requires an externally verified fully stopped service and the same cpuset bound
through that stop. It never treats a smaller post-stop counter as the total.
"""
    before=sample['monotonic_before'];after=sample['monotonic_after'];live=sample['service_CPU_seconds'];cpus=sample['effective_cpus']
    if (not all(_finite(x) for x in (before,after,live,stop_monotonic,prior_CPU_seconds,launcher_CPU_seconds,raw_final_service_CPU_seconds))
            or before>after or after>stop_monotonic or boot_id!=sample['boot_id']
            or not boot_id or type(cpus) is not list or not 1<=len(cpus)<=8
            or any(type(x) is not int or x<0 for x in cpus) or len(set(cpus))!=len(cpus)):
        raise ValueError('same-boot bounded stopped-service CPU evidence required')
    capacity=len(cpus) if capacity_cpus is None else capacity_cpus
    if type(capacity) is not int or not len(cpus)<=capacity<=8:raise ValueError('registered CPU capacity must cover actual effective CPUs')
    tail=capacity*(stop_monotonic-before)
    service=max(raw_final_service_CPU_seconds,live+tail)
    return dict(mode='measured-prefix-plus-cpuset-bounded-shutdown-tail',exact_CPU_measurement=False,
        measured_service_prefix_CPU_seconds=live,raw_final_service_CPU_seconds=raw_final_service_CPU_seconds,
        raw_final_counter_regressed=raw_final_service_CPU_seconds<live,
        shutdown_tail_CPU_charge=tail,service_CPU_charge=service,prior_CPU_seconds=prior_CPU_seconds,
        launcher_CPU_seconds=launcher_CPU_seconds,total_CPU_charge=prior_CPU_seconds+launcher_CPU_seconds+service,
        sample_monotonic_before=before,stop_monotonic=stop_monotonic,capacity_cpus=capacity,sampled_effective_cpus=list(cpus),boot_id=boot_id)
