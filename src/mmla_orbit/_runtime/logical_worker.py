"""Teacher-free worker for a validated derived logical world.

Retains the frozen native SAS guard, signed-state adapter, complete hidden-state
digest and action semantics. No planner, expert or old trajectory is loaded.
"""
from dataclasses import asdict
import contextlib,hashlib,json,os,random,resource,sys
from pathlib import Path
from mmla_pdsa.full.alfworld_episode import EpisodeLimits,_digest
from mmla_pdsa.full.alfworld_worker import Worker,verify_runtime
from logical_episode import LogicalWorldPermit


class LogicalWorker(Worker):
    def __init__(self,request):
        if request.get('purpose')!='validated-logical-execution':
            raise ValueError('only teacher-free validated-world execution')
        self.permit=LogicalWorldPermit(**request['permit']);self.limits=EpisodeLimits(**request['limits'])
        if self.limits.max_steps!=64 or self.limits.expose_admissible_commands:
            raise ValueError('fixed public-information and horizon contract required')
        cpu=request['cpu_seconds']
        if type(cpu) is not int or not 1<=cpu<=600:raise ValueError('finite worker CPU cap required')
        resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
        resource.setrlimit(resource.RLIMIT_FSIZE,(33554432,33554432))
        resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
        self.manifest=verify_runtime(request['dependency_manifest'],request['dependency_manifest_sha256'])
        self.identity,self.seed=request['identity'],request['seed']
        self._steps,self._started,self._teacher=0,False,None
        self._label,self._teacher_status,self._goal=None,'ABSENT',None
        game,provenance=self.permit.verify_files(request['data_root'])
        self._teacher_rng=random.Random(int(_digest(['unused-teacher-rng-v1',self.seed]),16)).getstate()
        random.seed(self.seed)
        import numpy as np
        np.random.seed(self.seed)
        import textworld
        from mmla_pdsa.full.runtime_attestation import retain_downward_library,attest
        retain_downward_library(self.manifest)
        from textworld.envs.pddl.pddl import PddlEnv
        from textworld.envs.pddl import logic as pddl_logic
        from mmla_pdsa.full.textworld_state import install_native_state_adapter
        install_native_state_adapter(pddl_logic)
        from alfworld.agents.utils.misc import Demangler
        import fast_downward
        from mmla_pdsa.full.native_sas_guard import GuardedNative
        self._env=PddlEnv(textworld.EnvInfos(won=True,admissible_commands=True,facts=False,policy_commands=False))
        self._env.downward_lib=GuardedNative(self._env.downward_lib,fast_downward.Atom,fast_downward.Operator)
        # Decode exactly the bytes checked again here; the admitted root must
        # remain immutable throughout the episode and original replay.
        raw=game.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=self.permit.game_sha256:raise ValueError('derived game changed')
        self._env.load(json.loads(raw))
        demangler=Demangler(game_infos=self._env._entity_infos,shuffle=False)
        for info in self._env._entity_infos.values():info.name=demangler.demangle_alfred_name(info.id)
        self._static=None;self.runtime_attestation=attest(self.manifest)

    def _bind_program(self):
        self._static=_digest(dict(schema='orbit-derived-logical-native-v1',permit=self.permit.identity,
            game=self.permit.game_sha256,provenance=self.permit.provenance_sha256,runtime=self.identity,
            native_guard='orbit-native-sas3-ascii-bounds-v1',fact_adapter='native-signed-state-v1',
            sas=self._env._pddl_state.sas,sas_replan=self._env._pddl_state.sas_replan,
            entity_infos={k:v.serialize() for k,v in self._env._entity_infos.items()}))

    def _update_teacher(self,last_action,*,initial=False):
        if self._teacher is not None:raise RuntimeError('a derived actor cannot acquire a teacher')
        self._label,self._teacher_status=None,'ABSENT'


def main():
    output=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1,encoding='utf-8')
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());sys.stdout=sys.stderr;worker=None
    while True:
        line=sys.stdin.buffer.readline(65537)
        if not line:return
        ok=False;result={}
        try:
            if len(line)>65536 or not line.endswith(b'\n'):raise ValueError('bounded complete RPC line required')
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                op=request['operation']
                if op=='initialize' and worker is None:
                    worker=LogicalWorker(request);result=dict(runtime_attestation=worker.runtime_attestation)
                elif op=='reset' and worker is not None:result=worker.reset()
                elif op=='step' and worker is not None:result=worker.step(request['action'])
                else:raise ValueError('invalid teacher-free backend operation/order')
            ok=True
        except Exception as exc:result=dict(error=type(exc).__name__+': '+str(exc))
        usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(ok=ok,identity=worker.identity if worker else None,
            cpu_seconds=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime)
        output.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');output.flush()
        if not ok:return


if __name__=='__main__':main()
