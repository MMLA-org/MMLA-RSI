"""Pinned TextWorld PDDL worker. No third-party imports until initialize RPC."""
from dataclasses import asdict
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import sys

from .alfworld_episode import BackendFrame, EpisodeLimits, TaskPermit, _digest


def _plain(value):
    """Deterministic known state types; never use pointer-bearing repr/pickle."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) in (tuple, list):
        return [_plain(v) for v in value]
    if type(value) in (set, frozenset):
        return sorted((_plain(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise TypeError('state dictionaries need string keys')
        return {k: _plain(v) for k, v in value.items()}
    if type(value).__module__.startswith('alfworld.agents.expert.'):
        return dict(type=type(value).__name__, fields=_plain(vars(value)))
    raise TypeError('unregistered hidden-state type: '+type(value).__name__)


def _facts(facts):
    return sorted((f.name, tuple((a.name, a.type) for a in f.arguments)) for f in facts)


def _action(action):
    if action is None:
        return None
    return dict(name=action.name, id=action.id, command_template=action.command_template,
                feedback_rule=action.feedback_rule,
                mapping=sorted((p.name, p.type, v.name, v.type) for p, v in action.mapping.items()))


def verify_runtime(manifest_path, expected_sha):
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('runtime manifest identity mismatch')
    manifest = json.loads(raw)
    if (manifest['schema'] != 'orbit-textworld-runtime-v1'
            or manifest['textworld_version'] != '1.7.0'
            or manifest['alfworld_revision'] != 'aaba6870f86c5be6a08a491f32a50b906227bc3e'
            or importlib.metadata.version('textworld') != '1.7.0'
            or not manifest['files']):
        raise ValueError('pinned dependency/source identity required')
    for entry in manifest['files']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('runtime component changed: '+entry['path'])
    # Completeness of source/native manifest is part of independent run review.
    return manifest


class Worker:
    def __init__(self, request):
        self.permit = TaskPermit(**request['permit'])
        self.limits = EpisodeLimits(**request['limits'])
        cpu = request['cpu_seconds']
        if type(cpu) is not int or not 1 <= cpu <= 600:
            raise ValueError('finite worker CPU cap required')
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu+1))
        # The pinned native loader copies a 32,251,456-byte library per worker.
        # Keep a finite 32MiB file cap within the unchanged whole-run disk bound.
        resource.setrlimit(resource.RLIMIT_FSIZE, (33554432, 33554432))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        self.manifest = verify_runtime(request['dependency_manifest'], request['dependency_manifest_sha256'])
        self.identity, self.seed = request['identity'], request['seed']
        self._steps, self._started, self._teacher = 0, False, None
        self._label, self._teacher_status, self._goal = None, 'ABSENT', None
        game, trajectory = Path(request['game']), Path(request['trajectory'])
        if (hashlib.sha256(game.read_bytes()).hexdigest() != self.permit.game_sha256
                or hashlib.sha256(trajectory.read_bytes()).hexdigest() != self.permit.trajectory_sha256
                or game.parent != trajectory.parent or trajectory.name != 'traj_data.json'):
            raise ValueError('original paired game and trajectory required')
        self._teacher_rng = random.Random(int(_digest(['teacher-rng-v1',self.seed]),16)).getstate()
        random.seed(self.seed)
        import numpy as np
        np.random.seed(self.seed)
        import textworld
        from .runtime_attestation import retain_downward_library
        retain_downward_library(self.manifest)
        from textworld.envs.pddl.pddl import PddlEnv
        from textworld.envs.pddl import logic as pddl_logic
        from .textworld_state import install_native_state_adapter
        install_native_state_adapter(pddl_logic)
        from alfworld.agents.utils.misc import Demangler
        self._env = PddlEnv(textworld.EnvInfos(won=True, admissible_commands=True,
                                              facts=self.permit.teacher_allowed, policy_commands=False))
        import fast_downward
        from .native_sas_guard import GuardedNative
        self._env.downward_lib = GuardedNative(self._env.downward_lib, fast_downward.Atom, fast_downward.Operator)
        self._env.load(str(game))
        demangler = Demangler(game_infos=self._env._entity_infos, shuffle=False)
        for info in self._env._entity_infos.values():
            info.name = demangler.demangle_alfred_name(info.id)
        if self.permit.teacher_allowed:
            from alfworld.agents.expert.handcoded_expert_tw import HandCodedTWAgent
            with self._teacher_random():
                self._teacher = HandCodedTWAgent(max_steps=self.limits.max_steps)
                self._teacher.reset(str(game))
        self._static = None  # reset reconstructs the actual native program.
        from .runtime_attestation import attest
        self.runtime_attestation=attest(self.manifest)

    def _bind_program(self):
        self._static = _digest(dict(game=self.permit.game_sha256, trajectory=self.permit.trajectory_sha256,
            runtime=self.identity, native_guard="orbit-native-sas3-ascii-bounds-v1", fact_adapter="native-signed-state-v1", teacher_facts="positive-only-v1", sas=self._env._pddl_state.sas,
            sas_replan=self._env._pddl_state.sas_replan,
            entity_infos={k: v.serialize() for k, v in self._env._entity_infos.items()}))

    @contextlib.contextmanager
    def _teacher_random(self):
        environment_rng = random.getstate()
        random.setstate(self._teacher_rng)
        try:
            yield
        finally:
            self._teacher_rng = random.getstate()
            random.setstate(environment_rng)

    def _update_teacher(self, last_action, *, initial=False):
        with self._teacher_random():
            return self._update_teacher_private(last_action, initial=initial)

    def _update_teacher_private(self, last_action, *, initial=False):
        self._label = None
        if self._teacher is None:
            self._teacher_status = 'ABSENT'
            return
        if self._state['won']:
            self._teacher_status = 'TERMINAL'
            return
        from alfworld.agents.expert.handcoded_expert import HandCodedAgentFailed, HandCodedAgentTimeout
        try:
            if initial:
                self._teacher.observe(self._state['feedback'])
                # A declared bootstrap action, not an inferred optimal label.
                candidate = 'look'
                status = 'BOOTSTRAP_LOOK'
            else:
                # FIT label service only: upstream two-object policy uses
                # substring tests that confuse not_inreceptacle with inreceptacle.
                # Give it true positive facts, while grammar/native identity keep
                # the complete signed state. No fact reaches the online actor.
                teacher_state = dict(self._state)
                teacher_state['facts'] = [f for f in self._state['facts'] if not f.is_negation]
                candidate = self._teacher.act(teacher_state, 0, False, last_action)
                status = 'CURRENT_STATE_EXPERT'
            if type(candidate) is str and candidate in self._state['admissible_commands']:
                self._label, self._teacher_status = candidate, status
            else:
                self._teacher_status = 'NO_ADMISSIBLE_LABEL'
        except (HandCodedAgentFailed, HandCodedAgentTimeout) as exc:
            # Preserve episode and negative teacher outcome, never replace by look.
            self._teacher_status = type(exc).__name__

    def _state_identity(self):
        import numpy as np
        from textworld.envs.pddl.logic import Atom
        native = self._env.downward_lib
        atoms = (Atom * native.get_state_size())()
        native.get_state(atoms)
        state = self._state
        rng = np.random.get_state()
        entity_infos = {k: v.serialize() for k, v in self._env._entity_infos.items()}
        return _digest(dict(static=self._static, steps=self._steps, goal=self._goal,
            native_atoms=sorted(atom.name for atom in atoms),
            facts=_facts(self._env._pddl_state.facts), moves=self._env._moves,
            last_action=_action(self._env._last_action),
            valid_actions=[_action(a) for a in state['_valid_actions']],
            valid_commands=list(state['_valid_commands']), entity_infos=entity_infos,
            native_operators=sorted((int(k), op.name, int(op.nb_effect_atoms))
                                    for k, op in self._env._pddl_state._operators.items()),
            observation=state['feedback'], won=bool(state['won']), lost=bool(state['lost']),
            teacher=_plain(self._teacher), teacher_label=self._label, teacher_status=self._teacher_status,
            teacher_random_state=_plain(self._teacher_rng), random_state=_plain(random.getstate()), numpy_state=[rng[0], rng[1].tolist(), *rng[2:]]))

    def _reply(self):
        frame = BackendFrame(self._goal, self._state['feedback'],
            tuple(self._state['admissible_commands']), bool(self._state['won'] or self._state['lost']),
            bool(self._state['won']), self._state_identity())
        return dict(frame=asdict(frame), teacher_label=self._label, teacher_status=self._teacher_status)

    def reset(self):
        if self._started:
            raise ValueError('one reset per process')
        self._state = self._env.reset(); self._started = True
        self._bind_program()
        marker = 'Your task is to: '
        if self._state['feedback'].count(marker) != 1:
            raise ValueError('original reset lacks one public task description')
        self._goal = self._state['feedback'].split(marker, 1)[1].strip()
        if not self._goal or 'UNKNOWN GOAL' in self._goal:
            raise ValueError('original game has no bound public goal')
        self._update_teacher('', initial=True)
        return self._reply()

    def step(self, action):
        if not self._started or self._steps >= self.limits.max_steps or self._state['won'] or self._state['lost']:
            raise ValueError('no action after terminal or before reset')
        if type(action) is not str or len(action.encode()) > self.limits.max_action_bytes:
            raise ValueError('bounded action required')
        self._state, _, _ = self._env.step(action)
        self._steps += 1
        self._update_teacher(action)
        return self._reply()


def main():
    # Keep a private wire fd; native writes to fd 1 go to diagnostic stderr.
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1, encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    worker = None
    while True:
        line = sys.stdin.buffer.readline(65537)
        if not line:
            return
        ok, result = False, {}
        try:
            if len(line) > 65536 or not line.endswith(b'\n'):
                raise ValueError('bounded complete RPC line required')
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                op = request['operation']
                if op == 'initialize' and worker is None:
                    worker = Worker(request)
                    result = dict(runtime_attestation=worker.runtime_attestation)
                elif op == 'reset' and worker is not None:
                    result = worker.reset()
                elif op == 'step' and worker is not None:
                    result = worker.step(request['action'])
                else:
                    raise ValueError('invalid backend operation/order')
            ok = True
        except Exception as exc:
            result = dict(error=type(exc).__name__+': '+str(exc))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(ok=ok, identity=worker.identity if worker else None,
                      cpu_seconds=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime)
        output.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n'); output.flush()
        if not ok:
            return


if __name__ == '__main__':
    main()
