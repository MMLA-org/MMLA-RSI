"""Finite Q0|(R and V) blocks, with whole-draw rejection and collision stop.

Production calls belong to the isolated support service, never a proposal role.
Model scores do not enter this sampler. The public actor only gets READY game
and provenance files; source goals, expert plans and rejection logs stay private.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import secrets

from mmla_pdsa.full.alfworld_episode import TaskPermit, load_permits, _sha
from placement_domain import compile_domain, instantiate, digest
from semantic_exclusion import semantic_fingerprint, SCHEMA as FINGERPRINT_SCHEMA
from logical_episode import LogicalWorldPermit
from validation_common import ValidationCandidate

HISTORY_COUNTS = {'FIT': 1969, 'DEV': 290, 'C_J0': 278, 'C_J1': 248, 'C_J2': 228,
                  'A_RSI1': 263, 'A_RSI2': 277, 'FINAL_OFFICIAL': 134, 'QUARANTINE': 340}
VALIDATOR = 'pinned-handcoded-tw-seed0-horizon64-fresh-teacher-free-replay-v1'


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = encode(value)
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SourceEntry:
    permit: TaskPermit
    domain_sha256: str


@dataclass(frozen=True)
class SourcePool:
    entries: tuple[SourceEntry, ...]
    catalog_sha256: str
    partition_sha256: str
    source_root: str

    @property
    def population_identity(self):
        return digest(dict(schema='orbit-conditional-logical-population-v1',
            catalog=self.catalog_sha256, partition=self.partition_sha256,
            sources=[(e.permit.identity, e.domain_sha256) for e in self.entries],
            draw='uniform-template-then-independent-uniform-destination-for-every-variable',
            condition='R:task-relevant-placement-changes;V:fixed-native-certificate-replayed',
            validator=VALIDATOR, max_attempts_per_world=32, horizon=64,
            collisions='STOP_WHOLE_CANDIDATE_NO_REPLACEMENT'))


def load_pool(*, catalog_path, catalog_sha256, partition_path, partition_sha256, source_root):
    raw = Path(catalog_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(catalog_sha256):
        raise ValueError('fixed complete FIT catalog changed')
    catalog = json.loads(raw)
    permits = load_permits(partition_path, expected_sha256=partition_sha256)
    fit = {k: v for k, v in permits.items() if v.role == 'FIT'}
    rows = catalog['rows']
    if (catalog['schema'] != 'orbit-complete-fit-structural-catalog-v1'
            or catalog['partition_sha256'] != partition_sha256
            or catalog['catalog_identity'] != digest(rows)
            or len(rows) != 1969 or len(fit) != 1969
            or len({e['task_id'] for e in rows}) != 1969
            or {e['task_id'] for e in rows} != set(fit)):
        raise ValueError('whole fixed FIT inventory required')
    good = []
    for row in rows:
        permit = fit[row['task_id']]
        if (row['source'] != permit.game_sha256 or row['path'] != permit.game_path
                or row['floor'] != permit.floor):
            raise ValueError('FIT catalog identity does not match its original partition')
        if row['status'] == 'STRUCTURAL_DOMAIN_ONLY':
            good.append(SourceEntry(permit, _sha(row['domain'])))
        elif (row['status'] != 'UNSUPPORTED_STRUCTURE'
              or row.get('reason') != 'no task-relevant object has multiple legal initial placements'):
            raise ValueError('undeclared source exclusion')
    if len(good) != 1919:
        raise ValueError('fixed1919 structural templates required')
    # The catalog order is part of Q0; do not reweight by family or success.
    return SourcePool(tuple(good), catalog_sha256, partition_sha256,
                      str(Path(source_root).resolve(strict=True)))


def check_history(value, *, expected_sha256, partition_sha256):
    if hashlib.sha256(encode(value)).hexdigest() != expected_sha256:
        raise ValueError('complete isolated exclusion inventory changed')
    if (value['schema'] != 'orbit-historical-logical-exclusions-v1'
            or value['fingerprint_schema'] != FINGERPRINT_SCHEMA
            or value['partition_sha256'] != partition_sha256
            or value['counts'] != HISTORY_COUNTS
            or value['without_executable_game'] != {'EXCLUDED_UNSUPPORTED': 3053}
            or value['raw_goals_or_plans_exported'] is not False
            or not value['fingerprints']
            or value['fingerprints'] != sorted(set(value['fingerprints']))):
        raise ValueError('full original inventory coverage required before new support')
    return {_sha(x) for x in value['fingerprints']}


class SupportExhausted(RuntimeError):
    pass


class SemanticCollision(RuntimeError):
    pass


def _verify_certificate(value, candidate):
    if (type(value) is not dict or value.get('schema') != 'orbit-private-validation-certificate-v1'
            or value.get('candidate') != candidate.identity or value.get('game_sha256') != candidate.game_sha256
            or value.get('seed') != 0 or value.get('horizon') != 64
            or value.get('operational_failures_are_rejections') is not False
            or type(value.get('valid')) is not bool):
        raise ValueError('fixed private certificate result required')
    if value['valid']:
        if (value.get('status') != 'NATIVE_CERTIFICATE_REPLAYED' or value.get('search_reason') != 'WON'
                or value.get('plan_replay_won') is not True or not 1 <= len(value['plan']) <= 64
                or len(value['search_trace']) != len(value['plan'])+1
                or len(value['replay_trace']) != len(value['plan'])+1):
            raise ValueError('complete search and fresh replay certificate required')
        for key in ('search_attestation', 'replay_attestation'):
            if value[key].get('schema') != 'orbit-worker-runtime-attestation-v1':
                raise ValueError('native validation attestation missing')
    elif (value.get('status') != 'NO_CERTIFICATE' or value.get('plan_replay_won') is not False
          or value.get('search_reason') not in {'INITIAL_TERMINAL', 'LOST', 'NO_CERTIFICATE_WITHIN_64',
              'NO_ADMISSIBLE_LABEL', 'HandCodedAgentFailed', 'HandCodedAgentTimeout'}):
        raise ValueError('undeclared finite validation rejection')


def generate_block(*, pool, role, n, history, history_sha256, previous_fingerprints,
                   candidate_lock_sha256, output, validator, on_event,
                   draw_below=secrets.randbelow):
    """All n worlds or a retained INCOMPLETE block; never replace collisions.

    Each draw uses fresh system random bits in production. Exact choices are
    retained for deterministic reconstruction; reader seeds are unrelated.
    A final caller must enforce role visibility and bind the candidate lock
    before invoking either audit generation. Python objects are not a sandbox.
    """
    if (type(pool) is not SourcePool or not pool.entries or role not in ('DEV', 'A_RSI1', 'A_RSI2')
            or type(n) is not int or n != (32 if role == 'DEV' else 1024)
            or not callable(validator) or not callable(on_event) or not callable(draw_below)
            or type(previous_fingerprints) is not frozenset):
        raise ValueError('fixed population, block size and private callbacks required')
    if role == 'DEV':
        if candidate_lock_sha256 is not None:
            raise ValueError('development has no post-selection candidate lock')
    else:
        _sha(candidate_lock_sha256)
    previous = {_sha(x) for x in previous_fingerprints}
    excluded = check_history(history, expected_sha256=history_sha256, partition_sha256=pool.partition_sha256)
    root = Path(output).absolute(); root.mkdir(parents=True, exist_ok=False)
    public = root/'public'; private = root/'private'; public.mkdir(); private.mkdir()
    initial = dict(schema='orbit-logical-block-request-v1', role=role, n=n,
        population_sha256=pool.population_identity, history_sha256=history_sha256,
        previous_fingerprints_sha256=digest(sorted(previous)), candidate_lock_sha256=candidate_lock_sha256,
        max_attempts_per_world=32, validator=VALIDATOR, randomness='fresh-system-bits; full draws recorded')
    save(private/'request.json', initial)
    accepted = []; seen = set(); attempts = 0; r_rejected = 0; v_rejected = 0
    try:
        for index in range(n):
            for attempt in range(32):
                source_index = draw_below(len(pool.entries))
                if type(source_index) is not int or not 0 <= source_index < len(pool.entries):
                    raise ValueError('invalid complete-template draw')
                entry = pool.entries[source_index]
                game_path, _ = entry.permit.verify_files(pool.source_root)
                domain = compile_domain(game_path.read_bytes(), expected_sha256=entry.permit.game_sha256)
                if domain.identity != entry.domain_sha256:
                    raise ValueError('source structural domain changed')
                choices = tuple(draw_below(len(v.destinations)) for v in domain.variables)
                if any(type(c) is not int or not 0 <= c < len(v.destinations)
                       for c, v in zip(choices, domain.variables)):
                    raise ValueError('invalid full placement vector')
                record = dict(index=index, attempt=attempt, source_index=source_index,
                    source_permit=asdict(entry.permit), choices=choices, status='DRAWN')
                slot = private/f'draw-{index:04d}-{attempt:02d}'; slot.mkdir()
                save(slot/'draw.json', record); attempts += 1
                on_event(dict(kind='support_draw', role=role, index=index, attempt=attempt))
                try:
                    game, _ = instantiate(domain, choices)
                except ValueError as error:
                    if str(error) != 'a task-relevant initial placement must actually change':
                        raise
                    save(slot/'result.json', dict(status='R_FALSE', reason=str(error)))
                    r_rejected += 1; continue
                game_sha = save(slot/'game.json', game)
                candidate = ValidationCandidate(entry.permit, choices,
                    str((slot/'game.json').relative_to(root)), game_sha)
                certificate = validator(candidate, root)
                _verify_certificate(certificate, candidate)
                validation_sha = save(slot/'certificate.json', certificate)
                if not certificate['valid']:
                    save(slot/'result.json', dict(status='V_FALSE', certificate_sha256=validation_sha))
                    v_rejected += 1; continue
                semantic = semantic_fingerprint(game)
                if semantic in excluded or semantic in previous or semantic in seen:
                    save(slot/'result.json', dict(status='SEMANTIC_COLLISION_STOP', semantic_sha256=semantic))
                    raise SemanticCollision('validated world duplicates historical or already generated logical content')
                seen.add(semantic)
                provenance = dict(schema='orbit-logical-world-provenance-v1',
                    source_game_sha256=entry.permit.game_sha256, game_sha256=game_sha,
                    semantic_sha256=semantic, validation_sha256=validation_sha,
                    population_sha256=pool.population_identity, role=role, source_floor=entry.permit.floor,
                    horizon=64, native_validated=True, plan_replay_won=True, plan_steps=len(certificate['plan']))
                public_game = f'worlds/{game_sha}.json'; public_provenance = f'worlds/{game_sha}.provenance.json'
                if save(public/public_game, game) != game_sha:
                    raise AssertionError('game re-encoding changed')
                provenance_sha = save(public/public_provenance, provenance)
                accepted.append(dict(task_id='logical-'+game_sha, floor=entry.permit.floor, role=role,
                    game_path=public_game, game_sha256=game_sha, provenance_path=public_provenance,
                    provenance_sha256=provenance_sha, source_game_sha256=entry.permit.game_sha256,
                    semantic_sha256=semantic, validation_sha256=validation_sha,
                    population_sha256=pool.population_identity))
                save(slot/'result.json', dict(status='CERTIFIED_NEW_WORLD', semantic_sha256=semantic))
                break
            else:
                raise SupportExhausted('no R-and-V world within32 full draws; no partial-block admission')
        block_sha = digest(dict(request=initial, worlds=accepted))
        permits = tuple(LogicalWorldPermit(**row, block_sha256=block_sha) for row in accepted)
        for permit in permits:
            permit.verify_files(public)
        receipt = dict(schema='orbit-logical-ready-block-v1', status='READY', request=initial,
            block_sha256=block_sha, permits=[asdict(p) for p in permits],
            draw_count=attempts, R_rejections=r_rejected, V_rejections=v_rejected,
            native_validation=True, teacher_or_plan_exported=False, audit_admission_performed=False)
        save(public/'READY.json', receipt)
        return receipt
    except BaseException as error:
        save(private/'INCOMPLETE.json', dict(status='INCOMPLETE', error=type(error).__name__+': '+str(error),
            completed_worlds=len(accepted), draw_count=attempts, R_rejections=r_rejected,
            V_rejections=v_rejected, automatic_retry=False, admission_performed=False))
        raise
