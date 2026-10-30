"""Bind a successful support-service return to an atomic public READY seal.

The private producer calls seal_generated_block only after generate_block has
returned successfully. A model process receives only the sealed public tree.
These files establish content identity; the native host still enforces producer
identity and visibility. No caller-supplied schema replaces that boundary.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import secrets

from logical_episode import LogicalWorldPermit
from placement_domain import digest
from support_generation import encode, VALIDATOR
from mmla_pdsa.full.alfworld_episode import _sha


def atomic_new(path, value):
    path = Path(path)
    raw = encode(value)
    temp = path.with_name('.'+path.name+'.'+secrets.token_hex(8)+'.tmp')
    with temp.open('xb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    # link is atomic and refuses to replace a previous publication.
    os.link(temp, path); temp.unlink()
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReadyBlock:
    public_root: str
    ready_sha256: str
    seal_sha256: str
    producer_sha256: str
    request_sha256: str
    block_sha256: str
    population_sha256: str
    role: str
    permits: tuple
    candidate_lock_sha256: str | None


def _inspect(public_root, *, expected_request):
    root = Path(public_root).resolve(strict=True)
    ready_path = root/'READY.json'
    if ready_path.is_symlink() or not 0 < ready_path.stat().st_size <= 8*1024**2:
        raise ValueError('bounded regular READY file required')
    raw = ready_path.read_bytes(); value = json.loads(raw)
    request = value['request']; role = request['role']; n = request['n']
    if (value['schema'] != 'orbit-logical-ready-block-v1' or value['status'] != 'READY'
            or request != expected_request or request['schema'] != 'orbit-logical-block-request-v1'
            or role not in ('DEV', 'A_RSI1', 'A_RSI2')
            or type(n) is not int or n != (32 if role == 'DEV' else 1024)
            or request['max_attempts_per_world'] != 32 or request['validator'] != VALIDATOR
            or request['randomness'] != 'fresh-system-bits; full draws recorded'
            or value['native_validation'] is not True or value['teacher_or_plan_exported'] is not False
            or value['audit_admission_performed'] is not False):
        raise ValueError('complete fixed support request and production barrier required')
    for key in ('population_sha256', 'history_sha256', 'previous_fingerprints_sha256'):
        _sha(request[key])
    if role == 'DEV':
        if request['candidate_lock_sha256'] is not None: raise ValueError('DEV cannot carry an audit lock')
    else: _sha(request['candidate_lock_sha256'])
    counts = [value[k] for k in ('draw_count', 'R_rejections', 'V_rejections')]
    if (any(type(x) is not int or x < 0 for x in counts)
            or counts[0] != n+counts[1]+counts[2] or not n <= counts[0] <= 32*n):
        raise ValueError('all draw outcomes must reconcile')
    permits = tuple(LogicalWorldPermit(**p) for p in value['permits'])
    if (len(permits) != n or len({p.identity for p in permits}) != n
            or len({p.semantic_sha256 for p in permits}) != n):
        raise ValueError('full unique support block required')
    world_rows = []
    allowed = {'READY.json', 'SEALED.json'}
    for p in permits:
        if (p.role != role or p.block_sha256 != value['block_sha256']
                or p.population_sha256 != request['population_sha256']):
            raise ValueError('world and block identity mismatch')
        for name in (p.game_path, p.provenance_path):
            q = root/name
            if any(part.is_symlink() for part in [q, *q.parents] if part.is_relative_to(root)):
                raise ValueError('public support cannot contain symlinks')
            allowed.add(name)
        p.verify_files(root)
        row = asdict(p); row.pop('block_sha256'); world_rows.append(row)
    if digest(dict(request=request, worlds=world_rows)) != value['block_sha256']:
        raise ValueError('complete block identity changed')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != allowed-{'SEALED.json'} and actual != allowed:
        raise ValueError('public support contains unexpected or missing files')
    return value, permits, hashlib.sha256(raw).hexdigest()


def seal_generated_block(output, returned, *, producer_sha256):
    """Private producer only; a failed or partially returned call cannot seal."""
    _sha(producer_sha256); root = Path(output).resolve(strict=True)
    if (root/'private/INCOMPLETE.json').exists():
        raise ValueError('failed support call cannot publish a usable block')
    request = json.loads((root/'private/request.json').read_bytes())
    value, _, ready_sha = _inspect(root/'public', expected_request=request)
    if returned != value:
        raise ValueError('successful complete producer return must match READY bytes')
    seal = dict(schema='orbit-support-success-seal-v1', status='COMPLETE',
        producer_sha256=producer_sha256, ready_sha256=ready_sha,
        request_sha256=digest(request), block_sha256=value['block_sha256'],
        role=request['role'], full_generate_call_returned=True,
        failed_or_partial_calls_usable=False)
    return atomic_new(root/'public/SEALED.json', seal)


def load_ready(public_root, *, expected_request, expected_seal_sha256, producer_sha256):
    """Read only the exact block exposed by the admitted producer/host."""
    root = Path(public_root).resolve(strict=True)
    path = root/'SEALED.json'; _sha(expected_seal_sha256); _sha(producer_sha256)
    if path.is_symlink() or not 0 < path.stat().st_size < 8192:
        raise ValueError('bounded regular producer seal required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_seal_sha256:
        raise ValueError('support seal changed')
    value, permits, ready_sha = _inspect(root, expected_request=expected_request)
    expected = dict(schema='orbit-support-success-seal-v1', status='COMPLETE',
        producer_sha256=producer_sha256, ready_sha256=ready_sha,
        request_sha256=digest(expected_request), block_sha256=value['block_sha256'],
        role=expected_request['role'], full_generate_call_returned=True,
        failed_or_partial_calls_usable=False)
    if json.loads(raw) != expected:
        raise ValueError('support producer contract differs')
    return ReadyBlock(str(root), ready_sha, expected_seal_sha256, producer_sha256,
        digest(expected_request), value['block_sha256'], expected_request['population_sha256'],
        expected_request['role'], permits,expected_request['candidate_lock_sha256'])
