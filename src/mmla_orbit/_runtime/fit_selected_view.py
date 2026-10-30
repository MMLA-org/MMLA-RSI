"""Materialize only the frozen 56 FIT tasks for the label-generation role.

The trusted native host owns the source tree. Training and inference actors
never receive that whole tree or this role's game/trajectory payloads.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import os
from curriculum import select_fit, digest
from training_state import _write, _json, _decode, sha
from fit_evaluation import PARTITION, check_fit_permits
from mmla_pdsa.full.alfworld_episode import TaskPermit

SELECTION='4674c5f60c95d8de2c21124ad58eff80d7f8aa7ac60ae57659df76733b448590'
PARTITION_BYTES=7137031

def permits_from_metadata(metadata,selection):
    if selection!=select_fit(metadata) or selection['identity']!=SELECTION:
        raise ValueError('exact original deterministic 56-task selection required')
    indexed={r['task_id']:r for r in metadata}
    permits=[]
    for g in selection['groups']:
        for side in ('support','query'):
            for selected in g[side]:
                r=indexed[selected['task_id']]
                if r['exclusions']:raise ValueError('excluded source may not become a FIT permit')
                permits.append(TaskPermit(r['task_id'],r['floor_plan'],'FIT',r['game_path'],
                    r['game_sha256'],r['source_path'],r['trajectory_sha256'],PARTITION))
    permits=tuple(permits);check_fit_permits(selection,permits)
    return permits

def _plain(root,relative):
    p=root/relative
    for part in (p,*p.parents):
        if part==root:break
        if part.is_symlink():raise ValueError('selected source aliases are not allowed')
    if not p.is_file() or not p.resolve(strict=True).is_relative_to(root):
        raise ValueError('regular source inside registered root required')
    return p

def build_view(metadata_path,source_root,destination,*,source_identity):
    """Host-only preparation inside the admitted scientific run; no execution."""
    sha(source_identity)
    p=Path(metadata_path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size!=PARTITION_BYTES:
        raise ValueError('exact original partition metadata required')
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARTITION:raise ValueError('partition changed')
    metadata=[_decode(line) for line in raw.splitlines()]
    selection=select_fit(metadata);permits=permits_from_metadata(metadata,selection)
    source=Path(source_root).resolve(strict=True)
    if not source.is_dir():raise ValueError('registered source root required')
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    files=[]
    for permit in permits:
        if (Path(permit.game_path).parent!=Path(permit.trajectory_path).parent
                or Path(permit.trajectory_path).name!='traj_data.json'):
            raise ValueError('original same-directory game/trajectory pair required')
        for name,expected in ((permit.game_path,permit.game_sha256),
                              (permit.trajectory_path,permit.trajectory_sha256)):
            original=_plain(source,name)
            if not 0<original.stat().st_size<=32*1024**2:raise ValueError('finite selected file cap')
            data=original.read_bytes()
            if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('selected source bytes changed')
            target=dest/'data'/name;target.parent.mkdir(parents=True,exist_ok=True);_write(target,data)
            files.append(dict(path='data/'+name,bytes=len(data),sha256=expected))
    for name,value in (('selection.json',selection),('permits.json',[asdict(p) for p in permits])):
        data=_json(value);_write(dest/name,data)
        files.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='orbit-selected-fit-view-v1',source=source_identity,partition=PARTITION,
        selection=SELECTION,permit_identities=[p.identity for p in permits],files=files,
        tasks=56,raw_files=112,role='FIT',no_nonfit_payloads=True)
    _write(dest/'READY.json',_json(manifest))
    fd=os.open(dest,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(manifest_sha256=hashlib.sha256(_json(manifest)).hexdigest(),files=len(files),tasks=56)

def read_view(directory,*,source_identity,manifest_sha256):
    """Label role verifies the exact sealed view before opening an environment."""
    sha(source_identity);sha(manifest_sha256)
    root=Path(directory)
    if root.is_symlink() or not root.is_dir():raise ValueError('regular selected FIT root required')
    manifest_file=root/'READY.json'
    if manifest_file.is_symlink() or not manifest_file.is_file() or manifest_file.stat().st_size>262144:
        raise ValueError('bounded selected FIT seal required')
    raw=manifest_file.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('FIT view seal changed')
    meta=_decode(raw)
    if (set(meta)!={'schema','source','partition','selection','permit_identities','files','tasks',
            'raw_files','role','no_nonfit_payloads'} or meta['schema']!='orbit-selected-fit-view-v1'
            or meta['source']!=source_identity or meta['partition']!=PARTITION or meta['selection']!=SELECTION
            or meta['tasks']!=56 or meta['raw_files']!=112 or meta['role']!='FIT'
            or meta['no_nonfit_payloads'] is not True or len(meta['files'])!=114):
        raise ValueError('bound complete FIT view required')
    expected={r['path'] for r in meta['files']}|{'READY.json'};actual=set()
    for p in root.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):raise ValueError('regular FIT inventory required')
        if p.is_file():actual.add(str(p.relative_to(root)))
    if len(expected)!=115 or actual!=expected:raise ValueError('FIT view includes missing or unselected payload')
    payloads={}
    for row in meta['files']:
        if set(row)!={'path','bytes','sha256'}:raise ValueError('exact FIT file identity required')
        name=row['path'];p=Path(name)
        if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('canonical FIT file path required')
        path=_plain(root.resolve(),name)
        if type(row['bytes']) is not int or not 0<row['bytes']<=32*1024**2 or path.stat().st_size!=row['bytes']:
            raise ValueError('bounded exact FIT file size required')
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('FIT view payload changed')
        if name in ('selection.json','permits.json'):payloads[name]=_decode(data)
    selection=payloads['selection.json'];permits=tuple(TaskPermit(**x) for x in payloads['permits.json'])
    check_fit_permits(selection,permits)
    if selection['identity']!=SELECTION or [p.identity for p in permits]!=meta['permit_identities']:
        raise ValueError('selected permit identities changed')
    file_map={r['path']:r['sha256'] for r in meta['files']}
    wanted={'selection.json','permits.json'}
    for p in permits:
        for name,expected_hash in ((p.game_path,p.game_sha256),(p.trajectory_path,p.trajectory_sha256)):
            wanted.add('data/'+name)
            if file_map.get('data/'+name)!=expected_hash:raise ValueError('permit/file binding changed')
    if set(file_map)!=wanted:raise ValueError('only exact selected source files allowed')
    return selection,permits,root/'data'
