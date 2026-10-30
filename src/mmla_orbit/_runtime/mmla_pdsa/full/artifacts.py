"""No-overwrite, bounded, portable stage artifacts containing plain data only.

These are trusted run outputs checked by byte identity, not an authentication
format for untrusted uploads. Loading uses torch weights_only and never imports
or instantiates a type named by stored audit metadata.
"""
from dataclasses import is_dataclass,fields
import hashlib,json,os,shutil,tempfile
from pathlib import Path
import torch
from .profile import wire,digest
from .q_features import ActionFeatures

MAX_ARTIFACT_BYTES=256*1024*1024


def plain_data(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if is_dataclass(value) and not isinstance(value,type):
        return {'__mmla_audit_type__':type(value).__module__+'.'+type(value).__qualname__,
                'fields':{f.name:plain_data(getattr(value,f.name)) for f in fields(value)}}
    if type(value) is dict:
        if any(type(k) not in (str,int) for k in value):raise ValueError('plain artifact dictionary keys required')
        return {k:plain_data(v) for k,v in value.items()}
    if type(value) in (list,tuple):return type(value)(plain_data(v) for v in value)
    if value is None or type(value) in (str,bytes,bool,int,float):return value
    raise TypeError('unsupported artifact value type: '+type(value).__name__)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def save_artifact(path,payload,*,kind):
    if type(kind) is not str or not 0<len(kind)<=128:raise ValueError('bounded explicit artifact kind')
    dest=Path(path)
    if dest.exists():raise FileExistsError('immutable stage artifact already exists')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
    try:
        body=tmp/'payload.pt';torch.save(plain_data(payload),body)
        if not 0<body.stat().st_size<=MAX_ARTIFACT_BYTES:raise ValueError('bounded stage payload exceeded')
        with body.open('rb') as f:os.fsync(f.fileno())
        manifest=dict(schema='mmla-stage-artifact-v1',kind=kind,
            payload=dict(name='payload.pt',bytes=body.stat().st_size,sha256=file_hash(body)))
        (tmp/'manifest.json').write_bytes(wire(manifest)+b'\n')
        with (tmp/'manifest.json').open('rb') as f:os.fsync(f.fileno())
        _sync_directory(tmp)
        if dest.exists():raise FileExistsError('immutable artifact publish collision')
        os.rename(tmp,dest);tmp=None;_sync_directory(dest.parent)
        return dict(manifest=manifest,identity=digest(manifest))
    finally:
        if tmp is not None:shutil.rmtree(tmp,ignore_errors=True)


def load_artifact(path,*,kind,identity):
    root=Path(path)
    manifest_path=root/'manifest.json'
    if manifest_path.stat().st_size>16*1024:raise ValueError('bounded artifact manifest required')
    manifest=json.loads(manifest_path.read_bytes())
    if (set(manifest)!={'schema','kind','payload'} or manifest['schema']!='mmla-stage-artifact-v1'
            or manifest['kind']!=kind or digest(manifest)!=identity):
        raise ValueError('exact stage artifact identity/kind required')
    spec=manifest['payload']
    if (set(spec)!={'name','bytes','sha256'} or spec['name']!='payload.pt'
            or type(spec['bytes']) is not int or not 0<spec['bytes']<=MAX_ARTIFACT_BYTES):
        raise ValueError('bounded original payload only')
    body=root/'payload.pt'
    if body.stat().st_size!=spec['bytes'] or file_hash(body)!=spec['sha256']:
        raise ValueError('stage artifact byte identity changed')
    # plain_data admits bytes, including empty authoritative-memory fields.
    # Torch's restricted unpickler needs this built-in constructor explicitly;
    # keep weights_only and scope the allowance to this verified artifact read.
    added=[] if bytes in torch.serialization.get_safe_globals() else [bytes]
    with torch.serialization.safe_globals(added):
        return torch.load(body,map_location='cpu',weights_only=True)


def q_rows_from_plain(rows):
    """Restore only the known Q feature carrier; other typed objects stay audit data."""
    result=[]
    for source in rows:
        row=dict(source);item=row['features']
        if (type(item) is not dict or set(item)!={'__mmla_audit_type__','fields'}
                or item['__mmla_audit_type__']!='mmla_pdsa.full.q_features.ActionFeatures'
                or set(item['fields'])!={'matrix','actions','binding'}):
            raise ValueError('exact stored ActionFeatures required')
        values=item['fields']
        row['features']=ActionFeatures(tuple(tuple(x) for x in values['matrix']),tuple(values['actions']),values['binding'])
        result.append(row)
    return result
