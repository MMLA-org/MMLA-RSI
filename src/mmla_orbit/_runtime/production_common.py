"""Small, data-only helpers for the finite fixed-Phi production entry."""
from pathlib import Path
import hashlib
import json
import os
import secrets


def encode(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(encode(value).rstrip(b'\n')).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def sha_identity(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('lowercase SHA256 identity required')
    return value


def read(path,*,expected=None,cap=8*1024**2):
    p=Path(path)
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=cap:
        raise ValueError('bounded regular JSON required')
    raw=p.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest()!=sha_identity(expected):
        raise ValueError('JSON identity changed')
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def new(path,value):
    p=Path(path);raw=encode(value)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def atomic(path,value):
    p=Path(path);tmp=p.with_name('.'+p.name+'.'+secrets.token_hex(8))
    new(tmp,value);os.replace(tmp,p)
    fd=os.open(p.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify_source(code,expected):
    code=Path(code);manifest=read(code/'source-manifest.json')
    if digest(manifest)!=sha_identity(expected):raise ValueError('exact production source identity required')
    review=read(code/'production-source-review.json')
    if review.get('decision')!='PASS' or review.get('candidate_sha')!=expected or review.get('role')!='reviewer':
        raise PermissionError('complete independent production review required')
    for name,spec in manifest['files'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('confined source entry required')
        p=code/rel
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=spec['bytes'] or sha(p)!=spec['sha256']:
            raise ValueError('frozen production source changed')
    return manifest


class EventLog:
    """Finite-size shards retain every event without one unbounded log file."""
    def __init__(self,root,*,shard_bytes=16*1024**2):
        self.root=Path(root);self.root.mkdir(exist_ok=False)
        if type(shard_bytes) is not int or not 1024<=shard_bytes<=64*1024**2:
            raise ValueError('bounded event shard size')
        self.bound=shard_bytes;self.size=0;self.count=0;self.rows=[];self.file=None
        self.hasher=None;self.closed=False
    def _seal(self):
        if self.file is None:return
        self.flush();self.file.close()
        self.rows.append(dict(path=Path(self.file.name).name,bytes=self.size,sha256=self.hasher.hexdigest()))
        self.file=None
    def write(self,line):
        if self.closed:raise ValueError('event log already closed')
        raw=line.encode() if type(line) is str else line
        if type(raw) is not bytes or not raw.endswith(b'\n') or len(raw)>self.bound:
            raise ValueError('bounded complete event line required')
        if self.file is not None and self.size+len(raw)>self.bound:self._seal()
        if self.file is None:
            self.file=(self.root/f'{len(self.rows):06d}.jsonl').open('xb')
            self.hasher=hashlib.sha256();self.size=0
        self.file.write(raw);self.hasher.update(raw);self.size+=len(raw);self.count+=1
    def flush(self):
        if self.file is not None:self.file.flush();os.fsync(self.file.fileno())
    def close(self):
        if self.closed:return
        self._seal();new(self.root/'manifest.json',dict(schema='orbit-production-event-shards-v1',
            events=self.count,shards=self.rows,shard_byte_limit=self.bound))
        self.closed=True

