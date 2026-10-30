"""Exact-size JSON+safetensors cuts, with capacity reserved before execution.

Single-writer transactions reserve full per-cut envelopes. A production factory
must justify the configured snapshot envelope from its finite state bounds.
Filesystem faults preserve the incomplete publication and forbid replay.
"""
from contextlib import contextmanager
import fcntl,hashlib,json,os,tempfile
from pathlib import Path
import torch
from safetensors.torch import save as tensor_bytes,load as load_tensors


class StaleController(ValueError):pass
_UNBOUND=object()


class StorageFull(RuntimeError):pass
class SnapshotTooLarge(RuntimeError):pass
class JournalIntegrityError(RuntimeError):pass


def digest(data):return hashlib.sha256(data).hexdigest()


def serialize(state):
    tensors={}
    def encode(x):
        if torch.is_tensor(x):
            key='t'+str(len(tensors));tensors[key]=x.detach().cpu().contiguous().clone()
            return {'kind':'tensor','value':key}
        if type(x) in (tuple,list):return {'kind':type(x).__name__,'value':[encode(v) for v in x]}
        if type(x) is dict:
            if any(type(k) is not str for k in x):raise ValueError('string checkpoint keys')
            return {'kind':'dict','value':{k:encode(v) for k,v in x.items()}}
        if x is None or type(x) in (str,int,float,bool):return {'kind':'scalar','value':x}
        raise TypeError('unregistered checkpoint type')
    data={'state.json':json.dumps(encode(state),ensure_ascii=True,allow_nan=False,
                                  separators=(',',':')).encode(),
          'tensors.safetensors':tensor_bytes(tensors)}
    manifest={k:dict(sha256=digest(v),bytes=len(v)) for k,v in data.items()}
    data['manifest.json']=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    return data


class Journal:
    def __init__(self,root,*,max_bytes,max_cuts,max_cut_bytes=16*1024*1024):
        self.root=Path(root)
        if (type(max_bytes) is not int or max_bytes<1 or type(max_cuts) is not int
                or not 1<=max_cuts<=999999 or type(max_cut_bytes) is not int or max_cut_bytes<4096):
            raise ValueError('finite storage, snapshot and cut limits')
        self.max_bytes,self.max_cuts,self.max_cut_bytes=max_bytes,max_cuts,max_cut_bytes
        self.root.mkdir(parents=True,exist_ok=True);self._reservation=None

    def has_evidence(self):return any(p.name!='.writer.lock' for p in self.root.iterdir())

    def _inventory(self):
        if self.root.is_symlink() or any(p.is_symlink() for p in self.root.rglob('*')):
            raise JournalIntegrityError('journal symlinks are not accepted')
        cuts=sorted(self.root.glob('cut-*'))
        if any(p.name not in {'.writer.lock','LATEST'} and p not in cuts for p in self.root.iterdir()):
            raise JournalIntegrityError('interrupted or unknown journal publication')
        if [p.name for p in cuts]!=[f'cut-{i:06d}' for i in range(len(cuts))]:
            raise JournalIntegrityError('noncontiguous journal')
        if cuts:
            latest=self.root/'LATEST'
            if not latest.is_file() or latest.read_text()!=cuts[-1].name+'\n':
                raise JournalIntegrityError('stale pointer/orphan cut; no rollback resume')
        elif (self.root/'LATEST').exists():raise JournalIntegrityError('pointer without checkpoint')
        used=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        return cuts,used

    @contextmanager
    def transaction(self,cuts,*,expected_cut=_UNBOUND):
        """Reserve before work; no other journal writer can consume this space."""
        if self._reservation is not None:raise ValueError('no nested journal transaction')
        if type(cuts) is not int or cuts<1:raise ValueError('positive cut reservation')
        with (self.root/'.writer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            existing,used=self._inventory()
            current=existing[-1].name if existing else None
            if expected_cut is not _UNBOUND and current!=expected_cut:
                raise StaleController('latest checkpoint no longer belongs to this controller')
            # Both LATEST and the transient replacement marker count. Each is
            # at most 11 bytes; 64 reserves both even on the initial save.
            if len(existing)+cuts>self.max_cuts or used+cuts*self.max_cut_bytes+64>self.max_bytes:
                raise StorageFull('complete publication and closure storage do not fit')
            self._reservation=cuts
            try:yield
            finally:self._reservation=None

    def save(self,state):
        if self._reservation is None:
            with self.transaction(1):return self.save(state)
        if self._reservation<1:raise StorageFull('publication exceeded reserved cuts')
        data=serialize(state);size=sum(map(len,data.values()))
        if size>self.max_cut_bytes:
            raise SnapshotTooLarge(f'exact snapshot {size} exceeds {self.max_cut_bytes}')
        cuts,used=self._inventory()
        if used+size+64>self.max_bytes:raise StorageFull('reserved storage changed outside journal')
        destination=self.root/f'cut-{len(cuts):06d}'
        tmp=Path(tempfile.mkdtemp(prefix='.inflight-',dir=self.root))
        for name,blob in data.items():
            with (tmp/name).open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
        fd=os.open(tmp,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        os.rename(tmp,destination)
        marker=self.root/'.LATEST.tmp'
        with marker.open('w') as f:f.write(destination.name+'\n');f.flush();os.fsync(f.fileno())
        os.replace(marker,self.root/'LATEST')
        fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        self._reservation-=1
        return destination.name

    def load(self):return self.load_bound()[1]

    def load_bound(self):
        # State bytes and their cursor are one read under the same writer fence.
        if self._reservation is not None:return self._read_bound()
        with (self.root/'.writer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            return self._read_bound()

    def _read_bound(self):
        cuts,_=self._inventory()
        if not cuts:raise ValueError('no checkpoint')
        root=cuts[-1];manifest=json.loads((root/'manifest.json').read_bytes())
        if set(manifest)!={'state.json','tensors.safetensors'}:raise ValueError('complete manifest required')
        data={}
        for name,m in manifest.items():
            data[name]=(root/name).read_bytes()
            if len(data[name])!=m['bytes'] or digest(data[name])!=m['sha256']:
                raise ValueError('damaged checkpoint')
        tensors=load_tensors(data['tensors.safetensors']);used=set()
        def decode(x):
            if set(x)!={'kind','value'}:raise ValueError('invalid checkpoint node')
            kind,value=x['kind'],x['value']
            if kind=='scalar':return value
            if kind=='tensor':
                if value in used:raise ValueError('tensor alias outside saved schema')
                used.add(value);return tensors[value]
            if kind=='dict':return {k:decode(v) for k,v in value.items()}
            if kind in ('list','tuple'):
                result=[decode(v) for v in value];return tuple(result) if kind=='tuple' else result
            raise ValueError('invalid checkpoint encoding')
        state=decode(json.loads(data['state.json']))
        if used!=set(tensors):raise ValueError('unbound tensor data')
        return root.name,state
