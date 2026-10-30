"""Small operation progress beacon; no model/data payload, no timer heartbeats.

Only completed source operations call advance. Repeated process liveness cannot
keep a stuck CUDA/environment call alive. It never changes model/RNG state.
"""
from pathlib import Path
import json,os,time
_context=None

def configure(root,phase):
    global _context
    if _context is not None:raise RuntimeError('one phase progress owner')
    root=Path(root).resolve(strict=True)
    _context=dict(path=root/'operation-progress.json',phase=phase,counter=0,last=0.)
    advance('phase_start',force=True)

def advance(kind,*,force=False):
    if _context is None:return
    c=_context;c['counter']+=1;now=time.monotonic()
    if not force and now-c['last']<5.:return
    from .outcome_resources import _json_atomic
    _json_atomic(c['path'],dict(phase=c['phase'],pid=os.getpid(),counter=c['counter'],
        monotonic=now,kind=str(kind)[:96]))
    c['last']=now

def finish():advance('phase_finished',force=True)
