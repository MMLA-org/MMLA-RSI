"""Fixed two-epoch schedule and data-only, identity-bound proposal checkpoints."""
from recipe import EXAMPLES,PRESENTATIONS
import hashlib
import json
import os
from pathlib import Path
import random
import re
from curriculum import digest, SALT, SEEDS


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('canonical SHA256 identity required')
    return value


class TrainingPlan:
    def __init__(self,examples,*,selection_identity,reader_identity,seed):
        sha(selection_identity);sha(reader_identity)
        if type(examples) is not list or len(examples)!=EXAMPLES or seed not in SEEDS:
            raise ValueError('fixed complete reader curriculum required')
        ids=[]
        for row in examples:
            if (row.get('seed')!=seed or row.get('source_roles')!=['FIT_SUPPORT','FIT_QUERY']
                    or row.get('future_scores_in_prompt') is not False
                    or row.get('recursive_benefit_claim') is not False
                    or row.get('identity')!=digest({k:v for k,v in row.items() if k!='identity'})):
                raise ValueError('changed, foreign or improperly labelled curriculum row')
            ids.append(row['identity'])
        if len(set(ids))!=EXAMPLES:raise ValueError('all distinct example identities required')
        self.examples=examples
        self.data=dict(schema='orbit-proposal-learning-data-v1',selection=selection_identity,
                       reader=reader_identity,seed=seed,example_identities=ids)
        self.data_identity=digest(self.data)
        self.order=[]
        for epoch in range(2):
            order=list(range(EXAMPLES))
            seedbytes=hashlib.sha256(f'{SALT}:schedule:{seed}:{epoch}'.encode()).digest()
            random.Random(int.from_bytes(seedbytes,'big')).shuffle(order)
            self.order.extend(order)
        self.manifest=dict(schema='orbit-proposal-learning-schedule-v1',data=self.data_identity,
                           seed=seed,order=self.order,batch_size=8,epochs=2)
        self.identity=digest(self.manifest)

    def verify(self):
        if (digest(self.data)!=self.data_identity or digest(self.manifest)!=self.identity
                or self.manifest['data']!=self.data_identity or len(self.examples)!=EXAMPLES):
            raise ValueError('bound schedule or dataset inventory changed')

    def batch(self,cursor):
        self.verify()
        if type(cursor) is not int or not 0<=cursor<PRESENTATIONS or cursor%8:
            raise ValueError('complete scheduled batch boundary required')
        rows=[self.examples[i] for i in self.order[cursor:cursor+8]]
        for i,row in zip(self.order[cursor:cursor+8],rows):
            if (row['identity']!=self.data['example_identities'][i]
                    or row['identity']!=digest({k:v for k,v in row.items() if k!='identity'})):
                raise ValueError('curriculum changed after schedule binding')
        return rows


def _write(path,data):
    with path.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def _json(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()+b'\n'


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def _decode(raw):
    def nonfinite(_):raise ValueError('nonfinite JSON')
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=nonfinite)


def save_checkpoint(path,trainer,plan,*,source_identity):
    """Final commit marker is written only after all checkpoint files are durable.

    On failure the new directory is retained as partial and cannot be reused.
    Parent contract controls all resource charging and later continuation rights.
    """
    from mmla_pdsa.full.safe_state import save_state
    sha(source_identity)
    plan.verify()
    if trainer.data_identity!=plan.data_identity:raise ValueError('trainer/data mismatch')
    state=trainer.export()
    directory=Path(path);directory.mkdir(parents=True,exist_ok=False)
    payload=dict(schema='orbit-proposal-learning-checkpoint-v1',source=source_identity,
                 schedule_identity=plan.identity,schedule=plan.manifest,data=plan.data,trainer=state)
    save_state(directory/'state',payload)
    files=[]
    for leaf in ('state.json','state.safetensors'):
        p=directory/leaf
        with p.open('rb') as f:os.fsync(f.fileno());data=f.read()
        files.append(dict(path=leaf,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(schema='orbit-proposal-learning-checkpoint-commit-v1',source=source_identity,
                  schedule=plan.identity,data=plan.data_identity,step=trainer.steps,
                  next_example=trainer.cursor,finished=trainer.finished,files=files)
    _write(directory/'COMMITTED.json',_json(manifest))
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(directory),manifest_sha256=hashlib.sha256((directory/'COMMITTED.json').read_bytes()).hexdigest(),
                **{k:manifest[k] for k in ('step','next_example','finished')})


def read_checkpoint(path,plan,*,source_identity,manifest_sha256):
    """Decode precisely the hashed bytes, never re-read an unchecked payload."""
    from safetensors.torch import load
    from mmla_pdsa.full.safe_state import decode,SCHEMA
    sha(source_identity);sha(manifest_sha256)
    plan.verify()
    directory=Path(path)
    if directory.is_symlink() or not directory.is_dir():raise ValueError('checkpoint directory required')
    expected={'COMMITTED.json','state.json','state.safetensors'}
    if {p.name for p in directory.iterdir()}!=expected:raise ValueError('complete exact checkpoint inventory required')
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():raise ValueError('regular checkpoint files required')
    if (directory/'COMMITTED.json').stat().st_size>4096:raise ValueError('bounded checkpoint manifest required')
    raw=(directory/'COMMITTED.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('checkpoint commit identity changed')
    manifest=_decode(raw)
    if (set(manifest)!={'schema','source','schedule','data','step','next_example','finished','files'}
            or manifest['schema']!='orbit-proposal-learning-checkpoint-commit-v1'
            or manifest['source']!=source_identity or manifest['schedule']!=plan.identity
            or manifest['data']!=plan.data_identity
            or [r['path'] for r in manifest['files']]!=['state.json','state.safetensors']):
        raise ValueError('checkpoint source, curriculum or schedule mismatch')
    payloads={}
    for row in manifest['files']:
        if set(row)!={'path','bytes','sha256'}:raise ValueError('complete file identity required')
        if (type(row['bytes']) is not int or not 0<row['bytes']<=16*1024**2
                or (directory/row['path']).stat().st_size!=row['bytes']):
            raise ValueError('bounded exact checkpoint payload size required')
        data=(directory/row['path']).read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('checkpoint payload changed')
        payloads[row['path']]=data
    tree=_decode(payloads['state.json'])
    if set(tree)!={'schema','tree'} or tree['schema']!=SCHEMA:raise ValueError('typed safe-state required')
    tensors=load(payloads['state.safetensors']);used=set();value=decode(tree['tree'],tensors,used)
    if used!=set(tensors):raise ValueError('unused checkpoint tensors')
    if (set(value)!={'schema','source','schedule_identity','schedule','data','trainer'}
            or value['schema']!='orbit-proposal-learning-checkpoint-v1'
            or value['source']!=source_identity or value['schedule_identity']!=plan.identity
            or value['schedule']!=plan.manifest or value['data']!=plan.data):
        raise ValueError('decoded training source/schedule identity mismatch')
    state=value['trainer']
    if (state['data_identity']!=plan.data_identity or state['steps']!=manifest['step']
            or state['cursor']!=manifest['next_example'] or state['finished']!=manifest['finished']):
        raise ValueError('checkpoint cursor/commit mismatch')
    return state
