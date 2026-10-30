"""Authenticate and merge one private worker's closed task in owner order."""
from pathlib import Path
import hashlib,json
from production_common import read,digest


def consume_job(directory,response,request,*,worker,source,terminal_kind,on_work,on_event):
    job=response['job'];relative=f'job-{job}/complete.json'
    if response['receipt']!=relative:raise ValueError('assigned worker result path differs')
    path=Path(directory)/f'worker-{worker}'/relative
    result=read(path,expected=response['sha256'],cap=2*1024**2)
    if (result['schema']!='orbit-scalar-worker-task-v1' or result['source']!=source
            or result['job']!=job or result['request']!=request
            or result['environment_closed'] is not True or result['GPU_synchronized'] is not True):
        raise ValueError('assigned task must close before feedback')
    manifest=read(path.parent/'events/manifest.json',expected=result['event_manifest_sha256'])
    if manifest['schema']!='orbit-production-event-shards-v1':raise ValueError('worker event manifest required')
    records=[];counts={};totals={};sequence=0;closed=[]
    fields=('input_tokens','target_tokens','valid_tokens','padded_positions','tokens','positions','examples','count','updates','candidates')
    for number,shard in enumerate(manifest['shards']):
        if shard['path']!=f'{number:06d}.jsonl':raise ValueError('ordered worker events required')
        raw=(path.parent/'events'/shard['path']).read_bytes()
        if len(raw)!=shard['bytes'] or hashlib.sha256(raw).hexdigest()!=shard['sha256']:
            raise ValueError('worker event bytes changed')
        for line in raw.splitlines():
            row=json.loads(line);sequence+=1
            if row['index']!=sequence:raise ValueError('worker event cursor changed')
            event=row['event'];records.append(event)
            if event['kind']==terminal_kind:closed.append(event['endpoint'] if terminal_kind=='FIT_RETURN_ENDPOINT_COMPLETE' else event['result'])
            if event['kind']!='WORK':continue
            item=event['record'];kind=item['kind'];counts[kind]=counts.get(kind,0)+1
            for field in fields:
                value=item.get(field)
                if type(value) is int:
                    key=kind+'.'+field;totals[key]=totals.get(key,0)+value
    if (sequence!=manifest['events'] or counts!=result['counts'] or totals!=result['totals']
            or closed!=[result['endpoint']]):raise ValueError('complete closed task work differs')
    # The parent ledger owns each physical WORK once. Private worker logs are
    # retained evidence and are not added again when releasing parent totals.
    on_event(dict(kind='SCALAR_TASK_REPLAY_BEGIN',worker=worker,job=job))
    for event in records:
        if event['kind']=='WORK':on_work(event['record'])
        else:on_event(event)
    reference=dict(kind='SCALAR_TASK_RECONCILED',worker=worker,job=job,request=request,
        receipt=f'worker-{worker}/'+relative,receipt_sha256=response['sha256'],
        event_manifest_sha256=result['event_manifest_sha256'],endpoint=digest(result['endpoint']),
        event_count=len(records),replayed_events_sha256=digest(records))
    on_event(reference)
    return result,reference
