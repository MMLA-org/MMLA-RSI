"""Result-blind finite schedule over the exact whole-floor partition."""
import hashlib
from collections import defaultdict

SEEDS=(17,29,43)
ROLES=('FIT','DEV','C_J0','C_J1','C_J2','A_RSI1','A_RSI2','FINAL_OFFICIAL')

def _rank(prefix,value):return hashlib.sha256((prefix+':'+value).encode()).hexdigest()

def make_schedule(permits,general_ids):
    if len(permits)!=3687 or len(general_ids)!=7890 or len(set(general_ids))!=len(general_ids):
        raise ValueError('exact qualified task and general populations required')
    by=defaultdict(lambda:defaultdict(list))
    for task,p in permits.items():
        if task!=p.task_id or p.role not in ROLES:raise ValueError('exact permitted task required')
        by[p.role][p.floor].append(task)
    expected={'FIT':(60,1969),'DEV':(8,290),'C_J0':(8,278),'C_J1':(8,248),'C_J2':(8,228),
        'A_RSI1':(8,263),'A_RSI2':(8,277),'FINAL_OFFICIAL':(4,134)}
    seen=set()
    for role,(floors,tasks) in expected.items():
        if len(by[role])!=floors or sum(map(len,by[role].values()))!=tasks or seen.intersection(by[role]):
            raise ValueError('whole-floor role counts or intersections changed')
        seen.update(by[role])
    buckets=defaultdict(list)
    for floor in by['FIT']:
        if not floor.startswith('FloorPlan') or not floor[9:].isdigit():raise ValueError('original room identifier required')
        bucket=int(floor[9:])//100;buckets[bucket].append(floor)
    if sorted(buckets)!=[0,2,3,4] or any(len(v)!=15 for v in buckets.values()):raise ValueError('fixed room-bucket support')
    train={}
    for seed in SEEDS:
        room_floors={b:sorted(v,key=lambda f:_rank(f'train-floor:{seed}',f)) for b,v in buckets.items()}
        counts=defaultdict(int);batches=[]
        general=sorted(general_ids,key=lambda x:_rank(f'general:{seed}',x))
        for update in range(64):
            tasks=[]
            for bucket in sorted(buckets):
                floor=room_floors[bucket][update%15]
                available=by['FIT'][floor]
                for _ in range(2):
                    offset=counts[floor];cycle,index=divmod(offset,len(available))
                    ordered=sorted(available,key=lambda task:_rank(f'train-task:{seed}:{floor}:{cycle}',task))
                    task=ordered[index];counts[floor]+=1
                    episode_seed=int(_rank(f'environment:{seed}:{offset}',task)[:8],16)
                    tasks.append(dict(task=task,floor=floor,episode_seed=episode_seed,
                        fast_seed=int(_rank(f'fast:{seed}:{offset}',task)[:8],16)))
            if len({r['task'] for r in tasks})!=8:raise ValueError('duplicated task inside outer batch')
            batches.append(dict(update=update,tasks=tasks,general=general[8*update:8*(update+1)]))
        train[str(seed)]=batches
    evaluation={}
    for role in ROLES[1:]:
        limit=2 if role in ('DEV','A_RSI1','A_RSI2') else 1
        selected=[]
        for floor,tasks in sorted(by[role].items()):
            ordered=sorted(tasks,key=lambda t:_rank('orbit-outcome-eval-r1',t))
            selected.extend(ordered if role=='FINAL_OFFICIAL' else ordered[:limit])
        evaluation[role]=selected
    jfit=[]
    for bucket,floors in sorted(buckets.items()):
        for floor in sorted(floors,key=lambda f:_rank('orbit-outcome-jfit-r1',f))[:4]:
            jfit.append(min(by['FIT'][floor],key=lambda t:_rank('orbit-outcome-eval-r1',t)))
    evaluation['J_FIT']=jfit
    return dict(schema='orbit-outcome-finite-schedule-r1',seeds=SEEDS,train=train,evaluation=evaluation,
        evaluation_environment_seed=20261017,steps=64,tasks_per_update=8,general_per_update=8,
        selection_uses='original metadata and fixed SHA only; no outcomes or labels')
