"""Bounded engineering corpus; truth is a deterministic symbolic state machine.

This module never imports model code. No hosted model, historical evaluation data,
network, or dependency download is used. Annotations are separate from observed
input; canonical semantic groups, not cosmetic IDs, determine the split.
"""
import argparse
import copy
import hashlib
import itertools
import json
import random
import string
from pathlib import Path


def wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(x):return hashlib.sha256(wire(x)).hexdigest()

# Each tuple is a distinct operation graph. Shared individual primitives may be
# learned, but the complete script/query/permission combination is held together.
SCRIPTS=(
    ('put_c',), ('update_a',), ('delete_a',), ('quarantine_a',),
    ('protect_a','try_update_a'), ('protect_a','unprotect_a','update_a'),
    ('restrict_a',), ('expire_a',), ('alias_b_a',), ('alias_a_b','update_b'),
    ('quarantine_a','repair_a'), ('update_a','rollback_a'), ('purge_a',),
    ('patch_a',), ('put_c','delete_c'), ('delete_a','invalid_repair_a'),
    ('null',), ('unauthorized_a',), ('update_a','update_b'),
    ('update_b','alias_a_b','restrict_b'),
)
QUERIES=('read_a','read_b','inventory','equal_ab')
CONTEXTS=('private','shared','reader_excluded','expiry_boundary')
SCHEMA='mmla-engineering-events-v1'


def semantic_spec(script,query,context):
    """Canonical whole-task graph under object renaming; no cosmetic new groups."""
    support='read_b' if query!='read_b' else 'read_a'
    acl=['owner','reader'] if context!='reader_excluded' else ['owner']
    initial={name:dict(acl=list(acl),expiry=4 if context=='expiry_boundary' else 99,
        unit='units',status='live',protected=False) for name in ('a','b')}
    if context=='shared':initial['b']['acl'].append('auditor')
    candidates=[]
    for swap in (False,True):
        rename=lambda word: '_'.join(('b' if x=='a' else 'a' if x=='b' else x) if swap else x for x in word.split('_'))
        # equal_ab is symmetric and inventory is permutation invariant.
        candidates.append(dict(operations=[rename(x) for x in script],query=rename(query),
            support_query=rename(support),initial={rename(k):v for k,v in initial.items()},
            constants='distinct values; only equality is queried; updates preserve distinctness',
            answer_semantics='authorized-current-value',expiry_semantics='inclusive-valid-until'))
    return min(candidates,key=wire)


class Truth:
    """Oracle state, intentionally independent of AMR/model implementations."""
    def __init__(self,context,values):
        self.context=context;self.values=values;self.rows={};self.clock=0;self.archive={}
        acl=['owner','reader'] if context!='reader_excluded' else ['owner']
        for name in ('A','B'):
            self.rows[name]=dict(status='live',value=values[name],unit='units',acl=list(acl),
                protected=False,expiry=4 if context=='expiry_boundary' else 99,version=1,alias=None)
        if context=='shared':self.rows['B']['acl'].append('auditor')

    def visible(self,name,principal='reader',seen=()):
        r=self.rows.get(name)
        if r is None or r['status']!='live' or principal not in r['acl'] or self.clock>r['expiry']:return None
        if name in seen:raise ValueError('oracle alias cycle')
        if r['alias'] is not None:return self.visible(r['alias'],principal,seen+(name,))
        return dict(value=r['value'],unit=r['unit'])

    def answer(self,query):
        if query.startswith('read_'):return self.visible(query[-1].upper())
        if query=='inventory':return {n:self.visible(n) for n in sorted(self.rows) if self.visible(n) is not None}
        if query=='equal_ab':
            a,b=self.visible('A'),self.visible('B')
            return None if a is None or b is None else a==b
        raise ValueError(query)

    def execute(self,op,envelope):
        before=copy.deepcopy(self.rows);self.clock+=1
        action={'put':'insert','try':'update','invalid':'repair','unauthorized':'update'}.get(op.split('_')[0],op.split('_')[0])
        if envelope['principal']!='owner' or action not in envelope['allowed']:
            return 'REJECT_AUTHORITY' 
        key='C' if op.endswith('_c') else 'B' if op.endswith('_b') else 'A'
        if op=='null':return 'NULL'
        if op=='invalid_repair_a':
            if self.rows['A']['status']!='quarantine':return 'REJECT_STATE'
            raise ValueError('invalid-repair scenario lacks its stated precondition')
        if op=='put_c':
            self.rows['C']=dict(status='live',value=self.values['C'],unit='units',acl=['owner','reader'],protected=False,expiry=99,version=1,alias=None)
            return 'COMMIT'
        if op in ('alias_b_a','alias_a_b'):key='B' if op=='alias_b_a' else 'A'
        row=self.rows[key]
        if op=='try_update_a' and row['protected']:return 'REJECT_PROTECTED'
        self.archive.setdefault(key,[]).append(copy.deepcopy(row))
        if op.startswith('update_') or op=='try_update_a':row.update(value=self.values[key]+7,alias=None)
        elif op=='patch_a':row['unit']='kilounits'
        elif op.startswith('delete_'):row.update(status='tombstone',value=None,alias=None)
        elif op=='quarantine_a':row.update(status='quarantine',value=None,alias=None)
        elif op=='repair_a':
            if row['status']!='quarantine':raise ValueError('oracle illegal repair')
            row.update(status='live',value=self.values['A']+7,alias=None)
        elif op=='rollback_a':
            target=self.archive[key][-2]
            row.update(status=target['status'],value=target['value'],unit=target['unit'],alias=target['alias'])
        elif op=='purge_a':row.update(status='purged',value=None,alias=None)
        elif op=='protect_a':row['protected']=True
        elif op=='unprotect_a':row['protected']=False
        elif op.startswith('restrict_'):row['acl']=['owner']
        elif op=='expire_a':row['expiry']=self.clock
        elif op in ('alias_b_a','alias_a_b'):row['alias']='A' if key=='B' else 'B'
        else:raise ValueError('unknown oracle operation')
        row['version']+=1
        if before==self.rows:raise ValueError('non-NULL operation made no change')
        return 'COMMIT'


OP_TEXT={
'put_c':('Create {C} with value {vC} units.','Store a new record {C}: value {vC}, unit units.'),
'update_a':('Set the value of {A} to {nA}; keep its other fields.','Change only {A} value to {nA}.'),
'update_b':('Set the value of {B} to {nB}; keep its other fields.','Change only {B} value to {nB}.'),
'delete_a':('Delete {A}.','Mark {A} as deleted.'),'delete_c':('Delete {C}.','Mark {C} as deleted.'),
'quarantine_a':('Quarantine {A}.','Suppress {A} pending repair.'),
'protect_a':('Protect {A} from ordinary replacement.','Enable write protection for {A}.'),
'unprotect_a':('Remove write protection from {A}.','Unprotect {A} with management authority.'),
' try_update_a':('Set {A} value to {nA}.','Replace {A} value by {nA}.'),
'restrict_a':('Restrict {A} to owner reads.','Allow only owner to read {A}.'),
'restrict_b':('Restrict {B} to owner reads.','Allow only owner to read {B}.'),
'expire_a':('Expire {A} at the current clock.','Set the validity end of {A} to now.'),
'alias_b_a':('Make {B} an alias of {A}.','Redirect {B} to record {A}.'),
'alias_a_b':('Make {A} an alias of {B}.','Redirect {A} to record {B}.'),
'repair_a':('Repair quarantined {A} with value {nA} units.','Restore {A} from quarantine: value {nA}, unit units.'),
'rollback_a':('Restore the prior committed content of {A}.','Roll back {A} content to its preceding committed version.'),
'purge_a':('Purge resident content of {A}.','Erase the current stored payload for {A}.'),
'patch_a':('Change only the unit of {A} to kilounits.','Keep {A} value and set unit to kilounits.'),
'invalid_repair_a':('Repair quarantined {A} with value {nA} units.','Restore {A} from quarantine: value {nA}, unit units.'),
'null':('Do not change memory.','Keep all stored records unchanged.'),
'unauthorized_a':('Set {A} value to {nA}.','Replace {A} value by {nA}.'),
}
OP_TEXT['try_update_a']=OP_TEXT.pop(' try_update_a')


def make_episode(script,query,context,variant):
    spec=semantic_spec(script,query,context);group=sha(spec)
    # Independent lexical PRNG; neither semantic hash nor split is rendered.
    presentation_index=(SCRIPTS.index(tuple(script))*len(QUERIES)+QUERIES.index(query))*len(CONTEXTS)+CONTEXTS.index(context)
    rng=random.Random(2026101507+presentation_index*2+variant)
    names={k:'item_'+''.join(rng.choices(string.ascii_lowercase,k=8)) for k in ('A','B','C')}
    if len(set(names.values()))!=3:raise ValueError('lexical collision')
    values={'A':rng.randrange(11,58),'B':rng.randrange(71,114),'C':rng.randrange(137,178)}
    fmt={**names,**{'v'+k:v for k,v in values.items()},'nA':values['A']+7,'nB':values['B']+7}
    truth=Truth(context,values)
    # Runtime input contains legally observed facts and authenticated envelope;
    # no future answer, truth rows, split, or semantic group is an input feature.
    initial=[]
    for name in ('A','B'):
        r=truth.rows[name]
        initial.append(dict(text=f"Create {names[name]} with value {values[name]} units.",
            authority=dict(tenant='engineering',principal='owner',object=names[name],
                acl=r['acl'],valid_until=r['expiry'],allowed=['insert'])))
    segments=[initial];annotations=[]
    events=[]
    for op in script:
        key='C' if op.endswith('_c') else 'A' if op=='alias_a_b' else 'B' if op.endswith('_b') or op=='alias_b_a' else 'A'
        principal='intruder' if op=='unauthorized_a' else 'owner'
        operation={'put':'insert','try':'update','invalid':'repair','unauthorized':'update'}.get(op.split('_')[0],op.split('_')[0])
        event=dict(text=OP_TEXT[op][variant].format(**fmt),authority=dict(tenant='engineering',
            principal=principal,object=names[key],source_clock=truth.clock+1,
            allowed=[] if principal=='intruder' else [operation],
            override_protection=op=='unprotect_a',
            acl=['owner','reader'] if op=='put_c' else None,
            valid_until=99 if op=='put_c' else None))
        before=copy.deepcopy(truth.rows)
        outcome=truth.execute(op,event['authority'])
        events.append(event)
        annotations.append(dict(operation=op,outcome=outcome,before=before,state=copy.deepcopy(truth.rows),clock=truth.clock))
    segments.extend([events[i:i+2] for i in range(0,len(events),2)])
    # Advance to the declared query clock. This is visible external time, not
    # model-controlled timing and not an observation of the future answer.
    truth.clock=4
    questions={'read_a':f'As reader, report the current value and unit of {names["A"]}, or null if unavailable.',
       'read_b':f'As reader, report the current value and unit of {names["B"]}, or null if unavailable.',
       'inventory':'As reader, list the available records with their current values and units.',
       'equal_ab':f'As reader, do {names["A"]} and {names["B"]} have equal values and units? Return null if either is unavailable.'}
    def answer(q):
        v=truth.answer(q)
        if q=='inventory':v={names[k]:x for k,x in v.items()}
        return v
    # First query is a supervised feedback opportunity; the distinct second
    # query is scored after adaptation. No model outcome is invented here.
    support='read_b' if query!='read_b' else 'read_a'
    return dict(schema=SCHEMA,id=sha(dict(script=list(script),query=query,context=context,variant=variant)),variant=variant,semantic_group=group,semantic_spec=spec,template=dict(script=list(script),query=query,context=context),
        observations=dict(segments=segments,query_clock=4),
        support=dict(prompt=questions[support],answer=answer(support),authority=dict(tenant='engineering',principal='reader',clock=4),
            feedback_kind='authoritative_answer_correction'),
        query=dict(prompt=questions[query],answer=answer(query),authority=dict(tenant='engineering',principal='reader',clock=4)),
        annotations=dict(events=annotations,final_state=truth.rows,values=values,names=names),
        limits=dict(memory_slots=8,max_events_per_segment=2,max_segments=4,max_feedback_updates=1),
        provenance=dict(generator=SCHEMA,external_calls=0,historical_data=False))


def inventory():
    specs=[(s,q,c) for s,q,c in itertools.product(SCRIPTS,QUERIES,CONTEXTS)]
    # Split canonical groups using one frozen hash rule. The resulting sizes are
    # reported, not adjusted using performance or to force a desired count.
    groups={sha(semantic_spec(s,q,c)) for s,q,c in specs}
    assignment={g:('fit' if int(g[:8],16)%100<80 else 'calibration'
        if int(g[:8],16)%100<90 else 'final') for g in groups}
    return specs,assignment


def validate_episode(row):
    spec=row['semantic_spec'];t=row['template'];expected=make_episode(tuple(t['script']),t['query'],t['context'],row['variant'])
    if row!=expected:raise ValueError('episode differs from deterministic generation')
    if row['semantic_group']!=sha(spec):raise ValueError('semantic identity mismatch')
    if len(row['observations']['segments'])>4 or any(len(x)>2 for x in row['observations']['segments']):raise ValueError('declared event capacity exceeded')
    # Independent interface invariants catch errors that regeneration cannot.
    names=row['annotations']['names'];inverse={v:k for k,v in names.items()}
    observed=[e for segment in row['observations']['segments'][1:] for e in segment]
    states=row['annotations']['events']
    if len(observed)!=len(states):raise ValueError('missing event annotation')
    for e,a in zip(observed,states):
        auth=e['authority'];target=inverse[auth['object']]
        mentioned=sorted((e['text'].find(name),role) for role,name in names.items() if name in e['text'])
        if a['operation']!='null' and (not mentioned or mentioned[0][1]!=target):
            raise ValueError('observed request and authority target disagree')
        changed={k for k in set(a['before'])|set(a['state']) if a['before'].get(k)!=a['state'].get(k)}
        if a['outcome'] in ('NULL','REJECT_AUTHORITY','REJECT_PROTECTED','REJECT_STATE'):
            if changed:raise ValueError('rejected/NULL event mutated truth')
        elif changed!={target}:raise ValueError('truth changed a different object than authority target')
        if a['operation']=='put_c' and (auth['acl']!=a['state']['C']['acl'] or auth['valid_until']!=a['state']['C']['expiry']):
            raise ValueError('create omits or changes trusted metadata')
    for name in ('support','query'):
        if row[name]['authority']!=dict(tenant='engineering',principal='reader',clock=row['observations']['query_clock']):
            raise ValueError('query authority/clock differs from truth adapter')
    for r in row['annotations']['final_state'].values():
        if r['status']!='live' and r['value'] is not None:raise ValueError('suppressed object leaked content')
    return True


def build(destination):
    dest=Path(destination)
    if dest.exists():raise FileExistsError('corpus is immutable; select a new version directory')
    specs,assignment=inventory();rows={s:[] for s in ('fit','calibration','final')}
    for script,query,context in specs:
        for variant in (0,1):
            row=make_episode(script,query,context,variant);validate_episode(row)
            rows[assignment[row['semantic_group']]].append(row)
    if sum(map(len,rows.values()))!=640 or not 1<=len(assignment)<=320:raise ValueError('fixed corpus bound violated')
    dest.mkdir(parents=True)
    files=[]
    for split,entries in rows.items():
        path=dest/(split+'.jsonl');path.write_bytes(b''.join(wire(x)+b'\n' for x in entries))
        files.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
            episodes=len(entries),semantic_groups=len({x['semantic_group'] for x in entries})))
    manifest=dict(schema=SCHEMA,purpose='engineering module training; not C evaluation or advantage evidence',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=files,
        split_semantics='structural whole-task combination; primitives shared; no claim of unseen primitive generalization',
        groups=assignment,generated_episodes=640,external_calls=0,model_calls=0,
        final_access='source/data validation only before training; no model selection on final',
        capacity='8 slots is an explicit engineering profile, not an established deployment requirement',
        status='GENERATED_NOT_TOKENIZED_NOT_TRAINED')
    (dest/'manifest.json').write_bytes(wire(manifest)+b'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build(args.output),indent=2))
