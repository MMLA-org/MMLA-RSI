"""Source-derived compiler supervision for the existing immutable 640 episodes.

No new tasks, remote annotator or old final corpus. The inference lexer is pure;
these annotation functions are training/export utilities, never model inputs.
"""
import copy
import hashlib
import json
import re
from pathlib import Path
from .data import validate_episode,wire

KINDS=('null','create','update_value','update_unit','delete','quarantine','protect',
       'unprotect','acl','validity','alias','repair','rollback','purge')
ROLES=('object','value','unit','reference')
PRIMITIVES=('bridge','entity','number','unit','predicate','connector','punctuation','other')
LEXEME=re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[+-]?\d+(?:\.\d+)?|[^\w\s]')
PREDICATES=frozenset('create store set change replace delete mark quarantine suppress protect enable remove unprotect restrict allow expire make redirect repair restore roll purge erase keep'.split())
CONNECTORS=frozenset('to of from with for by as an a only the at and into in on its'.split())
CUES={
 'object':frozenset('create record new of set only replace delete mark quarantine suppress protect for from unprotect restrict read expire make redirect repair restore purge payload keep back'.split()),
 'value':frozenset(('value','to','by')),
 'unit':frozenset(('value','unit','to')),
 'reference':frozenset(('alias','of','record','to')),
}


def lexicalize(text):
    """Input-only ASCII profile. Between every two raw lexemes is one bridge.

    The bridge is present for every pair, independent of supervision. Both
    anchor and copy pointer address original words (even lattice indices).
    """
    if type(text) is not str or not text or len(text.encode())>16384 or not text.isascii():
        raise ValueError('bounded nonempty ASCII control-language view required')
    matches=list(LEXEME.finditer(text));last=0
    for m in matches:
        if text[last:m.start()].strip():raise ValueError('unrepresented raw character')
        last=m.end()
    if text[last:].strip() or not matches or 2*len(matches)-1>512:
        raise ValueError('complete bounded lexical mapping required')
    return dict(text=text,words=[m.group() for m in matches],spans=[(m.start(),m.end()) for m in matches])


def kind_for(operation):
    if operation.startswith('put_'):return 'create'
    if operation.startswith(('update_','try_update_','unauthorized_')):return 'update_value'
    if operation=='patch_a':return 'update_unit'
    if operation.startswith('invalid_repair_'):return 'repair'
    first=operation.split('_')[0]
    return {'restrict':'acl','expire':'validity'}.get(first,first)


def _primitive(word):
    lower=word.lower()
    if word.startswith('item_'):return 1
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',word):return 2
    if lower in ('units','kilounits'):return 3
    if lower in PREDICATES:return 4
    if lower in CONNECTORS:return 5
    if len(word)==1 and not word.isalnum() and word!='_':return 6
    return 7


def _anchor(words,role,target,start,end):
    # Labels choose real relational cues, never a clone of the target filler.
    choices=[i for i in range(start,end+1) if i!=target and 1<=abs(i-target)<=4
             and words[i].lower() in CUES[role]]
    if not choices:raise ValueError(f'no registered cue-to-filler path for {role}: {words[start:end+1]}')
    # value is deliberately shared by numeric and unit roles when in range;
    # typed transport then sees more than one possible filler from that cue.
    preferred=[i for i in choices if words[i].lower()=='value'] if role in ('value','unit') else []
    return min(preferred or choices,key=lambda i:(abs(i-target),i>target,i))


def annotate_segment(events,kinds):
    if not 1<=len(events)<=2 or len(events)!=len(kinds):raise ValueError('fixed two-event source profile')
    text='\n'.join(e['text'] for e in events);lex=lexicalize(text);words=lex['words'];n=len(words)
    labels=[];cursor=0;filler=[0]*n
    for source,kind in zip(events,kinds):
        endchar=cursor+len(source['text'])
        indices=[i for i,(s,e) in enumerate(lex['spans']) if cursor<=s and e<=endchar]
        if not indices:raise ValueError('empty observed event')
        start,end=indices[0],indices[-1]
        names=[i for i in indices if words[i].startswith('item_')]
        numbers=[i for i in indices if _primitive(words[i])==2]
        units=[i for i in indices if words[i] in ('units','kilounits')]
        targets=[None]*4
        if kind!='null':
            if not names:raise ValueError('missing source object')
            targets[0]=names[0]
        if kind in ('create','update_value','repair'):
            if len(numbers)!=1:raise ValueError('ambiguous source numeric value')
            targets[1]=numbers[0]
        if kind in ('create','repair','update_unit'):
            if len(units)!=1:raise ValueError('ambiguous source unit')
            targets[2]=units[0]
        if kind=='alias':
            if len(names)!=2:raise ValueError('alias source lacks distinct target')
            targets[3]=names[1]
        anchors=[None if t is None else _anchor(words,role,t,start,end) for role,t in zip(ROLES,targets)]
        for t in targets:
            if t is not None:filler[t]=1
        pointers=[2*t if t is not None else 2*n-1 for t in targets]
        anchor_indices=[2*t if t is not None else 2*n-1 for t in anchors]
        # All labels are outside observations; authority is checked only here
        # for source consistency, and is not a semantic forward feature.
        if kind!='null' and words[targets[0]]!=source['authority']['object']:
            raise ValueError('label object differs from source authority')
        labels.append(dict(kind=KINDS.index(kind),start=start,end=end,
            pointers=pointers,anchors=anchor_indices,
            values=[None if t is None else words[t] for t in targets]))
        cursor=endchar+1
    relation=[]
    for i,w in enumerate(words):
        if i:relation.append(0)
        relation.append(_primitive(w))
    for event in labels:
        for a,f in zip(event['anchors'],event['pointers']):
            if f==2*n-1:
                if a!=f:raise ValueError('private NULL mismatch')
            elif a==f or not 2<=abs(a-f)<=8 or a%2 or f%2:
                raise ValueError('invalid cue/transport path')
    return dict(observation=lex,targets=dict(count=len(labels),events=labels,
                relation=relation,filler=filler),lattice='raw-even-uniform-bridge-odd-v1')


def derive(row):
    validate_episode(row)
    operations=iter(a['operation'] for a in row['annotations']['events'])
    segments=[]
    for i,events in enumerate(row['observations']['segments']):
        kinds=['create']*len(events) if i==0 else [kind_for(next(operations)) for _ in events]
        segments.append(annotate_segment(events,kinds))
    try:next(operations);raise ValueError('unbound source events')
    except StopIteration:pass
    return dict(schema='mmla-engineering-compiler-labels-v1',source_id=row['id'],semantic_group=row['semantic_group'],
                source_row_sha256=hashlib.sha256(wire(row)).hexdigest(),segments=segments)


def build(source,destination):
    source,dest=Path(source),Path(destination)
    if dest.exists():raise FileExistsError('derived data version is immutable')
    manifest=json.loads((source/'manifest.json').read_bytes());outputs={};files=[]
    counts=dict(episodes=0,segments=0,events=0,roles=0,null_roles=0,shared_cues=0)
    for spec in manifest['files']:
        path=source/spec['path'];raw=path.read_bytes()
        if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('source data identity mismatch')
        derived=[]
        for line in raw.splitlines():
            row=json.loads(line);new=derive(row);derived.append(new);counts['episodes']+=1
            for segment in new['segments']:
                counts['segments']+=1;n=2*len(segment['observation']['words'])-1
                for event in segment['targets']['events']:
                    counts['events']+=1
                    active=[a for a in event['anchors'] if a!=n]
                    counts['shared_cues']+=int(len(set(active))<len(active))
                    counts['roles']+=len(active);counts['null_roles']+=4-len(active)
        outputs[spec['path']]=b''.join(wire(x)+b'\n' for x in derived)
    if counts['episodes']!=640 or counts['shared_cues']==0:raise ValueError('source bound or transport support missing')
    # Validate the whole finite source before publishing any derived file.
    dest.mkdir(parents=True)
    for name,raw in outputs.items():
        (dest/name).write_bytes(raw);files.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    result=dict(schema=1,source_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=files,counts=counts,
                groups=manifest['groups'],model_calls=0,external_calls=0,
                status='DERIVED_COMPILER_TARGETS_ONLY_NOT_G_Q_OR_TRAINED_MODEL',
                inference_boundary='lexical observation only; targets, source_id, group and authority absent from neural frontend')
    (dest/'manifest.json').write_bytes(wire(result)+b'\n')
    return result
