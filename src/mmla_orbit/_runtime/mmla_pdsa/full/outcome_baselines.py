"""Fixed observable-only baselines; no task identifiers enter decision features.

Readers get current public frame and actual retained observation fragments.
The historical FIT course is an offline labeled corpus, never live feedback.
No model/environment/install occurs at import. This module does not select an
algorithm from evaluation outcomes or claim that a heuristic is optimal.
"""
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass
import hashlib,json,math,re
from pathlib import Path
from .alfworld_episode import VisibleFrame,_digest,_sha
from .alfworld_formatting import ACTION_INSTRUCTION,_controls
from .observation_memory import ObservationCodec

BASELINE_SCHEMA='orbit-observable-baselines-v1'
NAME=r'[a-z][a-z0-9_]* [0-9]+'

@dataclass(frozen=True)
class BaselineView:
    frame:VisibleFrame
    fragments:tuple[bytes,...]
    step:int
    memory_identity:str
    codec:ObservationCodec
    def __post_init__(self):
        self.validate()
    def validate(self):
        if (type(self.frame) is not VisibleFrame or self.frame.admissible_commands or
                type(self.step) is not int or not 0<=self.step<=64 or type(self.fragments) is not tuple):
            raise ValueError('public frame and bounded retained fragments required')
        _sha(self.memory_identity)
        if type(self.codec) is not ObservationCodec or self.codec.profile.memory_slots!=8 or len(self.fragments)>8:
            raise ValueError('the exact common codec and at most eight resident capsules required')
        intervals={}
        for b in self.fragments:
            x,_=self.codec.validate(b)
            if x['observed_step']>self.step:raise ValueError('future observation row')
            start=x['char_offset'];end=start+len(x['text'])
            for lo,hi in intervals.setdefault(x['observed_step'],[]):
                if start<hi and lo<end:raise ValueError('overlapping or duplicate source fragments')
            intervals[x['observed_step']].append((start,end))
    def passages(self):
        self.validate()
        # Never join over an evicted gap, even within the same old observation.
        out=[];step=None;end=None
        for b in sorted(self.fragments,key=lambda b:(json.loads(b)['observed_step'],json.loads(b)['char_offset'])):
            x=json.loads(b)
            if x['observed_step']==step and x['char_offset']==end:out[-1]+=x['text']
            else:out.append(x['text'])
            step=x['observed_step'];end=x['char_offset']+len(x['text'])
        return tuple(out)

    def complete_clauses(self):
        """Rule parser sees only sentences whose two edges are present.

        A fragment suffix 'apple 1' could be a prefix of 'apple 10'. Do not
        infer a finished entity from a missing right edge or join across gaps.
        """
        self.validate();groups=[];step=None;end=None
        for b in sorted(self.fragments,key=lambda b:(json.loads(b)['observed_step'],json.loads(b)['char_offset'])):
            x=json.loads(b)
            if x['observed_step']==step and x['char_offset']==end:groups[-1][1]+=x['text']
            else:groups.append([x['char_offset'],x['text']])
            step=x['observed_step'];end=x['char_offset']+len(x['text'])
        result=[]
        for offset,text in groups:
            if offset:
                first=text.find('.')
                if first<0:continue
                text=text[first+1:]
            last=text.rfind('.')
            if last>=0:result.append(text[:last+1])
        return tuple(result)

class ObservationFIFO:
    """Eight complete capsules using the SAME MiniCPM codec for every method.

    No private factual archive; cursor/time are control metadata. Complete
    history uses a separately declared diagnostic carrier and cost accounting.
    """
    def __init__(self,codec,*,slots=8):
        if type(codec) is not ObservationCodec or slots!=8 or codec.profile.memory_slots!=8:raise ValueError('common eight-slot source codec required')
        self.codec=codec;self.rows=[None]*slots;self.cursor=0;self.step=-1
        self.codec_identity=codec.identity
    def append(self,observation,*,step,source_receipt):
        if self.codec.identity!=self.codec_identity or step!=self.step+1:raise ValueError('sequential actual source required')
        batch=self.codec.split(observation,step=step,source_receipt=source_receipt)
        # split validates the ENTIRE payload before changing any resident row.
        rows=list(self.rows);cursor=self.cursor
        for b in batch.chunks:rows[cursor]=b;cursor=(cursor+1)%len(rows)
        self.rows,self.cursor,self.step=rows,cursor,step
        return dict(payload=batch.identity,fragments=len(batch.chunks),retained=sum(x is not None for x in rows),
                    canonical_bytes=sum(len(x) for x in rows if x is not None),identity=self.identity)
    @property
    def identity(self):return _digest(dict(schema=BASELINE_SCHEMA,codec=self.codec_identity,
        rows=[None if x is None else x.hex() for x in self.rows],cursor=self.cursor,step=self.step))
    def view(self,frame):
        if self.step<0:raise ValueError('initial observation must be committed')
        return BaselineView(frame,tuple(x for x in self.rows if x is not None),self.step,self.identity,self.codec)


def public_goal(goal):
    """Only the twelve public FIT-language forms; no structured task-name parse."""
    goal=goal.strip().lower().removeprefix('your task is to: ').rstrip('.')
    patterns=(
      (r'put (?:some|a) (\w+) (?:on|in) (\w+)',None,1),
      (r'put a (clean|hot|cool) (\w+) in (\w+)', 'state',1),
      (r'(clean|heat|cool) some (\w+) and put it in (\w+)', 'verb',1),
      (r'(?:put|find) two (\w+)(?: and put them)? in (\w+)',None,2),
      (r'examine the (\w+) with the (\w+)','examine',1),
      (r'look at (\w+) under the (\w+)','examine',1))
    for pattern,kind,count in patterns:
        m=re.fullmatch(pattern,goal)
        if not m:continue
        groups=m.groups()
        if kind in ('state','verb'):
            state,obj,dest=groups;operation={'hot':'heat','clean':'clean','cool':'cool','heat':'heat'}[state]
        else:
            obj,dest=groups;operation='examine' if kind=='examine' else 'place'
        return dict(object=obj,destination=dest,operation=operation,count=count)
    return None


def public_rule(view):
    """Deterministic retained-observation planner, recomputed for every action.

    Tracks only facts derivable from currently readable text, never an external
    per-task map. Lexical ties are fixed; step rotates fully visited locations.
    """
    if type(view) is not BaselineView:raise TypeError('observable-only view required')
    return public_rule_passages(view.frame.goal,view.frame.observation,view.complete_clauses(),view.step)


def public_rule_passages(goal_text,observation,clauses,step):
    """Same planning algorithm, shared with the declared full-history upper reference."""
    if type(goal_text) is not str or type(observation) is not str or type(clauses) is not tuple or any(type(x) is not str for x in clauses) or type(step) is not int or not 0<=step<=64:
        raise ValueError('public text and actual bounded time required')
    goal=public_goal(goal_text)
    if goal is None:return 'look'
    receptacles=set();objects={};open_state={};visited=set();held=set();processed=set();placed=set();location=None
    texts=(*clauses,observation)
    for text in texts:
        low=text.lower()
        if 'middle of a room' in low:
            receptacles.update(re.findall(NAME,low))
        for match in re.finditer(r'(?:on the|the) ('+NAME+r')(?:, you see| is (?:open|closed))',low):
            rec=match[1];receptacles.add(rec);visited.add(rec);location=rec
        arrival=re.search(r'you arrive at ('+NAME+r')\.',low)
        if arrival and not arrival[1].startswith('loc '):location=arrival[1];receptacles.add(location);visited.add(location)
        facing=re.search(r'you are facing the ('+NAME+r')',low)
        if facing:location=facing[1];receptacles.add(location);visited.add(location)
        for m in re.finditer(r'the ('+NAME+r') is (open|closed)',low):open_state[m[1]]=m[2]=='open'
        for m in re.finditer(r'(?:on the ('+NAME+r'), you see|the ('+NAME+r') is open\. in it, you see) ([^.]+)',low):
            rec=m[1] or m[2]
            for obj in re.findall(NAME,m[3]):objects[obj]=rec
        for m in re.finditer(r'you pick up the ('+NAME+r')(?: from the ('+NAME+r'))?',low):
            held.add(m[1]);objects.pop(m[1],None)
            if m[2]:location=m[2]
        for m in re.finditer(r'you (?:move|put) the ('+NAME+r') (?:to|in/on|in) the ('+NAME+r')',low):
            held.discard(m[1]);objects[m[1]]=m[2];location=m[2]
            if m[2].rsplit(' ',1)[0]==goal['destination']:placed.add(m[1])
        if 'you are not carrying anything' in low:held.clear()
        inv=re.search(r'you are carrying: ([^.]+)',low)
        if inv:held=set(re.findall(NAME,inv[1]))
        for m in re.finditer(r'you (heat|clean|cool|heated|cleaned|cooled) the ('+NAME+r')',low):processed.add((m[2],{'heat':'heat','clean':'clean','cool':'cool','heated':'heat','cleaned':'clean','cooled':'cool'}[m[1]]))
    def matching(names,kind):return sorted(x for x in names if x.rsplit(' ',1)[0]==kind)
    def navigate(rec,action):
        if location!=rec:return 'go to '+rec
        if open_state.get(rec) is False:return 'open '+rec
        return action
    carrying=matching(held,goal['object'])
    if carrying:
        obj=carrying[0];op=goal['operation']
        if op in ('heat','clean','cool') and (obj,op) not in processed:
            device={'heat':'microwave','clean':'sinkbasin','cool':'fridge'}[op];dest=matching(receptacles,device)
            if dest:return navigate(dest[0],op+' '+obj+' with '+dest[0])
        elif op=='examine':
            targets=matching(objects,goal['destination'])
            if targets:
                lamp=targets[0];rec=objects[lamp]
                return navigate(rec,'use '+lamp)
        else:
            dest=matching(receptacles,goal['destination'])
            if dest:return navigate(dest[0],'move '+obj+' to '+dest[0])
    targets=[x for x in matching(objects,goal['object']) if x not in placed]
    if not carrying and targets:
        obj=targets[0];rec=objects[obj];return navigate(rec,'take '+obj+' from '+rec)
    if location and open_state.get(location) is False:return 'open '+location
    unvisited=sorted(receptacles-visited)
    if unvisited:return 'go to '+unvisited[0]
    known=sorted(receptacles)
    return 'go to '+known[step%len(known)] if known else 'look'


def project_legacy_action(action):
    """Fixed offline-label spelling map, never applied to model generations."""
    if type(action) is not str or not action:raise ValueError('complete course action required')
    m=re.fullmatch(r'put ('+NAME+r') in/on ('+NAME+r')',action)
    return 'move '+m[1]+' to '+m[2] if m else action


def course_documents(path,*,expected_sha256,fit_permits):
    """Yield all40594 fixed offline rows without leaking structured metadata.

    Repeated stored context is retained and counted as stale. We do not call it
    a current actual environment observation or infer a missing transition.
    """
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=_sha(expected_sha256):raise ValueError('fixed FIT course changed')
    documents=[];provenance=[];stale=0;converted=0;seen=set()
    for line_index,line in enumerate(data.splitlines()):
        row=json.loads(line);task=row['task_id'];permit=fit_permits.get(task)
        if (permit is None or permit.role!='FIT' or row['role']!='FIT' or row['floor_plan']!=permit.floor
                or row['game_sha256']!=permit.game_sha256 or task in seen):raise ValueError('FIT-only original source identity required')
        seen.add(task);raw=row['raw_teacher_row']
        if set(raw)!={'g','g_id','steps','task'} or not isinstance(raw['task'],str) or not raw['task']:
            raise ValueError('exact offline schema/public goal required')
        previous=None
        for j,step in enumerate(raw['steps']):
            if set(step)!={'action','obs','step_id'} or step['step_id']!=j:raise ValueError('fixed sequential offline row required')
            parts=step['obs'].split(' [SEP] ')
            if len(parts)%2!=1 or (j==0 and len(parts)!=1) or (j>0 and len(parts)<3):raise ValueError('ambiguous stored observation schema')
            if previous is not None and parts[-1]!=previous['action']:
                if step['obs']!=previous['obs']:raise ValueError('unexplained stored history discontinuity')
                stale+=1
            observation=parts[-2] if len(parts)>1 else parts[0]
            action=project_legacy_action(step['action']);converted+=int(action!=step['action'])
            # Strings only. No hidden g/g_id, source path, physical/task id,
            # reward, future step or raw document enters the feature builder.
            documents.append((raw['task'],observation,action))
            provenance.append((row['archive_member'],row['source_row_index'],j))
            previous=step
    if len(set(provenance))!=len(provenance):raise ValueError('duplicate original course provenance')
    if len(documents)!=40594 or len(seen)!=1969:raise ValueError('all original course records required')
    return tuple(documents),dict(schema=BASELINE_SCHEMA,course=expected_sha256,tasks=len(seen),documents=len(documents),
        stale_stored_contexts=stale,legacy_labels_projected=converted,provenance_identity=_digest(provenance),
        document_identity=_digest(documents),context='last stored observation, not verified current-world feedback')


def _terms(goal,observation):
    # Fixed unigram TF-IDF. Goal factor2 in coordinates, hence4 in dot product.
    out={}
    for prefix,text,weight in (('g:',goal,2.),('o:',observation,1.)):
        counts=Counter(re.findall(r'[a-z0-9_]+',text.lower()))
        for token,count in counts.items():out[prefix+token]=weight*(1+math.log(count))
    return out

class FITRetrieval:
    def __init__(self,documents,*,corpus_identity):
        _sha(corpus_identity)
        if not documents or _digest(documents)!=corpus_identity:raise ValueError('exact nonempty FIT index content identity required')
        self.actions=tuple(row[2] for row in documents);n=len(documents);df=Counter()
        for goal,obs,_ in documents:df.update(_terms(goal,obs).keys())
        self.idf={t:1+math.log((1+n)/(1+c)) for t,c in df.items()};postings=defaultdict(list)
        for i,(goal,obs,_) in enumerate(documents):
            vector={t:x*self.idf[t] for t,x in _terms(goal,obs).items()};norm=math.sqrt(sum(x*x for x in vector.values()))
            for t,x in vector.items():postings[t].append((i,x/norm))
        self.postings=dict(postings);self.identity=_digest(dict(schema=BASELINE_SCHEMA,corpus=corpus_identity,
            projection='public-goal-last-stored-observation-legacy-put-v1',method='fielded-unigram-logTF-smoothedIDF-cosine; goal-coordinate2; first-original-row-tie',documents=n))
    def __call__(self,view):
        if type(view) is not BaselineView:raise TypeError('observable-only retrieval query required')
        text='\n'.join((*view.passages(),view.frame.observation));terms=_terms(view.frame.goal,text)
        vector={t:x*self.idf[t] for t,x in terms.items() if t in self.idf};norm=math.sqrt(sum(x*x for x in vector.values()))
        if not norm:return 'look'
        scores=defaultdict(float)
        for t,x in vector.items():
            for i,y in self.postings[t]:scores[i]+=(x/norm)*y
        winner=min(scores,key=lambda i:(-scores[i],i))
        return self.actions[winner]

class ReferenceFormatter:
    """Own original chat template, identical public content, no truncation."""
    def __init__(self,tokenizer,*,identity,eos_ids,max_tokens=4096,max_new_tokens=64):
        _sha(identity)
        if not tokenizer.chat_template or not eos_ids or max_tokens!=4096 or max_new_tokens!=64:raise ValueError('fixed reference prompt contract required')
        self.tokenizer=tokenizer;self.controls=_controls(tokenizer);self.eos=tuple(sorted(set(eos_ids)))
        self.template=_digest(tokenizer.chat_template);self.identity=_digest(dict(schema=BASELINE_SCHEMA,tokenizer=identity,
            template=self.template,eos=self.eos,context=4096,new_tokens=64,thinking=False,instruction=ACTION_INSTRUCTION))
    def prompt(self,view):
        if type(view) is not BaselineView or self.template!=_digest(self.tokenizer.chat_template) or self.controls!=_controls(self.tokenizer):
            raise ValueError('bound reference view/template required')
        view.validate()
        payload=view.frame.prompt_payload();payload['retained_historical_observation_fragments']=[json.loads(b) for b in view.fragments]
        body=ACTION_INSTRUCTION+json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        if any(x in body for x in self.controls):raise ValueError('literal role-control text in public payload')
        ids=self.tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if type(ids) not in (tuple,list) or not ids or any(type(t) is not int or t<0 for t in ids):
            raise ValueError('flat integer reference token IDs required')
        ids=tuple(ids)
        if not ids or len(ids)+64>4096:raise ValueError('complete reference prompt overflow; no truncation')
        return ids
