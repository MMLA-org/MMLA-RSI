"""Finite FIT-only manager curriculum. No model, environment or file I/O.

The grammar bank is an offline data instrument, never a proof of improved RSI.
No outcome is used to select source tasks. All model task scores must be real
terminal outcomes before a target is constructed.
"""
from recipe import EXAMPLES,PROGRAMS
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from itertools import combinations

SALT = 'orbit-task-outcome-proposal-v1'
SEEDS = (17, 29, 43)
TYPES = ('look_at_obj_in_light', 'pick_and_place_simple',
         'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
         'pick_heat_then_place_in_recep', 'pick_two_obj_and_place')
STRATA = ((0,TYPES[1]),(0,TYPES[2]),(0,TYPES[3]),(0,TYPES[4]),(0,TYPES[5]),
          (2,TYPES[0]),(2,TYPES[1]),(2,TYPES[5]),
          (3,TYPES[0]),(3,TYPES[1]),(3,TYPES[5]),
          (4,TYPES[1]),(4,TYPES[2]),(4,TYPES[5]))
PARENT_NAME = 'parent'
PARENT_SYMBOLS = ('add','sub','c1','x0','add','sub','c1','x1','x6')
from content_rules import Language, parse_rule
PINNED_LANGUAGE = Language(max_nodes=31,max_tokens=31,max_depth=3,max_instructions=31)
PARENT_IDENTITY = parse_rule(PINNED_LANGUAGE,'rtma',PARENT_SYMBOLS).identity

# Names describe algebra, not established efficacy. NULL remains legal.
from content_features import BANK

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def _hash(value):
    return hashlib.sha256((SALT+':'+value).encode()).hexdigest()

def select_fit(metadata):
    """Four distinct floors per existing stratum; first two support, last two query.

    Both subsets remain FIT. They are training roles, not independent audits.
    Full source permissions and raw bytes are verified by the later host.
    """
    if type(metadata) is not list or not metadata:
        raise ValueError('complete original partition metadata required')
    ids=[r['task_id'] for r in metadata]
    if len(set(ids))!=len(ids):
        raise ValueError('duplicate task identity in source metadata')
    fit=[r for r in metadata if r['role']=='FIT']
    if len(fit)!=1969 or len({r['floor_plan'] for r in fit})!=60:
        raise ValueError('exact original FIT support expected')
    strata=defaultdict(list)
    for r in fit:
        if r.get('eligible') is not True or r['reserved_floor_role']!='FIT':
            raise ValueError('eligible original FIT permissions required')
        if type(r['scene_num']) is not int or r['floor_plan']!='FloorPlan'+str(r['scene_num']):
            raise ValueError('source floor identity mismatch')
        strata[(r['scene_num']//100,r['task_type'])].append(r)
    if set(strata)!=set(STRATA):
        raise ValueError('source task strata changed')
    chosen=[];groups=[]
    for bucket,kind in STRATA:
        pool=sorted(strata[(bucket,kind)],key=lambda r:(_hash(r['task_id']),r['task_id']))
        floors=set();members=[]
        for r in pool:
            if r['floor_plan'] in floors:continue
            floors.add(r['floor_plan'])
            members.append({k:r[k] for k in ('task_id','floor_plan','scene_num','task_type',
                'role','game_sha256','trajectory_sha256','physical_layout_sha256','task_params_sha256')})
            if len(members)==4:break
        if len(members)!=4:raise ValueError('four existing FIT floors unavailable; no fallback')
        group=dict(stratum=[bucket,kind],support=members[:2],query=members[2:])
        groups.append(group);chosen.extend(members)
    for key in ('task_id','game_sha256','trajectory_sha256'):
        values=[r[key] for r in chosen]
        if any(type(x) is not str or not x for x in values) or len(values)!=len(set(values)):
            raise ValueError('duplicate or missing selected identity: '+key)
        outside={r.get(key) for r in metadata if r['role']!='FIT'}-{None}
        if set(values)&outside:raise ValueError('FIT/non-FIT identity overlap: '+key)
    result=dict(schema='orbit-outcome-proposal-fit-selection-v1',selection_salt=SALT,
        source_fit_tasks=len(fit),groups=groups,tasks=len(chosen),
        task_type_counts=dict(sorted(Counter(r['task_type'] for r in chosen).items())),
        distinct_floors=len({r['floor_plan'] for r in chosen}),
        support_query_are_training_only=True,independence_claim=False,
        selection_uses_outcomes=False)
    result['identity']=digest(result)
    return result

def bank_rules(language, *, actual_parent):
    from content_rules import parse_rule
    rules={name:parse_rule(language,'rtma',symbols) for name,symbols in BANK}
    if language != PINNED_LANGUAGE or len(rules)!=PROGRAMS or len({r.identity for r in rules.values()})!=PROGRAMS:
        raise ValueError('four distinct canonical content programs required')
    bind_parent(rules,actual_parent)
    return rules

def bind_parent(rules,actual_parent):
    """The host passes DemonstrationFactory.manager, not a label or guessed rule."""
    parent=rules.get(PARENT_NAME)
    if (parent is None or parent!=actual_parent or parent.symbols!=PARENT_SYMBOLS
            or parent.mode!='rtma' or parent.identity!=PARENT_IDENTITY):
        raise ValueError('actual original task manager wire and identity required')
    return parent

def checked_table(selection, records, *, seed, rules, reader_identity):
    """Reject incomplete, foreign, repeated or unscored terminal records.

    Each record carries one complete task failure and a separately metered
    token-cost proxy. Operational failures cannot be represented as task losses.
    """
    if seed not in SEEDS or type(records) is not list or len(rules)!=PROGRAMS:
        raise ValueError('fixed reader and full bank required')
    tasks={r['task_id']:r for g in selection['groups'] for side in ('support','query') for r in g[side]}
    if len(tasks)!=56:raise ValueError('exact complete FIT set required')
    result={}
    for r in records:
        if set(r)!={'seed','reader','rule','task_id','game_sha256','role','status','won',
                     'loss','cost','endpoint_sha256'}:
            raise ValueError('exact endpoint schema required')
        task=tasks.get(r['task_id']); key=(r['rule'],r['task_id'])
        if (r['seed']!=seed or r['reader']!=reader_identity or task is None or r['role']!='FIT'
                or r['game_sha256']!=task['game_sha256'] or r['status']!='COMPLETE'
                or type(r['won']) is not bool or type(r['loss']) not in (float,int)
                or r['loss']!=1.-float(r['won']) or key in result
                or r['rule'] not in {v.identity for v in rules.values()}
                or type(r['cost']) not in (float,int) or not math.isfinite(r['cost']) or not 0<=r['cost']<=1
                or type(r['endpoint_sha256']) is not str or len(r['endpoint_sha256'])!=64):
            raise ValueError('terminal evidence incomplete, changed or duplicated')
        try:int(r['endpoint_sha256'],16)
        except ValueError:raise ValueError('endpoint hash required') from None
        result[key]=dict(r)
    expected={(rule.identity,t) for rule in rules.values() for t in tasks}
    if set(result)!=expected:raise ValueError('all four x 56 outcomes required; no seed/cell deletion')
    return result

def mean_score(table, rule, tasks):
    rows=[table[rule.identity,t['task_id']] for t in tasks]
    return sum(r['loss'] for r in rows)/len(rows),sum(r['cost'] for r in rows)/len(rows)

def fit_opportunity(selection,table,rules,*,actual_parent,examples):
    """Require improvement both in the bank and in the emitted target mixture.

    This checks training opportunity, not model performance or RSI admission.
    Every example remains in the curriculum; no outcome-based row deletion.
    """
    bind_parent(rules,actual_parent)
    tasks=[r for g in selection['groups'] for side in ('support','query') for r in g[side]]
    loss={name:mean_score(table,rule,tasks)[0] for name,rule in rules.items()}
    if type(examples) is not list or len(examples)!=EXAMPLES:
        raise ValueError('complete actual target distribution required before training gate')
    for row in examples:
        if (row.get('target_name') not in rules
                or row.get('target')!=rules[row['target_name']].identity
                or row.get('identity')!=digest({k:v for k,v in row.items() if k!='identity'})):
            raise ValueError('changed target distribution')
    gain=loss[PARENT_NAME]-min(loss.values())
    target_loss=sum(loss[r['target_name']] for r in examples)/len(examples)
    target_gain=loss[PARENT_NAME]-target_loss
    return dict(status='FIT_OPPORTUNITY' if min(gain,target_gain)>=.02 else 'NO_LABELLED_FIT_OPPORTUNITY',
                parent=actual_parent.identity,mean_losses=loss,best_gain=gain,required_gain=.02,
                actual_target_counts=dict(sorted(Counter(r['target_name'] for r in examples).items())),
                actual_target_mixture_global_loss=target_loss,actual_target_mixture_gain=target_gain,
                units='56 fixed FIT tasks; balanced fourteen strata; descriptive training gate',
                independent_benefit_claim=False)

def mixed_views(selection):
    """Fourteen complete-mixture FIT views, in seven complementary pairs.

    Each view has two support and two query tasks in every one of the fourteen
    strata. Each physical task occurs in exactly seven support/query views.
    Views reuse the same 56 outcomes and are not independent experiments.
    """
    views=[dict(index=i,support=[],query=[]) for i in range(14)]
    for gi,group in enumerate(selection['groups']):
        members=group['support']+group['query']
        options=list(combinations(range(4),2))
        random.Random(int(_hash('mixed-view:'+str(gi)),16)).shuffle(options)
        for pair in range(7):
            # Two base-six digits distinguish the seven base views. A fixed
            # last stratum separates all base views from their complements.
            choice=(pair%6 if gi==0 else pair//6 if gi==1 else 0 if gi==13
                    else int(_hash(f'mixed-view:{gi}:{pair}'),16)%6)
            selected=set(options[choice])
            for parity in (0,1):
                take=selected if parity==0 else set(range(4))-selected
                view=views[2*pair+parity]
                view['support'].extend(members[j] for j in range(4) if j in take)
                view['query'].extend(members[j] for j in range(4) if j not in take)
    inventories=set()
    for view in views:
        s={t['task_id'] for t in view['support']};q={t['task_id'] for t in view['query']}
        if len(s)!=28 or len(q)!=28 or s&q:
            raise ValueError('complete disjoint mixed training view required')
        inventories.add(tuple(sorted(s)))
        view['identity']=digest(view)
    if len(inventories)!=14:raise ValueError('fixed distinct mixed views required; no fallback')
    for side in ('support','query'):
        counts=Counter(t['task_id'] for v in views for t in v[side])
        if len(counts)!=56 or set(counts.values())!={7}:
            raise ValueError('balanced complete-mixture exposure required')
    return views

def conditional_risk_diagnostics(selection,table,rules,examples):
    """Descriptive FIT supervision content, not an additional acceptance gate."""
    if len(examples)!=EXAMPLES or any(r['target']!=rules[r['target_name']].identity for r in examples):
        raise ValueError('complete actual emitted targets required')
    views=mixed_views(selection)
    scores={(v['index'],name):mean_score(table,rule,v['query'])[0]
            for v in views for name,rule in rules.items()}
    def risk(groups):
        total=0.
        for rows in groups.values():
            total+=min(sum(scores[r['view'],name] for r in rows) for name in rules)
        return total/len(examples)
    headers=defaultdict(list);full=defaultdict(list)
    for r in examples:
        headers[r['parent']].append(r);full[tuple(r['prompt'])].append(r)
    h=risk(headers);m=risk(full)
    targets={name:sum(r['target_name']==name for r in examples) for name in rules}
    return dict(schema='orbit-content-conditional-supervision-v1',header_only_query_risk=h,
        header_plus_M_query_risk=m,conditional_memory_information_increment=h-m,
        target_counts=targets,constant_global_target=sum(n>0 for n in targets.values())==1,
        unique_headers=len(headers),unique_complete_prompts=len(full),
        cost_ties_do_not_change_risk=True,changes_training_gate=False,
        evidence='reused FIT mixtures; diagnostic only; not learned dependence or independent benefit')

def context_order(seed,view_index,parent_name,order_index):
    if seed not in SEEDS or not 0<=view_index<14 or parent_name not in dict(BANK) or not 0<=order_index<4:
        raise ValueError('fixed training order domain required')
    values=[name for name,_ in BANK]
    random.Random(int(_hash(f'{seed}:{view_index}:{parent_name}:{order_index}'),16)).shuffle(values)
    return tuple(values[:3])

def build_examples(selection,table,*,seed,rules,make_memory,make_header,tokenize,codec,language,context_limit):
    """Build 896 teacher-forced examples, exclusively from FIT outcomes.

    A current parent V alone retains the three arrived support evaluations.
    Feedback covers the complete mixed support block, as deployment does.
    Identical visible inputs share a target minimizing mean query risk across
    their actual contexts, never the frequency of hidden-stratum winners.
    Query outcomes select a complete target; they never enter the prompt/M.
    The original run's 4-opportunity and 2-slot semantics are retained.
    """
    from content_evidence import EvidenceMemory,Record
    from content_rules import PrefixGrammar
    examples=[];views=mixed_views(selection)
    for vi,view in enumerate(views):
        for parent_name,parent in rules.items():
            header=make_header(parent)
            for order_index in range(4):
                memory=make_memory()
                if type(memory) is not EvidenceMemory or memory.version or any(memory.rows) or memory.bounds.slots!=2:
                    raise ValueError('fresh two-slot original evidence memory required')
                order=context_order(seed,vi,parent_name,order_index)
                for attempt in range(4):
                    text,read=memory.read()
                    prompt=tuple(tokenize(header+'\nMEMORY\n'+text+'\nPROGRAM\n'))
                    example=dict(schema='orbit-task-outcome-proposal-example-v2',seed=seed,
                        view=vi,view_identity=view['identity'],parent=parent.identity,order=order_index,attempt=attempt,
                        prompt=list(prompt),memory=memory.export(),memory_read=read,
                        source_roles=['FIT_SUPPORT','FIT_QUERY'],future_scores_in_prompt=False,
                        target_semantics='minimum conditional complete-mixture query risk for identical visible input; cost then canonical symbols',
                        offline_demonstration=True,recursive_benefit_claim=False)
                    examples.append(example)
                    if attempt==3:break
                    scored=rules[order[attempt]];loss,cost=mean_score(table,scored,view['support'])
                    record=Record(' '.join(scored.symbols),loss,cost,'completed',
                        memory.source_prefix+'proposal/'+str(attempt),attempt)
                    memory,receipt=memory.manage(record,language=language,rule=parent,opportunity_cap=4)
    if len(examples)!=EXAMPLES:raise ValueError('complete fixed curriculum required')
    visible=defaultdict(list)
    for row in examples:visible[tuple(row['prompt'])].append(row)
    query_scores={(v['index'],name):(
        sum(table[rule.identity,t['task_id']]['loss'] for t in v['query']),
        mean_score(table,rule,v['query'])[1]) for v in views for name,rule in rules.items()}
    for equivalent in visible.values():
        scores={name:(sum(query_scores[row['view'],name][0] for row in equivalent)/(28*len(equivalent)),
                      sum(query_scores[row['view'],name][1] for row in equivalent)/len(equivalent))
                for name in rules}
        # Preserve exact empirical task-risk ordering for indistinguishable
        # inputs; there is no weaker target chosen to manufacture a second edge.
        name=min(rules,key=lambda n:(*scores[n],tuple(rules[n].symbols),n))
        target=rules[name];grammar=PrefixGrammar(language,'rtma');tokens=[];masks=[]
        for symbol in target.symbols:
            masks.append(tuple(codec[s] for s in grammar.allowed()))
            tokens.append(codec[symbol]);grammar.append(symbol)
        if grammar.finish()!=target:raise ValueError('complete canonical target required')
        for row in equivalent:
            if not row['prompt'] or len(row['prompt'])+len(tokens)>context_limit:
                raise ValueError('whole prompt/target required; no truncation')
            row.update(tokens=tokens.copy(),masks=[list(x) for x in masks],target=target.identity,
                       target_name=name,conditional_query_loss=scores[name][0],
                       equivalent_visible_contexts=len(equivalent))
            row['identity']=digest(row)
    return examples
