"""Logical placements in an unchanged ALFWorld PDDL task specification.

This module does not import an environment, execute a planner, or use a model.
Compilation establishes a finite structural domain, not task solvability or a
physical THOR scene. Actual generated worlds require separate native validation.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re

SCHEMA = 'orbit-alfworld-logical-placement-domain-v1'
DOMAIN_SHA256 = '0af55e7f92d12b09d28515aa730e4d03fddd17228ef1fd7b18a3ee8ba49b27b9'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def parse(text):
    if type(text) is not str or len(text.encode()) > 4*1024*1024:
        raise ValueError('bounded complete PDDL text required')
    tokens=re.findall(r'\(|\)|[^\s()]+',re.sub(r';[^\n]*','',text))
    stack=[];roots=[]
    for token in tokens:
        if token=='(':
            if len(stack)>=128:raise ValueError('bounded PDDL depth')
            value=[]
            if stack:stack[-1].append(value)
            else:roots.append(value)
            stack.append(value)
        elif token==')':
            if not stack:raise ValueError('unbalanced PDDL')
            stack.pop()
        else:
            if not stack:raise ValueError('PDDL atom outside expression')
            stack[-1].append(token)
    if stack or len(roots)!=1 or not roots[0] or roots[0][0]!='define':
        raise ValueError('one complete PDDL definition required')
    return roots[0]


def render(tree):
    if isinstance(tree,str):return tree
    return '('+' '.join(render(x) for x in tree)+')'


def sections(tree):
    out={}
    for row in tree[1:]:
        if type(row) is not list or not row or type(row[0]) is not str or row[0] in out:
            raise ValueError('unique PDDL sections required')
        out[row[0]]=row
    if set(out)!={'problem',':domain',':objects',':init',':goal'}:
        raise ValueError('exact supported original problem sections required')
    return out


def atoms(tree,predicate):
    if type(tree) is not list:return []
    out=[tree] if tree and tree[0]==predicate else []
    for child in tree:
        if type(child) is list:out.extend(atoms(child,predicate))
    return out


def canonical_problem(text):
    """Ignore cosmetic problem IDs, fact order, whitespace and comments."""
    s=sections(parse(text))
    # Object declarations retain their exact original typed symbols. No object
    # or receptacle is renamed by the sampler.
    declared={};pending=[];items=s[':objects'][1:];i=0
    while i<len(items):
        atom=items[i]
        if type(atom) is not str:raise ValueError('flat typed declarations required')
        if atom=='-':
            if not pending or i+1>=len(items) or type(items[i+1]) is not str:
                raise ValueError('complete typed declarations required')
            typ=items[i+1]
            for name in pending:
                if name in declared and declared[name]!=typ:raise ValueError('object type conflict')
                declared[name]=typ
            pending=[];i+=2
        else:pending.append(atom);i+=1
    if pending:raise ValueError('all declared objects must be typed')
    return dict(domain=s[':domain'],objects=sorted(declared.items()),
        init=sorted(s[':init'][1:],key=render),goal=s[':goal'])


@dataclass(frozen=True)
class PlacementVariable:
    object: str
    object_type: str
    original_receptacle: str
    original_location: str
    destinations: tuple[tuple[str,str],...]


@dataclass(frozen=True)
class PlacementDomain:
    game_bytes: bytes
    source_sha256: str
    variables: tuple[PlacementVariable,...]
    goal_types: tuple[str,...]
    initial_semantics: str
    size: int

    @property
    def identity(self):
        from dataclasses import asdict
        return digest(dict(schema=SCHEMA,source=self.source_sha256,
            variables=[asdict(x) for x in self.variables],goal_types=self.goal_types,
            initial_semantics=self.initial_semantics,size=self.size))


def _pairs(facts,name):
    rows=[x for x in facts if x and x[0]==name]
    if any(len(x)!=3 or not all(type(a) is str for a in x) for x in rows):
        raise ValueError('flat binary predicate required: '+name)
    return rows


def _unique(rows,name):
    out={}
    for _,key,value in rows:
        if key in out and out[key]!=value:raise ValueError('ambiguous '+name)
        out[key]=value
    return out


def compile_domain(game_bytes, *, expected_sha256):
    if type(game_bytes) is not bytes or hashlib.sha256(game_bytes).hexdigest()!=expected_sha256:
        raise ValueError('exact original FIT game bytes required')
    game=json.loads(game_bytes)
    if (set(game)!={'pddl_domain','pddl_problem','grammar','solvable','walkthrough'}
            or hashlib.sha256(game['pddl_domain'].encode()).hexdigest()!=DOMAIN_SHA256):
        raise ValueError('unchanged original ALFWorld logical domain required')
    s=sections(parse(game['pddl_problem']));facts=s[':init'][1:]
    if any(type(x) is not list or not x for x in facts):raise ValueError('complete initial facts')
    # Flat objects only. Moving a container with live contents would require a
    # separate joint placement model; leave every participant unchanged here.
    nested={a for x in _pairs(facts,'inReceptacleObject') for a in x[1:]}
    held={x[2] for x in _pairs(facts,'holds')}
    object_types=_unique(_pairs(facts,'objectType'),'object types')
    receptacle_types=_unique(_pairs(facts,'receptacleType'),'receptacle types')
    object_locations=_unique(_pairs(facts,'objectAtLocation'),'object locations')
    receptacle_locations=_unique(_pairs(facts,'receptacleAtLocation'),'receptacle locations')
    parents={}
    for _,obj,rec in _pairs(facts,'inReceptacle'):parents.setdefault(obj,set()).add(rec)
    legal={(x[1],x[2]) for x in _pairs(facts,'canContain')}
    pickupable={x[1] for x in facts if len(x)==2 and x[0]=='pickupable'}
    goal_types=tuple(sorted({x[2] for x in atoms(s[':goal'],'objectType')}))
    variables=[]
    for obj in sorted(pickupable-nested-held):
        if obj not in object_types or obj not in object_locations or len(parents.get(obj,()))!=1:continue
        original=next(iter(parents[obj]))
        if original not in receptacle_locations:continue
        destinations=tuple(sorted((rec,loc) for rec,loc in receptacle_locations.items()
            if (receptacle_types.get(rec),object_types[obj]) in legal))
        if len(destinations)<2:continue
        variables.append(PlacementVariable(obj,object_types[obj],original,object_locations[obj],destinations))
    if not any(x.object_type in goal_types for x in variables):
        raise ValueError('no task-relevant object has multiple legal initial placements')
    return PlacementDomain(game_bytes,expected_sha256,tuple(variables),goal_types,
        digest(canonical_problem(game['pddl_problem'])),math.prod(len(x.destinations) for x in variables))


def instantiate(domain, choices):
    """Apply explicit choices, without random draws, validation or publication.

    The caller records independently sampled choices under a frozen contract.
    Domain rules, task goal, language, identities and all non-placement facts
    stay unchanged. A native validator must replay a new plan; old walkthrough
    and solvable flags are deliberately absent from the returned game.
    """
    if type(domain) is not PlacementDomain or type(choices) is not tuple or len(choices)!=len(domain.variables):
        raise ValueError('one complete registered placement vector required')
    selected={}
    for variable,index in zip(domain.variables,choices):
        if type(index) is not int or not 0<=index<len(variable.destinations):
            raise ValueError('finite destination index required')
        selected[variable.object]=variable.destinations[index]
    if compile_domain(domain.game_bytes,expected_sha256=domain.source_sha256)!=domain:
        raise ValueError('compiled placement domain differs from exact source')
    source=json.loads(domain.game_bytes);tree=parse(source['pddl_problem']);s=sections(tree)
    changed=[]
    for fact in s[':init'][1:]:
        if len(fact)==3 and fact[0] in ('inReceptacle','objectAtLocation') and fact[1] in selected:
            rec,loc=selected[fact[1]]
            fact[2]=rec if fact[0]=='inReceptacle' else loc
    problem=render(tree);semantic=digest(canonical_problem(problem))
    for variable in domain.variables:
        if selected[variable.object]!=(variable.original_receptacle,variable.original_location):
            changed.append(variable.object)
    relevant=[x.object for x in domain.variables if x.object in changed and x.object_type in domain.goal_types]
    if not relevant or semantic==domain.initial_semantics:
        raise ValueError('a task-relevant initial placement must actually change')
    return dict(pddl_domain=source['pddl_domain'],pddl_problem=problem,grammar=source['grammar']),dict(
        schema=SCHEMA,source=domain.source_sha256,domain=domain.identity,choices=choices,
        semantics=semantic,moved_objects=tuple(changed),moved_task_objects=tuple(relevant),
        native_validated=False,solvability_established=False)
