"""Conservative logical-world exclusion, including object/variable renaming.

The fingerprint is a typed relational-graph invariant, not a complete graph
isomorphism solver. Isomorphic supported graphs have equal fingerprints;
non-isomorphic collisions are possible and conservatively stop the run. Never
interpret a collision as permission to replace a draw or change the threshold.
Domain semantics are fixed. Arbitrary logical equivalence is not claimed.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile
from placement_domain import parse, sections, render, digest

SCHEMA = 'orbit-logical-exclusion-graph-v1'


def typed(items):
    result={};pending=[];i=0
    while i<len(items):
        token=items[i]
        if type(token) is not str:raise ValueError('flat typed declarations required')
        token=token.lower()
        if token=='-':
            if not pending or i+1>=len(items) or type(items[i+1]) is not str:
                raise ValueError('complete type binding required')
            typ=items[i+1].lower()
            for name in pending:
                if name in result and result[name]!=typ:raise ValueError('conflicting object types')
                result[name]=typ
            pending=[];i+=2
        else:pending.append(token);i+=1
    if pending:raise ValueError('all symbols need an explicit type')
    return result


def lower(tree):
    return tree.lower() if type(tree) is str else [lower(x) for x in tree]


def normalize_goal(tree, bound=None, depth=0):
    """Alpha-normalize local binders and flatten/deduplicate and/or clauses."""
    bound={} if bound is None else bound
    if type(tree) is str:return bound.get(tree,tree)
    if not tree or type(tree[0]) is not str:raise ValueError('typed goal formula required')
    op=tree[0]
    if op in ('exists','forall'):
        if len(tree)!=3 or type(tree[1]) is not list:raise ValueError('bounded quantifier body required')
        declarations=typed(tree[1]);local=dict(bound);decl=[]
        for i,(name,typ) in enumerate(declarations.items()):
            if not name.startswith('?'):raise ValueError('bound variable required')
            renamed=f'?bound{depth}_{i}';local[name]=renamed;decl.extend((renamed,'-',typ))
        body=normalize_goal(tree[2],local,depth+1)
        # Adjacent same-kind quantifiers commute; names are already unique
        # across nesting depths, including shadowed original variables.
        while type(body) is list and body and body[0]==op:
            decl.extend(body[1]);body=body[2]
        return [op,decl,body]
    values=[normalize_goal(x,bound,depth) for x in tree[1:]]
    if op in ('and','or'):
        flat=[]
        for value in values:
            flat.extend(value[1:] if type(value) is list and value and value[0]==op else [value])
        values=[v for _,v in sorted({render(v):v for v in flat}.items())]
        if len(values)==1:return values[0]
    return [op,*values]


def semantic_fingerprint(game):
    if (type(game) is not dict or not {'pddl_domain','pddl_problem','grammar'}<=set(game)
            or set(game)-{'pddl_domain','pddl_problem','grammar','walkthrough','solvable'}):
        raise ValueError('bounded known game schema required')
    s=sections(lower(parse(game['pddl_problem'].lower())));objects=typed(s[':objects'][1:])
    colors=[];edges=[];objects_by_name={};constants={}
    def node(color):
        if len(colors)>=100000:raise ValueError('bounded semantic graph exceeded')
        colors.append(digest(color));edges.append([]);return len(colors)-1
    def edge(a,b,label):
        edges[a].append((label,b));edges[b].append(('inverse/'+label,a))
    root=node(('problem',s[':domain']))
    for name,typ in objects.items():
        # otype/rtype constants denote semantic categories, not instance IDs.
        color=('object',typ,name) if typ in ('otype','rtype') else ('object',typ)
        objects_by_name[name]=node(color);edge(root,objects_by_name[name],'declared')
    def symbol(value,bindings):
        if type(value) is not str:raise ValueError('nested functional terms unsupported')
        if value.startswith('?'):
            if value not in bindings:raise ValueError('unbound goal variable')
            return bindings[value]
        if value in objects_by_name:return objects_by_name[value]
        if value not in constants:constants[value]=node(('constant',value))
        return constants[value]
    def atom(value,bindings):
        if not value or type(value[0]) is not str:raise ValueError('predicate required')
        n=node(('predicate',value[0],len(value)-1))
        for i,x in enumerate(value[1:]):
            edge(n,symbol(x,bindings),'arg' if value[0]=='=' else 'arg/'+str(i))
        return n
    facts={render(x):x for x in s[':init'][1:]}
    for fact in facts.values():edge(root,atom(fact,{}),'init')
    def formula(value,bindings):
        if type(value) is not list or not value:raise ValueError('goal formula required')
        op=value[0]
        if op in ('exists','forall'):
            n=node(('quantifier',op));local=dict(bindings)
            for name,typ in typed(value[1]).items():
                v=node(('variable',typ));local[name]=v;edge(n,v,'binds')
            edge(n,formula(value[2],local),'body');return n
        if op in ('and','or'):
            n=node(('connective',op))
            for x in value[1:]:edge(n,formula(x,bindings),'clause')
            return n
        if op=='not':
            if len(value)!=2:raise ValueError('one negated body required')
            n=node(('connective','not'));edge(n,formula(value[1],bindings),'body');return n
        return atom(value,bindings)
    if len(s[':goal'])!=2:raise ValueError('one goal formula required')
    edge(root,formula(normalize_goal(s[':goal'][1]),{}),'goal')
    if sum(map(len,edges))>1000000:raise ValueError('bounded semantic graph edges exceeded')
    # Refinement never merges color classes. Equal class counts mean the
    # partition has stabilized; isomorphic graphs stop on the same round.
    for _ in range(len(colors)+1):
        updated=[digest((colors[i],sorted((label,colors[j]) for label,j in links)))
                 for i,links in enumerate(edges)]
        stable=len(set(updated))==len(set(colors));colors=updated
        if stable:break
    else:raise RuntimeError('semantic refinement did not stabilize')
    return digest(dict(schema=SCHEMA,domain=hashlib.sha256(game['pddl_domain'].encode()).hexdigest(),
        colors=sorted(Counter(colors).items()),
        edges=sorted(Counter((colors[i],label,colors[j]) for i,links in enumerate(edges)
                            for label,j in links).items())))


@contextmanager
def historical_source(root, *, archive=None, expected_archive_sha256=None):
    """Decode only partition-bound members of one verified original source.

    The archive is read through the same open descriptor that was hashed. No
    extraction, alternate revision, omission or fallback to another source.
    """
    cap=4*1024**2
    def member_name(name):
        if (type(name) is not str or not name or '\\' in name
                or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                or str(PurePosixPath(name))!=name):
            raise ValueError('canonical relative historical member required')
        return name
    if archive is None:
        if root is None or expected_archive_sha256 is not None:
            raise ValueError('exact directory or verified archive required')
        base=Path(root).resolve(strict=True)
        def get(name):
            path=(base/member_name(name)).resolve(strict=True)
            if not path.is_relative_to(base) or not path.is_file() or path.stat().st_size>cap:
                raise ValueError('bounded original game missing or outside registered root')
            data=path.read_bytes()
            if len(data)>cap:raise ValueError('historical game exceeds byte limit')
            return data
        yield get
        return
    if root is not None or type(expected_archive_sha256) is not str or len(expected_archive_sha256)!=64:
        raise ValueError('one exact archive identity required; no directory fallback')
    path=Path(archive)
    if not path.is_file() or path.stat().st_size>64*1024**2:
        raise ValueError('bounded historical archive required')
    with path.open('rb') as handle:
        checksum=hashlib.sha256()
        for block in iter(lambda:handle.read(1024**2),b''):checksum.update(block)
        if checksum.hexdigest()!=expected_archive_sha256:raise ValueError('historical archive changed')
        handle.seek(0)
        with zipfile.ZipFile(handle) as source:
            infos=source.infolist();names=[info.filename for info in infos]
            if len(names)!=len(set(names)):raise ValueError('duplicate historical archive member')
            def get(name):
                name=member_name(name)
                try:info=source.getinfo(name)
                except KeyError as exc:raise FileNotFoundError('registered historical member missing') from exc
                mode=stat.S_IFMT(info.external_attr>>16)
                if info.is_dir() or mode not in (0,stat.S_IFREG) or not 0<info.file_size<=cap:
                    raise ValueError('bounded regular historical archive member required')
                with source.open(info) as stream:data=stream.read(cap+1)
                if len(data)!=info.file_size or len(data)>cap:
                    raise ValueError('historical archive member size differs')
                return data
            yield get


def historical_index(partition, root, *, expected_partition_sha256,
                     archive=None, expected_archive_sha256=None, on_event=None):
    """Isolated verifier only; never call from the proposal/model process.

Future admitted execution reads every available original game, including
reserved roles, and returns hashes/coverage only. Missing game files with a
registered SHA stop the whole operation. Unsupported rows without executable
games are counted explicitly; no claim covers unavailable pretraining data.
"""
    raw=Path(partition).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_partition_sha256:
        raise ValueError('exact historical partition required')
    counts=Counter();excluded=Counter();seen={};total=0
    allowed={'FIT','DEV','C_J0','C_J1','C_J2','A_RSI1','A_RSI2','FINAL_OFFICIAL','QUARANTINE','EXCLUDED_UNSUPPORTED'}
    with historical_source(root,archive=archive,expected_archive_sha256=expected_archive_sha256) as get:
        for line in raw.splitlines():
            row=json.loads(line);role=row['role']
            if role not in allowed:raise ValueError('unknown historical role')
            if not row.get('game_sha256'):
                if role!='EXCLUDED_UNSUPPORTED' or row.get('eligible') is not False or not row.get('exclusions'):
                    raise ValueError('missing original game cannot silently disappear')
                excluded[role]+=1;continue
            data=get(row['game_path']);total+=len(data)
            if hashlib.sha256(data).hexdigest()!=row['game_sha256']:raise ValueError('historical source changed')
            key=semantic_fingerprint(json.loads(data));counts[role]+=1;seen[key]=seen.get(key,0)+1
            if on_event is not None:on_event(dict(operation='historical_game_completed',role=role,
                completed=sum(counts.values()),bytes_read=total))
    if not counts:raise ValueError('empty historical exclusion inventory')
    return dict(schema='orbit-historical-logical-exclusions-v1',partition_sha256=expected_partition_sha256,
        fingerprint_schema=SCHEMA,counts=dict(counts),without_executable_game=dict(excluded),
        fingerprints=sorted(seen),duplicate_fingerprints=sum(v-1 for v in seen.values()),
        bytes_read=total,scope='all registered native game files; unavailable pretraining data not certified',
        raw_goals_or_plans_exported=False)
