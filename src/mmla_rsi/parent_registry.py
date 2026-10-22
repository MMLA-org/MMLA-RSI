"""Deployment routing prototype, isolated from the frozen Outcome training run.

Routes immutable rule/package identities; never loads a model or executes a task.
The installer supplies a qualified runtime package for the exact rule. Controller
evidence comes from the existing Journal, not a caller-supplied acceptance flag.
Trusted local journal roots and a verified controller binding are prerequisites.
"""
from dataclasses import asdict, dataclass
import copy, fcntl
from mmla_rsi.rules import identity, parse_rule
from mmla_rsi.evaluation import admission, Case
from mmla_rsi.journal import Journal, StaleController


def _hash(x):
    if type(x) is not str or len(x) != 64 or any(c not in '0123456789abcdef' for c in x):
        raise ValueError('exact lowercase SHA256 identity required')
    return x


@dataclass(frozen=True)
class Binding:
    controller: str
    role_pair: str
    session_adapter: str
    mode: str
    generations: int
    alpha: float
    delta: float
    cost_cap: float

    def __post_init__(self):
        for x in (self.controller, self.role_pair, self.session_adapter): _hash(x)
        if (self.mode not in ('rtt', 'rtma', 'joint') or type(self.generations) is not int
                or not 1 <= self.generations <= 32 or not 0 < self.alpha < 1
                or not 0 < self.delta <= 1 or not 0 <= self.cost_cap <= 1):
            raise ValueError('fixed controller admission and deployment binding required')


@dataclass(frozen=True)
class RuntimePackage:
    """Identity of an externally verified installer package, not a certificate.

    The registry cannot infer model qualification from hashes. The trusted caller
    must verify the real package before publishing. A rule-specific J protocol is
    mandatory for learned J; fixed routing carries no inherited J qualification.
    """
    rule: str
    role_pair: str
    session_adapter: str
    package: str
    j_policy: str
    j_protocol: str | None

    def __post_init__(self):
        for x in (self.rule, self.role_pair, self.session_adapter, self.package): _hash(x)
        if self.j_policy not in ('fixed', 'learned'): raise ValueError('explicit J routing')
        if self.j_policy == 'learned': _hash(self.j_protocol)
        elif self.j_protocol is not None: raise ValueError('fixed routing has no learned J qualification')


def _rule(language, wire):
    if type(wire) is not dict or set(wire) != {'mode', 'symbols', 'language'}:
        raise ValueError('complete canonical rule required')
    rule = parse_rule(language, wire['mode'], wire['symbols'])
    if wire['language'] != language.identity: raise ValueError('different language')
    return rule


def _audit_block(block, rule):
    """Bind the existing Evaluator wire to the exact rule and declared cases."""
    if (block.get('complete') is not True or block.get('rule') != rule.identity
            or not block.get('cases') or len(block['cases']) != len(block.get('results', []))):
        raise ValueError('complete audit for this exact rule required')
    ids=set();tapes=set()
    for declared,row in zip(block['cases'],block['results']):
        case=Case(**declared)
        if (row.get('rule') != rule.identity or row.get('task') != case.task_id
                or row.get('group') != case.physical_group or row.get('payload') != case.payload_sha256
                or row.get('tape') != case.seed or case.task_id in ids or case.seed in tapes):
            raise ValueError('audit rows must correspond one-to-one to declared cases and rule')
        ids.add(case.task_id);tapes.add(case.seed)
        for name in ('loss','cost'):
            if type(row.get(name)) not in (int,float) or not 0 <= row[name] <= 1:
                raise ValueError('finite bounded task endpoint and charged cost required')
    for name in ('loss','cost'):
        if block.get('mean_'+name) != sum(row[name] for row in block['results'])/len(block['results']):
            raise ValueError('audit aggregate differs from its complete rows')


def accepted_successor(journal, *, binding, language, initial, version, parent):
    """Read a durable accepted-generation cut; no controller restore or replay."""
    if type(journal) is not Journal: raise TypeError('existing Journal required')
    cut, saved = journal.load_bound()
    if (saved.get('schema') != 'rsi-controller-v1' or saved.get('contract') != binding.controller
            or saved.get('phase') not in ('READY_GENERATION', 'DONE')
            or saved.get('resumable') is not True or saved.get('persistence_complete') is not True
            or saved.get('failure') is not None or saved.get('role') is not None
            or saved.get('archive') != [] or saved.get('locked') is not None or saved.get('audit') is not None):
        raise ValueError('complete post-acceptance quiescent controller cut required')
    n = version + 1
    if (saved.get('generation') != n or n > binding.generations
            or len(saved.get('versions', [])) != n+1 or len(saved.get('edges', [])) != n
            or len(saved.get('archives', [])) != n
            or saved['phase'] != ('DONE' if n == binding.generations else 'READY_GENERATION')):
        raise ValueError('exactly one accepted successor; no skipped generations')
    if _rule(language, saved['versions'][0]) != initial:
        raise ValueError('different initial lineage')
    previous = initial
    for i, (edge, archive, wire) in enumerate(zip(saved['edges'], saved['archives'], saved['versions'][1:])):
        child = _rule(language, wire)
        locked = archive['locked']; audit = archive['audit']
        lock_payload = {k:v for k,v in locked.items() if k != 'identity'}
        if (child.mode != binding.mode or edge['version'] != i+1
                or archive['generation'] != i or locked['generation'] != i
                or locked['contract'] != binding.controller or locked['parent'] != previous.identity
                or _rule(language, archive['parent']) != previous
                or edge['parent'] != previous.identity or edge['child'] != child.identity
                or edge['lock'] != locked['identity'] or audit['lock'] != locked['identity']
                or identity(lock_payload) != locked['identity']
                or _rule(language, locked['candidate']) != child):
            raise ValueError('broken accepted-edge or candidate-lock chain')
        legal = [x for x in archive['attempts'] if x['eligible'] and x['candidate'] is not None]
        selection=[{k:x[k] for k in ('attempt','eligible','candidate','loss','cost','reward')}
                   for x in archive['attempts']]
        if locked.get('selection') != selection:
            raise ValueError('locked selection differs from development archive')
        if not legal: raise ValueError('no eligible attempt')
        selected = min(legal,key=lambda x:(-x['reward'],x['cost'],tuple(x['candidate']['symbols']),x['attempt']))
        if selected['attempt'] != locked['attempt'] or _rule(language, selected['candidate']) != child:
            raise ValueError('locked candidate differs from eligible development selection')
        _audit_block(audit['parent'],previous);_audit_block(audit['child'],child)
        decision = admission(audit['parent'], audit['child'], alpha=binding.alpha,
            generations=binding.generations, delta=binding.delta, cost_cap=binding.cost_cap)
        if not decision['accepted'] or edge['decision'] != decision or audit['decision'] != decision:
            raise ValueError('independent admission absent or different')
        previous = child
    if _rule(language, saved['versions'][version]) != parent or _rule(language, saved['parent']) != previous:
        raise ValueError('active parent differs from accepted lineage')
    return previous, dict(controller=binding.controller,cut=cut,generation=n,
        edge=copy.deepcopy(saved['edges'][-1]),archive_identity=identity(saved['archives'][-1]['locked']),
        functional_participation=saved['edges'][-1].get('functional_participation'))


class ParentRegistry:
    """Durable version routing. Outstanding leases keep their original package.

    Publications change only the registry pointer at its transaction boundary.
    The caller constructs a fresh OutcomeSession from each lease, never hot-edits
    a session. Reopening a registry does not resume an interrupted task; existing
    leases remain occupied until explicitly reconciled and closed with a receipt.
    """
    def __init__(self, journal, *, binding, language, initial, max_live_requests):
        if (type(journal) is not Journal or type(binding) is not Binding
                or type(max_live_requests) is not int or not 1 <= max_live_requests <= 65536
                or _rule(language, asdict(initial)) != initial or initial.mode != binding.mode):
            raise ValueError('bound registry, rule, and finite live-request capacity required')
        self.journal, self.binding, self.language, self.initial = journal, binding, language, initial
        self.max_live_requests = max_live_requests
        self.identity = identity(dict(schema='orbit-deployment-parent-registry-v1',binding=asdict(binding),
            language=language.identity,initial=asdict(initial),max_live_requests=max_live_requests))

    def _runtime(self, package, rule):
        if (type(package) is not RuntimePackage or package.rule != rule.identity
                or package.role_pair != self.binding.role_pair or package.session_adapter != self.binding.session_adapter):
            raise ValueError('installer package must bind exact rule, role pair, and adapter')
        return asdict(package)

    def initialize(self, package):
        runtime = self._runtime(package, self.initial)
        with self.journal.transaction(1, expected_cut=None):
            state = dict(schema='orbit-deployment-parent-registry-v1',binding=self.identity,version=0,
                active=dict(rule=asdict(self.initial),runtime=runtime),evidence=None,
                next_request=0,leases={},last_close=None)
            return self.journal.save(state)

    def _load(self):
        cut, state = self.journal.load_bound()
        if state.get('schema') != 'orbit-deployment-parent-registry-v1' or state.get('binding') != self.identity:
            raise ValueError('different registry binding')
        self._runtime(RuntimePackage(**state['active']['runtime']),_rule(self.language,state['active']['rule']))
        if (type(state.get('version')) is not int or not 0 <= state['version'] <= self.binding.generations
                or type(state.get('next_request')) is not int or state['next_request'] < 0
                or type(state.get('leases')) is not dict or len(state['leases']) > self.max_live_requests):
            raise ValueError('invalid registry cursor')
        return cut, state

    def snapshot(self): return self._load()[1]

    def acquire(self):
        cut, state = self._load()
        with self.journal.transaction(1, expected_cut=cut):
            if len(state['leases']) >= self.max_live_requests: raise ValueError('request capacity reached')
            sequence=state['next_request']
            lease=dict(registry=self.identity,sequence=sequence,version=state['version'],**copy.deepcopy(state['active']))
            lease['identity']=identity(lease)
            state['leases'][str(sequence)]=copy.deepcopy(lease);state['next_request']+=1
            self.journal.save(state)
            return lease

    def resolve(self, lease):
        _, state = self._load()
        if type(lease) is not dict or state['leases'].get(str(lease.get('sequence'))) != lease:
            raise ValueError('unknown, mutated, or already closed request')
        return copy.deepcopy(lease)

    def close(self, lease, *, outcome_receipt):
        _hash(outcome_receipt)
        cut,state=self._load()
        with self.journal.transaction(1,expected_cut=cut):
            key=str(lease.get('sequence'))
            if state['leases'].get(key) != lease: raise ValueError('stale request close')
            del state['leases'][key]
            state['last_close']=dict(lease=lease['identity'],outcome_receipt=outcome_receipt)
            return self.journal.save(state)

    def publish(self, controller_journal, *, expected_version, package):
        cut,state=self._load()
        if type(expected_version) is not int or state['version'] != expected_version:
            raise StaleController('parent changed before publication')
        if (type(controller_journal) is not Journal or
                controller_journal.root.resolve()==self.journal.root.resolve()):
            raise ValueError('distinct controller and routing journals required')
        # Lock order: controller shared fence, then registry CAS writer fence.
        # load_bound's own shared lock alone ends before publication. This outer
        # fence keeps the inspected cut current until the new pointer is durable.
        with (controller_journal.root/'.writer.lock').open('a') as controller_lock:
            fcntl.flock(controller_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            parent=_rule(self.language,state['active']['rule'])
            child,evidence=accepted_successor(controller_journal,binding=self.binding,language=self.language,
                initial=self.initial,version=expected_version,parent=parent)
            runtime=self._runtime(package,child)
            if (runtime['j_policy']=='learned' and state['active']['runtime']['j_policy']=='learned'
                    and runtime['j_protocol']==state['active']['runtime']['j_protocol']):
                raise ValueError('changed rule cannot inherit the old J protocol')
            with self.journal.transaction(1,expected_cut=cut):
                state['version']+=1;state['active']=dict(rule=asdict(child),runtime=runtime);state['evidence']=evidence
                return self.journal.save(state)
