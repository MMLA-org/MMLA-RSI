"""Selected-model subset of the completed fixed-Phi controller.

The evaluator, fixed-n acceptance and journal transitions are retained from the
three-reader implementation. The weight identity remains the selected model. Proposal RNG repetitions
are explicit and never select another trained role bundle. No training or execution on import.
"""
from dataclasses import asdict
import copy
from content_rules import parse_rule, identity
from mmla_pdsa.full.extensions.journal import Journal
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.alfworld_episode import _sha
from demonstration_factory import DemonstrationFactory
from rule_binding import ManagedCandidateFactory
from fixed_phi_evaluation import evaluate_logical_block
from paired_admission import checked_evaluation, paired_admission
from support_readiness import ReadyBlock
SEEDS = (17,)


def checked_rng_seeds(values):
    """Public run boundary: finite declared RNG repeats, with default17 fixed."""
    result=tuple(values)
    if (not result or result[0]!=17 or len(result)>3 or len(set(result))!=len(result)
            or any(type(v) is not int or v not in (17,29,43) for v in result)):
        raise ValueError('distinct proposal RNG seeds17/29/43, beginning with default17')
    return result

ATTEMPTS = 4
GENERATIONS = 2

def wire(rule): return asdict(rule)

def read_rule(language, value):
    rule = parse_rule(language, value['mode'], value['symbols'])
    if rule.mode != 'rtma' or rule.language != value['language']:
        raise ValueError('canonical RTMA rule required')
    return rule

class FixedPhiEvaluator:
    """Bind candidate-specific task interfaces to one immutable actual model."""
    def __init__(self, baseline, *, pair_seed, open_episode, on_work, on_progress, task_backend=None):
        if type(baseline) is not DemonstrationFactory or type(pair_seed) is not int or pair_seed not in (17,29,43):
            raise ValueError('original corrected factory and fixed reader seed required')
        if not all(callable(f) for f in (open_episode,on_work,on_progress)):
            raise ValueError('registered episode and accounting functions required')
        self.baseline=baseline; self.pair_seed=pair_seed
        self.open_episode=open_episode; self.on_work=on_work; self.on_progress=on_progress
        self.task_backend=task_backend
        self._factories={}
        self.model_identity=identity(dict(substrate=baseline.model.substrate_identity,
            modules=tensor_digest(baseline.model.new_state_dict()),profile=baseline.model.profile.identity))
        self._versions=self._current_versions()
        self._definition=identity(baseline.definition)

    def _current_versions(self):
        return tuple((k,id(t),t._version,t.data_ptr(),str(t.device),str(t.dtype),t.requires_grad)
            for k,t in tuple(self.baseline.model.named_parameters())+tuple(self.baseline.model.named_buffers()))

    def factory(self, rule):
        if (self._versions!=self._current_versions() or self._definition!=identity(self.baseline.definition)
                or any(m.training for m in self.baseline.model.modules())):
            raise ValueError('fixed model or task baseline changed')
        if rule.identity not in self._factories:
            self._factories[rule.identity]=ManagedCandidateFactory(self.baseline,rule,task_policy='fixed_rho')
        factory=self._factories[rule.identity];factory._check_binding()
        return factory

    def reader(self, rule): return self.factory(rule).reader_identity()

    def evaluate(self, rule, block):
        if type(block) is not ReadyBlock: raise ValueError('sealed complete block required')
        if self.task_backend is not None:
            result=self.task_backend(self,rule,block)
        else:
            result=evaluate_logical_block(self.factory(rule),block.permits,role=block.role,
                block_sha256=block.block_sha256,population_sha256=block.population_sha256,
                pair_seed=self.pair_seed,open_episode=self.open_episode,on_work=self.on_work,
                # Scalar world execution keeps this fixed-n estimand independent
                # of other worlds' stopping lengths and batch-dependent arithmetic.
                # Batched scientific execution needs a separate invariance result.
                on_progress=self.on_progress,microbatch=1,control='candidate')
        checked_evaluation(result,block,reader=self.reader(rule),rule=rule,pair_seed=self.pair_seed)
        return result

class RuleSuccession:
    """Durable rule owner. Replacement counts only after its journal commits.

    Reads from this registry, rather than a remembered initial rule, drive the
    next generation's actual task/proposal factory. A separate witness records
    the successor's subsequent functional participation. An interrupted live
    run is not resumed automatically by constructing a second owner.
    """
    def __init__(self, language, initial, journal, *, contract, rng_seeds=SEEDS):
        self.rng_seeds=checked_rng_seeds(rng_seeds)
        if (set(initial)!=set(self.rng_seeds) or type(journal) is not Journal or journal.has_evidence()
                or any(read_rule(language,wire(r))!=r for r in initial.values())):
            raise ValueError('the selected initial RTMA parent and a new finite journal required')
        _sha(contract)
        self.language,self.journal,self.contract=language,journal,contract
        self.parents={str(s):wire(initial[s]) for s in self.rng_seeds}
        self.versions={str(s):0 for s in self.rng_seeds};self.edges=[];self.uses=[];self.last_cut=None
        self.cohorts=[];self.decisions=[]
        self.poisoned=False
        self._save()

    def state(self):
        return copy.deepcopy(dict(schema='orbit-fixed-phi-rule-succession-v1',contract=self.contract,
            parents=self.parents,versions=self.versions,edges=self.edges,uses=self.uses,rng_seeds=list(self.rng_seeds),
            cohorts=self.cohorts,decisions=self.decisions,
            poisoned=self.poisoned,automatic_restart=False))

    def _save(self):
        try:
            with self.journal.transaction(1,expected_cut=self.last_cut):
                self.last_cut=self.journal.save(self.state())
                cut,saved=self.journal.load_bound()
                if cut!=self.last_cut or saved!=self.state():
                    raise RuntimeError('actual published rule state failed readback')
        except BaseException:
            self.poisoned=True;raise

    def parent(self, seed):
        if self.poisoned or seed not in self.rng_seeds:raise ValueError('live exact rule owner required')
        return read_rule(self.language,self.parents[str(seed)])



    def audit_and_accept(self, seed, *, generation, block, evaluator, parent_result, child_result,
                         native_resources_complete):
        parent=self.parent(seed)
        if (generation not in range(GENERATIONS) or generation>=len(self.cohorts)
                or generation!=self.versions[str(seed)] or type(evaluator) is not FixedPhiEvaluator
                or evaluator.pair_seed!=seed or type(block) is not ReadyBlock
                or block.role!='A_RSI'+str(generation+1)
                or any(d['seed']==seed and d['generation']==generation for d in self.decisions)):
            raise ValueError('one independent decision for the exact active reader and generation')
        cohort=self.cohorts[generation]
        lock=next((l for l in cohort['locks'] if l['seed']==seed),None)
        if (lock is None or block.candidate_lock_sha256!=cohort['identity']
                or block.population_sha256!=cohort['population']
                or identity({k:v for k,v in cohort.items() if k!='identity'})!=cohort['identity']
                or identity({k:v for k,v in lock.items() if k!='identity'})!=lock['identity']
                or lock['parent']!=wire(parent) or lock['model_identity']!=evaluator.model_identity
                or not lock['functional']
                or not any(u['seed']==seed and u['generation']==generation and u['parent']==parent.identity
                    and u['functional'] and u['completed_attempts']==ATTEMPTS for u in self.uses)):
            raise ValueError('durable all-reader lock and actual parent functional participation required')
        child=read_rule(self.language,lock['candidate'])
        decision=paired_admission(parent_result,child_result,block,model_identity=evaluator.model_identity,
            parent_reader=evaluator.reader(parent),child_reader=evaluator.reader(child),
            parent_rule=parent,child_rule=child,pair_seed=seed,
            candidate_lock_sha256=lock['identity'],native_resources_complete=native_resources_complete,
            family_id=self.contract)
        decision['parent_functional_participation_checked']=True
        self.decisions.append(dict(seed=seed,generation=generation,decision=copy.deepcopy(decision)))
        if not decision['accepted']:
            self._save();return copy.deepcopy(decision)
        edge=dict(seed=seed,generation=generation,parent=parent.identity,child=child.identity,
            lock=lock['identity'],decision=copy.deepcopy(decision),version=generation+1,
            child_rule=wire(child),successor_participation_observed=False)
        # A failed publication poisons this owner. It cannot serve changed rules.
        self.parents[str(seed)]=wire(child);self.versions[str(seed)]=generation+1;self.edges.append(edge)
        self._save()
        return dict(**decision,installed_version=generation+1,installed_cut=self.last_cut)

    def summary(self):
        if self.poisoned:raise ValueError('failed registry publication is not a completed succession')
        edges=copy.deepcopy(self.edges)
        for edge in edges:
            edge['successor_participation_observed']=any(u['seed']==edge['seed']
                and u['generation']==edge['generation']+1 and u['parent']==edge['child']
                and u['functional'] and u['completed_attempts']==ATTEMPTS for u in self.uses)
        chains=[]
        for seed in self.rng_seeds:
            rows=[e for e in edges if e['seed']==seed]
            ok=(len(rows)==2 and rows[0]['generation']==0 and rows[1]['generation']==1
                and rows[0]['child']==rows[1]['parent'] and rows[0]['successor_participation_observed'])
            chains.append(dict(seed=seed,beneficial_finite_chain=ok))
        return dict(edges=edges,chains=chains,acceptances=len(edges),successions=len(edges),
            C_passed=False,official_benchmark_or_SOTA=False,
            scope='RTMA route on the bound conditional population only')
