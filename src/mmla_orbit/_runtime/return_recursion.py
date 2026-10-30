"""Three matched edit processes on the selected model; at most two RTMA generations.

Reuse the original native task evaluator, fixed-n admission and durable journal.
The new edit process and three-way lock are explicit; they are not the previous
whole-program/two-process experiment under a new name. Native role isolation,
exact export authentication and complete resource enforcement remain the host's
responsibility; this module cannot launch them or grant its own admission.
"""
from dataclasses import asdict
import copy
from content_rules import identity, parse_rule
from content_evidence import EvidenceMemory
from edit_role import EditProcess, model_guard
from trained_pairs import identity as model_identity
from recursion_core import FixedPhiEvaluator, RuleSuccession, read_rule, wire
from support_readiness import ReadyBlock
from paired_admission import checked_evaluation, DELTA
from logical_public_rule import check_public_rule

PROCESSES = ('trained-keep', 'trained-frozen', 'original-keep')
SEEDS = (17,)


class ReturnRoleFactory:
    def __init__(self, evaluator, trained, original, interface, memory0, *, on_work, on_event):
        if (type(evaluator) is not FixedPhiEvaluator or int(trained.rho.version) != 184
                or int(original.rho.version) != 156
                or any(p.requires_grad for m in (trained,original) for p in m.parameters())):
            raise ValueError('fixed authentic task reader and trained/original proposal roles required')
        self.evaluator, self.models = evaluator, {'trained':trained, 'original':original}
        self.identities = {k:model_identity(m) for k,m in self.models.items()}
        self.guards = {k:model_guard(m) for k,m in self.models.items()}
        self.interface, self.memory0 = interface, memory0
        self.on_work, self.on_event = on_work, on_event

    def create(self, parent, generation, process):
        if process not in PROCESSES or generation not in (0,1):
            raise ValueError('fixed three-process two-generation schedule required')
        key = 'original' if process == 'original-keep' else 'trained'
        if model_guard(self.models[key]) != self.guards[key]:
            raise ValueError('frozen proposal model changed')
        self.evaluator.factory(parent)  # This same parent actually governs the task role.
        memory = EvidenceMemory.restore(self.memory0.export(), count_tokens=self.memory0._count_tokens,
                                         expected_contract=self.memory0.contract)
        return EditProcess(self.models[key], self.interface, parent, memory,
            control='frozen' if process == 'trained-frozen' else 'keep', seed=self.evaluator.pair_seed,
            phase='dev', round_index=generation, on_work=self.on_work, on_event=self.on_event,
            functional_probes=process != 'trained-frozen')


def develop_return_process(process, evaluator, dev, *, generation, process_name, parent_result,
                           proposal_identity, on_event, save_process):
    if (type(process) is not EditProcess or type(evaluator) is not FixedPhiEvaluator
            or type(dev) is not ReadyBlock or dev.role != 'DEV' or len(dev.permits) != 32
            or process_name not in PROCESSES or process.attempt != 0
            or process.phase != 'dev' or process.round_index != generation
            or process.seed != evaluator.pair_seed):
        raise ValueError('fresh matched edit process and complete seen DEV required')
    parent = process.plan.parent
    checked_evaluation(parent_result, dev, reader=evaluator.reader(parent), rule=parent, pair_seed=evaluator.pair_seed)
    archive = []
    try:
        save_process(process.export())
        for attempt in range(4):
            draw = process.draw()
            # Store sampled path/RNG before an environment can fail.
            save_process(process.export())
            evaluation = evaluator.evaluate(draw.sample.program, dev)
            loss, cost = evaluation['mean_loss'], evaluation['mean_cost']
            receipt = process.feedback(draw, loss=loss, cost=cost, tag='completed')
            sample = asdict(draw.sample)
            for key in ('rng_before','rng_after'):
                sample[key] = sample[key].tolist()
            item = dict(attempt=attempt, candidate=wire(draw.sample.program), loss=loss, cost=cost,
                evaluation=evaluation, eligible=draw.eligible, witness=copy.deepcopy(draw.witness),
                sampled_path=sample, sampled_path_identity=identity(sample), adaptation=receipt)
            archive.append(item)
            on_event(dict(kind='RETURN_DEV_ATTEMPT_COMPLETE', seed=evaluator.pair_seed,
                generation=generation, process=process_name, record=item))
            save_process(process.export())
    except BaseException as error:
        process.failed = True
        try:
            save_process(process.export())
        except BaseException as reporting:
            error.add_note('partial development process save: '+repr(reporting))
        raise
    return dict(schema='orbit-process-credit-development-v1', complete=True, seed=evaluator.pair_seed,
        generation=generation, process=process_name, parent=wire(parent), parent_result=parent_result,
        model_identity=evaluator.model_identity, proposal_identity=proposal_identity,
        dev_block=dev.block_sha256, population=dev.population_sha256, attempts=archive,
        role_contract=process.identity, memory_commits=process.memory_commits,
        numerical_deployment_updates=0, full_archive_visible_to_proposer=False)


def lock_return_cohort(generation, developments, *, language, parents, population):
    if generation not in (0,1) or not parents or set(parents) != set(developments) or not set(parents) <= set(SEEDS):
        raise ValueError('complete active generation required')
    locks, stops = [], []
    for seed in sorted(parents):
        group = developments[seed]
        if set(group) != set(PROCESSES):
            raise ValueError('all three processes must finish before any audit draw')
        main = group['trained-keep']; choices = {}
        for name in PROCESSES:
            data = group[name]
            if (data['schema'] != 'orbit-process-credit-development-v1' or data['complete'] is not True
                    or data['seed'] != seed or data['generation'] != generation or data['process'] != name
                    or data['parent'] != wire(parents[seed]) or data['population'] != population
                    or data['model_identity'] != main['model_identity'] or data['dev_block'] != main['dev_block']
                    or data['parent_result'] != main['parent_result'] or data['full_archive_visible_to_proposer'] is not False
                    or len(data['attempts']) != 4 or [r['attempt'] for r in data['attempts']] != list(range(4))):
                raise ValueError('paired complete task/read/source identity required')
            legal = []
            for row in data['attempts']:
                if row['candidate'] is None:
                    continue
                read_rule(language,row['candidate'])
                if row['evaluation'] is None or row['loss'] != row['evaluation']['mean_loss'] or row['cost'] != row['evaluation']['mean_cost']:
                    raise ValueError('selector must use closed complete-task returns')
                if name == 'trained-keep' and (not row['eligible'] or not row['witness']['functional']):
                    continue
                legal.append(row)
            choices[name] = min(legal,key=lambda r:(r['loss'],r['cost'],tuple(r['candidate']['symbols']),r['attempt'])) if legal else None
        if (main['proposal_identity'] != group['trained-frozen']['proposal_identity']
                or main['proposal_identity']['rho_version'] != 184 or group['original-keep']['proposal_identity']['rho_version'] != 156):
            raise ValueError('paired trained proposal and explicit original proposal required')
        chosen = choices['trained-keep']
        if any(choices[name] is None for name in PROCESSES):
            reason = 'MISSING_ELIGIBLE_MAIN_OR_LEGAL_COMPARATOR'
        elif chosen['candidate'] == wire(parents[seed]) or main['parent_result']['mean_loss']-chosen['loss'] < DELTA:
            reason = 'NO_MATERIAL_ELIGIBLE_DEV_SUCCESSOR'
        else:
            reason = None
        if reason:
            stops.append(dict(seed=seed,reason=reason,developments={k:identity(v) for k,v in group.items()}))
            continue
        lock = dict(schema='orbit-process-credit-candidate-lock-v1', seed=seed, generation=generation,
            model_identity=main['model_identity'], parent=wire(parents[seed]), candidate=chosen['candidate'],
            attempt=chosen['attempt'], functional=True, witness=chosen['witness'], rollout=chosen['sampled_path_identity'],
            developments={k:identity(v) for k,v in group.items()},
            proposal_identities={k:v['proposal_identity'] for k,v in group.items()},
            comparators={k:dict(candidate=choices[k]['candidate'],attempt=choices[k]['attempt']) for k in PROCESSES[1:]},
            population=population, dev_block=main['dev_block'], audit_exposed=False)
        lock['identity'] = identity(lock); locks.append(lock)
    cohort = dict(schema='orbit-process-credit-cohort-lock-v1', generation=generation, active_seeds=sorted(parents),
        population=population, locks=locks, stops=stops, audit_generation_allowed=bool(locks),
        all_development_complete=True, controls='conditional on the actual main-chain parent')
    cohort['identity'] = identity(cohort)
    return cohort


class ReturnSuccession(RuleSuccession):
    """Same journal and new fixed-n conditional-family acceptance; new process/lock semantics."""
    def state(self):
        value = super().state(); value['schema'] = 'orbit-process-credit-rule-succession-v1'
        return value

    def observe_process(self, seed, generation, process, development):
        parent = self.parent(seed)
        if (type(process) is not EditProcess or process.control != 'keep' or process.plan.parent != parent
                or process.failed or process.pending is not None or process.attempt != 4
                or generation != self.versions[str(seed)] or process.round_index != generation
                or process.seed != seed or int(process.model.rho.version) != 184
                or development['role_contract'] != process.identity or development['process'] != 'trained-keep'
                or model_guard(process.model) != process.guard):
            raise ValueError('actual accepted parent must govern the complete main proposal process')
        witnesses = [r for r in development['attempts'] if r['eligible'] and r['witness']['functional']]
        record = dict(seed=seed, generation=generation, parent=parent.identity, version=self.versions[str(seed)],
            role_contract=process.identity, memory_commits=process.memory_commits, functional=bool(witnesses),
            witness_rollouts=[r['sampled_path_identity'] for r in witnesses], completed_attempts=4)
        self.uses.append(record); self._save()
        return record

    def lock_generation(self, generation, developments, *, population):
        if self.poisoned or len(self.cohorts) != generation:
            raise ValueError('one durable cohort per generation')
        if generation and {d['seed'] for d in self.decisions if d['generation']==generation-1} != {l['seed'] for l in self.cohorts[-1]['locks']}:
            raise ValueError('all preceding decisions must close first')
        parents = {s:self.parent(s) for s in SEEDS if self.versions[str(s)] == generation}
        cohort = lock_return_cohort(generation, developments, language=self.language, parents=parents, population=population)
        self.cohorts.append(cohort); self._save()
        return copy.deepcopy(cohort)


def run_return_recursion(*, language, readers, initial_rules, dev, registry, role_factories,
                         prepare_audit, evaluate_strong_rule, resources_complete,
                         on_event, save_process, save_cohort):
    if (tuple(readers) != SEEDS or set(initial_rules) != set(SEEDS) or set(role_factories) != set(SEEDS)
            or type(registry) is not ReturnSuccession or type(dev) is not ReadyBlock or dev.role != 'DEV'
            or len(dev.permits) != 32 or dev.candidate_lock_sha256 is not None
            or any(type(e) is not FixedPhiEvaluator or e.pair_seed != s for s,e in readers.items())
            or any(type(f) is not ReturnRoleFactory or f.evaluator is not readers[s] for s,f in role_factories.items())
            or any(registry.parent(s) != initial_rules[s] for s in SEEDS)):
        raise ValueError('the selected model and all three proposal controls are required')
    seen = frozenset(p.semantic_sha256 for p in dev.permits); active=set(SEEDS); outcomes=[]
    strong_dev = evaluate_strong_rule(dev); check_public_rule(strong_dev,dev)
    coverage_rule = parse_rule(language,'rtma',('sub','x8','x9'))
    null_rule = parse_rule(language,'rtma',('c0',))
    for generation in (0,1):
        if not active:
            break
        developments = {}; references = {}
        for seed in sorted(active):
            parent=registry.parent(seed); evaluator=readers[seed]; factory=role_factories[seed]
            references[seed] = {name:evaluator.evaluate(rule,dev) for name,rule in
                               (('parent',parent),('coverage',coverage_rule),('task_NULL',null_rule))}
            developments[seed] = {}
            for name in PROCESSES:
                process = factory.create(parent,generation,name)
                result = develop_return_process(process,evaluator,dev,generation=generation,
                    process_name=name,parent_result=references[seed]['parent'],
                    proposal_identity=factory.identities['original' if name=='original-keep' else 'trained'],
                    on_event=on_event,save_process=lambda state,s=seed,g=generation,n=name:save_process(s,g,n,state))
                developments[seed][name] = result
                if name == 'trained-keep':
                    registry.observe_process(seed,generation,process,result)
        cohort = registry.lock_generation(generation,developments,population=dev.population_sha256)
        save_cohort(generation,cohort,dict(developments=developments,references=references))
        on_event(dict(kind='ALL_RETURN_CANDIDATES_LOCKED',cohort=cohort))
        if not cohort['locks']:
            outcomes.append(dict(generation=generation,cohort=cohort,audit_opened=False,reason='NO_DEV_CANDIDATE'))
            break
        audit = prepare_audit(generation,copy.deepcopy(cohort),seen)
        if (type(audit) is not ReadyBlock or audit.role != 'A_RSI'+str(generation+1)
                or len(audit.permits) != 1024 or audit.population_sha256 != dev.population_sha256
                or audit.candidate_lock_sha256 != cohort['identity']
                or len({p.semantic_sha256 for p in audit.permits}) != 1024
                or any(p.semantic_sha256 in seen for p in audit.permits)):
            raise ValueError('fresh exact independent support after complete cohort lock required')
        seen |= frozenset(p.semantic_sha256 for p in audit.permits)
        strong=evaluate_strong_rule(audit); check_public_rule(strong,audit)
        result=dict(generation=generation,cohort=cohort,audit_opened=True,
                    audit_block=audit.block_sha256,strong_rule=strong,edges=[])
        next_active=set()
        for lock in cohort['locks']:
            seed=lock['seed']; evaluator=readers[seed]; parent=registry.parent(seed)
            child=read_rule(language,lock['candidate'])
            scores={name:evaluator.evaluate(rule,audit) for name,rule in (
                ('parent',parent),('child',child),('coverage',coverage_rule),
                *((name,read_rule(language,lock['comparators'][name]['candidate'])) for name in PROCESSES[1:]))}
            decision=registry.audit_and_accept(seed,generation=generation,block=audit,evaluator=evaluator,
                parent_result=scores['parent'],child_result=scores['child'],
                native_resources_complete=resources_complete())
            row=dict(seed=seed,scores=scores,decision=decision,
                descriptive_child_gains={name:value['mean_loss']-scores['child']['mean_loss'] for name,value in scores.items() if name!='child'},
                controls_are_parent_conditional=True,seed_is_not_an_independent_world=True)
            result['edges'].append(row); on_event(dict(kind='RETURN_EDGE_DECIDED',generation=generation,record=row))
            if decision['accepted']:
                next_active.add(seed)
        outcomes.append(result); active=next_active
    closed=resources_complete()
    if type(closed) is not bool:
        raise ValueError('actual resource boundary required')
    return dict(schema='orbit-process-credit-recursion-result-v1',
        status='COMPLETE_FINITE_ALGORITHM' if closed else 'INCOMPLETE',generations=outcomes,
        strong_dev=strong_dev,registry=registry.summary(),actual_native_completion=closed,
        routes_executed=['RTMA'],RTT_executed=False,joint_executed=False,official_C_passed=False,
        publication='HOLD_PENDING_FULL_NATIVE_RESULT_REVIEW')
