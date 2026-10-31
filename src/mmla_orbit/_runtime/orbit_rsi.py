"""Public, finite RTMA execution for the selected Orbit model.

Task execution and resource enforcement belong to the deployment host. The
controller owns the three proposal processes, full-task feedback, independent
admission and installed rules. It never calls the proposal-training workflow.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

from content_rules import identity
from demonstration_factory import DemonstrationFactory
from recursion_core import FixedPhiEvaluator, read_rule, checked_rng_seeds
from return_recursion import ReturnRoleFactory, ReturnSuccession, run_return_recursion
from support_readiness import ReadyBlock
from logical_public_rule import evaluate_public_rule
from mmla_pdsa.full.extensions.journal import Journal
from reward_training import save_artifact, read_artifact
from trained_pairs import identity as model_identity


def model_binding(orbit):
    """Validate and identify the loaded release once at the execution boundary."""
    if (orbit.seed != 17 or orbit.config.get('selected_seed') != 17
            or set(orbit.config['roles']) != {'17'}):
        raise ValueError('this runtime requires the released single seed17 model')
    runtime = orbit.config['public_runtime_identity']
    if type(runtime) is not str or len(runtime) != 64:
        raise ValueError('bound public runtime identity required')
    int(runtime, 16)
    return dict(runtime=runtime, training_source=orbit.source_identity,
        task=model_identity(orbit.task), proposal=model_identity(orbit.proposal),
        original_proposal=model_identity(orbit.original_proposal),
        interface=orbit.interface.identity, memory=orbit.memory0.contract,
        initial_manager=asdict(orbit.initial_manager), selected_seed=orbit.seed)


def run_rsi(orbit, *, directory, run_identity, environment_identity, dev, open_episode,
            prepare_audit, on_work, on_event, resources_complete, task_backend=None, rng_seeds=(17,)):
    """Run a new finite deployment, without retraining or weakening admission.

    dev is the complete sealed 32-world block. open_episode opens the matching
    teacher-free LogicalWorldEpisode. prepare_audit is invoked only after all
    three controls finish and their candidate lock is saved; it must return a
    newly sealed 1024-world block bound to that lock. No audit scores are supplied
    by the caller: the controller evaluates parent, child and controls itself.

    task_backend may be the already qualified parallel scalar-world backend;
    its complete output is checked by FixedPhiEvaluator. Otherwise the exact
    scalar evaluator runs. The host must meter all work and enforce its declared
    limits; resources_complete must check actual quiescence/accounting. A Python
    callback is not a sandbox or an independent authorization mechanism.
    """
    binding = model_binding(orbit)
    rng_seeds=checked_rng_seeds(rng_seeds)
    for value in (run_identity, environment_identity):
        if type(value) is not str or len(value) != 64:
            raise ValueError('explicit run and environment identities required')
        int(value, 16)
    if (type(dev) is not ReadyBlock or dev.role != 'DEV' or len(dev.permits) != 32
            or dev.candidate_lock_sha256 is not None
            or not all(callable(f) for f in (open_episode, prepare_audit, on_work,
                                             on_event, resources_complete))):
        raise ValueError('sealed DEV and bound execution/accounting interfaces required')
    baseline = DemonstrationFactory(orbit.task, orbit)
    readers={s:FixedPhiEvaluator(baseline, pair_seed=s, open_episode=open_episode,
        on_work=on_work, on_progress=on_event, task_backend=task_backend) for s in rng_seeds}
    factories={s:ReturnRoleFactory(readers[s], orbit.proposal, orbit.original_proposal,
        orbit.interface, orbit.memory0, on_work=on_work, on_event=on_event) for s in rng_seeds}
    definition = dict(schema='orbit-selected-rtma-execution-v1', model=binding,
        run_identity=run_identity, environment_identity=environment_identity,
        dev=dict(ready=dev.ready_sha256, seal=dev.seal_sha256, block=dev.block_sha256,
                 producer=dev.producer_sha256, population=dev.population_sha256),
        attempts=4, generations=2, rng_seeds=list(rng_seeds), weight_seed=orbit.seed,
        default_manager_rng=17, same_weight_repetitions=True,
        processes=['trained-keep', 'trained-frozen', 'original-keep'],
        admission=dict(n=1024, delta=.02, alpha=.025/18, evalue_threshold=720),
        numerical_deployment_updates=0, automatic_retry=False)
    contract = identity(definition)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    save_artifact(root/'definition', definition, source=binding['runtime'], kind='rsi-definition')
    cuts=2+2*(2*len(rng_seeds)+1)
    journal = Journal(root/'registry', max_bytes=cuts*8*1024**2+4096, max_cuts=cuts,
                      max_cut_bytes=8*1024**2)
    registry = ReturnSuccession(orbit.interface.language, {s:orbit.initial_manager for s in rng_seeds},
                               journal, contract=contract,rng_seeds=rng_seeds)
    counts = {}

    def save_process(seed, generation, name, state):
        key = (seed, generation, name)
        index = counts.get(key, 0)
        save_artifact(root/'processes'/str(seed)/str(generation)/name/str(index), state,
                      source=binding['runtime'], kind='rsi-process-state')
        counts[key] = index+1

    def save_cohort(generation, cohort, developments):
        save_artifact(root/'cohorts'/str(generation),
            dict(cohort=cohort, developments=developments),
            source=binding['runtime'], kind='rsi-cohort')

    # Each publication is new. An interrupted call remains incomplete and this
    # same directory cannot be restarted through run_rsi.
    result = run_return_recursion(language=orbit.interface.language,
        readers=readers, initial_rules={s:orbit.initial_manager for s in rng_seeds}, dev=dev,
        registry=registry, role_factories=factories, prepare_audit=prepare_audit,
        evaluate_strong_rule=lambda block:evaluate_public_rule(block, codec=baseline.codec,
            open_episode=open_episode, on_work=on_work, on_event=on_event),
        resources_complete=resources_complete, on_event=on_event,
        save_process=save_process, save_cohort=save_cohort)
    value = dict(schema='orbit-selected-rtma-closed-v1', definition=definition,
        contract=contract, result=result, registry=registry.state(), cut=registry.last_cut,
        process_snapshots=[dict(rng_seed=s, generation=g, process=n, count=c)
                           for (s,g,n),c in sorted(counts.items())])
    receipt = save_artifact(root/'closed', value, source=binding['runtime'], kind='rsi-closed')
    return dict(result, controller_artifact=receipt)


@dataclass(frozen=True)
class RecoveredRTMA:
    """A closed controller state; it does not silently schedule another run."""
    orbit: object
    registry: object
    result: dict
    contract: str

    @property
    def manager(self):
        return self.registry.parent(17)

    def new_task_state(self, *, seed, tenant):
        """Deploy the installed manager, preserving this model's trained rho."""
        from rule_binding import ManagedCandidateFactory
        baseline = DemonstrationFactory(self.orbit.task, self.orbit)
        factory = ManagedCandidateFactory(baseline, self.manager, task_policy='fixed_rho')
        return factory.new_state(seed=seed, tenant=tenant)


def restore_rsi(orbit, directory, *, manifest_sha256):
    """Restore a complete installed rule registry without a model/task call.

    The host supplies the saved closed-artifact manifest hash. Pending or
    failed executions have no usable closed state. This recovers deployment
    state and history; it is not permission to repeat or extend a finite run.
    """
    binding = model_binding(orbit)
    root = Path(directory)
    value = read_artifact(root/'closed', source=binding['runtime'], kind='rsi-closed',
                          manifest_sha256=manifest_sha256)
    if (value['schema'] != 'orbit-selected-rtma-closed-v1'
            or value['definition']['model'] != binding
            or value['contract'] != identity(value['definition'])
            or value['result']['status'] != 'COMPLETE_FINITE_ALGORITHM'
            or value['result']['actual_native_completion'] is not True):
        raise ValueError('complete matching controller artifact required')
    rng_seeds=checked_rng_seeds(value['definition'].get('rng_seeds',(17,)))
    cuts=2+2*(2*len(rng_seeds)+1)
    journal = Journal(root/'registry', max_bytes=cuts*8*1024**2+4096, max_cuts=cuts,
                      max_cut_bytes=8*1024**2)
    cut, state = journal.load_bound()
    if cut != value['cut'] or state != value['registry']:
        raise ValueError('closed record and installed rule journal differ')
    if (state['schema'] != 'orbit-process-credit-rule-succession-v1'
            or state['contract'] != value['contract'] or state['poisoned'] is not False
            or set(state['parents']) != {str(s) for s in rng_seeds}
            or set(state['versions']) != {str(s) for s in rng_seeds}
            or tuple(state.get('rng_seeds',(17,))) != rng_seeds):
        raise ValueError('matching unpoisoned selected-model registry required')
    read_rule(orbit.interface.language, state['parents']['17'])
    # Bytes and computation identity have crossed their owning boundaries.
    # Reconstruct the same registry without __init__ writing an initial parent.
    registry = object.__new__(ReturnSuccession)
    registry.language, registry.journal = orbit.interface.language, journal
    registry.contract, registry.last_cut = value['contract'], cut
    registry.rng_seeds=rng_seeds
    for key in ('parents', 'versions', 'edges', 'uses', 'cohorts', 'decisions', 'poisoned'):
        setattr(registry, key, state[key])
    if registry.summary() != value['result']['registry']:
        raise ValueError('recovered succession differs from the complete result')
    return RecoveredRTMA(orbit, registry, value['result'], value['contract'])
