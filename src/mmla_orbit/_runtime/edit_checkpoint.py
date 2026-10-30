"""Quiet-boundary recovery for the task-return RTMA subtree-edit proposer.

Uses the production sampler and data-only artifact format unchanged. A caller
creates a fresh EditProcess with the intended model, manager, interface and
generation, then restores a checkpoint into it. Pending/failed draws stay in
the scientific archive; this API does not restart an interrupted experiment.
"""
from dataclasses import asdict
from content_evidence import EvidenceMemory, Record
from content_rules import parse_rule
from edit_role import EditProcess, model_guard
from mmla_pdsa.full.safe_state import exact_equal
from reward_training import read_artifact, save_artifact
from trained_pairs import identity as model_identity


def _binding(process):
    """Shared boundary for the complete, frozen proposal computation."""
    if (type(process) is not EditProcess or process.phase != 'dev'
            or process.failed or process.pending is not None
            or model_guard(process.model) != process.guard
            or any(m.training for m in process.model.modules())
            or any(p.requires_grad or p.grad is not None for p in process.model.parameters())):
        raise ValueError('quiet deployment process with unchanged frozen model required')
    return dict(model=model_identity(process.model), interface=process.interface.identity,
        parent=asdict(process.plan.parent), control=process.control, seed=process.seed,
        phase=process.phase, generation=process.round_index,
        functional_probes=process.functional_probes, memory0=process.memory0.export())


def save_edit_checkpoint(process, directory, *, source_identity):
    """Save between feedback and the next draw, including a completed process."""
    value = dict(schema='orbit-return-edit-checkpoint-v1', binding=_binding(process),
        state=process.export())
    return save_artifact(directory, value, source=source_identity, kind='return-edit-checkpoint')


def restore_edit_checkpoint(fresh, directory, *, source_identity, manifest_sha256):
    """Recover an authenticated checkpoint into a caller-owned fresh process.

    At most three deterministic memory decisions are replayed to check the
    saved state. No model forward, proposal sampling, environment execution,
    rule admission or scientific-run continuation occurs during this call.
    """
    binding = _binding(fresh)
    if fresh.attempt or fresh.memory_commits or fresh.receipts or fresh.memory.identity != fresh.memory0.identity:
        raise ValueError('restore requires a fresh caller-owned process')
    value = read_artifact(directory, source=source_identity, kind='return-edit-checkpoint',
        manifest_sha256=manifest_sha256)
    if (set(value) != {'schema', 'binding', 'state'}
            or value['schema'] != 'orbit-return-edit-checkpoint-v1'
            or not exact_equal(value['binding'], binding)):
        raise ValueError('checkpoint model, manager, interface or generation differs')
    state = value['state']
    initial = fresh.export()
    if (set(state) != set(initial) or state['schema'] != initial['schema']
            or state['process'] != fresh.identity or state['control'] != fresh.control
            or state['rho_version'] != initial['rho_version']
            or state['rho_parameters'] != initial['rho_parameters']
            or not exact_equal(state['memory0'], initial['memory0'])
            or state['pending'] is not None or state['failed'] is not False
            or type(state['attempt']) is not int or not 0 <= state['attempt'] <= 4
            or type(state['memory_commits']) is not int
            or not 0 <= state['memory_commits'] <= min(3, state['attempt'])
            or type(state['receipts']) is not list or len(state['receipts']) != state['attempt']):
        raise ValueError('complete quiet process state required')
    memory = EvidenceMemory.restore(state['memory0'], count_tokens=fresh.memory0._count_tokens,
        expected_contract=fresh.memory0.contract)
    commits = 0
    for attempt, receipt in enumerate(state['receipts']):
        record = Record(**receipt['record'])
        record.validate()
        program = parse_rule(fresh.interface.language, 'rtma', record.action.split())
        if (record.action != ' '.join(program.symbols) or record.opportunity != attempt
                or record.source != memory.source_prefix + 'proposal/' + str(attempt)):
            raise ValueError('ordered complete-program feedback required')
        before = memory.identity
        managed = None
        if fresh.control == 'keep' and attempt < 3:
            memory, managed = memory.manage(record, language=fresh.interface.language,
                rule=fresh.plan.parent, opportunity_cap=4)
            commits += int(managed['status'] == 'COMMIT')
        expected = dict(attempt=attempt, parent=fresh.plan.parent.identity,
            program=program.identity, record=asdict(record), pre_memory=before,
            post_memory=memory.identity, memory=managed, final_attempt=attempt == 3,
            numerical_deployment_updates=0, control=fresh.control)
        if not exact_equal(receipt, expected):
            raise ValueError('saved feedback differs from its parent-managed memory transition')
    if commits != state['memory_commits'] or not exact_equal(memory.export(), state['memory']):
        raise ValueError('saved memory does not match the complete feedback sequence')
    # The next attempt's independent RNG is reconstructed by the unchanged
    # production draw() seed rule. Archives never become proposal input.
    fresh.memory = memory
    fresh.memory_commits = commits
    fresh.attempt = state['attempt']
    fresh.receipts = state['receipts']
    return fresh
