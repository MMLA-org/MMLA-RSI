"""Arbitrary sampled manager programs on the original selected FIT tasks.

The native role authenticates the sealed view and permits once. This evaluator
owns the boundary from completed native episodes to typed task returns. It does
not call the closed four-rule-bank evaluator or reuse its result inventory.
"""
from dataclasses import asdict, dataclass
from content_rules import parse_rule
from curriculum import digest
from fit_evaluation import check_fit_permits, run_fit_task, rule_wire, PARTITION
from fit_selected_view import SELECTION
from rule_binding import ManagedCandidateFactory


@dataclass(frozen=True)
class FitReturns:
    task_ids: tuple[str, ...]
    rule: str
    reader: str
    losses: tuple[float, ...]
    costs: tuple[float, ...]
    endpoint_identities: tuple[str, ...]


class FitRewardEvaluator:
    def __init__(self, baseline, selection, permits, *, seed, source_identity,
                 open_episode, on_work, on_event):
        check_fit_permits(selection, permits)
        if selection['identity'] != SELECTION or seed not in (17, 29, 43):
            raise ValueError('fixed original 56-task FIT selection and reader required')
        self.baseline, self.selection, self.permits = baseline, selection, permits
        self.seed, self.source = seed, source_identity
        self.reader = baseline.reader_identity()
        self.open_episode, self.on_work, self.on_event = open_episode, on_work, on_event

    def evaluate(self, rule, indices):
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < 56 for i in indices)
                or rule.mode != 'rtma' or parse_rule(self.baseline.language, 'rtma', rule.symbols) != rule):
            raise ValueError('one legal complete program on a fixed FIT subset required')
        if self.baseline.reader_identity() != self.reader:
            raise ValueError('fixed task reader changed')
        factory = ManagedCandidateFactory(self.baseline, rule, task_policy='fixed_rho')
        execution_reader = factory.reader_identity()
        cap = 64 * factory.model.profile.max_new_tokens * factory.model.profile.local_tokens + 63 * 4096
        losses, costs, seals, ids = [], [], [], []
        for index in indices:
            permit = self.permits[index]
            forward = 0
            terminal, actions, writes = [], [], []
            def work(event):
                nonlocal forward
                self.on_work(event)
                if event['kind'] in ('generation_forward', 'student_trunk_forward'):
                    forward += event['valid_tokens']
                    if forward > cap:
                        raise RuntimeError('bounded FIT input-token cost exhausted')
            def progress(event):
                self.on_event(event)
                if event['kind'] == 'RTMA_TASK_TERMINAL':
                    terminal.append(event['result'])
                elif event['kind'] == 'RTMA_ACTION':
                    actions.append(event['action'])
                elif event['kind'] == 'RTMA_COMMIT':
                    r = event['receipt']['manager']['source']
                    writes.append((r['action'], len(r['row_commits'])))
            self.on_work(dict(kind='fit_task_reserved', permit=permit.identity, rule=rule.identity))
            state = factory.new_state(seed=0, tenant='fit/'+permit.identity)
            episode = self.open_episode(permit, 0, work)
            # This existing executor owns close/failure propagation. No reward is
            # returned before close succeeds; an I/O/native failure is not loss=1.
            rows = run_fit_task([episode], [state], expected_permits=(permit.identity,),
                role='FIT', control='candidate', on_work=work, on_progress=progress)
            if (len(rows) != 1 or terminal != rows or len(actions) != rows[0]['steps']
                    or len(writes) != rows[0]['feedback_commits'] or not 0 < forward <= cap):
                raise ValueError('complete closed task and action/write accounting required')
            outcome = dict(rows[0], behavior=dict(action_sequence_sha256=digest(actions),
                write_sequence_sha256=digest(writes), actions=len(actions), decisions=len(writes)))
            endpoint = dict(schema='orbit-fit-return-endpoint-v1', status='COMPLETE',
                source=self.source, selection=self.selection['identity'], partition=PARTITION,
                seed=self.seed, world_seed=0, reader=self.reader, execution_reader=execution_reader,
                rule=rule_wire(rule), permit=asdict(permit), outcome=outcome,
                forward_input_tokens=forward, forward_input_cap=cap, cost=forward/cap,
                teacher_actions_requested=0, all_environment_processes_closed=True)
            self.on_event(dict(kind='FIT_RETURN_ENDPOINT_COMPLETE', endpoint=endpoint))
            ids.append(permit.task_id); losses.append(outcome['loss'])
            costs.append(forward/cap); seals.append(digest(endpoint))
        factory._check_binding()
        if self.baseline.reader_identity() != self.reader:
            raise ValueError('FIT reward evaluation changed its task reader')
        return FitReturns(tuple(ids), rule.identity, self.reader, tuple(losses), tuple(costs), tuple(seals))
