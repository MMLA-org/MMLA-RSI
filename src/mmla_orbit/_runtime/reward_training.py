"""Task-return learning of slow proposal rho; task and online Phi stay fixed.

The isolated native owner supplies authenticated original models/FIT endpoints.
No launcher or data acquisition is exposed here. The course saves both learning
and frozen-process evidence, and performs the actual step14->15 restore within
its one finite run. A failed course is not automatically resumed.
"""
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import os
import torch
from safetensors.torch import load
from mmla_pdsa.full.safe_state import save_state, decode, SCHEMA, exact_equal
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.extensions.rsi_binding import isolated_model
from content_evidence import EvidenceMemory
from content_rules import identity
from edit_role import EditProcess, model_guard
from fit_reward import FitReturns
from proposal_edit import batch_task_indices, action_logprobs
from reward_update import RewardBatch, make_optimizer, apply_reward_update
from training_state import _write, _json, _decode, sha

STATUS = 'TRAINED_PROCESS_CREDIT_EDIT_PROPOSAL'


def save_artifact(directory, value, *, source, kind):
    """One data-only committed artifact; used for batches and complete states."""
    sha(source)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    save_state(root/'state', dict(source=source, kind=kind, value=value))
    files = []
    for name in ('state.json', 'state.safetensors'):
        p = root/name
        with p.open('rb') as handle:
            os.fsync(handle.fileno())
            data = handle.read()
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    seal = dict(schema='orbit-task-return-artifact-v1', source=source, kind=kind, files=files)
    raw = _json(seal)
    _write(root/'COMMITTED.json', raw)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(path=str(root), manifest_sha256=hashlib.sha256(raw).hexdigest())


def read_artifact(directory, *, source, kind, manifest_sha256):
    """At the file boundary, authenticate and decode the same verified bytes."""
    root = Path(directory)
    expected = {'COMMITTED.json', 'state.json', 'state.safetensors'}
    if (root.is_symlink() or {p.name for p in root.iterdir()} != expected
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())
            or (root/'COMMITTED.json').stat().st_size > 4096):
        raise ValueError('exact committed artifact inventory required')
    raw = (root/'COMMITTED.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError('artifact commit bytes changed')
    meta = _decode(raw)
    if (set(meta) != {'schema', 'source', 'kind', 'files'}
            or meta['schema'] != 'orbit-task-return-artifact-v1'
            or meta['source'] != source or meta['kind'] != kind
            or [f['path'] for f in meta['files']] != ['state.json', 'state.safetensors']):
        raise ValueError('artifact source or purpose changed')
    payloads = {}
    for item in meta['files']:
        path = root/item['path']
        if set(item) != {'path', 'bytes', 'sha256'} or not 0 < item['bytes'] <= 128*1024**2 or path.stat().st_size != item['bytes']:
            raise ValueError('exact bounded artifact file required')
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('artifact payload bytes changed')
        payloads[item['path']] = data
    tree = _decode(payloads['state.json'])
    if set(tree) != {'schema', 'tree'} or tree['schema'] != SCHEMA:
        raise ValueError('data-only serialized state required')
    tensors = load(payloads['state.safetensors']); used = set()
    payload = decode(tree['tree'], tensors, used)
    if used != set(tensors) or set(payload) != {'source', 'kind', 'value'} or payload['source'] != source or payload['kind'] != kind:
        raise ValueError('complete source-bound payload required')
    return payload['value']


class RewardTrainer:
    def __init__(self, model, *, course_identity):
        if int(model.rho.version) != 156:
            raise ValueError('retained original proposal rho156 required')
        self.model, self.course_identity = model, course_identity
        self.optimizer = make_optimizer(model)
        self.steps = 0
        self.finished = False
        self.frozen = tuple(row for row in model_guard(model) if row[0] not in ('rho.a', 'rho.b', 'rho.version'))
        self.initialization = copy.deepcopy(model.rho.initialization)
        self.statuses = dict(model.module_status)
        self.profile, self.substrate = model.profile.identity, copy.deepcopy(model.substrate_identity)

    def check(self):
        expected = dict(self.statuses, **({'rho': STATUS} if self.finished else {}))
        if (tuple(row for row in model_guard(self.model) if row[0] not in ('rho.a', 'rho.b', 'rho.version')) != self.frozen
                or int(self.model.rho.version) != 156+self.steps
                or self.model.module_status != expected or self.model.profile.identity != self.profile
                or self.model.substrate_identity != self.substrate
                or any(m.training for m in self.model.modules())
                or not exact_equal(self.model.rho.initialization, self.initialization)
                or tuple(n for n, p in self.model.named_parameters() if p.requires_grad) !=
                   (() if self.finished else ('rho.a', 'rho.b'))):
            raise ValueError('slow-rho ownership, original initialization or frozen computation changed')

    def step(self, batch, *, on_work):
        self.check()
        if self.finished or self.steps >= 28:
            raise RuntimeError('finite proposal course has ended')
        on_work(dict(kind='edit_reward_update_reserved', updates=1))
        result = apply_reward_update(self.model, self.optimizer, batch, on_work=on_work)
        self.steps += 1
        self.check()
        return dict(step=self.steps, **result)

    def state(self):
        self.check()
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError('checkpoint requires a completed update with no pending gradient')
        return dict(schema='orbit-process-credit-trainer-v1', course=self.course_identity, steps=self.steps,
            finished=self.finished, modules=self.model.new_state_dict(),
            initialization=copy.deepcopy(self.model.rho.initialization), statuses=dict(self.model.module_status),
            profile=self.profile, substrate=copy.deepcopy(self.substrate),
            optimizer=copy.deepcopy(self.optimizer.state_dict()), torch_rng=torch.get_rng_state().clone(),
            cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()] if self.model.rho.a.is_cuda else [])

    def restore(self, state):
        """Restore only into a fresh private trainer; any failure aborts its owner."""
        self.check()
        if (self.steps or self.finished or set(state) != {'schema', 'course', 'steps', 'finished', 'modules',
                'initialization', 'statuses', 'profile', 'substrate', 'optimizer', 'torch_rng', 'cuda_rng'}
                or state['schema'] != 'orbit-process-credit-trainer-v1' or state['course'] != self.course_identity
                or type(state['steps']) is not int or not 0 <= state['steps'] <= 28
                or type(state['finished']) is not bool or (state['finished'] and state['steps'] != 28)
                or state['profile'] != self.profile or state['substrate'] != self.substrate
                or not exact_equal(state['initialization'], self.initialization)
                or state['statuses'] != dict(self.statuses, **({'rho': STATUS} if state['finished'] else {}))):
            raise ValueError('fresh original-rho trainer and exact course state required')
        current = self.model.new_state_dict()
        if set(state['modules']) != set(current):
            raise ValueError('complete new-module inventory required')
        for name, expected in current.items():
            value = state['modules'][name]
            if (not torch.is_tensor(value) or value.shape != expected.shape or value.dtype != expected.dtype
                    or not torch.isfinite(value).all()
                    or (not name.startswith('rho.') and not exact_equal(value, expected))):
                raise ValueError('new-module shape/numeric/frozen identity mismatch: '+name)
        if int(state['modules']['rho.version']) != 156+state['steps']:
            raise ValueError('saved rho lineage mismatch')
        opt = state['optimizer']; initial_opt = self.optimizer.state_dict()
        if set(opt) != {'state', 'param_groups'} or opt['param_groups'] != initial_opt['param_groups']:
            raise ValueError('fixed optimizer configuration/order required')
        ids = initial_opt['param_groups'][0]['params']
        if set(opt['state']) != (set(ids) if state['steps'] else set()):
            raise ValueError('complete AdamW state required')
        for index, parameter in zip(ids, (self.model.rho.a, self.model.rho.b)):
            if not state['steps']:
                continue
            slots = opt['state'][index]
            if (set(slots) != {'step', 'exp_avg', 'exp_avg_sq'}
                    or not torch.is_tensor(slots['step']) or slots['step'].shape != torch.Size([])
                    or slots['step'].dtype != torch.float32 or slots['step'].device.type != 'cpu'
                    or float(slots['step']) != state['steps']):
                raise ValueError('AdamW cursor mismatch')
            for name in ('exp_avg', 'exp_avg_sq'):
                value = slots[name]
                if not torch.is_tensor(value) or value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise ValueError('AdamW moment layout/numeric mismatch')
            if (slots['exp_avg_sq'] < 0).any():
                raise ValueError('negative AdamW second moment')
        torch.Generator(device='cpu').set_state(state['torch_rng'])
        devices = torch.cuda.device_count() if self.model.rho.a.is_cuda else 0
        if len(state['cuda_rng']) != devices:
            raise ValueError('complete device random state required')
        for i, rng in enumerate(state['cuda_rng']):
            torch.Generator(device='cuda:'+str(i)).set_state(rng)
        self.model.rho.load_state_dict({k[4:]: v for k, v in state['modules'].items() if k.startswith('rho.')}, strict=True)
        self.optimizer.load_state_dict(opt)
        torch.set_rng_state(state['torch_rng'])
        if devices:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        self.steps, self.finished = state['steps'], state['finished']
        if self.finished:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.module_status['rho'] = STATUS
        self.check()
        if not exact_equal(self.state(), state):
            raise ValueError('restored full numerical state differs from saved state')

    def finish(self):
        self.check()
        if self.steps != 28:
            raise ValueError('all 28 updates required before finishing')
        self.finished = True
        self.model.module_status['rho'] = STATUS
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.check()


def train_return_course(initial_proposal, evaluator, interface, memory0, *, parent_returns: FitReturns, seed, directory,
                        source_identity, on_work, on_event):
    """28 paired batches; frozen control gets the same four task/draw opportunities.

    Main and restored step15 consume the same already-sampled batch. Only the
    restored branch enters the delivered lineage. The native role catches any
    operational failure once, saves its partial files/state and closes workers.
    """
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    physical_work = {}; original_on_work = on_work
    def on_work(event):
        kind = event['kind']; physical_work[kind] = physical_work.get(kind, 0) + 1
        original_on_work(event)
    if (int(initial_proposal.rho.version) != 156 or seed != evaluator.seed
            or any(p.requires_grad for p in initial_proposal.parameters())
            or any(m.training for m in initial_proposal.modules())):
        raise ValueError('frozen authentic original proposal and matching reader required')
    initial_modules = tensor_digest(initial_proposal.new_state_dict())
    original_guard = model_guard(initial_proposal)
    parent = evaluator.baseline.manager
    schedule = batch_task_indices(0) + batch_task_indices(1)
    course = dict(schema='orbit-process-credit-course-v1', source=source_identity, seed=seed,
        selection=evaluator.selection['identity'], task_reader=evaluator.reader,
        original_proposal=initial_modules, parent=parent.identity, interface=interface.identity,
        memory0=memory0.identity, schedule=schedule, initial_rho=156, final_rho=184)
    course_id = identity(course)
    _write(root/'course.json', _json(dict(course, identity=course_id)))
    if (parent_returns.task_ids != tuple(p.task_id for p in evaluator.permits)
            or parent_returns.reader != evaluator.reader or parent_returns.rule != parent.identity
            or len(parent_returns.losses) != 56 or len(parent_returns.endpoint_identities) != 56):
        raise ValueError('this new cohort must supply its own complete closed parent reference')
    save_artifact(root/'parent-reference', asdict(parent_returns), source=source_identity, kind='fit-parent-reference')
    trainer = RewardTrainer(isolated_model(initial_proposal), course_identity=course_id)
    effective = physical = physical_attempts = 0
    midpoint = None
    process = None
    active_trainer = trainer
    try:
        for batch_index, indices in enumerate(schedule):
            task_ids = tuple(evaluator.permits[i].task_id for i in indices)
            batch_losses = tuple(parent_returns.losses[i] for i in indices)
            process_states = {}; results = {}; draws = {}
            for process_name, model in (('learn', trainer.model), ('frozen-rho', initial_proposal)):
                memory = EvidenceMemory.restore(memory0.export(), count_tokens=memory0._count_tokens,
                                                 expected_contract=memory0.contract)
                process = EditProcess(model, interface, parent, memory, control='keep', seed=seed,
                    phase='fit', round_index=batch_index, on_work=on_work, on_event=on_event)
                process_draws = []; process_returns = []
                # Retained immediately on disk after each completed candidate, before
                # a subsequent candidate/native failure can prevent batch delivery.
                for attempt in range(4):
                    draw = process.draw()
                    returns = evaluator.evaluate(draw.sample.program, indices)
                    if returns.task_ids != task_ids or returns.rule != draw.sample.program.identity or returns.reader != evaluator.reader:
                        raise ValueError('task return is not bound to the actual sampled complete program')
                    process.feedback(draw, loss=sum(returns.losses)/4, cost=sum(returns.costs)/4, tag='completed')
                    process_draws.append(draw); process_returns.append(returns)
                    save_artifact(root/f'batch-{batch_index:02d}'/process_name/f'attempt-{attempt}',
                        dict(draw=asdict(draw), returns=asdict(returns), state=process.export()),
                        source=source_identity, kind='fit-sampled-edit-return')
                draws[process_name] = process_draws
                results[process_name] = process_returns
                process_states[process_name] = process.export()
            if model_guard(initial_proposal) != original_guard:
                raise ValueError('frozen original proposal/shared modules changed')
            samples = tuple(d.sample for d in draws['learn'])
            batch = RewardBatch(int(trainer.model.rho.version), task_ids,
                tuple(x.task_ids for x in results['learn']), batch_losses,
                tuple(x.losses for x in results['learn']), samples)
            # Authentication is supplied by the private process objects above, not
            # by a digest of caller-provided arbitrary samples.
            pending = dict(batch=asdict(batch), processes=process_states,
                all_returns={k: [asdict(v) for v in values] for k, values in results.items()},
                course=course_id, batch_index=batch_index)
            if batch_index == 14:
                pre = dict(trainer=trainer.state(), pending=pending)
                midpoint = save_artifact(root/'step-14-before-15', pre,
                    source=source_identity, kind='return-training-checkpoint')
                restored_pre = read_artifact(midpoint['path'], source=source_identity,
                    kind='return-training-checkpoint', manifest_sha256=midpoint['manifest_sha256'])
                if not exact_equal(pre, restored_pre):
                    raise ValueError('actual saved batch/RNG/model prestate failed byte restoration')
                active_trainer = trainer
                physical_attempts += 1
                live = trainer.step(batch, on_work=on_work); physical += 1
                expected = trainer.state()
                live_ref = save_artifact(root/'step-15-live', dict(trainer=expected, pending=pending),
                    source=source_identity, kind='return-training-restore-branch')
                resumed = RewardTrainer(isolated_model(initial_proposal), course_identity=course_id)
                resumed.restore(restored_pre['trainer'])
                restored_before_ref = save_artifact(root/'step-14-restored-before-15',
                    dict(trainer=resumed.state(), pending=pending), source=source_identity, kind='return-training-restore-branch')
                active_trainer = resumed
                physical_attempts += 1
                actual = resumed.step(batch, on_work=on_work); physical += 1; effective += 1
                resumed_state = resumed.state()
                restored_ref = save_artifact(root/'step-15-restored', dict(trainer=resumed_state, pending=pending),
                    source=source_identity, kind='return-training-restore-branch')
                comparison = dict(state_equal=exact_equal(expected, resumed_state),
                    update_equal=exact_equal(live, actual), midpoint=midpoint, live=live_ref,
                    restored_before=restored_before_ref, restored=restored_ref,
                    retained_lineage='restored', extra_physical_updates=1)
                _write(root/'restore-comparison.json', _json(comparison))
                if not comparison['state_equal'] or not comparison['update_equal']:
                    raise ValueError('actual live/restored update differs')
                trainer = resumed
                del resumed, pre, restored_pre, expected, resumed_state
                on_event(dict(kind='RETURN_RESTORE_CHECK_PASSED', seed=seed, comparison=comparison))
                result = actual
            else:
                active_trainer = trainer
                physical_attempts += 1
                result = trainer.step(batch, on_work=on_work); physical += 1; effective += 1
            on_work(dict(kind='edit_effective_update',updates=1))
            # The pre-update batch is retained even when advantages are all zero.
            save_artifact(root/f'batch-{batch_index:02d}'/'update', dict(pending=pending, result=result),
                          source=source_identity, kind='return-training-update')
            on_event(dict(kind='RETURN_TRAINING_STEP', seed=seed, batch=batch_index,
                effective_updates=effective, physical_updates=physical, update=result))
        trainer.finish()
        final = save_artifact(root/'final', trainer.state(), source=source_identity, kind='return-training-final')
        restored = RewardTrainer(isolated_model(initial_proposal), course_identity=course_id)
        restored.restore(read_artifact(final['path'], source=source_identity, kind='return-training-final',
                                      manifest_sha256=final['manifest_sha256']))
        # No extra draw: compare both final models on the same saved action's
        # site and subtree prefixes. This is restoration, not on-policy replay
        # against rho183's stored probabilities after advancing to rho184.
        with torch.no_grad():
            before = list(action_logprobs(trainer.model, batch.samples[0], on_work=on_work))
            after = list(action_logprobs(restored.model, batch.samples[0], on_work=on_work))
        equal = len(before) == len(after) and all(
            a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(before, after))
        if (not equal or effective != 28 or physical != 29 or midpoint is None
                or physical_attempts != 29
                or physical_work.get('edit_reward_backward',0) != 116):
            raise ValueError('final joint probability or physical training accounting mismatch')
        if model_guard(initial_proposal) != original_guard or tensor_digest(initial_proposal.new_state_dict()) != initial_modules:
            raise ValueError('retained original changed during proposal training')
        receipt = dict(schema='orbit-process-credit-proposal-trained-v1', status='COMPLETE_PENDING_NATIVE_SETTLEMENT',
            source=source_identity, course=course_id, seed=seed, selection=evaluator.selection['identity'],
            initial_rho_version=156, final_rho_version=184, effective_updates=effective, physical_updates=physical,
            sampled_candidates=224, candidate_task_endpoints=896, parent_task_endpoints=56,
            learned_process_batches=28, frozen_process_batches=28,
            backward_candidates=physical_work.get('edit_reward_backward',0),
            successful_update_backward_candidates=116, interrupted_backward_candidates=0,
            physical_update_calls=physical_attempts, failed_update_calls=0,
            segment_work=physical_work, retained_binding=None,
            segment_replay_forwards=physical_work.get('edit_replay_forward',0),
            segment_site_replay_forwards=physical_work.get('site_replay_forward',0),
            physical_backward_calls=physical_work.get('edit_replay_backward',0)+physical_work.get('site_replay_backward',0),
            original_modules=initial_modules, trained_modules=tensor_digest(restored.model.new_state_dict()),
            midpoint=midpoint, final=final, final_probability_forwards=2*(1+len(batch.samples[0].tokens)),
            final_probe_draws=0, final_probe_task_calls=0,
            exact_final_probability_equal=True,
            numerical_deployment_updates=0, task_reader_changed=False, original_initialization_preserved=True,
            independent_RSI_benefit_claim=False, publication='HOLD')
        _write(root/'COMPLETE.json', _json(receipt))
        return restored.model, receipt
    except BaseException as error:
        partial = dict(status='INCOMPLETE', error=repr(error), effective_updates=effective,
            completed_physical_updates=physical, physical_update_calls=physical_attempts,
            course=course_id, automatic_retry=False,
            process=None if process is None else process.export(),
            rho=copy.deepcopy(dict(active_trainer.model.rho.state_dict())),
            rho_gradients={k:None if getattr(active_trainer.model.rho,k).grad is None else getattr(active_trainer.model.rho,k).grad.detach().clone() for k in ('a','b')},
            optimizer=copy.deepcopy(active_trainer.optimizer.state_dict()),
            torch_rng=torch.get_rng_state().clone(),
            cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()] if active_trainer.model.rho.a.is_cuda else [])
        try:
            save_artifact(root/'INCOMPLETE', partial, source=source_identity, kind='return-training-incomplete')
        except BaseException as reporting:
            error.add_note('partial artifact could not be saved: '+repr(reporting))
        raise
