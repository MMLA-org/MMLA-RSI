"""Complete-task RTMA evaluation with trained Phi fixed and private episodes.

The scientific entry must authorize the exact block before constructing any
episode. This module does not open an audit, change a parent registry, retry an
operational failure or turn one into an ordinary task loss.
"""
from dataclasses import asdict
import hashlib
import torch

from mmla_pdsa.full.alfworld_episode import ReplayEpisode, _digest, _sha
from scoped_generation import generate_action_batch
from task_accounting import TaskAccounting
from mmla_pdsa.full.outcome_losses import outcome_losses
from mmla_pdsa.full.outcome_train_batch import _text_batch, _memories, _decode
from rule_binding import BoundManagedTaskState, ManagedCandidateFactory
from logical_episode import LogicalWorldEpisode, LogicalWorldPermit


def _episode_ok(episode, role):
    return ((type(episode) is ReplayEpisode and role == 'DEV')
            or type(episode) is LogicalWorldEpisode) and episode.permit.role == role


def run_fixed_phi_batch(episodes, states, *, expected_permits, role, control,
                        on_work, on_progress):
    """Execute every provided world to its terminal outcome under one rule.

control='candidate' applies V; control='frozen' keeps only common initial M.
Both use the same generation limits. Actual feedback-scoring costs are recorded
separately; frozen execution does not spend that unused scoring computation.
"""
    primary = None; result = None
    try:
        if (not episodes or len(episodes) != len(states) or len(states) > 8
                or len({id(e) for e in episodes}) != len(episodes)
                or len({id(s) for s in states}) != len(states)
                or role not in ('DEV', 'A_RSI1', 'A_RSI2')
                or control not in ('candidate', 'frozen')
                or not all(_episode_ok(e, role) for e in episodes)
                or tuple(e.permit.identity for e in episodes) != tuple(expected_permits)
                or len(set(expected_permits)) != len(expected_permits)
                or not all(type(s) is BoundManagedTaskState for s in states)
                or type(on_work) is not TaskAccounting or not callable(on_progress)):
            raise ValueError('exact private fixed-Phi states and authorized complete block required')
        factory = states[0]._factory
        if (type(factory) is not ManagedCandidateFactory or factory.task_policy != 'fixed_rho'
                or factory.rule.mode != 'rtma'):
            raise ValueError('explicit fixed-Phi RTMA factory required')
        factory._check_binding(); model = factory.model; device = model.rho.a.device
        identities = []; rows = [None]*len(states); live = list(range(len(states)))
        reads_after_write = [0]*len(states); last_write = [None]*len(states)
        for i,(ep,state) in enumerate(zip(episodes, states)):
            start=ep.checkpoint()
            rng=torch.Generator(device='cpu').manual_seed(ep.seed).get_state()
            if (state._factory is not factory or state.model is not model
                    or state.max_updates != 0 or state.fast.steps != 0 or state.step != 0
                    or state.frame is not None or state.pending is not None or state.receipts
                    or state.previous_loss != 0. or state.source.version != 0 or ep.terminal or start.steps
                    or not torch.equal(state.fast.initial_rng,rng)
                    or not torch.equal(state.fast.rng_state,rng)
                    or ep.limits.max_steps != 64 or ep.limits.expose_admissible_commands):
                raise ValueError('every world starts with its own unstarted state at trained rho')
            state._check(); identities.append(state.fast.identity)
            state.initialize(ep.visible, source_receipt=_digest(asdict(start)))
            on_progress(dict(kind='RTMA_INITIAL', task=ep.permit.task_id,
                permit=ep.permit.identity, control=control, receipt=state.receipts[-1]))
        while live:
            contexts = [states[i].action_input() for i in live]
            for i in live:
                if last_write[i] is not None and states[i].source.version == last_write[i]:
                    reads_after_write[i] += 1
            tokens = generate_action_batch(model, [c.prompt for c in contexts],
                fast_states=[c.fast for c in contexts], memories=[c.memory for c in contexts],
                eos_ids={factory.formatter._eos}, read_binding=_digest([c.binding for c in contexts]),
                accounting=on_work)
            if len(tokens) != len(live):raise ValueError('one generated command per live world')
            feedback = []; ongoing = []
            for i,ctx,ts in zip(live, contexts, tokens):
                state,ep = states[i],episodes[i]
                if ep.visible != ctx.frame:raise ValueError('current frame changed before action')
                action = _decode(factory.formatter.tokenizer, ts, {factory.formatter._eos})
                pair = ep.step(action)
                if pair.before != ctx.frame or pair.action != action:
                    raise ValueError('arrived feedback differs from executed action/prefix')
                seal = _digest(asdict(pair.receipt))
                on_progress(dict(kind='RTMA_ACTION', task=ep.permit.task_id, step=state.step,
                    action=action, tokens=len(ts), phi=ctx.fast.identity, read=ctx.source_read,
                    source_version=state.source.version, receipt=asdict(pair.receipt),terminal=pair.terminal))
                if pair.terminal:
                    state.finish_terminal(ctx, source_receipt=seal, action=action)
                    outcome = ep.outcome
                    if type(outcome['won']) is not bool:raise ValueError('real terminal Boolean required')
                    commits = [r for r in state.receipts if r['kind'] == 'ARRIVED_FEEDBACK']
                    if any(r['update'] is not None for r in commits):raise ValueError('fixed Phi was updated')
                    rows[i] = dict(task=ep.permit.task_id, permit=ep.permit.identity,
                        source_floor=ep.permit.floor, control=control, **outcome,
                        loss=1.-float(outcome['won']), phi=state.fast.identity,
                        numerical_commits=0, feedback_commits=len(commits),
                        source_writes=sum(r['manager']['selected'] != 0 for r in commits),
                        source_nulls=sum(r['manager']['selected'] == 0 for r in commits),
                        frozen_feedback=sum(r['kind']=='FROZEN_FEEDBACK' for r in state.receipts),
                        post_write_reads=reads_after_write[i], common_initial_writes_excluded=True)
                    # Preserve the original endpoint even if a peer later fails.
                    # A partial set remains unavailable to selection/admission.
                    on_progress(dict(kind='RTMA_TASK_TERMINAL',result=rows[i]))
                else:
                    ongoing.append(i)
                    if control == 'frozen':
                        state.freeze_feedback(ctx, after=pair.after, source_receipt=seal)
                    else:
                        text = state.feedback_text(ctx, action=action, after=pair.after)
                        feedback.append((i,ctx,pair,seal,len(ts),text))
            if feedback:
                texts = [x[5] for x in feedback]; contexts = [x[1] for x in feedback]
                with torch.no_grad():
                    losses,_ = outcome_losses(model, *_text_batch(texts,device),
                        fast_states=[c.fast for c in contexts],memory=_memories(model,contexts),
                        head_chunk=32,on_work=on_work)
                if losses.ndim != 1 or len(losses) != len(feedback):
                    raise ValueError('one observed-feedback loss per world required')
                for j,(i,ctx,pair,seal,n,_) in enumerate(feedback):
                    state = states[i]
                    record = state.commit_feedback(ctx,action=pair.action,after=pair.after,
                        source_receipt=seal,loss=losses[j],gradient=None,allow_update=False,action_tokens=n)
                    if record['manager']['selected'] != 0:last_write[i]=state.source.version
                    on_progress(dict(kind='RTMA_COMMIT',task=episodes[i].permit.task_id,receipt=record))
                del losses
            for i,state in enumerate(states):
                if state.fast.identity != identities[i] or state.fast.steps != 0:
                    raise ValueError('actual task Phi must remain at its initial trained rho')
            factory._check_binding(); live = ongoing
        if any(r is None for r in rows):raise RuntimeError('incomplete task block')
        result = rows
    except BaseException as exc:
        primary = exc
        for ep,state in zip(episodes,states):
            if type(state) is BoundManagedTaskState:state.failed = True
            try:
                on_progress(dict(kind='RTMA_INCOMPLETE',task=ep.permit.task_id,
                    step=state.step,error=repr(exc)[:512],source=state.source.identity,
                    phi=state.fast.identity,last_receipt=state.receipts[-1] if state.receipts else None))
            except BaseException as reporting:exc.add_note('partial evidence error: '+repr(reporting))
        raise
    finally:
        errors=[]; closed=set()
        for ep in episodes:
            if id(ep) in closed:continue
            closed.add(id(ep))
            try:ep.close()
            except BaseException as exc:errors.append(repr(exc))
        if errors:
            if primary is not None:primary.add_note('cleanup failures: '+'; '.join(errors))
            else:raise RuntimeError('task block cleanup incomplete: '+'; '.join(errors))
    # No complete block is returned before all environments close successfully.
    return result


def evaluate_logical_block(factory, permits, *, role, block_sha256, population_sha256,
                           pair_seed, open_episode, on_work, on_progress,
                           microbatch=8, control='candidate'):
    """Fresh per-world state and same parent/child random tapes; no audit unlock."""
    _sha(block_sha256); _sha(population_sha256)
    if (type(factory) is not ManagedCandidateFactory or factory.task_policy != 'fixed_rho'
            or type(pair_seed) is not int or not 0 <= pair_seed < 2**32
            or type(microbatch) is not int or not 1 <= microbatch <= 8
            or type(permits) is not tuple or not 1 <= len(permits) <= 4096
            or len({p.identity for p in permits}) != len(permits)
            or len({p.semantic_sha256 for p in permits}) != len(permits)):
        raise ValueError('bounded exact logical-world block required')
    for p in permits:
        if (type(p) is not LogicalWorldPermit or p.role != role or p.block_sha256 != block_sha256
                or p.population_sha256 != population_sha256):
            raise ValueError('world role/population/block mismatch')
    factory._check_binding(); rows=[]
    cap_per_world=64*factory.model.profile.max_new_tokens*factory.model.profile.local_tokens+63*4096
    cap=cap_per_world*len(permits)
    work=TaskAccounting(on_work,cap)
    for offset in range(0,len(permits),microbatch):
        batch=permits[offset:offset+microbatch]; episodes=[]; states=[]
        try:
            for permit in batch:
                # Seed0 is also the finite certificate/replay seed. Never derive
                # a world's tape from permit.identity: that contains the entire
                # block and cohort lock and would couple otherwise fresh worlds.
                # pair_seed labels the repeated reader condition, not new data.
                seed=0
                state=factory.new_state(seed=seed,tenant='world/'+permit.identity)
                work(dict(kind='task_reserved',permit=permit.identity))
                episode=open_episode(permit,seed,work)
                episodes.append(episode);states.append(state)
                if type(episode) is not LogicalWorldEpisode or episode.permit!=permit or episode.seed!=seed:
                    raise ValueError('exact independently created derived episode required')
        except BaseException as exc:
            for e in episodes:
                try:e.close()
                except BaseException as cleanup:exc.add_note('setup cleanup: '+repr(cleanup))
            raise
        rows.extend(run_fixed_phi_batch(episodes,states,expected_permits=tuple(p.identity for p in batch),
            role=role,control=control,on_work=work,on_progress=on_progress))
    factory._check_binding()
    return dict(schema='orbit-fixed-phi-logical-block-v1',status='COMPLETE',role=role,
        block_sha256=block_sha256,population_sha256=population_sha256,
        reader=factory.reader_identity(),rule=asdict(factory.rule),control=control,
        pair_seed=pair_seed,worlds=rows,mean_loss=sum(r['loss'] for r in rows)/len(rows),
        forward_input_tokens=work.forward_inputs,forward_input_cap=cap,mean_cost=work.forward_inputs/cap,
        cost_scope='bounded input-token proxy; physical and failed costs require the native ledger',
        independent_unit='sampled complete logical world under the admitted population contract',
        admission_performed=False,successor_installed=False)
