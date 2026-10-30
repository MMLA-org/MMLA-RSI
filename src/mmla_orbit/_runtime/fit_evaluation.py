"""Complete original FIT task endpoints for the finite proposal curriculum.

This isolated runner uses the same fixed-Phi task computation as the reviewed
RTMA evaluator, with an explicit FIT-only, scalar, seed-zero permission boundary.
No FIT role is relabelled DEV. No teacher action accessor is called. Native
mounts, complete resource accounting and scientific admission remain mandatory.
"""
from dataclasses import asdict
import re
import torch
from mmla_pdsa.full.alfworld_episode import ReplayEpisode, TaskPermit, _digest, _sha
from mmla_pdsa.full.outcome_generation import generate_action_batch
from mmla_pdsa.full.outcome_losses import outcome_losses
from mmla_pdsa.full.outcome_train_batch import _text_batch, _memories, _decode
from rule_binding import BoundManagedTaskState, ManagedCandidateFactory
from demonstration_factory import DemonstrationFactory
from curriculum import SEEDS, bank_rules, bind_parent, digest

PARTITION = 'a0a5a4b8f163785f599a2271e0e3c1c46ab0b3801d9289a79cbecfc4062e7e11'

def rule_wire(rule):
    return dict(mode=rule.mode,symbols=list(rule.symbols),language=rule.language)

def run_fit_task(episodes, states, *, expected_permits, role, control,
                        on_work, on_progress):
    """Execute every provided world to its terminal outcome under one rule.

Only control='candidate' is accepted, including the actual NULL program.
The inherited inactive frozen branch is unreachable; no alternate FIT baseline
silently skips candidate feedback costs.
"""
    primary = None; result = None
    try:
        if (not episodes or len(episodes) != len(states) or len(states) != 1
                or len({id(e) for e in episodes}) != len(episodes)
                or len({id(s) for s in states}) != len(states)
                or role != 'FIT'
                or control != 'candidate'
                or not all(type(e) is ReplayEpisode and e.permit.role == role and e.seed == 0 for e in episodes)
                or tuple(e.permit.identity for e in episodes) != tuple(expected_permits)
                or len(set(expected_permits)) != len(expected_permits)
                or not all(type(s) is BoundManagedTaskState for s in states)
                or not callable(on_work) or not callable(on_progress)):
            raise ValueError('exact private fixed-Phi states and authorized complete block required')
        factory = states[0]._factory
        if (type(factory) is not ManagedCandidateFactory or factory.task_policy != 'fixed_rho'
                or factory.rule.mode != 'rtma' or int(factory.model.rho.version)!=942
                or factory.model.profile.memory_slots!=8):
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
                on_forward=lambda n,valid,padded:on_work(dict(kind='generation_forward',rows=n,
                    valid_tokens=valid,padded_positions=padded)),
                on_tokens=lambda n,ts:on_work(dict(kind='generated_tokens',count=len(ts))))
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


def check_fit_permits(selection,permits):
    if (type(selection) is not dict
            or selection.get('identity')!=digest({k:v for k,v in selection.items() if k!='identity'})
            or type(permits) is not tuple or len(permits)!=56
            or len({p.identity for p in permits})!=56):
        raise ValueError('exact complete selected FIT permit block required')
    selected=[r for g in selection['groups'] for side in ('support','query') for r in g[side]]
    if len(selected)!=56:raise ValueError('complete 56-task inventory required')
    for row,permit in zip(selected,permits):
        if (type(permit) is not TaskPermit or permit.role!='FIT' or row['role']!='FIT'
                or permit.task_id!=row['task_id'] or permit.floor!=row['floor_plan']
                or permit.game_sha256!=row['game_sha256']
                or permit.trajectory_sha256!=row['trajectory_sha256']
                or permit.partition_sha256!=PARTITION):
            raise ValueError('original source/role/order mismatch')
    return selected


def evaluate_fit_rule(baseline,rule,permits,*,selection,seed,open_episode,on_work,on_event):
    """Execute all 56 selected tasks once under one bank rule and fixed reader.

    Individual completed endpoints are preserved; a complete result is returned
    only after every environment closes. Failed endpoints never become labels.
    """
    check_fit_permits(selection,permits)
    if type(baseline) is not DemonstrationFactory or seed not in SEEDS:
        raise ValueError('fixed corrected task reader required')
    rules=bank_rules(baseline.language,actual_parent=baseline.manager)
    if rule not in rules.values():raise ValueError('only the registered four content programs')
    baseline_reader=baseline.reader_identity()
    factory=ManagedCandidateFactory(baseline,rule,task_policy='fixed_rho')
    execution_reader=factory.reader_identity()
    cap=64*factory.model.profile.max_new_tokens*factory.model.profile.local_tokens+63*4096
    endpoints=[]
    for permit in permits:
        forward=0;terminal=[];actions=[];writes=[]
        def work(event):
            nonlocal forward
            on_work(event)
            if event['kind'] in ('generation_forward','student_trunk_forward'):
                count=event.get('valid_tokens')
                if type(count) is not int or count<0:raise ValueError('actual input-token count required')
                forward+=count
                if forward>cap:raise RuntimeError('registered complete FIT task token cap exhausted')
        def progress(event):
            on_event(event)
            if event['kind']=='RTMA_TASK_TERMINAL':terminal.append(event['result'])
            if event['kind']=='RTMA_ACTION':actions.append(event['action'])
            if event['kind']=='RTMA_COMMIT':
                source=event['receipt']['manager']['source']
                writes.append((source['action'],len(source['row_commits'])))
        work(dict(kind='fit_task_reserved',permit=permit.identity,rule=rule.identity,reader=baseline_reader))
        state=factory.new_state(seed=0,tenant='fit/'+permit.identity)
        episode=open_episode(permit,0,work)
        # run_fit_task owns and closes the episode even on validation failure.
        if type(episode) is not ReplayEpisode or episode.permit!=permit:
            try:episode.close()
            finally:raise ValueError('exact selected FIT episode required')
        rows=run_fit_task([episode],[state],expected_permits=(permit.identity,),role='FIT',
            control='candidate',on_work=work,on_progress=progress)
        if len(rows)!=1 or terminal!=rows:raise ValueError('saved/returned terminal mismatch')
        if len(actions)!=rows[0]['steps'] or len(writes)!=rows[0]['feedback_commits']:
            raise ValueError('complete actual action/write sequence required')
        rows[0]['behavior']=dict(action_sequence_sha256=digest(actions),write_sequence_sha256=digest(writes),
            actions=len(actions),decisions=len(writes),scope='observed action and write(action,row_count) sequence')
        endpoint=dict(schema='orbit-fit-rule-endpoint-v1',status='COMPLETE',role='FIT',
            seed=seed,world_seed=0,selection=selection['identity'],partition=PARTITION,
            reader=baseline_reader,execution_reader=execution_reader,rule=rule_wire(rule),
            permit=asdict(permit),outcome=rows[0],forward_input_tokens=forward,forward_input_cap=cap,
            cost=forward/cap,teacher_actions_requested=0)
        # Emit only after environment cleanup succeeded; before this event
        # prior terminal logs are partial and not trusted training labels.
        on_event(dict(kind='FIT_ENDPOINT_COMPLETE',endpoint=endpoint))
        endpoints.append(endpoint)
        if baseline.reader_identity()!=baseline_reader:raise ValueError('frozen reader changed')
    factory._check_binding()
    value=dict(schema='orbit-fit-rule-block-v1',status='COMPLETE',role='FIT',seed=seed,
        selection=selection['identity'],partition=PARTITION,reader=baseline_reader,
        execution_reader=execution_reader,rule=rule_wire(rule),world_seed=0,endpoints=endpoints)
    endpoint_records(value,selection=selection,permits=permits,seed=seed,
        reader_identity=baseline_reader,execution_reader=execution_reader,rule=rule,forward_input_cap=cap)
    return value


def endpoint_records(value,*,selection,permits,seed,reader_identity,execution_reader,rule,forward_input_cap):
    """Validate the full produced block before deriving the 56 label records.

    File identity and private producer ownership must also be checked by the
    native host. A digest of arbitrary caller data alone is not provenance.
    """
    check_fit_permits(selection,permits)
    for identity in (reader_identity,execution_reader):_sha(identity)
    if (set(value)!={'schema','status','role','seed','selection','partition','reader',
                    'execution_reader','rule','world_seed','endpoints'}
            or value['schema']!='orbit-fit-rule-block-v1' or value['status']!='COMPLETE'
            or value['role']!='FIT' or value['seed']!=seed or seed not in SEEDS
            or value['selection']!=selection['identity'] or value['partition']!=PARTITION
            or value['reader']!=reader_identity or value['execution_reader']!=execution_reader
            or value['rule']!=rule_wire(rule) or value['world_seed']!=0
            or type(value['endpoints']) is not list or len(value['endpoints'])!=56
            or type(forward_input_cap) is not int or forward_input_cap<=0):
        raise ValueError('complete actual reader/rule-bound FIT block required')
    records=[]
    for endpoint,permit in zip(value['endpoints'],permits):
        records.append(endpoint_record(endpoint,permit,selection=selection,seed=seed,
            reader_identity=reader_identity,execution_reader=execution_reader,rule=rule,forward_input_cap=forward_input_cap))
    return records


def endpoint_record(endpoint,permit,*,selection,seed,reader_identity,execution_reader,rule,forward_input_cap):
    if (set(endpoint)!={'schema','status','role','seed','world_seed','selection','partition',
            'reader','execution_reader','rule','permit','outcome','forward_input_tokens',
            'forward_input_cap','cost','teacher_actions_requested'}
            or endpoint['schema']!='orbit-fit-rule-endpoint-v1'
            or endpoint['status']!='COMPLETE' or endpoint['role']!='FIT'
            or endpoint['seed']!=seed or endpoint['world_seed']!=0
            or endpoint['selection']!=selection['identity'] or endpoint['partition']!=PARTITION
            or endpoint['reader']!=reader_identity or endpoint['execution_reader']!=execution_reader
            or endpoint['rule']!=rule_wire(rule) or endpoint['permit']!=asdict(permit)
            or endpoint['teacher_actions_requested']!=0
            or type(endpoint['forward_input_tokens']) is not int
            or not 0<endpoint['forward_input_tokens']<=forward_input_cap
            or endpoint['forward_input_cap']!=forward_input_cap
            or endpoint['cost']!=endpoint['forward_input_tokens']/forward_input_cap):
        raise ValueError('changed, partial, unmetered or foreign endpoint')
    row=endpoint['outcome']
    if (row.get('task')!=permit.task_id or row.get('permit')!=permit.identity
            or row.get('source_floor')!=permit.floor or row.get('control')!='candidate'
            or type(row.get('won')) is not bool or row.get('loss')!=1.-float(row['won'])
            or type(row.get('steps')) is not int or not 1<=row['steps']<=64
            or row.get('numerical_commits')!=0 or row.get('frozen_feedback')!=0
            or row.get('common_initial_writes_excluded') is not True):
        raise ValueError('actual complete fixed-Phi task outcome required')
    for key in ('feedback_commits','source_writes','source_nulls','post_write_reads'):
        if type(row.get(key)) is not int or not 0<=row[key]<=64:
            raise ValueError('actual M lifecycle counters required')
    # FastState.identity includes tensor hash, rho version, step, epoch
    # and the rho incarnation/hash. It is not itself a SHA256 string.
    if (type(row.get('phi')) is not str or re.fullmatch(
            r'[0-9a-f]{64}:942:0:0:[0-9a-f]{32}:[0-9a-f]{64}',row['phi']) is None):
        raise ValueError('actual unupdated fast state at the corrected rho942 required')
    if (row['feedback_commits']!=row['steps']-1
            or row['source_writes']+row['source_nulls']!=row['feedback_commits']):
        raise ValueError('feedback events do not match complete task')
    behavior=row['behavior']
    if behavior['actions']!=row['steps'] or behavior['decisions']!=row['feedback_commits']:
        raise ValueError('complete behavior signature accounting required')
    _sha(behavior['action_sequence_sha256']);_sha(behavior['write_sequence_sha256'])
    return dict(seed=seed,reader=reader_identity,rule=rule.identity,
        task_id=permit.task_id,game_sha256=permit.game_sha256,role='FIT',status='COMPLETE',
        won=row['won'],loss=row['loss'],cost=endpoint['cost'],endpoint_sha256=digest(endpoint))


def evaluate_fit_public_rule(permits,*,selection,codec,open_episode,on_work,on_event):
    """One same-information strong-rule pass, separate from model training labels."""
    from mmla_pdsa.full.outcome_baselines import ObservationFIFO,public_rule
    check_fit_permits(selection,permits)
    rows=[]
    for permit in permits:
        on_work(dict(kind='fit_rule_task_reserved',permit=permit.identity))
        episode=open_episode(permit,0,on_work);primary=None
        try:
            if (type(episode) is not ReplayEpisode or episode.permit!=permit or episode.seed!=0
                    or episode.checkpoint().steps or episode.terminal
                    or episode.limits.max_steps!=64 or episode.limits.expose_admissible_commands):
                raise ValueError('same unstarted FIT task and action information required')
            memory=ObservationFIFO(codec)
            memory.append(episode.visible.observation,step=0,source_receipt=_digest(asdict(episode.checkpoint())))
            while not episode.terminal:
                action=public_rule(memory.view(episode.visible));pair=episode.step(action)
                on_event(dict(kind='FIT_PUBLIC_RULE_ACTION',permit=permit.identity,
                    action=action,receipt=asdict(pair.receipt)))
                if not pair.terminal:
                    memory.append(pair.after.observation,step=pair.receipt.index+1,
                        source_receipt=_digest(asdict(pair.receipt)))
            row=dict(task=permit.task_id,permit=permit.identity,game_sha256=permit.game_sha256,
                source_floor=permit.floor,**episode.outcome,loss=1.-float(episode.outcome['won']))
        except BaseException as error:
            primary=error
            try:on_event(dict(kind='FIT_PUBLIC_RULE_INCOMPLETE',permit=permit.identity,error=repr(error)[:512]))
            except BaseException as reporting:error.add_note('partial rule evidence: '+repr(reporting))
            raise
        finally:
            try:episode.close()
            except BaseException as error:
                if primary is not None:primary.add_note('public-rule cleanup: '+repr(error))
                else:raise
        rows.append(row);on_event(dict(kind='FIT_PUBLIC_RULE_TERMINAL',result=row))
    return dict(schema='orbit-fit-public-rule-v1',status='COMPLETE',role='FIT',
        selection=selection['identity'],partition=PARTITION,codec=codec.identity,
        rule='unchanged observable public_rule + same eight-capsule FIFO',
        world_seed=0,worlds=rows,mean_loss=sum(r['loss'] for r in rows)/56,
        teacher_actions_requested=0,model_forwards=0)

