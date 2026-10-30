"""Complete-task evaluation and exact common-anchor future labels.

Evaluation never requests or consumes teacher labels, including on FIT tasks.
The native FIT backend eagerly prepares expert labels during reset/step; that
work is included in physical reset/step CPU cost, not claimed absent. The runner
owns task IDs, end points and branches. The selector receives features only.
"""
from dataclasses import asdict,dataclass
import torch
from .alfworld_episode import ReplayEpisode,_digest
from .outcome_generation import generate_action_batch,GENERATION_ID
from .outcome_losses import outcome_losses
from .outcome_session import OutcomeSession
from .outcome_train_batch import _text_batch,_memories,_decode,_close_all

MODES=('fixed_joint','phi_only','m_only','no_online','frozen','joint_j')

@dataclass(frozen=True)
class EvaluationPolicy:
    mode:str
    anchor_step:int=7
    def __post_init__(self):
        if self.mode not in MODES or type(self.anchor_step) is not int or not 0<=self.anchor_step<512:
            raise ValueError('registered finite evaluation policy required')
    @property
    def identity(self):return _digest(dict(schema='orbit-complete-evaluation-policy-v1',**asdict(self)))


def fixed_choice(session,policy):
    p=session._pending_check('PREPARED')
    memory=policy.mode in ('fixed_joint','m_only','joint_j')
    numerical=policy.mode in ('fixed_joint','phi_only','joint_j')
    name='WRITE_BATCH_'+str(session.step%session.model.profile.memory_slots)
    index=next((i for i,x in enumerate(p['options']) if x.action==name),0) if memory else 0
    return numerical and p['shadow'] is not None,index,memory


def trace_partial_failure(sessions,error,on_progress,*,scope):
    """Persist observed partial commits; never label a failed transaction closed.

    U/source can commit before V fails. The current last receipt then lacks a
    completed V field; preserve it as-is with the actual current state identity.
    """
    if on_progress is None:return
    for i,session in enumerate(sessions):
        if type(session) is not OutcomeSession:continue
        try:
            on_progress(dict(kind='INCOMPLETE_STATE',scope=scope,batch_index=i,tenant=session.tenant,
                error=repr(error),last_receipt=session.receipts[-1] if session.receipts else None,
                phi=session.fast.identity,evaluation_memory=session.evaluation.identity,
                source_version=session.source.version,transaction_complete=False))
        except BaseException as reporting:error.add_note('partial state reporting: '+repr(reporting))


def run_evaluation_batch(episodes,sessions,*,policy,on_work,pause_at_anchor=False,selector=None,head_chunk=32,on_progress=None):
    """Run all actual futures, optionally return at the sole prepared anchor.

    Caller owns and closes episodes, including successful/paused ones. Failed
    work poisons all sessions and attempts every close. A pause is not COMPLETE.
    Fresh inputs initialize M once; post-anchor branches must already be idle.
    """
    def progress(event):
        if on_progress is not None:on_progress(event)
    try:
        if on_progress is not None and not callable(on_progress):raise ValueError('bounded durable progress sink')
        if (type(policy) is not EvaluationPolicy or type(pause_at_anchor) is not bool
                or not callable(on_work) or not episodes or len(episodes)!=len(sessions)
                or len({id(x) for x in episodes})!=len(episodes) or len({id(x) for x in sessions})!=len(sessions)
                or (policy.mode=='joint_j' and not callable(selector))
                or (selector is not None and policy.mode!='joint_j')):
            raise ValueError('finite bound batch with one declared selector policy')
        model=sessions[0].model;device=model.rho.a.device
        results=[None]*len(episodes);live=[]
        for i,(episode,session) in enumerate(zip(episodes,sessions)):
            if (type(episode) is not ReplayEpisode or type(session) is not OutcomeSession or session.model is not model
                    or session.pending is not None or episode.limits.max_steps<=policy.anchor_step
                    or episode.limits.expose_admissible_commands):
                raise ValueError('isolated idle actual task with common public information required')
            if not session.receipts:
                session.initialize_observation(episode)
                progress(dict(kind='INITIAL_SOURCE',batch_index=i,task=episode.permit.task_id,receipt=session.receipts[-1]))
            if session.closed or episode.terminal:
                if not(session.closed and episode.terminal):raise ValueError('episode/session terminal state mismatch')
                results[i]=dict(status='COMPLETE',**episode.outcome)
            else:live.append(i)
        while live:
            contexts=[sessions[i].action_input(episodes[i].visible) for i in live]
            tokens=generate_action_batch(model,[x.prompt for x in contexts],fast_states=[x.fast for x in contexts],
                memories=[x.memory for x in contexts],eos_ids={sessions[live[0]].formatter._eos},
                read_binding=_digest([x.binding for x in contexts]),
                on_forward=lambda rows,valid,padded:on_work(dict(kind='generation_forward',rows=rows,valid_tokens=valid,padded_positions=padded)),
                on_tokens=lambda rows,ts:on_work(dict(kind='generated_tokens',count=len(ts))))
            ongoing=[];compute=[]
            for j,(i,ctx,ts) in enumerate(zip(live,contexts,tokens)):
                session=sessions[i];episode=episodes[i]
                action=_decode(session.formatter.tokenizer,ts,{session.formatter._eos})
                pair=session.execute(episode,action,generated_tokens=len(ts))
                progress(dict(kind='ACTUAL_ACTION',batch_index=i,task=episode.permit.task_id,policy=policy.identity,
                    read_binding=ctx.binding,receipt=asdict(pair.receipt),generated_tokens=len(ts),
                    terminal=pair.terminal,outcome=dict(episode.outcome) if pair.terminal else None))
                if pair.terminal:
                    session.finish_terminal();results[i]=dict(status='COMPLETE',**episode.outcome)
                    progress(dict(kind='TERMINAL',batch_index=i,task=episode.permit.task_id,receipt=session.receipts[-1],outcome=results[i]));continue
                if policy.mode in ('no_online','frozen') or (policy.mode=='phi_only' and session.fast.steps>=session.max_updates):
                    session.finish_frozen_feedback()
                    progress(dict(kind='FROZEN_FEEDBACK',batch_index=i,task=episode.permit.task_id,receipt=session.receipts[-1]))
                    ongoing.append(i);continue
                allow=policy.mode!='m_only' and session.fast.steps<session.max_updates
                compute.append((i,ctx,allow))
            # Preserve batch shape while avoiding graph building once all U
            # budgets are exhausted. Frozen M-only still measures real CE for V.
            if compute:
                need_grad=any(x[2] for x in compute)
                with torch.set_grad_enabled(need_grad):
                    losses,_=outcome_losses(model,*_text_batch([sessions[i].feedback_text() for i,_,_ in compute],device),
                        fast_states=[ctx.fast for _,ctx,_ in compute],memory=_memories(model,[ctx for _,ctx,_ in compute]),
                        head_chunk=head_chunk,on_work=on_work)
                    active=[j for j,x in enumerate(compute) if x[2]]
                    params=tuple(v for j in active for v in (compute[j][1].fast.a,compute[j][1].fast.b))
                    grads=torch.autograd.grad(losses[active].sum(),params) if params else ()
                mapping={row:grads[2*k:2*k+2] for k,row in enumerate(active)}
                for j,(i,ctx,allow) in enumerate(compute):
                    session=sessions[i];episode=episodes[i]
                    session.prepare(losses[j],opportunity_cap=episode.limits.max_steps,allow_update=allow,_gradient=mapping.get(j))
                    if pause_at_anchor and session.step==policy.anchor_step:
                        results[i]=dict(status='ANCHOR',anchor=session.branch_identity(),checkpoint=asdict(episode.checkpoint()))
                        progress(dict(kind='ANCHOR',batch_index=i,task=episode.permit.task_id,record=results[i]))
                        continue
                    update,index,manage=fixed_choice(session,policy)
                    if policy.mode=='joint_j' and session.step==policy.anchor_step:
                        from .outcome_j import outcome_features
                        features=outcome_features(session,protocol=selector.protocol,on_work=on_work)
                        choice,receipt=selector(features)
                        if type(choice) is not int or not 0<=choice<len(features.pairs):raise ValueError('issued J pair required')
                        update,name=features.pairs[choice]
                        index=next(k for k,x in enumerate(session.pending['options']) if x.action==name)
                        update=bool(update);session.commit(update=update,source_index=index,manage_evaluation=True)
                        session.receipts[-1]['selector']=receipt
                    else:session.commit(update=update,source_index=index,manage_evaluation=manage)
                    progress(dict(kind='STATE_COMMIT',batch_index=i,task=episode.permit.task_id,receipt=session.receipts[-1]))
                    ongoing.append(i)
                del losses,grads
            live=ongoing
        if any(x is None for x in results):raise RuntimeError('incomplete task result coverage')
        return tuple(results)
    except BaseException as original:
        trace_partial_failure(sessions,original,on_progress,scope='evaluation_batch')
        for session in sessions:
            if type(session) is OutcomeSession:session.failed=True
        errors=_close_all(episodes)
        if errors:original.add_note('Additional cleanup failures: '+'; '.join(repr(x) for x in errors))
        raise


def collect_future_labels(session,episode,*,protocol,factory,on_work,on_environment_work,branch_batch_size=8,on_progress=None):
    """All issued pairs; identical prefix, private state, own COMPLETE future.

    No arbitrary continuation callback can silently change the label policy.
    Fitting/calibration permit checks are repeated by the record validator.
    Every replay reset/step and model call is metered, even on failure.
    """
    from .outcome_j import outcome_features
    if episode.permit.role not in ('FIT','C_J0'):raise PermissionError('J labels only on fixed FIT/C_J0 tasks')
    if session.step!=protocol['anchor_step']:raise ValueError('declared real feedback anchor required')
    if type(branch_batch_size) is not int or branch_batch_size not in (1,2,4,8):
        raise ValueError('predeclared independent branch microbatch required')
    if on_progress is not None and not callable(on_progress):raise ValueError('bounded durable J progress sink')
    features=outcome_features(session,protocol=protocol,on_work=on_work)
    seal=session.branch_identity();actions=[]
    for start in range(0,len(features.pairs),branch_batch_size):
        pairs=features.pairs[start:start+branch_batch_size]
        branches=[];envs=[];commits=[];original=None
        try:
            for numerical,name in pairs:
                branch,env=session.fork_feedback(episode,factory=factory,on_work=on_environment_work)
                # Own every successfully created environment before any commit
                # can fail. Independent fast state, both M tables and RNG are
                # retained; only frozen model weights are shared by the batch.
                branches.append(branch);envs.append(env)
                index=next(i for i,x in enumerate(branch.pending['options']) if x.action==name)
                commits.append(branch.commit(update=bool(numerical),source_index=index))
                if on_progress is not None:on_progress(dict(kind='J_BRANCH_COMMIT',task=episode.permit.task_id,
                    anchor=seal,pair=(numerical,name),commit=commits[-1]))
            results=run_evaluation_batch(envs,branches,
                policy=EvaluationPolicy('fixed_joint',protocol['anchor_step']),on_work=on_work,
                on_progress=None if on_progress is None else lambda event:on_progress(dict(kind='J_FUTURE_PROGRESS',
                    anchor=seal,pairs=pairs,event=event)))
            for pair,branch,env,commit,result in zip(pairs,branches,envs,commits,results):
                if result['status']!='COMPLETE':raise RuntimeError('branch label is not a complete future')
                actions.append(dict(pair=pair,outcome=result,risk=1-int(result['won']),
                    commit=commit,final_checkpoint=_digest(asdict(env.checkpoint())),state_receipts=branch.receipts))
            if on_progress is not None:
                for item in actions[-len(pairs):]:on_progress(dict(kind='J_COMPLETE_FUTURE',task=episode.permit.task_id,
                    anchor=seal,pair=item['pair'],outcome=item['outcome'],risk=item['risk'],final_checkpoint=item['final_checkpoint']))
            if session.branch_identity()!=seal:raise RuntimeError('future evaluation changed live common predecessor')
        except BaseException as exc:
            original=exc
            trace_partial_failure(branches,exc,on_progress,scope=dict(kind='J_BRANCH_BATCH',anchor=seal,pairs=pairs))
            raise
        finally:
            errors=_close_all(envs)
            if errors:
                if original is not None:original.add_note('Additional branch cleanup failures: '+'; '.join(repr(x) for x in errors))
                else:raise RuntimeError('branch environment cleanup failed') from errors[0]
    return dict(schema='orbit-own-complete-futures-v1',coverage='COMPLETE',role=episode.permit.role,
        task=episode.permit.task_id,floor=episode.permit.floor,permit=asdict(episode.permit),
        anchor=seal,checkpoint=asdict(episode.checkpoint()),features=features,actions=tuple(actions),
        risks=tuple(x['risk'] for x in actions))
