"""One complete on-policy outer batch, connected to actual task feedback.

This module does not load a model, authorize a run, or choose a new recipe. The
entry binds the original artifacts, partition, schedule and total resource cap.
There is one slow update only after every task and the general objective finish.
"""
from dataclasses import asdict
import torch
from .alfworld_episode import ReplayEpisode,_digest
from .alfworld_formatting import SupervisedText
from .batching import collate_sequences,collate_memories
from .outcome_generation import generate_action_batch,GENERATION_ID
from .outcome_losses import outcome_losses
from .outcome_session import OutcomeSession,prepare_feedback_batch
from .outcome_trajectory import OutcomeTrajectoryCore


def _text_batch(rows,device):
    tensors=[]
    for row in rows:
        if type(row) is not SupervisedText:raise TypeError('complete registered supervision required')
        ids=torch.tensor(row.input_ids,dtype=torch.long)[None]
        labels=torch.tensor(row.labels,dtype=torch.long)[None]
        tensors.append((ids,torch.ones_like(ids),labels))
    return collate_sequences(tensors,pad_id=0,device=device,padding_side='right')


def _memories(model,contexts):
    return collate_memories([x.memory for x in contexts],key_width=model.profile.key_width,
        value_width=model.profile.neural_width,device=model.rho.a.device)


def _decode(tokenizer,tokens,eos):
    # Remove only the terminal EOS. Whitespace, extra explanation, blank output,
    # invalid commands and incomplete max-length outputs are not repaired.
    if not tokens:raise ValueError('generation must account for at least one token')
    if any(x in eos for x in tokens[:-1]):raise ValueError('generation continued past EOS')
    body=tokens[:-1] if tokens[-1] in eos else tokens
    return tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)


def _initial_checks(core,episodes,sessions,on_work):
    if (type(core) is not OutcomeTrajectoryCore or type(episodes) not in (tuple,list)
            or type(sessions) not in (tuple,list) or not episodes or len(episodes)!=len(sessions)
            or not callable(on_work) or len({id(e) for e in episodes})!=len(episodes)
            or len({id(s) for s in sessions})!=len(sessions)):
        raise ValueError('finite independent tasks and mandatory work accounting required')
    groups={};ids=set()
    for episode,session in zip(episodes,sessions):
        if (type(episode) is not ReplayEpisode or type(session) is not OutcomeSession
                or session.model is not core.model or episode.permit.role!='FIT'
                or episode.limits.max_steps!=core.horizon or episode.terminal
                or episode.checkpoint().steps or session.step or session.receipts
                or episode.permit.task_id in ids):
            raise ValueError('unstarted FIT tasks and private state on the same slow model required')
        ids.add(episode.permit.task_id)
        groups.setdefault(episode.permit.floor,[]).append(episode.permit.task_id)
    return groups


def _close_all(episodes):
    errors=[]
    if type(episodes) in (tuple,list):
        for episode in episodes:
            if type(episode) is not ReplayEpisode:continue
            try:episode.close()
            except BaseException as exc:errors.append(exc)
    return errors


def train_outer_batch(core,episodes,sessions,general_rows,*,teacher,on_work,head_chunk=32,microbatch=8,on_progress=None):
    """Fixed coverage: update each eligible feedback, rotate whole-source writes.

    Both objectives roll out their actual adaptive policy. Their only loss-side
    difference is task action CE at current rho (static) versus actual pending
    Phi (adapted). Neither gets the other arm's better trajectory or teacher.
    The direct-rho observation objective and general CE + SFT KL are identical.
    The original frozen SFT teacher is used only on general conversations.

    Caller supplies already validated general rows as SupervisedText. The fixed
    coefficient of KL is one; the core owns the common auxiliary/support weights.
    No old offline course observation is substituted for returned feedback.
    A prequalified microbatch partitions every neural forward and gradient
    contribution; it never starts a second outer batch or retries failed work.
    """
    if on_progress is None:on_progress=lambda event:None
    try:
        if not callable(on_progress):raise ValueError('bounded progress sink required')
        groups=_initial_checks(core,episodes,sessions,on_work)
        if type(microbatch) is not int or microbatch not in (1,2,4,8):
            raise ValueError("prequalified microbatch must be 8, 4, 2 or 1")
        if not general_rows or teacher is None:raise ValueError('bound general CE and original SFT KL required')
        model=core.model;device=model.rho.a.device
        core.begin(groups)
        for episode,session in zip(episodes,sessions):
            session.initialize_observation(episode)
            on_progress(dict(kind='INITIAL_SOURCE',task=episode.permit.task_id,receipt=session.receipts[-1]))
        live=list(range(len(episodes)));action_records=[]
        while live:
            contexts=[sessions[i].action_input(episodes[i].visible) for i in live]
            texts=[];records=[];endpoint_contexts=[];endpoint_states=[]
            for i,ctx in zip(live,contexts):
                episode=episodes[i];session=sessions[i];step=session.step
                label=episode.training_label()
                on_progress(dict(kind='TEACHER_EVENT',task=episode.permit.task_id,step=step,label=label))
                if label is None:
                    core.no_teacher(episode.permit.floor,episode.permit.task_id,step,reason='NO_ADMISSIBLE_LABEL')
                    continue
                texts.append(session.formatter.action_supervision(ctx.frame,label,evaluation_memory=ctx.evaluation))
                fast=ctx.fast if core.config.objective=='adapted' else model.rho.begin(episode.seed)
                endpoint_states.append(fast);endpoint_contexts.append(ctx)
                records.append((episode.permit.floor,episode.permit.task_id,step,fast))
            for start in range(0,len(texts),microbatch):
                stop=start+microbatch
                ce,_=outcome_losses(model,*_text_batch(texts[start:stop],device),
                    fast_states=endpoint_states[start:stop],memory=_memories(model,endpoint_contexts[start:stop]),
                    head_chunk=head_chunk,on_work=on_work)
                core.add_task_batch(records[start:stop],ce)
                del ce
            token_rows=[]
            for start in range(0,len(contexts),microbatch):
                chunk=contexts[start:start+microbatch]
                token_rows.extend(generate_action_batch(model,[x.prompt for x in chunk],
                    fast_states=[x.fast for x in chunk],memories=[x.memory for x in chunk],
                    eos_ids={sessions[live[0]].formatter._eos},read_binding=_digest([x.binding for x in chunk]),
                    on_forward=lambda rows,valid,padded:on_work(dict(kind='generation_forward',rows=rows,valid_tokens=valid,padded_positions=padded)),
                    on_tokens=lambda rows,tokens:on_work(dict(kind='generated_tokens',count=len(tokens)))))
            feedback=[];support_texts=[];support_records=[]
            for i,ctx,tokens in zip(live,contexts,token_rows):
                session=sessions[i];episode=episodes[i]
                action=_decode(session.formatter.tokenizer,tokens,{session.formatter._eos})
                pair=session.execute(episode,action,generated_tokens=len(tokens));feedback.append(pair)
                on_progress(dict(kind='ACTUAL_ACTION',task=episode.permit.task_id,step=session.step,
                    context=ctx.binding,receipt=asdict(pair.receipt),generated_tokens=len(tokens)))
                support_texts.append(session.feedback_text())
                support_records.append((episode.permit.floor,episode.permit.task_id,session.step))
                action_records.append(dict(task=episode.permit.task_id,step=session.step,
                    context=ctx.binding,receipt=asdict(pair.receipt),generated_tokens=len(tokens)))
            for start in range(0,len(support_texts),microbatch):
                stop=start+microbatch
                support,_=outcome_losses(model,*_text_batch(support_texts[start:stop],device),
                    memory=_memories(model,contexts[start:stop]),head_chunk=head_chunk,on_work=on_work)
                core.add_support_batch(support_records[start:stop],support)
                del support
            ongoing=[j for j,pair in enumerate(feedback) if not pair.terminal]
            for start in range(0,len(ongoing),microbatch):
                selected=ongoing[start:start+microbatch]
                inner_contexts=[contexts[j] for j in selected]
                inner,_=outcome_losses(model,*_text_batch([support_texts[j] for j in selected],device),
                    fast_states=[x.fast for x in inner_contexts],memory=_memories(model,inner_contexts),
                    head_chunk=head_chunk,on_work=on_work)
                chosen_sessions=[sessions[live[j]] for j in selected]
                candidates=prepare_feedback_batch(chosen_sessions,inner,opportunity_cap=core.horizon)
                del inner
                for session,(options,shadow) in zip(chosen_sessions,candidates):
                    name='WRITE_BATCH_'+str(session.step%model.profile.memory_slots)
                    # A protected slot is a legitimate absence, not replacement
                    # with a more favorable action. NULL is always option zero.
                    index=next((k for k,x in enumerate(options) if x.action==name),0)
                    session.commit(update=shadow is not None,source_index=index)
                    on_progress(dict(kind='STATE_COMMIT',tenant=session.tenant,receipt=session.receipts[-1]))
            for j,pair in enumerate(feedback):
                if not pair.terminal:continue
                i=live[j];session=sessions[i];episode=episodes[i]
                session.finish_terminal();outcome=episode.outcome
                on_progress(dict(kind='TERMINAL',task=episode.permit.task_id,outcome=outcome,receipt=session.receipts[-1]))
                end='WON' if outcome['won'] else ('STEP_LIMIT' if outcome['truncated'] else 'LOST')
                core.close_episode(episode.permit.floor,episode.permit.task_id,actions=outcome['steps'],termination=end)
            live=[live[j] for j in ongoing]
        ce_mean=0.;kl_mean=0.
        for start in range(0,len(general_rows),microbatch):
            chunk=general_rows[start:start+microbatch]
            general_ce,general_kl=outcome_losses(model,*_text_batch(chunk,device),teacher=teacher,
                head_chunk=head_chunk,on_work=on_work)
            core.add_auxiliary_part((general_ce+general_kl).mean(),offset=start,rows=len(chunk),total_rows=len(general_rows))
            weight=len(chunk)/len(general_rows)
            ce_mean+=float(general_ce.detach().mean())*weight
            kl_mean+=float(general_kl.detach().mean())*weight
            del general_ce,general_kl
        before_commit=dict(general_ce=ce_mean,general_kl=kl_mean,microbatch=microbatch,
            tasks=[dict(task=e.permit.task_id,floor=e.permit.floor,**e.outcome) for e in episodes],
            actions=action_records,state_receipts=[s.receipts for s in sessions])
        close_errors=_close_all(episodes)
        if close_errors:raise BaseExceptionGroup('episode close/accounting failed before slow commit',close_errors)
        on_progress(dict(kind='BEFORE_SLOW_COMMIT',completed_steps=core.steps,actual_actions=len(action_records),
            general_ce=ce_mean,general_kl=kl_mean))
        result=core.commit();result.update(before_commit)
        on_progress(dict(kind='SLOW_COMMIT',completed_steps=core.steps,contract=core.contract_identity))
        return result
    except BaseException as original:
        # Includes validation/begin and close accounting. Every owned episode
        # is attempted even if an earlier cleanup fails. Preserve the original
        # failure and attach additional close failures rather than replacing it.
        if type(core) is OutcomeTrajectoryCore:core.failed=True
        if type(sessions) in (tuple,list):
            # Same independently reviewed failure journal boundary as evaluation:
            # U/source may have committed before V raised. Preserve that fact.
            from .outcome_evaluation import trace_partial_failure
            trace_partial_failure(sessions,original,on_progress,scope='training_batch')
            for session in sessions:
                if type(session) is OutcomeSession:session.failed=True
        close_errors=_close_all(episodes)
        if close_errors:original.add_note('Additional episode cleanup failures: '+
            '; '.join(type(e).__name__+': '+str(e) for e in close_errors))
        raise
