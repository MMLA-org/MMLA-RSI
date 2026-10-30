"""Task V selects the actual authoritative observation-memory successor.

This is a versioned batch-write domain, distinct from single-row proposal
EvidenceMemory. There is no fixed source write before V and no second mutable
evaluation-memory input. Features use currently retained rows and this arrival.
"""
from dataclasses import dataclass, asdict
import math
import json
import re
from mmla_pdsa.full.memory import Memory
from mmla_pdsa.full.outcome_source import SourceOption, choose_source
from content_rules import evaluate, identity
from content_features import (Observation, observation_units, marginals,
                              goal_units, append_features, SCHEMA)
from mmla_pdsa.full.observation_memory import ObservationBatch, ObservationCodec

DOMAIN='orbit-v-managed-observation-content-batch-v1'


@dataclass(frozen=True)
class RetainedValue:
    row: str
    observed_step: int
    loss: float
    cost: float


def validate_retained(memory, values, *, clock):
    if (type(memory) is not Memory or type(values) is not tuple
            or len(values)!=memory.profile.memory_slots or type(clock) is not int or clock<0):
        raise ValueError('exact authoritative table and retained-only metadata required')
    for row,value in zip(memory._rows,values):
        if row is None:
            if value is not None:raise ValueError('metadata cannot retain an absent observation')
            continue
        memory._validate(row)
        if row.status!='live' or type(value) is not RetainedValue or value.row!=row.receipt:
            raise ValueError('metadata must bind the currently readable actual row')
        observed=json.loads(row.canonical)['observed_step']
        if (type(observed) is not int or not 0<=observed<=clock or observed!=value.observed_step
                or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in (value.loss,value.cost))):
            raise ValueError('actual arrived clock and bounded metadata required')


def initialize_values(memory):
    values=tuple(None if row is None else RetainedValue(row.receipt,0,0.,0.) for row in memory._rows)
    validate_retained(memory,values,clock=0)
    return values


def state_identity(memory, values):
    return identity(dict(domain=DOMAIN,memory=memory.identity,values=[None if v is None else asdict(v) for v in values]))


def manage_source(memory, values, options, *, language, rule, loss, cost, step, horizon,
                  batch, codec, goal):
    """Score issued NULL/WRITE_BATCH_i choices; commit only the selected one.

    Eight task features are NULL, fraction of touched slots already occupied,
    new bounded observation loss/cost, mean evicted-row loss/cost/age, and current
    table occupancy. A batch replacing multiple old rows uses their mean. These
    semantics are explicit; this is not the old single-row evidence protocol.
    """
    if (type(horizon) is not int or not 1<=horizon<=512 or type(step) is not int or not 1<=step<=horizon
            or rule.mode!='rtma' or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in (loss,cost))):
        raise ValueError('bounded actual arrived feedback and RTMA rule required')
    validate_retained(memory,values,clock=step-1)
    if (type(batch) is not ObservationBatch or type(codec) is not ObservationCodec
            or codec.profile.identity != memory.profile.identity
            or codec.identity != memory.encoder.codec.identity or batch.step != step
            or horizon != 64 or memory.profile.memory_slots != 8):
        raise ValueError('complete actual arrived batch, exact eight-slot codec and clock required')
    codec.verify_batch(batch)  # Common failure must precede NULL and scoring.
    if type(options) not in (tuple,list) or not options or options[0].action!='NULL':
        raise ValueError('validated payload and NULL-first source options required')
    slots=memory.profile.memory_slots;predecessor=memory.identity
    occupied=sum(v is not None for v in values)/slots
    decisions=[];seen=set();payload=options[0].payload
    if payload != batch.identity:
        raise ValueError('options must bind this exact complete arrival')
    def row_units(table, clock):
        result=[]
        for row in table._rows:
            if row is None:result.append(None);continue
            table._validate(row)
            data,_=codec.validate(row.canonical)
            result.append(observation_units(Observation(**data),clock=clock))
        return tuple(result)
    before_units=row_units(memory,step-1)
    incoming=frozenset().union(*(observation_units(Observation(**codec.validate(c)[0]),clock=step)
                                for c in batch.chunks))
    unit_successors=[]
    for index,choice in enumerate(options):
        if (type(choice) is not SourceOption or choice.predecessor!=predecessor or choice.payload!=payload
                or choice.memory.identity!=choice.successor or choice.action in seen
                or choice.memory.profile.identity!=memory.profile.identity
                or choice.memory.encoder.identity!=memory.encoder.identity):
            raise ValueError('stale, duplicated or changed observation candidate')
        seen.add(choice.action)
        if index==0:
            if choice.successor!=predecessor or choice.receipts:
                raise ValueError('NULL must retain the actual table')
            features=(1.,0.,float(loss),float(cost),0.,0.,0.,occupied)
        else:
            match=re.fullmatch(r'WRITE_BATCH_(\d+)',choice.action)
            if match is None or not 0<=int(match[1])<slots or not choice.receipts:
                raise ValueError('issued complete observation write required')
            changed=[i for i,(old,new) in enumerate(zip(memory._rows,choice.memory._rows))
                     if (None if old is None else old.receipt)!=(None if new is None else new.receipt)]
            if not changed:raise ValueError('non-NULL must change actual retained content')
            old_values=[values[i] for i in changed if values[i] is not None]
            for i in changed:
                row=choice.memory._rows[i]
                if row is None or row.status!='live' or json.loads(row.canonical)['observed_step']!=step:
                    raise ValueError('source write must contain this actual observation arrival')
            mean=lambda key:sum(getattr(v,key) for v in old_values)/len(old_values) if old_values else 0.
            age=sum((step-v.observed_step)/horizon for v in old_values)/len(old_values) if old_values else 0.
            features=(0.,len(old_values)/len(changed),float(loss),float(cost),mean('loss'),mean('cost'),age,occupied)
        decisions.append(dict(index=index,action=choice.action,features=features))
        unit_successors.append(row_units(choice.memory,step))
    additions,normalizer=marginals(before_units,tuple(unit_successors),incoming,goal_words=goal_units(goal))
    for decision,extra in zip(decisions,additions):
        features=append_features(decision['features'],extra)
        decision.update(features=features,score=evaluate(language,rule,memory_features=features)[0])
    # Production source_options is NULL then ascending start slot; require that tie order.
    starts=[int(x.action.split('_')[-1]) for x in options[1:]]
    if starts!=sorted(starts):raise ValueError('fixed NULL-first ascending-slot tie order required')
    selected=max(range(len(options)),key=lambda i:decisions[i]['score'])
    successor,receipt=choose_source(memory,options,selected)
    next_values=[]
    for old,new,previous in zip(memory._rows,successor._rows,values):
        if new is None:next_values.append(None)
        elif old is not None and old.receipt==new.receipt:next_values.append(previous)
        else:next_values.append(RetainedValue(new.receipt,step,float(loss),float(cost)))
    next_values=tuple(next_values);validate_retained(successor,next_values,clock=step)
    return successor,next_values,dict(domain=DOMAIN,rule=rule.identity,source=receipt,selected=selected,
        predecessor=state_identity(memory,values),successor=state_identity(successor,next_values),
        choices=decisions,source_commit_only_after_V=True,unselected_metadata_retained=False,
        content_schema=SCHEMA,content_normalizer=normalizer)
