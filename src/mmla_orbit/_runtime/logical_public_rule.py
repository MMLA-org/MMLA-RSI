"""Unchanged observable public_rule on the same certified derived worlds."""
from dataclasses import asdict
from mmla_pdsa.full.alfworld_episode import _digest
from mmla_pdsa.full.outcome_baselines import ObservationFIFO,public_rule
from logical_episode import LogicalWorldEpisode
from support_readiness import ReadyBlock

RULE='unchanged observable public_rule + same eight-capsule FIFO; no private factual archive'


def evaluate_public_rule(block,*,codec,open_episode,on_work,on_event):
    if type(block) is not ReadyBlock or not all(callable(f) for f in (open_episode,on_work,on_event)):
        raise ValueError('complete sealed block and bound native interfaces required')
    rows=[]
    for permit in block.permits:
        on_work(dict(kind='rule_task_reserved',permit=permit.identity))
        episode=open_episode(permit,0,on_work)
        primary=None
        try:
            if (type(episode) is not LogicalWorldEpisode or episode.permit!=permit or episode.seed!=0
                    or episode.checkpoint().steps or episode.limits.max_steps!=64
                    or episode.limits.expose_admissible_commands):
                raise ValueError('same unstarted teacher-free seed0 world required')
            memory=ObservationFIFO(codec)
            memory.append(episode.visible.observation,step=0,source_receipt=_digest(asdict(episode.checkpoint())))
            while not episode.terminal:
                action=public_rule(memory.view(episode.visible));pair=episode.step(action)
                on_event(dict(kind='PUBLIC_RULE_ACTION',permit=permit.identity,action=action,
                    receipt=asdict(pair.receipt)))
                if not pair.terminal:
                    memory.append(pair.after.observation,step=pair.receipt.index+1,
                        source_receipt=_digest(asdict(pair.receipt)))
            row=dict(task=permit.task_id,permit=permit.identity,source_floor=permit.floor,
                **episode.outcome,loss=1.-float(episode.outcome['won']))
            rows.append(row);on_event(dict(kind='PUBLIC_RULE_TERMINAL',result=row))
        except BaseException as error:
            primary=error
            on_event(dict(kind='PUBLIC_RULE_INCOMPLETE',permit=permit.identity,error=repr(error)[:512]))
            raise
        finally:
            try:episode.close()
            except BaseException as error:
                if primary is not None:primary.add_note('public-rule cleanup: '+repr(error))
                else:raise
    result=dict(schema='orbit-logical-public-rule-v1',status='COMPLETE',rule=RULE,
        role=block.role,block_sha256=block.block_sha256,population_sha256=block.population_sha256,
        codec=codec.identity,world_seed=0,worlds=rows,
        mean_loss=sum(r['loss'] for r in rows)/len(rows),model_forwards=0,
        physical_costs='charged separately by the same complete native ledger')
    check_public_rule(result,block)
    return result


def check_public_rule(value,block):
    if (value.get('schema')!='orbit-logical-public-rule-v1' or value.get('status')!='COMPLETE'
            or value.get('rule')!=RULE or value.get('role')!=block.role
            or value.get('block_sha256')!=block.block_sha256
            or value.get('population_sha256')!=block.population_sha256
            or value.get('world_seed')!=0 or value.get('model_forwards')!=0
            or len(value.get('worlds',()))!=len(block.permits)):
        raise ValueError('complete same-information same-world strong-rule result required')
    for row,permit in zip(value['worlds'],block.permits):
        if (row['task']!=permit.task_id or row['permit']!=permit.identity
                or row['source_floor']!=permit.floor or type(row['won']) is not bool
                or row['loss']!=1.-float(row['won']) or type(row['steps']) is not int
                or not 1<=row['steps']<=64):
            raise ValueError('actual complete strong-rule task outcome required')
    if value['mean_loss']!=sum(r['loss'] for r in value['worlds'])/len(value['worlds']):
        raise ValueError('strong-rule endpoint mean differs')
