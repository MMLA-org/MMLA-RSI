"""Fixed-n paired admission for the new conditional logical-world population.

This is a prospective gate, not a relabeling of historical audit outcomes.
Correct independent sampling, candidate locking and native isolation remain
separate requirements; a computed e-value alone does not attest those facts.
"""
from dataclasses import asdict
import math
from mmla_pdsa.full.alfworld_episode import _sha
from content_rules import identity, Rule
from support_readiness import ReadyBlock

AUDIT_N = 1024
DELTA = .02
FAMILY_SIZE = 18
FAMILY_ALPHA = .025
FAMILY_ID = 'orbit-process-credit-20261029-r1'
ALPHA = FAMILY_ALPHA/FAMILY_SIZE
CS = (1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1.)


def checked_evaluation(value, block, *, reader, rule, pair_seed):
    if (type(block) is not ReadyBlock or type(rule) is not Rule or rule.mode != 'rtma'
            or type(value) is not dict or value.get('schema') != 'orbit-fixed-phi-logical-block-v1'
            or value.get('status') != 'COMPLETE' or value.get('role') != block.role
            or value.get('block_sha256') != block.block_sha256
            or value.get('population_sha256') != block.population_sha256
            or value.get('reader') != reader or identity(value.get('rule')) != identity(asdict(rule))
            or value.get('control') != 'candidate' or value.get('pair_seed') != pair_seed
            or value.get('admission_performed') is not False
            or value.get('successor_installed') is not False):
        raise ValueError('exact complete fixed-Phi reader/rule/block/pair required')
    _sha(reader)
    rows = value['worlds']
    if len(rows) != len(block.permits): raise ValueError('incomplete evaluation')
    losses = []
    for row, permit in zip(rows, block.permits):
        if (row['permit'] != permit.identity or row['task'] != permit.task_id
                or row['source_floor'] != permit.floor or row['control'] != 'candidate'
                or type(row['won']) is not bool or row['loss'] != 1.-float(row['won'])
                or type(row['loss']) not in (int,float) or row['numerical_commits'] != 0
                or row['common_initial_writes_excluded'] is not True
                or type(row['steps']) is not int or not 1 <= row['steps'] <= 64):
            raise ValueError('complete paired world outcome and fixed Phi required')
        # FastState.identity is a digest plus rho/step/epoch/recovery tags.
        phi=row['phi']
        if type(phi) is not str or len(phi)>1024 or len(phi.split(':'))<5:
            raise ValueError('versioned fast-state identity required')
        tags=phi.split(':');_sha(tags[0])
        if tags[2]!='0':raise ValueError('numerical fast state was updated')
        losses.append(float(row['loss']))
    tokens = value['forward_input_tokens']; cap = value['forward_input_cap']
    if (type(tokens) is not int or type(cap) is not int or not 0 <= tokens <= cap or cap <= 0
            or value['mean_cost'] != tokens/cap or not math.isfinite(value['mean_cost'])
            or value['mean_loss'] != sum(losses)/len(losses)):
        raise ValueError('unrounded complete means and bounded actual work required')
    return tuple(losses)


def paired_evalue(differences, *, family_id=FAMILY_ID):
    if type(family_id) is not str or not family_id:
        raise ValueError('declared finite test family required')
    ds = tuple(differences)
    if (len(ds) != AUDIT_N or any(type(d) not in (int,float) or not math.isfinite(d)
                                or not -1 <= d <= 1 for d in ds)):
        raise ValueError('exact1024 finite paired differences required; no early decision')
    m = (1.+DELTA)/2.; logs = []
    for c in CS:
        terms = [1.+(c/m)*((d+1.)/2.-m) for d in ds]
        if any(t < 0 for t in terms): raise ArithmeticError('negative betting factor')
        logs.append(-math.inf if any(t == 0 for t in terms) else sum(math.log(t) for t in terms))
    top = max(logs)
    log_e = -math.inf if top == -math.inf else top+math.log(sum(math.exp(x-top) for x in logs))-math.log(len(CS))
    # JSON cannot serialize -inf; null means an exact zero e-value, not missing data.
    return dict(schema='orbit-fixed-n-paired-evalue-v1',n=AUDIT_N,delta=DELTA,
        family_size=FAMILY_SIZE,family_id=family_id,family_alpha=FAMILY_ALPHA,
        alpha=ALPHA,threshold=1./ALPHA,
        coverage_scope='this new finite family conditional on prior history; no recycled alpha',
        lambda_mixture=[c/m for c in CS],
        component_log_evalues=[None if x == -math.inf else x for x in logs],
        log_evalue=None if log_e == -math.inf else log_e,
        evalue_exactly_zero=log_e == -math.inf,
        mean_gain=sum(ds)/AUDIT_N,statistical_pass=log_e >= math.log(1./ALPHA),
        fixed_n=True,optional_stopping=False,seed_is_independent_task=False)


def paired_admission(parent, child, block, *, model_identity, parent_reader, child_reader,
                     parent_rule, child_rule, pair_seed,
                     candidate_lock_sha256, native_resources_complete, family_id):
    _sha(candidate_lock_sha256)
    _sha(model_identity)
    if (block.role not in ('A_RSI1','A_RSI2') or len(block.permits) != AUDIT_N
            or parent_rule.identity == child_rule.identity or type(native_resources_complete) is not bool):
        raise ValueError('new rule and complete fixed audit block required')
    # The production task-reader hash contains the candidate rule/interface.
    # Parent/child hashes therefore differ even with identical model weights.
    # The actual factory adapter must bind both to model_identity separately.
    a = checked_evaluation(parent,block,reader=parent_reader,rule=parent_rule,pair_seed=pair_seed)
    b = checked_evaluation(child,block,reader=child_reader,rule=child_rule,pair_seed=pair_seed)
    if parent['forward_input_cap'] != child['forward_input_cap']:
        raise ValueError('same complete per-world work allowance required')
    test = paired_evalue((x-y for x,y in zip(a,b)), family_id=family_id)
    return dict(schema='orbit-paired-edge-admission-v1',lock=candidate_lock_sha256,
        model_identity=model_identity,parent_reader=parent_reader,child_reader=child_reader,
        parent=parent_rule.identity,child=child_rule.identity,
        block=block.block_sha256,population=block.population_sha256,
        parent_result=identity(parent),child_result=identity(child),
        test=test,native_resources_complete=native_resources_complete,
        accepted=test['statistical_pass'] and native_resources_complete,
        parent_functional_participation_checked=False,successor_installed=False,
        scope='fixed conditional population; not official benchmark, C or universal dominance')
