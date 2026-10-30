"""Immutable outcome-J package, distinct from old engineering J artifacts."""
from dataclasses import asdict
import copy
import torch
from .artifacts import save_artifact,load_artifact
from .neural import tensor_digest
from .outcome_j import OutcomeHead,OutcomeCalibration,OutcomePredictor

KIND='orbit-complete-task-j-v1'
STATUSES=('EMPIRICALLY_CALIBRATED','UNQUALIFIED_FIT_NO_ANCHOR','UNQUALIFIED_CALIBRATION_NO_ANCHOR')


def _status(head,status,calibration):
    if type(head) is not OutcomeHead or status not in STATUSES:raise ValueError('explicit outcome J qualification status')
    expected=0 if status=='UNQUALIFIED_FIT_NO_ANCHOR' else 128
    if int(head.updates)!=expected or (status=='EMPIRICALLY_CALIBRATED')!=(calibration is not None):
        raise ValueError('qualification must match actual updates and complete calibration')
    if calibration is not None and (type(calibration) is not OutcomeCalibration or calibration.protocol!=head.protocol
            or calibration.head!=tensor_digest(head.state_dict())):raise ValueError('exact fitted head/calibration pair')


def save_j_package(path,head,*,seed,status,calibration,optimizer,fit_order,evidence_identity):
    from .alfworld_episode import _sha
    _sha(evidence_identity);_status(head,status,calibration)
    if seed not in (17,29,43):raise ValueError('paired training/head seed required')
    if (status=='UNQUALIFIED_FIT_NO_ANCHOR')!=(optimizer is None):raise ValueError('preserve actual fitted optimizer state')
    return save_artifact(path,dict(schema=KIND,protocol=copy.deepcopy(head.protocol),seed=seed,
        head=dict(head.state_dict()),head_identity=tensor_digest(head.state_dict()),status=status,
        calibration=None if calibration is None else asdict(calibration),optimizer=optimizer,
        fit_order=fit_order,evidence_identity=evidence_identity),kind=KIND)


def load_j_package(path,*,identity,reader,expected_protocol):
    payload=load_artifact(path,kind=KIND,identity=identity)
    if payload.get('schema')!=KIND or payload.get('protocol')!=expected_protocol or payload.get('seed') not in (17,29,43):
        raise ValueError('outcome artifact domain/reader contract mismatch')
    head=OutcomeHead(expected_protocol,seed=payload['seed'])
    reference=head.state_dict();state=payload['head']
    if set(reference)!=set(state) or any(not torch.is_tensor(state[k]) or state[k].shape!=v.shape
            or state[k].dtype!=v.dtype or not torch.isfinite(state[k]).all() for k,v in reference.items()):
        raise ValueError('complete finite exact-layout J tensors required')
    head.load_state_dict(state);head.to(reader.rho.a.device)
    if tensor_digest(head.state_dict())!=payload['head_identity']:raise ValueError('J tensor identity changed')
    raw=payload['calibration'];calibration=None if raw is None else OutcomeCalibration(**raw)
    _status(head,payload['status'],calibration)
    selector=OutcomePredictor(head,calibration,reader=reader)
    return head,selector,payload
