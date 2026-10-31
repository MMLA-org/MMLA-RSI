"""A source-bound shared readout update over three unchanged Orbit rho roles."""
from pathlib import Path
import torch
from mmla_pdsa.full.safe_state import save_state, load_state
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.extensions.rsi_binding import check_role_pair
from orbit_process_bundle import identity

ROLES=('task','proposal','original_proposal')
STATUS='TRAINED_COMPLETE_TASK_INPUT_READOUT'


def export_calibration(orbit, stem, *, original_roles, contract, course, steps):
    if steps != 247 or any(p.requires_grad or p.grad is not None for p in orbit.task.parameters()):
        raise ValueError('complete closed reader course required')
    for name in ROLES[1:]:check_role_pair(orbit.task,getattr(orbit,name))
    for name in ROLES:getattr(orbit,name).module_status['readout']=STATUS
    payload=dict(schema='orbit-shared-readout-calibration-v1',contract=contract,course=course,
        steps=steps,training_seed=17,original_roles=original_roles,
        roles={name:identity(getattr(orbit,name)) for name in ROLES},
        readout={key:value.detach().cpu().clone() for key,value in orbit.task.readout.state_dict().items()},
        RSI_qualification_inherited=False,role_rho_updated=False)
    save_state(Path(stem),payload)
    return payload


def apply_calibration(orbit, stem, *, contract):
    """Apply verified data into a freshly loaded original; no fallback/retraining.

    The containing artifact manifest must be checked by the owning loader. This
    boundary verifies the decoded role/parameter semantics before any mutation.
    Discard this fresh instance if restoration fails; no live state is rewound.
    """
    payload=load_state(Path(stem))
    current={name:identity(getattr(orbit,name)) for name in ROLES}
    if (payload['schema']!='orbit-shared-readout-calibration-v1' or payload['contract']!=contract
            or payload['steps']!=247 or payload['training_seed']!=17
            or payload['original_roles']!=current or payload['RSI_qualification_inherited'] is not False
            or payload['role_rho_updated'] is not False):
        raise ValueError('original three roles and completed bound calibration required')
    reference=orbit.task.readout.state_dict();values=payload['readout']
    if set(values)!=set(reference):raise ValueError('complete input readout required')
    for name,value in values.items():
        ref=reference[name]
        if not torch.is_tensor(value) or value.shape!=ref.shape or value.dtype!=ref.dtype or not torch.isfinite(value).all():
            raise ValueError('exact finite readout tensor required')
    for name in ROLES[1:]:check_role_pair(orbit.task,getattr(orbit,name))
    orbit.task.readout.load_state_dict(values,strict=True)
    for name in ROLES:getattr(orbit,name).module_status['readout']=STATUS
    if {name:identity(getattr(orbit,name)) for name in ROLES}!=payload['roles']:
        raise ValueError('restored calibrated roles differ from saved actual identities')
    orbit.metadata['reader_calibration']=dict(contract=contract,course=payload['course'],steps=247,
        readout=tensor_digest(values),RSI_qualification_inherited=False)
    return payload
