"""Inherit proposal rho while adopting a final task reader's frozen layout.

This constructs a role pair, not a runtime qualification or a successor. The
caller supplies independently verified source identities and retains original
artifacts. No model forward, random initializer, optimizer or J loader is used.
"""
from dataclasses import asdict
import copy
import torch
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.profile import digest
from mmla_pdsa.full.extensions.successor_model import SuccessorMMLAModel
from mmla_pdsa.full.extensions.rsi_binding import check_role_pair, rebind


def _digest(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('explicit SHA256 identity required')
    return value


def _initialization(rho):
    value=rho.initialization
    if set(value)!={'a','b','seed'} or type(value['seed']) is not int:
        raise ValueError('complete inherited initialization record')
    for name in ('a','b'):
        x=value[name];ref=getattr(rho,name)
        if (not torch.is_tensor(x) or x.shape!=ref.shape or x.dtype!=ref.dtype
                or not torch.isfinite(x).all()):raise ValueError('finite inherited initialization layout')
    return dict(seed=value['seed'],factors=tensor_digest({k:value[k] for k in ('a','b')}))


def inherit_proposal_role(task, original_proposal, *, task_modules, proposal_modules,
                          task_profile, proposal_profile, source_package):
    """Return a frozen proposal role and an explicit unqualified migration receipt.

    task is a closed final reader loaded from its actual export. original_proposal
    is the verified original role, not a newly initialized replacement. Metadata
    identities are consistency checks; they do not authenticate arbitrary input.
    Both objects must share the same immutable base/omega instance.
    """
    for x in (task_modules,proposal_modules,task_profile,proposal_profile,source_package):_digest(x)
    if (type(task) is not SuccessorMMLAModel or type(original_proposal) is not type(task)
            or task is original_proposal or task.mixer is not original_proposal.mixer
            or task.substrate_identity!=original_proposal.substrate_identity
            or set(task._modules)!=set(original_proposal._modules)):
        raise ValueError('original shared substrate and exact successor role types required')
    for m in (task,original_proposal):
        if any(x.training for x in m.modules()) or any(p.requires_grad for p in m.parameters()):
            raise ValueError('both actual input roles must be closed and frozen')
        m.validate_state_configuration()
        version=m.rho.version
        if version.dtype!=torch.int64 or version.ndim!=0 or not 0<=int(version)<2**63-1:
            raise ValueError('valid slow initialization version required')
        _initialization(m.rho)
        if any(not torch.isfinite(x).all() for x in m.new_state_dict().values()):
            raise ValueError('finite original module state required')
    if task.profile.identity!=task_profile or original_proposal.profile.identity!=proposal_profile:
        raise ValueError('actual admitted profile identity changed')
    old,new=asdict(original_proposal.profile),asdict(task.profile)
    if old['local_tokens']!=2048 or new['local_tokens']!=4096:
        raise ValueError('declared 2048 to 4096 migration only')
    if {k:v for k,v in old.items() if k!='local_tokens'}!={k:v for k,v in new.items() if k!='local_tokens'}:
        raise ValueError('no undeclared profile migration')
    actual_task=tensor_digest(task.new_state_dict());actual_proposal=tensor_digest(original_proposal.new_state_dict())
    if actual_task!=task_modules or actual_proposal!=proposal_modules:
        raise ValueError('actual original task/proposal bytes differ from admitted identities')
    initial=_initialization(original_proposal.rho)
    for k in ('a','b','version'):
        a,b=getattr(task.rho,k),getattr(original_proposal.rho,k)
        if a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device:
            raise ValueError('same rho layout, dtype and device required')
    inherited_rho=tensor_digest(original_proposal.rho.state_dict())
    # Common children are reused from the final reader; the proposal rho is
    # copied from the original role, with no random initialization or training.
    memo={id(m):m for k,m in task._modules.items() if k!='rho'}
    inherited=copy.deepcopy(original_proposal.rho)
    memo[id(task.rho)]=inherited
    result=copy.deepcopy(task,memo)
    result.module_status=dict(task.module_status,rho=original_proposal.module_status['rho'])
    result.eval().requires_grad_(False);rebind(result)
    check_role_pair(task,result)
    for k in ('a','b','version'):
        if getattr(result.rho,k).data_ptr()==getattr(original_proposal.rho,k).data_ptr():
            raise ValueError('inherited rho aliases source storage')
    for k in ('a','b'):
        if result.rho.initialization[k].data_ptr()==original_proposal.rho.initialization[k].data_ptr():
            raise ValueError('initialization record aliases source storage')
    if tensor_digest(result.rho.state_dict())!=inherited_rho or _initialization(result.rho)!=initial:
        raise ValueError('rho or initialization changed during inheritance')
    if (tensor_digest(task.new_state_dict())!=actual_task
            or tensor_digest(original_proposal.new_state_dict())!=actual_proposal):
        raise ValueError('migration mutated an input role')
    shared=tensor_digest({k:v for k,v in task.new_state_dict().items() if not k.startswith('rho.')})
    receipt=dict(schema='orbit-inherited-proposal-role-v1',source_package=source_package,
        task_modules=actual_task,source_proposal_modules=actual_proposal,proposal_modules=tensor_digest(result.new_state_dict()),
        task_profile=task.profile.identity,source_proposal_profile=original_proposal.profile.identity,
        proposal_profile=result.profile.identity,shared_modules=shared,
        inherited_rho=inherited_rho,inherited_initialization=initial,
        proposal_rho_status=result.module_status['rho'],status='CONSTRUCTED_NOT_RUNTIME_QUALIFIED',
        migration={'local_tokens':[2048,4096],'shared_modules':'final task reader','rho':'original proposal'},
        runtime_qualified=False,J_qualification_inherited=False,successor_admissions=0)
    return result,dict(receipt,identity=digest(receipt))
