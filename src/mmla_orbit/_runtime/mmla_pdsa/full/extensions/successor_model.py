"""Unified retained-layer hybrid with an independently trained joint risk head."""
import copy
import torch
from ..model import MMLAModel
from ..neural import ActionRisk,tensor_digest
from ..profile import digest
from .hybrid import HybridMMLAModel
from .joint import JOINT_SCHEMA


class SuccessorMMLAModel(HybridMMLAModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(kwargs['seed'])
            self.joint_risk=ActionRisk(self.profile.neural_width+self.profile.key_width+14)
        self.joint_risk.register_buffer('feature_schema',torch.tensor(list(bytes.fromhex(JOINT_SCHEMA)),dtype=torch.uint8))
        self.joint_risk.to(device=self.rho.a.device,dtype=torch.float32)
        self.module_status['joint_risk']='UNTRAINED_UNCALIBRATED'
        self._bind_joint_schema()
        self._reference_initialized=False
        self._initial_modules=tensor_digest(self.new_state_dict())
        self._initial_status=copy.deepcopy(self.module_status)

    def _bind_joint_schema(self):
        x=self.joint_risk.feature_schema
        if x.dtype!=torch.uint8 or tuple(x.shape)!=(32,) or bytes(x.cpu().tolist()).hex()!=JOINT_SCHEMA:
            raise ValueError('joint feature schema bytes differ')
        self._joint_schema_guard=(id(x),x._version,x.dtype,x.device)

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        if hasattr(self,'joint_risk'):self._bind_joint_schema()
        return result

    def validate_joint(self):
        x=self.joint_risk.feature_schema
        if ((id(x),x._version,x.dtype,x.device)!=self._joint_schema_guard
                or self.joint_risk.network[0].in_features!=self.profile.neural_width+self.profile.key_width+14
                or self.joint_risk.updates.dtype!=torch.int64 or self.joint_risk.updates.ndim!=0
                or not 0<=int(self.joint_risk.updates)<2**63-1):
            raise ValueError('joint head configuration/counter changed')

    def validate_state_configuration(self):
        self.validate_joint()
        super().validate_state_configuration()

    def new_state_dict(self):
        self.validate_joint()
        return super().new_state_dict()

    def restore_new_state(self,state):
        self.validate_joint()
        if ('joint_risk.feature_schema' not in state
                or not torch.equal(state['joint_risk.feature_schema'].cpu(),self.joint_risk.feature_schema.cpu())):
            raise ValueError('saved joint feature schema differs')
        counter=state.get('joint_risk.updates')
        if (not torch.is_tensor(counter) or counter.dtype!=torch.int64 or counter.ndim!=0
                or not 0<=int(counter)<2**63-1):raise ValueError('saved joint update counter is invalid')
        super().restore_new_state(state);self._bind_joint_schema()

    def initialize_reference_modules(self,reference):
        """Explicit original or hybrid warm start; J stays independently untrained."""
        if (self._reference_initialized or tensor_digest(self.new_state_dict())!=self._initial_modules
                or self.module_status!=self._initial_status):
            raise ValueError('warm start requires the exact fresh untrained successor instance')
        if type(reference) not in (MMLAModel,HybridMMLAModel):raise ValueError('explicit original/hybrid reference only')
        expected_substrate=self.substrate_identity if type(reference) is HybridMMLAModel else {
            k:v for k,v in self.substrate_identity.items() if k!='mmla_hybrid'}
        if reference.mixer is not self.mixer or reference.profile!=self.profile or reference.substrate_identity!=expected_substrate:
            raise ValueError('same original source, exact profile and shared frozen mixer required')
        source=reference.new_state_dict();target=self.new_state_dict()
        omit=('joint_risk.',) if type(reference) is HybridMMLAModel else ('joint_risk.','hybrid.')
        if set(source)!={k for k in target if not k.startswith(omit)}:raise ValueError('complete warm-start modules required')
        target.update(source);self.restore_new_state(target)
        self.rho.initialization=copy.deepcopy(reference.rho.initialization)
        for k,v in reference.module_status.items():self.module_status[k]=v
        self._reference_initialized=True
        return dict(source=tensor_digest(source),target=tensor_digest(self.new_state_dict()),
            joint_risk='NEW_UNTRAINED',reader_qualification='REQUIRED_FOR_NEW_READER')
