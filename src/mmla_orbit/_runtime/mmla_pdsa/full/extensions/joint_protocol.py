"""Frozen dependencies shared by joint labels and future-blind deployment."""
from dataclasses import asdict,dataclass
from ..neural import tensor_digest
from ..actual_tasks import BEHAVIOR_ID
from ..profile import digest

@dataclass(frozen=True)
class JointEndpoint:
    eos_ids:tuple=(1,130073)
    max_new_tokens:int=64
    schema:str='original-four-query-greedy-json-v1'
    world_clock:int=4
    query_forms:tuple=('read_a','read_b','inventory','equal_ab')
    generation:str='greedy'
    scorer:str='strict-json-exact-v1'
    def __post_init__(self):
        object.__setattr__(self,'eos_ids',tuple(self.eos_ids))
        object.__setattr__(self,'query_forms',tuple(self.query_forms))
        if (not self.eos_ids or len(set(self.eos_ids))!=len(self.eos_ids)
                or any(type(x) is not int or x<0 for x in self.eos_ids)
                or type(self.max_new_tokens) is not int or not 1<=self.max_new_tokens<=512
                or self.schema!='original-four-query-greedy-json-v1' or self.world_clock!=4
                or self.query_forms!=('read_a','read_b','inventory','equal_ab')
                or self.generation!='greedy' or self.scorer!='strict-json-exact-v1'):
            raise ValueError('fixed exact terminal action law and scorer required')
    @property
    def identity(self):return digest(asdict(self))


def joint_protocol(driver):
    from .joint import JointStreamDriver,JOINT_SCHEMA
    if type(driver) is not JointStreamDriver:raise ValueError('declared joint runtime required')
    driver._check();p=driver.problem
    return dict(schema='mmla-early-joint-frozen-dependencies-v1',features=JOINT_SCHEMA,
        reader=p.reader_identity(),rho=tensor_digest(p.model.rho.state_dict()),
        profile=p.model.profile.identity,substrate=p.model.substrate_identity,
        encoder=p.memory.encoder.identity,gate=p.memory.semantic_gate.identity,
        endpoint=asdict(driver.endpoint),tokenizer=driver.tokenizer_identity,boundary=asdict(driver.boundary),limits=asdict(driver.limits),
        updater=dict(driver.updater),continuation=BEHAVIOR_ID,
        source='EngineeringStreamTask-original-period-command-newline-v2',timing='early-next-original-event',
        tenant=p.tenant,principal=p.principal)
