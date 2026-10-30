"""Bound state factory shared by recorded-task training and future deployment."""
from dataclasses import asdict
from mmla_pdsa.full.alfworld_formatting import ALFWorldFormatter
from mmla_pdsa.full.encoding import FrozenCanonicalEncoder
from mmla_pdsa.full.observation_memory import ObservationCodec,observation_memory
from content_rules import Language,parse_rule
from mmla_pdsa.full.extensions.updater import UpdateBounds
from mmla_pdsa.full.neural import tensor_digest
from mmla_pdsa.full.profile import digest
from managed_task_state import ManagedTaskState
from task_source_manager import DOMAIN


class DemonstrationFactory:
    def __init__(self,model,original):
        self.model=model;tok=original.tokenizer;tid=original.config['tokenizer_identity']
        self.formatter=ALFWorldFormatter(tok,tokenizer_identity=tid,max_tokens=4096)
        self.codec=ObservationCodec(tok,model.profile,tokenizer_identity=tid)
        self.encoder=FrozenCanonicalEncoder(model,tok,tokenizer_identity=tid)
        interface=original.config['proposal_interface'];lang=dict(interface['language']);lang['constants']=tuple(lang['constants'])
        self.language=Language(**lang);self.bounds=UpdateBounds(**interface['update_bounds'])
        self.updater=parse_rule(self.language,'rtt',('c2','c1','c4'))
        self.manager=parse_rule(self.language,'rtma',('add','sub','c1','x0','add','sub','c1','x1','x6'))
        self.definition=dict(schema='orbit-managed-observation-interface-v1',profile=model.profile.identity,
            formatter=self.formatter.identity,codec=self.codec.identity,encoder=self.encoder.identity,
            language=asdict(self.language),bounds=asdict(self.bounds),updater=asdict(self.updater),
            manager=asdict(self.manager),task_memory_domain=DOMAIN,initial_source='common reset WRITE_BATCH_0',
            max_updates=8,horizon=64,evaluation_memory=False,retained_history='current authoritative observation rows only',
            fast_reset='current trained rho',past_caches='none across action/feedback boundaries')

    def reader_identity(self):
        return digest(dict(substrate=self.model.substrate_identity,
            modules=tensor_digest(self.model.new_state_dict()),interface=self.definition))

    def new_state(self,*,seed,tenant,manager=None):
        return ManagedTaskState(self.model,formatter=self.formatter,codec=self.codec,
            source=observation_memory(self.codec,self.encoder),reader_identity=self.reader_identity(),seed=seed,
            tenant=tenant,principal='actor',language=self.language,updater=self.updater,
            manager=self.manager if manager is None else manager,bounds=self.bounds,max_updates=8,horizon=64)

    def states(self,tasks,seeds):
        if len(tasks)!=len(seeds):raise ValueError('one distinct fast seed per task required')
        return [self.new_state(seed=seed,tenant=digest(dict(permit=task.permit_identity,
            trace=task.trace_identity,fast_seed=seed))) for task,seed in zip(tasks,seeds)]
