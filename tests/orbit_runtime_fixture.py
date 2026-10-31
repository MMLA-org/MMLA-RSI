"""Synthetic CPU roles and task endpoints; no trained Orbit or environment."""
import sys
from types import SimpleNamespace
import torch
from mmla_orbit import verify_runtime

runtime, _, _ = verify_runtime()
sys.path.insert(0, str(runtime))
from content_rules import Language, parse_rule, identity
from content_evidence import EvidenceMemory, MemoryBounds
from edit_role import EditInterface
from recursion_core import wire
from support_readiness import ReadyBlock
from logical_public_rule import RULE

LANG = Language(max_nodes=31, max_tokens=31, max_depth=3, max_instructions=31)
PARENT = ('add', 'sub', 'c1', 'x0', 'add', 'sub', 'c1', 'x1', 'x6')


class Rho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([.12, -.08]))
        self.b = torch.nn.Parameter(torch.tensor([.11, .03]))
        self.register_buffer('version', torch.tensor(156))
        self.initialization = dict(a=self.a.detach().clone(), b=self.b.detach().clone(), seed=17)

    def invalidate(self):
        pass


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rho = Rho()
        self.backbone = torch.nn.Parameter(torch.randn(32, 2,
            generator=torch.Generator().manual_seed(7)), requires_grad=False)
        self.mixer = torch.nn.Identity()
        self.profile = SimpleNamespace(local_tokens=4096, identity='a'*64)
        self.substrate_identity = {'base': 'b'*64}
        self.module_status = {'rho': 'TRAINED_TASK_RETURN_EDIT_PROPOSAL', 'backbone': 'FROZEN'}
        self.eval().requires_grad_(False)

    def forward_batch(self, ids, mask, *, fast_states, memory, logit_positions):
        assert fast_states is None and memory is None
        out = []
        for pos in logit_positions.tolist():
            scale = 1 + float(ids[0, :pos+1].sum() % 7) / 10
            out.append(self.backbone @ (self.rho.a + scale * self.rho.b))
        return torch.stack(out).unsqueeze(0)

    def validate_state_configuration(self):
        pass

    def restore_new_state(self, state):
        with torch.no_grad():
            for name, value in state.items():
                self.state_dict()[name].copy_(value)

    def new_state_dict(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if not k.startswith('mixer.')}


class Tokenizer:
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text in self.alphabet[:31]:
            return [self.alphabet.index(text)]
        return [ord(c) % 32 for c in text]

    def decode(self, ids, **kwargs):
        return ''.join(self.alphabet[i] for i in ids)


def interface_memory():
    return (EditInterface(Tokenizer(),tokenizer_identity="tokenizer",language=LANG,vocabulary_size=32),
        EvidenceMemory(MemoryBounds(),reader_identity="tokenizer",count_tokens=len,namespace="test",source_prefix="test/"))


class FakeTaskEvaluator:
    def __init__(self,seed):self.pair_seed=seed;self.model_identity='d'*64;self.calls=[]
    def factory(self,rule):return None
    def reader(self,rule):return 'e'*64
    def evaluate(self,rule,block):
        self.calls.append((rule.identity,len(block.permits)))
        rows=[dict(permit=p.identity,task=p.task_id,source_floor=p.floor,control='candidate',won=False,loss=1.,
                   numerical_commits=0,common_initial_writes_excluded=True,steps=2,phi='a'*64+':942:0:0:test') for p in block.permits]
        return dict(schema='orbit-fixed-phi-logical-block-v1',status='COMPLETE',role=block.role,
            block_sha256=block.block_sha256,population_sha256=block.population_sha256,reader='e'*64,
            rule=wire(rule),control='candidate',pair_seed=self.pair_seed,admission_performed=False,
            successor_installed=False,worlds=rows,forward_input_tokens=32,forward_input_cap=320,
            mean_loss=1.,mean_cost=.1)


class Evaluator(FakeTaskEvaluator):
    def __init__(self, baseline=None, *, pair_seed=17, **kwargs):
        super().__init__(pair_seed)


def block(role='DEV', lock=None):
    count = 32 if role == 'DEV' else 1024
    permits=tuple(SimpleNamespace(semantic_sha256=identity([role,i]),
        identity=identity(['permit',role,i]), task_id=role+str(i),floor=str(i)) for i in range(count))
    return ReadyBlock('synthetic', *(['a'*64]*6), role, permits, lock)


def strong(b, **unused):
    rows=[dict(task=p.task_id,permit=p.identity,source_floor=p.floor,
               won=False,loss=1.,steps=2) for p in b.permits]
    return dict(schema='orbit-logical-public-rule-v1',status='COMPLETE',rule=RULE,
        role=b.role,block_sha256=b.block_sha256,population_sha256=b.population_sha256,
        world_seed=0,model_forwards=0,worlds=rows,mean_loss=1.)


def orbit_fixture():
    task, proposal, original = Model(), Model(), Model()
    with torch.no_grad():
        task.rho.version.fill_(942); proposal.rho.version.fill_(184)
    interface, memory=interface_memory()
    return SimpleNamespace(task=task,proposal=proposal,original_proposal=original,
        config=dict(selected_seed=17,roles={'17':{}},public_runtime_identity='e'*64),
        source_identity='f'*64,seed=17,interface=interface,memory0=memory,
        initial_manager=parse_rule(LANG,'rtma',PARENT))
