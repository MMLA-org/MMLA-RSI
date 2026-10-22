"""CPU control-loop demonstration; synthetic endpoint, NOT an RSI benchmark."""
import argparse, json, hashlib
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from mmla_rsi.adapter import AdapterProfile, CausalLarcAdapter
from mmla_rsi.rules import Language, OPS, parse_rule, identity
from mmla_rsi.proposals import ModelProposer, TokenCodec
from mmla_rsi.evidence import EvidenceMemory, MemoryBounds
from mmla_rsi.updater import UpdateBounds
from mmla_rsi.roles import ProposalLearner
from mmla_rsi.evaluation import Case, Allowance, Evaluator, Meter
from mmla_rsi.generation import Controller, RunConfig
from mmla_rsi.journal import Journal

class Tok:
 def encode(self,text,add_special_tokens=False):return [ord(c)%32+1 for c in text]


class Factory:
 def __init__(self,model,language,attempts,mode):
  self.model=model;self.language=language;self.attempts=attempts;self.mode=mode
  self.identity=identity(dict(fixture='tiny-real-HF-proposal',mode=mode,language=language.identity,attempts=attempts))
 def create(self,parent,generation,seed,on_forward):
  symbols=tuple('c'+str(i) for i in range(7))+tuple('x'+str(i) for i in range(4 if self.mode=='rtt' else 8))+OPS
  codec=TokenCodec(tuple((s,i+1) for i,s in enumerate(symbols)))
  proposer=ModelProposer(self.model,Tok(),language=self.language,mode=self.mode,codec=codec,
   header='parent '+' '.join(parent.symbols),parent_identity=parent.identity,memory_token_allowance=1100,
   tokenizer_identity='fixture-ord32',on_forward=on_forward)
  mem=EvidenceMemory(MemoryBounds(slots=1,read_tokens=1100),reader_identity='fixture-ord32',count_tokens=len,
                     source_prefix='g'+str(generation)+'/',namespace='proposal/g'+str(generation))
  return ProposalLearner(proposer,parent,mem,seed=seed,opportunity_cap=self.attempts,update_bounds=UpdateBounds())


class EndpointFixture:
 """Explicit fake task endpoints for control-flow tests, never a research dataset."""
 identity='handwritten-terminal-loss-fixture-v1'
 def __init__(self,initial):self.initial=initial;self.calls=[]
 def run(self,rule,case,meter):
  meter.charge(1,'fixture_step');self.calls.append((rule.identity,case.task_id,case.seed))
  return 1. if rule.identity==self.initial.identity else .2


def case(i,prefix='dev',group=None):
 return Case(prefix+str(i),group or prefix+str(i),hashlib.sha256((prefix+str(i)).encode()).hexdigest(),1000+i+(0 if prefix=='dev' else 10000))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('rtt', 'rtma', 'joint'), default='joint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('choose a new demo output directory')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    mixer = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=48, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, attention_dropout=0.))
    model = CausalLarcAdapter(mixer, AdapterProfile(32, 2048, 8), seed=1,
                              substrate_identity={'random_demo_seed': 17})
    language = Language(max_nodes=12, max_depth=1, max_tokens=6)
    symbols = (['c3', 'c1', 'c0'] if args.mode in ('rtt', 'joint') else [])
    symbols += (['sub', 'c1', 'x0'] if args.mode in ('rtma', 'joint') else [])
    parent = parse_rule(language, args.mode, symbols)
    config = RunConfig(1, 3, (19,), 0., .05, .1, 1.,
        Allowance(1000000, 600), Allowance(10, 1), Allowance(10000, 3), Allowance(10000, 3), 1.)
    # This endpoint deliberately assigns lower fixture loss to non-parent rules.
    # It tests the handoff machinery; it does not measure useful adaptation.
    endpoint = EndpointFixture(parent)
    evaluator = Evaluator(endpoint, task_allowance=Allowance(2, .02))
    controller = Controller(language=language, initial_rule=parent, config=config,
        factory=Factory(model, language, 3, args.mode), evaluator=evaluator,
        dev=(case(0), case(1)), audits=(tuple(case(i, 'audit') for i in range(64)),),
        fit_groups=(), fit_payloads=(), journal=Journal(args.output, max_bytes=64<<20, max_cuts=64))
    result = controller.run()
    print(json.dumps({'purpose': 'synthetic control-flow fixture, not RSI effectiveness',
                      'result': result}, indent=2, default=str))


if __name__ == '__main__':
    main()
