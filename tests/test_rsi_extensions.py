import copy,dataclasses,unittest
import torch
from mmla_rsi.neural import LearningResidual
from mmla_rsi.rules import Language,OPS,parse_rule,evaluate,PrefixGrammar
from mmla_rsi.updater import UpdateBounds,apply_update
from mmla_rsi.evidence import Record,MemoryBounds,EvidenceMemory
from mmla_rsi.proposals import TokenCodec,ModelProposer

class RulesTests(unittest.TestCase):
 def test_prefix_domain_and_safe_division(self):
  lang=Language();rule=parse_rule(lang,'rtma',['sdiv','c1','c0'])
  self.assertAlmostEqual(evaluate(lang,rule,memory_features=[0]*8)[0],1e6,delta=1)
  with self.assertRaises(ValueError):parse_rule(lang,'rtt',['x7','c1','c0'])
  with self.assertRaises(ValueError):parse_rule(Language(max_depth=0),'rtma',['add','c0','c1'])
  with self.assertRaises(ValueError):parse_rule(lang,'rtma',['add','c0'])
 def test_paper_momentum_projection_and_no_mutation(self):
  rho=LearningResidual(3,seed=3);fast=rho.begin(7);before=fast.identity
  lang=Language();rule=parse_rule(lang,'rtt',['c3','c1','c4']) # .1, 1, .5
  ga,gb=torch.ones_like(fast.a),torch.ones_like(fast.b)
  out,receipt=apply_update(rho,fast,(ga,gb),language=lang,rule=rule,bounds=UpdateBounds(radius=.005),loss=.2,previous_loss=0,opportunity=0,opportunity_cap=4)
  expected=.5/(torch.cat((ga.flatten(),gb.flatten())).norm()+lang.epsilon)
  torch.testing.assert_close(out.momentum_a,torch.full_like(ga,expected))
  self.assertLessEqual(float((out.a-rho.a).abs().max().detach()),.005001)
  self.assertEqual(before,fast.identity);self.assertTrue(receipt['factors_changed'])
  out.momentum_a.fill_(.2);out.momentum_b.fill_(.2)
  next_state,_=apply_update(rho,out,(ga*0,gb*0),language=lang,rule=rule,bounds=UpdateBounds(radius=.5),loss=.2,previous_loss=.2,opportunity=1,opportunity_cap=4)
  self.assertFalse(torch.equal(out.a,next_state.a)) # zero current gradient, retained momentum
  original=next_state.identity
  with self.assertRaises(FloatingPointError):apply_update(rho,next_state,(ga*float('nan'),gb),language=lang,rule=rule,bounds=UpdateBounds(),loss=.2,previous_loss=0,opportunity=2,opportunity_cap=4)
  self.assertEqual(next_state.identity,original)
 def test_zero_b_first_gradient_and_reset_anchor(self):
  rho=LearningResidual(3,seed=4);state=rho.begin(9)
  loss=rho.inject(torch.ones(2,3),state).square().sum();ga,gb=torch.autograd.grad(loss,(state.a,state.b))
  self.assertEqual(int(ga.count_nonzero()),0);self.assertGreater(int(gb.count_nonzero()),0)
  with torch.no_grad():rho.b.fill_(.05)
  state=rho.begin(9);reset=rho.reset(state);self.assertTrue(torch.equal(reset.b,rho.b))

class MemoryTests(unittest.TestCase):
 def setUp(self):
  self.lang=Language(max_nodes=63,max_depth=8)
  self.mem=EvidenceMemory(MemoryBounds(slots=1),reader_identity='fixture-ascii',count_tokens=len,namespace='fixture',source_prefix='p/')
  # (1-null)*((1-occ)*(1-loss)+occ*(old_loss-loss))
  self.rule=parse_rule(self.lang,'rtma',['mul','sub','c1','x0','add','mul','sub','c1','x1','sub','c1','x2','mul','x1','sub','x4','x2'])
 def test_write_eviction_and_exact_null(self):
  m,r=self.mem.manage(Record('old-action',.8,.1,'completed','p/1',0),language=self.lang,rule=self.rule,opportunity_cap=4)
  self.assertEqual(r['status'],'COMMIT');self.assertEqual(self.mem.version,0)
  n,r=m.manage(Record('new-action',.2,.1,'completed','p/2',1),language=self.lang,rule=self.rule,opportunity_cap=4)
  self.assertNotIn('old-action',n.read()[0]);self.assertIn('new-action',n.read()[0])
  same,r=n.manage(Record('worse-action',.9,.1,'completed','p/3',2),language=self.lang,rule=self.rule,opportunity_cap=4)
  self.assertIs(same,n);self.assertEqual(r['status'],'NULL')
  loaded=EvidenceMemory.restore(n.export(),count_tokens=len,expected_contract=self.mem.contract);self.assertEqual(n.identity,loaded.identity)
 def test_bad_payload_is_failure_protected_slot_is_null(self):
  before=self.mem.identity
  with self.assertRaises(ValueError):self.mem.manage(Record('x',float('nan'),0,'completed','p/x',0),language=self.lang,rule=self.rule,opportunity_cap=4)
  self.assertEqual(before,self.mem.identity)
  m=EvidenceMemory(MemoryBounds(slots=1),reader_identity='fixture',count_tokens=len,namespace='fixture',protected=(0,))
  same,r=m.manage(Record('x',.1,0,'completed','s',0),language=self.lang,rule=self.rule,opportunity_cap=4)
  self.assertIs(m,same);self.assertEqual(r['status'],'NULL')
 def test_tampered_derived_view_and_duplicate_identity(self):
  m,_=self.mem.manage(Record('x',.2,0,'completed','p/x',0),language=self.lang,rule=self.rule,opportunity_cap=4)
  s=m.export();s['rows'][0]['derived_hex']='00'
  with self.assertRaises(ValueError):EvidenceMemory.restore(s,count_tokens=len,expected_contract=self.mem.contract)
  with self.assertRaises(ValueError):m.manage(Record('different',.1,0,'completed','p/x',0),language=self.lang,rule=self.rule,opportunity_cap=4)

class ProposalTests(unittest.TestCase):
 def test_actual_masked_sample_replay_gradient_and_incomplete_prefix(self):
  import tiny_helper as test_neural
  f=test_neural.NeuralTests();f.setUp();model=f.m.eval()
  class Tok:
   def encode(self,s,add_special_tokens=False):return [ord(c)%32+1 for c in s]
  for cap in (3,2):
   lang=Language(max_nodes=3,max_depth=0,max_tokens=cap)
   symbols=tuple('c'+str(i) for i in range(7))+tuple('x'+str(i) for i in range(4))+OPS
   codec=TokenCodec(tuple((s,i+1) for i,s in enumerate(symbols)))
   p=ModelProposer(model,Tok(),language=lang,mode='rtt',codec=codec,header='p',parent_identity='parent',memory_token_allowance=8,tokenizer_identity='fixture-ord-mod32')
   mem=EvidenceMemory(MemoryBounds(slots=1,read_tokens=8),reader_identity='ascii',count_tokens=len,namespace='fixture')
   fast=model.rho.begin(19);other=model.rho.begin(19)
   r=p.sample(fast,mem);rr=p.sample(other,mem)
   self.assertEqual(r.tokens,rr.tokens);self.assertEqual(r.rule is not None,cap==3)
   self.assertFalse(torch.equal(r.rng_before,r.rng_after))
   g,e=p.gradient(fast,r,.5)
   self.assertTrue(all(torch.isfinite(x).all() for x in g));self.assertLess(abs(e['replay_delta']),1e-5)
   self.assertGreater(float(g[1].abs().sum()),0)
   self.assertEqual(r.memory_read['memory'],mem.identity)


class RevisionTests(unittest.TestCase):
 def fixture(self, *, lang=None, temperature=1., header='p', tokenizer_identity='fixture-ord-mod32'):
  import tiny_helper as test_neural
  f=test_neural.NeuralTests();f.setUp();model=f.m.eval()
  class Tok:
   def encode(self,s,add_special_tokens=False):return [ord(c)%32+1 for c in s]
  lang=lang or Language(max_nodes=3,max_depth=0,max_tokens=3)
  symbols=tuple('c'+str(i) for i in range(7))+tuple('x'+str(i) for i in range(4))+OPS
  p=ModelProposer(model,Tok(),language=lang,mode='rtt',codec=TokenCodec(tuple((s,i+1) for i,s in enumerate(symbols))),header=header,parent_identity='parent',memory_token_allowance=8,tokenizer_identity=tokenizer_identity,temperature=temperature)
  mem=EvidenceMemory(MemoryBounds(slots=1,read_tokens=8),reader_identity='ascii',count_tokens=len,namespace='proposal/0')
  return model,p,mem
 def test_interpreter_uses_fp32_under_default_float64(self):
  lang=Language();rule=parse_rule(lang,'rtma',['sdiv','c1','c0'])
  expected=evaluate(lang,rule,memory_features=[0]*8)
  old=torch.get_default_dtype()
  try:
   torch.set_default_dtype(torch.float64)
   self.assertEqual(evaluate(lang,rule,memory_features=[0]*8),expected)
  finally:torch.set_default_dtype(old)
 def test_complete_serialization_failure_retains_prefix_and_gradient(self):
  model,p,mem=self.fixture(lang=Language(max_nodes=3,max_depth=0,max_tokens=3,max_bytes=1))
  fast=model.rho.begin(19);r=p.sample(fast,mem)
  self.assertEqual(len(r.tokens),3);self.assertIsNone(r.rule)
  self.assertEqual(r.failure,'serialization-capacity');self.assertTrue(r.sampled_log_probability<0)
  g,_=p.gradient(fast,r,1.);self.assertTrue(all(torch.isfinite(x).all() for x in g))
 def test_operational_failure_retains_partial_probability_rng_and_cost(self):
  from mmla_rsi.proposals import SamplingFailure
  model,p,mem=self.fixture();original=p._forward;calls=0
  def broken(*args):
   nonlocal calls
   calls+=1
   if calls==2:raise RuntimeError('injected fixture')
   return original(*args)
  p._forward=broken;fast=model.rho.begin(19)
  with self.assertRaises(SamplingFailure) as caught:p.sample(fast,mem)
  r=caught.exception.rollout
  self.assertEqual(len(r.tokens),1);self.assertEqual(len(r.masks),1)
  self.assertEqual(r.cost['forward_calls_attempted'],2)
  self.assertTrue(r.sampled_log_probability<0);self.assertFalse(torch.equal(r.rng_before,r.rng_after))
  self.assertTrue(torch.equal(fast.rng_state,r.rng_after));self.assertEqual(r.digest(),r.seal)
  with self.assertRaises(ValueError):p.gradient(fast,r,1.)
 def test_replay_rejects_configuration_prompt_mask_and_model_changes(self):
  model,p,mem=self.fixture();fast=model.rho.begin(19);r=p.sample(fast,mem)
  for key,value in [('tokens',r.tokens[:-1]),('masks',r.masks[:-1]),('prompt',(1,)+r.prompt),('memory_text','changed'),('sampled_log_probability',0.)]:
   altered=dataclasses.replace(r,**{key:value})
   with self.assertRaises(ValueError):p.gradient(fast,altered,.5)
  for kw in [dict(temperature=.5),dict(header='q'),dict(tokenizer_identity='different-tokenizer')]:
   q=ModelProposer(model,p.tokenizer,language=p.language,mode='rtt',codec=p.codec,header=kw.get('header','p'),parent_identity='parent',memory_token_allowance=8,tokenizer_identity=kw.get('tokenizer_identity','fixture-ord-mod32'),temperature=kw.get('temperature',1.))
   with self.assertRaises(ValueError):q.gradient(fast,r,.5)
  model.train()
  with self.assertRaises(ValueError):p.gradient(fast,r,.5)
  model.eval()
  with torch.no_grad():model.mixer.get_input_embeddings().weight.add_(1)
  with self.assertRaises(ValueError):p.gradient(fast,r,.5)
 def test_mutated_sampler_configuration_rejected_at_both_entries(self):
  model,p,mem=self.fixture();fast=model.rho.begin(19);r=p.sample(fast,mem)
  for key,value in [('temperature',.5),('header','q'),('mode','rtma'),('language',Language()),
                    ('mapping',{'c0':1}),('replay_absolute_tolerance',.1),('memory_token_allowance',9),
                    ('tokenizer_identity','other'),('parent_identity','other-parent')]:
   old=getattr(p,key);setattr(p,key,value)
   try:
    with self.assertRaises(ValueError):p.sample(fast,mem)
    with self.assertRaises(ValueError):p.gradient(fast,r,.5)
   finally:setattr(p,key,old)
 def test_probability_delta_rejects_before_gradient(self):
  model,p,mem=self.fixture();fast=model.rho.begin(19);r=p.sample(fast,mem)
  # Re-sealing simulates a corrupt producer, so the numerical check itself is tested.
  r=dataclasses.replace(r,sampled_log_probability=r.sampled_log_probability+.01)
  r=dataclasses.replace(r,seal=r.digest())
  with self.assertRaises(FloatingPointError):p.gradient(fast,r,.5)
 def test_restore_external_contract_and_strict_rows(self):
  m=EvidenceMemory(MemoryBounds(slots=1),reader_identity='fixed-tokenizer-reader',count_tokens=len,namespace='proposal/g0',source_prefix='p/')
  lang=Language();rule=parse_rule(lang,'rtma',['sub','c1','x0'])
  n,_=m.manage(Record('x',.2,.1,'completed','p/1',0),language=lang,rule=rule,opportunity_cap=4)
  for key,value in [('namespace','task/g0'),('reader','different'),('protected',[0]),('source_prefix',''),('bounds',{**m.contract['bounds'],'slots':2})]:
   s=n.export();s['contract'][key]=value
   with self.assertRaises(ValueError):EvidenceMemory.restore(s,count_tokens=len,expected_contract=m.contract)
  for key,value in [('version',1.),('slot',0.),('predecessor',1),('predecessor','bad')]:
   s=n.export();s['rows'][0][key]=value
   with self.assertRaises(ValueError):EvidenceMemory.restore(s,count_tokens=len,expected_contract=m.contract)
  s=n.export();s['version']=1.
  with self.assertRaises(ValueError):EvidenceMemory.restore(s,count_tokens=len,expected_contract=m.contract)
  with self.assertRaises(ValueError):Record('x',.2,.1,'task-failure','p/1',0).validate()
  self.assertTrue(Record('x',1.,.3,'task-failure','p/1',0).validate())

if __name__=='__main__':unittest.main()

