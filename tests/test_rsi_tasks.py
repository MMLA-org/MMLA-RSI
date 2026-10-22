"""Real tiny-HF action gradients; handwritten adapter fixtures, no task claim."""
from dataclasses import replace
import copy,hashlib,json,unittest
import torch
from mmla_rsi.rules import Language,parse_rule
from mmla_rsi.updater import UpdateBounds
from mmla_rsi.evidence import EvidenceMemory,MemoryBounds
from mmla_rsi.task_policy import TaskPolicy,TaskSampling
from mmla_rsi.task_learner import TaskLearner,TaskFeedback
from mmla_rsi.task_execution import TaskExecutor,TaskLimits,TaskPayloads
from mmla_rsi.evaluation import Case,Allowance,Meter,Evaluator,TaskRunFailure,TaskInterrupted,validate_partitions


class Tokenizer:
 def encode(self,text,add_special_tokens=False):return [ord(c)%32+1 for c in text]
 def decode(self,ids,skip_special_tokens=False):return ''.join(chr(65+i%26) for i in ids)

class Backend:
 identity='handwritten-syntax-feedback-fixture-not-a-benchmark'
 configuration={'schema':'handwritten-syntax-feedback-fixture-v1','shared_state':False}
 def __init__(self):self.seen=[]
 def feedback(self,action,support,*,opportunity,meter):
  self.seen.append(copy.deepcopy(support));meter.charge(1,'fixture_feedback')
  return (0.,True,'completed') if action=='OK' else (1.,False,'task-failure')
 def score(self,action,final,*,meter):
  self.seen.append(copy.deepcopy(final));meter.charge(1,'fixture_score')
  return float(action!='OK')


class TaskTests(unittest.TestCase):
 def setup(self,mode='joint',attempts=3,eos=(63,),tokens=2,tokenizer=None):
  import tiny_helper as test_neural
  fixture=test_neural.NeuralTests();fixture.setUp();m=fixture.m.eval()
  m.profile=replace(m.profile,local_tokens=2048)
  policy=TaskPolicy(m,tokenizer or Tokenizer(),sampling=TaskSampling(eos,tokens,32,1100),tokenizer_identity='fixture-char-v1')
  mem=EvidenceMemory(MemoryBounds(slots=2,read_tokens=1100),reader_identity='fixture-char-v1',count_tokens=len,namespace='task',source_prefix='task/')
  lang=Language(max_nodes=12,max_depth=1,max_tokens=6)
  rule=parse_rule(lang,mode,(['c3','c1','c0'] if mode in ('joint','rtt') else [])+(['sub','c1','x0'] if mode in ('joint','rtma') else []))
  role=TaskLearner(policy,lang,rule,mem,header='Produce OK',source='task/source',seed=17,attempts=attempts,update_bounds=UpdateBounds())
  return m,policy,mem,lang,rule,role
 def meter(self):return Meter(Allowance(100000,60))
 def feedback(self,role,rollout):
  role.accept_feedback(TaskFeedback(rollout.seal,role.attempt,role.source,1.,'task-failure',False),attempt_cost=.2,meter=self.meter())
 def test_full_vocab_capped_action_has_true_score_gradient(self):
  m,p,_,_,_,r=self.setup(mode='rtt',tokens=1)
  action=r.draw(self.meter());self.assertEqual(action.stop,'TOKEN_CAP')
  grads,receipt=p.gradient(r.fast,action,1.,meter=self.meter())
  self.assertEqual(float(grads[0].norm()),0.)
  self.assertGreater(float(grads[1].norm()),0.)
  self.assertLess(abs(receipt['replay_delta']),1e-4)
  self.feedback(r,action);self.assertEqual(r.fast.steps,1)
  self.assertTrue(all(x.grad is None for x in m.parameters()))
 def test_joint_next_action_reads_committed_M_and_private_Phi(self):
  m,p,mem,lang,rule,r=self.setup()
  first=r.draw(self.meter());self.feedback(r,first)
  self.assertEqual((r.numerical_commits,r.memory_commits),(1,1))
  second=r.draw(self.meter())
  self.assertEqual(second.memory_read['memory'],r.memory.identity)
  self.assertIn(first.action,second.memory_text)
  self.assertNotEqual(first.prompt,second.prompt)
  self.assertTrue(torch.equal(m.rho.b,torch.zeros_like(m.rho.b)))
 def test_single_opportunity_and_resolution_stop_without_unused_update(self):
  _,_,_,_,_,r=self.setup(attempts=1)
  a=r.draw(self.meter());self.feedback(r,a)
  self.assertTrue(r.closed);self.assertEqual((r.numerical_commits,r.memory_commits),(0,0))
  _,_,_,_,_,r=self.setup(eos=tuple(range(64)))
  a=r.draw(self.meter());self.assertEqual(a.stop,'EOS')
  r.accept_feedback(TaskFeedback(a.seal,0,r.source,0.,'completed',True),attempt_cost=.1,meter=self.meter())
  self.assertTrue(r.closed);self.assertEqual(r.fast.steps,0)
 def test_feedback_binding_and_restore_preserve_tape(self):
  _,p,mem,lang,rule,r=self.setup()
  a=r.draw(self.meter())
  with self.assertRaises(ValueError):r.accept_feedback(TaskFeedback(a.seal,1,r.source,1.,'task-failure',False),attempt_cost=.1,meter=self.meter())
  self.feedback(r,a);state=r.export()
  other=TaskLearner(p,lang,rule,mem,header='Produce OK',source='task/source',seed=17,attempts=3,update_bounds=UpdateBounds())
  other.restore(state)
  a=r.draw(self.meter());b=other.draw(self.meter())
  self.assertEqual(a.tokens,b.tokens);self.assertEqual(a.prompt,b.prompt)
  self.assertNotEqual(r.fast.a.data_ptr(),other.fast.a.data_ptr())
  self.assertNotEqual(r.epoch,other.epoch)
 def test_joint_memory_failure_keeps_numerical_commit(self):
  class Long(Tokenizer):
   def decode(self,ids,skip_special_tokens=False):return super().decode(ids,skip_special_tokens)*3000
  _,_,_,_,_,r=self.setup(tokens=1,tokenizer=Long())
  a=r.draw(self.meter())
  with self.assertRaises(ValueError):self.feedback(r,a)
  self.assertEqual((r.numerical_commits,r.memory_commits),(1,0))
  self.assertTrue(r.closed);self.assertEqual(r.receipts[-1]['status'],'FAILED')
  self.assertEqual(r.export()['fast']['steps'],1)
 def executor(self,**kw):
  _,policy,mem,lang,rule,_=self.setup(**kw)
  raw=json.dumps(dict(task_id='fixture',physical_group='physical-one',request='Produce OK',support={'visible':'syntax only'},final={'unread_answer':'OK'})).encode()
  bank=TaskPayloads({'fixture':raw},source_identity='handwritten-only');case=bank.case('fixture',51)
  limits=TaskLimits(3,Allowance(1,.1),Allowance(10000,.2),Allowance(4,.1),Allowance(10000,.2),Allowance(4,.1))
  backend=Backend()
  ex=TaskExecutor(policy,lang,payloads=bank,backend=backend,memory=mem,limits=limits,update_bounds=UpdateBounds())
  return ex,rule,case,backend
 def test_evaluator_preserves_actual_action_trace_and_complete_state(self):
  ex,rule,case,backend=self.executor(eos=tuple(range(64)))
  ev=Evaluator(ex,task_allowance=Allowance(100000,3))
  out=ev.block(rule,[case],meter=self.meter())
  row=out['results'][0];self.assertIsNone(row['error']);self.assertGreater(row['cost'],0)
  trace=row['trace'];self.assertTrue(trace['complete']);self.assertEqual(len(trace['actions']),3)
  self.assertEqual(trace['state']['numerical_commits'],2)
  for a in trace['actions']:self.assertNotIn('unread_answer',a['header']+a['memory_text'])
  self.assertEqual(backend.seen,[]) # Prototype never receives support or final.
  again=ex.run(rule,case,self.meter())
  self.assertEqual([a['tokens'] for a in trace['actions']],[a['tokens'] for a in again.trace['actions']])
 def test_bad_physical_group_or_payload_hash_is_not_relabelled_task(self):
  ex,rule,case,_=self.executor()
  with self.assertRaises(TaskRunFailure):ex.run(rule,replace(case,physical_group='new-name'),self.meter())
  with self.assertRaises(TaskRunFailure):ex.run(rule,replace(case,payload_sha256='0'*64),self.meter())
 def test_failure_after_real_update_is_retained_in_evaluation(self):
  ex,rule,case,backend=self.executor(eos=tuple(range(64)))
  def broken(own,*a,**kw):
   if kw['opportunity']==1:raise RuntimeError('handwritten injected operational failure')
   return Backend.feedback(own,*a,**kw)
  import types
  backend.feedback=types.MethodType(broken,backend)
  row=Evaluator(ex,task_allowance=Allowance(100000,3)).block(rule,[case],meter=self.meter())['results'][0]
  self.assertEqual(row['error'],'TaskRunFailure');self.assertEqual(row['loss'],1.)
  self.assertEqual(row['trace']['state']['numerical_commits'],1)
  self.assertEqual(len(row['trace']['actions']),2)
  self.assertGreater(row['cost'],0.)
 def test_changed_sampler_model_or_unclosed_terminal_restore_rejected(self):
  m,p,mem,lang,rule,r=self.setup(attempts=1)
  a=r.draw(self.meter());self.feedback(r,a);state=r.export();state['closed']=False
  other=TaskLearner(p,lang,rule,mem,header='Produce OK',source='task/source',seed=17,attempts=1,update_bounds=UpdateBounds())
  with self.assertRaises(ValueError):other.restore(state)
  p.sampling=replace(p.sampling,temperature=.9)
  with self.assertRaises(ValueError):p.check()

 def test_changed_role_and_executor_configuration_rejected(self):
  _,_,_,lang,rule,r=self.setup();a=r.draw(self.meter())
  r.rule=parse_rule(lang,'joint',['c0','c1','c0','sub','c1','x0'])
  with self.assertRaises(ValueError):self.feedback(r,a)
  ex,rule,case,_=self.executor();ex.bounds=replace(ex.bounds,radius=.5)
  with self.assertRaises(ValueError):ex.run(rule,case,self.meter())
  m,p,_,_,_,_=self.setup();m.profile=replace(m.profile,local_tokens=2047)
  with self.assertRaises(ValueError):p.check()
 def test_renamed_content_rejected_within_bank_and_across_partitions(self):
  ex,rule,case,_=self.executor()
  raw=ex.payloads._payloads['fixture'];obj=json.loads(raw);obj.update(task_id='renamed',physical_group='new-group')
  renamed=json.dumps(obj).encode()
  with self.assertRaises(ValueError):TaskPayloads({'fixture':raw,'renamed':renamed},source_identity='fixture')
  other=TaskPayloads({'renamed':renamed},source_identity='other-source').case('renamed',52)
  with self.assertRaises(ValueError):validate_partitions((),(),(case,),((other,),),(999,))
  with self.assertRaises(ValueError):validate_partitions((),(),(case,),((replace(other,content_sha256='a'*64),),),(999,),fit_contents=(case.content_sha256,))
 def test_interrupt_retains_evidence_and_never_starts_second_case(self):
  ex,rule,case,backend=self.executor(eos=tuple(range(64)))
  def interrupted(own,*a,**kw):
   kw['meter'].charge(1,'real_before_interrupt');raise KeyboardInterrupt('stop')
  import types
  backend.feedback=types.MethodType(interrupted,backend)
  out={};meter=self.meter()
  with self.assertRaises(TaskInterrupted) as caught:
   Evaluator(ex,task_allowance=Allowance(100000,3)).block(rule,[case,case],meter=meter,out=out)
  self.assertIsInstance(caught.exception.cause,KeyboardInterrupt)
  self.assertTrue(out['interrupted']);self.assertEqual(len(out['results']),1)
  self.assertGreater(meter.used,0);self.assertEqual(len(out['results'][0]['trace']['actions']),1)
  self.assertFalse(out['complete']);self.assertTrue(out['results'][0]['trace']['state']['closed'])

 def test_controller_persists_stop_on_interrupted_task_block(self):
  import tempfile
  from test_rsi_generation import GenerationTests
  with tempfile.TemporaryDirectory() as root:
   controller,_,runner=GenerationTests().setup_run(root)
   def interrupted(rule,case,meter):
    meter.charge(1,'real-before-interrupt')
    raise TaskInterrupted({'state':'handwritten partial evidence'},KeyboardInterrupt('stop'))
   runner.run=interrupted
   controller.step()
   with self.assertRaises(TaskInterrupted):controller.step()
   self.assertEqual(controller.phase,'STOPPED')
   self.assertTrue(controller.archive[0]['evaluation']['interrupted'])
   self.assertGreater(controller.used,0)
   _,saved=controller.journal.load_bound()
   self.assertEqual(saved['phase'],'STOPPED')
   self.assertTrue(saved['archive'][0]['evaluation']['interrupted'])

 def test_reader_identity_includes_fresh_mixer_configuration(self):
  m,p,_,_,_,_=self.setup()
  clone=copy.deepcopy(m);clone.mixer.config.attention_dropout=.125
  q=TaskPolicy(clone,Tokenizer(),sampling=p.sampling,tokenizer_identity=p.tokenizer_identity)
  self.assertNotEqual(p.identity,q.identity)
 def test_interrupted_sampling_keeps_actual_prefix_and_spent_rng(self):
  ex,rule,case,_=self.executor(tokens=2)
  forward=ex.policy.model.forward_batch;calls=0
  def interrupted(*args,**kwargs):
   nonlocal calls
   calls+=1
   if calls==2:raise KeyboardInterrupt('sample stop')
   return forward(*args,**kwargs)
  ex.policy.model.forward_batch=interrupted
  with self.assertRaises(TaskInterrupted) as caught:ex.run(rule,case,self.meter())
  trace=caught.exception.trace;self.assertIsInstance(caught.exception.cause,KeyboardInterrupt)
  self.assertEqual(len(trace['actions']),1);action=trace['actions'][0]
  self.assertEqual(len(action['tokens']),1);self.assertLess(action['log_probability'],0)
  self.assertFalse(torch.equal(action['rng_before'],action['rng_after']))
  self.assertTrue(trace['state']['closed']);self.assertGreater(trace['resources']['work'],0)
 def test_controller_storage_failure_does_not_swallow_interrupt(self):
  import tempfile
  from unittest.mock import patch
  from test_rsi_generation import GenerationTests
  from mmla_rsi.journal import SnapshotTooLarge
  with tempfile.TemporaryDirectory() as root:
   c,_,runner=GenerationTests().setup_run(root)
   def interrupted(rule,case,meter):
    meter.charge(1,'real-before-interrupt')
    raise TaskInterrupted({'state':'handwritten partial evidence'},KeyboardInterrupt('stop'))
   runner.run=interrupted;c.step();save=c.journal.save
   def limited(state):
    if state.get('schema')=='rsi-controller-v1' and state.get('phase')=='STOPPED':raise SnapshotTooLarge('fixture')
    return save(state)
   with patch.object(c.journal,'save',side_effect=limited):
    with self.assertRaises(TaskInterrupted):c.step()
   self.assertEqual(c.phase,'STOPPED');self.assertFalse(c.persistence_complete)
   _,saved=c.journal.load_bound()
   self.assertFalse(saved['full_snapshot_saved'])
   self.assertEqual(saved['failure']['control_interruption']['reason'],'TaskInterrupted')

if __name__=='__main__':unittest.main()
