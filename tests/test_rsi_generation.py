"""Handwritten controller engineering fixtures; no empirical RSI claim."""
from dataclasses import replace
import tempfile,unittest,hashlib,copy
import torch
from mmla_rsi.rules import Language,OPS,parse_rule,identity
from mmla_rsi.proposals import ModelProposer,TokenCodec
from mmla_rsi.evidence import EvidenceMemory,MemoryBounds
from mmla_rsi.updater import UpdateBounds
from mmla_rsi.roles import ProposalLearner,components
from mmla_rsi.evaluation import Case,Allowance,Evaluator,Meter,admission,validate_partitions,ResourceExhausted
from mmla_rsi.generation import Controller,RunConfig
from mmla_rsi.journal import Journal,JournalIntegrityError,SnapshotTooLarge,StaleController,serialize
from pathlib import Path
from unittest.mock import patch


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


class GenerationTests(unittest.TestCase):
 def setup_run(self,root,mode='rtt',N=3,K=1,extra=None):
  import tiny_helper as test_neural
  f=test_neural.NeuralTests();f.setUp();model=f.m.eval()
  model.profile=replace(model.profile,local_tokens=2048)
  lang=Language(max_nodes=8,max_depth=0,max_tokens=4 if mode=='joint' else 3)
  symbols=['c3','c1','c0'] if mode=='rtt' else ['c1'] if mode=='rtma' else ['c3','c1','c0','c1']
  # In memory modes positive constant ties with NULL, so use a write-preferring tree.
  if mode in ('rtma','joint'):
   lang=Language(max_nodes=8,max_depth=1,max_tokens=6)
   symbols=(symbols[:3] if mode=='joint' else [])+['sub','c1','x0']
  parent=parse_rule(lang,mode,symbols);factory=Factory(model,lang,N,mode)
  cfg=RunConfig(K,N,tuple(19+k for k in range(K)),0.,.05,.1,1.,
    Allowance(1000000,600),Allowance(10,1),Allowance(10000,3),Allowance(10000,3),1.)
  executor=EndpointFixture(parent);ev=Evaluator(executor,task_allowance=Allowance(2,.02))
  audits=tuple(tuple(case(i,'audit'+str(k)+'/') for i in range(64)) for k in range(K))
  # Distinct tapes between audit generations.
  audits=tuple(tuple(replace(x,seed=x.seed+k*1000) for x in b) for k,b in enumerate(audits))
  journal=Journal(root,max_bytes=64*1024*1024,max_cuts=64)
  args=dict(language=lang,initial_rule=parent,config=cfg,factory=factory,evaluator=ev,dev=(case(0),case(1)),
   audits=audits,fit_groups=(),fit_payloads=(),journal=journal)
  if extra:args.update(extra)
  return Controller(**args),args,executor
 def test_real_proposal_parent_update_lock_audit_and_adopt(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,runner=self.setup_run(root);result=c.run()
   self.assertEqual(result['status'],'DONE',result['failure'])
   self.assertEqual(len(c.edges),1)
   archive=c.archives[0]
   self.assertFalse(archive['attempts'][0]['eligible']);self.assertTrue(archive['attempts'][1]['eligible'])
   self.assertTrue(archive['proposal_state']['fast']['b'].abs().sum()>0)
   self.assertEqual(archive['proposal_state']['fast']['steps'],2)
   self.assertEqual(archive['locked']['candidate'],c.versions[-1])
   self.assertEqual(archive['audit']['decision']['independent_groups'],64)
   self.assertEqual(c.edges[0]['functional_participation'],'NOT_MEASURED')
   self.assertEqual(Journal(root,max_bytes=64*1024*1024,max_cuts=64).load()['phase'],'DONE')
   devcalls=[x for x in runner.calls if x[1].startswith('dev')]
   self.assertEqual({(x[1],x[2]) for x in devcalls},{('dev0',1000),('dev1',1001)})
 def test_no_null_parent_memory_successor_forced(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,_=self.setup_run(root,mode='rtma')
   null=parse_rule(c.language,'rtma',['c0']);args['initial_rule']=null;args['evaluator']=Evaluator(EndpointFixture(null),task_allowance=Allowance(2,.02))
   c=Controller(**args);out=c.run()
   self.assertEqual(out['status'],'STOPPED');self.assertEqual(out['failure']['reason'],'NO_ELIGIBLE_SUCCESSOR')
   self.assertEqual(c.versions,[dict(mode=null.mode,symbols=null.symbols,language=null.language)])
   self.assertEqual(c.role.fast.steps,0)
 def test_joint_components_and_memory_failure_preserves_phi(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,_=self.setup_run(root,mode='joint');c.step()
   role=c.role;u,v=components(c.language,c.parent)
   self.assertEqual(u.symbols,('c3','c1','c0'));self.assertEqual(v.symbols,('sub','c1','x0'))
   role.proposer.on_forward=Meter(Allowance(10000,10)).forward
   handle,rollout,_=role.draw();old=role.fast.b.detach().clone();memory_before=role.memory.identity
   def failed(*args,**kwargs):raise ValueError('injected memory commit failure')
   role.memory.manage=failed
   with self.assertRaises(ValueError):role.feedback(handle,rollout,loss=1.,cost=0.,tag='completed')
   self.assertFalse(torch.equal(old,role.fast.b));self.assertEqual(memory_before,role.memory.identity)
   self.assertTrue(role.closed);self.assertEqual(role.fast.steps,1)
 def test_quiescent_restore_preserves_state_rng_and_no_replay(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,runner=self.setup_run(root);c.step();c.step()
   old_fast=c.role.fast.fork();old_epoch=c.role.epoch;calls=len(runner.calls);used=c.used
   restored=Controller(**args);restored.restore()
   self.assertEqual(restored.role.attempt,1);self.assertNotEqual(restored.role.epoch,old_epoch)
   for name in ('a','b','momentum_a','momentum_b','rng_state'):
    self.assertTrue(torch.equal(getattr(old_fast,name),getattr(restored.role.fast,name)))
   self.assertEqual(restored.deadline,c.deadline);self.assertEqual(len(runner.calls),calls)
   self.assertGreaterEqual(restored.used,used)
   self.assertEqual(restored.run()['status'],'DONE')
 def test_inflight_recovery_rejected(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,_=self.setup_run(root);c.step();c._inflight('ATTEMPT')
   with self.assertRaises(ValueError):Controller(**args).restore()
 def test_generation_reservation_before_model_calls(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,runner=self.setup_run(root);args['config']=replace(args['config'],total=Allowance(1,600));c=Controller(**args)
   r=c.run();self.assertEqual(r['failure']['reason'],'ResourceExhausted');self.assertEqual(runner.calls,[]);self.assertIsNone(c.role)
 def test_second_generation_uses_admitted_parent_and_fresh_state(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,_=self.setup_run(root,K=2)
   while c.generation==0 and c.phase!='STOPPED':c.step()
   self.assertEqual(c.generation,1);accepted=c.parent
   c.step()
   self.assertEqual(c.role.parent,accepted);self.assertEqual(c.role.fast.steps,0)
   self.assertEqual(c.role.memory.version,0);self.assertEqual(c.role.attempt,0)
   result=c.run();self.assertEqual(result['status'],'STOPPED')
   self.assertEqual(len(c.edges),1);self.assertEqual(c.parent,accepted)
 def test_rtma_nonnull_commit_is_read_by_next_proposal(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,_=self.setup_run(root,mode='rtma');c.step();c.step()
   memory_id=c.role.memory.identity
   self.assertEqual(c.role.memory_commits,1);self.assertEqual(c.role.fast.steps,0)
   c.step();second=c.archive[1]
   self.assertTrue(second['eligible']);self.assertEqual(second['rollout']['memory_read']['memory'],memory_id)
 def test_joint_payload_assembly_failure_occurs_after_phi_commit(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,_=self.setup_run(root,mode='joint');c.step();base=c.role
   memory=EvidenceMemory(base.memory.bounds,reader_identity='fixture-ord32',count_tokens=len,
                         source_prefix='x'*120,namespace='proposal')
   role=ProposalLearner(base.proposer,c.parent,memory,seed=19,opportunity_cap=3,update_bounds=UpdateBounds())
   role.proposer.on_forward=Meter(Allowance(10000,10)).forward
   handle,rollout,_=role.draw();before=role.fast.b.detach().clone()
   with self.assertRaises(ValueError):role.feedback(handle,rollout,loss=1.,cost=0.,tag='completed')
   self.assertEqual(role.fast.steps,1);self.assertFalse(torch.equal(before,role.fast.b))
   self.assertEqual(role.memory.version,0);self.assertTrue(role.closed)
   self.assertEqual(role.receipts[-1]['numerical']['status'],'COMMIT')
 def test_timeout_after_completed_draw_retains_rollout_and_spent_rng(self):
  with tempfile.TemporaryDirectory() as root:
   c,_,_=self.setup_run(root);c.step();draw=c.role.draw
   def expire_after_draw():
    result=draw();c.role.proposer.on_forward.__self__.deadline=0;return result
   with patch.object(c.role,'draw',expire_after_draw):c.step()
   saved=c.journal.load();rollout=saved['failure']['partial_rollout']
   self.assertEqual(saved['phase'],'STOPPED');self.assertTrue(rollout['tokens'])
   self.assertTrue(torch.equal(rollout['rng_after'],saved['role']['fast']['rng_state']))
   self.assertGreater(saved['used'],0)
 def test_last_cut_and_insufficient_bytes_stop_before_setup(self):
  for limits in (dict(max_cuts=1,max_bytes=64*1024*1024),dict(max_cuts=64,max_bytes=32*1024*1024)):
   with tempfile.TemporaryDirectory() as root:
    _,args,_=self.setup_run(root);args['journal']=Journal(root,**limits)
    c=Controller(**args)
    with patch.object(c.factory,'create',side_effect=AssertionError('must not execute')) as create:c.step()
    create.assert_not_called();saved=c.journal.load()
    self.assertEqual(saved['phase'],'STOPPED');self.assertEqual(saved['used'],0)
    self.assertEqual(saved['failure']['reason'],'StorageFull')
 def test_nonempty_journal_without_pointer_cannot_restart(self):
  for name in ('cut-000000','.inflight-aborted'):
   with tempfile.TemporaryDirectory() as root:
    _,args,_=self.setup_run(root);(Path(root)/name).mkdir()
    c=Controller(**args)
    with self.assertRaises(ValueError):c.step()
    with self.assertRaises(JournalIntegrityError):c.restore()
    self.assertFalse((Path(root)/'LATEST').exists())
 def test_recovery_failure_persists_real_cost_and_cannot_repeat(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,_=self.setup_run(root);c.step();c.step();old=c.used
   def fail(parent,generation,seed,on_forward):on_forward(3,1);raise ValueError('charged recovery failure')
   with patch.object(c.factory,'create',side_effect=fail) as factory:
    restored=Controller(**args);restored.restore();saved=c.journal.load()
    self.assertEqual(saved['phase'],'STOPPED');self.assertEqual(saved['used'],old+3)
    self.assertEqual(saved['costs'][-1]['operation'],'recovery')
    with self.assertRaises(ValueError):Controller(**args).restore()
    self.assertEqual(factory.call_count,1)
 def test_interrupted_recovery_marks_nonresumable_before_factory(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,_=self.setup_run(root);c.step()
   def interrupt(*args):raise KeyboardInterrupt('simulated process interruption')
   with patch.object(c.factory,'create',side_effect=interrupt):
    with self.assertRaises(KeyboardInterrupt):Controller(**args).restore()
   self.assertEqual(c.journal.load()['phase'],'INFLIGHT_RECOVERY')
   with self.assertRaises(ValueError):Controller(**args).restore()
 def test_snapshot_envelope_violation_preserves_incomplete_cost_receipt(self):
  with tempfile.TemporaryDirectory() as root:
   c,args,_=self.setup_run(root);c.step();feedback=c.role.feedback
   def oversized(*a,**kw):
    feedback(*a,**kw);c.role.receipts[-1]['overspec']='x'*c.journal.max_cut_bytes
   with patch.object(c.role,'feedback',oversized):c.step()
   saved=c.journal.load();self.assertEqual(saved['schema'],'rsi-incomplete-journal-v1')
   self.assertGreater(saved['used'],0);self.assertFalse(saved['full_snapshot_saved'])
   self.assertEqual(c.run()['status'],'INCOMPLETE')
   with self.assertRaises(ValueError):Controller(**args).restore()
 def test_exact_tensor_header_accounting_not_fixed_overhead(self):
  with tempfile.TemporaryDirectory() as root:
   state={'many':[torch.zeros(1) for _ in range(2500)]};data=serialize(state)
   self.assertGreater(int.from_bytes(data['tensors.safetensors'][:8],'little'),65536)
   exact=sum(map(len,data.values()));j=Journal(root,max_bytes=exact+64,max_cuts=1,max_cut_bytes=exact)
   j.save(state);self.assertEqual(len(j.load()['many']),2500)
   self.assertLessEqual(sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()),j.max_bytes)
 def test_two_initial_controllers_cannot_rewind_each_other(self):
  with tempfile.TemporaryDirectory() as root:
   first,args,runner=self.setup_run(root);stale=Controller(**args);first.step()
   before=(Path(root)/'LATEST').read_bytes();used=first.used
   with self.assertRaises(StaleController):stale.step()
   self.assertEqual((Path(root)/'LATEST').read_bytes(),before)
   self.assertEqual(first.journal.load()['used'],used);self.assertEqual(runner.calls,[])
 def test_pre_restore_controller_cannot_overwrite_new_owner(self):
  with tempfile.TemporaryDirectory() as root:
   old,args,runner=self.setup_run(root);old.step();old.step()
   resumed=Controller(**args);resumed.restore();resumed.step()
   before=(Path(root)/'LATEST').read_bytes();calls=len(runner.calls);used=resumed.used
   with self.assertRaises(StaleController):old.step()
   self.assertEqual((Path(root)/'LATEST').read_bytes(),before)
   self.assertEqual(len(runner.calls),calls);self.assertEqual(resumed.journal.load()['used'],used)
 def test_restore_read_then_writer_advances_is_readonly_rejection(self):
  with tempfile.TemporaryDirectory() as root:
   current,args,runner=self.setup_run(root);current.step();restored=Controller(**args)
   read=current.journal.load_bound;latest=[]
   def interleave():
    result=read();current.step();latest.append((Path(root)/'LATEST').read_bytes());return result
   with patch.object(current.journal,'load_bound',interleave):
    with self.assertRaises(StaleController):restored.restore()
   self.assertEqual((Path(root)/'LATEST').read_bytes(),latest[0])
   self.assertEqual(current.journal.load()['phase'],'READY_ATTEMPT')
   self.assertEqual(current.journal.load()['role']['attempt'],1)
 def test_same_content_new_id_and_shared_audit_group_rejected(self):
  d=case(1);a=case(2,'audit')
  with self.assertRaises(ValueError):validate_partitions((),(),(d,),((replace(a,payload_sha256=d.payload_sha256),),),(19,))
  with self.assertRaises(ValueError):validate_partitions((),(),(d,),((replace(a,physical_group=d.physical_group),),),(19,))
  with self.assertRaises(ValueError):validate_partitions((),(),(d,),((a,),),(d.seed,))
 def test_failures_keep_full_denominator_and_real_cost(self):
  class Broken:
   identity='explicit-failing-fixture'
   def run(self,rule,case,meter):
    meter.charge(1,'step')
    if case.task_id=='dev0':raise ValueError('task-local failure')
    return 0.
  ev=Evaluator(Broken(),task_allowance=Allowance(2,.1));meter=Meter(Allowance(4,2))
  b=ev.block(parse_rule(Language(),'rtma',['c0']),(case(0),case(1)),meter=meter)
  self.assertEqual(b['mean_loss'],.5);self.assertEqual(b['mean_cost'],.5);self.assertEqual(len(b['results']),2)
 def test_audit_group_units_and_no_free_accept_flag(self):
  cases=[dict(task_id=str(i),physical_group='g'+str(i//2),payload_sha256=str(i),seed=i) for i in range(128)]
  def block(loss):return dict(complete=True,cases=cases,results=[dict(group=x['physical_group'],loss=loss) for x in cases],mean_cost=.5)
  r=admission(block(1.),block(0.),alpha=.05,generations=2,delta=.1,cost_cap=1.)
  self.assertTrue(r['accepted']);self.assertEqual(r['independent_groups'],64)
  self.assertFalse(admission(block(.5),block(.5),alpha=.05,generations=2,delta=.1,cost_cap=1.)['accepted'])
  self.assertFalse(admission(block(1.),block(0.),alpha=.05,generations=2,delta=.1,cost_cap=.2)['accepted'])

if __name__=='__main__':unittest.main()
