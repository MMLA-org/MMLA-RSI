"""Handwritten metadata fixtures: no models, environments, or scientific results."""
import copy, importlib.util, sys, tempfile, threading, unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import torch
from mmla_rsi.rules import Language, parse_rule, identity
from mmla_rsi.evaluation import admission, Case
from mmla_rsi.journal import Journal, JournalIntegrityError, StaleController

from mmla_rsi import parent_registry as mod

def H(value):return identity({'fixture':value})


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.lang=Language();self.setup('rtt')

    def setup(self,mode,generations=2):
        self.mode=mode
        symbols={'rtt':['c2','c1','c4'],'rtma':['sub','c1','x0'],
                 'joint':['c2','c1','c4','sub','c1','x0']}[mode]
        other={'rtt':['c3','c1','c4'],'rtma':['sub','c1','x1'],
               'joint':['c3','c1','c4','sub','c1','x1']}[mode]
        self.parent=parse_rule(self.lang,mode,symbols);self.child=parse_rule(self.lang,mode,other)
        self.binding=mod.Binding(H('controller-'+mode),H('roles'),H('session'),mode,generations,.05,.02,1.)
        self.rj=Journal(self.root/('registry-'+mode),max_bytes=64*1024*1024,max_cuts=64,max_cut_bytes=1024*1024)
        self.cj=Journal(self.root/('controller-'+mode),max_bytes=64*1024*1024,max_cuts=64,max_cut_bytes=1024*1024)
        self.registry=mod.ParentRegistry(self.rj,binding=self.binding,language=self.lang,initial=self.parent,max_live_requests=3)
        self.registry.initialize(self.package(self.parent))

    def package(self,rule,j='fixed',protocol=None):
        return mod.RuntimePackage(rule.identity,self.binding.role_pair,self.binding.session_adapter,H('runtime-'+rule.identity),j,protocol)

    def accepted(self,n=64,phase='READY_GENERATION'):
        blocks=[]
        for loss,rule in ((1.,self.parent),(0.,self.child)):
            cases=[Case(f'fixture-{i}',f'fixture-{i}',H(f'payload-{i}'),i) for i in range(n)]
            results=[dict(task=c.task_id,group=c.physical_group,payload=c.payload_sha256,tape=c.seed,
                rule=rule.identity,loss=loss,cost=.1,error=None,resources=dict(work=1,wall_seconds=0.,
                events=[('fixture_metadata',1)],allowance=dict(work=10,seconds=1.)),trace=None) for c in cases]
            blocks.append(dict(complete=True,rule=rule.identity,cases=[asdict(c) for c in cases],results=results,
                mean_cost=sum(x['cost'] for x in results)/n,mean_loss=loss))
        decision=admission(*blocks,alpha=.05,generations=self.binding.generations,delta=.02,cost_cap=1.)
        attempt=dict(attempt=2,eligible=True,candidate=asdict(self.child),loss=.1,cost=.1,reward=-.1)
        selection=[{k:attempt[k] for k in ('attempt','eligible','candidate','loss','cost','reward')}]
        lock=dict(candidate=asdict(self.child),attempt=2,generation=0,parent=self.parent.identity,
            contract=self.binding.controller,selection=selection);lock['identity']=identity(lock)
        audit=dict(lock=lock['identity'],parent=blocks[0],child=blocks[1],decision=decision)
        edge=dict(version=1,parent=self.parent.identity,child=self.child.identity,lock=lock['identity'],
            decision=decision,functional_participation='NOT_MEASURED')
        return dict(schema='rsi-controller-v1',contract=self.binding.controller,parent=asdict(self.child),generation=1,
            phase=phase,resumable=True,persistence_complete=True,failure=None,role=None,archive=[],locked=None,audit=None,
            versions=[asdict(self.parent),asdict(self.child)],edges=[edge],
            archives=[dict(generation=0,parent=asdict(self.parent),attempts=[attempt],locked=lock,audit=audit)])

    def save_accept(self,**kwargs):
        s=self.accepted(**kwargs);self.cj.save(s);return s

    def test_all_three_modes_pin_old_requests_and_publish_only_new_ones(self):
        for mode in ('rtt','rtma','joint'):
            if mode!='rtt':self.setup(mode)
            old=self.registry.acquire();self.save_accept()
            self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
            new=self.registry.acquire()
            self.assertEqual(old,self.registry.resolve(old));self.assertEqual(old['rule'],asdict(self.parent))
            self.assertEqual(new['rule'],asdict(self.child));self.assertEqual((old['version'],new['version']),(0,1))
            self.assertEqual(self.registry.snapshot()['evidence']['functional_participation'],'NOT_MEASURED')

    def test_audited_inflight_failed_and_incomplete_cuts_cannot_publish(self):
        for phase in ('AUDITED','INFLIGHT_AUDIT','STOPPED'):
            self.save_accept(phase=phase)
            with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        s=self.accepted();s['persistence_complete']=False;self.cj.save(s)
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        self.assertEqual(self.registry.snapshot()['version'],0)

    def test_eight_groups_cannot_bypass_original_admission_with_a_true_flag(self):
        s=self.accepted(n=8)
        self.assertFalse(s['edges'][0]['decision']['accepted'])
        s['edges'][0]['decision']['accepted']=True;s['archives'][0]['audit']['decision']['accepted']=True
        self.cj.save(s)
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))

    def test_corrupt_parent_lock_and_candidate_selection_are_rejected(self):
        for change in ('parent','lock','selection'):
            s=self.accepted()
            if change=='parent':s['edges'][0]['parent']=H('wrong-parent')
            elif change=='lock':s['archives'][0]['locked']['attempt']=3
            else:s['archives'][0]['attempts'][0]['eligible']=False
            self.cj.save(s)
            with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))

    def test_other_rule_audit_and_mismatched_case_results_are_rejected(self):
        for change in ('block_rule','row_rule','tape','group','payload','count','mean'):
            s=self.accepted();block=s['archives'][0]['audit']['child']
            if change=='block_rule':block['rule']=H('unrelated-rule')
            elif change=='row_rule':block['results'][0]['rule']=H('unrelated-rule')
            elif change=='tape':block['results'][0]['tape']=100000
            elif change=='group':block['results'][0]['group']='made-up-group'
            elif change=='payload':block['results'][0]['payload']=H('unrelated-payload')
            elif change=='count':block['results'].pop()
            else:block['mean_cost']=0.
            self.cj.save(s)
            with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        self.assertEqual(self.registry.snapshot()['version'],0)

    def test_self_consistent_lock_cannot_substitute_development_history(self):
        s=self.accepted();lock=s['archives'][0]['locked']
        lock['selection'][0]['cost']=.9
        lock['identity']=identity({k:v for k,v in lock.items() if k!='identity'})
        s['archives'][0]['audit']['lock']=lock['identity'];s['edges'][0]['lock']=lock['identity']
        self.cj.save(s)
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))

    def test_wrong_runtime_or_old_j_protocol_cannot_be_requalified_by_publication(self):
        self.save_accept()
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.parent))
        runtime=asdict(self.package(self.child));runtime['role_pair']=H('wrong-roles')
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=mod.RuntimePackage(**runtime))
        cut,s=self.rj.load_bound();s['active']['runtime']=asdict(self.package(self.parent,'learned',H('old-protocol')))
        with self.rj.transaction(1,expected_cut=cut):self.rj.save(s)
        with self.assertRaises(ValueError):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child,'learned',H('old-protocol')))
        self.registry.publish(self.cj,expected_version=0,package=self.package(self.child,'learned',H('new-protocol')))

    def test_reopen_keeps_outstanding_leases_without_resuming_tasks(self):
        old=self.registry.acquire()
        reopened=mod.ParentRegistry(self.rj,binding=self.binding,language=self.lang,initial=self.parent,max_live_requests=3)
        self.assertEqual(reopened.resolve(old),old)
        reopened.close(old,outcome_receipt=H('explicit-reconciliation'))
        with self.assertRaises(ValueError):reopened.resolve(old)
        with self.assertRaises(ValueError):reopened.close(old,outcome_receipt=H('duplicate'))
        self.assertEqual(reopened.acquire()['sequence'],1)

    def test_lease_capacity_mutation_and_stale_publication(self):
        a=self.registry.acquire();self.registry.acquire();self.registry.acquire()
        with self.assertRaises(ValueError):self.registry.acquire()
        altered=copy.deepcopy(a);altered['version']=1
        with self.assertRaises(ValueError):self.registry.resolve(altered)
        self.save_accept();self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        with self.assertRaises(StaleController):self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))

    def test_competing_request_allocations_do_not_lose_or_duplicate_leases(self):
        otherj=Journal(self.rj.root,max_bytes=self.rj.max_bytes,max_cuts=self.rj.max_cuts,max_cut_bytes=self.rj.max_cut_bytes)
        other=mod.ParentRegistry(otherj,binding=self.binding,language=self.lang,initial=self.parent,max_live_requests=3)
        barrier=threading.Barrier(2);out=[]
        original_a=self.registry._load;original_b=other._load
        def synchronized(original):
            result=original();barrier.wait(timeout=5);return result
        def work(reg):
            try:out.append(('ok',reg.acquire()))
            except (StaleController,BlockingIOError) as e:out.append(('rejected',type(e).__name__))
        with patch.object(self.registry,'_load',side_effect=lambda:synchronized(original_a)),patch.object(other,'_load',side_effect=lambda:synchronized(original_b)):
            threads=[threading.Thread(target=work,args=(x,)) for x in (self.registry,other)]
            for t in threads:t.start()
            for t in threads:t.join(timeout=8);self.assertFalse(t.is_alive())
        self.assertEqual(sorted(x[0] for x in out),['ok','rejected'])
        self.assertEqual(self.registry.snapshot()['next_request'],1)
        self.assertEqual(len(self.registry.snapshot()['leases']),1)

    def test_interrupted_publication_never_falls_back_to_older_cut(self):
        (self.rj.root/'.LATEST.tmp').write_text('interrupted')
        with self.assertRaises(JournalIntegrityError):self.registry.snapshot()

    def test_controller_cut_cannot_advance_during_registry_publication(self):
        self.save_accept();original_save=self.rj.save
        other=Journal(self.cj.root,max_bytes=self.cj.max_bytes,max_cuts=self.cj.max_cuts,max_cut_bytes=self.cj.max_cut_bytes)
        def while_publishing(state):
            with self.assertRaises(BlockingIOError):
                with other.transaction(1):self.fail('controller writer entered publication fence')
            return original_save(state)
        with patch.object(self.rj,'save',side_effect=while_publishing):
            self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        with other.transaction(1):pass

    def test_done_archive_publishes_without_resuming_controller(self):
        self.setup('rtma',generations=1);self.save_accept(phase='DONE')
        self.registry.publish(self.cj,expected_version=0,package=self.package(self.child))
        self.assertEqual(self.registry.snapshot()['version'],1)
        self.assertEqual(self.cj.load()['phase'],'DONE')

    def test_tests_never_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
