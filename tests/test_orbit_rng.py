"""Three independent proposal RNGs on one synthetic frozen role bundle."""
from contextlib import ExitStack
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from orbit_runtime_fixture import orbit_fixture,block,strong,Evaluator
from orbit_rsi import run_rsi,restore_rsi
from recursion_core import checked_rng_seeds
from orbit import Orbit

class SameWeightRNGTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_three_rng_runs_close_and_restore_without_three_weights(self):
        model=orbit_fixture();events=[];initial=(id(model.task),id(model.proposal),id(model.original_proposal))
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            for name in ('orbit_rsi.FixedPhiEvaluator','return_recursion.FixedPhiEvaluator','recursion_core.FixedPhiEvaluator'):
                stack.enter_context(patch(name,Evaluator))
            stack.enter_context(patch('orbit_rsi.DemonstrationFactory',return_value=SimpleNamespace(codec=None)))
            stack.enter_context(patch('orbit_rsi.evaluate_public_rule',strong))
            result=run_rsi(model,directory=Path(tmp)/'run',run_identity='1'*64,environment_identity='2'*64,
                dev=block(),open_episode=lambda *a:None,prepare_audit=lambda *a:(_ for _ in ()).throw(AssertionError('tied fixture cannot audit')),
                on_work=lambda e:None,on_event=events.append,resources_complete=lambda:True,rng_seeds=(17,29,43))
            self.assertEqual(result['status'],'COMPLETE_FINITE_ALGORITHM')
            self.assertEqual(result['registry']['acceptances'],0)
            attempts=[e for e in events if e['kind']=='RETURN_DEV_ATTEMPT_COMPLETE']
            self.assertEqual(len(attempts),36)
            self.assertEqual({e['seed'] for e in attempts},{17,29,43})
            self.assertEqual({(e['seed'],e['process']) for e in attempts},
                {(s,p) for s in (17,29,43) for p in ('trained-keep','trained-frozen','original-keep')})
            restored=restore_rsi(model,Path(tmp)/'run',manifest_sha256=result['controller_artifact']['manifest_sha256'])
            self.assertEqual(restored.registry.versions,{'17':0,'29':0,'43':0})
            self.assertEqual(restored.registry.rng_seeds,(17,29,43))
            self.assertEqual(restored.manager,model.initial_manager)
            self.assertEqual(initial,(id(model.task),id(model.proposal),id(model.original_proposal)))
            snapshot_dirs=sorted(p.name for p in (Path(tmp)/'run/processes').iterdir())
            self.assertEqual(snapshot_dirs,['17','29','43'])

    def test_explicit_rng_proposal_save_restore_is_distinct_from_weight_seed(self):
        model=orbit_fixture()
        model.new_proposal_process=lambda **kwargs:Orbit.new_proposal_process(model,**kwargs)
        process=model.new_proposal_process(rng_seed=29)
        self.assertEqual(model.seed,17);self.assertEqual(process.seed,29)
        with tempfile.TemporaryDirectory() as tmp:
            receipt=Orbit.save_proposal(model,process,Path(tmp)/'saved')
            restored=Orbit.restore_proposal(model,Path(tmp)/'saved',manifest_sha256=receipt['manifest_sha256'],rng_seed=29)
            self.assertEqual(restored.seed,29);self.assertEqual(restored.memory.identity,process.memory.identity)
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                Orbit.restore_proposal(model,Path(tmp)/'saved',manifest_sha256=receipt['manifest_sha256'],rng_seed=43)

    def test_rng_schedule_keeps_default_manager_and_rejects_duplicate_runs(self):
        self.assertEqual(checked_rng_seeds((17,29,43)),(17,29,43))
        for values in ((29,43),(17,17),(17,42),(17,True)):
            with self.assertRaises(ValueError):checked_rng_seeds(values)

if __name__=='__main__':unittest.main()
