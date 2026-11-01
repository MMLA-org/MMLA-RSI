import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from mmla_orbit import verify_package, verify_runtime
from mmla_orbit.consolidate import consolidate


class ConsolidationTests(unittest.TestCase):
    def test_lossless_namespaces_and_rejected_extra_tensor(self):
        runtime, _, runtime_id = verify_runtime()
        sys.path.insert(0, str(runtime))
        from mmla_pdsa.full.safe_state import save_state, exact_equal
        from consolidated_storage import load_consolidated
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            (source / 'base').mkdir(parents=True)
            base = {'lm_head.weight': torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)}
            omega = {'layer.16.q_proj.down': torch.arange(4, dtype=torch.float32)}
            save_file(base, str(source / 'base/weights.safetensors'))
            save_file(omega, str(source / 'omega.safetensors'))
            (source / 'base/model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {k:'weights.safetensors' for k in base}}))
            (source / 'base/config.json').write_text('{"model_type":"llama"}')
            (source / 'base/tokenizer_config.json').write_text('{}')
            (source / 'base/UPSTREAM_README.md').write_text('retained attribution')
            states = {}
            for index, stem in enumerate(('roles/17/task', 'roles/17/proposal',
                    'roles/17/original_proposal', 'qualification', 'calibration/readout')):
                state = {'weight':torch.full((2, 4), index, dtype=torch.float32),
                         'version':index, 'tuple':(True, b'abc')}
                save_state(source / stem, state)
                states[stem.replace('/', '.')] = state
            (source / 'roles/17/ROLES.json').write_text('{"role_fixture":true}')
            config = dict(schema='mmla-orbit-process-credit-config-v1', roles={'17':{}},
                public_runtime_identity=runtime_id, qualification={},
                runtime_package={'name':'mmla-rsi', 'version':'0.2.2', 'revision':'0' * 40},
                reader_calibration={'stem':'calibration/readout'})
            (source / 'orbit_config.json').write_text(json.dumps(config))
            (source / 'deployment.json').write_text(json.dumps({'model':{'runtime':runtime_id}}))
            files = [dict(path=p.relative_to(source).as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(source.rglob('*')) if p.is_file()]
            (source / 'MANIFEST.json').write_text(json.dumps(
                {'schema':'mmla-orbit-process-credit-package-v1', 'files':files}))
            destination = Path(directory) / 'merged'
            consolidate(source, destination, runtime_revision='1' * 40)
            verify_package(destination)
            self.assertEqual([p.name for p in destination.rglob('*.safetensors')], ['model.safetensors'])
            self.assertFalse((destination / 'base').exists())
            self.assertTrue((destination / 'tokenizer_config.json').is_file())
            self.assertTrue((destination / 'licenses/UPSTREAM_README.md').is_file())
            configuration = json.loads((destination / 'config.json').read_text())
            self.assertEqual(configuration['runtime_package']['revision'], '1' * 40)
            self.assertEqual(configuration['runtime_package']['version'], '0.2.3')
            actual_base, actual_omega, actual_states = load_consolidated(destination, configuration)
            self.assertTrue(exact_equal(base, actual_base))
            self.assertTrue(exact_equal(omega, actual_omega))
            self.assertTrue(exact_equal(states, actual_states))
            self.assertFalse(torch.equal(actual_states['roles.17.task']['weight'],
                                         actual_states['qualification']['weight']))
            with self.assertRaises(FileExistsError):
                consolidate(source, destination, runtime_revision='1' * 40)
            tensors = load_file(str(destination / 'model.safetensors'))
            tensors['unused'] = torch.ones(1)
            save_file(tensors, str(destination / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'inventory'):
                load_consolidated(destination, configuration)


if __name__ == '__main__':
    unittest.main()
