import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mmla_orbit import verify_package, verify_runtime


class OrbitPackageTests(unittest.TestCase):
    def test_public_import_is_lazy(self):
        output = subprocess.check_output([sys.executable, '-c',
            "import sys; import mmla_orbit; assert 'torch' not in sys.modules; "
            "assert 'content_rules' not in sys.modules; print('lazy')"], text=True)
        self.assertEqual(output.strip(), 'lazy')

    def test_installed_inventory_and_mismatch(self):
        root, supplied, identity = verify_runtime()
        self.assertIn('orbit_rsi', supplied)
        self.assertIn('mmla_pdsa.full.extensions.joint', supplied)
        self.assertTrue(all(p.is_relative_to(root) for p in supplied.values()))
        self.assertEqual(verify_runtime(identity)[2], identity)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            verify_runtime('0' * 64)

    def test_weights_directory_needs_no_python_and_rejects_extra(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'{"test_only":true}\n'
            (root / 'orbit_config.json').write_bytes(payload)
            manifest = {'schema': 'mmla-orbit-process-credit-package-v1', 'files': [
                {'path':'orbit_config.json','bytes':len(payload),
                 'sha256':hashlib.sha256(payload).hexdigest()}]}
            (root/'MANIFEST.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_package(root)[0], root)
            (root/'unbound.py').write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'unbound'):
                verify_package(root)
            (root/'unbound.py').unlink()
            (root/'orbit_config.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'member changed'):
                verify_package(root)


if __name__ == '__main__':
    unittest.main()
