"""The shipped manager is portable without copying private controller traces."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,tempfile,unittest
from mmla_orbit import load_manager
from orbit_runtime_fixture import orbit_fixture,parse_rule
from orbit_rsi import model_binding


class PackagedManagerTests(unittest.TestCase):
    def write(self,root,orbit,rule,mutate=None):
        value=dict(schema='orbit-packaged-manager-v1',model=model_binding(orbit),
            manager=asdict(rule),default_manager_rng=17,
            evidence=dict(result_sha256='a'*64,terminal_sha256='b'*64))
        if mutate is not None:mutate(value)
        raw=(json.dumps(value,sort_keys=True)+'\n').encode();path=root/'deployment.json';path.write_bytes(raw)
        orbit.metadata=dict(package=dict(files=[dict(path='deployment.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())]))
        return path

    def test_returned_initial_and_successor_leave_starting_rule_unchanged(self):
        model=orbit_fixture();original=model.initial_manager
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rule in (original,parse_rule(model.interface.language,'rtma',('c0',))):
                path=self.write(root,model,rule)
                self.assertEqual(load_manager(model,path),rule)
                self.assertEqual(model.initial_manager,original)
        self.assertEqual(int(model.task.rho.version),942)
        self.assertEqual(int(model.proposal.rho.version),184)

    def test_same_package_cannot_assign_manager_to_changed_reader(self):
        model=orbit_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write(Path(tmp),model,model.initial_manager)
            model.source_identity='0'*64
            with self.assertRaisesRegex(ValueError,'different Orbit'):
                load_manager(model,path)

    def test_modified_program_and_foreign_language_are_rejected(self):
        model=orbit_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.write(root,model,model.initial_manager)
            path.write_bytes(path.read_bytes().replace(b'"default_manager_rng": 17',b'"default_manager_rng": 29'))
            with self.assertRaisesRegex(ValueError,'record changed'):
                load_manager(model,path)
            path=self.write(root,model,model.initial_manager,
                mutate=lambda value:value['manager'].update(language='0'*64))
            with self.assertRaisesRegex(ValueError,'language'):
                load_manager(model,path)

    def test_unbound_or_nondefault_rng_is_not_a_release_selection(self):
        model=orbit_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.write(root,model,model.initial_manager,
                mutate=lambda value:value.update(default_manager_rng=29))
            with self.assertRaisesRegex(ValueError,'different Orbit'):
                load_manager(model,path)
            model.metadata['package']['files']=[]
            with self.assertRaisesRegex(ValueError,'must bind'):
                load_manager(model,path)
