"""Load the process-credit Orbit roles and create the matching joint site/subtree-edit process.

Use a fresh process and the runtime installed through the mmla-rsi orbit extra.
All model files are local; loading neither downloads assets nor accepts rules.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def verify_package(directory):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / 'MANIFEST.json').read_bytes())
    if manifest['schema'] != 'mmla-orbit-process-credit-package-v1':
        raise ValueError('Orbit process-credit package required')
    names = set()
    for row in manifest['files']:
        name = row['path']; rel = Path(name); path = root / rel
        if (not isinstance(name, str) or rel.is_absolute() or '..' in rel.parts
                or name in names or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(root)):
            raise ValueError('regular confined package member required')
        if path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
            raise ValueError('package member changed: ' + name)
        names.add(name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts
              and '.git' not in p.parts and not p.relative_to(root).as_posix().startswith('.cache/huggingface/')
              and p.relative_to(root).as_posix() not in ('MANIFEST.json', '.gitattributes')}
    if actual != names:
        raise ValueError('unbound or missing package member')
    return root, manifest


def verify_runtime(expected_identity=None):
    """Check the installed runtime once before importing its execution modules."""
    runtime = Path(__file__).resolve().parent
    manifest = json.loads((runtime / 'runtime-manifest.json').read_bytes())
    files = manifest['files']
    identity = hashlib.sha256(json.dumps({r['path']: r['sha256'] for r in files},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (manifest['schema'] != 'mmla-orbit-installed-runtime-v1'
            or manifest['identity'] != identity
            or (expected_identity is not None and expected_identity != identity)):
        raise ValueError('installed Orbit runtime does not match the model configuration')
    supplied = {}
    names = set()
    for row in files:
        rel = Path(row['path']); path = runtime / rel
        if (rel.is_absolute() or '..' in rel.parts or rel.suffix != '.py'
                or row['path'] in names or path.is_symlink() or not path.is_file()
                or not path.resolve(strict=True).is_relative_to(runtime)
                or path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']):
            raise ValueError('installed runtime member changed: ' + row['path'])
        names.add(row['path'])
        name = '.'.join(rel.with_suffix('').parts)
        if name.endswith('.__init__'):
            name = name[:-9]
        supplied[name] = path
    actual = {p.relative_to(runtime).as_posix() for p in runtime.rglob('*.py')}
    if actual != names:
        raise ValueError('unbound or missing installed runtime module')
    return runtime, supplied, identity


class Orbit:
    """Three inference roles plus the exact bounded RTMA proposal interface.

    Role tensors have already crossed restore_role_bundle's validation boundary.
    This constructor binds their tokenizer, language and empty proposal memory.
    Creating or restoring a process neither evaluates nor admits a successor.
    """
    def __init__(self, roles, tokenizer, config, *, package, qualification=None):
        from content_rules import Language, parse_rule
        from content_evidence import MemoryBounds, EvidenceMemory
        from edit_role import EditInterface
        self.task = roles['task']
        self.proposal = roles['proposal']
        self.original_proposal = roles['original_proposal']
        self.seed = roles['seed']
        self.tokenizer = tokenizer
        self.config = config
        self.qualification = qualification
        self.source_identity = config['source']
        spec = config['proposal_interface']
        values = dict(spec['language'])
        values['constants'] = tuple(values['constants'])
        language = Language(**values)
        self.interface = EditInterface(tokenizer, tokenizer_identity=config['tokenizer_identity'],
            language=language, vocabulary_size=self.task.mixer.get_input_embeddings().weight.shape[0])
        self.memory0 = EvidenceMemory(MemoryBounds(**spec['memory_bounds']),
            reader_identity=config['tokenizer_identity'],
            count_tokens=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
            namespace=spec['namespace'], source_prefix=spec['source_prefix'])
        self.initial_manager = parse_rule(language, 'rtma', tuple(config['initial_manager']))
        self.metadata = dict(roles=roles['metadata'], package=package,
            source=config['source'], deployment_Phi='fixed', RSI_qualification_inherited=False)

    def new_task_state(self, *, seed, tenant, policy='fixed_rho', rule=None):
        """Create a private observation-task state at the saved trained rho.

        fixed_rho executes the RTMA-only task path. adaptive enables numerical
        feedback as well as the bound memory manager, using the supplied rule
        or the matched default rules. This does not qualify a predictive head
        or transfer RTMA-only experimental results to the adaptive path.
        """
        from demonstration_factory import DemonstrationFactory
        from rule_binding import ManagedCandidateFactory
        baseline = DemonstrationFactory(self.task, self)
        factory = ManagedCandidateFactory(baseline,
            self.initial_manager if rule is None else rule, task_policy=policy)
        return factory.new_state(seed=seed, tenant=tenant)

    def new_engineering_memory(self):
        """Empty typed M with the original independent frozen qualification reader.

        Task rho/hybrid are not the qualification decoder. Their separate saved
        tensors are restored without training, then bound to this actual profile
        and device. No old cached qualification or predictive status is inherited.
        """
        import copy
        from types import SimpleNamespace
        from mmla_pdsa.full.encoding import FrozenCanonicalEncoder
        from mmla_pdsa.full.qualification import FrozenSemanticGate
        from mmla_pdsa.full.production_memory import production_memory
        from mmla_pdsa.full.typed_reads import EngineeringViewReader
        if self.qualification is None:
            raise ValueError('package lacks its independent frozen qualification role')
        saved = self.qualification  # Validated once by load_orbit.
        encoder = FrozenCanonicalEncoder(self.task, self.tokenizer,
            tokenizer_identity=self.config['tokenizer_identity'])
        frozen = SimpleNamespace(mixer=self.task.mixer, profile=self.task.profile,
            substrate_identity=self.task.substrate_identity, hybrid_config=self.task.hybrid_config)
        for name in ('rho', 'readout', 'hybrid'):
            module = copy.deepcopy(getattr(self.task, name))
            module.load_state_dict(saved[name]); module.eval().requires_grad_(False)
            setattr(frozen, name, module)
        gate = FrozenSemanticGate(frozen, encoder, self.tokenizer,
            tokenizer_identity=self.config['tokenizer_identity'],
            eos_ids=tuple(saved['definition']['eos_ids']),
            max_new_tokens=saved['definition']['max_new_tokens'], cache_capacity=saved['cache_capacity'])
        changes = {key for key in saved['definition'] if saved['definition'][key] != gate.definition[key]}
        if not changes <= {'profile', 'encoder'} or gate.reader.decoder_digest != saved['definition']['decoder']:
            raise ValueError('only explicit context/device encoder rebinding is supported')
        memory = production_memory(self.task.profile, encoder, gate)
        return memory, EngineeringViewReader(memory.encoder, gate), dict(
            original_gate=saved['identity'], current_gate=gate.identity,
            original_profile=saved['definition']['profile'], current_profile=self.task.profile.identity,
            original_decoder=saved['definition']['decoder'], changed_fields=sorted(changes),
            qualification_cache_inherited=False, predictive_qualification_inherited=False)

    def new_problem(self, memory, *, problem_id, tenant, principal, seed,
                    view_reader=None):
        """Bind this model to a caller's authoritative memory and source scope.

        Use FrameDriver, StreamDriver or JointStreamDriver for the source
        transport. Their boundary, permission and selector contracts apply;
        an old J calibration is not silently transferred to a new reader.
        """
        from mmla_pdsa.full.runtime import Problem
        return Problem(self.task, memory, problem_id=problem_id, tenant=tenant,
            principal=principal, seed=seed, view_reader=view_reader)

    def run_rsi(self, **execution):
        """Run the finite RTMA controller with a bound task environment/accounting.

        See the MMLA-RSI docs/orbit/RSI_RUNTIME.md guide. This evaluates real complete tasks through the
        supplied environment, preserves all controls, and never trains weights.
        """
        from orbit_rsi import run_rsi
        return run_rsi(self, **execution)

    def restore_rsi(self, directory, *, manifest_sha256):
        """Restore a closed controller's installed manager and complete record.

        This does not resume an interrupted scientific run or open a new audit.
        Quiet proposal/RNG recovery is available separately via restore_proposal.
        """
        from orbit_rsi import restore_rsi
        return restore_rsi(self, directory, manifest_sha256=manifest_sha256)

    def new_proposal_process(self, *, process='trained-keep', generation=0,
                             parent_symbols=None, on_work=None, on_event=None, rng_seed=None):
        """Create one of the three matched four-attempt proposal processes.

        The supplied manager actually controls retained evaluation records.
        parent_symbols may name a successor, but this call does not certify it.
        Independent evidence and rule succession remain the evaluator's job.
        """
        from content_rules import parse_rule
        from content_evidence import EvidenceMemory
        from edit_role import EditProcess
        if process not in ('trained-keep', 'trained-frozen', 'original-keep'):
            raise ValueError('explicit trained-keep, trained-frozen or original-keep required')
        parent = self.initial_manager if parent_symbols is None else parse_rule(
            self.interface.language, 'rtma', tuple(parent_symbols))
        memory = EvidenceMemory.restore(self.memory0.export(), count_tokens=self.memory0._count_tokens,
            expected_contract=self.memory0.contract)
        return EditProcess(self.original_proposal if process=='original-keep' else self.proposal,
            self.interface, parent, memory, control='frozen' if process=='trained-frozen' else 'keep',
            seed=self.seed if rng_seed is None else rng_seed, phase='dev', round_index=generation,
            on_work=on_work if on_work is not None else lambda value: None,
            on_event=on_event if on_event is not None else lambda value: None,
            functional_probes=process!='trained-frozen')

    def save_proposal(self, process, directory):
        """Save a quiet process owned by this loaded model, without model calls."""
        from edit_checkpoint import save_edit_checkpoint
        if process.model is not self.proposal and process.model is not self.original_proposal:
            raise ValueError('process belongs to another loaded model')
        if process.interface is not self.interface:
            raise ValueError('process belongs to another proposal interface')
        return save_edit_checkpoint(process, directory, source_identity=self.source_identity)

    def restore_proposal(self, directory, *, manifest_sha256, process='trained-keep',
                         generation=0, parent_symbols=None, on_work=None, on_event=None, rng_seed=None):
        """Restore between feedback and next proposal into a fresh owned process."""
        from edit_checkpoint import restore_edit_checkpoint
        fresh = self.new_proposal_process(process=process, generation=generation,
            parent_symbols=parent_symbols, on_work=on_work, on_event=on_event, rng_seed=rng_seed)
        return restore_edit_checkpoint(fresh, directory, source_identity=self.source_identity,
            manifest_sha256=manifest_sha256)


def load_orbit(directory, *, seed=None, device='cpu'):
    """Load selected task942 / trained proposal184 / original156 roles.

    Single-model packages default to their declared selected_seed. Historical
    multi-reader packages still require an explicit seed.

    Use a fresh process, local weights and the matching installed runtime. Loading preserves
    global torch RNG, downloads nothing, and makes no forward or update call.
    These roles belong to the fixed-Phi RTMA process-credit candidate; the original
    six-model release is a different artifact and keeps its own loader.
    """
    root, manifest = verify_package(directory)
    config = json.loads((root / 'orbit_config.json').read_bytes())
    selected = config.get('selected_seed')
    if selected is not None and (type(selected) is not int or selected not in (17, 29, 43)
            or set(config['roles']) != {str(selected)}):
        raise ValueError('single-model package must contain exactly its declared selected seed')
    if seed is None:
        seed = selected
    if type(seed) is not int or seed not in (17, 29, 43) or str(seed) not in config['roles']:
        raise ValueError('explicit seed must identify a supplied role bundle')
    runtime, supplied, _ = verify_runtime(config['public_runtime_identity'])
    for name, module in tuple(sys.modules.items()):
        if name in supplied:
            path = getattr(module, '__file__', None)
            if path is None or Path(path).resolve() != supplied[name]:
                raise ValueError('use the supplied runtime in a fresh process: ' + name)
    sys.path.insert(0, str(runtime))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from mmla_pdsa.full.profile import ArchitectureProfile
    from mmla_pdsa.full.substrate_helpers import lora_linear, tensors_hash
    from mmla_pdsa.full.extensions.hybrid import HybridConfig
    from mmla_pdsa.full.extensions.successor_model import SuccessorMMLAModel
    from mmla_pdsa.full.extensions.rsi_binding import rebind
    from mmla_pdsa.full.outcome_dense_arithmetic import bind
    from orbit_process_bundle import restore_role_bundle, COLLECTION
    if (config['schema'] != 'mmla-orbit-process-credit-config-v1'
            or config['collection'] != COLLECTION or config['deployment_Phi'] != 'fixed'
            or config['RSI_qualification_inherited'] is not False
            or (selected is None and set(config['roles']) != {'17', '29', '43'})):
        raise ValueError('explicit fixed-Phi process-credit package configuration required')
    specification = config['roles'][str(seed)]
    target = torch.device(device)
    if target.type not in ('cpu', 'cuda'):
        raise ValueError('CPU or CUDA device required')
    devices = list(range(torch.cuda.device_count())) if target.type == 'cuda' or torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        tokenizer = AutoTokenizer.from_pretrained(root / 'base', local_files_only=True, trust_remote_code=False)
        mixer = AutoModelForCausalLM.from_pretrained(root / 'base', local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager')
        mixer = mixer.to(target).eval().requires_grad_(False)
        if tensors_hash(torch, dict(mixer.named_parameters())) != config['substrate']['theta0']:
            raise ValueError('original theta0 identity changed')
        mixer.config._name_or_path = config['original_mixer_name_or_path']
        omega = load_file(str(root / 'omega.safetensors'), device='cpu')
        if tensors_hash(torch, omega) != config['substrate']['omega']:
            raise ValueError('always-on omega identity changed')
        recipe = config['omega_recipe']; actual = {}
        rng = torch.Generator(device=target).manual_seed(recipe['omega_seed'])
        for index in recipe['layers']:
            attention = mixer.model.layers[index].self_attn
            for name in recipe['targets']:
                layer = lora_linear(torch, getattr(attention, name), recipe['rank'], recipe['alpha'], rng)
                setattr(attention, name, layer)
                for leaf in ('down', 'up'):
                    key = f'layer.{index}.{name}.{leaf}'; value = omega[key]; dest = getattr(layer, leaf)
                    if value.shape != dest.shape or value.dtype != dest.dtype or not torch.isfinite(value).all():
                        raise ValueError('omega tensor layout changed')
                    with torch.no_grad():
                        dest.copy_(value.to(dest.device))
                    actual[key] = dest
        mixer.requires_grad_(False)
        if set(actual) != set(omega) or tensors_hash(torch, actual) != config['substrate']['omega']:
            raise ValueError('omega active attachment changed')
        original_profile = ArchitectureProfile(**config['original_profile'])
        if original_profile.identity != config['original_profile_identity']:
            raise ValueError('original architecture profile changed')
        prototype = SuccessorMMLAModel(mixer, original_profile, seed=config['construction_seed'],
            substrate_identity={k: v for k, v in config['substrate'].items() if k != 'mmla_hybrid'},
            hybrid_config=HybridConfig(**config['hybrid'])).eval().requires_grad_(False)
        prototype.profile = replace(original_profile, local_tokens=4096)
        rebind(prototype)
        # Validate the original arithmetic; the loader does not silently enable
        # reduced precision or select a replacement implementation.
        bind(prototype)
        roles = restore_role_bundle(prototype, root / 'roles' / str(seed),
            source_identity=config['source'], metadata_sha256=specification['metadata_sha256'],
            original_training_identity=specification['original_training_identity'])
        if roles['seed'] != seed:
            raise ValueError('restored role seed differs from the explicit selection')
        qualification = None
        if 'qualification' in config:
            from mmla_pdsa.full.safe_state import load_state
            from mmla_pdsa.full.profile import digest
            from mmla_pdsa.full.neural import tensor_digest
            qualification = load_state(root / 'qualification')
            reference = config['qualification']
            if (set(qualification) != {'schema','definition','identity','cache_capacity','rho','readout','hybrid'}
                    or qualification['schema'] != 2
                    or digest(qualification['definition']) != qualification['identity']
                    or qualification['identity'] != reference['original_gate_identity']
                    or qualification['definition']['profile'] != config['original_profile_identity']
                    or qualification['definition']['substrate'] != roles['task'].substrate_identity):
                raise ValueError('exact original independent qualification definition required')
            tensors = {}
            for name in ('rho','readout','hybrid'):
                expected = getattr(roles['task'], name).state_dict(); saved = qualification[name]
                if set(saved) != set(expected) or any(v.shape != expected[k].shape or v.dtype != expected[k].dtype
                        or not torch.isfinite(v).all() for k,v in saved.items()):
                    raise ValueError('complete original qualification tensor layout required')
                tensors.update({name+'.'+k:v for k,v in saved.items()})
            if tensor_digest(tensors) != qualification['definition']['decoder']:
                raise ValueError('original qualification decoder identity changed')
        orbit=Orbit(roles, tokenizer, config, package=manifest, qualification=qualification)
        if 'reader_calibration' in config:
            from readout_calibration import apply_calibration
            ref=config['reader_calibration']
            if set(ref)!={'stem','training_source','course','steps'} or ref['stem']!='calibration/readout' or ref['steps']!=247:
                raise ValueError('explicit complete readout calibration descriptor required')
            applied=apply_calibration(orbit,root/ref['stem'],contract=ref['training_source'])
            if applied['course']!=ref['course']:
                raise ValueError('calibration course identity changed')
        return orbit
