"""One fixed-rho four-attempt edit process with authoritative proposal memory.

The FIT owner and deployment controller use this same input/distribution. Only
current M enters the prompt; the receipt archive is output, never model input.
A process cannot survive a slow-rho update. Start the next batch/generation with
fresh M and independent, externally paired attempt RNGs.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import torch
from mmla_pdsa.full.neural import tensor_digest
from content_binding import real_token_codec
from content_evidence import EvidenceMemory, Record
from content_rules import PrefixGrammar, identity
from proposal_edit import EditPlan, SampledEdit, sample_edit, program_first_distribution


def model_guard(model):
    return tuple((name, id(value), value._version, value.data_ptr()) for name, value in
                 tuple(model.named_parameters()) + tuple(model.named_buffers()))


class EditInterface:
    def __init__(self, tokenizer, *, tokenizer_identity, language, vocabulary_size):
        self.tokenizer, self.tokenizer_identity, self.language = tokenizer, tokenizer_identity, language
        self.codec = real_token_codec(tokenizer, language, 'rtma', vocabulary_size)
        self.mapping = dict(self.codec.pairs)
        # Site tokens use a separate prompt/meaning, with one distinct roundtrip
        # token per possible node. Sharing literal tokens with the DSL is safe:
        # the joint action records which stage and mask owned every token.
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcde'
        if language.max_nodes != 31:
            raise ValueError('the fixed 31-node site codebook requires the bound language')
        site_tokens = []
        for character in alphabet:
            ids = tokenizer.encode(character, add_special_tokens=False)
            if (len(ids) != 1 or not 0 <= ids[0] < vocabulary_size
                    or tokenizer.decode(ids, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != character):
                raise ValueError('site code must be one roundtrip vocabulary token')
            site_tokens.append(ids[0])
        if len(set(site_tokens)) != 31:
            raise ValueError('distinct site tokens required')
        self.site_tokens = tuple(site_tokens)
        self.site_codebook = dict(enumerate(alphabet))
        self.identity = identity(dict(schema='orbit-joint-site-subtree-interface-v1',
            tokenizer=tokenizer_identity, language=language.identity, codec=self.codec.identity,
            temperature=1., site_tokens=self.site_tokens, learned_edit_sites=True,
            prompt_schema='task-success-and-retained-loss-v1'))
        alphabet = {s: tokenizer.decode([i], skip_special_tokens=False) for s, i in self.codec.pairs}
        self.instructions = (
            'Improve a bounded RTMA memory-management program to increase complete-task success. '
            'First choose a zero-based prefix node, including the root, then emit its replacement subtree. '
            'The rest of the parent program is unchanged. Binary operators: add, sub, mul, min, max, sdiv; '
            'sdiv uses the fixed language epsilon. V scores NULL or complete-row replacement; highest score wins, '
            'NULL first then ascending slot on ties. '
            'Features: x0=is_NULL, x1=slot_occupied, x2=new_loss, x3=new_cost, x4=old_loss, '
            'x5=old_cost, x6=min(1,age/opportunity_cap), x7=occupied_fraction, '
            'x8=newly retained coverage, x9=lost retained coverage. '
            'Old values and age are zero for empty slots; coverage marginals for NULL are zero. '
            'Only the current retained evaluation records below are available. '
            'Each record is a complete proposed program with its task failure rate and cost. '
            'Lower recorded loss means higher complete-task success; minimize failure rate. '
            'Evaluation records and task observation capsules are distinct memory payloads. '
            'Constants=' + json.dumps([language.constant(i) for i in range(len(language.constants))]) +
            '\nLanguage=' + json.dumps(asdict(language), sort_keys=True, separators=(',', ':')) +
            '\nCodebook=' + json.dumps(alphabet, sort_keys=True, separators=(',', ':')))

    def site_prompt(self, plan, memory_text):
        text = (self.instructions + '\nParent=' + ' '.join(plan.parent.symbols) +
            '\nSite codebook=' + json.dumps({i:self.site_codebook[i] for i in range(len(plan.sites))},
                sort_keys=True, separators=(',', ':')) +
            '\nCurrent memory=' + memory_text + '\nEdit site (one code token):')
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt(self, plan, site_index, memory_text):
        site = plan.sites[site_index]
        text = (self.instructions + '\nParent=' + ' '.join(plan.parent.symbols) +
            '\nEdit=' + json.dumps(dict(node=site.start, stop=site.stop, depth=site.depth,
                subtree_nodes=site.language.max_nodes, subtree_depth=site.language.max_depth), sort_keys=True) +
            '\nCurrent memory=' + memory_text + '\nReplacement subtree:')
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True)
class BoundEdit:
    sample: SampledEdit
    process: str
    attempt: int
    rho_version: int
    read_identity: str
    eligible: bool
    witness: dict


class EditProcess:
    def __init__(self, model, interface, parent, memory, *, control, seed, phase, round_index,
                 on_work, on_event, functional_probes=False):
        if (control not in ('keep', 'frozen') or phase not in ('fit', 'dev')
                or type(seed) is not int or seed not in (17, 29, 43)
                or type(round_index) is not int or not 0 <= round_index < (28 if phase == 'fit' else 2)
                or type(memory) is not EvidenceMemory or memory.version or any(memory.rows)
                or memory.reader_identity != interface.tokenizer_identity
                or memory.bounds.row_bytes != 4096 or memory.bounds.read_tokens != 1024
                or any(m.training for m in model.modules())
                or (functional_probes and phase != 'dev')):
            raise ValueError('fresh bounded fixed-rho proposal process required')
        self.model, self.interface, self.plan = model, interface, EditPlan(interface.language, parent)
        self.memory = memory
        self.memory0 = EvidenceMemory.restore(memory.export(), count_tokens=memory._count_tokens,
                                               expected_contract=memory.contract)
        self.control, self.seed, self.phase, self.round_index = control, seed, phase, round_index
        self.on_work, self.on_event, self.functional_probes = on_work, on_event, functional_probes
        self.rho_version = int(model.rho.version)
        self.guard = model_guard(model)
        self.rho_parameters = tensor_digest(dict(model.rho.state_dict()))
        self.attempt = self.memory_commits = 0
        self.pending = None
        self.receipts = []
        self.failed = False
        self.identity = identity(dict(schema='orbit-joint-edit-process-v1', interface=interface.identity,
            parent=parent.identity, memory0=memory.identity, control=control, seed=seed, phase=phase,
            round=round_index, rho=self.rho_version, rho_parameters=self.rho_parameters))

    def draw(self):
        if self.failed or self.pending is not None or self.attempt >= 4:
            raise RuntimeError('one unfinished draw, four attempts, no failed-state reuse')
        if model_guard(self.model) != self.guard:
            raise ValueError('slow rho or frozen computation changed inside a proposal process')
        text, read = self.memory.read()
        # Control name is intentionally absent; every process has the same exogenous tape seed.
        raw = f'orbit-process-credit-edit-v1:{self.seed}:{self.phase}:{self.round_index}:{self.attempt}'.encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.pending = dict(attempt=self.attempt, generator=generator, seed=seed, read=read,
            memory=self.memory.export(), memory_text=text, sample=None, trace=None)
        self.on_work(dict(kind='edit_attempt_reserved',count=1))
        self.on_event(dict(kind='EDIT_ATTEMPT_START', process=self.identity, attempt=self.attempt,
            parent=self.plan.parent.identity, rho_version=self.rho_version, seed=seed, memory_read=read))
        def progress(trace):
            self.pending['trace'] = trace
        # A failure leaves the pending draw/RNG accessible to the native failure owner.
        sample = sample_edit(self.model, self.plan, mapping=self.interface.mapping,
            site_prompt=self.interface.site_prompt(self.plan, text), site_tokens=self.interface.site_tokens,
            prompt_for_site=lambda j: self.interface.prompt(self.plan, j, text),
            generator=generator, on_work=self.on_work, on_progress=progress)
        witness = dict(status='NO_PRIOR_MEMORY_COMMIT' if self.control == 'keep' else 'FROZEN_CONTROL',
            functional=False, TV_lower_bound=0., forward_calls=0)
        if self.functional_probes and self.control == 'keep' and self.memory_commits:
            root_mask = tuple(self.interface.mapping[s] for s in PrefixGrammar(self.plan.sites[0].language, 'rtma').allowed())
            root_index = root_mask.index(self.interface.mapping[self.plan.parent.symbols[0]])
            reset_text, reset_read = self.memory0.read()
            distributions, site_distributions, root_distributions = [], [], []
            with torch.no_grad():
                for memory_text in (text, reset_text):
                    site_prompt = self.interface.site_prompt(self.plan, memory_text)
                    if not site_prompt or len(site_prompt)+1 > self.model.profile.local_tokens:
                        raise ValueError('whole site intervention prompt must fit')
                    ids = torch.tensor([site_prompt], device=self.model.rho.a.device, dtype=torch.long)
                    positions = torch.tensor([len(site_prompt)-1], device=ids.device)
                    self.on_work(dict(kind='site_functional_forward', input_tokens=len(site_prompt), positions=1))
                    scores = self.model.forward_batch(ids, torch.ones_like(ids), fast_states=None,
                        memory=None, logit_positions=positions)[0, 0, list(self.interface.site_tokens[:len(self.plan.sites)])].float().cpu()
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('nonfinite site intervention distribution')
                    site_distributions.append(scores.softmax(0).double())
                    prompt = self.interface.prompt(self.plan, 0, memory_text)
                    if not prompt or len(prompt) + self.plan.sites[0].language.max_tokens > self.model.profile.local_tokens:
                        raise ValueError('whole root-edit intervention prompt must fit')
                    ids = torch.tensor([prompt], device=self.model.rho.a.device, dtype=torch.long)
                    positions = torch.tensor([len(prompt)-1], device=ids.device)
                    self.on_work(dict(kind='edit_functional_forward', input_tokens=len(prompt), positions=1))
                    scores = self.model.forward_batch(ids, torch.ones_like(ids), fast_states=None,
                        memory=None, logit_positions=positions)[0, 0, list(root_mask)].float().cpu()
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError('nonfinite functional-probe distribution')
                    root_distributions.append(scores.softmax(0).double())
                    distributions.append(program_first_distribution(root_distributions[-1],
                        root_probability=site_distributions[-1][0], root_index=root_index))
            tv = float((distributions[0]-distributions[1]).abs().sum()/2)
            witness = dict(status='COMPLETE', functional=tv > 1e-4, TV_lower_bound=tv,
                tolerance=1e-4, forward_calls=4, retained_read=read, reset_read=reset_read,
                parent=self.plan.parent.identity, rho_version=self.rho_version,
                first_symbol_mask=list(root_mask),
                site_mask=list(self.interface.site_tokens[:len(self.plan.sites)]),
                retained_site_probability=site_distributions[0].tolist(),
                reset_site_probability=site_distributions[1].tolist(),
                retained_root_symbol_probability=root_distributions[0].tolist(),
                reset_root_symbol_probability=root_distributions[1].tolist(),
                retained_first_symbol_probability=distributions[0].tolist(),
                reset_first_symbol_probability=distributions[1].tolist(),
                estimand='complete-program first-symbol marginal under learned site/subtree editing')
        if model_guard(self.model) != self.guard:
            raise ValueError('sampling/probe changed the bound proposal computation')
        draw = BoundEdit(sample, self.identity, self.attempt, self.rho_version,
                         read['memory'], bool(self.memory_commits and witness['functional']), witness)
        self.pending['sample'] = draw
        self.on_event(dict(kind='EDIT_ATTEMPT_DRAWN', process=self.identity, attempt=self.attempt,
            site=sample.site_index, program=asdict(sample.program), subtree=list(sample.symbols),
            rho_version=self.rho_version, read=read, eligible=draw.eligible, witness=witness))
        return draw

    def feedback(self, draw: BoundEdit, *, loss, cost, tag):
        if (self.failed or self.pending is None or self.pending['sample'] is not draw
                or model_guard(self.model) != self.guard or self.memory.identity != draw.read_identity):
            raise ValueError('feedback must bind the actual unfinished draw and same-rho read')
        # This is the complete rebuilt program, not the sampled replacement subtree.
        record = Record(' '.join(draw.sample.program.symbols), loss, cost, tag,
            self.memory.source_prefix+'proposal/'+str(self.attempt), self.attempt)
        record.validate()
        managed = None
        if self.control == 'keep' and self.attempt < 3:
            self.memory, managed = self.memory.manage(record, language=self.interface.language,
                rule=self.plan.parent, opportunity_cap=4)
            self.memory_commits += int(managed['status'] == 'COMMIT')
        receipt = dict(attempt=self.attempt, parent=self.plan.parent.identity,
            program=draw.sample.program.identity, record=asdict(record), pre_memory=draw.read_identity,
            post_memory=self.memory.identity, memory=managed, final_attempt=self.attempt == 3,
            numerical_deployment_updates=0, control=self.control)
        self.receipts.append(receipt)
        self.on_event(dict(kind='EDIT_FEEDBACK', process=self.identity, receipt=receipt))
        self.pending = None
        self.attempt += 1
        return receipt

    def export(self):
        pending = None
        if self.pending is not None:
            pending = {k: v for k, v in self.pending.items() if k not in ('generator', 'sample')}
            pending['rng'] = self.pending['generator'].get_state().clone()
            pending['sample'] = None if self.pending['sample'] is None else asdict(self.pending['sample'])
        return dict(schema='orbit-joint-edit-process-state-v1', process=self.identity,
            attempt=self.attempt, rho_version=self.rho_version, rho_parameters=self.rho_parameters, control=self.control,
            memory=self.memory.export(), memory0=self.memory0.export(), memory_commits=self.memory_commits,
            receipts=self.receipts, pending=pending, failed=self.failed)
