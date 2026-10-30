"""Bind recursive rules to the corrected Orbit observation-memory domain.

This is role construction, not an admission controller. It does not load a
checkpoint, execute an environment, accept a rule, or consume audit data.
Production callers must pin the frozen imports and the full execution contract.
"""
from dataclasses import asdict
import copy
import json

from demonstration_factory import DemonstrationFactory
from managed_task_state import ManagedTaskState
from task_source_manager import DOMAIN
from mmla_pdsa.full.observation_memory import observation_memory, ObservationEncoder
from content_roles import components
from content_rules import parse_rule
from mmla_pdsa.full.extensions.rsi_binding import check_role_pair
from content_binding import ProductionProposalFactory
from mmla_pdsa.full.profile import digest


def _closed_model(model):
    if any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError('actual frozen task/proposal slow modules required')


def _formatter_execution(formatter):
    formatter._check()
    return dict(max_tokens=formatter.max_tokens, max_text_bytes=formatter.max_text_bytes,
                max_action_bytes=formatter.max_action_bytes, template=formatter._template,
                specials=formatter._specials, eos=formatter._eos)


class BoundManagedTaskState(ManagedTaskState):
    """Check the factory's live binding at every existing task-state boundary.

    This explicit subtype needs a reviewed RSI runner; the closed training
    runner's exact ManagedTaskState type check intentionally does not accept it.
    """
    def __init__(self, *, factory, **kwargs):
        self._factory = factory
        super().__init__(factory.model, **kwargs)
        self._fixed_fast_identity = self.fast.identity

    def _check(self):
        try:
            self._factory._check_binding()
            if (self.model is not self._factory.model
                    or self.formatter is not self._factory.formatter
                    or self.codec is not self._factory.codec
                    or type(self.source.encoder) is not ObservationEncoder
                    or self.source.encoder.encoder is not self._factory.encoder
                    or self.source.encoder.codec is not self._factory.codec):
                raise ValueError('task state changed outside its factory binding')
            super()._check()
            if self._factory.task_policy == "fixed_rho" and self.fast.identity != self._fixed_fast_identity:
                raise ValueError("fixed task Phi changed")
        except BaseException:
            self.failed = True
            raise


    def commit_feedback(self, context, **kwargs):
        self._check()
        if self._factory.task_policy == 'fixed_rho' and (
                kwargs.get('allow_update') is not False or kwargs.get('gradient') is not None):
            self.failed = True
            raise ValueError('fixed task Phi forbids numerical update requests')
        return super().commit_feedback(context, **kwargs)


class ManagedCandidateFactory(DemonstrationFactory):
    """Fresh task states with one declared rule component or a complete pair.

    Adaptive task execution keeps the inactive rule at its baseline. The
    explicit fixed_rho mode is RTMA-only: task factors remain at trained rho
    and every request for a numerical update is rejected.
    """
    def __init__(self, baseline, rule, *, task_policy="adaptive"):
        if type(baseline) is not DemonstrationFactory:
            raise ValueError('exact corrected observation factory required')
        _closed_model(baseline.model)
        if (task_policy not in ('adaptive', 'fixed_rho')
                or (task_policy == 'fixed_rho' and rule.mode != 'rtma')):
            raise ValueError('fixed task Phi is an explicit RTMA-only execution')
        if (baseline.definition['task_memory_domain'] != DOMAIN
                or parse_rule(baseline.language, rule.mode, rule.symbols) != rule):
            raise ValueError('canonical candidate in the bound observation domain')
        self.__dict__ = baseline.__dict__.copy()
        # Private execution bounds; only the pinned tokenizer is shared.
        self.formatter = copy.copy(baseline.formatter)
        self.definition = copy.deepcopy(baseline.definition)
        self.rule = rule
        self.task_policy = task_policy
        updater, manager = components(baseline.language, rule)
        self.updater = baseline.updater if updater is None else updater
        self.manager = baseline.manager if manager is None else manager
        self.definition.update(
            schema='orbit-managed-rule-task-interface-v2',
            task_policy=task_policy,
            max_updates=0 if task_policy == "fixed_rho" else 8,
            formatter_execution=_formatter_execution(self.formatter),
            updater=asdict(self.updater), manager=asdict(self.manager),
            candidate=asdict(rule), baseline_interface=digest(baseline.definition),
            task_execution=('RTMA-only; trained rho fixed; U never called' if task_policy == 'fixed_rho'
                else 'joint; fixed inactive rule component for single-rule search'),
            J_status='UNQUALIFIED_FOR_THIS_MEMORY_DOMAIN_NOT_USED_FOR_SELECTION')
        self._bound_definition = digest(self.definition)
        self._bound_versions = self._versions()
        self._bound_objects = (self.formatter, self.codec, self.encoder, self.language, self.bounds)
        self._bound_tokenizer = self.formatter.tokenizer

    def _versions(self):
        return tuple((n,id(t),t._version,str(t.device),str(t.dtype),t.requires_grad)
                     for n,t in tuple(self.model.named_parameters()) + tuple(self.model.named_buffers()))

    def _check_binding(self):
        _closed_model(self.model)
        if (digest(self.definition) != self._bound_definition
                or any(a is not b for a,b in zip(self._bound_objects,
                    (self.formatter,self.codec,self.encoder,self.language,self.bounds)))
                or self.definition['task_policy'] != self.task_policy
                or self.definition['max_updates'] != (0 if self.task_policy == 'fixed_rho' else 8)
                or self.definition['profile'] != self.model.profile.identity
                or self.definition['formatter'] != self.formatter.identity
                or self.formatter.tokenizer is not self._bound_tokenizer
                or self.definition['formatter_execution'] != _formatter_execution(self.formatter)
                or self.definition['codec'] != self.codec.identity
                or self.definition['encoder'] != self.encoder.identity
                or self.definition['language'] != asdict(self.language)
                or self.definition['bounds'] != asdict(self.bounds)
                or self.definition['candidate'] != asdict(self.rule)
                or self.definition['updater'] != asdict(self.updater)
                or self.definition['manager'] != asdict(self.manager)
                or self._versions() != self._bound_versions):
            raise ValueError('task reader or candidate changed outside a new binding')

    def reader_identity(self):
        self._check_binding()
        return super().reader_identity()

    def new_state(self, *, seed, tenant, manager=None):
        self._check_binding()
        if manager is not None:
            raise ValueError('candidate memory rule cannot be overridden per episode')
        return BoundManagedTaskState(factory=self, formatter=self.formatter, codec=self.codec,
            source=observation_memory(self.codec,self.encoder), reader_identity=self.reader_identity(),
            seed=seed, tenant=tenant, principal='actor', language=self.language,
            updater=self.updater, manager=self.manager, bounds=self.bounds,
            max_updates=self.definition["max_updates"], horizon=64)


def dual_role_interface(task_factory):
    if type(task_factory) is not ManagedCandidateFactory:
        raise ValueError('explicit new-domain candidate binding required')
    task_factory._check_binding()
    return dict(
        schema='orbit-task-observation-and-proposal-content-v1',
        task_reader=task_factory.reader_identity(),
        task_interface=copy.deepcopy(task_factory.definition),
        task=dict(domain=DOMAIN, slots=task_factory.model.profile.memory_slots,
            actions='NULL or a complete WRITE_BATCH_i observation replacement',
            common_initialization='WRITE_BATCH_0 before the first task action; not a V decision',
            features=[
                'x0=is_NULL', 'x1=occupied fraction of affected slots',
                'x2=new bounded observation loss', 'x3=new bounded action cost',
                'x4=mean replaced occupied rows loss, zero if none',
                'x5=mean replaced occupied rows cost, zero if none',
                'x6=mean min(1,age/64) over replaced occupied rows, zero if none',
                'x7=whole table occupied fraction',
                'x8=weighted actual successor units absent from current retained units / common Z',
                'x9=weighted current units absent from actual successor units / common Z'],
            content='lowercase ASCII word/pair units within each retained fragment; goal word overlap weight2, otherwise1; Z=max(1,weight(current union complete arrival)); no timestamp/receipt novelty or lost-row reconstruction',
            feedback='arrived next-observation CE; bounded loss=-expm1(-CE)',
            cost='actual command tokens / profile.max_new_tokens',
            update_limit=0 if task_factory.task_policy == 'fixed_rho' else 8, horizon=64,
            policy_state=task_factory.task_policy,
            feedback_use=('memory features only; no numerical update' if task_factory.task_policy == 'fixed_rho'
                else 'numerical update and memory features'),
            source_of_action_information='current visible frame plus current authoritative observation M'),
        proposal=dict(domain='complete-candidate-evaluation-content-v1',
            actions='NULL or replacement of one complete evaluation record',
            features=[
                'x0=is_NULL', 'x1=target slot occupied',
                'x2=new candidate task loss', 'x3=new candidate bounded cost',
                'x4=old record loss, zero if empty', 'x5=old record cost, zero if empty',
                'x6=min(1,record age / proposal opportunity cap), zero if empty',
                'x7=whole table occupied fraction',
                'x8=actual successor units absent from current retained units / common Z',
                'x9=current units absent from actual successor units / common Z'],
            content='program lowercase ASCII word/pair units, tag and min(15,floor(16*x)) loss/cost bins; all weights1; Z=max(1,units in current union complete arrival); no source/opportunity novelty',
            feedback='complete candidate development task loss in [0,1]; score-function gradient for numerical role',
            cost='bounded forward-input proxy declared by the complete scientific contract',
            read_boundary='immutable header plus current proposal evidence M only; no complete archive or old KV'),
        ownership='distinct task/proposal rho, Phi, momentum and RNG; only frozen common modules may be shared',
        rule_search='rtt changes U; rtma changes V; joint changes both; task_policy declares whether task U is used',
        proposal_adaptation='only the current parent components adapt the proposal learner; candidates are task-evaluated only',
        admission='external independent gate; this binding does not authorize or perform acceptance')


class ObservationProposalFactory(ProductionProposalFactory):
    """Proposal learner whose header and identity bind both memory payloads."""
    def __init__(self, model, tokenizer, *, task_factory, **kwargs):
        _closed_model(model)
        check_role_pair(task_factory.model, model)
        self.task_factory = task_factory
        self.role_interface = dual_role_interface(task_factory)
        self.bound_parent = task_factory.rule
        super().__init__(model, tokenizer, **kwargs)

    def configuration(self):
        return digest(dict(schema='orbit-content-dual-domain-proposal-factory-v1',
            production_configuration=super().configuration(),
            header_schema='complete-rsi-dual-content-interface-v1',
            role_interface=self.role_interface, parent=asdict(self.bound_parent)))

    def header(self, parent):
        if parent != self.bound_parent:
            raise ValueError('proposal parent must match the bound task rule')
        codec = self.codecs[parent.mode]
        alphabet = {s:self.tokenizer.decode([i],skip_special_tokens=False) for s,i in codec.pairs}
        return (
            'Propose a bounded adaptation program in prefix notation. Emit only code tokens. '
            'Binary operators add, sub, mul, min, max, sdiv; protected division uses the fixed language epsilon. '
            'RTT has three trees for step size, clipping, momentum. RTMA has one memory-action scoring tree. '
            'Joint concatenates RTT then RTMA. Current parent: '+json.dumps(asdict(parent),sort_keys=True)+
            '\nLanguage: '+json.dumps(asdict(self.language),sort_keys=True)+
            '\nConstants c_i: '+json.dumps([self.language.constant(i) for i in range(len(self.language.constants))])+
            '\nU features x0=loss, x1=clip(loss-previous_loss,-1,1), '
            'x2=min(1,gradient_L2/gradient_reference), x3=opportunity/opportunity_cap. '
            'The roles have different feedback and memory payloads as specified below. '
            'V chooses maximum score, NULL first then ascending slot on ties. '
            'U clamps outputs, uses mu*momentum+(1-mu)*clipped_gradient, clips momentum, '
            'and projects factors around the respective role rho. '
            '\nBounds: '+json.dumps(asdict(self.bounds),sort_keys=True)+
            '\nTask and proposal interfaces: '+json.dumps(self.role_interface,sort_keys=True)+
            '\nCodebook symbol to literal token: '+json.dumps(alphabet,sort_keys=True))

    def create(self, parent, generation, seed, on_forward):
        _closed_model(self.model)
        if parent != self.bound_parent:
            raise ValueError('a different parent requires a newly bound generation')
        self.task_factory._check_binding()
        check_role_pair(self.task_factory.model, self.model)
        if self.task_factory.reader_identity() != self.role_interface['task_reader']:
            raise ValueError('task role no longer matches the proposal binding')
        return super().create(parent, generation, seed, on_forward)
