"""First-order slow-update core for a complete-task outcome recipe.

This module owns task rho and shared hybrid parameters only. Task adapters must
first bind lawful feedback, actual memory/trajectory provenance and endpoint
permissions. This core neither supplies that provenance nor starts a run.

Endpoint forwards use independent fast leaves (first_order_outer=False). Their
factor gradients are transferred to rho with an identity Jacobian; hybrid gets
only its direct gradient with the fast leaves fixed. No gradient through an
inner optimizer or a discrete trajectory is claimed. Auxiliary CE/KL forwards
use rho directly. A complete physical group contributes one equal group weight.
"""
from dataclasses import asdict, dataclass
import copy
import math

import torch

from .neural import FastState
from .profile import digest
from .module_training import finite_optimizer
from .numerics import all_finite


@dataclass(frozen=True)
class OutcomeUpdateConfig:
    binding: str
    objective: str
    learning_rate: float
    max_steps: int
    outer_clip: float
    auxiliary_weight: float

    def __post_init__(self):
        if (type(self.binding) is not str or len(self.binding) != 64
                or self.objective not in ('static', 'adapted')):
            raise ValueError('exact recipe binding and objective required')
        int(self.binding, 16)
        for value in (self.learning_rate, self.outer_clip, self.auxiliary_weight):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('positive finite training settings required')
        if self.learning_rate > 1 or type(self.max_steps) is not int or not 1 <= self.max_steps <= 4096:
            raise ValueError('bounded rate and update count required')


class OutcomeUpdateCore:
    """Accumulate a whole outer batch, then perform one slow AdamW update.

    Inner losses/optimizers and scientific data validation belong to the task
    adapter. It must keep fast state, inner optimizer and RNG private per task.
    add_endpoint enforces independent detached fast factors and shared versions;
    it does not infer actual feedback from a tensor's numerical value.
    """

    def __init__(self, model, config: OutcomeUpdateConfig):
        if type(config) is not OutcomeUpdateConfig:
            raise TypeError('explicit immutable configuration required')
        self.model, self.config = model, config
        self.named = tuple((name, p) for name, p in model.named_parameters()
                           if name in ('rho.a', 'rho.b') or name.startswith('hybrid.'))
        names = [name for name, _ in self.named]
        if names[:2] != ['rho.a', 'rho.b'] or not any(n.startswith('hybrid.') for n in names):
            raise ValueError('task rho followed by a nonempty shared hybrid required')
        if len({id(p) for _, p in self.named}) != len(self.named):
            raise ValueError('duplicate parameter ownership')
        selected = set(names)
        for name, p in model.named_parameters():
            p.requires_grad_(name in selected)
            p.grad = None
        self.optimizer = torch.optim.AdamW([p for _, p in self.named],
            lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        self.contract = dict(schema='mmla-outcome-gradient-core-v1',
            config=asdict(config), parameters=names, betas=[0.9, 0.999],
            eps=1e-8, weight_decay=0., estimator='FO-rho/direct-hybrid')
        self.contract_identity = digest(self.contract)
        self.steps = 0
        self.failed = False
        self._batch = None

    def _check(self):
        if (self.failed or digest(self.contract) != self.contract_identity
                or asdict(self.config) != self.contract['config']
                or type(self.steps) is not int or not 0 <= self.steps <= self.config.max_steps):
            raise RuntimeError('failed or changed training core')
        actual = tuple((n, p) for n, p in self.model.named_parameters()
                       if n in ('rho.a', 'rho.b') or n.startswith('hybrid.'))
        if len(actual) != len(self.named) or any(n != m or p is not q
                for (n, p), (m, q) in zip(actual, self.named)):
            raise ValueError('parameter ownership changed')
        expected_names = set(self.contract['parameters'])
        if any(p.requires_grad != (n in expected_names) for n, p in self.model.named_parameters()):
            raise ValueError('gradient ownership changed')
        groups = self.optimizer.param_groups
        if len(groups) != 1 or any(p is not q for p, (_, q) in zip(groups[0]['params'], self.named)) or len(groups[0]['params']) != len(self.named):
            raise ValueError('optimizer ownership changed')
        expected = dict(lr=self.config.learning_rate, betas=(.9, .999), eps=1e-8,
                        weight_decay=0., amsgrad=False, maximize=False)
        if any(groups[0][k] != v for k, v in expected.items()):
            raise ValueError('optimizer settings changed')
        if self._batch is not None and self._versions() != self._batch['versions']:
            raise ValueError('slow parameters changed within the outer batch')

    def _versions(self):
        # Supported writers use copy_/optimizer.step; raw .data mutation is unsupported.
        # Frozen parameters must also remain unchanged during an outer batch.
        return tuple((n, id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device))
                     for n, p in self.model.named_parameters()) + (self.model.rho._state_identity(),)

    def begin(self, groups):
        self._check()
        if self._batch is not None or self.steps >= self.config.max_steps:
            raise RuntimeError('batch active or complete finite update budget consumed')
        if type(groups) is not dict or not 1 <= len(groups) <= 64:
            raise ValueError('complete physical-group layout required')
        layout = {}
        all_views = set()
        for group, views in groups.items():
            if (type(group) is not str or not group or type(views) not in (tuple, list)
                    or not 1 <= len(views) <= 64
                    or any(type(v) is not str or not v for v in views)
                    or len(set(views)) != len(views) or all_views.intersection(views)):
                raise ValueError('distinct identified task views required')
            all_views.update(views)
            layout[group] = tuple(views)
        self.optimizer.zero_grad(set_to_none=True)
        self._batch = dict(layout=layout, versions=self._versions(), seen=set(),
            fast=[], fast_storage=set(), auxiliary=False, auxiliary_progress=None, gradients=[torch.zeros_like(p) for _, p in self.named],
            endpoint_loss=0., auxiliary_loss=0.)

    @staticmethod
    def _loss(value):
        if (not torch.is_tensor(value) or value.ndim != 0 or not value.requires_grad
                or not bool(torch.isfinite(value))):
            raise ValueError('finite scalar differentiable loss required')

    def _validate_fast(self, fast):
        b = self._batch
        if b is None:raise ValueError('active outer batch required')
        if type(fast) is not FastState:
            raise TypeError('actual independent fast state required')
        self.model.rho.factors(fast)
        for value, reference in ((fast.a, self.model.rho.a), (fast.b, self.model.rho.b),
                (fast.momentum_a, self.model.rho.a), (fast.momentum_b, self.model.rho.b)):
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError('fast factors and optimizer state must match rho layout')
        if (fast.momentum_a.requires_grad or fast.momentum_b.requires_grad
                or type(fast.steps) is not int or fast.steps < 0
                or not all_finite((fast.a, fast.b, fast.momentum_a, fast.momentum_b))):
            raise ValueError('invalid detached fast optimizer state')
        if any(not p.is_leaf or not p.requires_grad or p.grad_fn is not None for p in (fast.a, fast.b)):
            raise ValueError('inner optimizer must return detached fast leaves')
        if any(p.untyped_storage().data_ptr() == q.untyped_storage().data_ptr() for p in (fast.a, fast.b)
               for q in (self.model.rho.a, self.model.rho.b)):
            raise ValueError('fast factors must not alias slow initialization')
        tensors = (fast.a, fast.b, fast.momentum_a, fast.momentum_b, fast.rng_state, fast.initial_rng)
        if any((str(p.device),p.untyped_storage().data_ptr()) in b['fast_storage'] for p in tensors):
            raise ValueError('episode factors, inner state or RNG alias another task')
        if self.config.objective == 'static' and (fast.steps != 0
                or not torch.equal(fast.a, self.model.rho.a) or not torch.equal(fast.b, self.model.rho.b)):
            raise ValueError('static endpoint must use the current rho state')

    def add_endpoint(self, group, view, fast, loss):
        self._check()
        b = self._batch
        if b is None or group not in b['layout'] or view not in b['layout'][group] or (group, view) in b['seen']:
            raise ValueError('unknown or duplicate episode contribution')
        self._validate_fast(fast)
        self._loss(loss)
        weight = 1 / len(b['layout']) / len(b['layout'][group])
        targets = (fast.a, fast.b) + tuple(p for _, p in self.named[2:]) + (self.model.rho.a, self.model.rho.b)
        try:
            gs = torch.autograd.grad(loss, targets, allow_unused=True)
            if gs[0] is None or gs[1] is None or gs[-2] is not None or gs[-1] is not None:
                raise ValueError('endpoint must depend on fast leaves, not directly on rho')
            self._accumulate(gs[:-2], weight)
        except BaseException:
            self.failed = True
            raise
        b['seen'].add((group, view));b['fast'].append(fast)
        b['fast_storage'].update((str(p.device),p.untyped_storage().data_ptr()) for p in
            (fast.a,fast.b,fast.momentum_a,fast.momentum_b,fast.rng_state,fast.initial_rng))
        b['endpoint_loss'] += float(loss.detach()) * weight

    def add_auxiliary(self, loss):
        """One mean over all general conversations, using slow rho directly."""
        self.add_auxiliary_part(loss, offset=0, rows=1, total_rows=1)

    def add_auxiliary_part(self, mean_loss, *, offset, rows, total_rows):
        """Stream disjoint consecutive rows of one complete auxiliary mean.

        Each incoming scalar is the mean of its rows; row counts recover the
        original whole-batch mean, including a short final microbatch. There is
        still exactly one auxiliary objective and one final optimizer step.
        """
        self._check();b = self._batch
        if (b is None or b['auxiliary'] or any(type(x) is not int for x in (offset,rows,total_rows))
                or not 1 <= total_rows <= 64 or not 1 <= rows <= total_rows
                or offset < 0 or offset+rows > total_rows):
            raise ValueError('one bounded complete auxiliary row partition required')
        progress = b['auxiliary_progress']
        if ((progress is None and offset != 0)
                or (progress is not None and progress != (offset,total_rows))):
            raise ValueError('auxiliary rows must cover the original batch once in order')
        self._loss(mean_loss)
        weight = rows/total_rows
        try:
            gs = torch.autograd.grad(mean_loss, tuple(p for _, p in self.named), allow_unused=True)
            if gs[0] is None or gs[1] is None:
                raise ValueError('auxiliary loss must use slow rho directly')
            self._accumulate(gs, self.config.auxiliary_weight*weight)
        except BaseException:
            self.failed = True
            raise
        b['auxiliary_progress'] = (offset+rows,total_rows)
        b['auxiliary'] = offset+rows == total_rows
        b['auxiliary_loss'] += float(mean_loss.detach())*weight

    def _accumulate(self, gradients, weight):
        if len(gradients) != len(self.named) or not all_finite(g for g in gradients if g is not None):
            raise FloatingPointError('nonfinite or incomplete gradient contribution')
        values = []
        for g, (_, p) in zip(gradients, self.named):
            if g is not None and (g.shape != p.shape or g.device != p.device or g.dtype != p.dtype):
                raise FloatingPointError('nonfinite or malformed contribution')
            values.append(torch.zeros_like(p) if g is None else g.detach() * weight)
        for total, value in zip(self._batch['gradients'], values):
            total.add_(value)

    def commit(self):
        self._check();b = self._batch
        if b is None or not b['auxiliary'] or len(b['seen']) != sum(map(len, b['layout'].values())):
            raise ValueError('whole outer batch and auxiliary objective required')
        if not all_finite(b['gradients']):
            self.failed = True
            raise FloatingPointError('nonfinite accumulated gradient')
        for (_, p), g in zip(self.named, b['gradients']):p.grad = g
        try:
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.named],
                self.config.outer_clip, error_if_nonfinite=True)
            self.optimizer.step()
            if not all_finite(p for _, p in self.named):
                raise FloatingPointError('nonfinite slow parameters after update')
            finite_optimizer(self.optimizer)
            with torch.no_grad():self.model.rho.version.add_(1)
        except BaseException:
            # No retry after a partial optimizer write; the caller saves failure
            # and may only resume via a separately approved complete checkpoint.
            self.failed = True
            raise
        self.steps += 1
        result = dict(step=self.steps, groups=len(b['layout']), views=len(b['seen']),
            endpoint_loss=b['endpoint_loss'], auxiliary_loss=b['auxiliary_loss'],
            preclip_gradient_norm=float(norm), rho_version=int(self.model.rho.version),
            estimator=self.contract['estimator'], shared_reader_rebinding_required=True)
        self._batch = None
        self.optimizer.zero_grad(set_to_none=True)
        return result

    def export(self):
        """Only complete slow updates; full model/RNG/data cursor live in outer checkpoint."""
        self._check()
        if self._batch is not None:
            raise ValueError('cannot export a partial outer batch as a completed step')
        finite_optimizer(self.optimizer)
        return dict(schema=self.contract['schema'], contract=copy.deepcopy(self.contract),
            steps=self.steps, rho_version=int(self.model.rho.version),
            parameters={n:p.detach().cpu().clone() for n,p in self.named},
            optimizer=copy.deepcopy(self.optimizer.state_dict()))
