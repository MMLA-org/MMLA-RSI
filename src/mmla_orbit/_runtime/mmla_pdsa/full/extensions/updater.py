"""Parent-rule numerical update shared by task and proposal roles.

Unlike the original fixed SGD, this uses (1-mu)*g, bounded momentum, and
coordinatewise projection around the bound role-specific slow initialization.
Rejection never mutates factors, optimizer, sampling RNG, or arrived history.
"""
from dataclasses import dataclass, asdict
import math
import torch
from .rules import Rule, evaluate, identity


@dataclass(frozen=True)
class UpdateBounds:
    learning_rate_max: float = .1
    clip_min: float = 1e-4
    clip_max: float = 1.
    momentum_max: float = .99
    momentum_abs_max: float = 1.
    radius: float = 1.
    gradient_reference: float = 1.

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError('finite numerical bounds')
        if not (self.learning_rate_max > 0 and 0 < self.clip_min <= self.clip_max
                and 0 <= self.momentum_max < 1 and self.momentum_abs_max > 0
                and self.radius > 0 and self.gradient_reference > 0):
            raise ValueError('invalid numerical bounds')


def apply_update(rho, fast, gradient, *, language, rule, bounds, loss,
                 previous_loss, opportunity, opportunity_cap):
    rho.factors(fast)  # Exact initialization identity/version, including incarnation.
    if (rule.mode != 'rtt' or not 0 <= loss <= 1 or not 0 <= previous_loss <= 1
            or type(opportunity) is not int or type(opportunity_cap) is not int
            or not 0 <= opportunity < opportunity_cap):
        raise ValueError('arrived role-local feedback and bounded opportunity required')
    a, b = gradient
    if a.shape != fast.a.shape or b.shape != fast.b.shape:
        raise ValueError('same row-major A/B gradient layout required')
    def checked(x):
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise FloatingPointError('nonfinite/incorrect-dtype update intermediate')
        return x
    g = checked(torch.cat((a.detach().flatten(), b.detach().flatten())))
    old = checked(torch.cat((fast.a.detach().flatten(), fast.b.detach().flatten())))
    anchor = checked(torch.cat((rho.a.detach().flatten(), rho.b.detach().flatten())))
    momentum = checked(torch.cat((fast.momentum_a.flatten(), fast.momentum_b.flatten())))
    squared = checked(g.square()); norm = checked(checked(squared.sum()).sqrt())
    z = (loss, max(-1., min(1., loss-previous_loss)),
         min(1., float(norm)/bounds.gradient_reference), min(1., opportunity/opportunity_cap))
    lr, clip, mu = evaluate(language, rule, numerical_features=z)
    lr = max(0., min(bounds.learning_rate_max, lr))
    clip = max(bounds.clip_min, min(bounds.clip_max, clip))
    mu = max(0., min(bounds.momentum_max, mu))
    scaled = checked(g * min(1., clip/(float(norm)+language.epsilon)))
    left, right = checked(mu*momentum), checked((1.-mu)*scaled)
    summed = checked(left+right)
    new_momentum = checked(summed.clamp(-bounds.momentum_abs_max, bounds.momentum_abs_max))
    delta = checked(-lr*new_momentum)
    candidate = checked(old+delta)
    lower, upper = checked(anchor-bounds.radius), checked(anchor+bounds.radius)
    candidate = checked(torch.maximum(lower, torch.minimum(upper, candidate)))
    new = fast.fork()
    n = fast.a.numel()
    new.a = candidate[:n].reshape_as(fast.a).detach().requires_grad_()
    new.b = candidate[n:].reshape_as(fast.b).detach().requires_grad_()
    new.momentum_a = new_momentum[:n].reshape_as(fast.a).detach().clone()
    new.momentum_b = new_momentum[n:].reshape_as(fast.b).detach().clone()
    new.steps += 1
    return new, dict(status='COMMIT', rule=rule.identity, language=language.identity,
        bounds=identity(asdict(bounds)), predecessor=fast.identity, successor=new.identity,
        factors_changed=not torch.equal(old, candidate),
        momentum_changed=not torch.equal(momentum, new_momentum),
        gradient_norm=float(norm), coefficients=dict(a=lr, kappa=clip, mu=mu),
        opportunity=opportunity, opportunity_cap=opportunity_cap, statistics=z)
