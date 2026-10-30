"""One same-rho batch update from closed whole-task returns.

The native cohort owns sampling, M, identity authentication, checkpoints and
resource closure. This numerical core consumes their typed batch. It never
starts an environment or turns an operational failure into a reward.
"""
from dataclasses import dataclass
from typing import Callable
import torch
from proposal_edit import SampledEdit, replay_objective


@dataclass(frozen=True)
class RewardBatch:
    rho_version: int
    task_ids: tuple[str, ...]
    candidate_task_ids: tuple[tuple[str, ...], ...]
    parent_losses: tuple[float, ...]
    candidate_losses: tuple[tuple[float, ...], ...]
    samples: tuple[SampledEdit, ...]

    def __post_init__(self):
        if (len(self.task_ids) != 4 or len(set(self.task_ids)) != 4
                or len(self.samples) != 4 or len(self.candidate_losses) != 4
                or self.candidate_task_ids != (self.task_ids,) * 4
                or len(self.parent_losses) != 4
                or any(len(row) != 4 for row in self.candidate_losses)):
            raise ValueError('four complete paired tasks for each of four proposals required')
        if any(value not in (0., 1.) for row in (self.parent_losses,) + self.candidate_losses for value in row):
            raise ValueError('only closed binary full-task losses may become returns')
        if len({sample.parent for sample in self.samples}) != 1:
            raise ValueError('one current parent owns this batch')


def make_optimizer(model):
    """Set ownership once at the start of the isolated proposal course."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in ('rho.a', 'rho.b'))
        parameter.grad = None
    model.eval()
    return torch.optim.AdamW((model.rho.a, model.rho.b), lr=1e-4, betas=(.9, .999),
                             eps=1e-8, weight_decay=0., foreach=False, fused=False)


def causal_returns(advantages):
    """Four complete-task advantages; include each draw's future M-mediated return.

    The batch constructor owns pairing and bounded outcome validation. The one
    factor of 1/4 implements the mean of four attempts, not a second batch mean.
    """
    return tuple(sum(advantages[j:])/4 for j in range(4))


def apply_reward_update(model, optimizer, batch: RewardBatch, *, on_work: Callable):
    """Accumulate four sampled sequence gradients, then commit one slow update.

    Repeated programs/samples are allowed; they remain separate actual draws.
    The host stops the course if this function raises, retaining the partial
    evidence. It must not reuse partially mutated state as a successful update.
    """
    if int(model.rho.version) != batch.rho_version or not 156 <= batch.rho_version < 184:
        raise ValueError('batch must bind the current pre-update slow rho')
    parameters = (model.rho.a, model.rho.b)
    before = tuple(p.detach().clone() for p in parameters)
    momenta = tuple(tuple(optimizer.state[p][k].detach().clone() if k in optimizer.state.get(p, {}) else None
                         for k in ('exp_avg', 'exp_avg_sq')) for p in parameters)
    advantages = tuple(sum(p-c for p,c in zip(batch.parent_losses,row))/4 for row in batch.candidate_losses)
    returns = causal_returns(advantages)
    optimizer.zero_grad(set_to_none=True)
    objective, replays = 0., []
    for sample, future_return in zip(batch.samples, returns):
        loss, evidence = replay_objective(model, sample, future_return, on_work=on_work, backward_scale=1.)
        on_work(dict(kind='edit_reward_backward', candidates=1))
        objective += float(loss.detach())
        replays.append(evidence)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('slow rho requires finite connected gradients')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError('nonfinite slow update')
    if any(torch.is_tensor(value) and not torch.isfinite(value).all()
           for state in optimizer.state.values() for value in state.values()):
        raise FloatingPointError('nonfinite optimizer state')
    changed = tuple(not torch.equal(old, p.detach()) for old,p in zip(before,parameters))
    moments_changed = any(old is None or not torch.equal(old, optimizer.state[p][key])
        for p, pair in zip(parameters, momenta) for old,key in zip(pair,('exp_avg','exp_avg_sq')))
    with torch.no_grad():
        model.rho.version.add_(1)
    optimizer.zero_grad(set_to_none=True)
    return dict(rho_version=int(model.rho.version), paired_advantages=advantages,
        returns_to_go=returns, objective_scope='mean complete-task success over four proposals',
        surrogate_objective=objective, preclip_gradient_norm=float(norm),
        factor_changed=changed, optimizer_moments_changed=moments_changed,
        numerical_deployment_updates=0, replay=replays)
