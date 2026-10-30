"""Differentiable carriers for the unified model; no loading or execution here."""
from dataclasses import dataclass
import copy
import hashlib
import math
import uuid

import torch
from torch import nn
from torch.nn import functional as F


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        x = value.detach().contiguous().cpu()
        h.update(str((name, tuple(x.shape), str(x.dtype))).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass
class FastState:
    a: torch.Tensor
    b: torch.Tensor
    momentum_a: torch.Tensor
    momentum_b: torch.Tensor
    rho_version: int
    steps: int
    epoch: int
    rng_state: torch.Tensor
    initial_rng: torch.Tensor
    rho_identity: str
    recovery_id: str = ""

    @property
    def identity(self):
        return tensor_digest(dict(a=self.a, b=self.b, ma=self.momentum_a, mb=self.momentum_b,
                                  rng=self.rng_state)) + f":{self.rho_version}:{self.steps}:{self.epoch}:{self.rho_identity}" + (":"+self.recovery_id if self.recovery_id else "")

    def fork(self):
        return FastState(self.a.detach().clone().requires_grad_(), self.b.detach().clone().requires_grad_(),
            self.momentum_a.clone(), self.momentum_b.clone(), self.rho_version, self.steps,
            self.epoch, self.rng_state.clone(), self.initial_rng.clone(), self.rho_identity, self.recovery_id)


class LearningResidual(nn.Module):
    def __init__(self, hidden, *, seed, rank=4):
        super().__init__()
        if rank != 4:
            raise ValueError("this projection has input LARC4")
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.a = nn.Parameter(torch.randn(rank, hidden, generator=g) * 0.02)
        self.b = nn.Parameter(torch.zeros(hidden, rank))
        self.register_buffer("version", torch.tensor(0, dtype=torch.int64))
        self._incarnation = uuid.uuid4().hex
        self.initialization = dict(a=self.a.detach().clone(), b=self.b.detach().clone(), seed=seed)

    def _state_identity(self):
        # Registered writers use optimizer.step/copy_/load_state_dict, all of
        # which advance PyTorch versions. Unsafe .data/storage mutation is not
        # supported. This caches identity bytes only, never mixer/Phi outputs.
        tensors=(self.a,self.b,self.version)
        guard=tuple((id(x),x._version,x.device,x.dtype,tuple(x.shape),x.data_ptr()) for x in tensors)
        saved=getattr(self,'_identity_cache',None)
        if saved is None or saved[0]!=guard:
            saved=(guard,tensor_digest(dict(a=self.a,b=self.b,version=self.version)),int(self.version))
            self._identity_cache=saved
        return self._incarnation+":"+saved[1],saved[2]

    @property
    def identity(self):
        return self._state_identity()[0]

    def invalidate(self):
        self._incarnation = uuid.uuid4().hex

    def begin(self, seed):
        rng = torch.Generator(device="cpu").manual_seed(seed).get_state()
        return FastState(self.a.detach().clone().requires_grad_(), self.b.detach().clone().requires_grad_(),
            torch.zeros_like(self.a), torch.zeros_like(self.b), self._state_identity()[1], 0, 0,
            rng.clone(), rng.clone(), self.identity)

    def reset(self, fast):
        identity,version=self._state_identity()
        if fast.rho_version != version or fast.rho_identity != identity:
            raise ValueError("rho changed during episode")
        fresh = self.begin(0)
        fresh.rng_state = fresh.initial_rng = fast.initial_rng.clone()
        fresh.epoch = fast.epoch + 1
        fresh.recovery_id = fast.recovery_id
        return fresh

    def factors(self, fast=None, *, first_order_outer=False):
        if fast is None:
            return self.a, self.b
        identity,version=self._state_identity()
        if fast.rho_version != version or fast.rho_identity != identity:
            raise ValueError("stale episode")
        if first_order_outer:
            return self.a + (fast.a-self.a).detach(), self.b + (fast.b-self.b).detach()
        return fast.a, fast.b

    def inject(self, value, fast=None, *, first_order_outer=False):
        a, b = self.factors(fast, first_order_outer=first_order_outer)
        return value + ((value.float() @ a.T) @ b.T).to(value.dtype)

    def propose(self, fast, loss, *, learning_rate, momentum, clip, max_steps):
        self.factors(fast)
        if fast.steps >= max_steps:
            raise RuntimeError("fast-update budget exhausted")
        if (not 0 < learning_rate <= 1 or not 0 <= momentum < 1 or not 0 < clip < math.inf
                or loss.ndim != 0 or not torch.isfinite(loss)):
            raise ValueError("invalid bounded update")
        ga, gb = torch.autograd.grad(loss, (fast.a, fast.b))
        if not all(torch.isfinite(v).all() for v in (ga, gb)):
            raise FloatingPointError("nonfinite gradient, original Phi/optimizer retained")
        norm = torch.sqrt(ga.square().sum()+gb.square().sum())
        scale = (clip/(norm+1e-12)).clamp(max=1.0)
        ma, mb = momentum*fast.momentum_a + ga*scale, momentum*fast.momentum_b + gb*scale
        a, b = fast.a-learning_rate*ma, fast.b-learning_rate*mb
        if not all(torch.isfinite(v).all() for v in (a, b, ma, mb)):
            raise FloatingPointError("nonfinite successor, original Phi/optimizer retained")
        new = fast.fork()
        new.a, new.b = a.detach().requires_grad_(), b.detach().requires_grad_()
        new.momentum_a, new.momentum_b = ma.detach(), mb.detach()
        new.steps += 1
        return new, dict(a_gradient_norm=float(ga.norm()), b_gradient_norm=float(gb.norm()),
                         parameter_changed=not (torch.equal(a, fast.a) and torch.equal(b, fast.b)))


class CompletedSegmentEncoder(nn.Module):
    """Bidirectional only on a completed observed view, never on future tokens."""
    def __init__(self, hidden, width):
        super().__init__()
        if width % 2:
            raise ValueError("even consolidation width required")
        self.input = nn.Linear(hidden, width)
        self.recurrent = nn.GRU(width, width//2, bidirectional=True, batch_first=True)

    def forward(self, observed_embeddings):
        if observed_embeddings.ndim != 3 or observed_embeddings.shape[1] == 0:
            raise ValueError("one nonempty completed view required")
        _, state = self.recurrent(torch.tanh(self.input(observed_embeddings.float())))
        return torch.cat((state[-2], state[-1]), dim=-1)


class MemoryRead(nn.Module):
    def __init__(self, hidden, key_width, value_width):
        super().__init__()
        self.query = nn.Linear(hidden, key_width, bias=False)
        self.output = nn.Linear(value_width, hidden, bias=False)
        self.gate = nn.Linear(hidden, 1)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, local, keys, values, *, top_k):
        if keys.shape[0] == 0:
            return local
        q = F.normalize(self.query(local.float()), dim=-1)
        k = F.normalize(keys.float(), dim=-1)
        scores = q @ k.T
        selected = scores.topk(min(top_k, keys.shape[0]), dim=-1)
        weights = selected.values.softmax(-1)
        read = (values[selected.indices] * weights.unsqueeze(-1)).sum(-2)
        delta = torch.sigmoid(self.gate(local.float())) * self.output(read)
        return local + delta.to(local.dtype)


class ActionRisk(nn.Module):
    """Continuous risk regression; random initialization is explicitly untrained."""
    def __init__(self, feature_width, width=64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_width, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.int64))

    def forward(self, causal_features):
        if not torch.isfinite(causal_features).all():
            raise ValueError("nonfinite causal features")
        return self.network(causal_features).squeeze(-1)

    def objective(self, causal_features, complete_action_risks):
        scores = self(causal_features)
        if scores.shape != complete_action_risks.shape or not torch.isfinite(complete_action_risks).all():
            raise ValueError("one detached teacher risk per complete feasible action")
        return F.huber_loss(scores, complete_action_risks.detach())
