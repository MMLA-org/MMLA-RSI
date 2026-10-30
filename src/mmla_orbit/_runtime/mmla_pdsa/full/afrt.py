"""AFRT transport migrated from the original v4 research implementation.

The numerical operator is preserved: anchor posterior + unsigned distance bias +
mean log-sigmoid typed path scores, shared across directions/repetitions. Input
`document_tokens` here is a registered relation-type lattice (-1/nonrelation),
NOT MiniCPM token IDs. A closed-segment semantic frontend must predict this
lattice, anchors, coverage and filler spans; that frontend is not supplied here.

Private NULLs are compressed into the final output column per role. The retained
D0/U0 heads are reference operators, not new evaluated baselines. No old weights
or synthetic token-ID parser are silently imported.
Original source SHA256: 13e0acff42cf12ac87461e6be5c5b48d308cbb6368d16d8da8614c2b36adc271
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class DynamicPartialSinkhorn(nn.Module):
    """Per-example partial one-to-one projection with explicit role NULLs."""

    def __init__(self, role_count: int, sinkhorn_iterations: int) -> None:
        super().__init__()
        if not 1 <= role_count <= 8 or not 1 <= sinkhorn_iterations <= 256:
            raise ValueError("invalid Sinkhorn registration")
        self.role_count = role_count
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(
        self,
        logits: torch.Tensor,
        candidate_mask: torch.Tensor,
        null_logits: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError("logits must be [B,E,R,L]")
        batch, events, roles, candidates = logits.shape
        if not 1 <= batch <= 64 or events > 16 or candidates > 512 or not torch.isfinite(logits).all() or not torch.isfinite(null_logits).all():
            raise ValueError("finite bounded assignment required")
        if roles != self.role_count:
            raise ValueError("role count drift")
        if candidate_mask.shape != (batch, events, candidates):
            raise ValueError("candidate mask geometry drift")
        if null_logits.shape != (batch, events, roles):
            raise ValueError("NULL logit geometry drift")
        flat = batch * events
        size = roles + candidates
        floor = torch.tensor(-1e4, dtype=logits.dtype, device=logits.device)
        matrix = floor.expand(flat, size, size).clone()
        flat_logits = logits.reshape(flat, roles, candidates)
        flat_mask = candidate_mask.reshape(flat, candidates)
        matrix[:, :roles, :candidates] = flat_logits.masked_fill(
            ~flat_mask[:, None, :], floor
        )
        role_indices = torch.arange(roles, device=logits.device)
        matrix[:, role_indices, candidates + role_indices] = null_logits.reshape(
            flat, roles
        )
        matrix[:, roles:, :] = 0.0
        log_assignment = matrix
        for _ in range(self.sinkhorn_iterations):
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-1, keepdim=True
            )
            log_assignment = log_assignment - torch.logsumexp(
                log_assignment, dim=-2, keepdim=True
            )
        assignment = log_assignment.exp()
        candidate_mass = assignment[:, :roles, :candidates]
        null_mass = assignment[:, role_indices, candidates + role_indices].unsqueeze(-1)
        partial = torch.cat((candidate_mass, null_mass), dim=-1)
        partial = partial / partial.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        candidate_mass = partial[..., :candidates] * flat_mask[:,None,:].to(partial.dtype)
        capacity = candidate_mass.sum(dim=-2, keepdim=True).clamp_min(1.0)
        candidate_mass = candidate_mass / capacity
        null_mass = 1.0 - candidate_mass.sum(dim=-1, keepdim=True)
        return torch.cat((candidate_mass, null_mass), dim=-1).reshape(
            batch, events, roles, candidates + 1
        )


class RichTopologyTransport(nn.Module):
    """Matched D0/U0/R1 transports over paths of length 2--8.

    D0 sees only unsigned anchor distance. U0 additionally sees whether every
    intermediate token is any registered relation primitive. R1 sees primitive
    type through one role-by-primitive kernel shared across direction and step.
    No decoder reads the candidate endpoint token identity.
    """

    def __init__(self, *, roles: int, primitives: int, max_path_length=8, sinkhorn_iterations=32) -> None:
        super().__init__()
        if not 1 <= roles <= 8 or not 1 <= primitives <= 64 or not 2 <= max_path_length <= 32:
            raise ValueError("bounded registered transport domain required")
        self.relation_tokens = tuple(range(primitives))
        self.max_path_length = max_path_length
        self.distance_bias = nn.Parameter(torch.zeros(roles, self.max_path_length + 1))
        self.role_relation_score = nn.Parameter(torch.zeros(roles, primitives))
        nn.init.normal_(self.role_relation_score, std=0.02)
        self.null_bias = nn.Parameter(torch.zeros(3, roles))
        self.projectors = nn.ModuleList(
            DynamicPartialSinkhorn(roles, sinkhorn_iterations) for _ in range(3)
        )

    def _relation_types(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        type_id = torch.full_like(tokens, -1)
        for index, token_id in enumerate(self.relation_tokens):
            type_id = torch.where(tokens == token_id, torch.full_like(type_id, index), type_id)
        return type_id, type_id >= 0

    def relation_logits(
        self,
        anchor_assignment: torch.Tensor,
        document_tokens: torch.Tensor,
        document_mask: torch.Tensor,
        event_coverage: torch.Tensor,
        filler_mask: torch.Tensor,
        *, r1_only: bool = False,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if anchor_assignment.ndim != 4:
            raise ValueError("anchor assignment must be [B,E,R,L+1]")
        batch, events, roles, lattice = anchor_assignment.shape
        if lattice < 1 or lattice > 513 or not 1 <= batch <= 64 or events > 16 or roles != self.distance_bias.shape[0]:
            raise ValueError("bounded transport lattice required")
        if not torch.isfinite(anchor_assignment).all() or (anchor_assignment < 0).any():
            raise ValueError("finite nonnegative anchor posterior required")
        if not torch.allclose(anchor_assignment.sum(-1),torch.ones_like(anchor_assignment[...,0]),atol=1e-5,rtol=0):
            raise ValueError("normalized predicted anchor posterior required")
        length = lattice - 1
        if document_tokens.shape != (batch, length) or document_mask.shape != (batch, length):
            raise ValueError("document geometry drift")
        if event_coverage.shape != (batch, events, length):
            raise ValueError("event coverage geometry drift")
        if filler_mask.shape != (batch, length):
            raise ValueError("filler mask geometry drift")
        if document_tokens.dtype not in (torch.int32,torch.int64) or ((document_tokens < -1) | (document_tokens >= len(self.relation_tokens))).any():
            raise ValueError("typed relation lattice required, never raw MiniCPM token IDs")
        if any(m.dtype!=torch.bool for m in (document_mask,event_coverage,filler_mask)):
            raise ValueError("boolean predicted coverage/masks required")
        if (event_coverage & ~document_mask[:,None,:]).any():
            raise ValueError("coverage outside observed document")
        floor = torch.tensor(-1e4, dtype=anchor_assignment.dtype, device=anchor_assignment.device)
        log_anchor = anchor_assignment[..., :length].clamp_min(1e-12).log()
        relation_types, relation_mask = self._relation_types(document_tokens)
        one_hot = F.one_hot(relation_types.clamp_min(0).long(), num_classes=len(self.relation_tokens)).to(
            anchor_assignment.dtype
        )
        one_hot = one_hot * relation_mask.unsqueeze(-1).to(one_hot.dtype)
        typed_step = torch.einsum(
            "blp,rp->brl", one_hot, F.logsigmoid(self.role_relation_score)
        )

        names=("R1",) if r1_only else ("D0", "U0", "R1")
        path_scores: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        path_valid: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        for distance in range(2, self.max_path_length + 1):
            span = length - distance
            if span <= 0:
                continue
            bridge_relation = torch.ones(
                (batch, span), dtype=torch.bool, device=document_tokens.device
            )
            bridge_typed = torch.zeros(
                (batch, roles, span), dtype=anchor_assignment.dtype, device=document_tokens.device
            )
            for step in range(1, distance):
                source = slice(step, span + step)
                bridge_relation = bridge_relation & relation_mask[:, source]
                bridge_typed = bridge_typed + typed_step[:, :, source]
            bridge_typed = bridge_typed / float(distance - 1)
            for direction in ("left", "right"):
                if direction == "left":
                    candidate_slice = slice(distance, length)
                    anchor_slice = slice(0, span)
                else:
                    candidate_slice = slice(0, span)
                    anchor_slice = slice(distance, length)
                base = log_anchor[..., anchor_slice] + self.distance_bias[
                    None, None, :, distance, None
                ]
                endpoint_valid = (
                    event_coverage[..., candidate_slice]
                    & event_coverage[..., anchor_slice]
                    & filler_mask[:, None, candidate_slice]
                    & document_mask[:, None, candidate_slice]
                )
                bridge_coverage = torch.ones_like(endpoint_valid)
                for step in range(1, distance):
                    bridge_slice = (
                        slice(step, span + step)
                        if direction == "right"
                        else slice(step, span + step)
                    )
                    bridge_coverage = bridge_coverage & event_coverage[..., bridge_slice]
                d0_valid = endpoint_valid
                typed_valid = endpoint_valid & bridge_coverage & bridge_relation[:, None, :]
                full_r1 = floor.expand(batch, events, roles, length).clone()
                full_r1[..., candidate_slice] = base + bridge_typed[:, None, :, :]
                valid_typed = torch.zeros((batch, events, length),dtype=torch.bool,device=document_tokens.device)
                valid_typed[..., candidate_slice] = typed_valid
                path_scores["R1"].append(full_r1);path_valid["R1"].append(valid_typed)
                if not r1_only:
                    full_d0 = floor.expand(batch, events, roles, length).clone()
                    full_u0 = floor.expand(batch, events, roles, length).clone()
                    full_d0[..., candidate_slice] = base;full_u0[..., candidate_slice] = base
                    valid_d0 = torch.zeros((batch, events, length),dtype=torch.bool,device=document_tokens.device)
                    valid_d0[..., candidate_slice] = d0_valid
                    path_scores["D0"].append(full_d0);path_scores["U0"].append(full_u0)
                    path_valid["D0"].append(valid_d0);path_valid["U0"].append(valid_typed)

        outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        null_anchor = anchor_assignment[..., length].clamp_min(1e-12).log()
        for decoder_index, name in enumerate(("D0", "U0", "R1")):
            if name not in names:continue
            if not path_scores[name]:
                logits=anchor_assignment.new_full((batch,events,roles,length),-1e4)
                mask=torch.zeros((batch,events,length),dtype=torch.bool,device=anchor_assignment.device)
                outputs[name]=(logits,mask,null_anchor+self.null_bias[decoder_index][None,None,:])
                continue
            stacked_scores = torch.stack(path_scores[name], dim=-1)
            stacked_valid = torch.stack(path_valid[name], dim=-1)
            safe_scores = stacked_scores.masked_fill(
                ~stacked_valid.unsqueeze(2), floor
            )
            logits = torch.logsumexp(safe_scores, dim=-1)
            candidate_mask = stacked_valid.any(dim=-1)
            logits = logits.masked_fill(~candidate_mask.unsqueeze(-2), floor)
            null_logits = null_anchor + self.null_bias[decoder_index][None, None, :]
            outputs[name] = (logits, candidate_mask, null_logits)
        return outputs

    def forward(
        self,
        anchor_assignment: torch.Tensor,
        document_tokens: torch.Tensor,
        document_mask: torch.Tensor,
        event_coverage: torch.Tensor,
        filler_mask: torch.Tensor,
        *, r1_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw = self.relation_logits(
            anchor_assignment,
            document_tokens,
            document_mask,
            event_coverage,
            filler_mask,r1_only=r1_only,
        )
        result: dict[str, torch.Tensor] = {}
        for index, name in enumerate(("D0", "U0", "R1")):
            if name not in raw:continue
            logits, mask, null_logits = raw[name]
            assignment = self.projectors[index](logits, mask, null_logits)
            result[f"{name}_logits"] = logits
            result[f"{name}_candidate_mask"] = mask
            result[f"{name}_null_logits"] = null_logits
            result[f"{name}_assignment"] = assignment
        return result



def capacity_preserving_decode(assignment):
    """Exact bounded matching for clipped log-probabilities; ties prefer NULL.

    Dynamic programming over role subsets consumes each real filler at most once.
    Input shape [1,E,R,C+1], R<=8; last column is each role's private NULL.
    """
    if assignment.ndim!=4 or assignment.shape[0]!=1 or assignment.shape[1]>16 or not 1<=assignment.shape[2]<=8 or assignment.shape[3]>513:
        raise ValueError('bounded assignment required')
    if not torch.isfinite(assignment).all() or (assignment<0).any():
        raise ValueError('invalid assignment')
    values=assignment.detach().cpu().double()
    if not torch.allclose(values.sum(-1),torch.ones_like(values[...,0]),atol=1e-5,rtol=0) or (values[...,:-1].sum(-2)>1+1e-5).any():
        raise ValueError('assignment does not satisfy partial capacity')
    _,events,roles,width=values.shape;count=width-1;out=[]
    for event in range(events):
        p=values[0,event]
        null=[math.log(max(float(p[r,count]),1e-30)) for r in range(roles)]
        dp={0:(sum(null),tuple([count]*roles))}
        for candidate in range(count):
            following=dict(dp)
            for mask,(score,chosen) in dp.items():
                for role in range(roles):
                    if mask & (1<<role) or p[role,candidate]<=0: continue
                    value=score+math.log(max(float(p[role,candidate]),1e-30))-null[role]
                    key=mask|(1<<role)
                    if key not in following or value>following[key][0]:
                        choices=list(chosen);choices[role]=candidate
                        following[key]=(value,tuple(choices))
            dp=following
        winner=max(dp.values(),key=lambda x:x[0])[1]
        out.append(winner)
    return torch.tensor(out,device=assignment.device,dtype=torch.long).reshape(1,events,roles)
