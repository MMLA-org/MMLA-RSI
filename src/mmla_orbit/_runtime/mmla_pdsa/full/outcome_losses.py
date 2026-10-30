"""Memory-bounded full-vocabulary task CE and frozen-SFT KL.

One causal trunk pass per batch; only scored predecessor states reach the head.
Checkpointed head chunks avoid retaining B*T*vocabulary probability tensors.
Frozen layer weights still propagate gradients to input Phi and shared hybrid.
"""
from contextlib import contextmanager, nullcontext
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .batching import read_memory, inject_residual
from .token_memory import prefix_rows


@contextmanager
def _recompute(on_work, kind, data):
    if on_work is not None:on_work(dict(kind=kind,**data))
    yield


def _positions(ids, mask, labels):
    if (ids.ndim != 2 or ids.shape != mask.shape or ids.shape != labels.shape
            or ids.dtype != torch.long or labels.dtype != torch.long
            or not torch.all((mask == 0) | (mask == 1)) or not mask.any(-1).all()):
        raise ValueError('nonempty independently masked integer rows required')
    valid = labels.ne(-100)
    predecessor = torch.zeros_like(valid);predecessor[:, 1:] = mask[:, :-1].bool()
    if (valid & (~mask.bool() | ~predecessor)).any() or not valid[:, 1:].any(-1).all():
        raise ValueError('each target has its own visible causal predecessor')
    row, target = valid.nonzero(as_tuple=True)
    return row, target-1, labels[row, target], valid.sum(-1)


def outcome_hidden(model, ids, mask, *, fast_states=None, memory=None, checkpoint_layers=True, on_work=None):
    """Exact retained-layer hybrid graph, with optional recomputation at backward.

    Only this new training path enables per-layer checkpointing; generation and
    the already delivered forward remain unchanged. No persistent hidden cache.
    """
    from transformers.masking_utils import create_causal_mask
    from .work_events import record_explicit_mixer
    model.validate_hybrid()
    if (model.training or model.mixer.training or model.mixer.config.attention_dropout != 0
            or type(checkpoint_layers) is not bool or ids.ndim != 2 or ids.shape != mask.shape
            or ids.shape[1] > model.profile.local_tokens
            or not torch.all((mask == 0) | (mask == 1)) or not mask.any(-1).all()):
        raise ValueError('fixed eval graph, no dropout and complete bounded masks required')
    prefix_width = 0
    if memory is not None:
        if memory.keys.shape[1] > model.profile.memory_slots:
            raise ValueError('source memory capacity exceeded')
        rows = prefix_rows(model.profile, memory);prefix_width = max(map(len, rows), default=0)
        if prefix_width:
            prefix = torch.tensor([[0]*(prefix_width-len(x))+list(x) for x in rows], device=ids.device, dtype=ids.dtype)
            prefix_mask = torch.tensor([[0]*(prefix_width-len(x))+[1]*len(x) for x in rows], device=ids.device, dtype=mask.dtype)
            ids = torch.cat((prefix, ids), 1);mask = torch.cat((prefix_mask, mask), 1)
    if ids.shape[1] > model.profile.local_tokens:
        raise ValueError('complete source memory and task text exceed context; no truncation')
    with torch.no_grad():embedded = model.mixer.get_input_embeddings()(ids).detach()
    if memory is not None:
        embedded = read_memory(model.readout, embedded, memory, top_k=model.profile.top_k)
    h = inject_residual(model.rho, embedded, fast_states, first_order_outer=False)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    trunk = model.mixer.model
    causal = create_causal_mask(config=trunk.config, inputs_embeds=h, attention_mask=mask,
                               past_key_values=None, position_ids=positions)
    rotary = trunk.rotary_emb(h, position_ids=positions)
    record_explicit_mixer(model.mixer, mask)
    work=dict(rows=ids.shape[0],width=ids.shape[1],valid_tokens=int(mask.sum()),padded_positions=mask.numel())
    if on_work is not None:on_work(dict(kind='student_trunk_forward',**work))
    for i, layer in enumerate(trunk.layers):
        branch = model.hybrid.blocks[str(i)] if str(i) in model.hybrid.blocks else None
        # Bind current objects as defaults: checkpoint recomputation occurs
        # after this loop advances and must not use the final layer's closure.
        def block(value, layer=layer, branch=branch):
            value = layer(value, attention_mask=causal, position_embeddings=rotary,
                          position_ids=positions, past_key_values=None, use_cache=False)
            return value if branch is None else branch(value, mask, positions, memory)
        h = checkpoint(block, h, use_reentrant=False,
            context_fn=lambda i=i:(nullcontext(),_recompute(on_work,'student_layer_recompute_entered',dict(layer=i,**work)))) if checkpoint_layers and torch.is_grad_enabled() else block(h)
    return trunk.norm(h)[:, prefix_width:]


def _head_losses(student_head, hidden, target, *, teacher_head=None, teacher_hidden=None):
    logits = student_head(hidden).float()
    ce = F.cross_entropy(logits, target, reduction='none')
    if teacher_head is None:return ce, torch.zeros_like(ce)
    with torch.no_grad():
        teacher_log = teacher_head(teacher_hidden).float().log_softmax(-1)
        teacher_p = teacher_log.exp()
    kl = (teacher_p*(teacher_log-logits.log_softmax(-1))).sum(-1)
    return ce, kl


def outcome_losses(model, ids, mask, labels, *, fast_states=None, memory=None,
                   teacher=None, head_chunk=32, checkpoint_layers=True, checkpoint_head=True, on_work=None):
    """Per-sequence mean CE and KL(P_SFT || P_student), temperature one.

    Teacher is allowed only for the empty-memory slow-initialization general
    objective. It sees the exact same full conversation and target positions.
    The run must bind its exact SFT revision/weights, not the student substrate.
    """
    if type(head_chunk) is not int or not 1 <= head_chunk <= 1024 or type(checkpoint_head) is not bool:
        raise ValueError('finite computational head chunk required')
    row, pos, target, counts = _positions(ids, mask, labels)
    if teacher is not None and (fast_states is not None or memory is not None
            or teacher.training or any(p.requires_grad for p in teacher.parameters())
            or teacher.config.vocab_size != model.mixer.config.vocab_size):
        raise ValueError('frozen matching-vocabulary SFT teacher on empty slow state only')
    hidden = outcome_hidden(model, ids, mask, fast_states=fast_states, memory=memory,
                            checkpoint_layers=checkpoint_layers,on_work=on_work)[row, pos]
    reference = None
    if teacher is not None:
        from .work_events import record_explicit_mixer
        record_explicit_mixer(teacher,mask)
        if on_work is not None:on_work(dict(kind='teacher_trunk_forward',rows=ids.shape[0],width=ids.shape[1],
            valid_tokens=int(mask.sum()),padded_positions=mask.numel()))
        with torch.no_grad():
            positions = (mask.cumsum(-1)-1).clamp_min(0)
            reference = teacher.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                      use_cache=False, return_dict=True).last_hidden_state[row, pos].detach()
    ce_parts, kl_parts = [], []
    for start in range(0, len(row), head_chunk):
        stop = min(len(row), start+head_chunk)
        hh, yy = hidden[start:stop], target[start:stop]
        tt = None if reference is None else reference[start:stop]
        def loss(hh, yy, tt):
            if on_work is not None:on_work(dict(kind='head_call',scored_positions=len(yy),
                vocabulary=model.mixer.config.vocab_size,teacher=teacher is not None))
            return _head_losses(model.mixer.lm_head, hh, yy,
                                teacher_head=None if teacher is None else teacher.lm_head, teacher_hidden=tt)
        ce, kl = checkpoint(loss, hh, yy, tt, use_reentrant=False) if checkpoint_head and torch.is_grad_enabled() else loss(hh, yy, tt)
        ce_parts.append(ce);kl_parts.append(kl)
    ce = torch.zeros(ids.shape[0], device=ids.device).scatter_add(0, row, torch.cat(ce_parts))/counts
    kl = torch.zeros_like(ce).scatter_add(0, row, torch.cat(kl_parts))/counts
    return ce, kl
