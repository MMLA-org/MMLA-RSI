"""Finite checks with one host synchronization per participating device."""
import torch


def all_finite(tensors):
    grouped={}
    for value in tensors:
        if value is not None:
            grouped.setdefault(value.device,[]).append(torch.isfinite(value.detach()).all())
    return all(bool(torch.stack(flags).all()) for flags in grouped.values())
