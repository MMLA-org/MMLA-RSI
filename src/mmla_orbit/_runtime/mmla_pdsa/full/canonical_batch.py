"""Batched canonical encoding: padding never changes a row's GRU state."""
import torch
from torch.nn.utils.rnn import pack_padded_sequence


def canonical_batch_tensors(model,episode_rows,*,pad_id=0):
    """List of bounded token rows per legal M; return independent key/value tables.

    This is the trainable encoder path. Runtime byte qualification still uses
    the exported deterministic FrozenCanonicalEncoder and its exact identity.
    """
    if not episode_rows:raise ValueError('nonempty batch of memories')
    flat=[];sizes=[]
    for rows in episode_rows:
        if len(rows)>model.profile.memory_slots:raise ValueError('per-episode memory capacity')
        sizes.append(len(rows))
        for ids in rows:
            if ids.ndim!=2 or ids.shape[0]!=1 or not 1<=ids.shape[1]<=model.profile.local_tokens:
                raise ValueError('one nonempty canonical token sequence')
            flat.append(ids[0])
    if not flat:return [None]*len(sizes)
    keys,values=canonical_vectors(model.mixer.get_input_embeddings(),model.consolidator,
        model.canonical_key,model.canonical_value,flat,pad_id=pad_id)
    result=[];start=0
    for size in sizes:
        item=None if not size else (keys[start:start+size],values[start:start+size])
        if item is not None and model.profile.memory_tokens:
            from .token_memory import token_table
            item=item+(token_table(model.profile,[ids.detach().cpu().tolist() for ids in flat[start:start+size]]),)
        result.append(item);start+=size
    return result


CANONICAL_ALGORITHM='packed-bidirectional-gru-fp32-on-model-device-v2'


def canonical_vectors(embedding,consolidator,key,value,token_rows,*,pad_id):
    """One shared length-aware path for training and frozen byte derivation.

    Batch shapes may still affect floating-point rounding. Only the actual
    exported reader's original exact-JSON qualification establishes capability.
    """
    if not token_rows or any(t.ndim!=1 or not len(t) for t in token_rows):
        raise ValueError('nonempty canonical token rows')
    device=key.weight.device
    if embedding.weight.device!=device:raise ValueError('canonical modules use actual model device')
    lengths=[len(row) for row in token_rows];width=max(lengths)
    assembly_device='cpu' if all(row.device.type=='cpu' for row in token_rows) else device
    ids=torch.full((len(token_rows),width),pad_id,device=assembly_device,dtype=torch.long)
    for i,row in enumerate(token_rows):ids[i,:len(row)]=row.to(assembly_device)
    ids=ids.to(device)
    with torch.no_grad():embedded=embedding(ids).detach().float()
    with torch.autocast(device_type=device.type,enabled=False):
        x=torch.tanh(consolidator.input(embedded))
        packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        _,state=consolidator.recurrent(packed)
        z=torch.cat((state[-2],state[-1]),dim=-1)
        keys=key(z);values=value(z)
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():raise FloatingPointError('nonfinite canonical encoding')
    return keys,values
