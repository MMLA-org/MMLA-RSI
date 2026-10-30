"""Bounded token-preserving capsule, stored in the same authoritative row.

The integer coordinates are an explicit additional neural representation. They
are not reconstructed from canonical bytes on the read path. All payload bytes
are covered by the existing row receipt and re-encoding conformance check.
"""
import struct
import torch

CODEC='fp32-semantic-plus-u32-length-i32-token-sequence-v1'
PREFIX_START=(130072,8448,220,26758,59038,5577,9155,990)
PREFIX_END=(220,130073,220)
PREFIX_SEPARATOR=(220,)
FORBIDDEN=(0,1,130072,130073,130074)


def pack_value(profile,semantic,tokens):
    if len(semantic)!=4*profile.neural_width:raise ValueError('exact semantic summary bytes')
    if not profile.memory_tokens:return semantic
    tokens=tuple(tokens)
    if (not 0<len(tokens)<=profile.memory_tokens or any(type(t) is not int or not 0<=t<profile.memory_vocab_size
            or t in profile.memory_forbidden_tokens for t in tokens)):
        raise ValueError('bounded ordinary canonical tokens required')
    return semantic+struct.pack('<I',len(tokens))+struct.pack('<'+'i'*profile.memory_tokens,*(tokens+(0,)*(profile.memory_tokens-len(tokens))))


def unpack_value(profile,blob):
    if type(blob) is not bytes or len(blob)!=profile.neural_value_bytes:raise ValueError('exact versioned neural value size')
    width=4*profile.neural_width;semantic=blob[:width]
    if not profile.memory_tokens:return semantic,()
    count=struct.unpack('<I',blob[width:width+4])[0]
    if count>profile.memory_tokens:raise ValueError('token capsule length')
    tokens=struct.unpack('<'+'i'*profile.memory_tokens,blob[width+4:])
    if (any(t!=0 for t in tokens[count:]) or any(not 0<=t<profile.memory_vocab_size or t in profile.memory_forbidden_tokens for t in tokens[:count])):
        raise ValueError('token capsule coordinates/padding')
    return semantic,tokens[:count]


def token_table(profile,rows):
    """CPU immutable integer rows; -1 pads tensors, zero pads serialized bytes."""
    if not profile.memory_tokens:return None
    if len(rows)>profile.memory_slots:raise ValueError('memory row cap')
    result=torch.full((len(rows),profile.memory_tokens),-1,dtype=torch.long)
    for i,tokens in enumerate(rows):
        tokens=tuple(int(t) for t in tokens)
        if (not 0<len(tokens)<=profile.memory_tokens or any(not 0<=t<profile.memory_vocab_size or t in profile.memory_forbidden_tokens for t in tokens)):
            raise ValueError('complete bounded ordinary token row')
        result[i,:len(tokens)]=torch.tensor(tokens,dtype=torch.long)
    return result


def memory_tensors(profile,pairs,*,device):
    if not pairs:return None
    rows=[unpack_value(profile,v) for k,v in pairs]
    keys=torch.tensor([struct.unpack('<'+'f'*profile.key_width,k) for k,v in pairs],device=device)
    values=torch.tensor([struct.unpack('<'+'f'*profile.neural_width,s) for s,t in rows],device=device)
    result=(keys,values)
    return result if not profile.memory_tokens else result+(token_table(profile,[t for s,t in rows]),)


def prefix_rows(profile,memory):
    """Only a pinned read tensor is consumed; no canonical/teacher access."""
    if memory.tokens is None:
        if profile.memory_tokens and memory.valid.any():raise ValueError('new profile requires complete token capsule')
        return [() for _ in range(len(memory.keys))]
    if not profile.memory_tokens:raise ValueError('token access not declared in this profile')
    tokens=memory.tokens
    if tokens.device.type!='cpu' or tokens.dtype!=torch.long or tokens.shape!=(*memory.valid.shape,profile.memory_tokens):
        raise ValueError('exact CPU integer capsule batch')
    valid=memory.valid.detach().cpu().tolist();result=[]
    for permitted,rows in zip(valid,tokens.tolist()):
        parts=[]
        for allowed,row in zip(permitted,rows):
            kept=[t for t in row if t!=-1]
            if not allowed:
                if kept:raise ValueError('padding row contains readable tokens')
                continue
            if (not kept or row[:len(kept)]!=kept or any(t!=-1 for t in row[len(kept):])
                    or any(not 0<=t<profile.memory_vocab_size or t in profile.memory_forbidden_tokens for t in kept)):
                raise ValueError('finite contiguous valid token rows')
            if parts:parts.extend(profile.memory_prefix_separator)
            parts.extend(kept)
        result.append(tuple(profile.memory_prefix_start)+tuple(parts)+tuple(profile.memory_prefix_end) if parts else ())
    return result
