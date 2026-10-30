"""Complete observed-payload candidates for the bounded source-memory table.

A WRITE_BATCH_i is a deterministic sequence of ordinary complete-row commits,
starting at slot i and advancing cyclically. All fragments are processed; only
the final bounded rows remain readable. It is a task-specific M action domain,
not the old single-row engineering domain or its calibrated predictor.
"""
from dataclasses import dataclass
import json

from .alfworld_episode import _digest
from .memory import Memory
from .observation_memory import ObservationCodec,ObservationBatch
from .token_memory import memory_tensors

SOURCE_DOMAIN = 'orbit-complete-observation-cyclic-batch-v1'

@dataclass(frozen=True)
class SourceOption:
    action: str
    predecessor: str
    payload: str
    successor: str
    memory: Memory
    receipts: tuple


def source_options(memory, batch, codec, *, tenant, principal):
    if type(memory) is not Memory or type(codec) is not ObservationCodec or type(batch) is not ObservationBatch:
        raise TypeError('complete typed source-memory decision required')
    if memory.profile.identity != codec.profile.identity or memory.encoder.codec.identity != codec.identity:
        raise ValueError('source encoder/codec/profile binding changed')
    codec.verify_batch(batch) # Common malformed/partial payload fails BEFORE NULL.
    proposals = [codec.proposal(batch,j,tenant=tenant,principal=principal) for j in range(len(batch.chunks))]
    predecessor = memory.identity
    options = [SourceOption('NULL',predecessor,batch.identity,predecessor,memory.fork(),())]
    excluded = []
    for start in range(memory.profile.memory_slots):
        shadow = memory.fork();receipts=[]
        try:
            for j,(proposal,authority) in enumerate(proposals):
                slot=(start+j)%memory.profile.memory_slots
                receipts.append(shadow.commit(shadow.assemble(slot,proposal,authority)))
        except PermissionError as exc:
            # Protection/tenant constraints can exclude this whole slot path.
            # Numeric, format, duplicate or source errors remain failures.
            excluded.append(dict(start=start,reason=type(exc).__name__+': '+str(exc)))
            continue
        options.append(SourceOption('WRITE_BATCH_'+str(start),predecessor,batch.identity,
                                     shadow.identity,shadow,tuple(receipts)))
    if memory.identity != predecessor:raise RuntimeError('candidate preparation mutated source memory')
    return tuple(options),tuple(excluded)


def choose_source(memory, options, index):
    if type(index) is not int or not 0<=index<len(options):raise ValueError('issued source action required')
    choice=options[index]
    if (type(choice) is not SourceOption or memory.identity!=choice.predecessor
            or choice.memory.identity!=choice.successor
            or any(x.predecessor!=choice.predecessor or x.payload!=choice.payload for x in options)):
        raise ValueError('stale or mutated source decision')
    return choice.memory.fork(),dict(domain=SOURCE_DOMAIN,action=choice.action,
        payload=choice.payload,predecessor=choice.predecessor,successor=choice.successor,
        row_commits=list(choice.receipts))


def source_read(memory, *, tenant, principal, clock, device):
    """Actual production neural/token reads; never recover from an archive."""
    pairs=[];receipts=[]
    for selector in memory.selectors(tenant=tenant,principal=principal,clock=clock,view='neural'):
        value=memory.read(selector,principal=principal,clock=clock)
        if value is None:raise RuntimeError('selected source view changed during pinned read')
        # The trusted carrier authenticates provenance; observed_step also
        # prevents a legal-looking future fragment from entering a local read.
        row=memory._rows[selector.slot]
        if json.loads(row.canonical)['observed_step']>clock:
            raise ValueError('future source observation is not readable')
        pairs.append(value);receipts.append(selector.receipt)
    return memory_tensors(memory.profile,pairs,device=device),_digest(dict(memory=memory.identity,
        tenant=tenant,principal=principal,clock=clock,view='neural',rows=receipts))
