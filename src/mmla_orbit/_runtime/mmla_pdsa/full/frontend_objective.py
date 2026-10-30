"""Active-event compiler objective, versioned separately from memory supervision."""
import torch
from torch.nn import functional as F
from .profile import digest

DEFINITION=dict(schema='mmla-active-event-compiler-v1',mask='slot < supervised count; never predicted count or non-NULL kind',
    reductions='kind/start/end: active event mean; anchor/transport: active event x 4 roles mean; zero count contributes zero',
    other_heads='original count CE, relation CE and filler BCE',class_weights='all one',
    aggregate='equal eight losses, then original segments/sources/families; genuine NULL remains supervised',
    topology='predicted discrete coverage/type/filler unchanged; no straight-through or oracle mask')

def feature_identity(data_identity):
    return digest(dict(data=data_identity,objective=DEFINITION))

def objective(result,targets):
    b,e,_=result['kind_logits'].shape
    count=targets['count'];mask=torch.arange(e,device=count.device)[None]<count[:,None]
    denom=count.clamp_min(1).to(result['kind_logits'].dtype)
    def ce(logits,label):
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),label.reshape(-1),reduction='none').reshape(label.shape)
    def event_mean(value):
        if value.ndim==2:return (value*mask).sum(1)/denom
        return (value*mask[:,:,None]).sum((1,2))/(denom*value.shape[2])
    losses=dict(count=ce(result['count_logits'],count),
        kind=event_mean(ce(result['kind_logits'],targets['kind'])),
        start=event_mean(ce(result['start_logits'],targets['start'])),
        end=event_mean(ce(result['end_logits'],targets['end'])),
        anchors=event_mean(ce(result['anchor_logits'],targets['anchors'])),
        relation=ce(result['relation_logits'],targets['relation']).mean(1),
        filler=F.binary_cross_entropy_with_logits(result['filler_logits'],targets['filler'].float(),reduction='none').mean(1))
    selected=result['R1_assignment'].gather(-1,targets['pointers'][...,None]).squeeze(-1)
    losses['transport']=event_mean(-selected.clamp_min(1e-12).log())
    return torch.stack(list(losses.values())).mean(0),losses
