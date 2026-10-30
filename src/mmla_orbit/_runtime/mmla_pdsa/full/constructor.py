"""Learned target-conditioned semantic constructor; no write authority here."""
from dataclasses import dataclass
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .compiler_data import KINDS
from .memory import canonical
from .frontend import TypedIR


@dataclass(frozen=True)
class SemanticIntent:
    kind:str
    object_id:str|None
    content:bytes|None
    reference:str|None


class SlotConstructor(nn.Module):
    """G(event, actual old row, current-M summary, slot/causal metadata).

    Binary field sources are OLD/EVENT. The head selects intent and field
    sources; deterministic copying constructs a complete bounded value. Host
    code separately supplies ACL, expiry, versions, protection and receipts.
    """
    def __init__(self,width):
        super().__init__();self.width=width
        self.network=nn.Sequential(nn.Linear(3*width+8,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.feasible=nn.Linear(width,1);self.kind=nn.Linear(width,len(KINDS))
        self.value=nn.Linear(width,2);self.unit=nn.Linear(width,2)

    def forward(self,event,old_rows,current_memory,metadata):
        if (event.ndim!=1 or event.shape[0]!=self.width or old_rows.ndim!=2
                or old_rows.shape[1]!=self.width or current_memory.shape!=event.shape
                or metadata.shape!=(len(old_rows),8) or not 1<=len(old_rows)<=1024
                or not all(torch.isfinite(x).all() for x in (event,old_rows,current_memory,metadata))):
            raise ValueError('finite bounded target-specific G inputs required')
        count=len(old_rows)
        h=self.network(torch.cat((event.expand(count,-1),old_rows,current_memory.expand(count,-1),metadata),-1))
        return dict(feasible=self.feasible(h).squeeze(-1),kind=self.kind(h),
                    value_source=self.value(h),unit_source=self.unit(h))

    @staticmethod
    def semantic(result,index,ir,old_content):
        if type(ir) is not TypedIR:raise ValueError('validated typed event required')
        if type(index) is not int or not 0<=index<len(result['feasible']):raise ValueError('constructor target index')
        if not all(torch.isfinite(x[index]).all() for x in result.values()):raise ValueError('nonfinite constructor output')
        if float(result['feasible'][index])<0:return None
        kind=KINDS[int(result['kind'][index].argmax())]
        if kind=='null':return None
        if not ir.object_id:raise ValueError('non-NULL intent requires predicted object')
        content=None
        if kind in ('create','update_value','update_unit','repair'):
            old={} if old_content is None else json.loads(old_content)
            if type(old) is not dict:raise ValueError('typed old content required')
            sources=(int(result['value_source'][index].argmax()),int(result['unit_source'][index].argmax()))
            if sources[0]:
                if ir.value is None:raise ValueError('missing predicted numeric filler')
                value=json.loads(ir.value)
            else:value=old.get('value')
            unit=ir.unit if sources[1] else old.get('unit')
            if type(value) not in (int,float) or not math.isfinite(value) or unit not in ('units','kilounits'):
                raise ValueError('incomplete or invalid constructed record')
            content=canonical(dict(value=value,unit=unit))
        if kind=='alias' and not ir.reference:raise ValueError('alias missing predicted reference')
        return SemanticIntent(kind,ir.object_id,content,ir.reference if kind=='alias' else None)


def constructor_objective(result,targets):
    """Teacher fields select losses only; no target is a constructor feature."""
    device=result['kind'].device;n=len(result['feasible'])
    if len(targets)!=n:raise ValueError('one target per actual candidate slot')
    feasible=torch.tensor([x['feasible'] for x in targets],device=device,dtype=torch.float32)
    if not torch.all((feasible==0)|(feasible==1)):raise ValueError('binary feasible target')
    loss=F.binary_cross_entropy_with_logits(result['feasible'],feasible)
    active=feasible.bool()
    kinds=torch.tensor([x['kind'] for x in targets],device=device,dtype=torch.long)
    if active.any():loss=loss+F.cross_entropy(result['kind'][active],kinds[active])
    content=active & torch.tensor([KINDS[x['kind']] in ('create','update_value','update_unit','repair') for x in targets],device=device)
    if content.any():
        for name in ('value_source','unit_source'):
            indices=torch.tensor([x[name] for x in targets],device=device,dtype=torch.long)
            loss=loss+F.cross_entropy(result[name][content],indices[content])
    return loss
