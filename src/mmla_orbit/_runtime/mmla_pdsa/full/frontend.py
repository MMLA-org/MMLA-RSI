"""Learned closed-view compiler for a bounded control-language profile.

Pure lexicalization supplies only raw word spans and uniform bridge positions.
Count, event boundaries, kinds, anchors, relation types and fillers are predicted.
Gold annotations are accepted only by compiler_objective, never by forward.
"""
from dataclasses import dataclass
import hashlib
import re
import torch
from torch import nn
from torch.nn import functional as F
from .afrt import RichTopologyTransport,capacity_preserving_decode
from .compiler_data import KINDS,ROLES,PRIMITIVES,lexicalize
from .profile import wire


def tokenize_view(tokenizer,text,*,max_tokens=2048):
    view=lexicalize(text)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids'];offsets=encoded['offset_mapping']
    if not ids or len(ids)>max_tokens or len(offsets)!=len(ids):raise ValueError('observed token capacity/mapping')
    pool=torch.zeros((len(view['words']),len(ids)),dtype=torch.float32)
    for i,(lo,hi) in enumerate(view['spans']):
        covered=[j for j,(a,b) in enumerate(offsets) if max(a,lo)<min(b,hi)]
        if not covered:raise ValueError('raw span not represented by tokenizer')
        pool[i,covered]=1/len(covered)
    return view,torch.tensor([ids],dtype=torch.long),pool


class PredictedFrameError(ValueError):
    """Valid decoder interface, but the predicted grammar cannot be executed."""


@dataclass(frozen=True)
class TypedIR:
    kind:str
    object_id:str|None
    value:str|None
    unit:str|None
    reference:str|None
    spans:tuple
    anchor_char:int
    ordinal:int
    source_digest:str

    def __post_init__(self):
        if self.kind not in KINDS or type(self.ordinal) is not int or not 0<=self.ordinal<16 or type(self.anchor_char) is not int or self.anchor_char<0:
            raise ValueError('bounded typed operation and event identity required')
        if type(self.source_digest) is not str or re.fullmatch('[0-9a-f]{64}',self.source_digest) is None:
            raise ValueError('exact observed source digest required')
        if type(self.spans) is not tuple or len(self.spans)!=4:
            raise ValueError('one raw span or NULL per role')
        for span in self.spans:
            if span is not None and (type(span) is not tuple or len(span)!=2 or any(type(v) is not int for v in span) or not 0<=span[0]<span[1]):
                raise ValueError('bounded original-character spans required')
        for value in (self.object_id,self.value,self.unit,self.reference):
            if value is not None and (type(value) is not str or not 0<len(value.encode())<=256):
                raise ValueError('typed bounded copied string or NULL required')
        if self.kind!='null' and self.object_id is None:
            raise ValueError('non-NULL operation requires an object')

    @property
    def identity(self):return hashlib.sha256(wire(self.__dict__)).hexdigest()


class ClosedFrontend(nn.Module):
    def __init__(self,hidden,*,width=128,max_events=2,roles=4):
        super().__init__()
        if width%2 or not 1<=max_events<=16 or roles!=4:raise ValueError('registered four-role bounded frontend')
        self.max_events=max_events;self.width=width
        self.input=nn.Linear(hidden,width)
        self.encoder=nn.GRU(width,width//2,bidirectional=True,batch_first=True)
        self.event_query=nn.Parameter(torch.randn(max_events,width)*.02)
        self.role_query=nn.Parameter(torch.randn(roles,width)*.02)
        self.node_type=nn.Parameter(torch.randn(2,width)*.02)
        self.start=nn.Linear(width,width,bias=False);self.end=nn.Linear(width,width,bias=False)
        self.boundary_null=nn.Linear(width,2)
        self.count=nn.Linear(width,max_events+1);self.kind=nn.Linear(width,len(KINDS))
        self.anchor=nn.Linear(width,width,bias=False);self.anchor_null=nn.Linear(width,roles)
        self.relation=nn.Linear(width,len(PRIMITIVES)+1)
        self.filler=nn.Linear(width,1)
        self.transport=RichTopologyTransport(roles=roles,primitives=len(PRIMITIVES),max_path_length=8,sinkhorn_iterations=32)

    def forward(self,observed_embeddings,lexeme_pool):
        if (observed_embeddings.ndim!=3 or observed_embeddings.shape[0]!=1
                or lexeme_pool.ndim!=2 or lexeme_pool.shape[1]!=observed_embeddings.shape[1]
                or not 1<=lexeme_pool.shape[0]<=256 or not torch.isfinite(lexeme_pool).all()
                or (lexeme_pool<0).any() or not torch.allclose(lexeme_pool.sum(-1),torch.ones_like(lexeme_pool[:,0]))):
            raise ValueError('one bounded complete observed view and pure lexical pool required')
        words=lexeme_pool.to(observed_embeddings.device)@observed_embeddings[0].float()
        encoded,_=self.encoder(torch.tanh(self.input(words))[None])
        words=encoded[0];n=len(words);length=2*n-1
        mean=words.mean(0);queries=self.event_query+mean
        weight=(queries@words.T/(self.width**.5)).softmax(-1)
        event=weight@words+self.event_query
        start=event@self.start(words).T/(self.width**.5)
        end=event@self.end(words).T/(self.width**.5)
        null=self.boundary_null(event)
        start=torch.cat((start,null[:,0,None]),-1);end=torch.cat((end,null[:,1,None]),-1)
        nodes=[]
        for i in range(n):
            if i:nodes.append((words[i-1]+words[i])*.5+self.node_type[1])
            nodes.append(words[i]+self.node_type[0])
        nodes=torch.stack(nodes)
        relation_logits=self.relation(nodes)
        typed=relation_logits.argmax(-1);typed=torch.where(typed==len(PRIMITIVES),-1,typed)
        role=event[:,None,:]+self.role_query[None]
        raw_anchor=torch.einsum('erw,lw->erl',role,self.anchor(nodes))/(self.width**.5)
        positions=torch.arange(length,device=nodes.device)
        raw_anchor=raw_anchor.masked_fill((positions%2==1)[None,None,:],-1e4)
        anchor_logits=torch.cat((raw_anchor,self.anchor_null(event)[:,:,None]),-1)
        anchor_assignment=anchor_logits.softmax(-1)[None]
        predicted_start=start.argmax(-1);predicted_end=end.argmax(-1)
        active=(torch.arange(self.max_events,device=nodes.device)<self.count(mean).argmax())
        coverage=((positions[None,:]>=2*predicted_start[:,None])
            &(positions[None,:]<=2*predicted_end[:,None])
            &(predicted_start<=predicted_end)[:,None]&(predicted_end<n)[:,None]&active[:,None])[None]
        filler_logits=self.filler(words).squeeze(-1)
        filler=torch.zeros(length,dtype=torch.bool,device=words.device);filler[::2]=filler_logits>=0
        transport=self.transport(anchor_assignment,typed[None],torch.ones((1,length),device=words.device,dtype=torch.bool),coverage,filler[None])
        return dict(count_logits=self.count(mean),kind_logits=self.kind(event),start_logits=start,end_logits=end,
            anchor_logits=anchor_logits,relation_logits=relation_logits,filler_logits=filler_logits,
            coverage=coverage,typed=typed,filler=filler,event_features=event,word_features=words,**transport)

    def forward_batch(self,observed_embeddings,lexeme_pool,*,r1_only=True):
        """Equal T/N batches only; no padded words or added Sinkhorn candidates.

        This shared operator preserves gradients to its input. Frozen-input
        caching, when used, belongs to the explicitly frozen module curriculum.
        """
        if (observed_embeddings.ndim!=3 or lexeme_pool.ndim!=3
                or lexeme_pool.shape[0]!=observed_embeddings.shape[0]
                or lexeme_pool.shape[2]!=observed_embeddings.shape[1]
                or not 1<=observed_embeddings.shape[0]<=64 or not 1<=lexeme_pool.shape[1]<=256
                or not torch.isfinite(lexeme_pool).all() or (lexeme_pool<0).any()
                or not torch.allclose(lexeme_pool.sum(-1),torch.ones_like(lexeme_pool[:,:,0]))):
            raise ValueError('complete equal-length observed views and lexical pools required')
        words=lexeme_pool.to(observed_embeddings.device)@observed_embeddings.float()
        words,_=self.encoder(torch.tanh(self.input(words)))
        batch,n,_=words.shape;length=2*n-1
        mean=words.mean(1);queries=self.event_query[None]+mean[:,None]
        weight=(queries@words.transpose(1,2)/(self.width**.5)).softmax(-1)
        event=weight@words+self.event_query[None]
        start=event@self.start(words).transpose(1,2)/(self.width**.5)
        end=event@self.end(words).transpose(1,2)/(self.width**.5)
        null=self.boundary_null(event)
        start=torch.cat((start,null[:,:,0,None]),-1);end=torch.cat((end,null[:,:,1,None]),-1)
        nodes=words.new_empty((batch,length,self.width))
        nodes[:,::2]=words+self.node_type[0]
        nodes[:,1::2]=(words[:,:-1]+words[:,1:])*.5+self.node_type[1]
        relation_logits=self.relation(nodes)
        typed=relation_logits.argmax(-1);typed=torch.where(typed==len(PRIMITIVES),-1,typed)
        role=event[:,:,None,:]+self.role_query[None,None]
        raw_anchor=torch.einsum('berw,blw->berl',role,self.anchor(nodes))/(self.width**.5)
        positions=torch.arange(length,device=nodes.device)
        raw_anchor=raw_anchor.masked_fill((positions%2==1)[None,None,None,:],-1e4)
        anchor_logits=torch.cat((raw_anchor,self.anchor_null(event)[:,:,:,None]),-1)
        anchor_assignment=anchor_logits.softmax(-1)
        predicted_start=start.argmax(-1);predicted_end=end.argmax(-1)
        count_logits=self.count(mean)
        active=(torch.arange(self.max_events,device=nodes.device)[None,:]<count_logits.argmax(-1)[:,None])
        coverage=((positions[None,None,:]>=2*predicted_start[:,:,None])
            &(positions[None,None,:]<=2*predicted_end[:,:,None])
            &(predicted_start<=predicted_end)[:,:,None]&(predicted_end<n)[:,:,None]&active[:,:,None])
        filler_logits=self.filler(words).squeeze(-1)
        filler=torch.zeros((batch,length),dtype=torch.bool,device=words.device);filler[:,::2]=filler_logits>=0
        transport=self.transport(anchor_assignment,typed,torch.ones((batch,length),device=words.device,dtype=torch.bool),
            coverage,filler,r1_only=r1_only)
        return dict(count_logits=count_logits,kind_logits=self.kind(event),start_logits=start,end_logits=end,
            anchor_logits=anchor_logits,relation_logits=relation_logits,filler_logits=filler_logits,
            coverage=coverage,typed=typed,filler=filler,event_features=event,word_features=words,**transport)

    def decode(self,result,view):
        words=view['words'];n=len(words);length=2*n-1
        if result['filler_logits'].shape!=(n,):raise ValueError('result/view mismatch')
        count=int(result['count_logits'].argmax());assignment=capacity_preserving_decode(result['R1_assignment'])[0]
        events=[];previous=-1
        source=hashlib.sha256(view['text'].encode()).hexdigest()
        for i in range(count):
            start=int(result['start_logits'][i].argmax());end=int(result['end_logits'][i].argmax())
            if not previous<start<=end<n:raise PredictedFrameError('invalid ordered predicted event boundaries')
            previous=end;values=[];spans=[]
            for p in assignment[i].tolist():
                if p==length:values.append(None);spans.append(None);continue
                if p%2 or not 2*start<=p<=2*end:raise PredictedFrameError('pointer outside predicted raw event')
                lo,hi=view['spans'][p//2];values.append(view['text'][lo:hi]);spans.append((lo,hi))
            kind=KINDS[int(result['kind_logits'][i].argmax())]
            if kind!='null' and values[0] is None:raise PredictedFrameError('missing predicted object; not a valid NULL event')
            events.append(TypedIR(kind,*values,tuple(spans),view['spans'][start][0],i,source))
        return tuple(events)


def compiler_objective(result,targets,*,active_events=False):
    """CE/BCE heads and R1 assignment loss. Discrete masks have no gradient.

    Separate supervised heads train those choices. This is not an estimator for
    gradient through discrete topology or event-count selection.
    """
    if active_events:
        from .frontend_batch import target_tensors
        from .frontend_objective import objective
        n=len(result['filler_logits']);e=len(result['kind_logits'])
        labels={k:v[None] for k,v in target_tensors(targets,n,e,result['count_logits'].device).items()}
        batch={k:(v if k=='coverage' or k.startswith(('D0_','U0_','R1_')) else v[None]) for k,v in result.items()}
        total,parts=objective(batch,labels)
        return total[0],{k:v[0] for k,v in parts.items()}
    device=result['count_logits'].device;n=len(result['filler_logits']);length=2*n-1
    count=targets['count'];events=targets['events'];max_events=len(result['kind_logits'])
    if not 0<=count<=max_events or len(events)!=count:raise ValueError('complete bounded target events')
    kind=[0]*max_events;start=[n]*max_events;end=[n]*max_events
    anchors=[[length]*4 for _ in range(max_events)];pointers=[[length]*4 for _ in range(max_events)]
    for i,event in enumerate(events):
        kind[i]=event['kind'];start[i]=event['start'];end[i]=event['end']
        anchors[i]=event['anchors'];pointers[i]=event['pointers']
    tensor=lambda x:torch.tensor(x,device=device,dtype=torch.long)
    losses={'count':F.cross_entropy(result['count_logits'][None],tensor([count])),
        'kind':F.cross_entropy(result['kind_logits'],tensor(kind)),
        'start':F.cross_entropy(result['start_logits'],tensor(start)),
        'end':F.cross_entropy(result['end_logits'],tensor(end)),
        'anchors':F.cross_entropy(result['anchor_logits'].reshape(-1,length+1),tensor(anchors).reshape(-1)),
        'relation':F.cross_entropy(result['relation_logits'],tensor(targets['relation'])),
        'filler':F.binary_cross_entropy_with_logits(result['filler_logits'],tensor(targets['filler']).float())}
    assignment=result['R1_assignment'][0]
    selected=assignment.gather(-1,tensor(pointers)[...,None]).squeeze(-1)
    losses['transport']=-selected.clamp_min(1e-12).log().mean()
    return torch.stack(list(losses.values())).mean(),losses
