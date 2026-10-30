"""Equal-shape compiler training shared by single and distributed module trainers."""
from collections import OrderedDict
import torch
from torch.nn import functional as F
from .frontend import tokenize_view
from .profile import digest,wire


class FrozenCompilerInputs:
    """Fit-only static inputs. Explicitly refuses a trainable or changed embedding.

    This cache is for the frozen-substrate module recipe, not pretraining.
    Learned frontend features, assignments and loss graphs are never cached.
    """
    def __init__(self,max_bytes=512*1024**2):
        self.rows=OrderedDict();self.bytes=0;self.max_bytes=max_bytes;self.binding=None

    def bind(self,model,formatter):
        embedding=model.mixer.get_input_embeddings().weight
        if embedding.requires_grad:raise ValueError('frozen compiler cache cannot be used for trainable embeddings')
        current=(id(embedding),embedding._version,embedding.data_ptr(),str(embedding.device),str(embedding.dtype),
            tuple(embedding.shape),digest(model.substrate_identity),formatter.identity)
        if self.binding is None:self.binding=current
        elif self.binding!=current:raise ValueError('cached frozen input identity changed')

    def get(self,model,formatter,segment):
        self.bind(model,formatter)
        text=segment['observation']['text'];key=digest(dict(text=text,targets=segment['targets']))
        if key not in self.rows:
            observed,ids,pool=tokenize_view(formatter.tokenizer,text,max_tokens=model.profile.local_tokens)
            if wire(observed)!=wire(segment['observation']):raise ValueError('compiler labels changed lexical observation')
            device=model.rho.a.device
            with torch.no_grad():embedding=model.mixer.get_input_embeddings()(ids.to(device)).detach()
            row=dict(embedding=embedding,pool=pool.to(device),targets=target_tensors(segment['targets'],len(pool),model.frontend.max_events,device))
            size=sum(t.numel()*t.element_size() for t in (row['embedding'],row['pool'],*row['targets'].values()))
            if size>self.max_bytes:raise ValueError('one frozen input exceeds cache budget')
            while self.rows and (self.bytes+size>self.max_bytes or len(self.rows)>=2048):
                _,old=self.rows.popitem(last=False);self.bytes-=old['bytes']
            row['bytes']=size;self.rows[key]=row;self.bytes+=size
        self.rows.move_to_end(key)
        return key,self.rows[key]


def target_tensors(targets,n,max_events,device):
    count=targets['count'];events=targets['events'];length=2*n-1
    if not 0<=count<=max_events or len(events)!=count:raise ValueError('complete bounded target events')
    kind=[0]*max_events;start=[n]*max_events;end=[n]*max_events
    anchors=[[length]*4 for _ in range(max_events)];pointers=[[length]*4 for _ in range(max_events)]
    for i,e in enumerate(events):
        kind[i]=e['kind'];start[i]=e['start'];end[i]=e['end'];anchors[i]=e['anchors'];pointers[i]=e['pointers']
    return {key:torch.tensor(value,device=device,dtype=torch.long) for key,value in
        dict(count=count,kind=kind,start=start,end=end,anchors=anchors,pointers=pointers,
            relation=targets['relation'],filler=targets['filler']).items()}


def compiler_objective_batch(result,targets,*,active_events=False):
    """Same eight per-segment means, retaining each original denominator."""
    if active_events:
        from .frontend_objective import objective
        return objective(result,targets)
    b,e,k=result['kind_logits'].shape;length=result['relation_logits'].shape[1]
    ce=lambda logits,label:F.cross_entropy(logits.reshape(-1,logits.shape[-1]),label.reshape(-1),reduction='none').reshape(b,-1).mean(-1)
    losses=dict(count=ce(result['count_logits'],targets['count']),kind=ce(result['kind_logits'],targets['kind']),
        start=ce(result['start_logits'],targets['start']),end=ce(result['end_logits'],targets['end']),
        anchors=ce(result['anchor_logits'],targets['anchors']),relation=ce(result['relation_logits'],targets['relation']),
        filler=F.binary_cross_entropy_with_logits(result['filler_logits'],targets['filler'].float(),reduction='none').mean(-1))
    selected=result['R1_assignment'].gather(-1,targets['pointers'][...,None]).squeeze(-1)
    losses['transport']=-selected.clamp_min(1e-12).log().mean((1,2))
    return torch.stack(list(losses.values())).mean(0),losses


def slice_result(result,index):
    """Public singleton layout for exact discrete/readout comparisons."""
    return {k:(v[index:index+1] if k=='coverage' or k.startswith(('D0_','U0_','R1_')) else v[index]) for k,v in result.items()}


def grouped_frontend_loss(data,model,families,*,active_events=False):
    cache=data._frontend_inputs;cache.bind(model,data.formatter)
    unique={};order=[];buckets={}
    for family in families:
        views=[]
        for view in data._views(family):
            labels=data.dataset.compiler_targets(view);keys=[]
            for segment in labels['segments']:
                key,row=cache.get(model,data.formatter,segment);keys.append(key)
                if key not in unique:
                    unique[key]=row
                    shape=(row['embedding'].shape[1],row['pool'].shape[0])
                    buckets.setdefault(shape,[]).append(key)
            views.append(keys)
        order.append(views)
    computed={}
    # Same token AND word lengths. Each row retains its exact Sinkhorn lattice.
    for keys in buckets.values():
        for offset in range(0,len(keys),24):
            subset=keys[offset:offset+24];rows=[unique[key] for key in subset]
            emb=torch.cat([row['embedding'] for row in rows]);pool=torch.stack([row['pool'] for row in rows])
            prediction=model.frontend.forward_batch(emb,pool,r1_only=True)
            targets={k:torch.stack([row['targets'][k] for row in rows]) for k in rows[0]['targets']}
            losses,_=compiler_objective_batch(prediction,targets,active_events=active_events)
            computed.update(zip(subset,losses.unbind()))
    # Recover occurrences before averaging: bucket/dedup counts never reweight tasks.
    return torch.stack([torch.stack([torch.stack([computed[key] for key in keys]).mean() for keys in views]).mean() for views in order])
