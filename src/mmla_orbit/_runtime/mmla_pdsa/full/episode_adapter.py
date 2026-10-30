"""Trusted finite engineering task adapter, separate from model inputs.

Original r1 remains immutable. Early/late feedback are two fixed timelines of
one source presentation, never new independent families. Early feedback is at
world clock 0; the old script follows at its original clocks 1..3, then query 4.
"""
from dataclasses import dataclass
import copy,json,random
import torch
from .data import Truth,QUERIES
from .profile import digest,wire
from .frames import ObservedCommand
from .construction import Envelope

TIMINGS=('early','late')


@dataclass(frozen=True)
class FamilyTask:
    group:str
    views:tuple


class FamilySchedule:
    def __init__(self,dataset,views,*,seed=20261015,global_batch=24):
        if dataset.family_manifest is None or type(global_batch) is not int or global_batch!=24:
            raise ValueError('production requires reviewed 24-family grouping')
        self.dataset=dataset;self.seed=seed;self.global_batch=global_batch
        families={}
        for view in views:
            dataset._validate_view(view)
            if view.split!='fit':raise PermissionError('training schedule may only read fit views')
            families.setdefault(view.semantic_group,[]).append(view)
        expected={k for k,v in dataset.family_manifest['families'].items() if v=='fit'}
        if set(families)!=expected or len(families)!=31:raise ValueError('complete 31 fit families required')
        for group,members in families.items():
            expected_ids={k for k,v in dataset.family_manifest['mapping'].items() if v['family']==group}
            if len(members)!=8 or {v.identity for v in members}!=expected_ids:raise ValueError('eight actual unique source presentations per family')
        self.tasks={g:FamilyTask(g,tuple(sorted(v,key=lambda x:x.identity))) for g,v in families.items()}
        self.identity=digest(dict(schema='mmla-24-family-schedule-v1',dataset=dataset.identity,
            members={g:[v.identity for v in self.tasks[g].views] for g in sorted(self.tasks)},
            seed=seed,global_batch=global_batch,timings=TIMINGS,
            timeline_rule='each consecutive pair of updates uses identical selected families, early then late'))

    def batch(self,step,*,rank,size):
        if type(step) is not int or step<0 or size not in (1,4,8) or not 0<=rank<size:
            raise ValueError('fixed 1/4/8-rank finite schedule position')
        selected=random.Random(self.seed+step//2).sample(sorted(self.tasks),self.global_batch)
        # Complete families stay on one rank; views are not independent groups.
        return tuple(self.tasks[g] for g in selected[rank::size]),TIMINGS[step%2]


class EngineeringTask:
    def __init__(self,dataset,view,*,timing):
        if timing not in TIMINGS:raise ValueError('fixed early/late feedback timing')
        self.dataset,self.view,self.timing=dataset,view,timing
        self.row=dataset._validate_view(view)
        self.identity=digest(dict(source=view.identity,timing=timing,dataset=dataset.identity))
        self.group=view.semantic_group
        self.support_clock=0 if timing=='early' else self.row['observations']['query_clock']
        self.names=self.row['annotations']['names']
        self.source_events=tuple(e for seg in self.row['observations']['segments'][1:] for e in seg)
        self.operations=tuple(self.row['template']['script'])

    def truth(self,*,completed_events,world_clock):
        if type(completed_events) is not int or not 0<=completed_events<=len(self.operations):raise ValueError('finite actual event prefix')
        if type(world_clock) is not int or not completed_events<=world_clock<=4:raise ValueError('truth clock outside registered timeline')
        t=Truth(self.row['template']['context'],copy.deepcopy(self.row['annotations']['values']))
        for op,event in zip(self.operations[:completed_events],self.source_events):t.execute(op,event['authority'])
        t.clock=world_clock
        return t

    def question(self,query):
        n=self.names
        if query not in QUERIES:raise ValueError('fixed four query forms')
        return dict(read_a=f'As reader, report the current value and unit of {n["A"]}, or null if unavailable.',
            read_b=f'As reader, report the current value and unit of {n["B"]}, or null if unavailable.',
            inventory='As reader, list the available records with their current values and units.',
            equal_ab=f'As reader, do {n["A"]} and {n["B"]} have equal values and units? Return null if either is unavailable.')[query]

    def answer(self,query,*,completed_events,world_clock):
        value=self.truth(completed_events=completed_events,world_clock=world_clock).answer(query)
        return {self.names[k]:v for k,v in value.items()} if query=='inventory' else value

    def support_query(self):return 'read_b' if self.row['template']['query']!='read_b' else 'read_a'

    def support_target(self):
        return self.answer(self.support_query(),completed_events=0 if self.timing=='early' else len(self.operations),world_clock=self.support_clock)

    def query_target(self):
        return self.answer(self.row['template']['query'],completed_events=len(self.operations),world_clock=4)

    def commands(self,index):
        """Authority semantics come from this registered host protocol, not labels.

        restrict grants owner-only ACL; expire grants its source-clock endpoint.
        Neural text cannot invent either grant or bypass the original allow list.
        """
        row=self.dataset._validate_view(self.view)
        segment=row['observations']['segments'][index];result=[]
        for i,event in enumerate(segment):
            a=event['authority'];allowed=tuple(a['allowed']);clock=a.get('source_clock',0)
            acl=('owner',) if allowed==('restrict',) else None if a.get('acl') is None else tuple(a['acl'])
            end=clock if allowed==('expire',) else a.get('valid_until')
            receipt=digest(dict(dataset=self.dataset.identity,source=self.view.identity,segment=index,event=i,observed=event))
            env=Envelope(a['tenant'],a['principal'],a['object'],receipt,clock,allowed,
                acl,end,a.get('override_protection',False),False)
            result.append(ObservedCommand(event['text'],env))
        return tuple(result)

    def release_feedback(self,problem):
        from .runtime import Problem
        formatter=self.dataset.formatter
        if type(problem) is not Problem or formatter is None:raise ValueError('actual live task and bound formatter required')
        self.dataset._validate_view(self.view)
        required=1 if self.timing=='early' else len(self.row['observations']['segments'])
        frames=[r for r in problem.receipts if r.get('phase')=='FRAME']
        if len(frames)!=required or problem.segment.number!=required or problem.segment._pending is not None:
            raise PermissionError('required source observation frames have not completed')
        for index,receipt in enumerate(frames):
            expected_sources=[c.envelope.source_receipt for c in self.commands(index)]
            if (receipt.get('status')!='COMPLETED' or receipt.get('source_frame_number')!=index+1
                    or receipt.get('source_receipts')!=expected_sources):
                raise PermissionError('feedback requires the bound original source frame sequence')
        pending=problem.pending;expected=formatter.prompt(self.question(self.support_query()))
        if (pending is None or problem.attempts!=1 or pending.number!=1 or problem.failed or problem.closed or problem.problem_id!=self.identity
                or problem.tenant!='engineering' or problem.principal!='reader'
                or pending.prompt!=expected or pending.world_clock!=self.support_clock
                or pending.memory!=problem.memory.identity or pending.phi!=problem.fast.identity):
            raise PermissionError('correct own pending support attempt required before feedback release')
        return copy.deepcopy(self.support_target())

    def teacher_rows(self,*,completed_events,world_clock,principal='reader'):
        """Loss/teacher only: true visible rows, including explicit alias resolution.

        This never populates production memory or substitutes for G predictions.
        """
        t=self.truth(completed_events=completed_events,world_clock=world_clock);rows=[]
        for name in sorted(t.rows):
            value=t.visible(name,principal=principal)
            if value is not None:rows.append(wire(dict(object_id=self.names[name],**value)))
        return tuple(rows)


def supervised_sequence(formatter,question,answer,*,eos_id=130073,device='cpu',max_tokens=2048):
    if eos_id!=130073 or formatter.tokenizer.convert_tokens_to_ids('<|im_end|>')!=eos_id:
        raise ValueError('original MiniCPM assistant message termination required')
    prompt=formatter.prompt(question)
    target=tuple(formatter.tokenizer.encode(wire(answer).decode(),add_special_tokens=False))+(eos_id,)
    if not target or len(prompt)+len(target)>max_tokens:raise ValueError('supervised sequence exceeds fixed capacity')
    ids=torch.tensor([prompt+target],device=device);labels=ids.clone();labels[:,:len(prompt)]=-100
    return ids,torch.ones_like(ids),labels
