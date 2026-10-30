"""Original engineering tasks on a causal newline-delimited token transport.

No new task, label or family is created. Source authority comes from the original
host envelopes. The host adapter can hold a full file, but feeds only its next
observed token; the compiler, selector and model never receive the future file.
Source completion is checked by original command receipts, not segment counts.
"""
from dataclasses import dataclass, asdict
import copy,hashlib
from ..episode_adapter import EngineeringTask
from ..profile import digest
from ..formatting import InputFormatter
from .stream import ObservedToken


@dataclass(frozen=True)
class CommandSpan:
    index: int
    original_segment: int
    start: int
    end: int
    source_receipt: str
    text_sha256: str


class EngineeringStreamTask:
    def __init__(self,task,tokenizer,*,tokenizer_identity,source_tokens=4096,command_tokens=512):
        if type(task) is not EngineeringTask or task.dataset.formatter is None:
            raise ValueError('bound original engineering task and formatter required')
        if (task.dataset.formatter.tokenizer is not tokenizer
                or task.dataset.formatter.identity!=InputFormatter(tokenizer,tokenizer_identity=tokenizer_identity).identity):
            raise ValueError('same bound source and task tokenizer required')
        if (type(source_tokens) is not int or not 1<=source_tokens<=65536
                or type(command_tokens) is not int or not 1<=command_tokens<=2048):
            raise ValueError('finite source and per-command token capacities')
        ending=tuple(tokenizer.encode('.\n',add_special_tokens=False))
        if not ending or not tokenizer.decode((ending[-1],),skip_special_tokens=False,clean_up_tokenization_spaces=False).endswith('\n'):
            raise ValueError('registered sentence-plus-newline terminal token required')
        self.task,self.tokenizer=task,tokenizer
        self.tokenizer_identity=tokenizer_identity;self.delimiter=ending[-1]
        self.source_cap,self.command_cap=source_tokens,command_tokens
        self.spans,self._tokens=self._encode()
        self.source_identity=self._source_digest()
        self.identity=digest(dict(schema='mmla-engineering-stream-task-v1',task=task.identity,
            source=self.source_identity,transport='original-period-terminated-command-plus-newline-v2'))
        self.group=task.group;self.split=task.view.split
        self.bootstrap_commands=sum(x.original_segment==0 for x in self.spans)
        self.feedback_commands=self.bootstrap_commands if task.timing=='early' else len(self.spans)
        self._task_content=self._task_digest()
        self._formatter_identity=task.dataset.formatter.identity
        self._immutable=self._guard()

    def _encode(self):
        task=self.task;task.dataset._validate_view(task.view)
        spans=[];observed=[]
        for seg in range(len(task.row['observations']['segments'])):
            for command in task.commands(seg):
                text=command.text+'\n'
                tokens=tuple(self.tokenizer.encode(text,add_special_tokens=False))
                if (not text.endswith('.\n') or not tokens or len(tokens)>self.command_cap or tokens[-1]!=self.delimiter
                        or self.delimiter in tokens[:-1] or any(type(x) is not int or x<0 for x in tokens)
                        or self.tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=text):
                    raise ValueError('complete original command does not fit lossless newline transport')
                start=len(observed)+1;index=len(spans)+1
                for i,token in enumerate(tokens):
                    observed.append(ObservedToken(start+i,token,index,start,i==len(tokens)-1,command.envelope))
                spans.append(CommandSpan(index,seg,start,len(observed),command.envelope.source_receipt,
                    hashlib.sha256(text.encode()).hexdigest()))
        if not spans or len(observed)>self.source_cap:raise ValueError('original finite source does not fit')
        return tuple(spans),tuple(observed)

    def _source_digest(self):
        return digest(dict(schema='original-observed-engineering-tokens-v1',tokenizer=self.tokenizer_identity,
            task=self.task.identity,spans=[asdict(s) for s in self.spans],
            tokens=[asdict(x) for x in self._tokens],source_cap=self.source_cap,command_cap=self.command_cap))

    def _task_digest(self):
        t=self.task
        actual=t.dataset._validate_view(t.view)
        if t.row!=actual:raise ValueError('actual task truth row differs from verified original source')
        return digest(dict(identity=t.identity,timing=t.timing,group=t.group,support_clock=t.support_clock,
            row=t.row,names=t.names,operations=t.operations,source_events=t.source_events))

    def _guard(self):
        return (id(self._tokens),id(self.spans),id(self.tokenizer),self.tokenizer_identity,self.delimiter,
                self.source_cap,self.command_cap,self.source_identity,self.identity,
                self.task.identity,self.task.dataset.identity,self.group,self.split,
                self.bootstrap_commands,self.feedback_commands)

    def check(self):
        self.task.dataset._validate_view(self.task.view)
        if (self.task.dataset.formatter.tokenizer is not self.tokenizer
                or self.task.dataset.formatter.identity!=self._formatter_identity
                or self._guard()!=self._immutable or self._task_digest()!=self._task_content
                or digest(self.tokenizer.chat_template)!=self.task.dataset.formatter.template_digest):
            raise ValueError('original source protocol or formatter changed')
        # Bind source rows and authority again; source changes cannot inherit a digest.
        actual=tuple(c.envelope.source_receipt for i in range(len(self.task.row['observations']['segments'])) for c in self.task.commands(i))
        if actual!=tuple(s.source_receipt for s in self.spans):raise ValueError('original source receipts changed')

    def admit(self,driver):
        self.check();driver._check();p=driver.problem
        if (driver.source_identity!=self.source_identity or driver.tokenizer_identity!=self.tokenizer_identity
                or driver.tokenizer is not self.tokenizer or p.problem_id!=self.identity
                or p.tenant!='engineering' or p.principal!='reader'
                or driver.boundary.min_tokens!=1 or driver.boundary.delimiters!=(self.delimiter,)
                or driver.limits.source_tokens< len(self._tokens)
                or driver.limits.source_commands<len(self.spans)):
            raise ValueError('exact causal task/source/detector driver required')

    def next_observation(self,driver):
        if self._guard()!=self._immutable or driver.source_identity!=self.source_identity or driver.problem.problem_id!=self.identity:
            raise ValueError('bound immutable source reader required')
        position=driver.problem.segment.position
        if position>=len(self._tokens):return None
        return self._tokens[position]

    def consume_to(self,driver,command,selector,*,on_observation=None):
        """Consume an original prefix; future data never enter the selection call."""
        self.admit(driver)
        if type(command) is not int or not driver._processed_command<=command<=len(self.spans):
            raise ValueError('monotone original command endpoint required')
        end=0 if command==0 else self.spans[command-1].end
        if driver.problem.segment.position>end:raise ValueError('cannot roll back an observed prefix')
        def guarded(p,context,candidates):
            bootstrap={s.source_receipt for s in self.spans[:self.bootstrap_commands]}
            if context.envelope.source_receipt in bootstrap:
                # Original host initialization precedes predictive selection.
                if (p.attempts or p.fast.steps or p.pending is not None
                        or p.memory.version!=driver._processed_command
                        or sum(row is not None for row in p.memory._rows)!=driver._processed_command
                        or context.event.kind!='create' or len(candidates)<2):
                    raise ValueError('original empty-M create bootstrap/qualification required')
                return 1,dict(mode='HOST_AUTHORIZED_BOOTSTRAP',predictive=False)
            return selector(p,context,candidates)
        guarded.identity=selector.identity
        if driver.problem.segment.stop is not None:driver.process(guarded)
        while driver.problem.segment.position<end:
            token=self.next_observation(driver)
            if on_observation is not None:on_observation(token)
            if driver.feed(token) is not None:driver.process(guarded)
        if driver._processed_command!=command or driver.problem.segment._core:
            raise ValueError('declared command endpoint was not actually published')
        if command==len(self.spans):driver.finish_source(offset=end)
        return tuple(driver.receipts)

    def validate_support_prefix(self,driver):
        """Validate arrived source only; this method does not expose any label."""
        self.admit(driver);p=driver.problem;task=self.task;s=p.segment
        if (driver._processing or s._pending is not None or s._core
                or driver._processed_command!=self.feedback_commands
                or s.position!=self.spans[self.feedback_commands-1].end):
            raise PermissionError('original support prefix is not completely published')
        if any(r['status']!='COMPLETED' or len(r['commands'])!=len(r['events']) for r in driver.receipts):
            raise PermissionError('every original command needs its actual event result')
        source=[x for r in driver.receipts for x in r['commands']]
        events=[x for r in driver.receipts for x in r['events']]
        if any(x.get('action')!='WRITE' or x.get('selection',{}).get('mode')!='HOST_AUTHORIZED_BOOTSTRAP'
               or x.get('predecessor')==x.get('successor') for x in events[:self.bootstrap_commands]):
            raise PermissionError('support requires actual original bootstrap commits')
        if self.task.timing=='early' and (p.memory.version!=self.bootstrap_commands
                or sum(row is not None for row in p.memory._rows)!=self.bootstrap_commands):
            raise PermissionError('support predecessor lacks the original initialized M')

        expected=self.spans[:self.feedback_commands]
        if (len(source)!=len(expected) or any(r['source']!=q.source_receipt or r['command']!=q.index
                or r['start']!=q.start or r['end']!=q.end or r['text_sha256']!=digest(self.tokenizer.decode(
                    tuple(x.token for x in self._tokens[q.start-1:q.end]),skip_special_tokens=False,clean_up_tokenization_spaces=False))
                for r,q in zip(source,expected))):
            raise PermissionError('complete original source ownership evidence required')
        return dict(source=self.source_identity,position=s.position,commands=driver._processed_command,
            root=p.memory.identity,world_clock=p.world_clock)

    def release_feedback(self,driver):
        """Original fixed-SGD interface retains its exact issued-prompt gate."""
        self.validate_support_prefix(driver)
        p=driver.problem;task=self.task
        pending=p.pending;prompt=task.dataset.formatter.prompt(task.question(task.support_query()))
        if (pending is None or p.attempts!=1 or pending.number!=1 or p.failed or p.closed
                or pending.problem!=self.identity or pending.prompt!=prompt
                or pending.world_clock!=task.support_clock or p.world_clock!=task.support_clock
                or pending.memory!=p.memory.identity or pending.phi!=p.fast.identity or pending.reader!=p._reader):
            raise PermissionError('exact own unresolved support attempt required')
        return copy.deepcopy(task.support_target())

    def metadata(self):
        self.check()
        return dict(identity=self.identity,original_task=self.task.identity,source=self.source_identity,
            family=self.group,split=self.split,commands=len(self.spans),tokens=len(self._tokens),
            bootstrap_commands=self.bootstrap_commands,feedback_commands=self.feedback_commands,
            max_command_tokens=max(s.end-s.start+1 for s in self.spans),delimiter=self.delimiter)
