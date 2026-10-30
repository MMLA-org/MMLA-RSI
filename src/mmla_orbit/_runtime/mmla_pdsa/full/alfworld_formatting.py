"""Exact pinned chat prompts for actions and pre-arrival observation learning.

No teacher, task id, hidden state or archived history enters these prompts.
The runner separately binds the pre-arrival Phi/M tensors for feedback loss.
"""
from dataclasses import dataclass
import json

from .alfworld_episode import VisibleFrame, FeedbackPair, _digest, _sha
from .command_data import POLICY as COMMAND_DATA_POLICY, escape_control_json

ACTION_INSTRUCTION = ('Complete the household task using the current observation and accessible memory. '
    'Return exactly one environment command, with no explanation. '
    'Public command forms: look; inventory; go to (receptacle); open (receptacle); '
    'close (receptacle); take (object) from (receptacle); move (object) to (receptacle); '
    'put (object) into (receptacle object); put (receptacle object) in (receptacle); '
    'examine (object or receptacle); use (object); heat (object) with (receptacle); '
    'clean (object) with (receptacle); cool (object) with (receptacle); '
    'slice (object) with (sharp object); help. Use names and numbers from observations.\n')
FEEDBACK_INSTRUCTION = ('Predict the next environment observation after the submitted command, '
    'using only the observation and accessible memory available before that command. '
    'Return the observation text only.\n')


def _controls(tokenizer):
    # all_special_tokens omits some backend-added special tokens in this
    # tokenizer. Inspect their actual AddedToken.special flags as well.
    return tuple(sorted(set(tokenizer.all_special_tokens) |
        {t.content for t in tokenizer.added_tokens_decoder.values() if t.special} |
        {'<think>', '</think>'}))


@dataclass(frozen=True)
class SupervisedText:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_length: int
    formatter_identity: str
    kind: str

    @property
    def target_tokens(self):
        return sum(label != -100 for label in self.labels[1:])


class ALFWorldFormatter:
    """One stateless prompt per action: dynamic history is read only through M.

    The tokenizer is the fixed original MiniCPM fast tokenizer. Prompt identity
    binds its supplied asset identity, template, public instructions and bounds.
    No truncation, format repair or alternate prompt is attempted on overflow.
    """
    def __init__(self, tokenizer, *, tokenizer_identity, max_tokens=2048,
                 max_text_bytes=65536, max_action_bytes=2048):
        _sha(tokenizer_identity)
        if (type(max_tokens) is not int or not 1 <= max_tokens <= 32768
                or type(max_text_bytes) is not int or not 1 <= max_text_bytes <= 65536
                or type(max_action_bytes) is not int or not 1 <= max_action_bytes <= 2048
                or not tokenizer.is_fast or not tokenizer.chat_template):
            raise ValueError('pinned fast tokenizer and finite text bounds required')
        self.tokenizer=tokenizer
        self.max_tokens,self.max_text_bytes,self.max_action_bytes=max_tokens,max_text_bytes,max_action_bytes
        self._template=_digest(tokenizer.chat_template)
        self._specials=_controls(tokenizer)
        self._eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
        if self._eos != 130073:
            raise ValueError('original MiniCPM assistant terminator required')
        self.identity=_digest(dict(schema='orbit-alfworld-prompts-v4',command_data=COMMAND_DATA_POLICY,tokenizer=tokenizer_identity,
            template=self._template,specials=self._specials,eos=self._eos,enable_thinking=False,
            action=ACTION_INSTRUCTION,feedback=FEEDBACK_INSTRUCTION,max_tokens=max_tokens,
            max_text_bytes=max_text_bytes,max_action_bytes=max_action_bytes,
            target='body-plus-im_end',history='pinned-memory-only',evaluation='complete-EvidenceMemory-read-or-absent'))

    def _check(self):
        if (_digest(self.tokenizer.chat_template)!=self._template
                or _controls(self.tokenizer)!=self._specials
                or self.tokenizer.convert_tokens_to_ids('<|im_end|>')!=self._eos):
            raise ValueError('pinned tokenizer/template changed')

    def _text(self,text,bound):
        if type(text) is not str or not text.strip() or len(text.encode())>bound:
            raise ValueError('complete bounded text required')
        if any(t in text for t in self._specials):
            raise ValueError('literal role/control token inside environment data')
        return text

    def _payload(self,frame):
        if type(frame) is not VisibleFrame or type(frame.admissible_commands) is not tuple:
            raise TypeError('actor-visible frame only')
        self._text(frame.goal,self.max_text_bytes);self._text(frame.observation,self.max_text_bytes)
        for a in frame.admissible_commands:self._text(a,self.max_action_bytes)
        return frame.prompt_payload()

    def _render(self,kind,frame,action=None,*,evaluation_memory=None):
        self._check()
        payload=self._payload(frame)
        if evaluation_memory is not None:
            from .extensions.evidence import EvidenceMemory
            if type(evaluation_memory) is not EvidenceMemory:
                raise TypeError('only a complete typed evaluation-memory read')
            text,receipt=evaluation_memory.read()
            if type(text) is not str or not text.strip() or len(text.encode())>self.max_text_bytes:
                raise ValueError('complete bounded typed evaluation read required')
            # Control spellings inside arrived records are quoted at the final
            # JSON boundary, after parsing; authoritative bytes stay unchanged.
            # Metadata receipt stays with the runner. The actor sees only the
            # retained rows, explicitly historical observations of its actions.
            payload['past_action_evaluations_not_current_world_facts']=json.loads(text)
        if kind=='feedback':
            if type(action) is not str or len(action.encode())>self.max_action_bytes:
                raise ValueError('bounded complete submitted command required')
            payload['submitted_command']=action
            instruction=FEEDBACK_INSTRUCTION
        elif kind=='action' and action is None:
            instruction=ACTION_INSTRUCTION
        else:raise ValueError('registered prompt kind')
        wire=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        body=instruction+escape_control_json(wire,self._specials)
        prompt=self.tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=tuple(self.tokenizer.encode(prompt,add_special_tokens=False))
        if not ids or len(ids)>self.max_tokens:raise ValueError('complete prompt exceeds context')
        return prompt,ids

    def evaluation_token_count(self,text):
        # Same safe representation as the final prompt, before its outer wrapper.
        wire=escape_control_json(text,self._specials)
        return len(self.tokenizer.encode(wire,add_special_tokens=False))

    def prompt(self,frame,*,evaluation_memory=None):
        return self._render('action',frame,evaluation_memory=evaluation_memory)[1]

    def _supervised(self,kind,frame,target,action=None,*,evaluation_memory=None):
        self._text(target,self.max_action_bytes if kind=='action' else self.max_text_bytes)
        prompt,prompt_ids=self._render(kind,frame,action,evaluation_memory=evaluation_memory)
        full=prompt+target+'<|im_end|>'
        encoded=self.tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
        ids=tuple(encoded['input_ids']);offsets=tuple(map(tuple,encoded['offset_mapping']))
        boundary=len(prompt)
        if (len(ids)>self.max_tokens or len(ids)!=len(offsets)
                or ids[:len(prompt_ids)]!=prompt_ids or len(ids)<=len(prompt_ids)
                or ids[-1]!=self._eos):
            raise ValueError('whole-sequence target must preserve the exact generation prefix')
        # Require every target token to live wholly after the public prompt.
        # The EOS has a real span in the pinned fast tokenizer (not a zero-span
        # tokenizer-inserted special); such silent insertions are rejected.
        if any(a<boundary or b<=a or b>len(full) for a,b in offsets[len(prompt_ids):]):
            raise ValueError('target token crosses the public/target boundary')
        labels=(-100,)*len(prompt_ids)+ids[len(prompt_ids):]
        return SupervisedText(ids,labels,len(prompt_ids),self.identity,kind)

    def action_supervision(self,frame,teacher_action,*,evaluation_memory=None):
        """Runner must supply FIT-only label from this exact actual state."""
        return self._supervised('action',frame,teacher_action,evaluation_memory=evaluation_memory)

    def feedback_supervision(self,pair,*,evaluation_memory=None):
        if type(pair) is not FeedbackPair or pair.after.observation!=pair.observation_target:
            raise ValueError('bound actual transition required')
        # Never serialize pair.after, hidden receipt, or observation target in
        # the prompt. The target is added only behind its loss boundary.
        return self._supervised('feedback',pair.before,pair.observation_target,pair.action,evaluation_memory=evaluation_memory)
