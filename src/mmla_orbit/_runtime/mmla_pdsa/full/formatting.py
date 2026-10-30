"""One registered MiniCPM-SFT prompt format, with no answers in generation input."""
import hashlib
from .profile import digest

INSTRUCTION='Use the accessible current memory to answer the query. Return only the requested JSON value.\nQuery: '


class InputFormatter:
    def __init__(self,tokenizer,*,tokenizer_identity):
        if type(tokenizer_identity) is not str or len(tokenizer_identity)!=64 or not tokenizer.chat_template:
            raise ValueError('verified tokenizer identity and original chat template required')
        int(tokenizer_identity,16)
        self.tokenizer=tokenizer
        self.template_digest=digest(tokenizer.chat_template)
        self.identity=digest(dict(schema='mmla-engineering-prompt-v1',tokenizer=tokenizer_identity,
            template=self.template_digest,instruction=INSTRUCTION,enable_thinking=False))

    def prompt(self,text):
        if type(text) is not str or not text or len(text.encode())>8192:
            raise ValueError('bounded observed query string required')
        if digest(self.tokenizer.chat_template)!=self.template_digest:raise ValueError('chat template changed')
        ids=self.tokenizer.apply_chat_template([dict(role='user',content=INSTRUCTION+text)],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if not ids or any(type(x) is not int or x<0 for x in ids):raise ValueError('complete prompt token sequence required')
        return tuple(ids)
