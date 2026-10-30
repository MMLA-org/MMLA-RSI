"""Owned task accounting for the internal scalar generation path.

Only the role's concrete WorkLog is accepted here. The public generator keeps
its original arbitrary-callback contract and checks. This owner has no model,
Phi, memory, tokenizer or environment reference and cannot change their state.
"""
from types import MethodType
from proposal_role_common import WorkLog


class TaskAccounting:
    def __init__(self, on_work, cap):
        if (type(on_work) is not MethodType or type(on_work.__self__) is not WorkLog
                or on_work.__func__ is not WorkLog.work):
            raise TypeError('internal generation requires the concrete role WorkLog.work owner')
        if type(cap) is not int or cap <= 0:
            raise ValueError('positive task input-token cap required')
        self.log = on_work.__self__
        self.cap = cap
        self.forward_inputs = 0

    def __call__(self, event):
        WorkLog.work(self.log, event)
        if event['kind'] in ('generation_forward', 'student_trunk_forward'):
            self.forward_inputs += event['valid_tokens']
            if self.forward_inputs > self.cap:
                raise RuntimeError('bounded task input-token cost exhausted')

    def forwarded(self, rows, valid, padded):
        self(dict(kind='generation_forward', rows=rows,
                  valid_tokens=valid, padded_positions=padded))

    def tokens(self, rows, tokens):
        self(dict(kind='generated_tokens', count=len(tokens)))
