"""Bounded RSI expression language from MMLA-RSI 2610.20294v1.

No Python evaluation, I/O, role branch, task identifier or hidden rule state.
Constants are an ordered table of exact FP32 bits. Prefix symbols are the wire
language; a bound model-token codec supplies the sampling alphabet separately.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import struct
import torch

OPS = ('add', 'sub', 'mul', 'min', 'max', 'sdiv')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def bits(value):
    return struct.pack('<f', value).hex()


@dataclass(frozen=True)
class Language:
    constants: tuple[str, ...] = tuple(bits(x) for x in (0., 1., .01, .1, .5, .9, -1.))
    max_nodes: int = 63
    max_depth: int = 5
    max_tokens: int = 63
    max_bytes: int = 4096
    max_instructions: int = 63
    epsilon: float = 1e-6

    def __post_init__(self):
        if (type(self.constants) is not tuple or len(self.constants) > 32 or not self.constants
                or any(type(x) is not str or len(x) != 8 for x in self.constants)):
            raise ValueError('ordered finite FP32 constant table required')
        values = [struct.unpack('<f', bytes.fromhex(x))[0] for x in self.constants]
        if not all(math.isfinite(x) for x in values) or 0. not in values or 1. not in values:
            raise ValueError('constant domain must contain zero and one')
        for name in ('max_nodes', 'max_tokens', 'max_bytes', 'max_instructions'):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 65536:
                raise ValueError('finite positive syntax bound')
        if type(self.max_depth) is not int or not 0 <= self.max_depth <= 32:
            raise ValueError('bounded expression depth')
        if not 0 < self.epsilon < math.inf:
            raise ValueError('positive finite division floor')

    @property
    def identity(self):
        return identity(dict(schema='rsi-prefix-fp32-v1', config=asdict(self),
                             operators=OPS, order='left-right-postorder',
                             numerics='torch-fp32-explicit-intermediate-checks', backend=dict(torch=torch.__version__, device='cpu', dtype='float32')))

    def constant(self, index):
        return struct.unpack('<f', bytes.fromhex(self.constants[index]))[0]


@dataclass(frozen=True)
class Rule:
    mode: str
    symbols: tuple[str, ...]
    language: str

    @property
    def identity(self):
        return identity(asdict(self))


class PrefixGrammar:
    """Each symbol consumes one syntax node. Completion itself draws no token."""
    def __init__(self, language, mode):
        if mode not in ('rtt', 'rtma', 'joint'):
            raise ValueError('explicit decision owner/domain required')
        self.language, self.mode = language, mode
        self.pending = [(0, 4)] * (3 if mode in ('rtt', 'joint') else 0)
        if mode in ('rtma', 'joint'):
            self.pending.append((0, 8))
        self.symbols = []

    @property
    def complete(self):
        return not self.pending

    def allowed(self):
        if not self.pending or len(self.symbols) >= self.language.max_tokens:
            return ()
        depth, features = self.pending[0]
        if len(self.symbols) + len(self.pending) > self.language.max_nodes:
            return ()
        leaves = tuple('c'+str(i) for i in range(len(self.language.constants)))
        leaves += tuple('x'+str(i) for i in range(features))
        # One binary node replaces a pending leaf by two children.
        operations = OPS if (depth < self.language.max_depth and
            len(self.symbols) + len(self.pending) + 2 <= self.language.max_nodes) else ()
        return leaves + operations

    def append(self, symbol):
        if symbol not in self.allowed():
            raise ValueError('illegal or over-budget grammar transition')
        depth, features = self.pending.pop(0)
        self.symbols.append(symbol)
        if symbol in OPS:
            self.pending[:0] = [(depth+1, features), (depth+1, features)]

    def finish(self):
        if not self.complete:
            raise ValueError('incomplete rollout is not a legal program')
        rule = Rule(self.mode, tuple(self.symbols), self.language.identity)
        if len(json.dumps(asdict(rule), separators=(',', ':')).encode()) > self.language.max_bytes:
            raise ValueError('program serialization capacity exceeded')
        return rule


def parse_rule(language, mode, symbols):
    if type(symbols) not in (tuple, list) or len(symbols) > language.max_tokens:
        raise ValueError('bounded symbols required')
    grammar = PrefixGrammar(language, mode)
    for symbol in symbols:
        grammar.append(symbol)
    return grammar.finish()


def evaluate(language, rule, numerical_features=(), memory_features=()):
    """FP32 scalar results, with every intermediate checked before clipping."""
    if rule.language != language.identity or parse_rule(language, rule.mode, rule.symbols) != rule:
        raise ValueError('program/language identity mismatch')
    pos = 0
    steps = 0

    def checked(x):
        if x.ndim != 0 or x.dtype != torch.float32 or x.device.type != 'cpu' or not torch.isfinite(x):
            raise FloatingPointError('nonfinite scalar interpreter intermediate')
        return x

    def expression(features):
        nonlocal pos, steps
        steps += 1
        if steps > language.max_instructions:
            raise RuntimeError('rule instruction budget exceeded')
        symbol = rule.symbols[pos]
        pos += 1
        if symbol.startswith('c'):
            return checked(torch.tensor(language.constant(int(symbol[1:])), dtype=torch.float32, device='cpu'))
        if symbol.startswith('x'):
            return checked(torch.tensor(features[int(symbol[1:])], dtype=torch.float32, device='cpu'))
        x, y = expression(features), expression(features)
        if symbol == 'add': result = x+y
        elif symbol == 'sub': result = x-y
        elif symbol == 'mul': result = x*y
        elif symbol == 'min': result = torch.minimum(x, y)
        elif symbol == 'max': result = torch.maximum(x, y)
        else:
            denominator = checked(torch.where(y < 0, -torch.ones_like(y), torch.ones_like(y)) * y.abs().clamp_min(language.epsilon))
            result = x/denominator
        return checked(result)

    outputs = []
    if rule.mode in ('rtt', 'joint'):
        if len(numerical_features) != 4:
            raise ValueError('four arrived numerical features required')
        outputs += [expression(numerical_features) for _ in range(3)]
    if rule.mode in ('rtma', 'joint'):
        if len(memory_features) != 8:
            raise ValueError('eight arrived memory features required')
        outputs.append(expression(memory_features))
    return tuple(float(x) for x in outputs)
