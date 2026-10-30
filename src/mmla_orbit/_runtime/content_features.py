"""Preparation reference for a new, content-aware RTMA input domain.

Pure CPU functions over explicit bounded inputs. They do not read files, call a
model/environment, authenticate records, mutate memory, or provide an executor.
The production adapter must authenticate complete payloads and each actual
successor BEFORE calling this layer. Existing eight features keep their meaning.
"""
from dataclasses import dataclass
import math
import re


SCHEMA = 'orbit-retained-content-marginals-v1'
MAX_ROWS = 8
MAX_ROW_BYTES = 4096
MAX_GOAL_BYTES = 4096
WORD = re.compile(r'[a-z0-9_]+', re.ASCII)


def lexical_units(text):
    """Unigrams + adjacent word bigrams INSIDE one row; no gap reconstruction."""
    if type(text) is not str or not text or len(text.encode('utf-8')) > MAX_ROW_BYTES:
        raise ValueError('bounded nonempty row text required')
    words = WORD.findall(text.lower())
    return frozenset([('word', w) for w in words] +
                     [('pair', a, b) for a, b in zip(words, words[1:])])


@dataclass(frozen=True)
class Observation:
    observed_step: int
    char_offset: int
    text: str


def observation_units(row, *, clock):
    if (type(row) is not Observation or type(clock) is not int or not 0 <= clock <= 64
            or type(row.observed_step) is not int or not 0 <= row.observed_step <= clock
            or type(row.char_offset) is not int or not 0 <= row.char_offset < 2**32):
        raise ValueError('arrived typed observation fragment required')
    # Provenance, step and offset establish authority elsewhere; they are NOT
    # artificial novelty. Only the fragment's actual observed text is counted.
    return lexical_units(row.text)


@dataclass(frozen=True)
class Evaluation:
    action: str
    loss: float
    cost: float
    tag: str
    opportunity: int


def evaluation_units(row, *, clock):
    if (type(row) is not Evaluation or type(clock) is not int or not 0 <= clock < 4
            or type(row.opportunity) is not int or not 0 <= row.opportunity <= clock
            or row.tag not in ('completed', 'invalid-or-incomplete', 'task-failure')
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in (row.loss, row.cost))
            or (row.tag == 'invalid-or-incomplete' and (row.loss != 1 or row.cost != 0))):
        raise ValueError('arrived bounded evaluation required')
    if type(row.action) is not str or len(row.action.encode()) > 2048:
        raise ValueError('bounded actual sampled program text required')
    # Empty/invalid rollout remains a record; no fabricated valid program.
    words = lexical_units(row.action) if row.action else frozenset()
    return words | frozenset({('loss-bin', min(15, math.floor(16*row.loss))),
                              ('cost-bin', min(15, math.floor(16*row.cost))),
                              ('tag', row.tag)})


def _units(rows):
    if (type(rows) is not tuple or len(rows) > MAX_ROWS
            or any(x is not None and (type(x) is not frozenset or
                                      any(type(u) is not tuple for u in x)) for x in rows)):
        raise ValueError('bounded explicit row-unit inventory required')
    return frozenset().union(*(x for x in rows if x is not None))


def marginals(before, successors, incoming, *, goal_words=frozenset()):
    """Coverage gained/lost by EACH actual successor; one common denominator.

    successors[0] is exact NULL. incoming is the complete arrived payload's
    units, even when a cyclic write retains only its final fragments. Scores
    describe lexical coverage, never correctness or future task utility.
    """
    old = _units(before)
    if (type(incoming) is not frozenset or type(goal_words) is not frozenset
            or any(type(w) is not str for w in goal_words)
            or type(successors) is not tuple or not successors or len(successors) > MAX_ROWS+1
            or successors[0] != before):
        raise ValueError('exact NULL-first issued successors and arrived units required')
    for rows in successors:
        if type(rows) is not tuple or len(rows) != len(before):
            raise ValueError('same bounded slots for each successor')
    union = old | incoming

    def weight(unit):
        if (not unit or unit[0] not in ('word', 'pair', 'loss-bin', 'cost-bin', 'tag')
                or (unit[0] == 'word' and (len(unit) != 2 or type(unit[1]) is not str))
                or (unit[0] == 'pair' and (len(unit) != 3 or any(type(w) is not str for w in unit[1:])))):
            raise ValueError('typed content unit required')
        return 1 + int(unit[0] in ('word', 'pair') and any(w in goal_words for w in unit[1:]))

    z = max(1, sum(weight(u) for u in union))
    answer = []
    for rows in successors:
        current = _units(rows)
        if not current <= union:
            raise ValueError('successor contains content absent from retained/arrived records')
        gained = sum(weight(u) for u in current - old)
        lost = sum(weight(u) for u in old - current)
        answer.append((gained/z, lost/z))
    return tuple(answer), z


def goal_units(goal):
    if type(goal) is not str or not goal or len(goal.encode()) > MAX_GOAL_BYTES:
        raise ValueError('bounded immutable public task goal required')
    return frozenset(WORD.findall(goal.lower()))


def append_features(original, marginal):
    if (type(original) is not tuple or len(original) != 8 or type(marginal) is not tuple or len(marginal) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
                   for x in original + marginal)):
        raise ValueError('eight unchanged old features plus two bounded marginals required')
    if original[0] == 1 and marginal != (0., 0.):
        raise ValueError('NULL has zero content gain and loss')
    return original + marginal


# Exact future bank, four programs only. No search/mutation on observed scores.
# These are hypothesis instruments, not programs already shown to improve tasks.
BANK = (
    ('parent', ('add','sub','c1','x0','add','sub','c1','x1','x6')),
    ('null', ('c0',)),
    ('coverage', ('sub','x8','x9')),
    ('coverage_age', ('add','sub','x8','x9','mul','c2','x6')),
)


def validate_program(symbols):
    """Syntax-only reference; production retains explicit FP32 interpretation."""
    if type(symbols) is not tuple or not 1 <= len(symbols) <= 31:
        raise ValueError('bounded exact wire')
    pending = [0]
    for s in symbols:
        if not pending:
            raise ValueError('trailing symbols')
        depth = pending.pop(0)
        if s in ('add', 'sub', 'mul', 'min', 'max', 'sdiv'):
            if depth >= 3:
                raise ValueError('depth cap')
            pending[:0] = [depth+1, depth+1]
        elif s not in tuple('c'+str(i) for i in range(7))+tuple('x'+str(i) for i in range(10)):
            raise ValueError('unknown symbol')
    if pending:
        raise ValueError('incomplete program')
    return symbols
