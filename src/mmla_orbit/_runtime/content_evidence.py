"""RTMA's independent authoritative evaluation-record carrier.

The fixed deterministic derived view is a byte encoding, not a learned semantic
certification. It deliberately does not borrow the old value/unit-only gate.
"""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
import re
import struct
from content_rules import identity, evaluate
from content_features import Evaluation, evaluation_units, marginals, append_features, SCHEMA


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


@dataclass(frozen=True)
class Record:
    action: str
    loss: float
    cost: float
    tag: str
    source: str
    opportunity: int

    def validate(self):
        if (type(self.action) is not str or len(self.action.encode()) > 2048
                or type(self.source) is not str or not 0 < len(self.source.encode()) <= 128
                or self.tag not in ('completed', 'invalid-or-incomplete', 'task-failure')
                or type(self.opportunity) is not int or not 0 <= self.opportunity < 2**32
                or type(self.loss) not in (int, float) or type(self.cost) not in (int, float)
                or not 0 <= self.loss <= 1 or not 0 <= self.cost <= 1):
            raise ValueError('malformed arrived evidence; not a legal NULL')
        if self.tag == 'invalid-or-incomplete' and (self.loss != 1 or self.cost != 0):
            raise ValueError('invalid rollout has loss one and zero development cost')
        if self.tag == 'task-failure' and self.loss != 1:
            raise ValueError('legal task failure retains actual cost and has loss one')
        return canonical(asdict(self))

    @property
    def object_id(self):
        return identity(dict(source=self.source, opportunity=self.opportunity))


@dataclass(frozen=True)
class EvidenceRow:
    slot: int
    version: int
    predecessor: str
    record: Record
    derived_hex: str

    def bytes(self):
        return canonical(asdict(self))


def derive(record):
    payload = record.validate()
    # Fixed IEEE64 storage of observed scalar values, plus complete record hash.
    return (struct.pack('<ddQ', record.loss, record.cost, record.opportunity) +
            hashlib.sha256(payload).digest()).hex()


@dataclass(frozen=True)
class MemoryBounds:
    slots: int = 2
    row_bytes: int = 4096
    read_tokens: int = 1024
    max_version: int = 2**32-1

    def __post_init__(self):
        for name in ('slots', 'row_bytes', 'read_tokens', 'max_version'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive memory capacity/version cap required')
        if self.slots > 256 or self.row_bytes > 65536 or self.read_tokens > 65536 or self.max_version >= 2**64:
            raise ValueError('bounded resident memory required')


class EvidenceMemory:
    def __init__(self, bounds, *, reader_identity, count_tokens, rows=None, version=0,
                 protected=(), source_prefix='', namespace):
        if type(bounds) is not MemoryBounds or bounds.slots != 2:
            raise ValueError('exact two-slot content proposal memory required')
        self.bounds = bounds
        self.namespace = namespace
        self.reader_identity = reader_identity
        self._count_tokens = count_tokens
        self.rows = tuple([None]*bounds.slots if rows is None else rows)
        self.version = version
        self.protected = frozenset(protected)
        self.source_prefix = source_prefix
        self.validate()

    @property
    def contract(self):
        # The reader identity binds both tokenization and deterministic text read.
        return dict(schema='rtma-content-evidence-v1', encoding='canonical-ascii-ieee64-le-sha256-v1',
            content_features=SCHEMA,
            bounds=asdict(self.bounds), reader=self.reader_identity,
            protected=sorted(self.protected), source_prefix=self.source_prefix,
            namespace=self.namespace)

    @property
    def identity(self):
        return identity(dict(contract=self.contract,
            rows=[None if x is None else asdict(x) for x in self.rows], version=self.version))

    def validate(self):
        if (type(self.namespace) is not str or not 0 < len(self.namespace.encode()) <= 128
                or type(self.source_prefix) is not str or len(self.source_prefix.encode()) > 128):
            raise ValueError('fixed role namespace and source permission required')
        if len(self.rows) != self.bounds.slots or type(self.version) is not int or not 0 <= self.version <= self.bounds.max_version:
            raise ValueError('invalid resident/version state')
        seen = set()
        for i, row in enumerate(self.rows):
            if row is None:
                continue
            if type(row) is not EvidenceRow or type(row.record) is not Record:
                raise ValueError('complete typed authoritative row required')
            row.record.validate()
            if (type(row.slot) is not int or row.slot != i
                    or type(row.version) is not int or not 1 <= row.version <= self.version
                    or type(row.predecessor) is not str or re.fullmatch('[0-9a-f]{64}',row.predecessor) is None
                    or row.derived_hex != derive(row.record)
                    or len(row.bytes()) > self.bounds.row_bytes
                    or row.record.object_id in seen
                    or not row.record.source.startswith(self.source_prefix)):
                raise ValueError('authoritative row integrity/identity failure')
            seen.add(row.record.object_id)
        if any(type(i) is not int or not 0 <= i < self.bounds.slots for i in self.protected):
            raise ValueError('invalid protected slot')
        if type(self.reader_identity) is not str or not self.reader_identity:
            raise ValueError('bound deterministic reader required')
        tokens = self._count_tokens(self._text())
        if type(tokens) is not int or not 0 <= tokens <= self.bounds.read_tokens:
            raise ValueError('complete memory read exceeds bound; no truncation')

    def _text(self):
        return canonical([dict(slot=i, evidence=asdict(row.record))
                          for i, row in enumerate(self.rows) if row is not None]).decode('ascii')

    def read(self):
        self.validate()
        return self._text(), dict(memory=self.identity, reader=self.reader_identity,
                                 version=self.version, read_tokens=self._count_tokens(self._text()))

    def export(self):
        self.validate()
        return dict(contract=self.contract,
                    rows=[None if x is None else asdict(x) for x in self.rows],
                    version=self.version, identity=self.identity)

    @classmethod
    def restore(cls, state, *, count_tokens, expected_contract):
        # Expected permissions come from the caller's fixed role contract,
        # never from the serialized state being restored.
        if (set(state) != {'contract','rows','version','identity'}
                or canonical(state['contract']) != canonical(expected_contract)):
            raise ValueError('memory carrier contract changed on restore')
        c = expected_contract
        rows = [None if x is None else EvidenceRow(**{**x, 'record':Record(**x['record'])})
                for x in state['rows']]
        obj = cls(MemoryBounds(**c['bounds']), reader_identity=c['reader'],
                  count_tokens=count_tokens, rows=rows, version=state['version'],
                  protected=c['protected'], source_prefix=c['source_prefix'], namespace=c['namespace'])
        if obj.contract != c or obj.identity != state['identity']:
            raise ValueError('memory serialization identity changed')
        return obj

    def manage(self, record, *, language, rule, opportunity_cap):
        self.validate()
        record.validate()  # Common payload failures cannot be downgraded to NULL.
        if (rule.mode != 'rtma' or not record.source.startswith(self.source_prefix)
                or type(opportunity_cap) is not int or opportunity_cap != 4
                or not 0 <= record.opportunity < opportunity_cap):
            raise ValueError('arrived source/domain/clock failure')
        if any(row and row.record.object_id == record.object_id for row in self.rows):
            raise ValueError('duplicate insertion, not a new object')
        # Common serialized-size check before slot filtering.
        stub = EvidenceRow(0, min(self.version+1, self.bounds.max_version), self.identity,
                           record, derive(record))
        if len(stub.bytes()) > self.bounds.row_bytes:
            raise ValueError('oversized common payload, not NULL')
        occupied = sum(x is not None for x in self.rows)/self.bounds.slots
        features = (1., 0., record.loss, record.cost, 0., 0., 0., occupied)
        choices = [(features, -1, self)]
        for slot, old in enumerate(self.rows):
            if slot in self.protected or self.version == self.bounds.max_version:
                continue
            if old and old.record.opportunity > record.opportunity:
                raise ValueError('future evidence is not an arrived row')
            row = replace(stub, slot=slot, version=self.version+1)
            rows = list(self.rows); rows[slot] = row
            try:
                successor = EvidenceMemory(self.bounds, reader_identity=self.reader_identity,
                    count_tokens=self._count_tokens, rows=rows, version=self.version+1,
                    protected=self.protected, source_prefix=self.source_prefix, namespace=self.namespace)
            except ValueError:
                continue  # Slot-specific capacity/protection exclusion after payload validation.
            f = (0., float(old is not None), record.loss, record.cost,
                 old.record.loss if old else 0., old.record.cost if old else 0.,
                 min(1., (record.opportunity-old.record.opportunity)/opportunity_cap) if old else 0., occupied)
            choices.append((f, slot, successor))
        def units(row):
            if row is None:return None
            r=row.record
            return evaluation_units(Evaluation(r.action,r.loss,r.cost,r.tag,r.opportunity),
                                    clock=record.opportunity)
        before=tuple(units(row) for row in self.rows)
        incoming=evaluation_units(Evaluation(record.action,record.loss,record.cost,
                                             record.tag,record.opportunity),clock=record.opportunity)
        extras,normalizer=marginals(before,tuple(tuple(units(r) for r in m.rows) for _,_,m in choices),incoming)
        scored=[];evidence=[]
        for (base,slot,successor),extra in zip(choices,extras):
            f=append_features(base,extra)
            score=evaluate(language,rule,memory_features=f)[0]
            scored.append((score,slot,successor))
            evidence.append(dict(slot=slot,score=score,features=f))
        # max returns the first tie: NULL, then ascending slot. Any score failure
        # above propagates before a successor is exposed; neither carrier mutates.
        score, slot, successor = max(scored, key=lambda x:x[0])
        successor.validate()
        return successor, dict(status='NULL' if slot == -1 else 'COMMIT', slot=slot,
            rule=rule.identity, predecessor=self.identity, successor=successor.identity,
            score=score, legal_scores=evidence, content_schema=SCHEMA,content_normalizer=normalizer)
