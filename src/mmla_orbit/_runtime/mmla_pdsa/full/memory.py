"""Complete-row AMR mechanics for one trusted, serialized local runtime.

Byte consistency is enforced independently of the optional semantic gate. Hashes
identify versions; they do not authenticate an adversarial caller or prove truth.
No original experiment state or source file is imported or mutated here.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import struct
import uuid
from typing import Callable, Protocol

from .profile import ArchitectureProfile, digest, wire


MAX_VERSION = 2**64 - 1


def canonical(value) -> bytes:
    """Normalize a JSON value, rejecting non-JSON keys and nonfinite values."""
    def check(v):
        if v is None or type(v) in (str, bool, int, float):
            return
        if type(v) is list:
            for x in v:
                check(x)
            return
        if type(v) is dict and all(type(k) is str for k in v):
            for x in v.values():
                check(x)
            return
        raise ValueError("canonical values must be JSON, with string object keys")
    check(value)
    return wire(value)


class Encoder(Protocol):
    identity: str

    def encode(self, canonical_bytes: bytes) -> tuple[bytes, bytes]:
        """Deterministic FP32 little-endian key and value, derived from content."""


@dataclass(frozen=True)
class Authority:
    tenant: str
    acl: tuple[str, ...]
    source_receipt: str
    object_id: str
    protected: bool = False
    valid_until: int = MAX_VERSION


@dataclass(frozen=True)
class Row:
    status: str
    canonical: bytes
    key: bytes
    value: bytes
    metadata: bytes

    @property
    def fields(self):
        return json.loads(self.metadata)

    @property
    def receipt(self):
        return digest(dict(status=self.status, metadata=self.metadata.hex(),
                           canonical=self.canonical.hex(), key=self.key.hex(), value=self.value.hex()))


@dataclass(frozen=True)
class Selector:
    epoch: int
    tenant: str
    slot: int
    object_id: str
    version: int
    receipt: str
    view: str
    object_version: int
    incarnation: str
    recovery_id: str = ""


@dataclass(frozen=True)
class Proposal:
    """A semantic proposal has no authority/version/security fields."""
    content: bytes
    operation: str = "upsert"


@dataclass(frozen=True)
class Candidate:
    predecessor: str
    epoch: int
    slot: int
    row: Row


class Memory:
    def __init__(self, profile: ArchitectureProfile, encoder: Encoder, *,
                 semantic_gate: Callable[[bytes, bytes, bytes], bool] | None = None):
        self.profile, self.encoder, self.semantic_gate = profile, encoder, semantic_gate
        self._rows: tuple[Row | None, ...] = (None,) * profile.memory_slots
        self.version = self.epoch = 0
        self.recovery_id = ""
        self._pending: dict[tuple[int, str], Candidate] = {}

    @property
    def identity(self):
        content = dict(profile=self.profile.identity, encoder=self.encoder.identity,
                       epoch=self.epoch, version=self.version,
                       rows=[None if r is None else r.receipt for r in self._rows])
        if self.recovery_id: content["recovery_id"] = self.recovery_id
        return digest(content)

    def _slot(self, slot):
        if type(slot) is not int or not 0 <= slot < self.profile.memory_slots:
            raise ValueError("slot outside bounded table")

    def _validate(self, row: Row):
        p = self.profile
        if row.status not in ("live", "tombstone", "quarantine", "purged"):
            raise ValueError("invalid row status")
        if len(row.canonical) > p.canonical_bytes or len(row.metadata) > p.metadata_bytes:
            raise ValueError("fixed row capacity exceeded")
        if len(row.key) != p.key_width * 4 or len(row.value) != p.neural_value_bytes:
            raise ValueError("neural shape mismatch")
        import math
        from .token_memory import unpack_value
        semantic,tokens=unpack_value(p,row.value)
        if row.status=='live' and p.memory_tokens and not tokens:raise ValueError('live token capsule is empty')
        if not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", row.key + semantic)):
            raise ValueError("nonfinite neural bytes")
        f = row.fields
        required = {"tenant", "acl", "source_receipt", "object_id", "protected", "valid_until",
                    "version", "encoder", "parent", "semantic_qualified", "object_version", "incarnation"}
        if set(f) != required or f["encoder"] != self.encoder.identity:
            raise ValueError("metadata/encoder mismatch")
        if type(f["version"]) is not int or not 0 < f["version"] <= MAX_VERSION:
            raise ValueError("version outside unsigned range")
        if type(f["object_version"]) is not int or not 0 < f["object_version"] <= MAX_VERSION or type(f["incarnation"]) is not str or len(f["incarnation"]) != 64:
            raise ValueError("object version/incarnation invalid")
        if type(f["valid_until"]) is not int or not 0 <= f["valid_until"] <= MAX_VERSION:
            raise ValueError("invalid validity boundary")
        if any(type(f[k]) is not str or not f[k] for k in ("tenant", "object_id", "source_receipt")):
            raise ValueError("missing trusted provenance/identity")
        if type(f["protected"]) is not bool or type(f["semantic_qualified"]) is not bool:
            raise ValueError("invalid gate/protection bit")
        if type(f["acl"]) is not list or not f["acl"] or any(type(x) is not str or not x for x in f["acl"]):
            raise ValueError("invalid ACL")
        if row.status == "live":
            if canonical(json.loads(row.canonical)) != row.canonical:
                raise ValueError("noncanonical row content")
            expected = self.encoder.encode(row.canonical)
            if expected != (row.key, row.value):
                raise ValueError("canonical/neural byte mismatch")
            if f["semantic_qualified"]:
                if self.semantic_gate is None or self.semantic_gate(row.canonical, row.key, row.value) is not True:
                    raise ValueError("semantic qualification cannot be reconstructed")
        elif row.canonical or any(row.key) or any(row.value) or f["semantic_qualified"]:
            raise ValueError("nonlive row must suppress both content views")

        validator=getattr(self.encoder,'validate_row',None)
        if validator is not None:validator(row)

    def assemble(self, slot: int, proposal: Proposal, authority: Authority) -> Candidate:
        """Trusted caller supplies Authority; semantic proposer cannot set it.

        Invalid payload raises a typed failure before a feasible action set exists.
        The caller may record a rejection/NULL; it may not call it a scored action.
        """
        self._slot(slot)
        if type(proposal) is not Proposal or proposal.operation not in ("upsert", "delete", "quarantine"):
            raise ValueError("invalid proposal")
        if type(authority) is not Authority or type(authority.protected) is not bool:
            raise ValueError("trusted authority required")
        old = self._rows[slot]
        if old is not None:
            self._validate(old)
            f = old.fields
            if f["tenant"] != authority.tenant or f["protected"]:
                raise PermissionError("foreign tenant or protected row")
            version, parent = f["version"] + 1, old.receipt
        else:
            version, parent = 1, "EMPTY"
        if version > MAX_VERSION or self.version == MAX_VERSION:
            raise OverflowError("explicit namespace migration required")
        same = old is not None and old.fields["object_id"] == authority.object_id
        object_version = old.fields["object_version"]+1 if same else 1
        incarnation = old.fields["incarnation"] if same else digest(dict(predecessor=self.identity, object=authority.object_id, slot=slot))
        if object_version > MAX_VERSION: raise OverflowError("object version exhausted")
        if old is not None and old.status == "purged" and same:
            raise PermissionError("purged incarnation cannot be resurrected")
        if proposal.operation != "upsert" and not same:
            raise ValueError("delete/quarantine must address the existing object")
        if proposal.operation == "upsert":
            if len(proposal.content) > self.profile.canonical_bytes:
                raise ValueError("canonical capacity exceeded")
            if canonical(json.loads(proposal.content)) != proposal.content:
                raise ValueError("proposal is not canonical JSON")
            decoded = json.loads(proposal.content)
            if type(decoded) is dict and set(decoded) == {'alias_object'}:
                raise PermissionError('typed aliases require a lifecycle grant')
            key, value = self.encoder.encode(proposal.content)
            qualified = self.semantic_gate is not None and self.semantic_gate(proposal.content, key, value) is True
            status, content = "live", proposal.content
        else:
            if proposal.content:
                raise ValueError("deletion/quarantine cannot carry replacement content")
            status = "tombstone" if proposal.operation == "delete" else "quarantine"
            content, key, value, qualified = b"", bytes(4*self.profile.key_width), bytes(self.profile.neural_value_bytes), False
        fields = asdict(authority)
        fields.update(version=version, object_version=object_version, incarnation=incarnation, encoder=self.encoder.identity, parent=parent, semantic_qualified=qualified)
        row = Row(status, content, key, value, wire(fields))
        self._validate(row)
        return self._issue(slot, row)

    def _issue(self, slot, row):
        self._slot(slot)
        self._validate(row)
        f = row.fields
        if row.status == "live" and any(r is not None and r.status == "live" and i != slot and r.fields["tenant"] == f["tenant"] and r.fields["object_id"] == f["object_id"] for i,r in enumerate(self._rows)):
            raise ValueError("duplicate live object: infeasible before pricing")
        candidate = Candidate(self.identity, self.epoch, slot, row)
        key = (slot, row.receipt)
        if key not in self._pending and len(self._pending) >= self.profile.max_candidates:
            raise ValueError("candidate capacity exhausted")
        self._pending[key] = candidate
        return candidate

    def commit(self, candidate: Candidate | None):
        if candidate is None:
            return dict(action="NULL", predecessor=self.identity, successor=self.identity)
        self._slot(candidate.slot)
        if candidate.predecessor != self.identity or candidate.epoch != self.epoch:
            raise ValueError("stale candidate must be reconstructed from its own predecessor")
        if self._pending.get((candidate.slot, candidate.row.receipt)) != candidate:
            raise ValueError("candidate was not issued by this trusted assembler")
        self._validate(candidate.row)
        old = self._rows[candidate.slot]
        expected_version = 1 if old is None else old.fields["version"] + 1
        if candidate.row.fields["version"] != expected_version:
            raise ValueError("nonmonotone row version")
        f = candidate.row.fields
        if candidate.row.status == "live" and any(r is not None and r.status == "live" and i != candidate.slot
               and r.fields["tenant"] == f["tenant"] and r.fields["object_id"] == f["object_id"]
               for i, r in enumerate(self._rows)):
            raise ValueError("duplicate live object identity")
        if self.version == MAX_VERSION:
            raise OverflowError("memory version exhausted")
        before = self.identity
        rows = list(self._rows)
        rows[candidate.slot] = candidate.row
        self._rows = tuple(rows)
        self.version += 1
        self._pending.clear()
        return dict(action="WRITE", slot=candidate.slot, predecessor=before, successor=self.identity)

    def selectors(self, *, tenant: str, principal: str, clock: int, view: str):
        if view not in ("canonical", "neural"):
            raise ValueError("explicit view required")
        result = []
        for slot, row in enumerate(self._rows):
            if row is None or row.status != "live":
                continue
            f = row.fields
            s = Selector(self.epoch, tenant, slot, f["object_id"], f["version"], row.receipt, view, f["object_version"], f["incarnation"], self.recovery_id)
            if self.read(s, principal=principal, clock=clock) is not None:
                result.append(s)
        return tuple(result)

    def read(self, selector: Selector, *, principal: str, clock: int):
        self._slot(selector.slot)
        if selector.view not in ("canonical", "neural") or type(clock) is not int or clock < 0:
            raise ValueError("invalid read contract")
        row = self._rows[selector.slot]
        if row is None or row.status != "live" or selector.epoch != self.epoch or selector.recovery_id != self.recovery_id:
            return None
        try:
            self._validate(row)
        except (ValueError, TypeError, KeyError):
            return None  # Read suppression never mutates authority.
        f = row.fields
        if (selector.tenant != f["tenant"] or selector.object_id != f["object_id"]
                or selector.version != f["version"] or selector.receipt != row.receipt
                or selector.object_version != f["object_version"] or selector.incarnation != f["incarnation"]
                or principal not in f["acl"] or clock > f["valid_until"]):
            return None
        if selector.view == "neural":
            return (row.key, row.value) if f["semantic_qualified"] else None
        return row.canonical

    def snapshot(self):
        validator=getattr(self.encoder,'validate_export',None)
        if validator is not None:
            validator()
            for row in self._rows:
                if row is not None:self._validate(row)
        return dict(profile=self.profile.identity, encoder=self.encoder.identity,
                    epoch=self.epoch, version=self.version, recovery_id=self.recovery_id,
                    rows=[None if r is None else dict(status=r.status, canonical=r.canonical.hex(),
                          key=r.key.hex(), value=r.value.hex(), metadata=r.metadata.hex()) for r in self._rows])

    def restore(self, snapshot, *, rollback=True):
        if not rollback and (self.version != 0 or self.epoch != 0 or any(self._rows)):
            raise ValueError("exact import is only permitted into a fresh instance")
        if snapshot["profile"] != self.profile.identity or snapshot["encoder"] != self.encoder.identity:
            raise ValueError("snapshot profile identity mismatch")
        if len(snapshot["rows"]) != self.profile.memory_slots:
            raise ValueError("snapshot table shape mismatch")
        for k in ("epoch", "version"):
            if type(snapshot[k]) is not int or not 0 <= snapshot[k] <= MAX_VERSION:
                raise ValueError("snapshot namespace mismatch")
        recovery = snapshot.get("recovery_id", "")
        if type(recovery) is not str or (recovery and (len(recovery)!=32 or any(c not in "0123456789abcdef" for c in recovery))):
            raise ValueError("invalid recovery namespace")
        rows = []
        for value in snapshot["rows"]:
            row = None if value is None else Row(value["status"], *(bytes.fromhex(value[k]) for k in ("canonical", "key", "value", "metadata")))
            if row is not None:
                self._validate(row)
            rows.append(row)
        live = [(x.fields["tenant"], x.fields["object_id"]) for x in rows if x is not None and x.status == "live"]
        if len(live) != len(set(live)):
            raise ValueError("snapshot contains duplicate live objects")
        epoch = max(self.epoch, snapshot["epoch"]) + 1 if rollback else snapshot["epoch"]
        if epoch > MAX_VERSION:
            raise OverflowError("namespace exhausted")
        if rollback:
            # OS entropy is independent of the saved learning RNG. Even loading
            # the same snapshot into fresh objects invalidates prior handles.
            prior = {recovery, self.recovery_id}
            while True:
                recovery = uuid.uuid4().hex
                if recovery not in prior: break
        self._rows, self.epoch, self.version = tuple(rows), epoch, snapshot["version"]
        self.recovery_id = recovery
        self._pending.clear()

    def fork(self):
        child = Memory(self.profile, self.encoder, semantic_gate=self.semantic_gate)
        child.restore(self.snapshot(), rollback=False)
        child._pending = dict(self._pending)
        return child

    def packed_rows(self):
        """Resident accounting format; JSON snapshots are separately billed."""
        p, result = self.profile, bytearray()
        for row in self._rows:
            if row is None:
                result.extend(bytes(p.row_bytes))
                continue
            self._validate(row)
            result.extend(bytes([{"live": 1, "tombstone": 2, "quarantine": 3, "purged": 4}[row.status]]))
            result.extend(struct.pack("<I", len(row.canonical)))
            result.extend(row.canonical.ljust(p.canonical_bytes, b"\0"))
            result.extend(row.key + row.value)
            result.extend(struct.pack("<I", len(row.metadata)))
            result.extend(row.metadata.ljust(p.metadata_bytes, b"\0"))
        return bytes(result)
