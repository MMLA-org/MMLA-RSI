"""Process-local reconciliation with the pinned native logical transition.

The upstream C interface returns all syntactic effects, including effects whose
conditions did not fire. These records must not drive observations or teachers.
Original PDDL, native transitions and task goals remain unchanged. Every actor
uses this same, separately identified backend adapter; this is not a claim of
byte-identical unmodified upstream observations.
"""

from collections import Counter, defaultdict


def native_facts(state, atom_type):
    count = state.downward_lib.get_state_size()
    if type(count) is not int or not 0 <= count <= 1000000:
        raise RuntimeError('bounded native logical state required')
    atoms = (atom_type * count)()
    state.downward_lib.get_state(atoms)
    signed, positive = [], set()
    for atom in atoms:
        if not (atom.name.startswith('Atom ') or atom.name.startswith('NegatedAtom ')):
            raise RuntimeError('primary SAS must contain complete signed singleton atoms')
        fact = atom.get_fact(state.name2type)
        base = fact.negate() if fact.is_negation else fact
        if base in positive:
            raise RuntimeError('duplicate native logical variable')
        positive.add(base); signed.append(fact)
    return tuple(signed), frozenset(positive)


def _reconcile(state, atom_type, *, initialize=False):
    signed, domain = native_facts(state, atom_type)
    if initialize:
        state._orbit_native_domain = domain
    elif domain != state._orbit_native_domain:
        raise RuntimeError('native fluent universe changed')
    before = set(state.facts)
    # Rebuild the four indexes from unique desired facts. Upstream add_fact
    # increments variable counts even for duplicates and remove_fact decrements
    # them even for absent facts; a blind two-polarity remove would corrupt them.
    desired = (before - domain - {p.negate() for p in domain}) | set(signed)
    state._facts = defaultdict(set)
    state._vars_by_name = {}
    state._vars_by_type = defaultdict(set)
    state._var_counts = Counter()
    state.add_facts(sorted(desired, key=lambda p: (p.name, tuple((a.name, a.type) for a in p.arguments))))
    return [fact for fact in signed if fact not in before]


def reconcile(state, atom_type, *, initialize=False):
    try:
        return _reconcile(state, atom_type, initialize=initialize)
    except Exception as exc:
        # PddlEnv.step catches ValueError as an invalid user command. A decode
        # or index failure after a real transition must instead abort the run.
        raise RuntimeError('authoritative native-state reconciliation failed') from exc


def install_native_state_adapter(logic):
    """Install once in an isolated worker, before PddlEnv.load or reset.

    The primary translator disables invariant groups: each variable is a
    positive/negative atom pair. Runtime checks reject other encodings. Both
    initial and post-action facts use the same complete signed representation.
    No third-party file is modified and no alternate task is created.
    """
    base = logic.PddlState
    if getattr(base, '_orbit_authoritative_state', False):
        raise RuntimeError('native state adapter already installed')

    class AuthoritativeState(base):
        _orbit_authoritative_state = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            reconcile(self, logic.Atom, initialize=True)

        def apply(self, action):
            operator = self._operators[action.id]
            # ABI storage remains complete, but these syntactic effects are
            # discarded. The actual native transition executes exactly once.
            effects = (logic.Atom * operator.nb_effect_atoms)()
            self.downward_lib.apply_operator(operator.id, effects)
            return reconcile(self, logic.Atom)

    logic.PddlState = AuthoritativeState
    return AuthoritativeState
