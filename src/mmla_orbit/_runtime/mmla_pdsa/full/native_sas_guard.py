"""Validate translated SAS before native fixed-buffer access; exact C ABI.

This adapter does not alter a task, action, transition, goal or planner result.
Out-of-envelope data aborts the candidate; it is never silently dropped/replaced.
ASCII is required because this pinned binding decodes native names as cp1252.
"""
import ctypes as C
import re

GUARD_SCHEMA = 'orbit-native-sas3-ascii-bounds-v1'
MAX_SAS_BYTES = 16*1024*1024
MAX_COUNT = 100000
MAX_TOTAL = 1000000


def validate_sas(data):
    if type(data) is not bytes or not 0 < len(data) <= MAX_SAS_BYTES or b'\x00' in data:
        raise RuntimeError('finite NUL-free SAS bytes required before native load')
    try:text=data.decode('ascii')
    except UnicodeDecodeError as exc:raise RuntimeError('ASCII native-name domain required') from exc
    if any(x!='\n' and not ' '<=x<='~' for x in text):
        raise RuntimeError('only LF and printable ASCII are permitted in SAS')
    lines=text.split('\n')
    if lines[-1]=='':lines.pop()
    if any(not line or line!=line.strip() or len(line)>1023 or '\r' in line for line in lines):
        raise RuntimeError('all SAS fields must fit the native 1024-byte buffer')
    index=0;total=0
    def take():
        nonlocal index
        if index>=len(lines):raise RuntimeError('incomplete SAS')
        value=lines[index];index+=1;return value
    def expect(value):
        if take()!=value:raise RuntimeError('unexpected SAS section')
    def ints(n=None):
        row=take().split()
        if n is not None and len(row)!=n:raise RuntimeError('SAS field count')
        if any(re.fullmatch(r'-?(?:0|[1-9][0-9]*)',x) is None for x in row):
            raise RuntimeError('canonical decimal SAS integer required')
        try:values=[int(x) for x in row]
        except ValueError as exc:raise RuntimeError('SAS integer required') from exc
        if any(abs(x)>2**31-1 for x in values):raise RuntimeError('SAS int32 overflow')
        return values
    def count():
        nonlocal total
        n,=ints(1)
        if not 0<=n<=MAX_COUNT:raise RuntimeError('bounded SAS count required')
        total+=n
        if total>MAX_TOTAL:raise RuntimeError('bounded total SAS entries required')
        return n
    expect('begin_version');expect('3');expect('end_version')
    expect('begin_metric');metric,=ints(1)
    if metric not in (0,1):raise RuntimeError('SAS metric flag')
    expect('end_metric');variables=count();domains=[]
    for variable_index in range(variables):
        expect('begin_variable');expect('var'+str(variable_index));layer,=ints(1)
        if not -1<=layer<=MAX_COUNT:raise RuntimeError('bounded axiom layer')
        size=count()
        if size<1:raise RuntimeError('nonempty SAS variable domain')
        domains.append(size)
        for _ in range(size):take()
        expect('end_variable')
    def fact(pair):
        if len(pair)!=2:raise RuntimeError('SAS fact width')
        var,value=pair
        if not 0<=var<variables or not 0<=value<domains[var]:raise RuntimeError('SAS fact range')
    for _ in range(count()):
        expect('begin_mutex_group')
        for _ in range(count()):fact(ints(2))
        expect('end_mutex_group')
    expect('begin_state')
    for var in range(variables):fact((var,ints(1)[0]))
    expect('end_state');expect('begin_goal')
    for _ in range(count()):fact(ints(2))
    expect('end_goal');operators=[]
    for _ in range(count()):
        expect('begin_operator');name=take()
        for _ in range(count()):fact(ints(2))
        effects=count()
        for _ in range(effects):
            values=ints();conditions=values[0] if values else -1
            if not 0<=conditions<=MAX_COUNT or len(values)!=2*conditions+4:raise RuntimeError('SAS conditional effect width')
            for i in range(conditions):fact(values[1+2*i:3+2*i])
            var,pre,post=values[-3:];fact((var,post))
            if pre!=-1:fact((var,pre))
        cost,=ints(1)
        if cost<0:raise RuntimeError('nonnegative SAS cost')
        expect('end_operator');operators.append((name,effects))
    for _ in range(count()):
        expect('begin_rule')
        for _ in range(count()):fact(ints(2))
        var,pre,post=ints(3);fact((var,pre));fact((var,post));expect('end_rule')
    if index!=len(lines):raise RuntimeError('trailing SAS data')
    return dict(variables=variables,operators=tuple(operators),total_entries=total)


class GuardedNative:
    """Only APIs used by this fixed PDDL environment; no hidden replanning."""
    def __init__(self,library,atom_type,operator_type):
        if (C.sizeof(atom_type)!=1024 or C.alignment(atom_type)!=1
                or dict(atom_type._fields_)!={'_name':C.c_char*1024}
                or atom_type._name.offset!=0
                or C.sizeof(operator_type)!=1032 or C.alignment(operator_type)!=4
                or dict(operator_type._fields_)!={'id':C.c_int,'_name':C.c_char*1024,'nb_effect_atoms':C.c_int}
                or (operator_type.id.offset,operator_type._name.offset,operator_type.nb_effect_atoms.offset)!=(0,4,1028)):
            raise RuntimeError('exact fixed native structure layout required')
        self._lib=library;self._primary=None;self._applicable=None
        signatures={
            'load_sas':([C.c_char_p],C.c_int),'load_sas_replan':([C.c_char_p],C.c_int),
            'cleanup':([],None),'get_state_size':([],C.c_int),
            'get_state':([C.POINTER(atom_type)],None),
            'get_applicable_operators_count':([],C.c_int),
            'get_applicable_operators':([C.POINTER(operator_type)],None),
            'apply_operator':([C.c_size_t,C.POINTER(atom_type)],C.c_size_t),
            'check_goal':([],C.c_bool)}
        for name,(args,result) in signatures.items():
            fn=getattr(library,name);fn.argtypes=args;fn.restype=result
    @property
    def _handle(self):return self._lib._handle
    def load_sas(self,data):
        qualified=validate_sas(data)
        if self._lib.load_sas(data)!=1:raise RuntimeError('native primary load failed')
        self._primary=qualified;self._applicable=None;return 1
    def load_sas_replan(self,data):
        validate_sas(data)
        if self._lib.load_sas_replan(data)!=1:raise RuntimeError('native replan payload load failed')
        return 1
    def get_state_size(self):
        value=self._lib.get_state_size()
        if self._primary is None or value!=self._primary['variables']:raise RuntimeError('native state size differs from qualified SAS')
        return value
    def get_state(self,out):
        if len(out)!=self.get_state_size():raise RuntimeError('exact native state allocation required')
        self._lib.get_state(out)
    def get_applicable_operators_count(self):
        value=self._lib.get_applicable_operators_count()
        if self._primary is None or not 0<=value<=len(self._primary['operators']):raise RuntimeError('native operator count exceeds qualified SAS')
        self._applicable=value;return value
    def get_applicable_operators(self,out):
        if self._applicable is None or len(out)!=self._applicable:raise RuntimeError('exact applicable operator allocation required')
        self._lib.get_applicable_operators(out)
        for op in out:
            if not 0<=op.id<len(self._primary['operators']):raise RuntimeError('native operator id out of range')
            if (op.name,op.nb_effect_atoms)!=self._primary['operators'][op.id]:raise RuntimeError('native operator identity differs from qualified SAS')
    def apply_operator(self,idx,effects):
        if type(idx) is not int or self._primary is None or not 0<=idx<len(self._primary['operators']):raise RuntimeError('qualified native operator required')
        expected=self._primary['operators'][idx][1]
        if len(effects)!=expected:raise RuntimeError('complete native effect allocation required')
        result=self._lib.apply_operator(idx,effects)
        if result!=expected:raise RuntimeError('native effect count differs')
        self._applicable=None;return result
    def check_goal(self):return self._lib.check_goal()
    def cleanup(self):self._lib.cleanup();self._primary=None;self._applicable=None
