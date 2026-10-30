"""Lossless data-only checkpoint trees; no pickle or executable class names."""
import json
import math
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

SCHEMA = 'mmla-safe-state-v1'

def _pairs(items):
    out = {}
    for k, v in items:
        if k in out: raise ValueError('duplicate JSON key')
        out[k] = v
    return out

def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def encode(value, tensors):
    if torch.is_tensor(value):
        if value.layout != torch.strided: raise TypeError('dense tensors only')
        key = 't' + str(len(tensors))
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {'type': 'tensor', 'key': key}
    if type(value) in (tuple, list):
        return {'type': 'tuple' if type(value) is tuple else 'list',
                'items': [encode(v, tensors) for v in value]}
    if type(value) is dict:
        if any(type(k) not in (str, int) for k in value):
            raise TypeError('only string/integer mapping keys')
        return {'type': 'dict', 'items': [[k, encode(v, tensors)] for k, v in value.items()]}
    if type(value) is bytes: return {'type': 'bytes', 'hex': value.hex()}
    if value is None or type(value) in (str, int, bool): return {'type': 'scalar', 'value': value}
    if type(value) is float and math.isfinite(value): return {'type': 'scalar', 'value': value}
    raise TypeError('not a finite data-only value: ' + type(value).__name__)

def decode(node, tensors, used):
    if type(node) is not dict: raise ValueError('typed node required')
    kind = node.get('type')
    if kind == 'tensor' and set(node) == {'type', 'key'}:
        key = node['key']
        if type(key) is not str or key in used or key not in tensors: raise ValueError('tensor reference')
        used.add(key)
        return tensors[key]
    if kind in ('tuple', 'list') and set(node) == {'type', 'items'} and type(node['items']) is list:
        values = [decode(v, tensors, used) for v in node['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'dict' and set(node) == {'type', 'items'} and type(node['items']) is list:
        out = {}
        for row in node['items']:
            if type(row) is not list or len(row) != 2: raise ValueError('mapping entry')
            key, value = row
            if type(key) not in (str, int) or key in out: raise ValueError('mapping key')
            out[key] = decode(value, tensors, used)
        return out
    if kind == 'bytes' and set(node) == {'type', 'hex'} and type(node['hex']) is str:
        raw = bytes.fromhex(node['hex'])
        if raw.hex() != node['hex']: raise ValueError('canonical bytes required')
        return raw
    if kind == 'scalar' and set(node) == {'type', 'value'}:
        value = node['value']
        if value is None or type(value) in (str, int, bool): return value
        if type(value) is float and math.isfinite(value): return value
    raise ValueError('invalid data-only node')

def save_state(stem, value):
    stem = Path(stem); meta = stem.with_suffix('.json'); weights = stem.with_suffix('.safetensors')
    if meta.exists() or weights.exists(): raise FileExistsError(stem)
    tensors = {}; tree = encode(value, tensors)
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(weights), metadata={'format': 'pt', 'schema': SCHEMA})
    with meta.open('x') as f:
        json.dump({'schema': SCHEMA, 'tree': tree}, f, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def load_state(stem):
    stem = Path(stem); data = read_json(stem.with_suffix('.json'))
    if set(data) != {'schema', 'tree'} or data['schema'] != SCHEMA: raise ValueError('state schema')
    tensors = load_file(str(stem.with_suffix('.safetensors')), device='cpu'); used = set()
    value = decode(data['tree'], tensors, used)
    if used != set(tensors): raise ValueError('unreferenced checkpoint tensor')
    return value

def exact_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b): return False
    if type(a) is dict: return a.keys() == b.keys() and all(exact_equal(v, b[k]) for k, v in a.items())
    if type(a) in (tuple, list): return len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b
