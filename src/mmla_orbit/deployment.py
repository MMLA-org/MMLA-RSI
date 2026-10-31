"""Load the model package's returned RTMA program, without running RSI."""
from pathlib import Path
import hashlib
import json


def load_manager(orbit, path):
    """Return the exact packaged manager for ``orbit.new_task_state(rule=...)``.

    The package manifest and actual loaded model own the binding. Source and
    result hashes in deployment.json document the completed run; loading this
    program does not repeat its evaluation or independently certify a benefit.
    The initial manager and a complete private controller remain separate.
    """
    path=Path(path)
    entry=next((row for row in orbit.metadata['package']['files']
                if row['path']=='deployment.json'),None)
    if entry is None or path.stat().st_size!=entry['bytes'] or entry['bytes']>256*1024:
        raise ValueError('model package must bind a bounded deployment.json')
    raw=path.read_bytes()
    if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
        raise ValueError('packaged deployment record changed')
    value=json.loads(raw)
    from orbit_rsi import model_binding
    from content_rules import parse_rule,identity
    if (value['schema']!='orbit-packaged-manager-v1' or identity(value['model'])!=identity(model_binding(orbit))
            or value['default_manager_rng']!=17):
        raise ValueError('returned manager belongs to a different Orbit binding')
    wire=value['manager'];language=orbit.interface.language
    if wire['mode']!='rtma' or wire['language']!=language.identity:
        raise ValueError('matched RTMA language required')
    return parse_rule(language,'rtma',tuple(wire['symbols']))
