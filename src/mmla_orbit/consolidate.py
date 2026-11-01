"""Losslessly consolidate a verified legacy Orbit package into one weight file.

Run ``python -m mmla_orbit.consolidate SOURCE DEST --runtime-revision COMMIT``.
The output is a new directory; the source is never modified. Loading still uses
``mmla_orbit.load_orbit``. This is storage conversion, not an adapter weight merge.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil
import sys

from ._runtime.orbit import verify_package, verify_runtime, file_sha


def consolidate(source, destination, *, runtime_revision):
    if (len(runtime_revision) != 40
            or any(c not in '0123456789abcdef' for c in runtime_revision)):
        raise ValueError('exact runtime Git revision required')
    source, original_manifest = verify_package(source)
    runtime, _, runtime_identity = verify_runtime()
    sys.path.insert(0, str(runtime))
    from safetensors.torch import load_file, save_file
    from mmla_pdsa.full.safe_state import SCHEMA, decode
    from consolidated_storage import FORMAT

    config = json.loads((source / 'orbit_config.json').read_bytes())
    if config['schema'] != 'mmla-orbit-process-credit-config-v1':
        raise ValueError('original process-credit configuration required')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    tensors = {}
    index = json.loads((source / 'base/model.safetensors.index.json').read_bytes())
    base = {}
    for name in sorted(set(index['weight_map'].values())):
        values = load_file(str(source / 'base' / name), device='cpu')
        if base.keys() & values.keys():
            raise ValueError('duplicate base tensor')
        if any(index['weight_map'].get(key) != name for key in values):
            raise ValueError('base index does not bind its actual tensors')
        base.update(values)
    if set(base) != set(index['weight_map']):
        raise ValueError('complete base index required')
    tensors.update({'mixer.' + k: v for k, v in base.items()})
    tensors.update({'omega.' + k: v for k, v in
                    load_file(str(source / 'omega.safetensors')).items()})
    trees = {}
    stems = ['roles/' + seed + '/' + role for seed in config['roles']
             for role in ('task', 'proposal', 'original_proposal')]
    if 'qualification' in config:
        stems.append('qualification')
    if 'reader_calibration' in config:
        stems.append(config['reader_calibration']['stem'])
    for stem in stems:
        document = json.loads((source / (stem + '.json')).read_bytes())
        if set(document) != {'schema', 'tree'} or document['schema'] != SCHEMA:
            raise ValueError('typed state required')
        values = load_file(str(source / (stem + '.safetensors')), device='cpu')
        used = set()
        decode(document['tree'], values, used)
        if used != set(values):
            raise ValueError('unreferenced original tensor')
        prefix = stem.replace('/', '.')
        # Preserve the original tree and dtype exactly; only tensor keys change.
        tree = copy.deepcopy(document['tree'])
        pending = [tree]
        while pending:
            node = pending.pop()
            if node['type'] == 'tensor':
                node['key'] = prefix + '.' + node['key']
            elif node['type'] == 'dict':
                pending.extend(row[1] for row in node['items'])
            elif node['type'] in ('list', 'tuple'):
                pending.extend(node['items'])
        tensors.update({prefix + '.' + k: v for k, v in values.items()})
        trees[prefix] = tree
    config.update(model_type='mmla_orbit', architectures=['Orbit'],
        weights_format=FORMAT, tensor_trees=trees,
        backbone_config=json.loads((source / 'base/config.json').read_bytes()),
        role_metadata={seed: json.loads((source / 'roles' / seed / 'ROLES.json').read_bytes())
                       for seed in config['roles']},
        public_runtime_identity=runtime_identity,
        public_runtime_version='0.2.3', public_runtime_revision=runtime_revision,
        storage_provenance={'source_manifest_sha256':file_sha(source / 'MANIFEST.json'),
                            'source_runtime_identity':json.loads((source / 'orbit_config.json').read_bytes())['public_runtime_identity'],
                            'conversion':'unchanged tensors; namespace-only consolidation'})
    # Preserve public documents; update runtime binding and storage attribution.
    destination.mkdir(parents=True)
    for row in original_manifest['files']:
        rel = row['path']
        if (rel.endswith('.safetensors') or rel == 'orbit_config.json'
                or rel.startswith(('roles/', 'calibration/'))
                or rel == 'qualification.json'
                or rel in ('base/config.json', 'base/model.safetensors.index.json')):
            continue
        public_path = rel[5:] if rel.startswith('base/') else rel
        if public_path == 'UPSTREAM_README.md':
            public_path = 'licenses/UPSTREAM_README.md'
        target = destination / public_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
    if (source / '.gitattributes').is_file():
        shutil.copyfile(source / '.gitattributes', destination / '.gitattributes')
    save_file(tensors, str(destination / 'model.safetensors'),
              metadata={'format':'pt', 'schema':FORMAT})
    (destination / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    deployment = json.loads((destination / 'deployment.json').read_bytes())
    deployment['model']['runtime'] = runtime_identity
    (destination / 'deployment.json').write_text(json.dumps(deployment, indent=2) + '\n')
    (destination / 'requirements.txt').write_text(
        '--extra-index-url https://download.pytorch.org/whl/cu128\n'
        'mmla-rsi[orbit] @ git+https://github.com/MMLA-org/MMLA-RSI.git@' + runtime_revision + '\n')
    notice = destination / 'NOTICE'
    if notice.is_file():
        text = notice.read_text().replace('The base/ model and tokenizer files',
            'The model substrate and tokenizer files').replace(
            'base/UPSTREAM_README.md', 'licenses/UPSTREAM_README.md').replace(
            'The original base weights/configuration/tokenizer are copied byte-for-byte.',
            'Original tensor values and tokenizer bytes are unchanged. The base configuration\n'
            'is retained inside config.json, and model tensors are consolidated into\n'
            'model.safetensors under separate namespaces.').replace(
            'results/PROVENANCE.json records the substrate, runtime, result and review identities.',
            'results/PROVENANCE.json retains the original substrate, runtime, result and review\n'
            'identities as historical provenance. config.json and MANIFEST.json bind this\n'
            'consolidated package and its matching runtime.')
        notice.write_text(text)
    manifest = dict(original_manifest)
    manifest['files'] = [dict(path=p.relative_to(destination).as_posix(),
        bytes=p.stat().st_size, sha256=file_sha(p))
        for p in sorted(destination.rglob('*'))
        if p.is_file() and p.name not in ('MANIFEST.json', '.gitattributes')]
    (destination / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify_package(destination)
    return {'destination':str(destination), 'tensors':len(tensors),
            'model_bytes':(destination / 'model.safetensors').stat().st_size,
            'source_manifest':config['storage_provenance']['source_manifest_sha256'],
            'runtime_identity':runtime_identity}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--runtime-revision', required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.source, args.destination,
                                runtime_revision=args.runtime_revision), indent=2))
