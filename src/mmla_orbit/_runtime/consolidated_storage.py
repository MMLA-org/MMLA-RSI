"""One physical weight archive; task and qualification roles remain distinct."""
from safetensors.torch import load_file
from mmla_pdsa.full.safe_state import decode

FORMAT = 'mmla-orbit-consolidated-v1'


def load_consolidated(root, config):
    """Decode tensors after the owning loader verifies the package manifest."""
    if (config['model_type'] != 'mmla_orbit'
            or config['weights_format'] != FORMAT):
        raise ValueError('consolidated Orbit configuration required')
    tensors = load_file(str(root / 'model.safetensors'), device='cpu')
    base = {k[6:]: v for k, v in tensors.items() if k.startswith('mixer.')}
    omega = {k[6:]: v for k, v in tensors.items() if k.startswith('omega.')}
    used = {'mixer.' + k for k in base} | {'omega.' + k for k in omega}
    states = {name: decode(tree, tensors, used)
              for name, tree in config['tensor_trees'].items()}
    expected = {'roles.' + seed + '.' + role for seed in config['roles']
                for role in ('task', 'proposal', 'original_proposal')}
    if 'qualification' in config:
        expected.add('qualification')
    if 'reader_calibration' in config:
        expected.add('calibration.readout')
    if (not base or not omega or set(states) != expected
            or set(config['role_metadata']) != set(config['roles'])
            or used != set(tensors)):
        raise ValueError('complete consolidated model and state inventory required')
    return base, omega, states
