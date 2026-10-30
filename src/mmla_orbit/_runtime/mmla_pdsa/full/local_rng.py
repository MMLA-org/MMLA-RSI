"""One CUDA device per worker; never initialize every visible GPU for RNG I/O."""
import torch


def capture_cuda():
    if not torch.cuda.is_initialized():return None
    index=torch.cuda.current_device()
    return dict(device_index=index,state=torch.cuda.get_rng_state(index).cpu().clone())


def validate_cuda(state):
    if state is None:
        if torch.cuda.is_initialized():raise ValueError('missing initialized worker CUDA RNG')
        return
    if (type(state) is not dict or set(state)!={'device_index','state'}
            or type(state['device_index']) is not int or not torch.cuda.is_initialized()
            or state['device_index']!=torch.cuda.current_device()):
        raise ValueError('local CUDA RNG topology mismatch')
    value=state['state']
    if not torch.is_tensor(value) or value.device.type!='cpu' or value.dtype!=torch.uint8 or value.ndim!=1:
        raise ValueError('CPU byte tensor required for CUDA RNG')
    # Validate against this worker's existing device only, without publishing it.
    torch.Generator(device=torch.device('cuda',state['device_index'])).set_state(value)


def restore_cuda(state):
    validate_cuda(state)
    if state is not None:torch.cuda.set_rng_state(state['state'],device=state['device_index'])
