"""Original BF16/eager execution for the prospective full-recompute delivery.

No cache-specific kernel, tensor replacement, or silent arithmetic fallback.
Frozen base weights continue to propagate input gradients through ordinary
PyTorch operations. Omega, rho and hybrid retain their original stored dtypes.
"""
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaAttention
ARITHMETIC_ID='original-pytorch-bf16-eager-dense-v1'
STATIC_ID='original-auxiliary-pytorch-linear-fp32-v1'
RUNTIME=dict(torch='2.11.0+cu128',transformers='5.12.1',float32_matmul_precision='highest',bf16_reduced_precision_reduction=True)


def _runtime():
    import transformers
    if (torch.__version__!=RUNTIME['torch'] or transformers.__version__!=RUNTIME['transformers']
            or torch.get_float32_matmul_precision()!='highest' or torch.backends.cuda.matmul.allow_tf32
            or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
        raise ValueError('declared ordinary dense runtime required')


def _projections(model,dtype):
    rows=[]
    for name,m in model.mixer.named_modules():
        if not isinstance(m,nn.Linear):continue
        if (type(m) is not nn.Linear or m.weight.dtype!=dtype or m.weight.requires_grad
                or m.bias is not None and (m.bias.dtype!=dtype or m.bias.requires_grad)):
            raise ValueError('ordinary frozen projection dtype/type required: '+name)
        rows.append(dict(name=name,shape=list(m.weight.shape),bias=m.bias is not None))
    layers=model.mixer.model.layers
    if (not rows or not layers or model.mixer.config._attn_implementation!='eager'
            or any(type(x.self_attn) is not LlamaAttention or x.self_attn.layer_idx!=i for i,x in enumerate(layers))):
        raise ValueError('original eager Llama attention and layer routing required')
    return rows


def identity(model):
    _runtime()
    return dict(identity=ARITHMETIC_ID,runtime=dict(RUNTIME),projections=_projections(model,torch.bfloat16),
        teacher='original frozen SFT PyTorch arithmetic',backward='ordinary PyTorch autograd; frozen weights, live input gradients')


def bind(model):
    # Validation only: never mutate the pristine/shared base or install kernels.
    return identity(model)


def promote_auxiliary(model,*,phase):
    if phase!='fixed-auxiliary-final' or any(p.requires_grad for p in model.parameters()):
        raise ValueError('closed frozen reader at fixed auxiliary boundary required')
    _runtime()
    dtype=next(m.weight.dtype for m in model.mixer.modules() if isinstance(m,nn.Linear))
    if dtype not in (torch.bfloat16,torch.float32):raise ValueError('declared dense dtype required')
    _projections(model,dtype)
    model.float()
    return dict(before=ARITHMETIC_ID if dtype==torch.bfloat16 else STATIC_ID,after=STATIC_ID,
        phase=phase,export_requires_fresh_original=True)
