"""Strict numerical execution, without a new loss or tolerance relaxation."""
import os
POLICY='strict-deterministic-v1'
WORKSPACE=':4096:8'


def settings():
    import torch
    return dict(policy=POLICY,torch=torch.__version__,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
        sdpa=dict(flash=torch.backends.cuda.flash_sdp_enabled(),
            efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),cudnn=torch.backends.cuda.cudnn_sdp_enabled()))


def require():
    value=settings()
    expected=dict(policy=POLICY,cublas_workspace_config=WORKSPACE,
        deterministic_algorithms=True,warn_only=False,cudnn_deterministic=True,
        cudnn_benchmark=False,matmul_precision='highest',matmul_tf32=False,
        cudnn_tf32=True,bf16_reduced_precision_reduction=True,
        fill_uninitialized_memory=True,
        sdpa=dict(flash=True,efficient=True,math=True,cudnn=True))
    if any(value[k]!=v for k,v in expected.items()):
        raise RuntimeError('registered deterministic numerical execution policy changed')
    return value


def establish(source):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('deterministic policy must precede CUDA context initialization')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=WORKSPACE:
        raise RuntimeError('fixed cuBLAS workspace must be present at process creation')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.utils.deterministic.fill_uninitialized_memory=True
    # Automatic SDPA dispatch stays enabled. Strict Torch determinism governs
    # supported kernels; unsupported operations must stop, never warn-and-run.
    value=require()
    return dict(source=source,settings=value,CUDA_initialized_at_configuration=False,
        unsupported_operation='STOP_AND_PRESERVE',strict_restore_tolerance=0,
        scalar_loss_logs_are_not_state_equivalence=True)
