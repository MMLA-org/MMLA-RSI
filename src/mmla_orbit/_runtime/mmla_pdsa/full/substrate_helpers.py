import math
import hashlib

def lora_linear(torch, original, rank, alpha, generator):
    """Standard linear LoRA: frozen W(x) + alpha/r * B(A(x)); zero initial B."""
    class LinearLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = original
            original.requires_grad_(False)
            self.down = torch.nn.Parameter(torch.empty(rank, original.in_features,
                dtype=torch.float32, device=original.weight.device))
            self.up = torch.nn.Parameter(torch.zeros(original.out_features, rank,
                dtype=torch.float32, device=original.weight.device))
            with torch.no_grad():
                self.down.uniform_(-1/math.sqrt(original.in_features),
                                    1/math.sqrt(original.in_features), generator=generator)
            self.scaling = alpha/rank

        def forward(self, value):
            frozen = self.original(value)
            delta = ((value.float() @ self.down.T) @ self.up.T) * self.scaling
            return frozen + delta.to(frozen.dtype)

    if not isinstance(original, torch.nn.Linear) or rank < 1 or alpha <= 0:
        raise ValueError('LoRA requires an ordinary existing Linear and positive rank/scale')
    return LinearLoRA()


def tensors_hash(torch, named):
    """Stream one tensor at a time; no full theta CPU clone or checkpoint."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named.items()):
        digest.update(str((name, list(parameter.shape), str(parameter.dtype))).encode())
        data = parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(data).cast('B'))
    return digest.hexdigest()


