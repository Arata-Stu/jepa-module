import torch
import torch.nn as nn


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Wrapper used by exported V-JEPA modules."""
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def repeat_interleave_batch(
    x: torch.Tensor, batch_size: int, repeat: int
) -> torch.Tensor:
    """
    Repeat each batch entry `repeat` times.
    Input shape: [B, ...], where B == batch_size.
    Output shape: [B * repeat, ...].
    """
    if x.shape[0] != batch_size:
        raise ValueError(
            f"Expected batch dimension {batch_size}, got {x.shape[0]}"
        )
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    return torch.repeat_interleave(x, repeats=repeat, dim=0)
