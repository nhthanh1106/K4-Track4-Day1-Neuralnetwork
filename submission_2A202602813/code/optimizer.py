"""Optimizer, scheduler, and pre-clip gradient norm helpers."""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {name!r}")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay must be non-negative")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    if name is None:
        return None
    if name == "cosine":
        if total_steps <= 0:
            raise ValueError("total_steps must be positive for a cosine scheduler")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, **kwargs
        )
    raise ValueError("supported scheduler is None or 'cosine'")


def clip_gradients(params, max_norm: float | None) -> float:
    """Return the global gradient norm measured before optional clipping."""
    if max_norm is not None and max_norm <= 0:
        raise ValueError("max_norm must be positive or None")
    params = list(params)
    grads = [parameter.grad.detach() for parameter in params if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(grad.float(), ord=2) for grad in grads])
    total_norm = torch.linalg.vector_norm(norms, ord=2)
    if max_norm is None:
        return float(total_norm.item())
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(norm.detach().item() if isinstance(norm, torch.Tensor) else norm)
