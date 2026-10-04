"""Compliant MLP architectures and initialization helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,
    (512, 256): 161_287,
    (256, 128, 64): 55_687,
}


class MLP(nn.Module):
    """A ReLU MLP that returns raw class logits."""

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not hidden or any(int(width) <= 0 for width in hidden):
            raise ValueError("hidden must contain positive layer widths")
        dims = [in_features, *(int(width) for width in hidden), num_classes]
        layers: list[nn.Module] = []
        for index, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(n_in, n_out, bias=True))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)
        expected = EXPECTED_PARAMS.get(tuple(hidden))
        if in_features == 54 and num_classes == 7 and expected is not None:
            assert count_params(self) == expected, (tuple(hidden), count_params(self), expected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Initialize every Linear layer; all supported schemes set bias to zero."""
    supported = {"zeros", "normal", "xavier", "he", "default"}
    if init not in supported:
        raise ValueError(f"init must be one of {sorted(supported)}, got {init!r}")
    if init == "default":
        return
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(module.weight)
        elif init == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif init == "he":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def activation_stats(model: MLP, x: torch.Tensor) -> list[float]:
    """Return activation standard deviations after each hidden ReLU at initialization."""
    was_training = model.training
    model.eval()
    h = x
    stats: list[float] = []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stats.append(float(h.std(unbiased=False).item()))
    model.train(was_training)
    return stats
