"""Amplitude representations shared by the Rice and DAD ablation benchmarks.

All learned alternatives return 16 features by default. Their encoder parameter
counts are matched as closely as a one-hidden-layer MLP permits to the existing
AmplitudeDeepSet. Matching parameters does not match FLOPs: Deep Sets applies its
element network to each measurement. No pretrained parameters are loaded here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from modules.dad.encoder_amp import AmplitudeDeepSet


ENCODER_KINDS = ("deepsets", "raw", "mean_var", "expanded_stats")


def _positive_root(value: torch.Tensor, exponent: float) -> torch.Tensor:
    """Exact clipped root with a finite, zero derivative on the clipped branch.

    Computing ``clamp(value, min=0).pow(exponent)`` directly has an infinite
    derivative at zero for exponents below one. Evaluating the inactive branch
    at one prevents NaNs when statistics are differentiated through by DAD.
    """
    positive = value > 0
    safe_value = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe_value.pow(exponent), torch.zeros_like(value))


def known_sigma_estimate(r: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Estimate nu from m2 with known complex-noise RMS sigma.

    ``r`` has shape ``[..., N]`` and sigma is scalar or broadcastable to ``[...]``.
    A trailing singleton measurement axis in sigma is accepted as well. This is
    an oracle comparator if learned models are not supplied the noise level.
    """
    m2 = r.square().mean(dim=-1)
    sigma = torch.as_tensor(sigma, dtype=r.dtype, device=r.device)
    if sigma.ndim == r.ndim and sigma.shape[-1] == 1:
        sigma = sigma.squeeze(-1)
    return _positive_root(m2 - sigma.square(), 0.5)


def moment_estimate(r: torch.Tensor) -> torch.Tensor:
    """Unknown-noise estimate: max(2*m2**2 - m4, 0)**(1/4)."""
    squared = r.square()
    m2 = squared.mean(dim=-1)
    m4 = squared.square().mean(dim=-1)
    return _positive_root(2 * m2.square() - m4, 0.25)


def amplitude_statistics(r: torch.Tensor, expanded: bool = False) -> torch.Tensor:
    """Return mean/variance, or the repository's six hand-crafted features.

    Expanded ordering is mean, population variance, skewness, non-excess
    kurtosis, moment estimate of nu**2, moment estimate of sigma**2. Standardized
    moments use a small variance floor and clipped moment estimates can be zero.
    The floor only stabilizes arithmetic; no dataset statistics are fitted.
    """
    mean = r.mean(dim=-1, keepdim=True)
    centered = r - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    if not expanded:
        return torch.cat((mean, variance), dim=-1)

    epsilon = max(1e-8, torch.finfo(r.dtype).tiny)
    standardized = centered / torch.sqrt(variance + epsilon)
    skew = standardized.pow(3).mean(dim=-1, keepdim=True)
    kurtosis = standardized.pow(4).mean(dim=-1, keepdim=True)
    squared = r.square()
    m2 = squared.mean(dim=-1, keepdim=True)
    m4 = squared.square().mean(dim=-1, keepdim=True)
    nu2 = _positive_root(2 * m2.square() - m4, 0.5)
    sigma2 = (m2 - nu2).clamp_min(0)
    return torch.cat((mean, variance, skew, kurtosis, nu2, sigma2), dim=-1)


def _encoder_parameter_budget(output_dim: int) -> int:
    # Inspect the unchanged production architecture without allocating weights
    # or consuming the caller's random-number stream.
    with torch.device("meta"):
        reference = AmplitudeDeepSet(output_dim=output_dim)
    return sum(parameter.numel() for parameter in reference.parameters())


class _MLPAmplitudeEncoder(nn.Module):
    def __init__(self, kind: str, num_amplitudes: int, output_dim: int):
        super().__init__()
        self.kind = kind
        self.num_amplitudes = num_amplitudes
        feature_dim = {"raw": num_amplitudes, "mean_var": 2, "expanded_stats": 6}[kind]
        budget = _encoder_parameter_budget(output_dim)
        # A Linear(d,h)->Linear(h,o) MLP has h*(d+o+1)+o parameters.
        per_hidden = feature_dim + output_dim + 1
        max_width = max(1, math.ceil((budget - output_dim) / per_hidden))
        self.hidden_dim = min(
            range(1, max_width + 1),
            key=lambda width: abs(width * per_hidden + output_dim - budget),
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if r.shape[-1] != self.num_amplitudes:
            raise ValueError(f"Expected {self.num_amplitudes} amplitudes; got {r.shape[-1]}")
        if self.kind == "raw":
            features = r
        else:
            features = amplitude_statistics(r, expanded=self.kind == "expanded_stats")
        return self.net(features)


def make_encoder(kind: str, num_amplitudes: int = 127, output_dim: int = 16) -> nn.Module:
    """Create a fresh encoder mapping ``[..., num_amplitudes]`` to ``[..., output_dim]``.

    Kinds: ``deepsets``, ``raw``, ``mean_var``, ``expanded_stats``. At the default
    dimensions the encoder counts are respectively 5360, 5344, 5355, and 5352.
    The raw MLP uses amplitude order; the other three are permutation invariant.
    """
    if kind not in ENCODER_KINDS:
        raise ValueError(f"Unknown encoder {kind!r}; choose one of {ENCODER_KINDS}")
    if num_amplitudes < 1 or output_dim < 1:
        raise ValueError("num_amplitudes and output_dim must be positive")
    if kind == "deepsets":
        return AmplitudeDeepSet(output_dim=output_dim)
    return _MLPAmplitudeEncoder(kind, num_amplitudes, output_dim)


class AmplitudeRegressor(nn.Module):
    """A chosen amplitude encoder followed by the same positive regression head."""

    def __init__(self, kind: str, num_amplitudes: int = 127):
        super().__init__()
        self.encoder = make_encoder(kind, num_amplitudes=num_amplitudes)
        self.head = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(r)).squeeze(-1)
