import torch
import math

__all__ = [
    "beam_from_phases",
    "phase_to_beam",
    "steering_vector",
    "steering_vector_batch",
]


def _unit_beam_from_phases(eta, num_elements):
    return torch.exp(1j * eta) / math.sqrt(num_elements)


def phase_to_beam(eta):
    """
    Batch-oriented phase-to-beam transform.

    eta : [..., K]
    return : [..., K]
    """

    if eta.ndim < 1:
        raise ValueError(
            "eta must have at least one dimension, with antenna phases on the last axis"
        )

    return _unit_beam_from_phases(
        eta,
        eta.shape[-1],
    )


def steering_vector(theta, params):
    """
    Column-oriented steering matrix used by the EIG baseline.

    theta : any tensor shape
    return : [K, theta.numel()]
    """

    theta_flat = theta.reshape(-1)

    n = torch.arange(
        params.K,
        dtype=theta.dtype,
        device=theta.device
    ).reshape(-1, 1)

    theta_grid = theta_flat.reshape(1, -1)

    phase = (-1j * 2 * torch.pi * params.d * torch.cos(theta_grid) / params.lambda_c)

    a = torch.exp(n * phase)

    a = a / torch.linalg.vector_norm(
        a,
        dim=0,
        keepdim=True
    )

    return a


def steering_vector_batch(theta, params):
    """
    Batch-oriented steering vectors used by DAD.

    theta : [...]
    return : [..., K]
    """

    return steering_vector(
        theta,
        params,
    ).transpose(0, 1).reshape(
        theta.shape + (params.K,)
    )


def beam_from_phases(eta):
    """
    Column-oriented phase-to-beam transform used by the EIG baseline.

    eta [K] -> [K, 1], eta stack [K, R] -> [K, R]
    """

    if eta.ndim == 1:
        eta = eta[:, None]
    elif eta.ndim != 2:
        raise ValueError(
            "eta must be a column vector [K, 1] or a stack [K, R]"
        )

    K = eta.shape[0]

    return _unit_beam_from_phases(
        eta,
        K,
    )
