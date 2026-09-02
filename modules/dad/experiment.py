import torch

from modules.beam_eig.array_model import (
    phase_to_beam,
    steering_vector_batch,
)

__all__ = [
    "phase_to_beam",
    "rollout",
    "steering_vector_batch",
]


def rollout(
    policy,
    theta,
    n_steps,
    observation_fn,
):
    """
    Generate a full adaptive DAD trajectory.

    Parameters
    ----------
    policy : DADPolicy

    theta : Tensor [B, ...]
        True channel parameter for each Monte-Carlo simulation.

    n_steps : int
        Total number of adaptive experiments.

    observation_fn : callable
        Function r = observation_fn(theta, b), where b is the complex beam
        generated from the policy phases. It must return either [B] for scalar
        observations or [B, Ns] for vector observations.

    Returns
    -------
    eta_history : Tensor [B, T, N]

    r_history : Tensor [B, T, Ns]
    """

    B = theta.shape[0]
    N = policy.design_dim
    Ns = policy.observation_dim

    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    theta = theta.to(
        device=device,
        dtype=dtype,
    )

    eta_history = torch.empty(
        B,
        0,
        N,
        device=device,
        dtype=dtype,
    )

    r_history = torch.empty(
        B,
        0,
        Ns,
        device=device,
        dtype=dtype,
    )

    for _ in range(n_steps):
        eta_t = policy(
            eta_history,
            r_history,
        )

        # Lambda layer: phases [B, N] -> constant-modulus beams [B, N].
        b_t = phase_to_beam(eta_t)

        r_t = observation_fn(
            theta,
            b_t,
        )

        if r_t.ndim == 0:
            r_t = r_t.reshape(1, 1)

        elif r_t.ndim == 1:
            r_t = r_t.unsqueeze(-1)

        elif r_t.ndim != 2:
            raise ValueError(
                "observation_fn must return observations shaped [B] or [B, Ns]."
            )

        if r_t.shape != (B, Ns):
            raise ValueError(
                "observation_fn returned an incompatible shape: "
                f"expected {(B, Ns)}, got {tuple(r_t.shape)}."
            )

        r_t = r_t.to(
            device=device,
            dtype=dtype,
        )

        eta_history = torch.cat(
            [
                eta_history,
                eta_t.unsqueeze(1),
            ],
            dim=1,
        )

        r_history = torch.cat(
            [
                r_history,
                r_t.unsqueeze(1),
            ],
            dim=1,
        )

    return eta_history, r_history
