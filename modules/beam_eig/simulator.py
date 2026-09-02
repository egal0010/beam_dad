import math
import torch

from modules.beam_eig.array_model import (
    steering_vector,
    beam_from_phases,
)


def simulate_y(alpha, s, sigma):
    """
    alpha :
        scalar      -> retourne [Ns]
        [N]         -> retourne [N, Ns]
        [... ]      -> retourne [..., Ns]

    s :
        [Ns]
    """

    # Ajoute une dimension à alpha avant la dimension des symboles
    mu = alpha[..., None] * s

    sigma = torch.as_tensor(
        sigma,
        dtype=mu.real.dtype,
        device=mu.device,
    )

    while sigma.ndim < mu.ndim:
        sigma = sigma.unsqueeze(-1)

    noise_real = torch.randn_like(mu.real)
    noise_imag = torch.randn_like(mu.real)

    w = sigma / torch.sqrt(
        torch.tensor(
            2.0,
            dtype=mu.real.dtype,
            device=mu.device
        )
    ) * (
        noise_real
        + 1j * noise_imag
    )

    return mu + w


def generate_amplitude_measurement(
    theta,
    rho,
    eta,
    s,
    sigma,
    params,
):
    a = steering_vector(theta, params)
    b = beam_from_phases(eta)

    alpha = rho * (
        b.conj().T @ a
    ).squeeze()

    y = simulate_y(alpha, s, sigma)

    return torch.abs(y)

def sigma_from_snr(s, snr_db, rho_true):
    Es = torch.mean(torch.abs(s) ** 2)

    snr_lin = 10.0 ** (snr_db / 10.0)

    sigma = torch.sqrt(
        rho_true**2 * Es / snr_lin
    )

    return sigma

