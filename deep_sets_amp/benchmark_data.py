"""Paired physical and direct-Rice data for the encoder ablation."""

from dataclasses import dataclass
import math

import torch

from modules.beam_eig.array_model import phase_to_beam, steering_vector_batch
from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.beam_eig.simulator import simulate_y


SCENARIOS = ("physical_fixed", "physical_snr", "uniform_fixed")
SNR_VALUES = (-10.0, -5.0, 0.0, 5.0, 10.0)


@dataclass
class Batch:
    r: torch.Tensor
    nu: torch.Tensor
    sigma: torch.Tensor
    gain: torch.Tensor
    theta: torch.Tensor
    rho: torch.Tensor
    eta: torch.Tensor
    reference_snr_db: torch.Tensor

    @property
    def effective_snr_db(self):
        return 20 * torch.log10((self.nu / self.sigma).clamp_min(1e-12))


class Sampler:
    """Independent RNG: resetting its seed pairs examples across architectures.

    physical_fixed: actual random beams, sigma=1.
    physical_snr: actual random beams, reference SNR uniform in [-10, 10] dB.
    uniform_fixed: direct Rice, nu uniform in [0, 1], sigma=1.
    Physical priors match the original Rice scripts and DAD's latent priors;
    random beams are not the distribution induced by a trained adaptive policy.
    """

    def __init__(self, scenario, seed, device="cpu", nx=8):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.params = Params(Nx=nx, Ny=1)
        self.pilot = generate_pilot_sequence(self.device, sequence_type="PSS")

    def rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.generator)

    def sample(self, n, reference_snr_db=None):
        if n < 1:
            raise ValueError("n must be positive")
        theta = math.pi / 6 + (2 * math.pi / 3) * self.rand(n)
        rho = 0.05 + 0.95 * self.rand(n)
        eta = 2 * math.pi * self.rand(n, self.params.K)
        # Fix the physically irrelevant common phase, as the policy does.
        eta = eta - eta[:, :1]
        beam = phase_to_beam(eta)
        steering = steering_vector_batch(theta, self.params)
        alpha = rho * (beam.conj() * steering).sum(-1)
        gain = alpha.abs() / rho
        nu = alpha.abs()
        sampled_snr = -10 + 20 * self.rand(n)
        if self.scenario == "uniform_fixed":
            nu = self.rand(n)
            alpha = nu.to(torch.complex64)
            # No physical gain or channel interpretation in this control.
            gain = torch.full_like(nu, float("nan"))
        if self.scenario == "physical_snr":
            snr = sampled_snr if reference_snr_db is None else torch.full_like(rho, reference_snr_db)
            sigma = rho * 10 ** (-snr / 20)
        else:
            if reference_snr_db is not None:
                raise ValueError("reference_snr_db is only meaningful for physical_snr")
            sigma = torch.ones_like(nu)
            snr = 20 * rho.log10()
            if self.scenario == "uniform_fixed":
                snr = torch.full_like(nu, float("nan"))
        noise = tuple(torch.randn(n, self.pilot.numel(), device=self.device,
                                  generator=self.generator) for _ in range(2))
        r = simulate_y(alpha, self.pilot, sigma, noise=noise).abs()
        return Batch(r, nu, sigma, gain, theta, rho, eta, snr)
