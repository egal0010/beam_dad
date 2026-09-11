"""Scientific checks for the data shared by the amplitude encoder benchmarks.

Run from the repository root with
``python -m unittest deep_sets_amp.test_benchmark_data``.
"""

from dataclasses import fields
import math
import unittest
from unittest.mock import patch

import torch

from deep_sets_amp import benchmark_data
from deep_sets_amp.benchmark_data import Batch, SCENARIOS, Sampler
from modules.beam_eig.simulator import generate_amplitude_measurement, simulate_y


class BenchmarkDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_num_threads)

    def test_physical_prior_and_normalized_random_beam_gain(self):
        batch = Sampler("physical_fixed", seed=713).sample(16384)
        self.assertEqual(batch.r.shape, (16384, 127))
        self.assertEqual(batch.eta.shape, (16384, 8))
        self.assertTrue(torch.isfinite(batch.r).all())
        self.assertTrue((batch.r >= 0).all())
        self.assertTrue(((batch.theta >= math.pi / 6) & (batch.theta <= 5 * math.pi / 6)).all())
        self.assertTrue(((batch.rho >= 0.05) & (batch.rho <= 1)).all())
        self.assertTrue(((batch.gain >= 0) & (batch.gain <= 1 + 1e-6)).all())
        self.assertTrue(((batch.nu >= 0) & (batch.nu <= 1 + 1e-6)).all())
        torch.testing.assert_close(batch.nu, batch.rho * batch.gain)
        torch.testing.assert_close(batch.sigma, torch.ones_like(batch.sigma))
        torch.testing.assert_close(batch.eta[:, 0], torch.zeros_like(batch.nu))
        # Independent uniform beam phases cancel all cross terms in E[|b^H a|^2].
        self.assertAlmostEqual(batch.gain.square().mean().item(), 1 / 8, delta=0.004)

    def test_noncentrality_matches_independent_array_sum(self):
        for nx in (1, 8):
            with self.subTest(nx=nx):
                batch = Sampler("physical_fixed", seed=409, nx=nx).sample(128)
                element = torch.arange(nx, dtype=batch.theta.dtype)
                # Half-wavelength spacing, unit-norm steering and beam vectors.
                phase = -math.pi * batch.theta.cos()[:, None] * element - batch.eta
                alpha = batch.rho * torch.exp(1j * phase).sum(-1) / nx
                torch.testing.assert_close(batch.nu, alpha.abs(), rtol=2e-5, atol=2e-6)
                if nx == 1:
                    torch.testing.assert_close(batch.nu, batch.rho)

    def test_vectorized_samples_match_scalar_physical_simulator_with_shared_noise(self):
        for scenario in ("physical_fixed", "physical_snr"):
            with self.subTest(scenario=scenario):
                sampler = Sampler(scenario, seed=145)
                captured = {}

                def capture_noise(alpha, pilot, sigma, *, noise):
                    captured["noise"] = noise
                    return simulate_y(alpha, pilot, sigma, noise=noise)

                with patch.object(benchmark_data, "simulate_y", side_effect=capture_noise):
                    batch = sampler.sample(11)
                scalar_samples = torch.stack([
                    generate_amplitude_measurement(
                        batch.theta[i], batch.rho[i], batch.eta[i],
                        sampler.pilot, batch.sigma[i], sampler.params,
                        noise=tuple(component[i] for component in captured["noise"]),
                    )
                    for i in range(batch.nu.numel())
                ])
                torch.testing.assert_close(batch.r, scalar_samples, rtol=1e-5, atol=2e-6)

    def test_rice_second_and_fourth_moments_with_monte_carlo_uncertainty(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                batch = Sampler(scenario, seed=998).sample(8192)
                nu2 = batch.nu.double().square()
                sigma2 = batch.sigma.double().square()
                r2 = batch.r.double().square()
                expected_m2 = nu2 + sigma2
                expected_m4 = nu2.square() + 4 * nu2 * sigma2 + 2 * sigma2.square()
                expected_m8 = (
                    nu2.pow(4) + 16 * nu2.pow(3) * sigma2
                    + 72 * nu2.square() * sigma2.square()
                    + 96 * nu2 * sigma2.pow(3) + 24 * sigma2.pow(4)
                )
                for name, empirical, expected, single_draw_variance in (
                    ("second", r2.mean(-1), expected_m2, expected_m4 - expected_m2.square()),
                    ("fourth", r2.square().mean(-1), expected_m4, expected_m8 - expected_m4.square()),
                ):
                    # Conditional moments account for every example's own nu/sigma;
                    # their average residual has a known Monte Carlo standard error.
                    standard_error = (single_draw_variance.sum() / batch.r.numel() / batch.r.shape[0]).sqrt()
                    residual = (empirical - expected).mean().abs()
                    self.assertLess(
                        residual.item(), 6 * standard_error.item(),
                        msg=f"{scenario}: {name} Rice moment disagrees beyond 6 standard errors",
                    )

    def test_uniform_control_covers_current_zero_to_one_prior(self):
        batch = Sampler("uniform_fixed", seed=652).sample(8192)
        self.assertTrue(((batch.nu >= 0) & (batch.nu <= 1)).all())
        self.assertLess(batch.nu.min().item(), 0.001)
        self.assertGreater(batch.nu.max().item(), 0.999)
        self.assertAlmostEqual(batch.nu.mean().item(), 0.5, delta=0.015)
        self.assertAlmostEqual(batch.nu.var().item(), 1 / 12, delta=0.004)
        self.assertTrue(torch.isnan(batch.gain).all())
        self.assertTrue(torch.isnan(batch.reference_snr_db).all())
        torch.testing.assert_close(batch.sigma, torch.ones_like(batch.sigma))

    def test_private_rng_reproducibility_ignores_global_model_randomness(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario), torch.random.fork_rng(devices=[]):
                first = Sampler(scenario, seed=443)
                second = Sampler(scenario, seed=443)
                for _ in range(2):
                    batch_a = first.sample(37)
                    # Different architectures can consume arbitrary global RNG draws.
                    torch.random.default_generator.manual_seed(952)
                    torch.rand(501)
                    batch_b = second.sample(37)
                    for field in fields(Batch):
                        torch.testing.assert_close(
                            getattr(batch_a, field.name), getattr(batch_b, field.name),
                            rtol=0, atol=0, equal_nan=True,
                        )

    def test_sampler_does_not_advance_global_rng(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(138)
            previous_state = torch.random.get_rng_state().clone()
            Sampler("physical_snr", seed=225).sample(19)
            torch.testing.assert_close(torch.random.get_rng_state(), previous_state, rtol=0, atol=0)

    def test_reference_and_effective_snr_relations(self):
        batch = Sampler("physical_snr", seed=390).sample(1024)
        self.assertTrue(((batch.reference_snr_db >= -10) & (batch.reference_snr_db <= 10)).all())
        torch.testing.assert_close(
            20 * torch.log10(batch.rho / batch.sigma), batch.reference_snr_db,
            rtol=1e-5, atol=2e-6,
        )
        torch.testing.assert_close(
            batch.effective_snr_db,
            batch.reference_snr_db + 20 * batch.gain.log10(),
            rtol=1e-5, atol=3e-6,
        )

    def test_snr_sweeps_keep_the_same_physical_examples(self):
        low = Sampler("physical_snr", seed=727).sample(173, reference_snr_db=-10)
        high = Sampler("physical_snr", seed=727).sample(173, reference_snr_db=10)
        for name in ("theta", "rho", "eta", "nu", "gain"):
            torch.testing.assert_close(getattr(low, name), getattr(high, name), rtol=0, atol=0)
        torch.testing.assert_close(low.sigma, 10 * high.sigma)
        torch.testing.assert_close(low.reference_snr_db, torch.full_like(low.nu, -10))
        torch.testing.assert_close(high.reference_snr_db, torch.full_like(high.nu, 10))
        self.assertFalse(torch.equal(low.r, high.r))


if __name__ == "__main__":
    unittest.main()
