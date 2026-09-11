"""Focused numerical and architectural checks; run with unittest discovery."""

import unittest

import torch

from deep_sets_amp.benchmark_models import (
    ENCODER_KINDS,
    AmplitudeRegressor,
    amplitude_statistics,
    known_sigma_estimate,
    make_encoder,
    moment_estimate,
)
from modules.dad.encoder_amp import AmplitudeDeepSet


class BenchmarkModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_regression_and_encoder_preserve_leading_dimensions(self):
        for kind in ENCODER_KINDS:
            model = AmplitudeRegressor(kind)
            for shape in ((127,), (1, 127), (2, 3, 127)):
                with self.subTest(kind=kind, shape=shape):
                    r = torch.rand(shape)
                    self.assertEqual(model.encoder(r).shape, (*shape[:-1], 16))
                    prediction = model(r)
                    self.assertEqual(prediction.shape, shape[:-1])
                    self.assertTrue((prediction > 0).all())

    def test_set_encoders_are_permutation_invariant(self):
        r = torch.rand(3, 2, 127)
        permutation = torch.randperm(127)
        for kind in ("deepsets", "mean_var", "expanded_stats"):
            with self.subTest(kind=kind):
                encoder = make_encoder(kind)
                torch.testing.assert_close(encoder(r), encoder(r[..., permutation]))

    def test_raw_encoder_can_use_amplitude_order(self):
        encoder = make_encoder("raw")
        r = torch.zeros(1, 127)
        r[0, 0] = 1
        self.assertFalse(torch.allclose(encoder(r), encoder(r.flip(-1))))

    def test_parameter_counts_are_matched_and_production_deepset_is_reused(self):
        expected = {"deepsets": 5360, "raw": 5344, "mean_var": 5355, "expanded_stats": 5352}
        self.assertIsInstance(make_encoder("deepsets"), AmplitudeDeepSet)
        for kind, count in expected.items():
            with self.subTest(kind=kind):
                model = AmplitudeRegressor(kind)
                self.assertEqual(sum(p.numel() for p in model.encoder.parameters()), count)
                self.assertEqual(sum(p.numel() for p in model.head.parameters()), 577)
                self.assertLess(abs(count - expected["deepsets"]) / expected["deepsets"], 0.01)

    def test_custom_embedding_and_set_size(self):
        for kind in ENCODER_KINDS:
            with self.subTest(kind=kind):
                encoder = make_encoder(kind, num_amplitudes=11, output_dim=7)
                self.assertEqual(encoder(torch.rand(2, 11)).shape, (2, 7))

    def test_noiseless_moment_estimates_are_exact(self):
        nu = torch.tensor([0., 0.1, 0.5, 1.], dtype=torch.float64)
        r = nu[:, None].expand(-1, 127)
        torch.testing.assert_close(moment_estimate(r), nu)
        torch.testing.assert_close(known_sigma_estimate(r, 0.), nu)

    def test_known_sigma_second_moment_and_broadcasting(self):
        # r**2 equals nu**2 + sigma**2 exactly; no Monte Carlo tolerance needed.
        nu = torch.tensor([[0.25, 0.5], [0.75, 1.]], dtype=torch.float64)
        sigma = torch.tensor([[0.5, 1.], [0.25, 0.75]], dtype=torch.float64)
        r = (nu.square() + sigma.square()).sqrt()[..., None].expand(-1, -1, 127)
        torch.testing.assert_close(known_sigma_estimate(r, sigma), nu)
        torch.testing.assert_close(known_sigma_estimate(r, sigma[..., None]), nu)
        torch.testing.assert_close(known_sigma_estimate(torch.zeros(2, 127), 1.), torch.zeros(2))

    def test_negative_fourth_moment_combination_clips_to_zero(self):
        r = torch.tensor([[0., 0., 0., 1.]], requires_grad=True)
        prediction = moment_estimate(r)
        torch.testing.assert_close(prediction, torch.zeros(1))
        prediction.sum().backward()
        self.assertTrue(torch.isfinite(r.grad).all())

    def test_expanded_statistics_definitions(self):
        r = torch.tensor([[1., 3.]], dtype=torch.float64)
        expected = torch.tensor([[2., 1., 0., 1. / (1. + 1e-8)**2, 3., 2.]], dtype=torch.float64)
        torch.testing.assert_close(amplitude_statistics(r, expanded=True), expected)
        torch.testing.assert_close(amplitude_statistics(r), expected[:, :2])

    def test_zero_and_constant_data_have_finite_values_and_gradients(self):
        for value in (0., 1e-12, 0.7):
            for kind in ENCODER_KINDS:
                with self.subTest(value=value, kind=kind):
                    r = torch.full((2, 127), value, requires_grad=True)
                    model = AmplitudeRegressor(kind)
                    output = model(r)
                    output.sum().backward()
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(r.grad).all())
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
            for estimator in (moment_estimate, lambda x: known_sigma_estimate(x, 0.)):
                r = torch.full((2, 127), value, requires_grad=True)
                output = estimator(r)
                output.sum().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(r.grad).all())

    def test_fresh_initialization_is_reproducible(self):
        for kind in ENCODER_KINDS:
            with self.subTest(kind=kind):
                torch.manual_seed(42)
                first = make_encoder(kind)
                torch.manual_seed(42)
                second = make_encoder(kind)
                for a, b in zip(first.parameters(), second.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "Unknown encoder"):
            make_encoder("missing")
        for kwargs in ({"num_amplitudes": 0}, {"output_dim": 0}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                make_encoder("deepsets", **kwargs)


if __name__ == "__main__":
    unittest.main()
