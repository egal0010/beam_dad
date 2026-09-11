"""Focused checks for controlled DAD representation ablations."""

import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import torch

from deep_sets_amp.benchmark_dad import (
    METHODS, batch_bound, derived_seed, make_policy, parse_args, train_one,
)
from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.dad.contrastive import make_log_likelihood_fn


class DADBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.args = parse_args([
            "--steps", "1", "--batch-size", "2", "--L", "2",
            "--rho-grid-size", "15", "--eval-batches", "1",
            "--eval-batch-size", "2", "--eval-every", "1", "--seeds", "11",
            "--warmup-steps", "1", "--no-use-checkpoint",
        ])
        self.device = torch.device("cpu")
        self.params = Params(Nx=8, Ny=1)
        self.pilot = generate_pilot_sequence(self.device, "PSS")
        rho_grid = torch.linspace(0.05, 1.0, self.args.rho_grid_size)
        self.log_rho_prior = torch.full_like(rho_grid, -math.log(rho_grid.numel()))
        self.likelihood = make_log_likelihood_fn(rho_grid, self.pilot, self.args.snr_db, self.params)

    def bound(self, policy, seed=123, training=False):
        return batch_bound(
            policy, self.args, self.params, self.pilot, self.likelihood, self.log_rho_prior,
            batch_size=2, seed=seed, training=training,
        )

    def test_common_policy_layers_have_identical_initialization(self):
        reference = make_policy("deepsets", 11, self.args, self.device).state_dict()
        for method in METHODS:
            with self.subTest(method=method):
                state = make_policy(method, 11, self.args, self.device).state_dict()
                for key, value in reference.items():
                    if not key.startswith("encoder.amp."):
                        torch.testing.assert_close(state[key], value, rtol=0, atol=0)

    def test_empty_and_nonempty_history_shapes_and_amplitude_gradients(self):
        for method in METHODS:
            with self.subTest(method=method):
                policy = make_policy(method, 11, self.args, self.device)
                first = policy(None, None, batch_size=2)
                self.assertEqual(first.shape, (2, 8))
                torch.testing.assert_close(first[:, 0], torch.zeros(2))
                eta = torch.randn(2, 2, 8)
                eta[..., 0] = 0
                amplitudes = torch.rand(2, 2, 127, requires_grad=True)
                next_beam = policy(eta, amplitudes)
                self.assertEqual(next_beam.shape, (2, 8))
                next_beam.square().sum().backward()
                self.assertTrue(torch.isfinite(amplitudes.grad).all())
                self.assertGreater(amplitudes.grad.abs().sum().item(), 0)
                gradients = [p.grad for p in policy.encoder.amp.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_dad_objective_reaches_all_amplitude_encoders(self):
        for method in METHODS:
            with self.subTest(method=method):
                policy = make_policy(method, 11, self.args, self.device)
                bound, per_trajectory = self.bound(policy, training=True)
                self.assertEqual(per_trajectory.shape, (2,))
                self.assertTrue(torch.isfinite(per_trajectory).all())
                self.assertLessEqual(per_trajectory.max().item(), math.log(self.args.L+1) + 1e-5)
                (-bound).backward()
                gradients = [p.grad for p in policy.encoder.amp.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_batch_randomness_is_reproducible_and_split_streams_differ(self):
        policy = make_policy("deepsets", 11, self.args, self.device).eval()
        with torch.no_grad():
            first, first_values = self.bound(policy)
            torch.rand(100)
            repeated, repeated_values = self.bound(policy)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        torch.testing.assert_close(first_values, repeated_values, rtol=0, atol=0)
        streams = [derived_seed(11, split, 0) for split in ("training", "validation", "test", "warmup")]
        self.assertEqual(len(set(streams)), len(streams))

    def test_single_experiment_is_rejected_because_encoder_is_unused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--experiments", "1"])

    def test_checkpoints_support_weights_only_reload(self):
        with tempfile.TemporaryDirectory(prefix="dad_benchmark_test_") as directory:
            out = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics, history = train_one(
                    "deepsets", 11, self.args, self.device, self.params, self.pilot,
                    self.likelihood, self.log_rho_prior, out,
                )
            destination = out / "seed_11" / "deepsets"
            best = torch.load(destination / "best.pt", weights_only=True)
            final = torch.load(destination / "final.pt", weights_only=True)
            self.assertEqual(best["step"], metrics["selected_step"])
            self.assertEqual(final["step"], 1)
            self.assertEqual(best["encoder_type"], "deepsets")
            self.assertEqual(len(history), 2)
            reloaded = make_policy("deepsets", 11, self.args, self.device)
            reloaded.load_state_dict(best["model_state_dict"])
            with torch.no_grad():
                bound, _ = self.bound(reloaded)
            self.assertTrue(torch.isfinite(bound))
            values = torch.load(destination / "test_per_trajectory_bounds.pt", weights_only=True)
            self.assertEqual(values.shape, (2,))


if __name__ == "__main__":
    unittest.main()
