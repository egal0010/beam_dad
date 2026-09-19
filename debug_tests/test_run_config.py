"""Run with ``python -m unittest debug_tests.test_run_config``."""

from contextlib import redirect_stdout, redirect_stderr
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from modules.beam_eig.array_model import steering_vector
from modules.beam_eig.params import Params
from modules.dad.policy import DADPolicy
from modules.run_config import default_checkpoint_dir
from modules.training import training_chunked
from training import dad_training
import test_dad_vs_eig_nmc as evaluation


class RunConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_num_threads)

    def setUp(self):
        self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.temp_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.params = Params(Nx=2, T=7, N=3)
        self.pilot = torch.ones(3)
        torch.manual_seed(142)

    def make_context(self, params=None, **kwargs):
        with (
            patch.object(evaluation, "N_REAL", 2),
            patch.object(evaluation, "L_EVAL", 3),
            patch.object(evaluation, "generate_pilot_sequence", return_value=self.pilot),
        ):
            return evaluation.make_test_set(params or self.params, **kwargs)

    def make_policy(self):
        return DADPolicy(
            design_dim=self.params.K,
            observation_dim=self.pilot.numel(),
            hidden_dim=8,
            encoding_dim=4,
        )

    def test_entrypoints_share_default_and_accept_positive_horizons(self):
        for parse_args in (dad_training.parse_args, evaluation.parse_args):
            with self.subTest(entrypoint=parse_args.__module__):
                self.assertEqual(parse_args([]).T, Params.T)
                self.assertEqual(parse_args(["--T", "7"]).T, 7)
                for invalid in ("0", "-1", "1.5"):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse_args(["--T", invalid])

    def test_default_paths_and_generated_observations_follow_run_dimensions(self):
        training_root = Path(__file__).resolve().parents[1] / "training"
        self.assertEqual(
            default_checkpoint_dir(Params()),
            training_root / "checkpoints_fixed_sigma_nx8_T3_ds",
        )
        self.assertEqual(
            default_checkpoint_dir(Params(Nx=5, Ny=2, T=7)),
            training_root / "checkpoints_fixed_sigma_nx5_ny2_T7_ds",
        )
        ctx = self.make_context()
        self.assertEqual(ctx.device.type, "cpu")
        self.assertEqual(ctx.noise_real.shape, (2, 7, 3))
        self.assertEqual(ctx.noise_imag.shape, (2, 7, 3))
        self.assertEqual(
            ctx.checkpoint_path,
            training_root / "checkpoints_fixed_sigma_nx2_T7_ds" / "dad_T7_best.pt",
        )
        self.assertEqual(ctx.output_path, Path("comparison_nmc_vs_dad_T7.pt"))

        checkpoint_path = self.temp_dir / "custom.pt"
        output_path = self.temp_dir / "results.pt"
        custom = self.make_context(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )
        self.assertEqual(custom.checkpoint_path, checkpoint_path)
        self.assertEqual(custom.output_path, output_path)

    def test_training_checkpoints_reload_for_seven_step_evaluation(self):
        policy = self.make_policy()
        initial_weights = {
            key: tensor.clone() for key, tensor in policy.state_dict().items()
        }
        checkpoint_dir = self.temp_dir / "checkpoints"

        # One real optimizer update at the periodic boundary covers both saves.
        with (
            patch.object(training_chunked, "range", return_value=(2000,), create=True),
            patch.object(
                training_chunked, "default_checkpoint_dir", return_value=checkpoint_dir
            ) as checkpoint_path,
            patch.object(training_chunked, "rollout", wraps=training_chunked.rollout) as rollout,
        ):
            history = training_chunked.train_dad_chunked(
                policy,
                self.params,
                self.pilot,
                num_steps=2000,
                batch_size=2,
                L=3,
                snr_db=0.0,
                rho_grid_size=3,
                use_checkpoint=False,
            )

        checkpoint_path.assert_called_once_with(self.params)
        self.assertEqual(rollout.call_args.kwargs["n_steps"], 7)
        self.assertEqual(len(history["loss"]), 1)
        self.assertTrue(math.isfinite(history["loss"][0]))
        self.assertTrue(any(
            not torch.equal(initial_weights[key], tensor)
            for key, tensor in policy.state_dict().items()
        ))

        for filename in ("dad_T7_best.pt", "dad_T7_step_2000.pt"):
            with self.subTest(checkpoint=filename):
                saved_path = checkpoint_dir / filename
                checkpoint = torch.load(saved_path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["T"], 7)
                self.assertEqual(checkpoint["encoder_type"], "deepsets")
                self.assertEqual(checkpoint["sigma_mode"], "sigma_fixed")
                self.assertEqual(checkpoint["snr_db"], 0.0)
                self.assertEqual(checkpoint["Nx"], 2)
                self.assertEqual(checkpoint["Ny"], 1)
                self.assertEqual(checkpoint["Ns"], 3)
                ctx = self.make_context(checkpoint_path=saved_path)
                loaded = evaluation.load_policy(ctx)
                for key, tensor in loaded.state_dict().items():
                    torch.testing.assert_close(tensor, policy.state_dict()[key])

        ctx.theta_grid = torch.linspace(math.pi / 6, 5 * math.pi / 6, 5)
        ctx.rho_grid = torch.linspace(0.05, 1.0, 3)
        ctx.a_grid = steering_vector(ctx.theta_grid, ctx.params)
        ctx.log_p_rho_prior = torch.full((3,), -math.log(3))
        history_lengths = []

        def select_beam(posterior, p_theta, sigma, eta_history, r_history):
            history_lengths.append(eta_history.shape[1])
            self.assertEqual(eta_history.shape[1], r_history.shape[1])
            return loaded(eta_history, r_history)[0], None

        with patch.object(evaluation, "N_RECOMPUTE", 3):
            results = evaluation.evaluate_method(
                ctx, "dad", select_beam, eig_mode="marginal"
            )
        self.assertEqual(history_lengths, list(range(7)) * 2)
        self.assertEqual(results["eig_per_step"].shape, (2, 7))
        self.assertEqual(results["decision_times"].shape, (14,))
        self.assertEqual(results["g_L"].shape, (2,))
        for values in results.values():
            self.assertTrue(torch.isfinite(values).all())

        ctx.params = Params(Nx=2, T=3)
        with self.assertRaisesRegex(ValueError, r"T=7.*T=3"):
            evaluation.load_policy(ctx)

    def test_checkpoint_milestones_preserve_scheduler_cadence(self):
        milestones = (2000, 4000, 6000, 10000, 15000, 20000)
        selected_steps = (
            100, 999, 1000, 1999, 2000, 2001, 3000, 4000,
            5000, 6000, 8000, 10000, 15000, 20000, 21000,
        )
        checkpoint_dir = self.temp_dir / "milestones"
        learning_rate = 1e-3

        # Exercise real updates at saving boundaries without a full training run.
        with patch.object(
            training_chunked, "range", return_value=selected_steps, create=True
        ):
            history = training_chunked.train_dad_chunked(
                self.make_policy(), self.params, self.pilot,
                num_steps=21000, batch_size=2, L=3, snr_db=0.0,
                rho_grid_size=3, learning_rate=learning_rate,
                use_checkpoint=False, checkpoint_dir=checkpoint_dir,
            )

        self.assertEqual(len(history["loss"]), len(selected_steps))
        self.assertEqual(
            {path.name for path in checkpoint_dir.iterdir()},
            {"dad_T7_best.pt"}
            | {f"dad_T7_step_{step}.pt" for step in milestones},
        )
        for step in milestones:
            with self.subTest(step=step):
                checkpoint = torch.load(
                    checkpoint_dir / f"dad_T7_step_{step}.pt",
                    map_location="cpu", weights_only=True,
                )
                expected_decays = sum(
                    s <= step and s % 1000 == 0 for s in selected_steps
                )
                self.assertEqual(checkpoint["step"], step)
                self.assertEqual(
                    checkpoint["scheduler_state_dict"]["last_epoch"], expected_decays
                )
                self.assertAlmostEqual(
                    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"],
                    learning_rate * 0.98 ** expected_decays,
                )

    def test_legacy_checkpoint_without_horizon_can_still_load(self):
        policy = self.make_policy()
        checkpoint_path = self.temp_dir / "legacy.pt"
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "encoder_type": "mean_var",
        }, checkpoint_path)
        loaded = evaluation.load_policy(self.make_context(checkpoint_path=checkpoint_path))
        for key, tensor in loaded.state_dict().items():
            torch.testing.assert_close(tensor, policy.state_dict()[key])

    def test_eig_selection_and_recomputation_use_fixed_sigma_in_both_modes(self):
        ctx = self.make_context(Params(Nx=2, T=2, N=3))
        ctx.theta_grid = torch.linspace(math.pi / 6, 5 * math.pi / 6, 5)
        ctx.rho_grid = torch.linspace(0.05, 1.0, 3)
        ctx.a_grid = steering_vector(ctx.theta_grid, ctx.params)
        ctx.log_p_rho_prior = torch.full((3,), -math.log(3))
        candidates = SimpleNamespace(
            search="random", beam_space="continuous", n_candidates=3
        )
        for mode in ("mean", "marginal"):
            with (
                self.subTest(mode=mode),
                patch.object(evaluation, "N_RECOMPUTE", 3),
                patch.dict(evaluation.NAMES),
            ):
                results = evaluation.eig_nmc(ctx, mode, candidates)
                self.assertEqual(results["eig_per_step"].shape, (2, 2))
                for values in results.values():
                    self.assertTrue(torch.isfinite(values).all())

    def test_explicit_training_horizon_mismatch_fails_before_creating_directory(self):
        checkpoint_dir = self.temp_dir / "unused"
        with self.assertRaisesRegex(ValueError, r"n_experiments=3.*params.T=7"):
            training_chunked.train_dad_chunked(
                self.make_policy(), self.params, self.pilot,
                num_steps=1, batch_size=2, L=3, n_experiments=3,
                snr_db=0.0, checkpoint_dir=checkpoint_dir,
            )
        self.assertFalse(checkpoint_dir.exists())


if __name__ == "__main__":
    unittest.main()
