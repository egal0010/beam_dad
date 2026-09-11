"""Compare amplitude representations while training actual DAD policies from scratch.

The downstream policy, physical model, contrastive objective, and random draws
are shared across methods. Only the amplitude encoder changes. The encoders have
approximately matched parameter counts and the same 16-dimensional output; raw
and statistics variants are controlled bottleneck baselines, not reproductions
of the archived direct-concatenation encoders.

Run from the repository root, for example::

    python -m deep_sets_amp.benchmark_dad --device cuda:1 \
        --steps 5000 --seeds 11 22 33 --output /tmp/dad_encoder_comparison

A small pipeline check (not evidence of convergence)::

    python -m deep_sets_amp.benchmark_dad --steps 2 --batch-size 4 --L 3 \
        --rho-grid-size 5 --eval-batches 1 --eval-batch-size 4 \
        --eval-every 1 --seeds 11 --output /tmp/dad_encoder_smoke

Held-out contrastive bound measures policy information gain, not angle accuracy.
With L contrastive candidates the bound is capped at log(L+1); increase L when
curves approach that ceiling. Validation alone selects checkpoints. Test data
are evaluated once, after training, and never choose a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from deep_sets_amp.benchmark_models import make_encoder
from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.dad.contrastive import (
    contrastive_bound,
    make_log_likelihood_fn,
    make_observation_fn,
)
from modules.dad.experiment import rollout
from modules.dad.policy import DADPolicy

METHODS = ("deepsets", "raw", "mean_var", "expanded_stats")


def derived_seed(seed: int, stream: str, index: int = 0) -> int:
    payload = f"dad-encoder-v1:{seed}:{stream}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cpu_state(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_policy(method: str, seed: int, args: argparse.Namespace, device: torch.device):
    # Creating the common policy before the variable-sized encoder makes both
    # the beam/embedding net and emitter initialization identical across methods.
    torch.manual_seed(derived_seed(seed, "common_initialization"))
    policy = DADPolicy(
        design_dim=args.nx,
        observation_dim=127,
        hidden_dim=args.hidden_dim,
        encoding_dim=args.encoding_dim,
    )
    torch.manual_seed(derived_seed(seed, "amplitude_initialization"))
    policy.encoder.amp = make_encoder(method, num_amplitudes=127, output_dim=16)
    return policy.to(device)


def batch_bound(policy, args, params, pilot, likelihood, log_rho_prior, *,
                batch_size: int, seed: int, training: bool):
    # Every method receives identical theta, rho, Gaussian noise, and candidates
    # for a given seed and step. Adaptive observations differ because beams do.
    torch.manual_seed(seed)
    device = next(policy.parameters()).device
    theta = math.radians(30.0) + math.radians(120.0) * torch.rand(batch_size, device=device)
    rho = 0.05 + 0.95 * torch.rand(batch_size, device=device)
    observation_fn = make_observation_fn(rho, pilot, args.snr_db, params)
    eta_history, r_history = rollout(policy, theta, args.experiments, observation_fn)
    contrasts = math.radians(30.0) + math.radians(120.0) * torch.rand(
        batch_size, args.L, device=device,
    )
    candidates = torch.cat((theta[:, None], contrasts), dim=1)
    return contrastive_bound(
        candidates, eta_history, r_history, likelihood, log_rho_prior,
        candidate_chunk_size=args.candidate_chunk_size,
        use_checkpoint=training and args.use_checkpoint,
    )


@torch.no_grad()
def evaluate(policy, args, params, pilot, likelihood, log_rho_prior, *, seed: int, split: str):
    policy.eval()
    values = []
    for index in range(args.eval_batches):
        _, per_trajectory = batch_bound(
            policy, args, params, pilot, likelihood, log_rho_prior,
            batch_size=args.eval_batch_size,
            seed=derived_seed(seed, split, index), training=False,
        )
        if not torch.isfinite(per_trajectory).all():
            raise RuntimeError(f"Non-finite {split} contrastive bounds.")
        values.append(per_trajectory.detach().cpu().double())
    values = torch.cat(values)
    mc_se = values.std(unbiased=True).item() / math.sqrt(values.numel()) if values.numel() > 1 else 0.0
    return {"bound": values.mean().item(), "mc_se": mc_se, "values": values}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(method, seed, args, device, params, pilot, likelihood, log_rho_prior, out):
    policy = make_policy(method, seed, args, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, betas=(0.8, 0.998))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # Warm each architecture's kernels and checkpoint machinery, then restore
    # its exact starting weights and a fresh optimizer. Exclude warm-up from
    # training seconds and sample counts, and use a separate random stream.
    if args.warmup_steps:
        initial_state = cpu_state(policy)
        policy.train()
        for index in range(args.warmup_steps):
            warmup_bound, _ = batch_bound(
                policy, args, params, pilot, likelihood, log_rho_prior,
                batch_size=args.batch_size,
                seed=derived_seed(seed, "warmup", index), training=True,
            )
            optimizer.zero_grad(set_to_none=True)
            (-warmup_bound).backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.grad_clip, norm_type=float("inf"),
                error_if_nonfinite=True,
            )
            optimizer.step()
        synchronize(device)
        policy.load_state_dict(initial_state)
        policy.zero_grad(set_to_none=True)
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, betas=(0.8, 0.998))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        del initial_state, warmup_bound
    destination = out / f"seed_{seed}" / method
    destination.mkdir(parents=True)
    history = []
    train_seconds = 0.0
    best_bound = -math.inf
    best_state = None
    best_step = 0
    best_seconds = 0.0
    interval_bounds = []
    interval_gradient_norms = []
    metadata = {
        "method": method, "seed": seed, "config": vars(args),
        "Nx": params.Nx, "Ny": params.Ny, "Ns": pilot.numel(),
        "encoder_type": method, "amplitude_embedding_dim": 16,
        "hidden_dim": args.hidden_dim, "encoding_dim": args.encoding_dim,
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "amplitude_parameter_count": sum(p.numel() for p in policy.encoder.amp.parameters()),
    }

    for step in range(args.steps + 1):
        if step:
            policy.train()
            synchronize(device)
            started = time.perf_counter()
            bound, _ = batch_bound(
                policy, args, params, pilot, likelihood, log_rho_prior,
                batch_size=args.batch_size,
                seed=derived_seed(seed, "training", step), training=True,
            )
            if not torch.isfinite(bound):
                raise RuntimeError(f"Non-finite training bound: method={method}, seed={seed}, step={step}")
            optimizer.zero_grad(set_to_none=True)
            (-bound).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.grad_clip, norm_type=float("inf"),
                error_if_nonfinite=True,
            )
            optimizer.step()
            if step % 1000 == 0:
                scheduler.step()
            synchronize(device)
            train_seconds += time.perf_counter() - started
            interval_bounds.append(bound.detach().item())
            interval_gradient_norms.append(float(grad_norm))

        if step % args.eval_every and step != args.steps:
            continue
        validation = evaluate(
            policy, args, params, pilot, likelihood, log_rho_prior,
            seed=seed, split="validation",
        )
        row = {
            "seed": seed, "method": method, "step": step,
            "trajectories_seen": step * args.batch_size,
            "train_seconds": train_seconds,
            "training_bound_interval_mean": statistics.mean(interval_bounds) if interval_bounds else "",
            "gradient_inf_norm_interval_mean": statistics.mean(interval_gradient_norms) if interval_gradient_norms else "",
            "validation_bound": validation["bound"],
            "validation_mc_se": validation["mc_se"],
        }
        history.append(row)
        interval_bounds.clear()
        interval_gradient_norms.clear()
        if validation["bound"] > best_bound:
            best_bound = validation["bound"]
            best_state = cpu_state(policy)
            best_step, best_seconds = step, train_seconds
            torch.save({
                **metadata, "model_state_dict": best_state,
                "step": step, "train_seconds": train_seconds,
                "validation_bound": best_bound,
                "selection": "maximum held-out validation contrastive bound",
            }, destination / "best.pt")
        write_csv(destination / "history.csv", history)
        print(
            f"DAD {method:14s} seed={seed} step={step:6d} "
            f"validation_bound={validation['bound']:.5f} "
            f"train_seconds={train_seconds:.2f}", flush=True,
        )

    torch.save({
        **metadata, "model_state_dict": cpu_state(policy),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": args.steps, "train_seconds": train_seconds,
        "validation_bound": history[-1]["validation_bound"],
        "selection": "final optimizer step; not selected by test data",
    }, destination / "final.pt")
    policy.load_state_dict(best_state)
    test = evaluate(
        policy, args, params, pilot, likelihood, log_rho_prior,
        seed=seed, split="test",
    )
    torch.save(test["values"], destination / "test_per_trajectory_bounds.pt")
    crossings = [row for row in history if args.target_bound is not None and row["validation_bound"] >= args.target_bound]
    result = {
        "target_bound": args.target_bound,
        "first_target_step": crossings[0]["step"] if crossings else None,
        "first_target_train_seconds": crossings[0]["train_seconds"] if crossings else None,
        "seed": seed, "method": method,
        "parameter_count": metadata["parameter_count"],
        "amplitude_parameter_count": metadata["amplitude_parameter_count"],
        "selected_step": best_step, "selected_train_seconds": best_seconds,
        "validation_bound": best_bound,
        "test_bound": test["bound"], "test_mc_se": test["mc_se"],
        "test_trajectories": test["values"].numel(),
        "total_train_seconds": train_seconds,
        "train_trajectories_per_second": args.steps * args.batch_size / train_seconds,
    }
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result, history


def summarize(results, history, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    write_csv(out / "metrics.csv", results)
    write_csv(out / "learning_curves.csv", history)
    summary = []
    for method in args.methods:
        entries = [row for row in results if row["method"] == method]
        def mean_sd(key):
            values = [row[key] for row in entries]
            return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
        bound_mean, bound_sd = mean_sd("test_bound")
        time_mean, time_sd = mean_sd("total_train_seconds")
        summary.append({
            "method": method, "seeds": len(entries),
            "parameter_count": entries[0]["parameter_count"],
            "test_bound_mean": bound_mean, "test_bound_sd": bound_sd,
            "train_seconds_mean": time_mean, "train_seconds_sd": time_sd,
        })
    write_csv(out / "summary.csv", summary)

    lookup = {(row["seed"], row["method"]): row for row in results}
    pairs = []
    if "deepsets" in args.methods:
        for seed in args.seeds:
            for method in args.methods:
                if method == "deepsets":
                    continue
                ds, other = lookup[seed, "deepsets"], lookup[seed, method]
                pairs.append({
                    "seed": seed, "other_method": method,
                    "test_bound_deepsets_minus_other": ds["test_bound"] - other["test_bound"],
                    "train_seconds_deepsets_minus_other": ds["total_train_seconds"] - other["total_train_seconds"],
                })
    write_csv(out / "paired_seed_differences.csv", pairs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in args.methods:
        rows_by_seed = [
            [row for row in history if row["method"] == method and row["seed"] == seed]
            for seed in args.seeds
        ]
        scores = np.asarray([[row["validation_bound"] for row in rows] for rows in rows_by_seed])
        means = scores.mean(axis=0)
        spread = scores.std(axis=0, ddof=1) if len(args.seeds) > 1 else np.zeros_like(means)
        steps = [row["step"] for row in rows_by_seed[0]]
        line, = axes[0].plot(steps, means, label=method)
        axes[0].fill_between(steps, means-spread, means+spread, color=line.get_color(), alpha=0.15)
        # Each seed has its own time axis. Avoid averaging scores at unmatched
        # times; show one line per seed in the wall-clock panel.
        for index, rows in enumerate(rows_by_seed):
            axes[1].plot(
                [row["train_seconds"] for row in rows],
                [row["validation_bound"] for row in rows],
                color=line.get_color(), alpha=0.7,
                label=method if index == 0 else None,
            )
    for axis in axes:
        axis.set_ylabel("Held-out validation contrastive bound (nats)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_xlabel("Optimizer updates")
    axes[0].set_title("Mean across seeds; shading = ±1 sample SD")
    axes[1].set_xlabel("Synchronized training seconds (validation excluded)")
    axes[1].set_title("One line per method and seed")
    fig.tight_layout()
    fig.savefig(out / "dad_learning_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(len(summary))
    axis.bar(positions, [row["test_bound_mean"] for row in summary], color="steelblue", alpha=0.7)
    if len(args.seeds) > 1:
        axis.errorbar(positions, [row["test_bound_mean"] for row in summary],
                      yerr=[row["test_bound_sd"] for row in summary], fmt="none", color="black", capsize=4)
    for index, method in enumerate(args.methods):
        values = [row["test_bound"] for row in results if row["method"] == method]
        axis.scatter(np.full(len(values), index), values, color="black", s=18)
    axis.set_xticks(positions, args.methods)
    axis.set_ylabel("Held-out test contrastive bound (nats; higher is better)")
    axis.set_title("Validation-selected checkpoints; bars = mean, whiskers = sample SD")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "dad_test_bound.png", dpi=180)
    plt.close(fig)

    lines = [
        "# DAD amplitude encoder comparison", "",
        "All policies train from scratch on the DAD contrastive objective. Only the amplitude encoder changes; the common policy layers start identically within each seed. No Rice regression weights are loaded.", "",
        f"T={args.experiments}, reference SNR={args.snr_db:g} dB, L={args.L}, rho grid={args.rho_grid_size}, steps={args.steps}, seeds={args.seeds}.", "",
        "The measured outcome is held-out contrastive bound (higher is better), not angle estimation accuracy. Finite-L contrastive bounds are capped at log(L+1); this is not an exact EIG estimate. A coarse rho grid also changes the objective approximation.", "",
        "| Method | Parameters | Test bound mean ± seed SD (nats) | Training seconds mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['method']} | {row['parameter_count']} | {row['test_bound_mean']:.5f} ± {row['test_bound_sd']:.5f} | {row['train_seconds_mean']:.2f} ± {row['train_seconds_sd']:.2f} |")
    lines.extend(["", "Reported seed SD describes run variability; with one seed it is zero by convention and does not establish uncertainty. Test Monte Carlo standard errors are saved separately in metrics.csv.", ""])
    if pairs:
        lines.append("Paired differences use the same training/test random streams per seed. Positive bound differences favor Deep Sets; negative training-time differences favor Deep Sets for this fixed update budget.")
        lines.append("")
        for method in args.methods:
            values = [row["test_bound_deepsets_minus_other"] for row in pairs if row["other_method"] == method]
            if values:
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                lines.append(f"- Deep Sets minus {method}: {statistics.mean(values):+.5f} ± {sd:.5f} nats (mean ± seed SD).")
    lines.extend([
        "", "Training seconds include simulation, likelihood, backward pass, clipping, and optimizer updates, with CUDA synchronization. They exclude validation, checkpoint writes, plotting, and process setup; Each method receives excluded warm-up updates, followed by restoration of its exact initial weights and a fresh optimizer; optimizer allocation costs during actual training are included. Method order rotates across seeds. Smaller step counts are pipeline checks and cannot establish convergence or a speed advantage.", "",
        "If a --target-bound was specified before the run, metrics.csv reports the first validation checkpoint to reach it and its cumulative training seconds; unreached targets are null/blank. Thresholds are observed only at validation checkpoints and are subject to Monte Carlo noise.", "",
        "Raw and statistics encoders project to the same 16-dimensional bottleneck with approximately matched encoder parameters. These are controlled representation comparisons, not reproductions of the archived direct-concatenation policies. All methods see 127 amplitudes only; sigma is not supplied to the policy.", "",
        "Training, validation, and test streams are separate. Validation is reused for checkpoint selection; test is evaluated once after selection. Common theta, rho, Gaussian noise, and contrastive candidates are paired across methods. Adaptive beams and observations naturally differ between policies.", "",
        "The physical prior is theta uniform on [30°,150°] and rho uniform on [0.05,1]. Noise RMS is rho * 10^(-reference_SNR/20), matching the existing DAD objective. Beam gains come from the learned sequential policies, not a uniform gain proxy.", "",
        "Artifacts: `learning_curves.csv`, `dad_learning_curves.png`, `dad_test_bound.png`, `metrics.csv`, `summary.csv`, `paired_seed_differences.csv`, and per-seed model checkpoints and held-out per-trajectory bounds.", "",
    ])
    (out / "report.md").write_text("\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--L", type=int, default=64, help="Contrastive candidates per trajectory; objective ceiling log(L+1).")
    parser.add_argument("--experiments", "--T", type=int, default=3)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--rho-grid-size", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoding-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-bound", type=float, default=None, help="Optional predeclared validation bound target; report first observed step/time reaching it.")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="Use an explicit available device, e.g. cuda:1; defaults to cpu.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Excluded warm-up updates; restore initial weights and optimizer before actual training.")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; record for reproducible timings.")
    parser.add_argument("--output", default=None, help="New output directory; an existing path is rejected.")
    args = parser.parse_args(argv)
    positive = ("steps", "batch_size", "L", "experiments", "nx", "rho_grid_size", "hidden_dim", "encoding_dim", "candidate_chunk_size", "eval_every", "eval_batches", "eval_batch_size", "threads")
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_bound is not None and not math.isfinite(args.target_bound):
        parser.error("--target-bound must be finite")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    if args.nx < 2 or args.rho_grid_size < 2:
        parser.error("--nx and --rho-grid-size must be at least 2")
    if args.experiments < 2:
        parser.error("--experiments must be at least 2; with one beam no amplitude encoder is used")
    if not math.isfinite(args.snr_db) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.grad_clip) or args.grad_clip <= 0:
        parser.error("SNR must be finite, and learning rate/gradient clip must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("--seeds and --methods must not contain duplicates")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = str(Path("deep_sets_amp/results") / f"dad_{stamp}")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This benchmark supports cpu and cuda devices.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    params = Params(Nx=args.nx, Ny=1)
    pilot = generate_pilot_sequence(device=device, sequence_type="PSS")
    rho_grid = torch.linspace(0.05, 1.0, args.rho_grid_size, device=device)
    log_rho_prior = torch.full_like(rho_grid, -math.log(args.rho_grid_size))
    likelihood = make_log_likelihood_fn(rho_grid, pilot, args.snr_db, params)
    source_files = (
        "deep_sets_amp/benchmark_dad.py", "deep_sets_amp/benchmark_models.py",
        "modules/dad/policy.py", "modules/dad/encoder.py", "modules/dad/encoder_amp.py", "modules/dad/emitter.py",
        "modules/dad/contrastive.py", "modules/dad/experiment.py",
        "modules/beam_eig/likelihood.py", "modules/beam_eig/simulator.py",
        "modules/beam_eig/array_model.py", "modules/beam_eig/pilot.py", "modules/beam_eig/params.py",
    )
    root = Path(__file__).resolve().parents[1]
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    manifest = {
        "config": vars(args), "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "platform": platform.platform(), "device": str(device), "torch_threads": torch.get_num_threads(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "git_head": git_head,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "objective": "contrastive bound on information about theta with rho marginalized; not angle error",
        "timing": "synchronized simulation+objective+backward+optimizer seconds, excluding warm-up, validation and I/O",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results, history = [], []
    for index, seed in enumerate(args.seeds):
        offset = index % len(args.methods)
        order = args.methods[offset:] + args.methods[:offset]
        for method in order:
            result, trace = train_one(
                method, seed, args, device, params, pilot, likelihood, log_rho_prior, out,
            )
            results.append(result)
            history.extend(trace)
            write_csv(out / "metrics.csv", results)
            write_csv(out / "learning_curves.csv", history)
    summarize(results, history, args, out)
    print(f"DAD benchmark artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
