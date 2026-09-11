"""Controlled encoder comparison; run with python -m deep_sets_amp.benchmark_rice."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_sets_amp.benchmark_data import SCENARIOS, SNR_VALUES, Sampler
from deep_sets_amp.benchmark_models import AmplitudeRegressor, known_sigma_estimate, moment_estimate


METHODS = ("deepsets", "raw", "mean_var", "expanded_stats")
LABELS = {"deepsets": "Deep Sets", "raw": "Raw 127", "mean_var": "Mean + variance",
          "expanded_stats": "Six statistics", "moments": "Moments",
          "moments_clipped": "Moments clipped to [0,1]", "known_sigma": "Known sigma second moment",
          "prior_mean": "Prior mean", "prior_median": "Prior median"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def predict(model, r, batch_size):
    model.eval()
    return torch.cat([model(chunk) for chunk in r.split(batch_size)])


def metrics(pred, true):
    error = (pred - true).double()
    # Correlation is undefined for a constant prediction.
    corr = None
    if pred.double().std(unbiased=False) > 1e-10 and true.double().std(unbiased=False) > 1e-10:
        corr = torch.corrcoef(torch.stack((pred.double(), true.double())))[0, 1].item()
    return {"n": true.numel(), "mae": error.abs().mean().item(),
            "rmse": error.square().mean().sqrt().item(), "bias": error.mean().item(), "corr": corr}


def evaluate_predictions(pred, data, identity):
    row = {**identity, **metrics(pred, data.nu),
           "median_effective_snr_db": data.effective_snr_db.median().item()}
    bins = []
    for i in range(10):
        mask = (data.nu >= i / 10) & (data.nu < (i + 1) / 10)
        if mask.any():
            bins.append({**identity, "nu_low": i / 10, "nu_high": (i + 1) / 10,
                         **metrics(pred[mask], data.nu[mask])})
    return row, bins


@torch.no_grad()
def inference_ms(model, r, repeats, device):
    for _ in range(3):
        model(r)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(r)
    synchronize(device)
    return 1000 * (time.perf_counter() - start) / repeats


def train_one(args, scenario, seed, method, validation, output, metadata):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = AmplitudeRegressor(method).to(device)
    # Architecture-dependent initialization consumes different RNG counts.
    # Reset the common head explicitly so it starts identically for all methods.
    torch.manual_seed(seed + 10_000)
    for layer in model.head:
        if isinstance(layer, torch.nn.Linear):
            layer.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Warm libraries without retaining weight updates or advancing training RNG.
    initial_state = cpu_state(model)
    warmup = Sampler(scenario, seed + 5_000_000, device, args.nx)
    synchronize(device)
    warmup_start = time.perf_counter()
    for _ in range(args.warmup_steps):
        data = warmup.sample(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        (model(data.r) - data.nu).square().mean().backward()
        optimizer.step()
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    del initial_state, warmup
    sampler = Sampler(scenario, seed + 1_000_000, device, args.nx)
    history = []
    best_mse, best_step, best_state = float("inf"), 0, None
    compute_seconds = data_seconds = 0.0
    loss_sum = loss_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.steps + 1):
        if step:
            synchronize(device)
            start = time.perf_counter()
            data = sampler.sample(args.batch_size)
            synchronize(device)
            data_seconds += time.perf_counter() - start
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(data.r) - data.nu).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {scenario}/{seed}/{method}/{step}")
            loss.backward()
            optimizer.step()
            synchronize(device)
            compute_seconds += time.perf_counter() - start
            loss_sum += loss.item()
            loss_count += 1
        if step % args.eval_every == 0 or step == args.steps:
            val_pred = predict(model, validation.r, args.eval_batch_size)
            val_mse = (val_pred - validation.nu).double().square().mean().item()
            if val_mse < best_mse:
                best_mse, best_step, best_state = val_mse, step, cpu_state(model)
            history.append({"scenario": scenario, "seed": seed, "method": method, "step": step,
                            "examples": step * args.batch_size, "compute_seconds": compute_seconds,
                            "data_seconds": data_seconds, "train_seconds": compute_seconds + data_seconds,
                            "train_mse": loss_sum / loss_count if loss_count else None,
                            "validation_rmse": val_mse ** 0.5, "best_validation_rmse": best_mse ** 0.5})
            loss_sum = loss_count = 0
            print(f"{scenario} seed={seed} {method:14s} step={step:5d} "
                  f"val RMSE={val_mse ** .5:.5f} train={compute_seconds + data_seconds:.1f}s", flush=True)

    checkpoint = {"model_state_dict": best_state, "method": method, "seed": seed,
                  "scenario": scenario, "best_step": best_step, "best_validation_mse": best_mse,
                  "config": vars(args), "metadata": metadata}
    torch.save(checkpoint, output / f"{scenario}_seed{seed}_{method}.pt")
    model.load_state_dict(best_state)
    model.eval()
    permutation = torch.arange(126, -1, -1, device=device)
    r_probe = validation.r[:min(args.batch_size, validation.nu.numel())]
    with torch.no_grad():
        difference = (model(r_probe) - model(r_probe[:, permutation])).abs()
    timings = {"scenario": scenario, "seed": seed, "method": method,
               "parameters": sum(p.numel() for p in model.parameters()), "best_step": best_step,
               "warmup_seconds": warmup_seconds,
               "compute_seconds": compute_seconds, "data_seconds": data_seconds,
               "train_seconds": compute_seconds + data_seconds,
               "examples_per_compute_second": args.steps * args.batch_size / compute_seconds,
               "inference_batch_size": r_probe.shape[0],
               "inference_ms": inference_ms(model, r_probe, args.timing_repeats, device),
               "permutation_max_difference": difference.max().item(),
               "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
    return model, history, timings


def aggregate(rows, keys, values):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, members in groups.items():
        item = dict(zip(keys, group))
        item["seeds"] = len(members)
        for key in values:
            numbers = [x[key] for x in members]
            item[key + "_mean"] = statistics.mean(numbers)
            item[key + "_std"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        result.append(item)
    return result


def paired_differences(rows):
    index = {(r["scenario"], r["level"], r["seed"], r["method"]): r for r in rows}
    result = []
    for row in rows:
        if row["method"] == "deepsets":
            for other in METHODS[1:]:
                key = (row["scenario"], row["level"], row["seed"], other)
                if key in index:
                    result.append({"scenario": row["scenario"], "level": row["level"], "seed": row["seed"],
                                   "comparison": "deepsets_minus_" + other,
                                   "mae_delta": row["mae"] - index[key]["mae"],
                                   "rmse_delta": row["rmse"] - index[key]["rmse"]})
    return result


def plots(args, output, histories, summaries, bin_rows):
    for scenario in args.scenarios:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for method in args.methods:
            for ax, x in zip(axes, ("step", "train_seconds")):
                curves = [[r for r in histories if r["scenario"] == scenario and
                           r["method"] == method and r["seed"] == seed] for seed in args.seeds]
                yy = np.array([[r["validation_rmse"] for r in c] for c in curves])
                if x == "step":
                    xx = [r[x] for r in curves[0]]
                    line, = ax.plot(xx, yy.mean(0), label=LABELS[method])
                    spread = yy.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(xx, dtype=float)
                    ax.fill_between(xx, yy.mean(0) - spread, yy.mean(0) + spread,
                                    color=line.get_color(), alpha=.15)
                else:
                    # Do not average scores from different elapsed times.
                    color = None
                    for index, curve in enumerate(curves):
                        line, = ax.plot([r[x] for r in curve], [r["validation_rmse"] for r in curve],
                                        color=color, alpha=.7, label=LABELS[method] if index == 0 else None)
                        color = line.get_color()
                ax.set(xlabel="Optimizer updates" if x == "step" else "Training seconds (data + model)",
                       ylabel="Validation RMSE")
                ax.grid(alpha=.3)
                ax.legend(fontsize=8)
        fig.suptitle(f"{scenario}: mean ± seed SD (left); individual seeds (right)")
        fig.tight_layout()
        fig.savefig(output / f"{scenario}_learning.png", dpi=180)
        plt.close(fig)

        if scenario == "physical_snr":
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for method in (*args.methods, "moments_clipped", "known_sigma", "prior_mean"):
                selected = sorted([r for r in summaries if r["scenario"] == scenario and r["method"] == method],
                                  key=lambda r: float(r["level"]))
                label = LABELS[method] + (" (oracle)" if method == "known_sigma" else "")
                for ax, metric in zip(axes, ("mae", "rmse")):
                    ax.errorbar([float(r["level"]) for r in selected],
                                [r[metric + "_mean"] for r in selected],
                                yerr=[r[metric + "_std"] for r in selected], marker="o",
                                linestyle="-" if method in METHODS else "--", label=label, capsize=3)
                    ax.set(xlabel="Reference SNR [dB]", ylabel=metric.upper() + " on nu")
                    ax.grid(alpha=.3)
                    ax.legend(fontsize=7)
            fig.suptitle("Matched physical prior: held-out test, mean and seed SD")
            fig.tight_layout()
            fig.savefig(output / "physical_snr_test.png", dpi=180)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            for method in (*args.methods, "moments_clipped", "known_sigma"):
                selected = [r for r in bin_rows if r["scenario"] == scenario and r["method"] == method]
                # Weight each bin by its actual number of held-out observations.
                xx, yy = [], []
                for low in sorted({r["nu_low"] for r in selected}):
                    group = [r for r in selected if r["nu_low"] == low]
                    xx.append(low + .05)
                    yy.append(sum(r["mae"] * r["n"] for r in group) / sum(r["n"] for r in group))
                ax.plot(xx, yy, marker="o", label=LABELS[method])
            ax.set(xlabel="True nu (bin center)", ylabel="Held-out MAE", title=scenario)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / f"{scenario}_mae_by_nu.png", dpi=180)
            plt.close(fig)


def report(args, output, summaries, timings, paired):
    lines = ["# Amplitude encoder benchmark", "",
             f"{args.steps} updates × {args.batch_size} sets per method/seed; seeds {args.seeds}; device {args.device}.",
             "All encoders train from scratch. Checkpoint selection uses independent validation MSE; test data are used once afterward.",
             "Training and validation/test examples share the same prior within each scenario; physical scenarios use actual random beams.",
             "The physical_snr validation prior is continuous uniform reference SNR [-10,10] dB; its test slices fix reference SNR.",
             "Known sigma is available implicitly to every model in fixed-noise runs, but is an oracle side-information baseline in physical_snr.",
             "Constants are fitted on a separate prior-calibration sample, never on validation/test targets. Clipped moments uses the known [0,1] support.",
             "", "## Held-out accuracy", "", "Mean ± sample SD across seeds (not confidence intervals).", "",
             "| Scenario | Reference SNR | Method | MAE | RMSE |", "|---|---|---|---:|---:|"]
    for r in summaries:
        label = LABELS[r["method"]] + (" (oracle)" if r["scenario"] == "physical_snr" and r["method"] == "known_sigma" else "")
        lines.append(f"| {r['scenario']} | {r['level']} | {label} | {r['mae_mean']:.5f} ± {r['mae_std']:.5f} | "
                     f"{r['rmse_mean']:.5f} ± {r['rmse_std']:.5f} |")
    lines += ["", "## Training cost", "", "Training seconds include data generation and synchronized forward/backward/Adam; warm-up, validation, checkpointing, and plots are excluded.",
              "Inference timings include feature extraction. Parameter budgets are approximately matched; compute budgets are measured, not assumed equal.",
              "GPU peak allocation includes resident evaluation data and model/optimizer tensors, not only activations.", "",
              "| Scenario | Method | Parameters | Train seconds | Inference ms/batch |", "|---|---|---:|---:|---:|"]
    for r in aggregate(timings, ("scenario", "method"), ("parameters", "train_seconds", "inference_ms")):
        lines.append(f"| {r['scenario']} | {LABELS[r['method']]} | {r['parameters_mean']:.0f} | {r['train_seconds_mean']:.2f} | {r['inference_ms_mean']:.3f} |")
    lines += ["", "## Paired Deep Sets differences", "", "Negative error differences favor Deep Sets; each subtraction uses the same held-out sets and seed.",
              "", "| Scenario | SNR | Comparison | RMSE difference (mean ± SD) |", "|---|---|---|---:|"]
    for r in aggregate(paired, ("scenario", "level", "comparison"), ("rmse_delta",)):
        lines.append(f"| {r['scenario']} | {r['level']} | {r['comparison']} | {r['rmse_delta_mean']:.5f} ± {r['rmse_delta_std']:.5f} |")
    lines += ["", "## Scope", "", "This supervised nu task measures representation utility, not DAD policy quality. Use benchmark_dad for the downstream contrastive objective.",
              "A small run validates the experiment and provides preliminary measurements; it does not establish convergence or general superiority.",
              "All architectures use the same optimizer and learning rate. A final architecture claim should also check comparable hyperparameter-search budgets.",
              "Random physical beams differ from beams chosen by a learned adaptive policy. Sparse high-nu bins must be interpreted using the counts in nu_bins.csv.",
              "Learning-curve bands and tables show sample SD across seeds. No uncertainty estimate is meaningful with a single seed.", ""]
    (output / "report.md").write_text("\n".join(lines))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--validation-samples", type=int, default=4096)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--calibration-samples", type=int, default=16384)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--timing-repeats", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--device", default="cpu", help="Explicit CUDA device, e.g. cuda:1; defaults to CPU")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--save-predictions", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    for key in ("steps", "batch_size", "eval_every", "validation_samples", "test_samples", "calibration_samples",
                "eval_batch_size", "timing_repeats", "nx", "threads"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("--learning-rate must be positive")
    if args.warmup_steps < 0:
        p.error("--warmup-steps must be nonnegative")
    if any(len(set(getattr(args, k))) != len(getattr(args, k)) for k in ("seeds", "methods", "scenarios")):
        p.error("Duplicate seeds, methods, or scenarios are not allowed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sources = [Path(__file__), Path(__file__).with_name("benchmark_data.py"), Path(__file__).with_name("benchmark_models.py")]
    root = Path(__file__).resolve().parents[1]
    sources += [root / name for name in ("modules/dad/encoder_amp.py", "modules/beam_eig/array_model.py",
                                        "modules/beam_eig/simulator.py", "modules/beam_eig/pilot.py", "modules/beam_eig/params.py")]
    metadata = {"torch": str(torch.__version__), "python": platform.python_version(),
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "cuda": torch.version.cuda, "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
                "training_seed_offset": 1_000_000, "validation_seed_offset": 2_000_000,
                "test_seed_offset": 3_000_000, "calibration_seed_offset": 4_000_000}
    (output / "config.json").write_text(json.dumps({"config": vars(args), "metadata": metadata}, indent=2))
    rows, bins, histories, timings = [], [], [], []
    for scenario in args.scenarios:
        for seed in args.seeds:
            validation = Sampler(scenario, seed + 2_000_000, device, args.nx).sample(args.validation_samples)
            calibration = Sampler(scenario, seed + 4_000_000, device, args.nx).sample(args.calibration_samples)
            prior_mean, prior_median = calibration.nu.mean().item(), calibration.nu.median().item()
            del calibration
            levels = SNR_VALUES if scenario == "physical_snr" else (None,)
            # Reset seed at each SNR to share theta/rho/beam/noise draws across slices.
            tests = {str(level) if level is not None else "fixed": Sampler(scenario, seed + 3_000_000, device, args.nx).sample(args.test_samples, level)
                     for level in levels}
            saved_predictions = {level: {"nu": d.nu.cpu(), "sigma": d.sigma.cpu(), "gain": d.gain.cpu(),
                                          "effective_snr_db": d.effective_snr_db.cpu()} for level, d in tests.items()}
            for level, data in tests.items():
                estimates = {"moments": moment_estimate(data.r), "moments_clipped": moment_estimate(data.r).clamp(0, 1),
                             "known_sigma": known_sigma_estimate(data.r, data.sigma),
                             "prior_mean": torch.full_like(data.nu, prior_mean), "prior_median": torch.full_like(data.nu, prior_median)}
                for method, pred in estimates.items():
                    identity = {"scenario": scenario, "seed": seed, "method": method, "level": level}
                    row, grouped = evaluate_predictions(pred, data, identity)
                    rows.append(row)
                    bins.extend(grouped)
                    saved_predictions[level][method] = pred.cpu()
            # Rotate model order across seeds to reduce systematic timing-order bias.
            order = list(args.methods)
            shift = args.seeds.index(seed) % len(order)
            order = order[shift:] + order[:shift]
            for method in order:
                model, curve, timing = train_one(args, scenario, seed, method, validation, output, metadata)
                histories.extend(curve)
                timings.append(timing)
                for level, data in tests.items():
                    pred = predict(model, data.r, args.eval_batch_size)
                    identity = {"scenario": scenario, "seed": seed, "method": method, "level": level}
                    row, grouped = evaluate_predictions(pred, data, identity)
                    rows.append(row)
                    bins.extend(grouped)
                    saved_predictions[level][method] = pred.cpu()
                del model
                write_csv(output / "metrics.csv", rows)
                write_csv(output / "learning_curves.csv", histories)
                write_csv(output / "timings.csv", timings)
                write_csv(output / "nu_bins.csv", bins)
            if args.save_predictions:
                torch.save(saved_predictions, output / f"{scenario}_seed{seed}_predictions.pt")
    summaries = aggregate(rows, ("scenario", "level", "method"), ("mae", "rmse", "bias"))
    paired = paired_differences(rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_differences.csv", paired)
    plots(args, output, histories, summaries, bins)
    report(args, output, summaries, timings, paired)
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
