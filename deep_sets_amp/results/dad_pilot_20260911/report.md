# DAD amplitude encoder comparison

All policies train from scratch on the DAD contrastive objective. Only the amplitude encoder changes; the common policy layers start identically within each seed. No Rice regression weights are loaded.

T=3, reference SNR=0 dB, L=32, rho grid=50, steps=500, seeds=[11, 22, 33].

The measured outcome is held-out contrastive bound (higher is better), not angle estimation accuracy. Finite-L contrastive bounds are capped at log(L+1); this is not an exact EIG estimate. A coarse rho grid also changes the objective approximation.

| Method | Parameters | Test bound mean ± seed SD (nats) | Training seconds mean ± SD |
|---|---:|---:|---:|
| deepsets | 34359 | 0.93732 ± 0.20714 | 8.34 ± 0.09 |
| raw | 34343 | 0.91945 ± 0.15202 | 8.16 ± 0.02 |
| mean_var | 34354 | 1.03755 ± 0.09407 | 8.24 ± 0.03 |
| expanded_stats | 34351 | 1.08605 ± 0.07838 | 8.99 ± 0.05 |

Reported seed SD describes run variability; with one seed it is zero by convention and does not establish uncertainty. Test Monte Carlo standard errors are saved separately in metrics.csv.

Paired differences use the same training/test random streams per seed. Positive bound differences favor Deep Sets; negative training-time differences favor Deep Sets for this fixed update budget.

- Deep Sets minus raw: +0.01787 ± 0.25018 nats (mean ± seed SD).
- Deep Sets minus mean_var: -0.10023 ± 0.13944 nats (mean ± seed SD).
- Deep Sets minus expanded_stats: -0.14873 ± 0.22585 nats (mean ± seed SD).

Training seconds include simulation, likelihood, backward pass, clipping, and optimizer updates, with CUDA synchronization. They exclude validation, checkpoint writes, plotting, and process setup; Each method receives excluded warm-up updates, followed by restoration of its exact initial weights and a fresh optimizer; optimizer allocation costs during actual training are included. Method order rotates across seeds. Smaller step counts are pipeline checks and cannot establish convergence or a speed advantage.

If a --target-bound was specified before the run, metrics.csv reports the first validation checkpoint to reach it and its cumulative training seconds; unreached targets are null/blank. Thresholds are observed only at validation checkpoints and are subject to Monte Carlo noise.

Raw and statistics encoders project to the same 16-dimensional bottleneck with approximately matched encoder parameters. These are controlled representation comparisons, not reproductions of the archived direct-concatenation policies. All methods see 127 amplitudes only; sigma is not supplied to the policy.

Training, validation, and test streams are separate. Validation is reused for checkpoint selection; test is evaluated once after selection. Common theta, rho, Gaussian noise, and contrastive candidates are paired across methods. Adaptive beams and observations naturally differ between policies.

The physical prior is theta uniform on [30°,150°] and rho uniform on [0.05,1]. Noise RMS is rho * 10^(-reference_SNR/20), matching the existing DAD objective. Beam gains come from the learned sequential policies, not a uniform gain proxy.

Artifacts: `learning_curves.csv`, `dad_learning_curves.png`, `dad_test_bound.png`, `metrics.csv`, `summary.csv`, `paired_seed_differences.csv`, and per-seed model checkpoints and held-out per-trajectory bounds.
