# Amplitude encoder experiments

The original `training_rice_estimator.py` and `training_rice_snr.py` are preserved.
Their saved plots are historical: the fixed-noise plots cover nu in `[0,5]`,
whereas the current fixed-noise script specifies `[0,1]`.

The new benchmarks compare representations without assuming Deep Sets wins.
They train from scratch and do not load Rice weights into a DAD policy.
Run the commands below **from the repository root** with a Python environment
containing PyTorch, NumPy, and Matplotlib. In this workspace the interpreter is
`/home/egalarraga/.conda/envs/beam_dad/bin/python`.

## Four controlled representations

All representations produce 16 learned features. The regression head, or the
remaining DAD network, is identical and initialized identically within a seed.
The encoder parameter counts are approximately matched; the computational cost
is measured separately.

| Method | Features available to encoder | Encoder parameters |
|---|---|---:|
| `deepsets` | Shared per-amplitude network, mean pooling, set network | 5,360 |
| `raw` | Ordered vector of 127 amplitudes through an MLP | 5,344 |
| `mean_var` | Sample mean and population variance through an MLP | 5,355 |
| `expanded_stats` | Mean, variance, skewness, kurtosis, moment nu² and sigma² through an MLP | 5,352 |

Deep Sets uses the existing `modules/dad/encoder_amp.py` unchanged. Expanded
statistics follow the six-feature experiment already in the repository, with
safe clipped roots to permit backpropagation through observations in DAD.
The expanded set is a diagnostic for sensitivity to feature selection, not a
claim that these are the best possible six features.

The raw/statistics variants use a common 16-feature bottleneck. They are
controlled alternatives, not exact reproductions of archived policies that
concatenated raw amplitudes or statistics directly with beam features. Matching
parameter counts alone does not guarantee equal capacity, optimization
difficulty, or FLOPs. All methods share optimizer settings; architecture-specific
tuning would need comparable search budgets.

## Rice estimation benchmark

```bash
python -m deep_sets_amp.benchmark_rice \
  --device cuda:1 --steps 3000 --seeds 11 22 33 \
  --output deep_sets_amp/results/rice_comparison --save-predictions
```

Select an available device; CPU is the safe default. Output directories must be
new, so historical plots/checkpoints cannot be overwritten accidentally.

The default suite contains three separate experiments:

| Scenario | Training and evaluation prior | Noise |
|---|---|---|
| `physical_fixed` | theta U(30°,150°), rho U(.05,1), independent uniform beam phases, nu=rho times actual normalized beam gain | sigma=1 |
| `physical_snr` | Same physical priors | sigma=rho × 10^(-reference SNR/20) |
| `uniform_fixed` | nu U(0,1), direct Rice control matching the current original script | sigma=1 |

The physical generator calls the repository array functions and `simulate_y`
with the actual 127-symbol unit-magnitude PSS. It does not use uniform synthetic
gain. Physical train and test sets have the same distribution. They still use
random beams; an adaptive policy produces a different gain distribution.

In `physical_snr`, training and validation sample reference SNR uniformly in
`[-10,10]` dB. Test slices fix reference SNR at `-10,-5,0,5,10` dB and share
latent variables and standardized noise across the slices. The effective SNR is
`reference SNR + 20*log10(beam gain)`, and can be much lower.

Each training example has 127 magnitudes. Every model sees exactly the same
training stream within a seed. Validation and test streams are independent;
validation MSE selects the checkpoint, and each selected model is evaluated on
test data once. Test arrays are paired across architectures. Three seeds are the
default; reported SD describes run variability, not a confidence interval.

Analytical baselines are the fourth-moment estimate, that estimate clipped to
the known `[0,1]` support, and `sqrt(max(mean(r²)-sigma²,0))`. The last is a fair
known-noise comparator when sigma is fixed, and an **oracle** when sigma varies:
none of the neural estimators receive sigma. Constant mean/median predictors
are fitted on a separate prior sample, never on the test targets.

Artifacts include:

- `config.json`: complete arguments, device/library versions, random-stream
  offsets, and hashes of the relevant source files.
- `metrics.csv`, `summary.csv`: MAE, RMSE, bias, correlation, sample counts, and
  aggregation across seeds.
- `nu_bins.csv`: errors and counts in ten bins spanning `[0,1]`, including sparse
  high-amplitude bins under physical priors.
- `learning_curves.csv` and `*_learning.png`: validation accuracy versus updates
  and measured training time. The time panel shows individual seeds.
- `timings.csv`: data/model time, throughput, inference latency including feature
  extraction, permutation sensitivity, and GPU peak allocation.
- `physical_snr_test.png`, `*_mae_by_nu.png`, `paired_differences.csv`, `report.md`.
- Per-method validation-selected checkpoints, with configuration and metadata;
  optional paired prediction tensors for independent reanalysis.

Timing synchronizes the selected GPU, includes data generation and model
forward/backward/Adam, and excludes validation and artifact writing. Discarded
warm-up updates use a separate stream; weights and optimizer are reset before
measurement. Model order rotates across seeds. Inference uses a recorded batch
size. GPU memory measurements include resident evaluation tensors.

## Actual DAD training benchmark

```bash
python -m deep_sets_amp.benchmark_dad \
  --device cuda:1 --steps 5000 --seeds 11 22 33 \
  --T 3 --snr-db 0 --batch-size 64 --L 64 --rho-grid-size 50 \
  --output deep_sets_amp/results/dad_comparison
```

This replaces only `policy.encoder.amp` in a fresh `DADPolicy`. Beam phase
features, the downstream encoder, history aggregation, emitter, physical
rollout, and contrastive objective are retained. The learned representation can
therefore adapt to the policy objective; no nu regression loss is used.

Training randomness is paired across methods: true angles, channel amplitudes,
Gaussian noises, and contrastive candidates. Observations themselves differ as
the policies choose different beams. Each method uses independent validation
and test streams, with the best validation bound choosing its checkpoint.
`T` must be at least two, since a one-experiment policy never uses observations
to select a later beam.

The score is held-out **contrastive bound in nats**, higher being better. It is
neither angle-estimation accuracy nor exact EIG. The finite-candidate bound has
ceiling `log(L+1)`, and the rho integration grid is another approximation. A
coarse grid can produce misleading scores, especially at high reference SNR.
Keep objective approximations identical across methods and examine convergence
in these numerical settings before making a final policy claim.

The runner writes `report.md`, CSV results, learning curves versus updates and
synchronized time, test-bound plots, per-seed paired differences, best/final
checkpoints, and per-trajectory test bounds. The manifest records configuration,
hardware, versions, and source hashes. `--target-bound VALUE` optionally records
the first validation checkpoint reaching a target chosen before the run; this
is a useful way to compare time to comparable policy quality. Thresholds are
observed only at the validation interval and are subject to Monte Carlo noise.

## Interpreting the outcome

A lower Rice error demonstrates better nu estimation for that training budget
and prior. It does not establish a better DAD policy. A higher DAD bound at the
same number of updates demonstrates better objective learning per update; a
higher bound at the same training time addresses computational efficiency.
An encoder may win one comparison and lose another.

The purpose is to test the hypothesis, including outcomes where mean/variance
already provides enough useful information for this task. A fixed-size learned
representation can remain a useful design choice even if it does not beat every
hand-crafted baseline on this particular Rice family.

## Verification and small pipeline runs

```bash
python -m unittest discover -s deep_sets_amp -p 'test_benchmark_*.py' -v

python -m deep_sets_amp.benchmark_rice \
  --steps 2 --batch-size 8 --eval-every 1 --seeds 11 \
  --validation-samples 16 --test-samples 32 --calibration-samples 32 \
  --timing-repeats 2 --output /tmp/rice_pipeline_check

python -m deep_sets_amp.benchmark_dad \
  --steps 2 --batch-size 4 --L 3 --rho-grid-size 15 \
  --eval-batches 1 --eval-batch-size 4 --eval-every 1 --seeds 11 \
  --output /tmp/dad_pipeline_check
```

The tests check physical normalization, Rice moments, scalar/batch simulator
agreement, paired random streams, shape and permutation properties, stable
statistic gradients, parameter budgets, and DAD integration. Tiny pipeline runs
check executability; their scores do not establish convergence.
