# Amplitude encoder benchmark

3000 updates × 512 sets per method/seed; seeds [11, 22, 33]; device cuda:1.
All encoders train from scratch. Checkpoint selection uses independent validation MSE; test data are used once afterward.
Training and validation/test examples share the same prior within each scenario; physical scenarios use actual random beams.
The physical_snr validation prior is continuous uniform reference SNR [-10,10] dB; its test slices fix reference SNR.
Known sigma is available implicitly to every model in fixed-noise runs, but is an oracle side-information baseline in physical_snr.
Constants are fitted on a separate prior-calibration sample, never on validation/test targets. Clipped moments uses the known [0,1] support.

## Held-out accuracy

Mean ± sample SD across seeds (not confidence intervals).

| Scenario | Reference SNR | Method | MAE | RMSE |
|---|---|---|---:|---:|
| physical_fixed | fixed | Moments | 0.31003 ± 0.00137 | 0.36456 ± 0.00124 |
| physical_fixed | fixed | Moments clipped to [0,1] | 0.31003 ± 0.00137 | 0.36456 ± 0.00124 |
| physical_fixed | fixed | Known sigma second moment | 0.12991 ± 0.00062 | 0.16037 ± 0.00075 |
| physical_fixed | fixed | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_fixed | fixed | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_fixed | fixed | Deep Sets | 0.08362 ± 0.00126 | 0.10400 ± 0.00148 |
| physical_fixed | fixed | Raw 127 | 0.08572 ± 0.00218 | 0.10596 ± 0.00177 |
| physical_fixed | fixed | Mean + variance | 0.08374 ± 0.00141 | 0.10396 ± 0.00147 |
| physical_fixed | fixed | Six statistics | 0.08369 ± 0.00132 | 0.10403 ± 0.00140 |
| physical_snr | -10.0 | Moments | 0.52205 ± 0.00467 | 0.70857 ± 0.00335 |
| physical_snr | -10.0 | Moments clipped to [0,1] | 0.40335 ± 0.00463 | 0.49642 ± 0.00464 |
| physical_snr | -10.0 | Known sigma second moment (oracle) | 0.21827 ± 0.00146 | 0.29814 ± 0.00193 |
| physical_snr | -10.0 | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_snr | -10.0 | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_snr | -5.0 | Moments | 0.28682 ± 0.00207 | 0.36805 ± 0.00182 |
| physical_snr | -5.0 | Moments clipped to [0,1] | 0.28080 ± 0.00239 | 0.35522 ± 0.00253 |
| physical_snr | -5.0 | Known sigma second moment (oracle) | 0.12645 ± 0.00090 | 0.16896 ± 0.00073 |
| physical_snr | -5.0 | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_snr | -5.0 | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_snr | 0.0 | Moments | 0.15473 ± 0.00107 | 0.19655 ± 0.00106 |
| physical_snr | 0.0 | Moments clipped to [0,1] | 0.15473 ± 0.00107 | 0.19655 ± 0.00106 |
| physical_snr | 0.0 | Known sigma second moment (oracle) | 0.06415 ± 0.00031 | 0.08963 ± 0.00057 |
| physical_snr | 0.0 | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_snr | 0.0 | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_snr | 5.0 | Moments | 0.07286 ± 0.00044 | 0.10291 ± 0.00067 |
| physical_snr | 5.0 | Moments clipped to [0,1] | 0.07286 ± 0.00044 | 0.10291 ± 0.00067 |
| physical_snr | 5.0 | Known sigma second moment (oracle) | 0.02821 ± 0.00030 | 0.04134 ± 0.00046 |
| physical_snr | 5.0 | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_snr | 5.0 | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_snr | 10.0 | Moments | 0.02696 ± 0.00030 | 0.04513 ± 0.00047 |
| physical_snr | 10.0 | Moments clipped to [0,1] | 0.02696 ± 0.00030 | 0.04513 ± 0.00047 |
| physical_snr | 10.0 | Known sigma second moment (oracle) | 0.01221 ± 0.00016 | 0.01829 ± 0.00032 |
| physical_snr | 10.0 | Prior mean | 0.10121 ± 0.00161 | 0.12784 ± 0.00102 |
| physical_snr | 10.0 | Prior median | 0.09823 ± 0.00158 | 0.13161 ± 0.00095 |
| physical_snr | -10.0 | Deep Sets | 0.11181 ± 0.00343 | 0.12935 ± 0.00380 |
| physical_snr | -5.0 | Deep Sets | 0.09057 ± 0.00206 | 0.11276 ± 0.00212 |
| physical_snr | 0.0 | Deep Sets | 0.07122 ± 0.00109 | 0.10045 ± 0.00071 |
| physical_snr | 5.0 | Deep Sets | 0.06019 ± 0.00291 | 0.08738 ± 0.00434 |
| physical_snr | 10.0 | Deep Sets | 0.05708 ± 0.00967 | 0.07236 ± 0.01415 |
| physical_snr | -10.0 | Raw 127 | 0.12968 ± 0.03087 | 0.15577 ± 0.04646 |
| physical_snr | -5.0 | Raw 127 | 0.11579 ± 0.04244 | 0.14484 ± 0.05584 |
| physical_snr | 0.0 | Raw 127 | 0.10250 ± 0.05352 | 0.13653 ± 0.06288 |
| physical_snr | 5.0 | Raw 127 | 0.09724 ± 0.05747 | 0.13284 ± 0.06583 |
| physical_snr | 10.0 | Raw 127 | 0.10341 ± 0.05111 | 0.13834 ± 0.06066 |
| physical_snr | -10.0 | Mean + variance | 0.10700 ± 0.00157 | 0.12475 ± 0.00122 |
| physical_snr | -5.0 | Mean + variance | 0.08646 ± 0.00129 | 0.10942 ± 0.00169 |
| physical_snr | 0.0 | Mean + variance | 0.07060 ± 0.00158 | 0.10008 ± 0.00199 |
| physical_snr | 5.0 | Mean + variance | 0.05762 ± 0.00093 | 0.08244 ± 0.00134 |
| physical_snr | 10.0 | Mean + variance | 0.04748 ± 0.00235 | 0.05878 ± 0.00321 |
| physical_snr | -10.0 | Six statistics | 0.10892 ± 0.00036 | 0.12694 ± 0.00022 |
| physical_snr | -5.0 | Six statistics | 0.08535 ± 0.00111 | 0.10858 ± 0.00166 |
| physical_snr | 0.0 | Six statistics | 0.07016 ± 0.00035 | 0.09960 ± 0.00046 |
| physical_snr | 5.0 | Six statistics | 0.05658 ± 0.00073 | 0.08091 ± 0.00115 |
| physical_snr | 10.0 | Six statistics | 0.03697 ± 0.00089 | 0.04778 ± 0.00095 |
| uniform_fixed | fixed | Moments | 0.25576 ± 0.00137 | 0.32472 ± 0.00111 |
| uniform_fixed | fixed | Moments clipped to [0,1] | 0.25229 ± 0.00146 | 0.32349 ± 0.00115 |
| uniform_fixed | fixed | Known sigma second moment | 0.09966 ± 0.00096 | 0.13010 ± 0.00124 |
| uniform_fixed | fixed | Prior mean | 0.25013 ± 0.00063 | 0.28905 ± 0.00026 |
| uniform_fixed | fixed | Prior median | 0.25012 ± 0.00063 | 0.28903 ± 0.00026 |
| uniform_fixed | fixed | Deep Sets | 0.08654 ± 0.00039 | 0.11108 ± 0.00047 |
| uniform_fixed | fixed | Raw 127 | 0.08888 ± 0.00046 | 0.11343 ± 0.00052 |
| uniform_fixed | fixed | Mean + variance | 0.08671 ± 0.00037 | 0.11114 ± 0.00039 |
| uniform_fixed | fixed | Six statistics | 0.08665 ± 0.00044 | 0.11107 ± 0.00048 |

## Training cost

Training seconds include data generation and synchronized forward/backward/Adam; warm-up, validation, checkpointing, and plots are excluded.
Inference timings include feature extraction. Parameter budgets are approximately matched; compute budgets are measured, not assumed equal.
GPU peak allocation includes resident evaluation data and model/optimizer tensors, not only activations.

| Scenario | Method | Parameters | Train seconds | Inference ms/batch |
|---|---|---:|---:|---:|
| physical_fixed | Deep Sets | 5937 | 4.96 | 0.352 |
| physical_fixed | Raw 127 | 5921 | 3.82 | 0.073 |
| physical_fixed | Mean + variance | 5932 | 3.95 | 0.108 |
| physical_fixed | Six statistics | 5929 | 4.50 | 0.246 |
| physical_snr | Deep Sets | 5937 | 4.91 | 0.353 |
| physical_snr | Raw 127 | 5921 | 3.86 | 0.073 |
| physical_snr | Mean + variance | 5932 | 4.08 | 0.108 |
| physical_snr | Six statistics | 5929 | 4.57 | 0.247 |
| uniform_fixed | Deep Sets | 5937 | 5.00 | 0.350 |
| uniform_fixed | Raw 127 | 5921 | 3.88 | 0.073 |
| uniform_fixed | Mean + variance | 5932 | 4.13 | 0.108 |
| uniform_fixed | Six statistics | 5929 | 4.51 | 0.246 |

## Paired Deep Sets differences

Negative error differences favor Deep Sets; each subtraction uses the same held-out sets and seed.

| Scenario | SNR | Comparison | RMSE difference (mean ± SD) |
|---|---|---|---:|
| physical_fixed | fixed | deepsets_minus_raw | -0.00196 ± 0.00097 |
| physical_fixed | fixed | deepsets_minus_mean_var | 0.00004 ± 0.00008 |
| physical_fixed | fixed | deepsets_minus_expanded_stats | -0.00003 ± 0.00013 |
| physical_snr | -10.0 | deepsets_minus_raw | -0.02642 ± 0.04973 |
| physical_snr | -10.0 | deepsets_minus_mean_var | 0.00460 ± 0.00388 |
| physical_snr | -10.0 | deepsets_minus_expanded_stats | 0.00241 ± 0.00402 |
| physical_snr | -5.0 | deepsets_minus_raw | -0.03208 ± 0.05742 |
| physical_snr | -5.0 | deepsets_minus_mean_var | 0.00334 ± 0.00045 |
| physical_snr | -5.0 | deepsets_minus_expanded_stats | 0.00418 ± 0.00049 |
| physical_snr | 0.0 | deepsets_minus_raw | -0.03608 ± 0.06352 |
| physical_snr | 0.0 | deepsets_minus_mean_var | 0.00037 ± 0.00192 |
| physical_snr | 0.0 | deepsets_minus_expanded_stats | 0.00085 ± 0.00032 |
| physical_snr | 5.0 | deepsets_minus_raw | -0.04545 ± 0.06148 |
| physical_snr | 5.0 | deepsets_minus_mean_var | 0.00494 ± 0.00433 |
| physical_snr | 5.0 | deepsets_minus_expanded_stats | 0.00647 ± 0.00391 |
| physical_snr | 10.0 | deepsets_minus_raw | -0.06598 ± 0.04845 |
| physical_snr | 10.0 | deepsets_minus_mean_var | 0.01357 ± 0.01675 |
| physical_snr | 10.0 | deepsets_minus_expanded_stats | 0.02458 ± 0.01366 |
| uniform_fixed | fixed | deepsets_minus_raw | -0.00235 ± 0.00044 |
| uniform_fixed | fixed | deepsets_minus_mean_var | -0.00006 ± 0.00012 |
| uniform_fixed | fixed | deepsets_minus_expanded_stats | 0.00002 ± 0.00001 |

## Scope

This supervised nu task measures representation utility, not DAD policy quality. Use benchmark_dad for the downstream contrastive objective.
A small run validates the experiment and provides preliminary measurements; it does not establish convergence or general superiority.
All architectures use the same optimizer and learning rate. A final architecture claim should also check comparable hyperparameter-search budgets.
Random physical beams differ from beams chosen by a learned adaptive policy. Sparse high-nu bins must be interpreted using the counts in nu_bins.csv.
Learning-curve bands and tables show sample SD across seeds. No uncertainty estimate is meaningful with a single seed.
