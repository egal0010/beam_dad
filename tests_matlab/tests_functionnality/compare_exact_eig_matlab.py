"""
Exact deterministic MATLAB <-> PyTorch EIG comparison,
using the REAL production function estimate_eig_for_eta_mean().

Prerequisite:
    replace estimate_eig_for_eta_mean() in baseline_eig.py
    with the version supporting:
        idx_theta_samples=None,
        noise_samples=None,
        return_terms=False

Run MATLAB first:
    eig_exact_mc_debug.m

Then:
    python compare_exact_eig_using_baseline.py
"""

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from modules.beam_eig.baseline_eig import estimate_eig_for_eta_mean


MAT_PATH = Path(__file__).resolve().parent / "eig_exact_mc_debug.mat"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

data = loadmat(MAT_PATH)

p_theta_np = data["p_theta"].squeeze()
rho_mean = float(np.asarray(data["rho_mean"]).squeeze())
sigma = float(np.asarray(data["sigma"]).squeeze())

s_np = np.asarray(data["s"]).reshape(-1)
a_np = np.asarray(data["a"])
eta_grid_np = np.asarray(data["eta_grid"])

# MATLAB indices are 1-based -> Python 0-based
theta_sample_idx_np = (
    np.asarray(data["theta_sample_idx"], dtype=np.int64) - 1
)

# MATLAB stores [Ns, N, R]
noise_all_np = np.asarray(data["noise_all"])

eig_terms_matlab_np = np.asarray(data["eig_terms_all"])
eig_matlab_np = np.asarray(data["EIG_matlab"]).squeeze()

R = eta_grid_np.shape[1]
N = theta_sample_idx_np.shape[1]

s = torch.as_tensor(
    s_np,
    dtype=torch.complex128,
    device=device,
)

a = torch.as_tensor(
    a_np,
    dtype=torch.complex128,
    device=device,
)

p_theta = torch.as_tensor(
    p_theta_np,
    dtype=torch.float64,
    device=device,
)

eta_grid = torch.as_tensor(
    eta_grid_np,
    dtype=torch.float64,
    device=device,
)

sigma_t = torch.tensor(
    sigma,
    dtype=torch.float64,
    device=device,
)

eig_python = np.zeros(R, dtype=np.float64)
eig_terms_python = np.zeros((N, R), dtype=np.float64)

for d in range(R):

    eta = eta_grid[:, d:d+1]

    idx_theta_samples = torch.as_tensor(
        theta_sample_idx_np[d],
        dtype=torch.long,
        device=device,
    )

    # MATLAB: [Ns, N, R]
    # Production estimator expects: [N, Ns]
    noise_samples = torch.as_tensor(
        noise_all_np[:, :, d].T,
        dtype=torch.complex128,
        device=device,
    )

    eig_value, eig_terms = estimate_eig_for_eta_mean(
        eta=eta,
        a_grid=a,
        s=s,
        sigma=sigma_t,
        p_theta=p_theta,
        rho_mean=rho_mean,
        N=N,
        idx_theta_samples=idx_theta_samples,
        noise_samples=noise_samples,
        return_terms=True,
    )

    eig_python[d] = eig_value.item()
    eig_terms_python[:, d] = (
        eig_terms.detach().cpu().numpy()
    )

eig_diff = np.abs(eig_python - eig_matlab_np)
term_diff = np.abs(
    eig_terms_python - eig_terms_matlab_np
)

print()
print("=" * 76)
print("EXACT TEST USING PRODUCTION estimate_eig_for_eta_mean()")
print("=" * 76)

for d in range(R):
    print(
        f"Beam {d+1:2d} | "
        f"MATLAB={eig_matlab_np[d]: .12f} | "
        f"PyTorch={eig_python[d]: .12f} | "
        f"|diff|={eig_diff[d]:.3e}"
    )

print()
print("-" * 76)
print(
    "max |MC term Python - MATLAB| = "
    f"{term_diff.max():.3e}"
)
print(
    "mean |MC term difference|     = "
    f"{term_diff.mean():.3e}"
)
print(
    "max |EIG Python - MATLAB|     = "
    f"{eig_diff.max():.3e}"
)

best_matlab = int(np.argmax(eig_matlab_np))
best_python = int(np.argmax(eig_python))

print(f"best MATLAB beam = {best_matlab + 1}")
print(f"best Python beam = {best_python + 1}")
print(f"same argmax      = {best_matlab == best_python}")

tol = 1e-10

if (
    term_diff.max() < tol
    and eig_diff.max() < tol
    and best_matlab == best_python
):
    print()
    print(
        "PASS: the actual production estimator used by choose_beam "
        "matches MATLAB to numerical precision."
    )
else:
    print()
    print("FAIL: discrepancy in the production estimator.")
