import time
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODULES_DIR = PROJECT_ROOT / "modules"

if str(MODULES_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODULES_DIR),
    )

import numpy as np
import torch
from scipy.io import loadmat, savemat

from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.beam_eig.array_model import steering_vector
from modules.beam_eig.codebook import generate_quantized_codebook

from modules.beam_eig.baseline_eig import choose_beam

from modules.beam_eig.posterior import (
    update_posterior,
    marginal_theta,
    marginal_rho,
    compute_rho_mean,
)

from modules.beam_eig.simulator import (
    generate_amplitude_measurement,
    sigma_from_snr,
)


# ============================================================
# Setup
# ============================================================

params = Params()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Load Amelia MATLAB baseline
# ============================================================

data_matlab = loadmat(
    SCRIPT_DIR / "results_amplitude_vector_EIGtheta_only.mat"
)

theta_true_matlab = torch.tensor(
    data_matlab["theta_true_all"],
    dtype=torch.float32,
    device=device,
)

rho_true_matlab = torch.tensor(
    data_matlab["rho_true_all"],
    dtype=torch.float32,
    device=device,
)

theta_grid = torch.tensor(
    data_matlab["theta_grid"].squeeze(),
    dtype=torch.float32,
    device=device,
)

rho_grid = torch.tensor(
    data_matlab["rho_grid"].squeeze(),
    dtype=torch.float32,
    device=device,
)

snr_db_vec = torch.tensor(
    data_matlab["snr_dB_vec"].squeeze(),
    dtype=torch.float32,
    device=device,
)


# ============================================================
# Pilot
# ============================================================

s = generate_pilot_sequence(
    device
)

Ns = s.numel()


# ============================================================
# Dimensions
# ============================================================

n_snr = snr_db_vec.numel()
n_real = theta_true_matlab.shape[1]

L = theta_grid.numel()
Lrho = rho_grid.numel()


# ============================================================
# Array
# ============================================================

a_grid = steering_vector(
    theta_grid,
    params
)


# ============================================================
# Codebook
# ============================================================

eta_grid = generate_quantized_codebook(
    params.K,
    params.B,
    device,
)


# ============================================================
# Python output storage
# ============================================================

theta_est_python = np.zeros(
    (n_snr, n_real)
)

rho_est_python = np.zeros(
    (n_snr, n_real)
)

error_deg_python = np.zeros(
    (n_snr, n_real)
)


# ============================================================
# 9. Main experiment
# ============================================================

start = time.perf_counter()

with torch.no_grad():

    for i_snr, snr_db in enumerate(
        snr_db_vec
    ):

        print(
            f"\n========== "
            f"SNR = {snr_db.item():.0f} dB "
            f"=========="
        )

        for r in range(n_real):

            # --------------------------------------------
            # True parameters
            # --------------------------------------------

            theta_true = theta_true_matlab[i_snr, r]
            rho_true = rho_true_matlab[i_snr, r]

            """ 
            theta_true = (
                theta_true_vec[r]
            )

            rho_true = (
                rho_true_vec[r]
            )

            theta_true_all[
                i_snr, r
            ] = torch.rad2deg(
                theta_true
            ).item()

            rho_true_all[
                i_snr, r
            ] = rho_true.item()

             """
            # --------------------------------------------
            # Noise
            # --------------------------------------------

            sigma = sigma_from_snr(
                s,
                snr_db.item(),
                rho_true,
            )


            # --------------------------------------------
            # Uniform initial posterior
            # --------------------------------------------

            posterior = torch.ones(
                (L, Lrho),
                dtype=torch.float32,
                device=device,
            )

            posterior /= (
                posterior.sum()
            )


            # ============================================
            # Adaptive loop
            # ============================================

            for t in range(
                params.T
            ):

                # ----------------------------------------
                # Current marginal p(theta)
                # ----------------------------------------

                p_theta = (
                    marginal_theta(
                        posterior
                    )
                )


                # ----------------------------------------
                # Current rho mean
                # ----------------------------------------

                rho_mean = (
                    compute_rho_mean(
                        posterior,
                        rho_grid,
                    )
                )


                # ----------------------------------------
                # 1) Select beam
                # ----------------------------------------

                eta_star, eig_values = (
                    choose_beam(
                        eta_grid,
                        a_grid,
                        s,
                        sigma,
                        p_theta,
                        rho_mean,
                        params.N,
                    )
                )


                # ----------------------------------------
                # 2) True observation
                # ----------------------------------------

                amp_vec_true = (
                    generate_amplitude_measurement(
                        theta_true,
                        rho_true,
                        eta_star,
                        s,
                        sigma,
                        params,
                    )
                )


                # ----------------------------------------
                # 3) Posterior update
                # ----------------------------------------

                posterior = (
                    update_posterior(
                        eta_star,
                        a_grid,
                        amp_vec_true,
                        posterior,
                        rho_grid,
                        s,
                        sigma,
                    )
                )


                # ----------------------------------------
                # 4) Current rho estimate
                # ----------------------------------------

                p_rho = marginal_rho(
                    posterior
                )

                rho_hat = rho_grid[
                    torch.argmax(p_rho)
                ]



            # ============================================
            # Final estimates
            # ============================================

            p_theta = marginal_theta(
                posterior
            )

            p_rho = marginal_rho(
                posterior
            )

            theta_hat = theta_grid[
                torch.argmax(p_theta)
            ]

            rho_hat = rho_grid[
                torch.argmax(p_rho)
            ]

            error_deg = torch.abs(
                torch.rad2deg(
                    theta_hat
                    - theta_true
                )
            )
            # --------------------------------------------
            # Save Python final results
            # --------------------------------------------

            theta_est_python[
                i_snr, r
            ] = theta_hat.item()   # radians, comme MATLAB

            rho_est_python[
                i_snr, r
            ] = rho_hat.item()

            error_deg_python[
                i_snr, r
            ] = error_deg.item()


            print(
                f"Realisation {r + 1:2d} | "
                f"Error = {error_deg.item():.3f} deg | "
                f"rho_true = {rho_true.item():.3f} | "
                f"rho_hat = {rho_hat.item():.3f}"
            )



# ============================================================
# 10. Runtime
# ============================================================

if torch.cuda.is_available():
    torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - start
)

print(
    f"\nElapsed time: "
    f"{elapsed:.2f} s"
)

theta_est_matlab = data_matlab[
    "theta_est_all"
]

rho_est_matlab = data_matlab[
    "rho_est_all"
]

error_deg_matlab = data_matlab[
    "error_deg_all"
]


print("\n==========================================")
print("MATLAB vs PYTORCH")
print("==========================================")

for i_snr in range(n_snr):

    snr = snr_db_vec[
        i_snr
    ].item()

    matlab_mean = np.mean(
        error_deg_matlab[
            i_snr, :
        ]
    )

    python_mean = np.mean(
        error_deg_python[
            i_snr, :
        ]
    )

    matlab_median = np.median(
        error_deg_matlab[
            i_snr, :
        ]
    )

    python_median = np.median(
        error_deg_python[
            i_snr, :
        ]
    )

    print(
        f"SNR = {snr:>4.0f} dB | "
        f"MATLAB mean = {matlab_mean:7.3f} deg | "
        f"PyTorch mean = {python_mean:7.3f} deg | "
        f"MATLAB median = {matlab_median:7.3f} deg | "
        f"PyTorch median = {python_median:7.3f} deg"
    )

for i_snr in range(n_snr):

    print(
        f"\nSNR = "
        f"{snr_db_vec[i_snr].item():.0f} dB"
    )

    for r in range(n_real):

        print(
            f"Realisation {r + 1:2d} | "
            f"MATLAB = "
            f"{error_deg_matlab[i_snr, r]:7.3f} deg | "
            f"PyTorch = "
            f"{error_deg_python[i_snr, r]:7.3f} deg"
        )

# ============================================================
# Save PyTorch results
# ============================================================

savemat(
    SCRIPT_DIR / "results_pytorch_baseline.mat",
    {
        # SNR
        "snr_dB_vec":
            snr_db_vec.detach().cpu().numpy(),

        # Python estimates
        "theta_est_python":
            theta_est_python,

        "rho_est_python":
            rho_est_python,

        "error_deg_python":
            error_deg_python,

        # IMPORTANT:
        # exact same ground truth as MATLAB
        "theta_true_all":
            theta_true_matlab.detach().cpu().numpy(),

        "rho_true_all":
            rho_true_matlab.detach().cpu().numpy(),

        # Same parameter grids
        "theta_grid":
            theta_grid.detach().cpu().numpy(),

        "rho_grid":
            rho_grid.detach().cpu().numpy(),
    },
)

print(
    "\nResults saved to "
    str(SCRIPT_DIR / "results_pytorch_baseline.mat")
)        
