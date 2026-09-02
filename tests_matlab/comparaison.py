from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# 1. Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MATLAB_FILE = SCRIPT_DIR / "results_amplitude_vector_EIGtheta_only.mat"

PYTHON_FILE = SCRIPT_DIR / "results_pytorch_baseline.mat"

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


# Temps mesurés
MATLAB_RUNTIME_S = 372.222532
PYTHON_RUNTIME_S = 54.65


# ============================================================
# 2. Helper functions
# ============================================================

def load_required(data, *names):
    """
    Retourne la première variable trouvée parmi names.
    L'ordre est donc important.
    """

    for name in names:

        if name in data:

            print(
                f"Using variable: {name}"
            )

            return np.asarray(
                data[name]
            )

    available = [
        key
        for key in data.keys()
        if not key.startswith("__")
    ]

    raise KeyError(
        f"Aucune des variables {names} "
        f"n'a été trouvée.\n"
        f"Variables disponibles:\n"
        f"{available}"
    )


def angle_array_to_deg(array):
    """
    Détecte automatiquement si les angles
    semblent être en radians ou degrés.
    """

    x = np.asarray(
        array,
        dtype=float,
    )

    finite = x[
        np.isfinite(x)
    ]

    if finite.size == 0:
        return x

    # Dans notre problème:
    # theta est entre 30 et 150 deg
    #
    # donc s'il est < 2*pi,
    # il est très probablement en radians.

    if (
        np.max(
            np.abs(finite)
        )
        <= 2 * np.pi + 1e-6
    ):

        return np.rad2deg(
            x
        )

    return x


def separator(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 3. Check files
# ============================================================

if not MATLAB_FILE.exists():

    raise FileNotFoundError(
        f"MATLAB file not found:\n"
        f"{MATLAB_FILE.resolve()}"
    )


if not PYTHON_FILE.exists():

    raise FileNotFoundError(
        f"PyTorch file not found:\n"
        f"{PYTHON_FILE.resolve()}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Load MAT files
# ============================================================

separator(
    "LOADING FILES"
)

matlab = loadmat(
    MATLAB_FILE
)

python = loadmat(
    PYTHON_FILE
)

print(
    "MATLAB file:",
    MATLAB_FILE
)

print(
    "PyTorch file:",
    PYTHON_FILE
)


# ============================================================
# 5. SNR vectors
# ============================================================

snr_matlab = np.asarray(
    load_required(
        matlab,
        "snr_dB_vec",
    ),
    dtype=float,
).squeeze()


snr_python = np.asarray(
    load_required(
        python,
        "snr_dB_vec",
    ),
    dtype=float,
).squeeze()


assert snr_matlab.shape == snr_python.shape, (
    "MATLAB and PyTorch SNR vectors "
    "do not have the same shape."
)


assert np.allclose(
    snr_matlab,
    snr_python,
), (
    "MATLAB and PyTorch do not use "
    "the same SNR values."
)


snr = snr_matlab


# ============================================================
# 6. Load TRUE parameters
# ============================================================

separator(
    "TRUE PARAMETER VALIDATION"
)


theta_true_matlab = angle_array_to_deg(
    load_required(
        matlab,
        "theta_true_all",
    )
)


rho_true_matlab = np.asarray(
    load_required(
        matlab,
        "rho_true_all",
    ),
    dtype=float,
)


theta_true_python = angle_array_to_deg(
    load_required(
        python,
        "theta_true_all",
    )
)


rho_true_python = np.asarray(
    load_required(
        python,
        "rho_true_all",
    ),
    dtype=float,
)


print()
print(
    "theta_true MATLAB shape:",
    theta_true_matlab.shape
)

print(
    "theta_true PyTorch shape:",
    theta_true_python.shape
)

print(
    "rho_true MATLAB shape:",
    rho_true_matlab.shape
)

print(
    "rho_true PyTorch shape:",
    rho_true_python.shape
)


# ============================================================
# 7. STRICT truth validation
# ============================================================

assert (
    theta_true_matlab.shape
    == theta_true_python.shape
), (
    "theta_true shapes are different."
)


assert (
    rho_true_matlab.shape
    == rho_true_python.shape
), (
    "rho_true shapes are different."
)


theta_true_max_diff = np.max(
    np.abs(
        theta_true_matlab
        - theta_true_python
    )
)


rho_true_max_diff = np.max(
    np.abs(
        rho_true_matlab
        - rho_true_python
    )
)


print()
print(
    "Max theta_true difference:",
    f"{theta_true_max_diff:.6e} deg"
)

print(
    "Max rho_true difference:",
    f"{rho_true_max_diff:.6e}"
)


assert np.allclose(
    theta_true_matlab,
    theta_true_python,
    atol=1e-5,
    rtol=0.0,
), (
    "ERROR: MATLAB and PyTorch "
    "did not use the same theta_true."
)


assert np.allclose(
    rho_true_matlab,
    rho_true_python,
    atol=1e-6,
    rtol=0.0,
), (
    "ERROR: MATLAB and PyTorch "
    "did not use the same rho_true."
)


print()
print(
    "✓ Same theta_true used in MATLAB and PyTorch"
)

print(
    "✓ Same rho_true used in MATLAB and PyTorch"
)


# ============================================================
# 8. Debug: print true rho
# ============================================================

print()
print(
    "rho_true MATLAB, first SNR:"
)

print(
    rho_true_matlab[0]
)

print()
print(
    "rho_true PyTorch, first SNR:"
)

print(
    rho_true_python[0]
)


# ============================================================
# 9. Load errors
# ============================================================

separator(
    "LOADING ESTIMATION RESULTS"
)


error_matlab = np.asarray(
    load_required(
        matlab,
        "error_deg_all",
    ),
    dtype=float,
)


# IMPORTANT:
# Python-specific name has priority.
error_python = np.asarray(
    load_required(
        python,
        "error_deg_python",
        "error_deg_all",
    ),
    dtype=float,
)


assert (
    error_matlab.shape
    == error_python.shape
), (
    "MATLAB and PyTorch error arrays "
    "do not have the same shape."
)


n_snr, n_real = (
    error_matlab.shape
)


# ============================================================
# 10. Load theta estimates
# ============================================================

theta_est_matlab = angle_array_to_deg(
    load_required(
        matlab,
        "theta_est_all",
    )
)


# IMPORTANT:
# Python-specific name first.
theta_est_python = angle_array_to_deg(
    load_required(
        python,
        "theta_est_python",
        "theta_est_all",
    )
)


# ============================================================
# 11. Load rho estimates
# ============================================================

rho_est_matlab = np.asarray(
    load_required(
        matlab,
        "rho_est_all",
    ),
    dtype=float,
)


# IMPORTANT:
# Python-specific name first.
rho_est_python = np.asarray(
    load_required(
        python,
        "rho_est_python",
        "rho_est_all",
    ),
    dtype=float,
)


# ============================================================
# 12. Debug loaded rho estimates
# ============================================================

print()
print(
    "rho_hat MATLAB at 10 dB:"
)

print(
    rho_est_matlab[-1]
)


print()
print(
    "rho_hat PyTorch at 10 dB:"
)

print(
    rho_est_python[-1]
)


print()
print(
    "rho_true at 10 dB:"
)

print(
    rho_true_matlab[-1]
)


# ============================================================
# 13. Angular error statistics
# ============================================================

mean_matlab = np.mean(
    error_matlab,
    axis=1,
)

mean_python = np.mean(
    error_python,
    axis=1,
)


median_matlab = np.median(
    error_matlab,
    axis=1,
)

median_python = np.median(
    error_python,
    axis=1,
)


std_matlab = np.std(
    error_matlab,
    axis=1,
)

std_python = np.std(
    error_python,
    axis=1,
)


# ============================================================
# 14. Print angular comparison table
# ============================================================

separator(
    "ANGULAR ERROR: MATLAB VS PYTORCH"
)


header = (
    f"{'SNR':>7} | "
    f"{'MAT mean':>10} | "
    f"{'Torch mean':>10} | "
    f"{'MAT med':>10} | "
    f"{'Torch med':>10} | "
    f"{'MAT std':>10} | "
    f"{'Torch std':>10}"
)

print(
    header
)

print(
    "-" * len(header)
)


for i in range(
    n_snr
):

    print(
        f"{snr[i]:7.0f} | "
        f"{mean_matlab[i]:10.3f} | "
        f"{mean_python[i]:10.3f} | "
        f"{median_matlab[i]:10.3f} | "
        f"{median_python[i]:10.3f} | "
        f"{std_matlab[i]:10.3f} | "
        f"{std_python[i]:10.3f}"
    )


# ============================================================
# 15. Rho MAE
# ============================================================

#
# IMPORTANT:
#
# MATLAB estimate compared to MATLAB truth.
#
# PyTorch estimate compared to PyTorch truth.
#
# We already asserted that both truths are identical.
#

rho_error_matlab = np.abs(
    rho_est_matlab
    - rho_true_matlab
)


rho_error_python = np.abs(
    rho_est_python
    - rho_true_python
)


rho_mae_matlab = np.mean(
    rho_error_matlab,
    axis=1,
)


rho_mae_python = np.mean(
    rho_error_python,
    axis=1,
)


rho_median_matlab = np.median(
    rho_error_matlab,
    axis=1,
)


rho_median_python = np.median(
    rho_error_python,
    axis=1,
)


# ============================================================
# 16. Print rho comparison
# ============================================================

separator(
    "RHO ERROR: MATLAB VS PYTORCH"
)


for i in range(
    n_snr
):

    print(
        f"SNR = {snr[i]:>4.0f} dB | "
        f"MATLAB MAE = "
        f"{rho_mae_matlab[i]:.4f} | "
        f"PyTorch MAE = "
        f"{rho_mae_python[i]:.4f} | "
        f"MATLAB median = "
        f"{rho_median_matlab[i]:.4f} | "
        f"PyTorch median = "
        f"{rho_median_python[i]:.4f}"
    )


# ============================================================
# 17. Realisation-by-realisation comparison
# ============================================================

separator(
    "REALISATION-BY-REALISATION ANGULAR ERROR"
)


for i in range(
    n_snr
):

    print()
    print(
        f"SNR = {snr[i]:.0f} dB"
    )

    for r in range(
        n_real
    ):

        print(
            f"Realisation {r + 1:2d} | "
            f"MATLAB = "
            f"{error_matlab[i, r]:7.3f} deg | "
            f"PyTorch = "
            f"{error_python[i, r]:7.3f} deg"
        )


# ============================================================
# 18. Plot angular mean
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.errorbar(
    snr,
    mean_matlab,
    yerr=std_matlab,
    marker="o",
    capsize=4,
    label="MATLAB",
)


plt.errorbar(
    snr,
    mean_python,
    yerr=std_python,
    marker="s",
    capsize=4,
    label="PyTorch",
)


plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Mean angular error (deg)"
)

plt.title(
    "Mean angular error: MATLAB vs PyTorch"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "mean_error_vs_snr.png",
    dpi=180,
)


plt.close()


# ============================================================
# 19. Plot angular median
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    snr,
    median_matlab,
    marker="o",
    label="MATLAB",
)


plt.plot(
    snr,
    median_python,
    marker="s",
    label="PyTorch",
)


plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Median angular error (deg)"
)

plt.title(
    "Median angular error: MATLAB vs PyTorch"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "median_error_vs_snr.png",
    dpi=180,
)


plt.close()


# ============================================================
# 20. Plot rho MAE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    snr,
    rho_mae_matlab,
    marker="o",
    label="MATLAB",
)


plt.plot(
    snr,
    rho_mae_python,
    marker="s",
    label="PyTorch",
)


plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Mean absolute rho error"
)

plt.title(
    "Rho estimation error: MATLAB vs PyTorch"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "rho_mae_vs_snr.png",
    dpi=180,
)


plt.close()


# ============================================================
# 21. Plot rho median error
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    snr,
    rho_median_matlab,
    marker="o",
    label="MATLAB",
)


plt.plot(
    snr,
    rho_median_python,
    marker="s",
    label="PyTorch",
)


plt.xlabel(
    "SNR (dB)"
)

plt.ylabel(
    "Median absolute rho error"
)

plt.title(
    "Median rho error: MATLAB vs PyTorch"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "rho_median_vs_snr.png",
    dpi=180,
)


plt.close()


# ============================================================
# 22. Scatter: MATLAB error vs PyTorch error
# ============================================================

plt.figure(
    figsize=(6, 6)
)


x = error_matlab.flatten()
y = error_python.flatten()


plt.scatter(
    x,
    y,
)


limit = max(
    np.max(x),
    np.max(y),
)


plt.plot(
    [0, limit],
    [0, limit],
    linestyle="--",
    label="y = x",
)


plt.xlabel(
    "MATLAB angular error (deg)"
)

plt.ylabel(
    "PyTorch angular error (deg)"
)

plt.title(
    "Realisation-level angular error"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "angular_error_scatter.png",
    dpi=180,
)


plt.close()


# ============================================================
# 23. Runtime comparison
# ============================================================

separator(
    "RUNTIME"
)


speedup = (
    MATLAB_RUNTIME_S
    / PYTHON_RUNTIME_S
)


print(
    f"MATLAB runtime  = "
    f"{MATLAB_RUNTIME_S:.2f} s"
)

print(
    f"PyTorch runtime = "
    f"{PYTHON_RUNTIME_S:.2f} s"
)

print(
    f"Speed-up        = "
    f"{speedup:.2f}x"
)


plt.figure(
    figsize=(6, 5)
)


labels = [
    "MATLAB",
    "PyTorch GPU",
]


runtimes = [
    MATLAB_RUNTIME_S,
    PYTHON_RUNTIME_S,
]


bars = plt.bar(
    labels,
    runtimes,
)


plt.ylabel(
    "Runtime (s)"
)

plt.title(
    "Full baseline runtime"
)

plt.grid(
    True,
    axis="y",
)


for bar, value in zip(
    bars,
    runtimes,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.1f} s",
        ha="center",
        va="bottom",
    )


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "runtime_comparison.png",
    dpi=180,
)


plt.close()


# ============================================================
# 24. CSV summary
# ============================================================

summary = np.column_stack(
    (
        snr,
        mean_matlab,
        mean_python,
        median_matlab,
        median_python,
        rho_mae_matlab,
        rho_mae_python,
        rho_median_matlab,
        rho_median_python,
    )
)


np.savetxt(
    OUTPUT_DIR
    / "comparison_summary.csv",
    summary,
    delimiter=",",
    header=(
        "snr_db,"
        "matlab_mean_theta_error_deg,"
        "pytorch_mean_theta_error_deg,"
        "matlab_median_theta_error_deg,"
        "pytorch_median_theta_error_deg,"
        "matlab_rho_mae,"
        "pytorch_rho_mae,"
        "matlab_rho_median_error,"
        "pytorch_rho_median_error"
    ),
    comments="",
)


# ============================================================
# 25. Done
# ============================================================

separator(
    "COMPARISON COMPLETE"
)


print(
    "All figures saved to:"
)

print(
    OUTPUT_DIR.resolve()
)


print()
print(
    "Generated files:"
)


for file in sorted(
    OUTPUT_DIR.glob("*")
):

    print(
        " -",
        file.name
    )
