"""
Replay and compare MATLAB vs PyTorch posterior trajectories.

The MATLAB debug file must contain:
    s, a, eta_grid_t, sigma_all,
    eta_star_all, amp_history_all, posterior_history_all,
    EIG_values_all, best_idx_all,
    p_theta_before_all, rho_mean_before_all,
    theta_hat_history_all, rho_hat_history_all,
    theta_grid, rho_grid, snr_dB_vec, theta_true_all, rho_true_all.

This script deliberately reuses the MATLAB beam eta_t and MATLAB amplitude
measurement r_t. Therefore, if the PyTorch likelihood/posterior update is the
same as MATLAB's, the posteriors should agree up to numerical precision.

By default the replay uses MATLAB's saved steering matrix `a` and pilot `s`
to isolate the posterior update itself. It also reports whether the local
Python steering-vector implementation agrees with MATLAB's `a`.

Example:
    python compare_matlab_trajectory.py \
        --mat results_amplitude_vector_EIGtheta_only_debug.mat \
        --snr 0 --realization 1

All trajectories:
    python compare_matlab_trajectory.py \
        --mat results_amplitude_vector_EIGtheta_only_debug.mat \
        --all

Optional EIG diagnostic (not expected to be identical because MATLAB and
PyTorch use different Monte-Carlo random draws):
    python compare_matlab_trajectory.py \
        --mat results_amplitude_vector_EIGtheta_only_debug.mat \
        --snr 0 --realization 1 --compare-eig --eig-n 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from modules.beam_eig.params import Params
from modules.beam_eig.array_model import steering_vector
from modules.beam_eig.posterior import (
    update_posterior,
    marginal_theta,
    marginal_rho,
    compute_rho_mean,
)

try:
    from modules.beam_eig.baseline_eig import choose_beam
except ImportError:
    choose_beam = None


def _cell(mat: dict, name: str, i_snr: int, i_real: int) -> np.ndarray:
    """Return one MATLAB cell entry as a numeric ndarray."""
    arr = mat[name]
    value = arr[i_snr, i_real]
    return np.asarray(value)


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).squeeze()


def _find_snr_index(snr_values: np.ndarray, target: float) -> int:
    snr_values = _flat(snr_values).astype(float)
    idx = int(np.argmin(np.abs(snr_values - target)))
    if not np.isclose(snr_values[idx], target):
        raise ValueError(
            f"SNR {target:g} dB absent du fichier. "
            f"Valeurs disponibles: {snr_values.tolist()}"
        )
    return idx


def _phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum wrapped phase distance between two vectors."""
    delta = np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))
    return float(np.max(np.abs(delta)))


def _posterior_metrics(p_py: np.ndarray, p_mat: np.ndarray) -> dict[str, float]:
    diff = p_py - p_mat
    l1 = float(np.sum(np.abs(diff)))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "l1": l1,
        "tv": 0.5 * l1,
    }


def _make_params_for_saved_array(k: int) -> Params:
    """
    MATLAB debug case is a 1D ULA, so Nx=K, Ny=1.
    Only array geometry is used here.
    """
    try:
        return Params(Nx=k, Ny=1)
    except TypeError:
        # Fallback if the local Params class has a more restrictive signature.
        p = Params()
        p.Nx = k
        p.Ny = 1
        return p


def compare_one(
    mat: dict,
    i_snr: int,
    i_real: int,
    device: torch.device,
    compare_eig: bool = False,
    eig_n: int = 1000,
    verbose: bool = True,
) -> dict[str, float]:

    snr_db = float(_flat(mat["snr_dB_vec"])[i_snr])
    theta_grid_np = _flat(mat["theta_grid"]).astype(np.float64)
    rho_grid_np = _flat(mat["rho_grid"]).astype(np.float64)

    s_np = np.asarray(mat["s"]).reshape(-1).astype(np.complex128)
    a_mat_np = np.asarray(mat["a"]).astype(np.complex128)
    eta_grid_np = np.asarray(mat["eta_grid_t"]).astype(np.float64)

    eta_hist_np = _cell(mat, "eta_star_all", i_snr, i_real).astype(np.float64)
    amp_hist_np = _cell(mat, "amp_history_all", i_snr, i_real).astype(np.float64)
    post_hist_np = _cell(
        mat, "posterior_history_all", i_snr, i_real
    ).astype(np.float64)

    eig_values_np = _cell(mat, "EIG_values_all", i_snr, i_real).astype(np.float64)
    best_idx_np = _flat(_cell(mat, "best_idx_all", i_snr, i_real)).astype(int) - 1

    p_theta_before_np = _cell(
        mat, "p_theta_before_all", i_snr, i_real
    ).astype(np.float64)
    rho_mean_before_np = _flat(
        _cell(mat, "rho_mean_before_all", i_snr, i_real)
    ).astype(np.float64)

    theta_hat_hist_np = _flat(
        _cell(mat, "theta_hat_history_all", i_snr, i_real)
    ).astype(np.float64)
    rho_hat_hist_np = _flat(
        _cell(mat, "rho_hat_history_all", i_snr, i_real)
    ).astype(np.float64)

    theta_true = float(mat["theta_true_all"][i_snr, i_real])
    rho_true = float(mat["rho_true_all"][i_snr, i_real])

    if "sigma_all" in mat:
        sigma = float(mat["sigma_all"][i_snr, i_real])
    else:
        es = float(np.mean(np.abs(s_np) ** 2))
        sigma = rho_true * np.sqrt(es / (10.0 ** (snr_db / 10.0)))

    k, r_candidates = eta_grid_np.shape
    t_steps = eta_hist_np.shape[1]
    l = theta_grid_np.size
    lrho = rho_grid_np.size

    # Use MATLAB's own s and a to isolate the posterior-update implementation.
    s = torch.as_tensor(s_np, dtype=torch.complex128, device=device)
    a_mat = torch.as_tensor(a_mat_np, dtype=torch.complex128, device=device)
    theta_grid = torch.as_tensor(theta_grid_np, dtype=torch.float64, device=device)
    rho_grid = torch.as_tensor(rho_grid_np, dtype=torch.float64, device=device)

    posterior = torch.full(
        (l, lrho),
        1.0 / (l * lrho),
        dtype=torch.float64,
        device=device,
    )

    # Independent check of the steering implementation.
    params = _make_params_for_saved_array(k)
    try:
        a_python = steering_vector(theta_grid, params)
        steering_max_abs = float(
            torch.max(torch.abs(a_python.to(torch.complex128) - a_mat)).item()
        )
    except Exception as exc:
        steering_max_abs = float("nan")
        if verbose:
            print(f"[warning] steering_vector check impossible: {exc}")

    if verbose:
        print("\n" + "=" * 78)
        print(
            f"SNR={snr_db:g} dB | réalisation={i_real + 1} | "
            f"K={k} | R={r_candidates} | T={t_steps}"
        )
        print(
            f"theta_true={np.rad2deg(theta_true):.6f} deg | "
            f"rho_true={rho_true:.6f} | sigma={sigma:.8g}"
        )
        print(f"max |a_python - a_MATLAB| = {steering_max_abs:.3e}")
        print("-" * 78)
        print(
            " t | beam ok | post max abs |    post L1 |      TV | "
            "theta MAP py/mat [deg] | rho MAP py/mat"
        )
        print("-" * 78)

    worst_max_abs = 0.0
    worst_l1 = 0.0
    worst_ptheta_l1 = 0.0
    worst_prho_l1 = 0.0
    all_beams_ok = True

    for t in range(t_steps):
        eta_np = eta_hist_np[:, t]
        amp_np = amp_hist_np[:, t]

        # Check that MATLAB's saved selected beam is exactly the candidate argmax.
        idx = int(best_idx_np[t])
        beam_argmax_idx = int(np.argmax(eig_values_np[:, t]))

        beam_saved_vs_idx = _phase_distance(eta_np, eta_grid_np[:, idx])
        beam_saved_vs_argmax = _phase_distance(
            eta_np, eta_grid_np[:, beam_argmax_idx]
        )
        beam_ok = (
            idx == beam_argmax_idx
            and beam_saved_vs_idx < 1e-10
            and beam_saved_vs_argmax < 1e-10
        )
        all_beams_ok = all_beams_ok and beam_ok

        # Compare pre-selection state reconstructed from the PyTorch replay.
        ptheta_py_before = marginal_theta(posterior)
        rho_mean_py_before = compute_rho_mean(posterior, rho_grid)

        ptheta_before_l1 = float(
            torch.sum(
                torch.abs(
                    ptheta_py_before
                    - torch.as_tensor(
                        p_theta_before_np[:, t],
                        dtype=torch.float64,
                        device=device,
                    )
                )
            ).item()
        )
        rho_mean_before_err = abs(
            float(rho_mean_py_before.item()) - float(rho_mean_before_np[t])
        )

        eta = torch.as_tensor(
            eta_np[:, None], dtype=torch.float64, device=device
        )
        amp = torch.as_tensor(amp_np, dtype=torch.float64, device=device)

        posterior = update_posterior(
            eta,
            a_mat,
            amp,
            posterior,
            rho_grid,
            s,
            snr_db,
            torch.as_tensor(sigma, dtype=torch.float64, device=device),
            mode="sigma_fixed",
        )

        p_py = posterior.detach().cpu().numpy()
        p_mat = post_hist_np[:, :, t]

        metrics = _posterior_metrics(p_py, p_mat)
        worst_max_abs = max(worst_max_abs, metrics["max_abs"])
        worst_l1 = max(worst_l1, metrics["l1"])

        ptheta_py = marginal_theta(posterior)
        prho_py = marginal_rho(posterior)

        ptheta_mat = np.sum(p_mat, axis=1)
        ptheta_mat /= np.sum(ptheta_mat)

        prho_mat = np.sum(p_mat, axis=0)
        prho_mat /= np.sum(prho_mat)

        ptheta_l1 = float(
            np.sum(np.abs(ptheta_py.detach().cpu().numpy() - ptheta_mat))
        )
        prho_l1 = float(
            np.sum(np.abs(prho_py.detach().cpu().numpy() - prho_mat))
        )

        worst_ptheta_l1 = max(worst_ptheta_l1, ptheta_l1)
        worst_prho_l1 = max(worst_prho_l1, prho_l1)

        theta_idx_py = int(torch.argmax(ptheta_py).item())
        rho_idx_py = int(torch.argmax(prho_py).item())

        theta_py_deg = float(np.rad2deg(theta_grid_np[theta_idx_py]))
        rho_py = float(rho_grid_np[rho_idx_py])

        theta_mat_deg = float(np.rad2deg(theta_hat_hist_np[t]))
        rho_mat = float(rho_hat_hist_np[t])

        if verbose:
            print(
                f"{t+1:2d} | {'yes':>7s} | "
                f"{metrics['max_abs']:12.3e} | "
                f"{metrics['l1']:10.3e} | "
                f"{metrics['tv']:7.3e} | "
                f"{theta_py_deg:8.3f}/{theta_mat_deg:8.3f} | "
                f"{rho_py:6.3f}/{rho_mat:6.3f}"
            )
            print(
                f"    pre-state: L1 p(theta)={ptheta_before_l1:.3e}, "
                f"|rho_mean_py-rho_mean_mat|={rho_mean_before_err:.3e}, "
                f"MATLAB EIG max={eig_values_np[beam_argmax_idx, t]:.6f}"
            )

        # Optional EIG comparison. Not exact because random Monte Carlo draws differ.
        if compare_eig:
            if choose_beam is None:
                raise RuntimeError(
                    "--compare-eig demandé, mais choose_beam n'a pas pu être importé."
                )

            # Beam selection at step t used the *pre-update* posterior.
            if t == 0:
                pre_posterior = torch.full(
                    (l, lrho),
                    1.0 / (l * lrho),
                    dtype=torch.float64,
                    device=device,
                )
            else:
                pre_posterior = torch.as_tensor(
                    post_hist_np[:, :, t - 1],
                    dtype=torch.float64,
                    device=device,
                )

            p_theta_pre = marginal_theta(pre_posterior)
            eta_grid_torch = torch.as_tensor(
                eta_grid_np, dtype=torch.float64, device=device
            )

            with torch.no_grad():
                _, eig_py = choose_beam(
                    eta_grid_torch,
                    a_mat,
                    s,
                    snr_db,
                    torch.as_tensor(sigma, dtype=torch.float64, device=device),
                    p_theta_pre,
                    pre_posterior,
                    rho_grid,
                    mode="mean",
                    N=eig_n,
                )

            eig_py_np = eig_py.detach().cpu().numpy().reshape(-1)
            eig_mat_np = eig_values_np[:, t].reshape(-1)

            corr = float(np.corrcoef(eig_py_np, eig_mat_np)[0, 1])
            py_argmax = int(np.argmax(eig_py_np))
            matlab_choice_rank_py = int(
                np.where(np.argsort(-eig_py_np) == beam_argmax_idx)[0][0] + 1
            )

            if verbose:
                print(
                    f"    EIG MC diagnostic N={eig_n}: "
                    f"corr(Matlab,Python)={corr:.4f}, "
                    f"argmax_mat={beam_argmax_idx+1}, "
                    f"argmax_py={py_argmax+1}, "
                    f"rang du beam MATLAB selon Python={matlab_choice_rank_py}/{r_candidates}"
                )

    ptheta_final = marginal_theta(posterior)
    prho_final = marginal_rho(posterior)

    theta_final_py = float(
        theta_grid_np[int(torch.argmax(ptheta_final).item())]
    )
    rho_final_py = float(
        rho_grid_np[int(torch.argmax(prho_final).item())]
    )

    theta_final_mat = float(mat["theta_est_all"][i_snr, i_real])
    rho_final_mat = float(mat["rho_est_all"][i_snr, i_real])

    error_py = abs(np.rad2deg(theta_final_py - theta_true))
    error_mat = abs(np.rad2deg(theta_final_mat - theta_true))

    if verbose:
        print("-" * 78)
        print(
            f"Final theta error: PyTorch={error_py:.6f} deg | "
            f"MATLAB={error_mat:.6f} deg"
        )
        print(
            f"Final rho: PyTorch={rho_final_py:.6f} | "
            f"MATLAB={rho_final_mat:.6f}"
        )
        print(
            f"Worst trajectory discrepancies: "
            f"max_abs={worst_max_abs:.3e}, "
            f"L1={worst_l1:.3e}, "
            f"L1 p(theta)={worst_ptheta_l1:.3e}, "
            f"L1 p(rho)={worst_prho_l1:.3e}"
        )
        print(f"All saved MATLAB beam/argmax checks passed: {all_beams_ok}")

    return {
        "worst_max_abs": worst_max_abs,
        "worst_l1": worst_l1,
        "worst_ptheta_l1": worst_ptheta_l1,
        "worst_prho_l1": worst_prho_l1,
        "error_py": float(error_py),
        "error_mat": float(error_mat),
        "steering_max_abs": steering_max_abs,
        "all_beams_ok": float(all_beams_ok),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("results_amplitude_vector_EIGtheta_only_debug.mat"),
        help="Fichier .mat produit par baseline_amplitude_debug.m",
    )
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument(
        "--realization",
        type=int,
        default=1,
        help="Index humain (1,2,...) et non index Python.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rejoue toutes les trajectoires du fichier.",
    )
    parser.add_argument(
        "--compare-eig",
        action="store_true",
        help=(
            "Recalcule aussi les EIG Python. C'est un diagnostic statistique, "
            "pas une égalité sample-by-sample."
        ),
    )
    parser.add_argument(
        "--eig-n",
        type=int,
        default=1000,
        help="NMC utilisé uniquement avec --compare-eig.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force le CPU.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print("Device:", device)

    mat = loadmat(args.mat)

    required = [
        "s",
        "a",
        "eta_grid_t",
        "sigma_all",
        "theta_grid",
        "rho_grid",
        "snr_dB_vec",
        "theta_true_all",
        "rho_true_all",
        "theta_est_all",
        "rho_est_all",
        "eta_star_all",
        "best_idx_all",
        "amp_history_all",
        "posterior_history_all",
        "EIG_values_all",
        "p_theta_before_all",
        "rho_mean_before_all",
        "theta_hat_history_all",
        "rho_hat_history_all",
    ]
    missing = [name for name in required if name not in mat]
    if missing:
        raise KeyError(
            "Le .mat ne contient pas les champs de debug suivants: "
            + ", ".join(missing)
            + "\nRelance d'abord baseline_amplitude_debug.m."
        )

    if args.all:
        snrs = _flat(mat["snr_dB_vec"]).astype(float)
        n_snr = snrs.size
        n_real = mat["theta_true_all"].shape[1]

        rows = []
        for i_snr in range(n_snr):
            for i_real in range(n_real):
                result = compare_one(
                    mat,
                    i_snr,
                    i_real,
                    device=device,
                    compare_eig=False,  # Avoid huge accidental runs.
                    verbose=False,
                )
                rows.append((i_snr, i_real, result))

        worst = max(rows, key=lambda x: x[2]["worst_max_abs"])

        print("\n" + "=" * 78)
        print("GLOBAL REPLAY SUMMARY")
        print("=" * 78)
        print(f"Trajectoires rejouées : {len(rows)}")
        print(
            "Worst posterior max_abs : "
            f"{worst[2]['worst_max_abs']:.3e} "
            f"(SNR={snrs[worst[0]]:g} dB, réalisation={worst[1]+1})"
        )
        print(
            "Worst posterior L1      : "
            f"{max(x[2]['worst_l1'] for x in rows):.3e}"
        )
        print(
            "Worst marginal theta L1 : "
            f"{max(x[2]['worst_ptheta_l1'] for x in rows):.3e}"
        )
        print(
            "Worst marginal rho L1   : "
            f"{max(x[2]['worst_prho_l1'] for x in rows):.3e}"
        )
        print(
            "Worst steering max_abs  : "
            f"{np.nanmax([x[2]['steering_max_abs'] for x in rows]):.3e}"
        )
        print(
            "Beam/argmax checks       : "
            f"{sum(bool(x[2]['all_beams_ok']) for x in rows)}/{len(rows)} passed"
        )
        print(
            "Final theta errors agree (max abs difference): "
            f"{max(abs(x[2]['error_py'] - x[2]['error_mat']) for x in rows):.3e} deg"
        )

    else:
        i_snr = _find_snr_index(mat["snr_dB_vec"], args.snr)
        i_real = args.realization - 1

        n_real = mat["theta_true_all"].shape[1]
        if not (0 <= i_real < n_real):
            raise ValueError(
                f"Réalisation invalide: {args.realization}. "
                f"Le fichier contient 1..{n_real}."
            )

        compare_one(
            mat,
            i_snr,
            i_real,
            device=device,
            compare_eig=args.compare_eig,
            eig_n=args.eig_n,
            verbose=True,
        )


if __name__ == "__main__":
    main()
