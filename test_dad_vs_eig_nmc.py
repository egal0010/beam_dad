"""Compare les méthodes choisies sur les mêmes réalisations physiques.

Cette version permet de choisir :
  - EIG rho-marginalisé ou rho_mean plug-in ;
  - recherche EIG exhaustive quantifiée ;
  - recherche EIG aléatoire quantifiée ou continue.

Le posterior reste ici en mode ``sigma_snr``.
"""

import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch

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
from modules.beam_eig.simulator import generate_amplitude_measurement, sigma_from_snr
from modules.dad.policy import DADPolicy
from modules.dad.contrastive import contrastive_bound, make_log_likelihood_fn


# ============================================================
# Configuration
# ============================================================

SNR_DB = 0
N_REAL = 100
L_EVAL = 3000       # Contrastifs pour l'évaluation finale.
N_RECOMPUTE = 5000  # Recalcul EIG du beam choisi pour toutes les méthodes.
SEED = 42

CHECKPOINT_PATH = Path(
    "model_checkpoints/checkpoints_nx8/checkpoints_T3/dad_T10_best.pt"
)
OUTPUT_PATH = Path("comparison_nmc_vs_dad.pt")

NAMES = {
    "baseline": "EIG / NMC",
    "dad": "DAD",
    "random": "RANDOM CONTINUOUS",
}

# Le modèle d'inférence utilisé dans ce script.
POSTERIOR_MODE = "sigma_snr"


# ============================================================
# Interactive configuration
# ============================================================


def choose_methods():
    print("Méthodes à tester : 1 = EIG / NMC, 2 = DAD (réseau), 3 = Random")
    print("Combinaisons possibles : 1, 2, 3, 1 2, 1 3, 2 3 ou 1 2 3.")
    print("Chaque méthode : erreurs finales, sum EIG et g_L. Seul le cas 1 sélectionne par EIG.")

    while True:
        choices = set(re.split(r"[\s,;+]+", input("Ton choix : ").strip()))
        if choices and choices <= {"1", "2", "3"}:
            return choices
        print("Entre au moins un numéro parmi 1, 2 et 3 (exemple : 2 3).")


def choose_eig_mode():
    print("\nEstimateur EIG pour la méthode NMC :")
    print("  1 = rho marginalisé : p(r|theta,h) = sum_rho p(rho|theta,h)p(r|theta,rho)")
    print("  2 = rho_mean plug-in : rho -> E[rho|h]")

    while True:
        choice = input("Mode EIG : ").strip()
        if choice == "1":
            return "marginal"
        if choice == "2":
            return "mean"
        print("Choisis 1 ou 2.")


def choose_eig_candidates(params):
    """Choisit le type de recherche et, si nécessaire, le nombre de beams."""

    print("\nRecherche des beams pour EIG / NMC :")
    print(f"  1 = codebook exhaustif quantifié sur {params.B} bits")
    print("  2 = sous-ensemble aléatoire de beams")

    while True:
        search_choice = input("Sélection des beams : ").strip()
        if search_choice in {"1", "2"}:
            break
        print("Choisis 1 ou 2.")

    if search_choice == "1":
        return SimpleNamespace(
            search="exhaustive",
            beam_space="quantized",
            n_candidates=None,
        )

    print("\nEspace des beams échantillonnés :")
    print(f"  1 = quantifié sur {params.B} bits")
    print("  2 = continu, phases uniformes dans [0, 2pi)")

    while True:
        space_choice = input("Type de beams aléatoires : ").strip()
        if space_choice == "1":
            beam_space = "quantized"
            break
        if space_choice == "2":
            beam_space = "continuous"
            break
        print("Choisis 1 ou 2.")

    if beam_space == "quantized":
        while True:
            try:
                count = int(
                    input(
                        f"Nombre de beams aléatoires quantifiés "
                        f"(1 à {params.R - 1}) : "
                    )
                )
                if 1 <= count < params.R:
                    break
            except ValueError:
                pass
            print(f"Entre un entier entre 1 et {params.R - 1}.")
    else:
        while True:
            try:
                count = int(input("Nombre de beams aléatoires continus (> 0) : "))
                if count >= 1:
                    break
            except ValueError:
                pass
            print("Entre un entier strictement positif.")

    return SimpleNamespace(
        search="random",
        beam_space=beam_space,
        n_candidates=count,
    )


# ============================================================
# Helpers
# ============================================================


def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_test_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    params = Params()
    s = generate_pilot_sequence(device=device, sequence_type="PSS")

    theta_grid = torch.deg2rad(
        torch.linspace(30.0, 150.0, 121, device=device)
    )
    rho_grid = torch.linspace(0.05, 1.0, 50, device=device)

    theta_min = math.radians(30.0)
    theta_range = math.radians(120.0)

    # Générés une seule fois, partagés par toutes les méthodes.
    theta_true = theta_min + theta_range * torch.rand(N_REAL, device=device)
    rho_true = 0.05 + 0.95 * torch.rand(N_REAL, device=device)

    noise_real = torch.randn(
        N_REAL, params.T, s.numel(), device=device
    )
    noise_imag = torch.randn_like(noise_real)

    theta_contrast = theta_min + theta_range * torch.rand(
        N_REAL, L_EVAL, device=device
    )

    print(
        f"Device: {device} | Antennes: {params.K} | "
        f"T: {params.T} | Ns: {s.numel()}"
    )
    print(f"Posterior mode: {POSTERIOR_MODE}")

    return SimpleNamespace(
        device=device,
        params=params,
        s=s,
        theta_grid=theta_grid,
        rho_grid=rho_grid,
        a_grid=steering_vector(theta_grid, params),
        theta_true=theta_true,
        rho_true=rho_true,
        noise_real=noise_real,
        noise_imag=noise_imag,
        theta_contrast=theta_contrast,
        log_likelihood_fn=make_log_likelihood_fn(
            rho_grid=rho_grid,
            s=s,
            snr_db=SNR_DB,
            params=params,
        ),
        log_p_rho_prior=torch.full(
            (rho_grid.numel(),),
            -math.log(rho_grid.numel()),
            device=device,
        ),
    )


def evaluate_g_L(ctx, realization, eta_history, r_history):
    theta_candidates = torch.cat(
        [
            ctx.theta_true[realization].reshape(1, 1),
            ctx.theta_contrast[realization].reshape(1, -1),
        ],
        dim=1,
    )

    bound, _ = contrastive_bound(
        theta_candidates=theta_candidates,
        eta_history=eta_history,
        r_history=r_history,
        log_likelihood_fn=ctx.log_likelihood_fn,
        log_p_rho_prior=ctx.log_p_rho_prior,
    )

    return bound.item()


def sigma_for_mean_eig(ctx, posterior):
    """Sigma cohérent avec le plug-in rho_mean sous le modèle sigma_snr.

    Le posterior est en mode sigma_snr, donc pour l'approximation rho_mean on
    remplace rho par E[rho|h] aussi dans sigma(rho), au lieu d'utiliser
    sigma(rho_true), qui donnerait une information oracle à la méthode EIG.
    """

    rho_mean = compute_rho_mean(posterior, ctx.rho_grid)
    return sigma_from_snr(
        s=ctx.s,
        snr_db=SNR_DB,
        rho_true=rho_mean,
    )


def estimate_eig_for_selected_beam(
    ctx,
    eta_vec,
    posterior,
    sigma_scenario,
    eig_mode,
    N,
):
    """Évalue un beam déjà choisi avec exactement le mode EIG sélectionné."""

    p_theta = marginal_theta(posterior)

    if eig_mode == "mean":
        sigma_eig = sigma_for_mean_eig(ctx, posterior)
    else:
        # En mode marginal, estimate_eig_for_eta_marginal recalcule sigma(rho_j)
        # pour chaque candidat rho_j ; ce sigma n'est donc pas utilisé.
        sigma_eig = sigma_scenario

    _, eig_values = choose_beam(
        eta_grid=eta_vec[:, None],
        a_grid=ctx.a_grid,
        s=ctx.s,
        snr_db=SNR_DB,
        sigma=sigma_eig,
        p_theta=p_theta,
        posterior=posterior,
        rho_grid=ctx.rho_grid,
        mode=eig_mode,
        N=N,
    )

    return eig_values[0]


# ============================================================
# Common evaluation loop
# ============================================================


@torch.inference_mode()
def evaluate_method(ctx, name, select_beam, *, eig_mode, nmc=False):
    """Boucle commune : décision, observation, posterior et scores finaux."""

    print(f"\n{'=' * 60}\n{NAMES[name]}\n{'=' * 60}")

    # Reproductibilité indépendante de la combinaison de méthodes choisie.
    torch.manual_seed(SEED + {"baseline": 1, "dad": 2, "random": 3}[name])

    results = {
        key: []
        for key in (
            "theta_errors",
            "rho_errors",
            "g_L",
            "decision_times",
            "eig_sums",
            "eig_per_step",
        )
    }
    results["eig_sums_recomputed"] = []

    for r in range(ctx.theta_true.numel()):
        posterior = torch.ones(
            ctx.theta_grid.numel(),
            ctx.rho_grid.numel(),
            device=ctx.device,
        )
        posterior /= posterior.sum()

        eta_history = torch.empty(
            1, 0, ctx.params.K, device=ctx.device
        )
        r_history = torch.empty(
            1, 0, ctx.s.numel(), device=ctx.device
        )

        # Ce sigma sert à générer la réalisation physique.
        sigma_scenario = sigma_from_snr(
            s=ctx.s,
            snr_db=SNR_DB,
            rho_true=ctx.rho_true[r],
        )

        eig_steps = []
        eig_sum_recomputed = 0.0

        for t in range(ctx.params.T):
            p_theta = marginal_theta(posterior) if nmc else None

            sync_cuda(ctx.device)
            tic = time.perf_counter()

            eta_vec, eig_t = select_beam(
                posterior,
                p_theta,
                sigma_scenario,
                eta_history,
                r_history,
            )

            sync_cuda(ctx.device)
            results["decision_times"].append(time.perf_counter() - tic)

            # Pour DAD/random, l'EIG est seulement diagnostique.
            # On l'évalue avec le MEME mode (mean ou marginal) que la baseline.
            if not nmc:
                eig_t = estimate_eig_for_selected_beam(
                    ctx=ctx,
                    eta_vec=eta_vec,
                    posterior=posterior,
                    sigma_scenario=sigma_scenario,
                    eig_mode=eig_mode,
                    N=ctx.params.N,
                )

            eig_steps.append(eig_t.item())

            eig_recomputed = estimate_eig_for_selected_beam(
                ctx=ctx,
                eta_vec=eta_vec,
                posterior=posterior,
                sigma_scenario=sigma_scenario,
                eig_mode=eig_mode,
                N=N_RECOMPUTE,
            )
            eig_sum_recomputed += eig_recomputed.item()

            amp_vec = generate_amplitude_measurement(
                theta=ctx.theta_true[r],
                rho=ctx.rho_true[r],
                eta=eta_vec,
                s=ctx.s,
                sigma=sigma_scenario,
                params=ctx.params,
                noise=(ctx.noise_real[r, t], ctx.noise_imag[r, t]),
            )

            posterior = update_posterior(
                eta=eta_vec,
                a_grid=ctx.a_grid,
                amp_vec=amp_vec,
                posterior=posterior,
                rho_grid=ctx.rho_grid,
                s=ctx.s,
                snr_db=SNR_DB,
                mode=POSTERIOR_MODE,
            )

            eta_history = torch.cat(
                [eta_history, eta_vec.reshape(1, 1, -1)],
                dim=1,
            )
            r_history = torch.cat(
                [r_history, amp_vec.reshape(1, 1, -1)],
                dim=1,
            )

        theta_hat = ctx.theta_grid[
            torch.argmax(marginal_theta(posterior))
        ]
        rho_hat = ctx.rho_grid[
            torch.argmax(marginal_rho(posterior))
        ]

        theta_error = torch.abs(
            torch.rad2deg(theta_hat - ctx.theta_true[r])
        ).item()
        rho_error = torch.abs(
            rho_hat - ctx.rho_true[r]
        ).item()

        g_L = evaluate_g_L(
            ctx,
            r,
            eta_history,
            r_history,
        )

        results["theta_errors"].append(theta_error)
        results["rho_errors"].append(rho_error)
        results["g_L"].append(g_L)
        results["eig_sums"].append(sum(eig_steps))
        results["eig_per_step"].append(eig_steps)
        results["eig_sums_recomputed"].append(eig_sum_recomputed)

        print(
            f"Réalisation {r + 1:3d} | "
            f"theta err = {theta_error:7.3f} deg | "
            f"rho err = {rho_error:.4f} | "
            f"sum EIG = {sum(eig_steps):.3f} | "
            f"g_L = {g_L:.3f}"
        )

    return {
        key: torch.tensor(values)
        for key, values in results.items()
    }


# ============================================================
# EIG candidate generation
# ============================================================


def generate_random_continuous_candidates(
    K,
    n_candidates,
    device,
    generator,
):
    eta_grid = torch.zeros(K, n_candidates, device=device)
    eta_grid[1:, :] = 2.0 * math.pi * torch.rand(
        K - 1,
        n_candidates,
        device=device,
        generator=generator,
    )
    return eta_grid


def sample_quantized_candidates(
    full_codebook,
    n_candidates,
    generator,
):
    """Échantillonne sans remplacement dans le codebook quantifié."""

    num_total = full_codebook.shape[1]
    if n_candidates > num_total:
        raise ValueError(
            f"n_candidates={n_candidates} > taille codebook={num_total}."
        )

    idx = torch.randperm(
        num_total,
        device=full_codebook.device,
        generator=generator,
    )[:n_candidates]

    return full_codebook[:, idx]


# ============================================================
# EIG / NMC baseline
# ============================================================


def eig_nmc(ctx, eig_mode, candidate_cfg):
    beam_generator = torch.Generator(
        device=ctx.device
    ).manual_seed(SEED + 1000)

    full_codebook = None

    if candidate_cfg.search == "exhaustive":
        full_codebook = generate_quantized_codebook(
            ctx.params.K,
            ctx.params.B,
            ctx.device,
        )
        NAMES["baseline"] = (
            f"EIG / NMC {eig_mode.upper()} EXHAUSTIF QUANTIFIE"
        )

    elif candidate_cfg.beam_space == "quantized":
        # Le full codebook n'a que 4^(K-1)=16384 beams pour K=8, B=2 :
        # on peut le générer une fois puis tirer des indices uniques à chaque décision.
        full_codebook = generate_quantized_codebook(
            ctx.params.K,
            ctx.params.B,
            ctx.device,
        )
        NAMES["baseline"] = (
            f"EIG / NMC {eig_mode.upper()} RANDOM QUANTIFIE "
            f"({candidate_cfg.n_candidates})"
        )

    else:
        NAMES["baseline"] = (
            f"EIG / NMC {eig_mode.upper()} RANDOM CONTINU "
            f"({candidate_cfg.n_candidates})"
        )

    if candidate_cfg.search == "exhaustive":
        beam_text = f"{full_codebook.shape[1]} exhaustifs quantifiés"
    else:
        beam_text = (
            f"{candidate_cfg.n_candidates} aléatoires "
            f"{candidate_cfg.beam_space}"
        )

    print(f"Baseline beams: {beam_text} | NMC N: {ctx.params.N}")
    print(f"EIG mode: {eig_mode}")

    if eig_mode == "mean":
        print(
            "rho_mean plug-in sous sigma_snr : "
            "sigma est aussi évalué à rho_mean (pas à rho_true)."
        )

    def select_beam(
        posterior,
        p_theta,
        sigma_scenario,
        eta_history,
        r_history,
    ):
        del eta_history, r_history

        # Nouveau pool à chaque décision pour la recherche aléatoire.
        if candidate_cfg.search == "exhaustive":
            candidates = full_codebook

        elif candidate_cfg.beam_space == "quantized":
            candidates = sample_quantized_candidates(
                full_codebook=full_codebook,
                n_candidates=candidate_cfg.n_candidates,
                generator=beam_generator,
            )

        else:
            candidates = generate_random_continuous_candidates(
                K=ctx.params.K,
                n_candidates=candidate_cfg.n_candidates,
                device=ctx.device,
                generator=beam_generator,
            )

        if eig_mode == "mean":
            sigma_eig = sigma_for_mean_eig(ctx, posterior)
        else:
            # Ignoré par le mode marginal, qui recalcule sigma(rho_j).
            sigma_eig = sigma_scenario

        eta, eig_values = choose_beam(
            eta_grid=candidates,
            a_grid=ctx.a_grid,
            s=ctx.s,
            snr_db=SNR_DB,
            sigma=sigma_eig,
            p_theta=p_theta,
            posterior=posterior,
            rho_grid=ctx.rho_grid,
            mode=eig_mode,
            N=ctx.params.N,
        )

        return eta.squeeze(-1), eig_values.max()

    return evaluate_method(
        ctx,
        "baseline",
        select_beam,
        eig_mode=eig_mode,
        nmc=True,
    )


# ============================================================
# DAD
# ============================================================


def load_policy(ctx):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=ctx.device,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        raise ValueError(
            "Checkpoint invalide : clé model_state_dict ou "
            "policy_state_dict attendue."
        )

    design_dim = (
        checkpoint.get("Nx", ctx.params.Nx)
        * checkpoint.get("Ny", ctx.params.Ny)
    )
    observation_dim = checkpoint.get("Ns", ctx.s.numel())

    if (
        design_dim != ctx.params.K
        or observation_dim != ctx.s.numel()
    ):
        raise ValueError(
            "Les dimensions du checkpoint ne correspondent pas "
            "au test (antennes / Ns)."
        )

    policy = DADPolicy(
        design_dim=design_dim,
        observation_dim=observation_dim,
        hidden_dim=state_dict["encoder.net.0.weight"].shape[0],
        encoding_dim=state_dict["encoder.net.2.weight"].shape[0],
    ).to(ctx.device)

    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            f"Le format de {CHECKPOINT_PATH} est reconnu, mais ses poids "
            "ne sont pas compatibles avec l'architecture DAD actuelle.\n"
            f"{exc}"
        ) from exc

    policy.eval()
    print(f"Réseau chargé : {CHECKPOINT_PATH}")

    return policy


def dad(ctx, policy, eig_mode):
    # Chauffe aussi l'encodeur avec un historique non vide.
    with torch.inference_mode():
        for t in (0, max(1, ctx.params.T - 1)):
            eta = torch.zeros(
                1, t, ctx.params.K, device=ctx.device
            )
            obs = torch.zeros(
                1, t, ctx.s.numel(), device=ctx.device
            )
            for _ in range(20):
                policy(eta, obs)

    sync_cuda(ctx.device)

    def select_beam(
        posterior,
        p_theta,
        sigma_scenario,
        eta_history,
        r_history,
    ):
        del posterior, p_theta, sigma_scenario
        return policy(eta_history, r_history)[0], None

    return evaluate_method(
        ctx,
        "dad",
        select_beam,
        eig_mode=eig_mode,
    )


# ============================================================
# Random continuous baseline
# ============================================================


def random_continuous_beam(K, device):
    eta = torch.zeros(K, device=device)
    eta[1:] = 2.0 * math.pi * torch.rand(
        K - 1,
        device=device,
    )
    return eta


def random(ctx, eig_mode):
    def select_beam(
        posterior,
        p_theta,
        sigma_scenario,
        eta_history,
        r_history,
    ):
        del posterior, p_theta, sigma_scenario, eta_history, r_history
        return random_continuous_beam(
            ctx.params.K,
            ctx.device,
        ), None

    return evaluate_method(
        ctx,
        "random",
        select_beam,
        eig_mode=eig_mode,
    )


# ============================================================
# Results
# ============================================================


def print_results(results, n_diagnostic, eig_mode):
    for name, values in results.items():
        theta = values["theta_errors"]

        print(f"\n{'=' * 60}\n{NAMES[name]}\n{'=' * 60}")
        print(f"Theta MAE       : {theta.mean().item():.4f} deg")
        print(f"Theta median    : {theta.median().item():.4f} deg")
        print(
            f"Theta RMSE      : "
            f"{theta.square().mean().sqrt().item():.4f} deg"
        )
        print(
            f"rho MAE         : "
            f"{values['rho_errors'].mean().item():.6f}"
        )
        print(
            f"Mean g_L        : "
            f"{values['g_L'].mean().item():.4f} nats"
        )
        print(
            f"Decision time   : "
            f"{1e3 * values['decision_times'].mean().item():.4f} ms / beam"
        )

        print(
            f"Sum EIG {eig_mode} diagnostic (N={n_diagnostic}) : "
            f"{values['eig_sums'].mean().item():.4f} nats"
        )
        print(
            f"Sum EIG {eig_mode} recomputed (N={N_RECOMPUTE}) : "
            f"{values['eig_sums_recomputed'].mean().item():.4f} nats"
        )

        print("Mean EIG per step :")
        for t, eig in enumerate(
            values["eig_per_step"].mean(dim=0),
            start=1,
        ):
            print(f"  {t:2d} : {eig.item():.4f} nats")

    if "baseline" in results and "dad" in results:
        baseline_time = results["baseline"][
            "decision_times"
        ].mean().item()
        dad_time = results["dad"][
            "decision_times"
        ].mean().item()

        print(
            f"\nSpeed-up NMC / DAD : "
            f"{baseline_time / dad_time:.1f} x"
        )


def save_results(
    ctx,
    results,
    eig_mode,
    candidate_cfg,
):
    saved = {
        "snr_db": SNR_DB,
        "seed": SEED,
        "L_eval": L_EVAL,
        "posterior_mode": POSTERIOR_MODE,
        "eig_mode": eig_mode,
        "methods": list(results),
        "theta_true": ctx.theta_true.cpu(),
        "rho_true": ctx.rho_true.cpu(),
    }

    if candidate_cfg is not None:
        saved["eig_search"] = candidate_cfg.search
        saved["eig_beam_space"] = candidate_cfg.beam_space
        saved["eig_n_candidates"] = candidate_cfg.n_candidates

    if "dad" in results:
        saved["checkpoint_path"] = str(CHECKPOINT_PATH)

    for name, values in results.items():
        for metric, value in values.items():
            saved[f"{name}_{metric}"] = value

    torch.save(saved, OUTPUT_PATH)
    print(f"\nResults saved to {OUTPUT_PATH}")


# ============================================================
# Main
# ============================================================


def main():
    choices = choose_methods()
    ctx = make_test_set()

    # Par défaut, les diagnostics DAD/random restent en EIG marginal.
    eig_mode = "marginal"
    candidate_cfg = None

    if "1" in choices:
        eig_mode = choose_eig_mode()
        candidate_cfg = choose_eig_candidates(ctx.params)

    print("\nRésumé configuration :")
    print(f"  posterior       : {POSTERIOR_MODE}")
    print(f"  EIG diagnostic : {eig_mode}")

    if candidate_cfg is not None:
        print(f"  recherche EIG  : {candidate_cfg.search}")
        print(f"  espace beams   : {candidate_cfg.beam_space}")
        if candidate_cfg.n_candidates is not None:
            print(f"  nb beams        : {candidate_cfg.n_candidates}")

    # Vérifie le réseau avant une éventuelle longue évaluation NMC.
    policy = load_policy(ctx) if "2" in choices else None

    results = {}

    if "1" in choices:
        results["baseline"] = eig_nmc(
            ctx,
            eig_mode=eig_mode,
            candidate_cfg=candidate_cfg,
        )

    if "2" in choices:
        results["dad"] = dad(
            ctx,
            policy,
            eig_mode=eig_mode,
        )

    if "3" in choices:
        results["random"] = random(
            ctx,
            eig_mode=eig_mode,
        )

    print_results(
        results,
        n_diagnostic=ctx.params.N,
        eig_mode=eig_mode,
    )

    save_results(
        ctx,
        results,
        eig_mode=eig_mode,
        candidate_cfg=candidate_cfg,
    )


if __name__ == "__main__":
    main()
