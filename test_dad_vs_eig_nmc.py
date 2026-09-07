"""Compare les méthodes choisies sur les mêmes réalisations physiques."""

import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.beam_eig.array_model import steering_vector, beam_from_phases
from modules.beam_eig.codebook import generate_quantized_codebook
from modules.beam_eig.baseline_eig import choose_beam, estimate_eig_for_eta_marginal
from modules.beam_eig.posterior import update_posterior, marginal_theta, marginal_rho
from modules.beam_eig.simulator import sigma_from_snr
from modules.dad.policy import DADPolicy
from modules.dad.contrastive import contrastive_bound, make_log_likelihood_fn


# Configuration
SNR_DB = 0.0
N_REAL = 200
L_EVAL = 3000  # Contrastifs pour l'évaluation finale, indépendants de l'entraînement.
N_RECOMPUTE = 5000  # Recalcul EIG du beam choisi pour toutes les méthodes.
SEED = 42
CHECKPOINT_PATH = Path("dad_nx8_smoke_127.pt")
OUTPUT_PATH = Path("comparison_nmc_vs_dad.pt")
NAMES = {"baseline": "EIG / NMC", "dad": "DAD", "random": "RANDOM CONTINUOUS"}


def choose_methods():
    print("Méthodes à tester : 1 = EIG / NMC, 2 = DAD (réseau), 3 = Random")
    print("Combinaisons possibles : 1, 2, 3, 1 2, 1 3, 2 3 ou 1 2 3.")
    print("Chaque méthode : erreurs finales, sum_eig et g_L. Seul le cas 1 sélectionne par EIG.")
    while True:
        choices = set(re.split(r"[\s,;+]+", input("Ton choix : ").strip()))
        if choices and choices <= {"1", "2", "3"}:
            return choices
        print("Entre au moins un numéro parmi 1, 2 et 3 (exemple : 2 3).")


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
    theta_grid = torch.deg2rad(torch.linspace(30.0, 150.0, 121, device=device))
    rho_grid = torch.linspace(0.05, 1.0, 50, device=device)
    theta_min = math.radians(30.0)
    theta_range = math.radians(120.0)

    # Générés une seule fois, partagés par toutes les méthodes sélectionnées.
    theta_true = theta_min + theta_range * torch.rand(N_REAL, device=device)
    rho_true = 0.05 + 0.95 * torch.rand(N_REAL, device=device)
    noise_real = torch.randn(N_REAL, params.T, s.numel(), device=device)
    noise_imag = torch.randn_like(noise_real)
    theta_contrast = theta_min + theta_range * torch.rand(N_REAL, L_EVAL, device=device)

    print(f"Device: {device} | Antennes: {params.K} | T: {params.T} | Ns: {s.numel()}")
    return SimpleNamespace(
        device=device, params=params, s=s, theta_grid=theta_grid, rho_grid=rho_grid,
        a_grid=steering_vector(theta_grid, params), theta_true=theta_true,
        rho_true=rho_true, noise_real=noise_real, noise_imag=noise_imag,
        theta_contrast=theta_contrast,
        log_likelihood_fn=make_log_likelihood_fn(
            rho_grid=rho_grid, s=s, snr_db=SNR_DB, params=params,
        ),
        log_p_rho_prior=torch.full(
            (rho_grid.numel(),), -math.log(rho_grid.numel()), device=device,
        ),
    )


def generate_measurement_fixed_noise(ctx, realization, step, eta, sigma):
    a = steering_vector(ctx.theta_true[realization], ctx.params)
    b = beam_from_phases(eta)
    alpha = ctx.rho_true[realization] * (b.conj().T @ a).squeeze()
    noise = sigma / math.sqrt(2.0) * (
        ctx.noise_real[realization, step] + 1j * ctx.noise_imag[realization, step]
    )
    return torch.abs(alpha * ctx.s + noise)


def evaluate_g_L(ctx, realization, eta_history, r_history):
    theta_candidates = torch.cat([
        ctx.theta_true[realization].reshape(1, 1),
        ctx.theta_contrast[realization].reshape(1, -1),
    ], dim=1)
    bound, _ = contrastive_bound(
        theta_candidates=theta_candidates, eta_history=eta_history,
        r_history=r_history, log_likelihood_fn=ctx.log_likelihood_fn,
        log_p_rho_prior=ctx.log_p_rho_prior,
    )
    return bound.item()


@torch.inference_mode()
def evaluate_method(ctx, name, select_beam, *, nmc=False):
    """Boucle commune : décision, observation, posterior et scores finaux.

    select_beam renvoie (phases [K], EIG optionnel). Le posterior ne sert
    à choisir le beam que pour NMC ; DAD utilise uniquement l'historique.
    """
    print(f"\n{'=' * 60}\n{NAMES[name]}\n{'=' * 60}")
    # Rend chaque méthode reproductible indépendamment de la combinaison choisie.
    torch.manual_seed(SEED + {"baseline": 1, "dad": 2, "random": 3}[name])
    results = {key: [] for key in ("theta_errors", "rho_errors", "g_L", "decision_times", "eig_sums", "eig_per_step")}
    results["eig_sums_recomputed"] = []

    for r in range(ctx.theta_true.numel()):
        posterior = torch.ones(
            ctx.theta_grid.numel(), ctx.rho_grid.numel(), device=ctx.device,
        )
        posterior /= posterior.sum()
        eta_history = torch.empty(1, 0, ctx.params.K, device=ctx.device)
        r_history = torch.empty(1, 0, ctx.s.numel(), device=ctx.device)
        sigma = sigma_from_snr(s=ctx.s, snr_db=SNR_DB, rho_true=ctx.rho_true[r])
        eig_steps = []
        eig_sum_recomputed = 0.0

        for t in range(ctx.params.T):
            # Le marginal du posterior est préparé hors du temps de décision.
            p_theta = marginal_theta(posterior) if nmc else None
            sync_cuda(ctx.device)
            tic = time.perf_counter()
            eta_vec, eig_t = select_beam(posterior, p_theta, sigma, eta_history, r_history)
            sync_cuda(ctx.device)
            results["decision_times"].append(time.perf_counter() - tic)

            # Diagnostic du beam déjà choisi, hors du temps de décision
            # et avant d'incorporer la nouvelle observation au posterior.
            if not nmc:
                eig_t = estimate_eig_for_eta_marginal(
                    eta=eta_vec[:, None], a_grid=ctx.a_grid, s=ctx.s,
                    snr_db=SNR_DB, posterior=posterior, rho_grid=ctx.rho_grid,
                    N=ctx.params.N,
                )
            eig_steps.append(eig_t.item())
            eig_recomputed = estimate_eig_for_eta_marginal(
                eta=eta_vec[:, None], a_grid=ctx.a_grid, s=ctx.s,
                snr_db=SNR_DB, posterior=posterior, rho_grid=ctx.rho_grid,
                N=N_RECOMPUTE,
            )
            eig_sum_recomputed += eig_recomputed.item()

            amp_vec = generate_measurement_fixed_noise(ctx, r, t, eta_vec, sigma)
            posterior = update_posterior(
                eta=eta_vec, a_grid=ctx.a_grid, amp_vec=amp_vec,
                posterior=posterior, rho_grid=ctx.rho_grid, s=ctx.s, snr_db=SNR_DB,
            )
            eta_history = torch.cat([eta_history, eta_vec.reshape(1, 1, -1)], dim=1)
            r_history = torch.cat([r_history, amp_vec.reshape(1, 1, -1)], dim=1)

        theta_hat = ctx.theta_grid[torch.argmax(marginal_theta(posterior))]
        rho_hat = ctx.rho_grid[torch.argmax(marginal_rho(posterior))]
        theta_error = torch.abs(torch.rad2deg(theta_hat - ctx.theta_true[r])).item()
        rho_error = torch.abs(rho_hat - ctx.rho_true[r]).item()
        g_L = evaluate_g_L(ctx, r, eta_history, r_history)
        results["theta_errors"].append(theta_error)
        results["rho_errors"].append(rho_error)
        results["g_L"].append(g_L)
        results["eig_sums"].append(sum(eig_steps))
        results["eig_per_step"].append(eig_steps)
        results["eig_sums_recomputed"].append(eig_sum_recomputed)
        eig_text = f"sum EIG = {sum(eig_steps):.3f} | "
        print(
            f"Réalisation {r + 1:3d} | theta err = {theta_error:7.3f} deg | "
            f"rho err = {rho_error:.4f} | {eig_text}g_L = {g_L:.3f}"
        )

    return {key: torch.tensor(values) for key, values in results.items()}


def choose_eig_candidates(params):
    print(f"EIG / NMC : 1 = exhaustif, 2 = aléatoire (phases quantifiées sur {params.B} bits)")
    while True:
        choice = input("Sélection des beams : ").strip()
        if choice == "1":
            return None
        if choice == "2" and params.R > 1:
            break
        print("Choisis 1 ou 2." if params.R > 1 else "Un seul beam possible : choisis 1.")
    while True:
        try:
            count = int(input(f"Nombre de beams aléatoires (1 à {params.R - 1}) : "))
            if 1 <= count < params.R:
                return count
        except ValueError:
            pass
        print(f"Entre un entier strictement positif et inférieur à {params.R}.")


def generate_random_beam_candidates(K, B, n_candidates, device, generator):
    eta_grid = torch.zeros(K, n_candidates, device=device)
    eta_grid[1:, :] = (2.0 * math.pi / (2**B)) * torch.randint(
        2**B, (K - 1, n_candidates), device=device, generator=generator,
    )
    return eta_grid


def eig_nmc(ctx, n_candidates=None):
    if n_candidates is None:
        eta_grid = generate_quantized_codebook(ctx.params.K, ctx.params.B, ctx.device)
        NAMES["baseline"] = "EIG / NMC EXHAUSTIF"
    else:
        if not 1 <= n_candidates < ctx.params.R:
            raise ValueError(f"Le nombre de beams doit être compris entre 1 et {ctx.params.R - 1}.")
        beam_generator = torch.Generator(device=ctx.device).manual_seed(SEED + 1000)
        NAMES["baseline"] = f"EIG / NMC RANDOM SEARCH ({n_candidates})"
    print(f"Baseline beams: {ctx.params.R if n_candidates is None else n_candidates} | NMC N: {ctx.params.N}")

    def select_beam(posterior, p_theta, sigma, eta_history, r_history):
        # Nouveau pool à chaque décision, indépendant du RNG des estimations NMC.
        candidates = eta_grid if n_candidates is None else generate_random_beam_candidates(
            ctx.params.K, ctx.params.B, n_candidates, ctx.device, beam_generator,
        )
        eta, eig_values = choose_beam(
            eta_grid=candidates, a_grid=ctx.a_grid, s=ctx.s, snr_db=SNR_DB,
            sigma=sigma, p_theta=p_theta, posterior=posterior,
            rho_grid=ctx.rho_grid, mode="mean", N=ctx.params.N,
        )
        return eta.squeeze(-1), eig_values.max()

    return evaluate_method(ctx, "baseline", select_beam, nmc=True)


def load_policy(ctx):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=ctx.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        raise ValueError(
            "Checkpoint invalide : clé model_state_dict ou policy_state_dict attendue."
        )

    # Les checkpoints intermédiaires n'enregistrent pas les paramètres physiques.
    design_dim = checkpoint.get("Nx", ctx.params.Nx) * checkpoint.get("Ny", ctx.params.Ny)
    observation_dim = checkpoint.get("Ns", ctx.s.numel())
    if design_dim != ctx.params.K or observation_dim != ctx.s.numel():
        raise ValueError("Les dimensions du checkpoint ne correspondent pas au test (antennes / Ns).")
    policy = DADPolicy(
        design_dim=design_dim, observation_dim=observation_dim,
        hidden_dim=state_dict["encoder.net.0.weight"].shape[0],
        encoding_dim=state_dict["encoder.net.2.weight"].shape[0],
    ).to(ctx.device)
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            f"Le format de {CHECKPOINT_PATH} est reconnu, mais ses poids ne sont pas "
            "compatibles avec l'architecture DAD actuelle. Utilise l'encodeur et "
            "l'émetteur correspondant à l'entraînement de ce checkpoint.\n"
            f"{exc}"
        ) from exc
    policy.eval()
    print(f"Réseau chargé : {CHECKPOINT_PATH}")
    return policy


def dad(ctx, policy):
    # Chauffe aussi l'encodeur, avec un historique non vide.
    with torch.inference_mode():
        for t in (0, max(1, ctx.params.T - 1)):
            eta = torch.zeros(1, t, ctx.params.K, device=ctx.device)
            obs = torch.zeros(1, t, ctx.s.numel(), device=ctx.device)
            for _ in range(20):
                policy(eta, obs)
    sync_cuda(ctx.device)

    def select_beam(posterior, p_theta, sigma, eta_history, r_history):
        return policy(eta_history, r_history)[0], None

    return evaluate_method(ctx, "dad", select_beam)


def random_continuous_beam(K, device):
    eta = torch.zeros(K, device=device)
    eta[1:] = 2.0 * math.pi * torch.rand(K - 1, device=device)
    return eta


def random(ctx):
    def select_beam(posterior, p_theta, sigma, eta_history, r_history):
        return random_continuous_beam(ctx.params.K, ctx.device), None

    return evaluate_method(ctx, "random", select_beam)


def print_results(results, n_diagnostic):
    for name, values in results.items():
        theta = values["theta_errors"]
        print(f"\n{'=' * 60}\n{NAMES[name]}\n{'=' * 60}")
        print(f"Theta MAE       : {theta.mean().item():.4f} deg")
        print(f"Theta median    : {theta.median().item():.4f} deg")
        print(f"Theta RMSE      : {theta.square().mean().sqrt().item():.4f} deg")
        print(f"rho MAE         : {values['rho_errors'].mean().item():.6f}")
        print(f"Mean g_L        : {values['g_L'].mean().item():.4f} nats")
        print(f"Decision time   : {1e3 * values['decision_times'].mean().item():.4f} ms / beam")
        if "eig_sums" in values:
            print(f"Sum EIG diagnostic (N={n_diagnostic}) : {values['eig_sums'].mean().item():.4f} nats")
            if "eig_sums_recomputed" in values:
                print(f"Sum EIG recomputed (N={N_RECOMPUTE}) : {values['eig_sums_recomputed'].mean().item():.4f} nats")
            print("Mean EIG per step :")
            for t, eig in enumerate(values["eig_per_step"].mean(dim=0), start=1):
                print(f"  {t:2d} : {eig.item():.4f} nats")

    if "baseline" in results and "dad" in results:
        baseline_time = results["baseline"]["decision_times"].mean().item()
        dad_time = results["dad"]["decision_times"].mean().item()
        print(f"\nSpeed-up NMC / DAD : {baseline_time / dad_time:.1f} x")


def save_results(ctx, results):
    # Conserve les noms de clés existants et ajoute ceux du random si sélectionné.
    saved = {
        "snr_db": SNR_DB, "seed": SEED, "L_eval": L_EVAL,
        "methods": list(results),
        "theta_true": ctx.theta_true.cpu(), "rho_true": ctx.rho_true.cpu(),
    }
    if "dad" in results:
        saved["checkpoint_path"] = str(CHECKPOINT_PATH)
    for name, values in results.items():
        for metric, value in values.items():
            saved[f"{name}_{metric}"] = value
    torch.save(saved, OUTPUT_PATH)
    print(f"\nResults saved to {OUTPUT_PATH}")


def main():
    choices = choose_methods()
    ctx = make_test_set()
    if "1" in choices:
        ctx.n_candidates = choose_eig_candidates(ctx.params)
    # Vérifie le réseau avant de lancer une éventuelle longue évaluation NMC.
    policy = load_policy(ctx) if "2" in choices else None
    results = {}
    if "1" in choices:
        results["baseline"] = eig_nmc(ctx, ctx.n_candidates)
    if "2" in choices:
        results["dad"] = dad(ctx, policy)
    if "3" in choices:
        results["random"] = random(ctx)
    print_results(results, ctx.params.N)
    save_results(ctx, results)


if __name__ == "__main__":
    main()
