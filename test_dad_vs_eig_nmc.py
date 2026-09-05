import math
import time
from pathlib import Path

import torch

from modules.beam_eig.params import Params

from modules.beam_eig.pilot import (
    generate_pilot_sequence,
)

from modules.beam_eig.array_model import (
    steering_vector,
    beam_from_phases,
)

from modules.beam_eig.codebook import (
    generate_quantized_codebook,
)

from modules.beam_eig.baseline_eig import (
    choose_beam,
    estimate_eig_for_eta_marginal,
    estimate_eig_for_eta_mean,
)

from modules.beam_eig.posterior import (
    update_posterior,
    marginal_theta,
    marginal_rho,
    compute_rho_mean,
)

from modules.beam_eig.simulator import (
    sigma_from_snr,
)

from modules.dad.policy import (
    DADPolicy,
)

from modules.dad.contrastive import (
    contrastive_bound,
    make_log_likelihood_fn,
)


# ============================================================
# Configuration
# ============================================================

SNR_DB = 0.0

N_REAL = 200

# Nombre de contrastifs uniquement pour l'EVALUATION finale
# de g_L. Ça n'a rien à voir avec le L utilisé pendant
# l'entraînement.
L_EVAL = 3000

SEED = 42

CHECKPOINT_PATH = Path(
    "dad_nx4_smoke.pt"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Helpers
# ============================================================

def sync_cuda():
    """
    Nécessaire pour mesurer correctement les temps GPU.
    """

    if device.type == "cuda":
        torch.cuda.synchronize()


def make_uniform_posterior(
    theta_grid,
    rho_grid,
):
    posterior = torch.ones(
        theta_grid.numel(),
        rho_grid.numel(),
        dtype=torch.float32,
        device=device,
    )

    return posterior / posterior.sum()


def generate_measurement_fixed_noise(
    theta,
    rho,
    eta,
    s,
    sigma,
    params,
    noise_real,
    noise_imag,
):
    """
    Même réalisation de bruit pour baseline et DAD.

    Le beam peut être différent, donc le signal reçu
    est différent, mais le bruit sous-jacent est identique.
    """

    a = steering_vector(
        theta,
        params,
    )

    b = beam_from_phases(
        eta,
    )

    alpha = rho * (
        b.conj().T @ a
    ).squeeze()

    mu = alpha * s

    w = (
        sigma
        / math.sqrt(2.0)
        * (
            noise_real
            + 1j * noise_imag
        )
    )

    y = mu + w

    return torch.abs(y)


def get_final_estimates(
    posterior,
    theta_grid,
    rho_grid,
):
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

    return theta_hat, rho_hat


def evaluate_g_L(
    theta_true,
    theta_contrast,
    eta_history,
    r_history,
    rho_grid,
    s,
    snr_db,
    params,
):
    """
    Évalue la trajectoire entière avec exactement le
    même score contrastif pour baseline et DAD.

    theta_true      : scalar
    theta_contrast  : [L_EVAL]
    eta_history     : [1, T, K]
    r_history       : [1, T, Ns]
    """

    theta_candidates = torch.cat(
        [
            theta_true.reshape(1, 1),
            theta_contrast.reshape(1, -1),
        ],
        dim=1,
    )

    R = rho_grid.numel()

    log_p_rho_prior = torch.full(
        (R,),
        -math.log(R),
        device=device,
        dtype=torch.float32,
    )

    log_likelihood_fn = (
        make_log_likelihood_fn(
            rho_grid=rho_grid,
            s=s,
            snr_db=snr_db,
            params=params,
        )
    )

    bound, _ = contrastive_bound(
        theta_candidates=theta_candidates,
        eta_history=eta_history,
        r_history=r_history,
        log_likelihood_fn=log_likelihood_fn,
        log_p_rho_prior=log_p_rho_prior,
    )

    return bound.item()


def print_summary(
    name,
    theta_errors,
    rho_errors,
    eig_sums,
    eig_sums_recomputed,
    g_L_values,
    decision_times,
):
    theta_errors = torch.tensor(
        theta_errors
    )

    rho_errors = torch.tensor(
        rho_errors
    )

    eig_sums = torch.tensor(
        eig_sums
    )

    eig_sums_recomputed = torch.tensor(
        eig_sums_recomputed
    )

    g_L_values = torch.tensor(
        g_L_values
    )

    decision_times = torch.tensor(
        decision_times
    )

    print()
    print(
        "=" * 60
    )
    print(name)
    print(
        "=" * 60
    )

    print(
        f"Theta MAE       : "
        f"{theta_errors.mean().item():.4f} deg"
    )

    print(
        f"Theta median    : "
        f"{theta_errors.median().item():.4f} deg"
    )

    print(
        f"Theta RMSE      : "
        f"{torch.sqrt(torch.mean(theta_errors**2)).item():.4f} deg"
    )

    print(
        f"rho MAE         : "
        f"{rho_errors.mean().item():.6f}"
    )

    print(
        f"Sum NMC EIG     : "
        f"{eig_sums.mean().item():.4f} nats"
    )

    print(
        f"Sum NMC EIG recomputed : "
        f"{eig_sums_recomputed.mean().item():.4f} nats"
    )

    print(
        f"Mean g_L        : "
        f"{g_L_values.mean().item():.4f} nats"
    )

    print(
        f"Decision time   : "
        f"{1e3 * decision_times.mean().item():.4f} ms / beam"
    )


# ============================================================
# Physical setup
# ============================================================

params = Params(
    Nx=4,
    Ny=1,
)

print(
    "N antennas :",
    params.K,
)

print(
    "T          :",
    params.T,
)

print(
    "NMC N      :",
    params.N,
)


# ============================================================
# Pilot
# ============================================================

s = generate_pilot_sequence(
    device=device,
    sequence_type="PSS",
).to(
    dtype=torch.float32,
)

Ns = s.numel()

print(
    "Ns         :",
    Ns,
)


# ============================================================
# theta / rho grids
#
# Same grids as baseline
# ============================================================

theta_grid = torch.deg2rad(
    torch.linspace(
        30.0,
        150.0,
        121,
        dtype=torch.float32,
        device=device,
    )
)

rho_grid = torch.linspace(
    0.05,
    1.0,
    50,
    dtype=torch.float32,
    device=device,
)

L_theta = theta_grid.numel()
L_rho = rho_grid.numel()


# ============================================================
# Steering matrix
# ============================================================

a_grid = steering_vector(
    theta_grid,
    params,
)


# ============================================================
# Baseline codebook
#
# B = 2 -> 4^(K-1) = 64 beams
# ============================================================

eta_grid = generate_quantized_codebook(
    params.K,
    params.B,
    device,
)

print(
    "Baseline beams:",
    eta_grid.shape[1],
)


# ============================================================
# FIXED TEST SET
#
# Important:
# baseline et DAD voient exactement les mêmes
# theta_true et rho_true.
# ============================================================

theta_min = math.radians(30.0)
theta_max = math.radians(150.0)

theta_true_all = (
    theta_min
    + (
        theta_max
        - theta_min
    )
    * torch.rand(
        N_REAL,
        device=device,
    )
)

rho_true_all = (
    0.05
    + 0.95
    * torch.rand(
        N_REAL,
        device=device,
    )
)


# ============================================================
# FIXED ENVIRONMENT NOISE
#
# Même bruit physique pour baseline et DAD.
#
# Attention :
# le bruit Monte-Carlo utilisé à l'intérieur de
# estimate_eig_for_eta reste indépendant.
# ============================================================

noise_real_all = torch.randn(
    N_REAL,
    params.T,
    Ns,
    device=device,
)

noise_imag_all = torch.randn(
    N_REAL,
    params.T,
    Ns,
    device=device,
)


# ============================================================
# Fixed contrastive theta for g_L evaluation
#
# SAME theta contrastifs for baseline and DAD.
# ============================================================

theta_contrast_all = (
    theta_min
    + (
        theta_max
        - theta_min
    )
    * torch.rand(
        N_REAL,
        L_EVAL,
        device=device,
    )
)


# ============================================================
# Storage
# ============================================================

baseline_theta_errors = []
baseline_rho_errors = []

baseline_eig_sums = []
baseline_eig_sums_recomputed = []
baseline_eig_per_step = []

baseline_g_L = []

baseline_decision_times = []


dad_theta_errors = []
dad_rho_errors = []

dad_eig_sums = []
dad_eig_sums_recomputed = []
dad_eig_per_step = []

dad_g_L = []

dad_decision_times = []


# ============================================================
#
# PART 1
#
# BASELINE EIG / NMC
#
# ============================================================

print()
print(
    "=" * 60
)
print(
    "BASELINE EIG / NMC"
)
print(
    "=" * 60
)


with torch.no_grad():

    for r in range(
        N_REAL
    ):

        theta_true = (
            theta_true_all[r]
        )

        rho_true = (
            rho_true_all[r]
        )

        sigma = sigma_from_snr(
            s=s,
            snr_db=SNR_DB,
            rho_true=rho_true,
        )


        # ====================================================
        # Initial posterior
        # ====================================================

        posterior = (
            make_uniform_posterior(
                theta_grid,
                rho_grid,
            )
        )


        # ====================================================
        # Histories
        #
        # We store them in DAD convention:
        #
        # eta_history : [1,T,K]
        # r_history   : [1,T,Ns]
        # ====================================================

        eta_history = torch.empty(
            1,
            0,
            params.K,
            device=device,
        )

        r_history = torch.empty(
            1,
            0,
            Ns,
            device=device,
        )


        eig_sum = 0.0
        eig_sum_recomputed = 0.0
        eig_steps = []


        # ====================================================
        # Adaptive baseline
        # ====================================================

        for t in range(
            params.T
        ):

            # --------------------------------------------
            # Current posterior p(theta)
            # --------------------------------------------

            p_theta = marginal_theta(
                posterior
            )



            # ============================================
            # Beam selection timing
            # ============================================

            sync_cuda()

            tic = time.perf_counter()

            eta_star, eig_values = (
                choose_beam(
                    eta_grid=eta_grid,
                    a_grid=a_grid,
                    s=s,
                    snr_db=SNR_DB,
                    p_theta=p_theta,
                    posterior=posterior,
                    rho_grid=rho_grid,
                    mode="marginal",
                    N=params.N,
                )
            )

            sync_cuda()

            baseline_decision_times.append(
                time.perf_counter()
                - tic
            )


            # --------------------------------------------
            # NMC EIG of selected beam
            # --------------------------------------------

            eig_t = eig_values.max()

            eig_recomputed= estimate_eig_for_eta_marginal(
                eta=eta_star.squeeze(-1)[:, None],
                a_grid=a_grid,
                s=s,
                snr_db=SNR_DB,
                posterior=posterior,
                rho_grid=rho_grid,
                N=5000,
            )

            eig_sum_recomputed += eig_recomputed.item()

            eig_sum += eig_t.item()

            eig_steps.append(
                eig_t.item()
            )


            # --------------------------------------------
            # [K,1] -> [K]
            # --------------------------------------------

            eta_vec = (
                eta_star.squeeze(-1)
            )


            # --------------------------------------------
            # True physical observation
            #
            # IMPORTANT:
            # fixed noise shared with DAD
            # --------------------------------------------

            amp_vec = (
                generate_measurement_fixed_noise(
                    theta=theta_true,
                    rho=rho_true,
                    eta=eta_vec,
                    s=s,
                    sigma=sigma,
                    params=params,
                    noise_real=(
                        noise_real_all[
                            r,
                            t,
                        ]
                    ),
                    noise_imag=(
                        noise_imag_all[
                            r,
                            t,
                        ]
                    ),
                )
            )


            # --------------------------------------------
            # Posterior update
            # --------------------------------------------

            posterior = update_posterior(
                eta=eta_vec,
                a_grid=a_grid,
                amp_vec=amp_vec,
                posterior=posterior,
                rho_grid=rho_grid,
                s=s,
                snr_db=SNR_DB,
            )


            # --------------------------------------------
            # Save trajectory
            # --------------------------------------------

            eta_history = torch.cat(
                [
                    eta_history,
                    eta_vec.reshape(
                        1,
                        1,
                        params.K,
                    ),
                ],
                dim=1,
            )

            r_history = torch.cat(
                [
                    r_history,
                    amp_vec.reshape(
                        1,
                        1,
                        Ns,
                    ),
                ],
                dim=1,
            )


        # ====================================================
        # Final theta / rho estimates
        # ====================================================

        theta_hat, rho_hat = (
            get_final_estimates(
                posterior,
                theta_grid,
                rho_grid,
            )
        )


        theta_error = torch.abs(
            torch.rad2deg(
                theta_hat
                - theta_true
            )
        )

        rho_error = torch.abs(
            rho_hat
            - rho_true
        )


        # ====================================================
        # Common trajectory-level g_L evaluation
        # ====================================================

        g_L_value = evaluate_g_L(
            theta_true=theta_true,
            theta_contrast=(
                theta_contrast_all[r]
            ),
            eta_history=eta_history,
            r_history=r_history,
            rho_grid=rho_grid,
            s=s,
            sigma=sigma,
            params=params,
        )


        # ====================================================
        # Storage
        # ====================================================

        baseline_theta_errors.append(
            theta_error.item()
        )

        baseline_rho_errors.append(
            rho_error.item()
        )

        baseline_eig_sums.append(
            eig_sum
        )

        baseline_eig_sums_recomputed.append(
            eig_sum_recomputed
        )

        baseline_eig_per_step.append(
            eig_steps
        )

        baseline_g_L.append(
            g_L_value
        )


        print(
            f"Realisation {r + 1:3d} | "
            f"theta err = "
            f"{theta_error.item():7.3f} deg | "
            f"rho err = "
            f"{rho_error.item():.4f} | "
            f"sum EIG = "
            f"{eig_sum:.3f} | "
            f"g_L = "
            f"{g_L_value:.3f}"
        )


# ============================================================
#
# PART 2
#
# LOAD DAD POLICY
#
# ============================================================

print()
print(
    "=" * 60
)
print(
    "LOADING DAD POLICY"
)
print(
    "=" * 60
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)


# ============================================================
# Recover architecture
#
# For your current checkpoint:
#
# encoder:
#   131 -> 256 -> 16
#
# emitter:
#   16 -> 4
#
# We infer 256 and 16 from the state_dict.
# ============================================================

state_dict = (
    checkpoint[
        "model_state_dict"
    ]
)

design_dim = (
    checkpoint["Nx"]
    * checkpoint["Ny"]
)

observation_dim = (
    checkpoint["Ns"]
)

hidden_dim = (
    state_dict[
        "encoder.net.0.weight"
    ].shape[0]
)

encoding_dim = (
    state_dict[
        "encoder.net.2.weight"
    ].shape[0]
)


print(
    "design_dim      :",
    design_dim,
)

print(
    "observation_dim :",
    observation_dim,
)

print(
    "hidden_dim      :",
    hidden_dim,
)

print(
    "encoding_dim    :",
    encoding_dim,
)


# ============================================================
# Safety checks
# ============================================================

if design_dim != params.K:

    raise ValueError(
        "Checkpoint antenna dimension "
        "does not match comparison setup."
    )

if observation_dim != Ns:

    raise ValueError(
        "Checkpoint observation dimension "
        "does not match PSS length."
    )


# ============================================================
# Recreate architecture + load trained weights
# ============================================================

policy = DADPolicy(
    design_dim=design_dim,
    observation_dim=observation_dim,
    hidden_dim=hidden_dim,
    encoding_dim=encoding_dim,
).to(
    device
)

policy.load_state_dict(
    state_dict
)

policy.eval()


# ============================================================
# Small GPU warmup
# ============================================================

with torch.inference_mode():

    dummy_eta = torch.empty(
        1,
        0,
        params.K,
        device=device,
    )

    dummy_r = torch.empty(
        1,
        0,
        Ns,
        device=device,
    )

    for _ in range(20):

        _ = policy(
            dummy_eta,
            dummy_r,
        )

sync_cuda()


# ============================================================
#
# PART 3
#
# DAD
#
# ============================================================

print()
print(
    "=" * 60
)
print(
    "DAD"
)
print(
    "=" * 60
)


with torch.inference_mode():

    for r in range(
        N_REAL
    ):

        theta_true = (
            theta_true_all[r]
        )

        rho_true = (
            rho_true_all[r]
        )


        sigma = sigma_from_snr(
            s=s,
            snr_db=SNR_DB,
            rho_true=rho_true,
        )


        # ====================================================
        # Same initial posterior
        #
        # DAD DOES NOT USE THIS posterior to choose its beam.
        #
        # It is only here:
        #
        # 1. to calculate the final estimator
        # 2. to evaluate the NMC EIG of DAD's beam
        # ====================================================

        posterior = (
            make_uniform_posterior(
                theta_grid,
                rho_grid,
            )
        )


        # ====================================================
        # Empty DAD history
        # ====================================================

        eta_history = torch.empty(
            1,
            0,
            params.K,
            dtype=torch.float32,
            device=device,
        )

        r_history = torch.empty(
            1,
            0,
            Ns,
            dtype=torch.float32,
            device=device,
        )


        eig_sum = 0.0
        eig_steps = []


        # ====================================================
        # Adaptive DAD loop
        # ====================================================

        for t in range(
            params.T
        ):

            # ============================================
            # DAD FORWARD PASS
            #
            # This is the online design decision.
            # ============================================

            sync_cuda()

            tic = time.perf_counter()

            eta_t = policy(
                eta_history,
                r_history,
            )

            sync_cuda()

            dad_decision_times.append(
                time.perf_counter()
                - tic
            )


            # eta_t : [1,K]
            eta_vec = eta_t[0]


            # ============================================
            # Evaluate immediate EIG of DAD beam
            #
            # IMPORTANT:
            # this is NOT used by DAD to make its choice.
            #
            # It is only a diagnostic so that the DAD
            # beam can be evaluated with the SAME NMC
            # estimator as the baseline.
            # ============================================

            p_theta = marginal_theta(
                posterior
            )

            rho_mean = (
                compute_rho_mean(
                    posterior,
                    rho_grid,
                )
            )

            eig_t = estimate_eig_for_eta_marginal(
                eta=eta_vec[:, None],
                a_grid=a_grid,
                s=s,
                snr_db=SNR_DB,
                posterior=posterior,
                rho_grid=rho_grid,
                N=params.N,
            )

            eig_sum += eig_t.item()

            eig_steps.append(
                eig_t.item()
            )


            # ============================================
            # True physical observation
            #
            # SAME theta, rho and noise as baseline.
            # ============================================

            amp_vec = (
                generate_measurement_fixed_noise(
                    theta=theta_true,
                    rho=rho_true,
                    eta=eta_vec,
                    s=s,
                    sigma=sigma,
                    params=params,
                    noise_real=(
                        noise_real_all[
                            r,
                            t,
                        ]
                    ),
                    noise_imag=(
                        noise_imag_all[
                            r,
                            t,
                        ]
                    ),
                )
            )


            # ============================================
            # Evaluation posterior
            #
            # Again:
            # DAD does NOT see this posterior.
            # ============================================

            posterior = update_posterior(
                eta=eta_vec,
                a_grid=a_grid,
                amp_vec=amp_vec,
                posterior=posterior,
                rho_grid=rho_grid,
                s=s,
                snr_db=SNR_DB,
            )


            # ============================================
            # Update history seen by DAD
            # ============================================

            eta_history = torch.cat(
                [
                    eta_history,
                    eta_t.unsqueeze(1),
                ],
                dim=1,
            )

            r_history = torch.cat(
                [
                    r_history,
                    amp_vec.reshape(
                        1,
                        1,
                        Ns,
                    ),
                ],
                dim=1,
            )


        # ====================================================
        # Final theta / rho estimates
        #
        # EXACT SAME estimator as baseline.
        # ====================================================

        theta_hat, rho_hat = (
            get_final_estimates(
                posterior,
                theta_grid,
                rho_grid,
            )
        )


        theta_error = torch.abs(
            torch.rad2deg(
                theta_hat
                - theta_true
            )
        )

        rho_error = torch.abs(
            rho_hat
            - rho_true
        )


        # ====================================================
        # Same trajectory g_L evaluator
        # ====================================================

        g_L_value = evaluate_g_L(
            theta_true=theta_true,
            theta_contrast=(
                theta_contrast_all[r]
            ),
            eta_history=eta_history,
            r_history=r_history,
            rho_grid=rho_grid,
            s=s,
            sigma=sigma,
            params=params,
        )


        # ====================================================
        # Storage
        # ====================================================

        dad_theta_errors.append(
            theta_error.item()
        )

        dad_rho_errors.append(
            rho_error.item()
        )

        dad_eig_sums.append(
            eig_sum
        )

        dad_eig_per_step.append(
            eig_steps
        )

        dad_g_L.append(
            g_L_value
        )


        print(
            f"Realisation {r + 1:3d} | "
            f"theta err = "
            f"{theta_error.item():7.3f} deg | "
            f"rho err = "
            f"{rho_error.item():.4f} | "
            f"sum EIG = "
            f"{eig_sum:.3f} | "
            f"g_L = "
            f"{g_L_value:.3f}"
        )


# ============================================================
# RANDOM CONTINUOUS BASELINE
# ============================================================

random_theta_errors = []
random_rho_errors = []
random_eig_sums = []
random_eig_per_step = []
random_g_L = []
random_decision_times = []

def random_continuous_beam(
    K,
    device,
    dtype=torch.float32,
):
    eta = torch.zeros(
        K,
        device=device,
        dtype=dtype,
    )

    eta[1:] = (
        2.0
        * math.pi
        * torch.rand(
            K - 1,
            device=device,
            dtype=dtype,
        )
    )

    return eta

eta_vec = random_continuous_beam(
    params.K,
    device,
)

idx = torch.randint(
    low=0,
    high=eta_grid.shape[1],
    size=(1,),
    device=device,
)

eta_vec = eta_grid[:, idx].squeeze()

with torch.inference_mode():

    for r in range(N_REAL):

        theta_true = theta_true_all[r]
        rho_true = rho_true_all[r]

        sigma = sigma_from_snr(
            s=s,
            snr_db=SNR_DB,
            rho_true=rho_true,
        )

        posterior = make_uniform_posterior(
            theta_grid,
            rho_grid,
        )

        eta_history = torch.empty(
            1,
            0,
            params.K,
            device=device,
        )

        r_history = torch.empty(
            1,
            0,
            Ns,
            device=device,
        )

        eig_sum = 0.0
        eig_steps = []

        for t in range(params.T):

            # ============================================
            # RANDOM BEAM
            # ============================================

            sync_cuda()

            tic = time.perf_counter()

            eta_vec = random_continuous_beam(
                params.K,
                device,
            )

            sync_cuda()

            random_decision_times.append(
                time.perf_counter() - tic
            )


            # ============================================
            # Diagnostic EIG
            # ============================================

            p_theta = marginal_theta(
                posterior
            )

            rho_mean = compute_rho_mean(
                posterior,
                rho_grid,
            )

            eig_t = estimate_eig_for_eta_marginal(
                eta=eta_vec[:, None],
                a_grid=a_grid,
                s=s,
                snr_db=SNR_DB,
                p_theta=p_theta,
                rho_mean=rho_mean,
                N=params.N,
            )

            eig_sum += eig_t.item()
            eig_steps.append(
                eig_t.item()
            )


            # ============================================
            # SAME PHYSICAL NOISE
            # ============================================

            amp_vec = (
                generate_measurement_fixed_noise(
                    theta=theta_true,
                    rho=rho_true,
                    eta=eta_vec,
                    s=s,
                    sigma=sigma,
                    params=params,
                    noise_real=noise_real_all[r, t],
                    noise_imag=noise_imag_all[r, t],
                )
            )


            # ============================================
            # SAME posterior estimator
            # ============================================

            posterior = update_posterior(
                eta=eta_vec,
                a_grid=a_grid,
                amp_vec=amp_vec,
                posterior=posterior,
                rho_grid=rho_grid,
                s=s,
                snr_db=SNR_DB,
            )


            # ============================================
            # Save history
            # ============================================

            eta_history = torch.cat(
                [
                    eta_history,
                    eta_vec.reshape(
                        1,
                        1,
                        params.K,
                    ),
                ],
                dim=1,
            )

            r_history = torch.cat(
                [
                    r_history,
                    amp_vec.reshape(
                        1,
                        1,
                        Ns,
                    ),
                ],
                dim=1,
            )


        # ================================================
        # SAME final estimator
        # ================================================

        theta_hat, rho_hat = (
            get_final_estimates(
                posterior,
                theta_grid,
                rho_grid,
            )
        )

        theta_error = torch.abs(
            torch.rad2deg(
                theta_hat
                - theta_true
            )
        )

        rho_error = torch.abs(
            rho_hat
            - rho_true
        )


        # ================================================
        # SAME g_L evaluator
        # ================================================

        g_L_value = evaluate_g_L(
            theta_true=theta_true,
            theta_contrast=theta_contrast_all[r],
            eta_history=eta_history,
            r_history=r_history,
            rho_grid=rho_grid,
            s=s,
            snr_db=SNR_DB,
            params=params,
        )


        random_theta_errors.append(
            theta_error.item()
        )

        random_rho_errors.append(
            rho_error.item()
        )

        random_eig_sums.append(
            eig_sum
        )

        random_eig_per_step.append(
            eig_steps
        )

        random_g_L.append(
            g_L_value
        )

# ============================================================
#
# PART 4
#
# SUMMARY
#
# ============================================================

print_summary(
    name="BASELINE EIG / NMC",
    theta_errors=(
        baseline_theta_errors
    ),
    rho_errors=(
        baseline_rho_errors
    ),
    eig_sums=(
        baseline_eig_sums
    ),
    eig_sums_recomputed=(
        baseline_eig_sums_recomputed
    ),
    g_L_values=(
        baseline_g_L
    ),
    decision_times=(
        baseline_decision_times
    ),
)


print_summary(
    name="DAD",
    theta_errors=(
        dad_theta_errors
    ),
    rho_errors=(
        dad_rho_errors
    ),
    eig_sums=(
        dad_eig_sums
    ),
    eig_sums_recomputed=(0.0,),
    g_L_values=(
        dad_g_L
    ),
    decision_times=(
        dad_decision_times
    ),
)

print_summary(
    name="RANDOM CONTINUOUS",
    theta_errors=random_theta_errors,
    rho_errors=random_rho_errors,
    eig_sums=random_eig_sums,
    eig_sums_recomputed=(0.0,),
    g_L_values=random_g_L,
    decision_times=random_decision_times,
)


# ============================================================
# Per-step EIG
# ============================================================

baseline_eig_per_step = torch.tensor(
    baseline_eig_per_step
)

dad_eig_per_step = torch.tensor(
    dad_eig_per_step
)


print()
print(
    "=" * 60
)
print(
    "MEAN EIG PER STEP"
)
print(
    "=" * 60
)

print(
    " t | baseline |    DAD"
)

print(
    "---+----------+----------"
)

for t in range(
    params.T
):

    print(
        f"{t + 1:2d} | "
        f"{baseline_eig_per_step[:, t].mean().item():8.4f} | "
        f"{dad_eig_per_step[:, t].mean().item():8.4f}"
    )


# ============================================================
# Speedup
# ============================================================

baseline_time_mean = (
    torch.tensor(
        baseline_decision_times
    ).mean()
)

dad_time_mean = (
    torch.tensor(
        dad_decision_times
    ).mean()
)


print()
print(
    "=" * 60
)
print(
    "ONLINE DECISION SPEED"
)
print(
    "=" * 60
)

print(
    f"Baseline : "
    f"{1e3 * baseline_time_mean.item():.4f} ms"
)

print(
    f"DAD      : "
    f"{1e3 * dad_time_mean.item():.4f} ms"
)

print(
    f"Speed-up : "
    f"{(baseline_time_mean / dad_time_mean).item():.1f} x"
)

# ============================================================
# Save comparison
# ============================================================

torch.save(
    {
        "snr_db":
            SNR_DB,

        "theta_true":
            theta_true_all.detach().cpu(),

        "rho_true":
            rho_true_all.detach().cpu(),

        "baseline_theta_errors":
            torch.tensor(
                baseline_theta_errors
            ),

        "dad_theta_errors":
            torch.tensor(
                dad_theta_errors
            ),

        "baseline_rho_errors":
            torch.tensor(
                baseline_rho_errors
            ),

        "dad_rho_errors":
            torch.tensor(
                dad_rho_errors
            ),

        "baseline_eig_per_step":
            baseline_eig_per_step,

        "dad_eig_per_step":
            dad_eig_per_step,

        "baseline_g_L":
            torch.tensor(
                baseline_g_L
            ),

        "dad_g_L":
            torch.tensor(
                dad_g_L
            ),

        "baseline_decision_times":
            torch.tensor(
                baseline_decision_times
            ),

        "dad_decision_times":
            torch.tensor(
                dad_decision_times
            ),
    },
    "comparison_nmc_vs_dad.pt",
)

print()
print(
    "Results saved to "
    "comparison_nmc_vs_dad.pt"
)