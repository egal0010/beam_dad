import math

import torch

from modules.beam_eig.params import Params

from modules.beam_eig.pilot import (
    generate_pilot_sequence,
)

from modules.beam_eig.array_model import (
    beam_from_phases,
    steering_vector_batch,
)

from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)

from modules.dad.contrastive import (
    contrastive_bound_from_log_prob,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_THETA = 241
N_RHO = 50

BATCH_SIZE = 16
N_BATCHES = 50

SNR_DB = 0.0

# For model A:
SIGMA_FIXED = 0.5

L_VALUES = [
    1,
    5,
    10,
    30,
    70,
    128,
    300,
    1000,
    3000,
]

MAX_L = max(L_VALUES)

T_VALUES = [1, 2, 3, 5, 10]
T_MAX = max(T_VALUES)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dtype = torch.float32

print("Device:", device)


# ============================================================
# Seed
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Physical model
# ============================================================

params = Params(
    Nx=4,
    Ny=1,
)

K = params.K


# ============================================================
# Pilot
# ============================================================

s = generate_pilot_sequence(
    device=device,
    sequence_type="PSS",
).to(dtype=dtype)

Ns = s.numel()

Es = torch.mean(
    torch.abs(s) ** 2
)

snr_lin = 10.0 ** (
    SNR_DB / 10.0
)

# sigma(rho) = c * rho
c_sigma = torch.sqrt(
    Es / snr_lin
)

print("Es       :", Es.item())
print("SNR      :", SNR_DB, "dB")
print("c_sigma  :", c_sigma.item())


# ============================================================
# theta prior
# ============================================================

theta_grid = torch.linspace(
    math.radians(30.0),
    math.radians(150.0),
    N_THETA,
    device=device,
    dtype=dtype,
)

p_theta = torch.full(
    (N_THETA,),
    1.0 / N_THETA,
    device=device,
    dtype=dtype,
)

log_p_theta = torch.log(
    p_theta
)


# ============================================================
# rho prior
# ============================================================

rho_grid = torch.linspace(
    0.05,
    1.0,
    N_RHO,
    device=device,
    dtype=dtype,
)

p_rho = torch.full(
    (N_RHO,),
    1.0 / N_RHO,
    device=device,
    dtype=dtype,
)

log_p_rho = torch.log(
    p_rho
)

# ============================================================
# Mean rho used by the plug-in approximation
# ============================================================

rho_mean = torch.sum(
    p_rho * rho_grid
)

print(
    "rho_mean  :",
    rho_mean.item(),
)

# ============================================================
# Fixed beam
# ============================================================

eta = torch.zeros(
    K,
    1,
    device=device,
    dtype=dtype,
)

b = beam_from_phases(
    eta
)


# ============================================================
# Array responses
# ============================================================

a_grid_batch = steering_vector_batch(
    theta_grid,
    params,
)

# [N_theta, K]

a_grid = a_grid_batch.T

# [K, N_theta]

array_response_theta = (
    b.conj().T
    @ a_grid
).squeeze(0)

# [N_theta]


# ============================================================
# Local simulator
#
# This makes the sigma broadcasting explicit.
# ============================================================

def simulate_amplitude(
    alpha,
    s,
    sigma,
):

    mu = (
        alpha[..., None]
        * s
    )

    sigma = torch.as_tensor(
        sigma,
        device=mu.device,
        dtype=mu.real.dtype,
    )

    # sigma scalar -> OK
    #
    # sigma [B] -> [B,1]
    while sigma.ndim < mu.ndim:
        sigma = sigma.unsqueeze(-1)

    noise_real = torch.randn_like(
        mu.real
    )

    noise_imag = torch.randn_like(
        mu.real
    )

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


# ============================================================
# Likelihood helper
# ============================================================

def log_like_for_theta(
    amp,
    alpha_theta,
    sigma,
):
    """
    amp         : [B,Ns]
    alpha_theta : [B,Ntheta] or [1,Ntheta]
    sigma       : scalar or [B]

    Returns
    -------
    [B,Ntheta]
    """

    sigma = torch.as_tensor(
        sigma,
        device=amp.device,
        dtype=amp.dtype,
    )

    sigma2 = sigma**2

    # If sigma differs per realization,
    # reshape for correct broadcasting:
    #
    # [B] -> [B,1,1]
    if sigma2.ndim == 1:
        sigma2 = sigma2[
            :,
            None,
            None,
        ]

    return (
        log_amplitude_vector_likelihood(
            amp[:, None, :],
            s,
            alpha_theta,
            sigma2,
        )
    )


# ============================================================
# Model A
#
# Fixed sigma:
#
# p(r|theta)
# =
# sum_rho p(rho)
# p(r|theta,rho,sigma_fixed)
# ============================================================

def marginal_fixed_noise(
    amp,
):

    terms = []

    sigma_fixed = torch.tensor(
        SIGMA_FIXED,
        device=device,
        dtype=dtype,
    )

    for j in range(N_RHO):

        rho_j = rho_grid[j]

        alpha_theta_j = (
            rho_j
            * array_response_theta
        )[None, :]

        log_like_j = log_like_for_theta(
            amp,
            alpha_theta_j,
            sigma_fixed,
        )

        terms.append(
            log_p_rho[j]
            + log_like_j
        )

    terms = torch.stack(
        terms,
        dim=-1,
    )

    # [B,Ntheta,Nrho]

    return torch.logsumexp(
        terms,
        dim=-1,
    )


# ============================================================
# Model B
#
# Constant SNR, coherent model:
#
# sigma_j = c * rho_j
#
# p(r|theta)
# =
# sum_rho p(rho)
# p(r|theta,rho,sigma(rho))
# ============================================================

def marginal_fixed_snr_correct(
    amp,
):

    terms = []

    for j in range(N_RHO):

        rho_j = rho_grid[j]

        sigma_j = (
            c_sigma
            * rho_j
        )

        alpha_theta_j = (
            rho_j
            * array_response_theta
        )[None, :]

        log_like_j = log_like_for_theta(
            amp,
            alpha_theta_j,
            sigma_j,
        )

        terms.append(
            log_p_rho[j]
            + log_like_j
        )

    terms = torch.stack(
        terms,
        dim=-1,
    )

    return torch.logsumexp(
        terms,
        dim=-1,
    )


# ============================================================
# Model C
#
# CURRENT IMPLEMENTATION:
#
# data generated with
#
# sigma_true = c * rho_true
#
# but EVERY rho candidate is tested with
#
# sigma = sigma_true
#
# p_current(r|theta)
# =
# sum_rho p(rho)
# p(r|theta,rho,sigma_true)
#
# This is NOT the generating marginal.
# ============================================================

def marginal_current_model(
    amp,
    sigma_true,
):

    terms = []

    for j in range(N_RHO):

        rho_j = rho_grid[j]

        alpha_theta_j = (
            rho_j
            * array_response_theta
        )[None, :]

        log_like_j = log_like_for_theta(
            amp,
            alpha_theta_j,
            sigma_true,
        )

        terms.append(
            log_p_rho[j]
            + log_like_j
        )

    terms = torch.stack(
        terms,
        dim=-1,
    )

    return torch.logsumexp(
        terms,
        dim=-1,
    )


# ============================================================
# Model D
#
# sigma_true is observed.
#
# Since sigma = c*rho,
# rho is then effectively known.
#
# p(r|theta,rho_true)
# ============================================================

def log_like_rho_known(
    amp,
    rho_true,
    sigma_true,
):

    alpha_theta = (
        rho_true[:, None]
        * array_response_theta[
            None, :
        ]
    )

    return log_like_for_theta(
        amp,
        alpha_theta,
        sigma_true,
    )


# ============================================================
# Joint-grid likelihood helpers for trajectories
# ============================================================

def log_like_joint_theta_rho(amp, sigma):
    """Evaluate log p(r | theta_m, rho_j, sigma) on the joint grid."""

    alpha_theta_rho = (
        array_response_theta[None, :, None]
        * rho_grid[None, None, :]
    )
    sigma2 = (sigma**2)[:, None, None, None]

    return log_amplitude_vector_likelihood(
        amp[:, None, None, :],
        s,
        alpha_theta_rho,
        sigma2,
    )


def log_like_joint_fixed_snr(amp):
    """Evaluate log p(r | theta_m, rho_j, sigma_j=c*rho_j)."""

    alpha_theta_rho = (
        array_response_theta[None, :, None]
        * rho_grid[None, None, :]
    )
    sigma2 = (
        (c_sigma * rho_grid)**2
    )[None, None, :, None]

    return log_amplitude_vector_likelihood(
        amp[:, None, None, :],
        s,
        alpha_theta_rho,
        sigma2,
    )


def rho_mean_from_joint_posterior(log_posterior):
    """Return E[rho | h] from a normalized joint log posterior."""

    log_p_rho_post = torch.logsumexp(
        log_posterior,
        dim=1,
    )
    p_rho_post = torch.exp(log_p_rho_post)

    return torch.sum(
        p_rho_post * rho_grid[None, :],
        dim=1,
    )


def log_like_rho_mean_dynamic(amp, rho_mean_batch, sigma):
    """Evaluate F with the rho posterior mean preceding this observation."""

    alpha_theta = (
        rho_mean_batch[:, None]
        * array_response_theta[None, :]
    )

    return log_like_for_theta(
        amp,
        alpha_theta,
        sigma,
    )


# ============================================================
# Model E
#
# rho_mean plug-in + FIXED noise
#
# Data are generated with:
#
# rho_true ~ p(rho)
# sigma = sigma_fixed
#
# but likelihood assumes:
#
# rho = E[rho]
#
# p_plugin(r | theta)
# =
# p(r | theta, rho_mean, sigma_fixed)
# ============================================================

def log_like_rho_mean_fixed_noise(
    amp,
):

    sigma_fixed = torch.tensor(
        SIGMA_FIXED,
        device=device,
        dtype=dtype,
    )

    alpha_theta = (
        rho_mean
        * array_response_theta
    )[None, :]

    return log_like_for_theta(
        amp,
        alpha_theta,
        sigma_fixed,
    )


# ============================================================
# Model F
#
# rho_mean plug-in + sigma_true
#
# This is the closest version to the OLD Amelia setup:
#
# data:
#   rho_true ~ p(rho)
#   sigma_true = c * rho_true
#
# likelihood:
#   rho -> rho_mean
#   sigma -> sigma_true
#
# p_plugin(r | theta)
# =
# p(r | theta, rho_mean, sigma_true)
#
# IMPORTANT:
# This is a mismatched / plug-in model.
# ============================================================

def log_like_rho_mean_current(
    amp,
    sigma_true,
):

    alpha_theta = (
        rho_mean
        * array_response_theta
    )[None, :]

    return log_like_for_theta(
        amp,
        alpha_theta,
        sigma_true,
    )

# ============================================================
# Information terms
# ============================================================

def information_terms(
    log_like_theta,
    idx_theta_true,
):

    B = log_like_theta.shape[0]

    batch_idx = torch.arange(
        B,
        device=device,
    )

    log_num = log_like_theta[
        batch_idx,
        idx_theta_true,
    ]

    log_den = torch.logsumexp(
        log_p_theta[None, :]
        + log_like_theta,
        dim=1,
    )

    return (
        log_num
        - log_den
    )


# ============================================================
# Contrastive g_L from an already computed
# log p(r|theta) table
# ============================================================

def compute_g_values(
    log_like_theta,
    idx_theta_true,
    idx_theta_contrast,
):

    idx_candidates = torch.cat(
        [
            idx_theta_true[:, None],
            idx_theta_contrast,
        ],
        dim=1,
    )

    candidate_logs = torch.gather(
        log_like_theta,
        dim=1,
        index=idx_candidates,
    )

    results = {}

    for L in L_VALUES:

        logs_L = (
            candidate_logs[:, :L + 1]
        )

        _, results[L] = contrastive_bound_from_log_prob(
            logs_L
        )

    return results


# ============================================================
# Statistics
# ============================================================

def mean_and_se(x):

    x = torch.cat(x).float()

    mean = x.mean()

    se = (
        x.std(unbiased=True)
        / math.sqrt(x.numel())
    )

    return (
        mean.item(),
        se.item(),
    )


# ============================================================
# Storage
# ============================================================

fixed_noise_info = []

fixed_snr_info = []

current_pseudo_info = []

rho_known_info = []

rho_mean_fixed_noise_info = []

rho_mean_current_info = []

rho_mean_fixed_noise_consistent_info = []

rho_mean_amelia_consistent_info = []

g_fixed_noise = {
    L: []
    for L in L_VALUES
}

g_fixed_snr = {
    L: []
    for L in L_VALUES
}

g_current = {
    L: []
    for L in L_VALUES
}

g_rho_known = {
    L: []
    for L in L_VALUES
}

g_rho_mean_fixed_noise = {
    L: []
    for L in L_VALUES
}

g_rho_mean_current = {
    L: []
    for L in L_VALUES
}

g_rho_mean_fixed_noise_consistent = {
    L: []
    for L in L_VALUES
}

g_rho_mean_amelia_consistent = {
    L: []
    for L in L_VALUES
}

TRAJECTORY_MODELS = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
)

g_trajectory = {
    model: {
        T: {L: [] for L in L_VALUES}
        for T in T_VALUES
    }
    for model in TRAJECTORY_MODELS
}

summed_g_trajectory = {
    model: {
        T: {L: [] for L in L_VALUES}
        for T in T_VALUES
    }
    for model in TRAJECTORY_MODELS
}

pseudo_trajectory = {
    model: {
        T: []
        for T in T_VALUES
    }
    for model in TRAJECTORY_MODELS
}

# ============================================================
# Experiment
# ============================================================

print()
print("=" * 80)
print("COMPARING NOISE / RHO MODELS")
print("=" * 80)


with torch.inference_mode():

    for batch_i in range(N_BATCHES):

        # ----------------------------------------------------
        # Same latent draws
        # ----------------------------------------------------

        idx_theta_true = torch.multinomial(
            p_theta,
            BATCH_SIZE,
            replacement=True,
        )

        theta_true = theta_grid[
            idx_theta_true
        ]

        idx_rho_true = torch.multinomial(
            p_rho,
            BATCH_SIZE,
            replacement=True,
        )

        rho_true = rho_grid[
            idx_rho_true
        ]

        array_true = array_response_theta[
            idx_theta_true
        ]

        alpha_true = (
            rho_true
            * array_true
        )


        # ----------------------------------------------------
        # Same contrastive theta indices
        # ----------------------------------------------------

        idx_theta_contrast = torch.multinomial(
            p_theta,
            BATCH_SIZE * MAX_L,
            replacement=True,
        ).reshape(
            BATCH_SIZE,
            MAX_L,
        )


        # ====================================================
        # A. FIXED-NOISE experiment
        # ====================================================

        sigma_fixed = torch.tensor(
            SIGMA_FIXED,
            device=device,
            dtype=dtype,
        )

        amp_fixed_noise = simulate_amplitude(
            alpha_true,
            s,
            sigma_fixed,
        )

        log_like_A = marginal_fixed_noise(
            amp_fixed_noise,
        )

        terms_A = information_terms(
            log_like_A,
            idx_theta_true,
        )

        fixed_noise_info.append(
            terms_A.cpu()
        )

        g_A = compute_g_values(
            log_like_A,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_fixed_noise[L].append(
                g_A[L].cpu()
            )


        # ====================================================
        # E. rho_mean plug-in + FIXED noise
        #
        # Same observations as model A.
        #
        # Generation:
        #   rho_true random
        #   sigma fixed
        #
        # Scoring:
        #   rho = rho_mean
        # ====================================================

        log_like_E = (
            log_like_rho_mean_fixed_noise(
                amp_fixed_noise,
            )
        )

        terms_E = information_terms(
            log_like_E,
            idx_theta_true,
        )

        rho_mean_fixed_noise_info.append(
            terms_E.cpu()
        )

        g_E = compute_g_values(
            log_like_E,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_rho_mean_fixed_noise[L].append(
                g_E[L].cpu()
            )

        # ====================================================
        # B/C/D use SAME fixed-SNR-generated observations
        # ====================================================

        sigma_true = (
            c_sigma
            * rho_true
        )

        # ====================================================
        # H. Amelia plug-in model
        #    CONSISTENT with estimate_eig_for_eta()
        #
        # sigma comes from the outer realization,
        # but BOTH generation and likelihood use rho_mean.
        # ====================================================

        alpha_true_rho_mean = (
            rho_mean
            * array_true
        )

        amp_amelia = simulate_amplitude(
            alpha_true_rho_mean,
            s,
            sigma_true,
        )

        log_like_H = (
            log_like_rho_mean_current(
                amp_amelia,
                sigma_true,
            )
        )

        terms_H = information_terms(
            log_like_H,
            idx_theta_true,
        )

        rho_mean_amelia_consistent_info.append(
            terms_H.cpu()
        )

        g_H = compute_g_values(
            log_like_H,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_rho_mean_amelia_consistent[L].append(
                g_H[L].cpu()
            )

        amp_fixed_snr = simulate_amplitude(
            alpha_true,
            s,
            sigma_true,
        )


        # ====================================================
        # G. rho_mean + FIXED noise
        #    CONSISTENT plug-in model
        #
        # Both generation AND likelihood use rho_mean.
        # ====================================================

        alpha_true_rho_mean = (
            rho_mean
            * array_true
        )

        amp_rho_mean_fixed = simulate_amplitude(
            alpha_true_rho_mean,
            s,
            sigma_fixed,
        )

        log_like_G = (
            log_like_rho_mean_fixed_noise(
                amp_rho_mean_fixed,
            )
        )

        terms_G = information_terms(
            log_like_G,
            idx_theta_true,
        )

        rho_mean_fixed_noise_consistent_info.append(
            terms_G.cpu()
        )

        g_G = compute_g_values(
            log_like_G,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_rho_mean_fixed_noise_consistent[L].append(
                g_G[L].cpu()
            )

        # ====================================================
        # B. Correct constant-SNR marginal
        # ====================================================

        log_like_B = (
            marginal_fixed_snr_correct(
                amp_fixed_snr,
            )
        )

        terms_B = information_terms(
            log_like_B,
            idx_theta_true,
        )

        fixed_snr_info.append(
            terms_B.cpu()
        )

        g_B = compute_g_values(
            log_like_B,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_fixed_snr[L].append(
                g_B[L].cpu()
            )


        # ====================================================
        # C. Current mismatched model
        # ====================================================

        log_like_C = (
            marginal_current_model(
                amp_fixed_snr,
                sigma_true,
            )
        )

        terms_C = information_terms(
            log_like_C,
            idx_theta_true,
        )

        current_pseudo_info.append(
            terms_C.cpu()
        )

        g_C = compute_g_values(
            log_like_C,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_current[L].append(
                g_C[L].cpu()
            )

        # ====================================================
        # F. rho_mean plug-in + sigma_true
        #
        # Closest case to the OLD Amelia implementation.
        #
        # Generation:
        #   rho_true random
        #   sigma_true = c * rho_true
        #
        # Scoring:
        #   rho = rho_mean
        #   sigma = sigma_true
        # ====================================================

        log_like_F = (
            log_like_rho_mean_current(
                amp_fixed_snr,
                sigma_true,
            )
        )

        terms_F = information_terms(
            log_like_F,
            idx_theta_true,
        )

        rho_mean_current_info.append(
            terms_F.cpu()
        )

        g_F = compute_g_values(
            log_like_F,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_rho_mean_current[L].append(
                g_F[L].cpu()
            )

        # ====================================================
        # D. sigma observed -> rho known
        # ====================================================

        log_like_D = log_like_rho_known(
            amp_fixed_snr,
            rho_true,
            sigma_true,
        )

        terms_D = information_terms(
            log_like_D,
            idx_theta_true,
        )

        rho_known_info.append(
            terms_D.cpu()
        )

        g_D = compute_g_values(
            log_like_D,
            idx_theta_true,
            idx_theta_contrast,
        )

        for L in L_VALUES:

            g_rho_known[L].append(
                g_D[L].cpu()
            )


        # ====================================================
        # A-H trajectories with a fixed beam
        #
        # rho is static over a trajectory. For A/B/C, likelihoods
        # are accumulated conditionally on rho and marginalized
        # only afterwards.
        # ====================================================

        cumulative_joint_A = torch.zeros(
            BATCH_SIZE,
            N_THETA,
            N_RHO,
            device=device,
            dtype=dtype,
        )
        cumulative_joint_B = torch.zeros_like(cumulative_joint_A)
        cumulative_joint_C = torch.zeros_like(cumulative_joint_A)

        cumulative_D = torch.zeros(
            BATCH_SIZE,
            N_THETA,
            device=device,
            dtype=dtype,
        )
        cumulative_E = torch.zeros_like(cumulative_D)
        cumulative_F = torch.zeros_like(cumulative_D)
        cumulative_G = torch.zeros_like(cumulative_D)
        cumulative_H = torch.zeros_like(cumulative_D)

        log_posterior_F = (
            log_p_theta[None, :, None]
            + log_p_rho[None, None, :]
        ).expand(
            BATCH_SIZE,
            N_THETA,
            N_RHO,
        ).clone()

        previous_log_history = {
            model: torch.zeros_like(cumulative_D)
            for model in TRAJECTORY_MODELS
        }
        running_g_sum = {
            model: {
                L: torch.zeros(
                    BATCH_SIZE,
                    device=device,
                    dtype=dtype,
                )
                for L in L_VALUES
            }
            for model in TRAJECTORY_MODELS
        }

        for t in range(1, T_MAX + 1):

            # A/E share data generated with rho_true and fixed sigma.
            amp_A_t = simulate_amplitude(
                alpha_true,
                s,
                sigma_fixed,
            )

            # B/C/D/F share data generated at fixed SNR.
            amp_B_t = simulate_amplitude(
                alpha_true,
                s,
                sigma_true,
            )

            # G is the consistent fixed-noise plug-in model.
            amp_G_t = simulate_amplitude(
                alpha_true_rho_mean,
                s,
                sigma_fixed,
            )

            # H is the consistent Amelia plug-in model.
            amp_H_t = simulate_amplitude(
                alpha_true_rho_mean,
                s,
                sigma_true,
            )

            # F uses E[rho | h_(t-1)] before observing r_t.
            rho_mean_F_t = rho_mean_from_joint_posterior(
                log_posterior_F
            )

            joint_A_t = log_like_joint_theta_rho(
                amp_A_t,
                sigma_fixed.expand(BATCH_SIZE),
            )
            joint_B_t = log_like_joint_fixed_snr(
                amp_B_t
            )
            joint_C_t = log_like_joint_theta_rho(
                amp_B_t,
                sigma_true,
            )

            cumulative_joint_A += joint_A_t
            cumulative_joint_B += joint_B_t
            cumulative_joint_C += joint_C_t

            cumulative_D += log_like_rho_known(
                amp_B_t,
                rho_true,
                sigma_true,
            )
            cumulative_E += log_like_rho_mean_fixed_noise(
                amp_A_t
            )
            cumulative_F += log_like_rho_mean_dynamic(
                amp_B_t,
                rho_mean_F_t,
                sigma_true,
            )
            cumulative_G += log_like_rho_mean_fixed_noise(
                amp_G_t
            )
            cumulative_H += log_like_rho_mean_current(
                amp_H_t,
                sigma_true,
            )

            log_history_by_model = {
                "A": torch.logsumexp(
                    log_p_rho[None, None, :]
                    + cumulative_joint_A,
                    dim=-1,
                ),
                "B": torch.logsumexp(
                    log_p_rho[None, None, :]
                    + cumulative_joint_B,
                    dim=-1,
                ),
                "C": torch.logsumexp(
                    log_p_rho[None, None, :]
                    + cumulative_joint_C,
                    dim=-1,
                ),
                "D": cumulative_D,
                "E": cumulative_E,
                "F": cumulative_F,
                "G": cumulative_G,
                "H": cumulative_H,
            }

            for model, log_history_theta in (
                log_history_by_model.items()
            ):
                # Conditional predictive log likelihood for experiment t:
                # log q(r_t | theta, h_(t-1)).
                step_log_prob = (
                    log_history_theta
                    - previous_log_history[model]
                )
                step_g_values = compute_g_values(
                    step_log_prob,
                    idx_theta_true,
                    idx_theta_contrast,
                )

                for L in L_VALUES:
                    running_g_sum[model][L] += (
                        step_g_values[L]
                    )

                if t in T_VALUES:
                    trajectory_g_values = compute_g_values(
                        log_history_theta,
                        idx_theta_true,
                        idx_theta_contrast,
                    )

                    for L in L_VALUES:
                        g_trajectory[model][t][L].append(
                            trajectory_g_values[L].cpu()
                        )
                        summed_g_trajectory[model][t][L].append(
                            running_g_sum[model][L].cpu().clone()
                        )

                    pseudo_trajectory[model][t].append(
                        information_terms(
                            log_history_theta,
                            idx_theta_true,
                        ).cpu()
                    )

                # Clone because D/E/F/G/H histories are in-place
                # accumulators that will be mutated at the next step.
                previous_log_history[model] = (
                    log_history_theta.clone()
                )

            # F's posterior is updated only after r_t was scored.
            log_posterior_F += joint_C_t
            log_norm_F = torch.logsumexp(
                log_posterior_F.reshape(BATCH_SIZE, -1),
                dim=1,
            )
            log_posterior_F -= log_norm_F[:, None, None]


        print(
            f"batch "
            f"{batch_i + 1:3d}/"
            f"{N_BATCHES}"
        )


# ============================================================
# Summary
# ============================================================

A_mean, A_se = mean_and_se(
    fixed_noise_info
)

B_mean, B_se = mean_and_se(
    fixed_snr_info
)

C_mean, C_se = mean_and_se(
    current_pseudo_info
)

D_mean, D_se = mean_and_se(
    rho_known_info
)

E_mean, E_se = mean_and_se(
    rho_mean_fixed_noise_info
)

F_mean, F_se = mean_and_se(
    rho_mean_current_info
)

G_mean, G_se = mean_and_se(
    rho_mean_fixed_noise_consistent_info
)

H_mean, H_se = mean_and_se(
    rho_mean_amelia_consistent_info
)

print()
print("=" * 80)
print("INFORMATION SCORES")
print("=" * 80)

print(
    f"A - fixed noise, correct EIG       : "
    f"{A_mean:.6f} +/- {2*A_se:.6f}"
)

print(
    f"B - fixed SNR, correct EIG         : "
    f"{B_mean:.6f} +/- {2*B_se:.6f}"
)

print(
    f"C - CURRENT hybrid pseudo-objective: "
    f"{C_mean:.6f} +/- {2*C_se:.6f}"
)

print(
    f"D - sigma observed / rho known EIG : "
    f"{D_mean:.6f} +/- {2*D_se:.6f}"
)

print(
    f"E - rho_mean + fixed noise pseudo   : "
    f"{E_mean:.6f} +/- {2*E_se:.6f}"
)

print(
    f"F - rho_mean + sigma_true (Amelia)  : "
    f"{F_mean:.6f} +/- {2*F_se:.6f}"
)

# ============================================================
# g_L convergence
# ============================================================

print()
print("=" * 80)
print("g_L CONVERGENCE")
print("=" * 80)

print(
    f"{'L':>7} | "
    f"{'A fixed':>11} | "
    f"{'B SNR':>11} | "
    f"{'C current':>11} | "
    f"{'D known':>11} | "
    f"{'E mean/fix':>11} | "
    f"{'F Amelia':>11} | "
    f"{'G fix correct':>11} | "
    f"{'H Amelia':>11}"     
)

print("-" * 95)


for L in L_VALUES:

    A_g, _ = mean_and_se(
        g_fixed_noise[L]
    )

    B_g, _ = mean_and_se(
        g_fixed_snr[L]
    )

    C_g, _ = mean_and_se(
        g_current[L]
    )

    D_g, _ = mean_and_se(
        g_rho_known[L]
    )

    E_g, _ = mean_and_se(
        g_rho_mean_fixed_noise[L]
    )

    F_g, _ = mean_and_se(
        g_rho_mean_current[L]
    )

    G_g, _ = mean_and_se(
    g_rho_mean_fixed_noise_consistent[L]
)

    H_g, _ = mean_and_se(
    g_rho_mean_amelia_consistent[L]
    )

    print(
        f"{L:7d} | "
        f"{A_g:11.6f} | "
        f"{B_g:11.6f} | "
        f"{C_g:11.6f} | "
        f"{D_g:11.6f} | "
        f"{E_g:11.6f} | "
        f"{F_g:11.6f} | "
        f"{G_g:11.6f} | "
        f"{H_g:11.6f}"
    )


# ============================================================
# Useful gaps
# ============================================================

print()
print("=" * 80)
print("KEY DIFFERENCES")
print("=" * 80)

print(
    "Current - correct fixed-SNR : "
    f"{C_mean - B_mean:+.6f} nats"
)

print(
    "rho-known - fixed-SNR       : "
    f"{D_mean - B_mean:+.6f} nats"
)

print(
    "fixed-noise - fixed-SNR     : "
    f"{A_mean - B_mean:+.6f} nats"
)

print(
    "rho_mean/fixed - exact fixed noise : "
    f"{E_mean - A_mean:+.6f} nats"
)

print(
    "rho_mean Amelia - current marginal : "
    f"{F_mean - C_mean:+.6f} nats"
)

print(
    "rho_mean Amelia - correct fixed SNR: "
    f"{F_mean - B_mean:+.6f} nats"
)

print()
print("=" * 80)
print("rho_mean COMPARISON")
print("=" * 80)

print(
    f"{'L':>7} | "
    f"{'E mismatch':>12} | "
    f"{'G fix correct':>14} | "
    f"{'F mismatch':>12} | "
    f"{'H Amelia':>12}"
)

print("-" * 70)

for L in L_VALUES:

    E_g, _ = mean_and_se(
        g_rho_mean_fixed_noise[L]
    )

    G_g, _ = mean_and_se(
        g_rho_mean_fixed_noise_consistent[L]
    )

    F_g, _ = mean_and_se(
        g_rho_mean_current[L]
    )

    H_g, _ = mean_and_se(
        g_rho_mean_amelia_consistent[L]
    )

    print(
        f"{L:7d} | "
        f"{E_g:12.6f} | "
        f"{G_g:14.6f} | "
        f"{F_g:12.6f} | "
        f"{H_g:12.6f}"
    )

# ============================================================
# A-H trajectory convergence
# ============================================================

print()
print("=" * 120)
print("A-H FIXED-BEAM TRAJECTORIES: pseudo limit")
print("=" * 120)

print(
    f"{'model':>7} | "
    + " | ".join(
        f"T={T:>2}"
        for T in T_VALUES
    )
)
print("-" * 70)

for model in TRAJECTORY_MODELS:
    row = f"{model:>7}"

    for T in T_VALUES:
        pseudo_mean, _ = mean_and_se(
            pseudo_trajectory[model][T]
        )
        row += f" | {pseudo_mean:8.3f}"

    print(row)


for T in T_VALUES:
    print()
    print("=" * 120)
    print(f"A-H FIXED-BEAM TRAJECTORIES: g_L at T={T}")
    print("=" * 120)

    print(
        f"{'model':>7} | "
        + " | ".join(
            f"L={L:>4}"
            for L in L_VALUES
        )
    )
    print("-" * 120)

    for model in TRAJECTORY_MODELS:
        row = f"{model:>7}"

        for L in L_VALUES:
            g_mean, _ = mean_and_se(
                g_trajectory[model][T][L]
            )
            row += f" | {g_mean:8.3f}"

        print(row)


for T in T_VALUES:
    print()
    print("=" * 120)
    print(f"A-H SUM OF PER-STEP g_L THROUGH T={T}")
    print("=" * 120)

    print(
        f"{'model':>7} | "
        + " | ".join(
            f"L={L:>4}"
            for L in L_VALUES
        )
    )
    print("-" * 120)

    for model in TRAJECTORY_MODELS:
        row = f"{model:>7}"

        for L in L_VALUES:
            summed_g_mean, _ = mean_and_se(
                summed_g_trajectory[model][T][L]
            )
            row += f" | {summed_g_mean:8.3f}"

        print(row)
