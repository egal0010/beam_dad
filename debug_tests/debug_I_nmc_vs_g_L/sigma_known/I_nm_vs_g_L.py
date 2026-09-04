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

from modules.beam_eig.simulator import (
    simulate_y,
    sigma_from_snr,
)

from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)

from modules.beam_eig.baseline_eig import (
    estimate_eig_for_eta,
)

from modules.dad.contrastive import (
    trajectory_log_likelihood,
    make_log_likelihood_fn,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_THETA = 241

BATCH_SIZE = 128
N_BATCHES = 20

# We will study convergence as L increases
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
    5000,
    10000,
]

MAX_L = max(L_VALUES)

RHO_KNOWN = 0.5
SNR_DB = 0.0


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
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Physical setup
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


# ============================================================
# DISCRETE theta prior
#
# IMPORTANT:
# EIG and g_L will use exactly the SAME prior.
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


# steering_vector_batch:
# [N_theta, K]
a_grid_batch = steering_vector_batch(
    theta_grid,
    params,
)

# Baseline convention:
# [K, N_theta]
a_grid = a_grid_batch.T


# ============================================================
# Fixed beam
#
# eta does NOT change from one realization to another.
# ============================================================

eta = torch.zeros(
    K,
    1,
    device=device,
    dtype=dtype,
)

b = beam_from_phases(
    eta,
)


# ============================================================
# Known rho
# ============================================================

rho_known = torch.tensor(
    RHO_KNOWN,
    device=device,
    dtype=dtype,
)


# ============================================================
# Fixed sigma
#
# Since rho_known itself is globally fixed here,
# computing sigma from it ONCE is fine.
#
# sigma does not vary between realizations.
# ============================================================

sigma = sigma_from_snr(
    s=s,
    snr_db=SNR_DB,
    rho_true=rho_known,
)

print("rho known :", rho_known.item())
print("sigma     :", sigma.item())


# ============================================================
# alpha(theta) for every theta of the discrete prior
# ============================================================

alpha_theta = (
    rho_known
    * (
        b.conj().T
        @ a_grid
    ).squeeze(0)
)

# [N_theta]


# ============================================================
# For the DAD likelihood:
#
# Known rho can simply be represented as a rho grid
# containing ONE point.
#
# sum_rho p(rho) ... then becomes exactly
# p(... | rho_known)
# ============================================================

rho_grid_known = rho_known.reshape(1)

log_p_rho_prior_known = torch.zeros(
    1,
    device=device,
    dtype=dtype,
)

log_likelihood_fn = make_log_likelihood_fn(
    rho_grid=rho_grid_known,
    s=s,
    sigma=sigma,
    params=params,
)


# ============================================================
# Storage
# ============================================================

all_eig_terms = []

all_g_terms = {
    L: []
    for L in L_VALUES
}


# ============================================================
# Monte-Carlo experiment
# ============================================================

print()
print("=" * 75)
print("T=1 | KNOWN rho | FIXED sigma | FIXED eta")
print("=" * 75)


with torch.inference_mode():

    for batch_idx in range(N_BATCHES):

        # ====================================================
        # 1. theta_true ~ EXACT SAME discrete prior
        #    used by EIG
        # ====================================================

        idx_true = torch.multinomial(
            p_theta,
            num_samples=BATCH_SIZE,
            replacement=True,
        )

        theta_true = theta_grid[
            idx_true
        ]

        # [B]


        # ====================================================
        # 2. Generate r ~ p(r | theta_true, rho, eta)
        # ====================================================

        alpha_true = alpha_theta[
            idx_true
        ]

        # [B]

        y = simulate_y(
            alpha_true,
            s,
            sigma,
        )

        amp = torch.abs(
            y
        )

        # [B, Ns]


        # ====================================================
        # 3. REFERENCE EIG
        #
        # Compute likelihood of EVERY generated observation
        # under EVERY possible theta from the prior.
        #
        # This denominator is exact w.r.t. the discrete
        # theta prior.
        # ====================================================

        log_like_all = (
            log_amplitude_vector_likelihood(
                amp[:, None, :],
                s,
                alpha_theta[None, :],
                sigma**2,
            )
        )

        # [B, N_theta]


        batch_indices = torch.arange(
            BATCH_SIZE,
            device=device,
        )

        # Numerator:
        #
        # log p(r | theta_true, eta)
        log_num = log_like_all[
            batch_indices,
            idx_true,
        ]

        # Denominator:
        #
        # log sum_theta
        # p(theta) p(r | theta, eta)
        log_den = torch.logsumexp(
            log_p_theta[None, :]
            + log_like_all,
            dim=1,
        )

        eig_terms = (
            log_num
            - log_den
        )

        all_eig_terms.append(
            eig_terms.cpu()
        )


        # ====================================================
        # 4. Build DAD history
        #
        # T = 1
        # ====================================================

        eta_history = (
            eta.squeeze(-1)
            .view(1, 1, K)
            .expand(
                BATCH_SIZE,
                1,
                K,
            )
        )

        r_history = amp.unsqueeze(1)

        # [B,1,Ns]


        # ====================================================
        # 5. Contrastive theta samples
        #
        # Again sampled from EXACT SAME p_theta.
        # ====================================================

        idx_contrast = torch.multinomial(
            p_theta,
            num_samples=(
                BATCH_SIZE
                * MAX_L
            ),
            replacement=True,
        ).reshape(
            BATCH_SIZE,
            MAX_L,
        )

        theta_contrast = theta_grid[
            idx_contrast
        ]

        theta_candidates = torch.cat(
            [
                theta_true.unsqueeze(1),
                theta_contrast,
            ],
            dim=1,
        )

        # [B, MAX_L + 1]


        # ====================================================
        # 6. Compute log p(r | theta_candidate)
        #
        # Since rho_grid contains only rho_known,
        # no actual nuisance marginalization remains.
        # ====================================================

        log_prob = trajectory_log_likelihood(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior_known,
        )

        # [B, MAX_L + 1]


        # ====================================================
        # 7. Evaluate g_L for many L using the SAME
        # observations and SAME nested contrastive samples
        # ====================================================

        for L in L_VALUES:

            log_prob_L = (
                log_prob[:, :L + 1]
            )

            log_prob_true = (
                log_prob_L[:, 0]
            )

            log_evidence_L = (
                torch.logsumexp(
                    log_prob_L,
                    dim=1,
                )
                - math.log(L + 1)
            )

            g_L = (
                log_prob_true
                - log_evidence_L
            )

            all_g_terms[L].append(
                g_L.cpu()
            )


        print(
            f"batch "
            f"{batch_idx + 1:3d}/"
            f"{N_BATCHES}"
        )


# ============================================================
# Concatenate samples
# ============================================================

eig_terms = torch.cat(
    all_eig_terms
)


# ============================================================
# Utility for standard error
# ============================================================

def mean_and_se(x):

    x = x.float()

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
# Reference EIG
# ============================================================

eig_mean, eig_se = mean_and_se(
    eig_terms
)


print()
print("=" * 75)
print("RESULTS")
print("=" * 75)

print(
    f"Reference EIG : "
    f"{eig_mean:.6f} "
    f"+/- {2 * eig_se:.6f} nats "
    f"(approx. 95% MC interval)"
)

print()

print(
    f"{'L':>8} | "
    f"{'E[g_L]':>12} | "
    f"{'2 SE':>10} | "
    f"{'gap to EIG':>12} | "
    f"{'log(L+1)':>10}"
)

print(
    "-" * 65
)


for L in L_VALUES:

    g = torch.cat(
        all_g_terms[L]
    )

    g_mean, g_se = mean_and_se(
        g
    )

    gap = (
        eig_mean
        - g_mean
    )

    print(
        f"{L:8d} | "
        f"{g_mean:12.6f} | "
        f"{2 * g_se:10.6f} | "
        f"{gap:12.6f} | "
        f"{math.log(L + 1):10.6f}"
    )


# ============================================================
# Optional:
# validate your EXISTING estimate_eig_for_eta()
# ============================================================

print()
print("=" * 75)
print("CHECK EXISTING estimate_eig_for_eta()")
print("=" * 75)

N_EIG = 5000
N_EIG_REPEATS = 10

old_eig_values = []

with torch.inference_mode():

    for i in range(N_EIG_REPEATS):

        eig_old = estimate_eig_for_eta(
            eta=eta,
            a_grid=a_grid,
            s=s,
            sigma=sigma,
            p_theta=p_theta,
            rho_mean=rho_known,
            N=N_EIG,
        )

        old_eig_values.append(
            eig_old.item()
        )

old_eig_values = torch.tensor(
    old_eig_values
)

print(
    "Existing estimator mean : "
    f"{old_eig_values.mean().item():.6f} nats"
)

print(
    "Reference EIG           : "
    f"{eig_mean:.6f} nats"
)

difference = (
    old_eig_values.mean().item()
    - eig_mean
)

print(
    f"Difference              : "
    f"{difference:.6f} nats"
)