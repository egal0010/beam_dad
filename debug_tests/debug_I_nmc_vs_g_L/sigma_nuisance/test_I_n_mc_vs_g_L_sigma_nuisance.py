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
)

from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)

from modules.dad.contrastive import (
    trajectory_log_likelihood,
    make_log_likelihood_fn,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

# theta prior
N_THETA = 241

# rho prior
N_RHO = 50

# Monte-Carlo outer expectation
#
# Keep B relatively small because the contrastive likelihood
# scales approximately as:
#
# B * (L+1) * N_RHO * Ns
#
BATCH_SIZE = 16
N_BATCHES = 100

# Convergence of the contrastive bound
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

# IMPORTANT:
#
# sigma is FIXED and does not depend on rho_true.
SIGMA = 0.5


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
).to(
    dtype=dtype,
)

Ns = s.numel()

print("N antennas :", K)
print("Ns         :", Ns)


# ============================================================
# Discrete theta prior
#
# IMPORTANT:
# Both the reference EIG and g_L use EXACTLY this prior.
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
# Discrete rho prior
#
# Again:
# generation AND marginalization use exactly the same prior.
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

# Same thing under the name expected by
# trajectory_log_likelihood()
log_p_rho_prior = log_p_rho


# ============================================================
# Fixed sigma
# ============================================================

sigma = torch.tensor(
    SIGMA,
    device=device,
    dtype=dtype,
)

print("sigma      :", sigma.item())
print(
    "rho range  :",
    rho_grid[0].item(),
    "->",
    rho_grid[-1].item(),
)


# ============================================================
# Fixed beam
#
# eta is fixed for ALL realizations.
# There is no policy/adaptation in this experiment.
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

# [K,1]


# ============================================================
# Steering vectors for every theta
# ============================================================

a_grid_batch = steering_vector_batch(
    theta_grid,
    params,
)

# expected [N_THETA, K]

a_grid = a_grid_batch.T

# expected [K, N_THETA]


# ============================================================
# Array response for every theta
#
# b^H a(theta)
# ============================================================

array_response_theta = (
    b.conj().T
    @ a_grid
).squeeze(0)

# [N_THETA]


# ============================================================
# alpha(theta, rho)
#
# alpha_mj = rho_j * b^H a(theta_m)
# ============================================================

alpha_theta_rho = (
    array_response_theta[:, None]
    * rho_grid[None, :]
)

# [N_THETA, N_RHO]


# ============================================================
# Likelihood function used by the DAD / sPCE side
# ============================================================

log_likelihood_fn = make_log_likelihood_fn(
    rho_grid=rho_grid,
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

likelihood_consistency_errors = []


# ============================================================
# Helper
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
# Experiment
# ============================================================

print()
print("=" * 78)
print(
    "T=1 | NUISANCE rho | FIXED sigma | FIXED eta"
)
print("=" * 78)


with torch.inference_mode():

    for batch_idx in range(N_BATCHES):

        # ====================================================
        # 1. theta_true ~ p(theta)
        #
        # IMPORTANT:
        # sampled from the exact discrete prior used
        # in the reference evidence.
        # ====================================================

        idx_theta_true = torch.multinomial(
            p_theta,
            num_samples=BATCH_SIZE,
            replacement=True,
        )

        theta_true = theta_grid[
            idx_theta_true
        ]

        # [B]


        # ====================================================
        # 2. rho_true ~ p(rho)
        #
        # IMPORTANT:
        # sampled from the SAME discrete rho prior used
        # for marginalization.
        # ====================================================

        idx_rho_true = torch.multinomial(
            p_rho,
            num_samples=BATCH_SIZE,
            replacement=True,
        )

        rho_true = rho_grid[
            idx_rho_true
        ]

        # [B]


        # ====================================================
        # 3. Generate observations
        #
        # r ~ p(r | theta_true, rho_true, eta)
        #
        # sigma is the SAME scalar for every realization.
        # ====================================================

        array_response_true = (
            array_response_theta[
                idx_theta_true
            ]
        )

        # [B]

        alpha_true = (
            rho_true
            * array_response_true
        )

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
        # 4. REFERENCE TARGETED EIG
        #
        # First evaluate:
        #
        # log p(r_b | theta_m, rho_j, eta)
        #
        # for:
        #
        # b = 1...B
        # m = 1...N_THETA
        # j = 1...N_RHO
        # ====================================================

        log_like_theta_rho = (
            log_amplitude_vector_likelihood(
                amp[:, None, None, :],
                s,
                alpha_theta_rho[
                    None, :, :
                ],
                sigma**2,
            )
        )

        # [B, N_THETA, N_RHO]


        # ====================================================
        # 5. Marginalize rho
        #
        # log p(r | theta)
        #
        # =
        #
        # log sum_rho [
        #     p(rho)
        #     p(r | theta, rho)
        # ]
        # ====================================================

        log_like_theta = torch.logsumexp(
            log_p_rho[
                None,
                None,
                :
            ]
            + log_like_theta_rho,
            dim=-1,
        )

        # [B, N_THETA]


        # We no longer need the huge theta-rho tensor.
        del log_like_theta_rho


        # ====================================================
        # 6. Numerator of targeted EIG
        #
        # IMPORTANT:
        #
        # This is NOT
        #
        # p(r | theta_true, rho_true)
        #
        # but
        #
        # p(r | theta_true)
        #
        # because rho is a nuisance parameter.
        # ====================================================

        batch_indices = torch.arange(
            BATCH_SIZE,
            device=device,
        )

        log_num = log_like_theta[
            batch_indices,
            idx_theta_true,
        ]

        # [B]


        # ====================================================
        # 7. Evidence
        #
        # p(r)
        #
        # =
        #
        # sum_theta
        # p(theta) p(r | theta)
        # ====================================================

        log_den = torch.logsumexp(
            log_p_theta[None, :]
            + log_like_theta,
            dim=1,
        )

        # [B]


        # ====================================================
        # 8. Reference targeted EIG terms
        # ====================================================

        eig_terms = (
            log_num
            - log_den
        )

        all_eig_terms.append(
            eig_terms.cpu()
        )


        # ====================================================
        # 9. Construct T=1 history for trajectory likelihood
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

        # [B,1,K]

        r_history = amp.unsqueeze(1)

        # [B,1,Ns]


        # ====================================================
        # 10. Contrastive theta samples
        #
        # Again:
        # EXACT SAME theta prior.
        # ====================================================

        idx_theta_contrast = torch.multinomial(
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
            idx_theta_contrast
        ]

        # [B,MAX_L]


        # Candidate indices are useful for an additional
        # consistency test below.
        idx_theta_candidates = torch.cat(
            [
                idx_theta_true.unsqueeze(1),
                idx_theta_contrast,
            ],
            dim=1,
        )

        # [B,MAX_L+1]


        theta_candidates = theta_grid[
            idx_theta_candidates
        ]

        # [B,MAX_L+1]


        # ====================================================
        # 11. DAD trajectory likelihood
        #
        # For T=1 this calculates:
        #
        # log p(r | theta)
        #
        # =
        #
        # log sum_rho
        # p(rho) p(r | theta,rho)
        # ====================================================

        log_prob = trajectory_log_likelihood(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
        )

        # [B,MAX_L+1]


        # ====================================================
        # 12. IMPORTANT SANITY CHECK
        #
        # We have independently calculated log p(r|theta)
        # above for EVERY theta on theta_grid.
        #
        # Therefore trajectory_log_likelihood() must give
        # exactly the corresponding entries.
        # ====================================================

        direct_candidate_log_prob = (
            log_like_theta.gather(
                dim=1,
                index=idx_theta_candidates,
            )
        )

        likelihood_diff = torch.abs(
            log_prob
            - direct_candidate_log_prob
        )

        max_likelihood_diff = (
            likelihood_diff.max().item()
        )

        likelihood_consistency_errors.append(
            max_likelihood_diff
        )


        # ====================================================
        # 13. g_L for different L
        #
        # Same true theta,
        # same observation,
        # same nested contrastive samples.
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
            f" | likelihood diff = "
            f"{max_likelihood_diff:.3e}"
        )


# ============================================================
# Concatenate Monte-Carlo samples
# ============================================================

eig_terms = torch.cat(
    all_eig_terms
)


# ============================================================
# Reference targeted EIG
# ============================================================

eig_mean, eig_se = mean_and_se(
    eig_terms
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 78)
print("RESULTS")
print("=" * 78)

print(
    f"Reference targeted EIG : "
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
# Likelihood consistency
# ============================================================

max_consistency_error = max(
    likelihood_consistency_errors
)

mean_consistency_error = (
    sum(likelihood_consistency_errors)
    / len(likelihood_consistency_errors)
)

print()
print("=" * 78)
print("TARGETED LIKELIHOOD CONSISTENCY")
print("=" * 78)

print(
    f"Maximum error : "
    f"{max_consistency_error:.3e}"
)

print(
    f"Mean max error: "
    f"{mean_consistency_error:.3e}"
)


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 78)
print("EXPECTED RESULT")
print("=" * 78)

print(
    "If the implementation is correct:"
)

print(
    "1) likelihood consistency errors "
    "should be ~ numerical precision;"
)

print(
    "2) E[g_L] should approach the "
    "reference targeted EIG as L grows."
)