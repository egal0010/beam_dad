"""
Quick equivalence test for candidate chunking.

Run from the project root after replacing modules/dad/contrastive.py:

    python test_chunking_equivalence.py

The test does NOT use the physical beam model. It checks that:
  1) chunked and unchunked trajectory likelihoods match;
  2) the resulting contrastive bound matches;
  3) gradients with respect to eta_history and r_history match.

Small floating-point differences are expected.
"""

import torch

from modules.dad.contrastive import (
    trajectory_log_likelihood,
    trajectory_log_likelihood_chunked,
    contrastive_bound_from_log_prob,
)


torch.manual_seed(0)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

B = 4
C = 11
T = 3
N = 8
Ns = 7
R = 5

theta = torch.randn(
    B, C,
    device=device,
    dtype=dtype,
)

eta_base = torch.randn(
    B, T, N,
    device=device,
    dtype=dtype,
)

r_base = torch.rand(
    B, T, Ns,
    device=device,
    dtype=dtype,
)

log_p_rho_prior = torch.log_softmax(
    torch.randn(
        R,
        device=device,
        dtype=dtype,
    ),
    dim=0,
)


def dummy_log_likelihood(
    theta_candidates,
    eta,
    amp_vec,
):
    """
    Produces a differentiable [B,C,R] tensor while relying on broadcasting
    in exactly the same shape pattern as the real likelihood.
    """

    # eta: [B,1,N], amp_vec: [B,1,Ns]
    eta_summary = eta.mean(dim=-1)       # [B,1]
    r_summary = amp_vec.mean(dim=-1)     # [B,1]

    rho_grid = torch.linspace(
        0.1,
        1.0,
        R,
        device=theta_candidates.device,
        dtype=theta_candidates.dtype,
    )

    center = (
        theta_candidates
        + 0.3 * eta_summary
        + 0.2 * r_summary
    )
    # [B,C]

    return -(
        center.unsqueeze(-1)
        - rho_grid.view(1, 1, R)
    ) ** 2


# ============================================================
# Unchunked
# ============================================================

eta_1 = eta_base.clone().requires_grad_(True)
r_1 = r_base.clone().requires_grad_(True)

log_prob_full = trajectory_log_likelihood(
    theta_candidates=theta,
    eta_history=eta_1,
    r_history=r_1,
    log_likelihood_fn=dummy_log_likelihood,
    log_p_rho_prior=log_p_rho_prior,
)

bound_full, _ = contrastive_bound_from_log_prob(
    log_prob_full
)

(-bound_full).backward()

grad_eta_full = eta_1.grad.detach().clone()
grad_r_full = r_1.grad.detach().clone()


# ============================================================
# Chunked + checkpointed
# ============================================================

eta_2 = eta_base.clone().requires_grad_(True)
r_2 = r_base.clone().requires_grad_(True)

log_prob_chunk = trajectory_log_likelihood_chunked(
    theta_candidates=theta,
    eta_history=eta_2,
    r_history=r_2,
    log_likelihood_fn=dummy_log_likelihood,
    log_p_rho_prior=log_p_rho_prior,
    candidate_chunk_size=3,
    use_checkpoint=True,
)

bound_chunk, _ = contrastive_bound_from_log_prob(
    log_prob_chunk
)

(-bound_chunk).backward()

grad_eta_chunk = eta_2.grad.detach().clone()
grad_r_chunk = r_2.grad.detach().clone()


print("Device:", device)

print(
    "max |log_prob full - chunk| =",
    (
        log_prob_full.detach()
        - log_prob_chunk.detach()
    ).abs().max().item()
)

print(
    "|bound full - chunk| =",
    abs(
        bound_full.detach().item()
        - bound_chunk.detach().item()
    )
)

print(
    "max |grad eta full - chunk| =",
    (
        grad_eta_full
        - grad_eta_chunk
    ).abs().max().item()
)

print(
    "max |grad r full - chunk| =",
    (
        grad_r_full
        - grad_r_chunk
    ).abs().max().item()
)
