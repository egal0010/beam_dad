import math

import torch
from torch.utils.checkpoint import checkpoint

from modules.beam_eig.array_model import (
    phase_to_beam,
    steering_vector_batch,
)
from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)
from modules.beam_eig.simulator import (
    simulate_y,
    sigma_from_snr,
)


def trajectory_log_likelihood(
    theta_candidates,
    eta_history,
    r_history,
    log_likelihood_fn,
    log_p_rho_prior,
):
    """
    Compute log p(h_T | theta_c) for all theta candidates, with rho
    marginalized at the end of the trajectory.

    Parameters
    ----------
    theta_candidates : [B, C]
        C = L + 1 candidates. Column 0 is the true theta.
    eta_history : [B, T, N]
    r_history : [B, T, Ns]
    log_p_rho_prior : [R]

    Returns
    -------
    log_prob : [B, C]
        log p(h_T | theta_c), with rho marginalized.

    Notes
    -----
    Because rho is static across the whole trajectory,

        p(h_T | theta)
        = sum_rho p(rho) prod_t p(r_t | theta, rho, eta_t)

    so we can accumulate the conditional log-likelihood over t and
    marginalize rho only once at the end.

    This is mathematically equivalent to the sequential Bayes update used
    previously, but avoids repeated posterior normalization operations.
    """

    if theta_candidates.ndim != 2:
        raise ValueError(
            "theta_candidates must have shape [B, C], "
            f"got {tuple(theta_candidates.shape)}."
        )

    B, T, _ = eta_history.shape
    C = theta_candidates.shape[1]
    R = log_p_rho_prior.shape[0]

    if theta_candidates.shape[0] != B:
        raise ValueError("Batch dimension mismatch between candidates and history.")

    # [1, 1, R]. Broadcasting creates [B, C, R] only when needed.
    log_joint_rho = log_p_rho_prior.view(1, 1, R)

    for t in range(T):
        # IMPORTANT: do not expand over C here.
        # Broadcasting inside log_likelihood_fn handles the candidate dimension.
        eta_t = eta_history[:, t, :].unsqueeze(1)  # [B, 1, N]
        r_t = r_history[:, t, :].unsqueeze(1)      # [B, 1, Ns]

        log_like_t = log_likelihood_fn(
            theta_candidates,
            eta_t,
            r_t,
        )
        # [B, C, R]

        log_joint_rho = log_joint_rho + log_like_t

    # [B, C]
    return torch.logsumexp(
        log_joint_rho,
        dim=-1,
    )


def trajectory_log_likelihood_chunked(
    theta_candidates,
    eta_history,
    r_history,
    log_likelihood_fn,
    log_p_rho_prior,
    candidate_chunk_size=32,
    use_checkpoint=True,
):
    """
    Same result as trajectory_log_likelihood(), but evaluates the candidate
    dimension C=L+1 in chunks.

    The expensive Rice likelihood internally creates tensors with a dimension
    similar to [B, C, R, Ns]. Chunking replaces C by a much smaller C_chunk.

    If use_checkpoint=True, activation checkpointing recomputes each candidate
    chunk during backward instead of storing all large intermediate tensors.
    This trades extra compute time for substantially lower GPU memory usage.
    """

    if candidate_chunk_size is None:
        return trajectory_log_likelihood(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
        )

    if candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive or None.")

    C = theta_candidates.shape[1]

    # If the chunk already covers all candidates, avoid unnecessary Python loop.
    if candidate_chunk_size >= C and not use_checkpoint:
        return trajectory_log_likelihood(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
        )

    log_prob_chunks = []

    def compute_chunk(theta_chunk, eta_hist, r_hist):
        return trajectory_log_likelihood(
            theta_candidates=theta_chunk,
            eta_history=eta_hist,
            r_history=r_hist,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
        )

    for start in range(0, C, candidate_chunk_size):
        end = min(start + candidate_chunk_size, C)

        theta_chunk = theta_candidates[:, start:end]

        if use_checkpoint:
            log_prob_chunk = checkpoint(
                compute_chunk,
                theta_chunk,
                eta_history,
                r_history,
                use_reentrant=False,
            )
        else:
            log_prob_chunk = compute_chunk(
                theta_chunk,
                eta_history,
                r_history,
            )

        log_prob_chunks.append(log_prob_chunk)

    # [B, C]. This tensor is small compared with [B, C, R, Ns].
    return torch.cat(
        log_prob_chunks,
        dim=1,
    )


def contrastive_bound_from_log_prob(log_prob):
    """
    Compute g_L from precomputed trajectory log probabilities.

    The first column must correspond to the true theta and all remaining
    columns to contrastive candidates.

    Parameters
    ----------
    log_prob : Tensor [B, L + 1]
        log p(h_T | theta_c) for every candidate.

    Returns
    -------
    bound : scalar Tensor
        Monte-Carlo batch mean of g_L.
    g_L : Tensor [B]
        Per-realization contrastive bounds.
    """

    if log_prob.ndim != 2:
        raise ValueError(
            "log_prob must have shape [B, L + 1], "
            f"got {tuple(log_prob.shape)}."
        )

    C = log_prob.shape[1]

    if C < 2:
        raise ValueError(
            "log_prob must contain the true theta and at least "
            "one contrastive candidate."
        )

    log_prob_true = log_prob[:, 0]

    log_evidence = (
        torch.logsumexp(log_prob, dim=1)
        - math.log(C)
    )

    g_L = log_prob_true - log_evidence

    return g_L.mean(), g_L


def contrastive_bound(
    theta_candidates,
    eta_history,
    r_history,
    log_likelihood_fn,
    log_p_rho_prior,
    candidate_chunk_size=32,
    use_checkpoint=True,
):
    """
    Chunked/checkpointed version of the DAD contrastive bound.

    Set candidate_chunk_size=None to recover the unchunked calculation.
    Set use_checkpoint=False to chunk without activation checkpointing.
    """

    log_prob = trajectory_log_likelihood_chunked(
        theta_candidates=theta_candidates,
        eta_history=eta_history,
        r_history=r_history,
        log_likelihood_fn=log_likelihood_fn,
        log_p_rho_prior=log_p_rho_prior,
        candidate_chunk_size=candidate_chunk_size,
        use_checkpoint=use_checkpoint,
    )

    return contrastive_bound_from_log_prob(log_prob)


def _as_beam(design):
    if torch.is_complex(design):
        return design

    return phase_to_beam(design)


def _to_real_tensor(
    value,
    reference,
):
    dtype = reference.real.dtype

    if torch.is_tensor(value):
        return value.to(
            device=reference.device,
            dtype=dtype,
        )

    return torch.tensor(
        value,
        device=reference.device,
        dtype=dtype,
    )


def make_log_likelihood_fn(
    rho_grid,
    s,
    snr_db,
    params,
):
    """
    rho_grid : [R]

    Returns a likelihood function evaluated for all rho values.
    """

    rho_grid = _to_real_tensor(
        rho_grid,
        s,
    )

    sigma_grid = sigma_from_snr(
        s=s,
        snr_db=snr_db,
        rho_true=rho_grid,
    )
    # [R]

    sigma2_grid = (
        sigma_grid**2
    )[None, None, :, None]
    # [1, 1, R, 1]

    def log_likelihood_fn(
        theta,
        eta,
        amp_vec,
    ):
        """
        theta   : [B, C]
        eta     : [B, 1, N] or [B, C, N]
        amp_vec : [B, 1, Ns] or [B, C, Ns]

        Returns
        -------
        [B, C, R]
        """

        a = steering_vector_batch(
            theta,
            params,
        )
        # [B, C, N]

        b = phase_to_beam(
            eta,
        )
        # [B, 1, N] is sufficient; broadcasting handles C.

        array_response = torch.sum(
            torch.conj(b) * a,
            dim=-1,
        )
        # [B, C]

        alpha = (
            array_response.unsqueeze(-1)
            * rho_grid.view(1, 1, -1)
        )
        # [B, C, R]

        amp_vec = amp_vec.unsqueeze(-2)
        # [B, 1, 1, Ns] if amp_vec came in as [B,1,Ns]

        return log_amplitude_vector_likelihood(
            amp_vec=amp_vec,
            s=s,
            alpha=alpha,
            sigma2=sigma2_grid,
        )
        # [B, C, R]

    return log_likelihood_fn


def make_observation_fn(
    rho,
    s,
    snr_db,
    params,
):
    """
    Here rho is the true channel amplitude used by the simulator.
    """

    rho = _to_real_tensor(
        rho,
        s,
    )

    sigma = sigma_from_snr(
        s=s,
        snr_db=snr_db,
        rho_true=rho,
    )

    def observation_fn(
        theta,
        design,
    ):
        a = steering_vector_batch(
            theta,
            params,
        )

        b = _as_beam(
            design,
        )

        alpha = torch.sum(
            torch.conj(b) * a,
            dim=-1,
        )

        y = simulate_y(
            rho * alpha,
            s,
            sigma,
        )

        return torch.abs(y)

    return observation_fn
