import torch

from modules.beam_eig.array_model import beam_from_phases
from modules.beam_eig.posterior import compute_rho_mean
from modules.beam_eig.simulator import simulate_y
from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)


# ============================================================
# rho_mean plug-in EIG
# ============================================================

def estimate_eig_for_eta_mean(
    eta,
    a_grid,
    s,
    sigma,
    p_theta,
    rho_mean,
    N,
):
    """
    Myopic EIG with rho replaced by its posterior mean.

    eta      : [K,1]
    a_grid   : [K,L]
    p_theta  : [L]
    rho_mean : scalar
    """

    # --------------------------------------------------------
    # Beam
    # --------------------------------------------------------

    b = beam_from_phases(eta)

    # b^H a(theta_l)
    beam_response = (
        b.conj().T @ a_grid
    ).squeeze(0)  # [L]

    alpha_theta = (
        rho_mean * beam_response
    )  # [L]


    # --------------------------------------------------------
    # p(theta | h_t)
    # --------------------------------------------------------

    p_theta = (
        p_theta / p_theta.sum()
    )

    tiny = torch.finfo(
        p_theta.dtype
    ).tiny

    log_p_theta = torch.log(
        p_theta + tiny
    )


    # --------------------------------------------------------
    # theta_n ~ p(theta | h_t)
    # --------------------------------------------------------

    idx_theta_samples = torch.multinomial(
        p_theta,
        num_samples=N,
        replacement=True,
    )  # [N]

    alpha_samples = alpha_theta[
        idx_theta_samples
    ]  # [N]


    # --------------------------------------------------------
    # r_n ~ p(r | theta_n, rho_mean, eta)
    # --------------------------------------------------------

    y_samples = simulate_y(
        alpha_samples,
        s,
        sigma,
    )

    amp_samples = torch.abs(
        y_samples
    )  # [N,Ns]


    # --------------------------------------------------------
    # p(r_n | theta_l, rho_mean, eta)
    #
    # [N,L]
    # --------------------------------------------------------

    log_likelihood_all = (
        log_amplitude_vector_likelihood(
            amp_samples[:, None, :],
            s,
            alpha_theta[None, :],
            sigma**2,
        )
    )


    # --------------------------------------------------------
    # Numerator
    #
    # log p(r_n | theta_n, rho_mean, eta)
    # --------------------------------------------------------

    n_idx = torch.arange(
        N,
        device=eta.device,
    )

    log_p_num = log_likelihood_all[
        n_idx,
        idx_theta_samples,
    ]


    # --------------------------------------------------------
    # Denominator
    #
    # log sum_theta
    #
    # p(theta | h_t)
    # p(r_n | theta, rho_mean, eta)
    # --------------------------------------------------------

    log_p_den = torch.logsumexp(
        log_p_theta[None, :]
        + log_likelihood_all,
        dim=1,
    )


    # --------------------------------------------------------
    # NMC EIG
    # --------------------------------------------------------

    return (
        log_p_num - log_p_den
    ).mean()

# ============================================================
# Marginalized-rho EIG
# ============================================================

def estimate_eig_for_eta_marginal(
    eta,
    a_grid,
    s,
    sigma,
    posterior,
    rho_grid,
    N,
    rho_chunk_size=10,
):
    """
    Targeted EIG for theta with rho marginalized.

    posterior:
        p(theta,rho | h_t)
        shape [L,R]

    rho_grid:
        [R]

    The predictive distribution is

        p(r | theta,h_t,eta)
        =
        sum_rho
        p(rho | theta,h_t)
        p(r | theta,rho,eta).
    """

    # --------------------------------------------------------
    # Normalize joint posterior
    # --------------------------------------------------------

    posterior = (
        posterior / posterior.sum()
    )

    tiny = torch.finfo(
        posterior.dtype
    ).tiny


    # --------------------------------------------------------
    # p(theta | h_t)
    # --------------------------------------------------------

    p_theta = posterior.sum(
        dim=1
    )  # [L]

    p_theta = (
        p_theta / p_theta.sum()
    )

    log_p_theta = torch.log(
        p_theta + tiny
    )


    # --------------------------------------------------------
    # p(rho | theta,h_t)
    #
    # posterior [L,R]
    # --------------------------------------------------------

    row_sum = posterior.sum(
        dim=1,
        keepdim=True,
    )

    p_rho_given_theta = (
        posterior
        / row_sum.clamp_min(tiny)
    )  # [L,R]

    log_p_rho_given_theta = torch.log(
        p_rho_given_theta + tiny
    )


    # --------------------------------------------------------
    # Beam response
    # --------------------------------------------------------

    b = beam_from_phases(eta)

    beam_response = (
        b.conj().T @ a_grid
    ).squeeze(0)  # [L]


    # --------------------------------------------------------
    # theta_n ~ p(theta | h_t)
    # --------------------------------------------------------

    idx_theta_samples = torch.multinomial(
        p_theta,
        num_samples=N,
        replacement=True,
    )  # [N]


    # --------------------------------------------------------
    # rho_n ~ p(rho | theta_n,h_t)
    # --------------------------------------------------------

    rho_probs_samples = (
        p_rho_given_theta[
            idx_theta_samples
        ]
    )  # [N,R]

    idx_rho_samples = torch.multinomial(
        rho_probs_samples,
        num_samples=1,
        replacement=True,
    ).squeeze(-1)  # [N]

    rho_samples = rho_grid[
        idx_rho_samples
    ]  # [N]


    # --------------------------------------------------------
    # Generate future observations
    #
    # r_n ~
    # p(r | theta_n,rho_n,eta)
    # --------------------------------------------------------

    beam_response_samples = (
        beam_response[
            idx_theta_samples
        ]
    )

    alpha_samples = (
        rho_samples
        * beam_response_samples
    )  # [N]

    y_samples = simulate_y(
        alpha_samples,
        s,
        sigma,
    )

    amp_samples = torch.abs(
        y_samples
    )  # [N,Ns]


    # --------------------------------------------------------
    # Marginal likelihood
    #
    # log p(r_n | theta_l,h_t,eta)
    #
    # =
    #
    # log sum_j [
    #   p(rho_j | theta_l,h_t)
    #   p(r_n | theta_l,rho_j,eta)
    # ]
    #
    # Result: [N,L]
    # --------------------------------------------------------

    L = p_theta.numel()
    R = rho_grid.numel()

    log_likelihood_all = torch.full(
        (N, L),
        -torch.inf,
        dtype=p_theta.dtype,
        device=p_theta.device,
    )

    # Chunk rho so that we do not allocate
    # [N,L,R,Ns] all at once.
    for start in range(
        0,
        R,
        rho_chunk_size,
    ):

        end = min(
            start + rho_chunk_size,
            R,
        )

        rho_chunk = rho_grid[
            start:end
        ]  # [Rc]

        # alpha(theta_l,rho_j)
        #
        # [L,Rc]
        alpha_chunk = (
            beam_response[:, None]
            * rho_chunk[None, :]
        )

        # Likelihood:
        #
        # amp              [N,1,1,Ns]
        # alpha            [1,L,Rc]
        #
        # result           [N,L,Rc]
        log_like_chunk = (
            log_amplitude_vector_likelihood(
                amp_samples[
                    :, None, None, :
                ],
                s,
                alpha_chunk[
                    None, :, :
                ],
                sigma**2,
            )
        )

        log_weight_chunk = (
            log_p_rho_given_theta[
                None,
                :,
                start:end,
            ]
        )  # [1,L,Rc]

        # Marginalize rho inside this chunk
        chunk_marginal = torch.logsumexp(
            log_weight_chunk
            + log_like_chunk,
            dim=-1,
        )  # [N,L]

        # Combine chunks:
        #
        # log(exp(old) + exp(chunk))
        log_likelihood_all = torch.logaddexp(
            log_likelihood_all,
            chunk_marginal,
        )
    # --------------------------------------------------------
    # Numerator
    #
    # IMPORTANT:
    #
    # because the information target is THETA,
    # numerator is also marginalized over rho:
    #
    # p(r_n | theta_n,h_t,eta)
    # --------------------------------------------------------

    n_idx = torch.arange(
        N,
        device=eta.device,
    )

    log_p_num = log_likelihood_all[
        n_idx,
        idx_theta_samples,
    ]
    # --------------------------------------------------------
    # Denominator
    #
    # p(r_n | h_t,eta)
    #
    # =
    #
    # sum_theta
    # p(theta | h_t)
    # p(r_n | theta,h_t,eta)
    # --------------------------------------------------------

    log_p_den = torch.logsumexp(
        log_p_theta[None, :]
        + log_likelihood_all,
        dim=1,
    )
    # --------------------------------------------------------
    # Targeted NMC EIG
    # --------------------------------------------------------
    eig_terms = (
        log_p_num
        - log_p_den
    )

    return eig_terms.mean()


# ============================================================
# Central beam selector
# ============================================================

def choose_beam(
    eta_grid,
    a_grid,
    s,
    sigma,
    p_theta,
    posterior,
    rho_grid,
    mode,
    N,
    rho_chunk_size=10,
):
    """
    Choose beam maximizing one-step EIG.

    mode:
        "mean"
            rho -> E[rho | h_t]

        "marginal"
            rho marginalized using
            p(rho | theta,h_t)
    """

    num_beams = eta_grid.shape[1]

    eig_values = torch.empty(
        num_beams,
        dtype=p_theta.dtype,
        device=eta_grid.device,
    )

    # ========================================================
    # rho_mean approximation
    # ========================================================

    if mode == "mean":

        rho_mean = compute_rho_mean(
            posterior,
            rho_grid,
        )

        for d in range(num_beams):

            eta_d = eta_grid[
                :,
                [d],
            ]

            eig_values[d] = (
                estimate_eig_for_eta_mean(
                    eta=eta_d,
                    a_grid=a_grid,
                    s=s,
                    sigma=sigma,
                    p_theta=p_theta,
                    rho_mean=rho_mean,
                    N=N,
                )
            )
    # ========================================================
    # rho marginalization
    # ========================================================

    elif mode == "marginal":

        for d in range(num_beams):

            eta_d = eta_grid[
                :,
                [d],
            ]

            eig_values[d] = (
                estimate_eig_for_eta_marginal(
                    eta=eta_d,
                    a_grid=a_grid,
                    s=s,
                    sigma=sigma,
                    posterior=posterior,
                    rho_grid=rho_grid,
                    N=N,
                    rho_chunk_size=rho_chunk_size,
                )
            )

    else:

        raise ValueError(
            "mode must be either "
            "'mean' or 'marginal', "
            f"got {mode!r}"
        )


    # ========================================================
    # Best beam
    # ========================================================

    best_idx = torch.argmax(eig_values)

    eta_star = eta_grid[:,best_idx,][:,None,]

    return (
        eta_star,
        eig_values,
    )