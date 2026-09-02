import torch

from modules.beam_eig.array_model import beam_from_phases
from modules.beam_eig.simulator import simulate_y
from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)


def estimate_eig_for_eta(
    eta,
    a_grid,
    s,
    sigma,
    p_theta,
    rho_mean,
    N,
):
    """
    Reproduction de
    estimate_eig_for_eta_amplitude_theta_only()

    eta      : [K, 1]
    a_grid   : [K, L]
    p_theta  : [L]
    """

    # -----------------------------------------
    # Beam candidate
    # -----------------------------------------

    b = beam_from_phases(eta)

    # b^H a(theta_l) for all thetas, we have thus alpha_theta = [rho_mean * b^H a(theta_1), ..., rho_mean * b^H a(theta_L)]
    alpha_theta = rho_mean * (
        b.conj().T @ a_grid
    ).squeeze(0)  # [L]

    # -----------------------------------------
    # Current marginal posterior over theta
    # -----------------------------------------

    p_theta = p_theta / p_theta.sum()

    tiny = torch.finfo(p_theta.dtype).tiny

    log_p_theta = torch.log(
        p_theta + tiny
    )  # [L]

    # -----------------------------------------
    # Sample theta_n ~ p(theta | h_t)
    # -----------------------------------------

    #torch.mutinomial samples from p_theta, with replacement, N times
    idx_theta_samples = torch.multinomial(
        p_theta,
        num_samples=N,
        replacement=True,
    )  # [N]

    #because the idx_theta_samples indice corresponds to a(theta_l), we can get the corresponding alpha_theta for each sampled theta_n
    #alpha_samples contains our N samples for Monte Carlo estimation of EIG, each sample corresponds to a sampled theta_n
    alpha_samples = alpha_theta[
        idx_theta_samples
    ]  # [N]

    # -----------------------------------------
    # Simulate N future observations
    # -----------------------------------------

    y_samples = simulate_y(
        alpha_samples,
        s,
        sigma,
    )  # [N, Ns]

    amp_samples = torch.abs(
        y_samples
    )  # [N, Ns]

    # -----------------------------------------
    # Likelihood:
    # every observation n under every theta_l
    # -----------------------------------------

    log_likelihood_all = (
        log_amplitude_vector_likelihood(
            amp_samples[:, None, :],   # [N,1,Ns]
            s,
            alpha_theta[None, :],      # [1,L]
            sigma**2,
        )
    )

    # result: [N,L]

    # -----------------------------------------
    # Numerator
    # log p(r_n | theta_n, eta)
    # -----------------------------------------

    n_idx = torch.arange(
        N,
        device=eta.device,
    )

    log_p_num = log_likelihood_all[
        n_idx,
        idx_theta_samples,
    ]  # [N]

    # -----------------------------------------
    # Denominator
    #
    # log sum_l [
    #   p(theta_l | h_t)
    #   p(r_n | theta_l, eta)
    # ]
    # -----------------------------------------

    log_p_den = torch.logsumexp(
        log_p_theta[None, :]
        + log_likelihood_all,
        dim=1,
    )  # [N]

    # -----------------------------------------
    # Monte Carlo EIG
    # -----------------------------------------

    eig_terms = (
        log_p_num - log_p_den
    )

    return eig_terms.mean()

def choose_beam(
    eta_grid,
    a_grid,
    s,
    sigma,
    p_theta,
    rho_mean,
    N,
):
    R = eta_grid.shape[1]

    eig_values = torch.empty(
        R,
        dtype=p_theta.dtype,
        device=eta_grid.device,
    )

    for d in range(R):

        eig_values[d] = estimate_eig_for_eta(
            eta_grid[:, [d]],
            a_grid,
            s,
            sigma,
            p_theta,
            rho_mean,
            N,
        )

    best_idx = torch.argmax(eig_values)

    eta_star = eta_grid[:, best_idx][:, None]

    return eta_star, eig_values
