import math

import torch

from modules.beam_eig.array_model import (
    phase_to_beam,
    steering_vector_batch,
)

from modules.beam_eig.likelihood import (
    log_amplitude_vector_likelihood,
)

from modules.beam_eig.simulator import simulate_y


def trajectory_log_likelihood(
    theta_candidates,
    eta_history,
    r_history,
    log_likelihood_fn,
    log_p_rho_prior,
):
    """
    theta_candidates : [B, C] avec C = L + 1
    eta_history      : [B, T, N]
    r_history        : [B, T, Ns]
    log_p_rho_prior  : [R]

    Returns
    -------
    log_prob : [B, C]

        log p(h_T | theta_c)

    avec rho marginalisé.
    """

    B, T, N = eta_history.shape
    C = theta_candidates.shape[1]
    Ns = r_history.shape[-1]
    R = log_p_rho_prior.shape[0]

    # p(rho | theta, h_0) = p(rho)
    log_p_rho = (
        log_p_rho_prior
        .view(1, 1, R)
        .expand(B, C, R)
    )

    total_log_prob = torch.zeros(
        B,
        C,
        device=eta_history.device,
        dtype=eta_history.dtype,
    )

    for t in range(T):

        eta_t = eta_history[:, t, :]
        # [B, N]

        r_t = r_history[:, t, :]
        # [B, Ns]

        eta_t = eta_t.unsqueeze(1).expand(
            B,
            C,
            N,
        )

        r_t = r_t.unsqueeze(1).expand(
            B,
            C,
            Ns,
        )

        # -------------------------------------------------
        # log p(r_t | theta, rho, eta_t)
        # -------------------------------------------------

        log_like_t = log_likelihood_fn(
            theta_candidates,
            eta_t,
            r_t,
        )

        # [B, C, R]

        # -------------------------------------------------
        # p(rho | theta,h_{t-1})
        # *
        # p(r_t | theta,rho,eta_t)
        # -------------------------------------------------

        log_joint_rho = (
            log_p_rho
            + log_like_t
        )

        # -------------------------------------------------
        # p(r_t | theta,h_{t-1},eta_t)
        #
        # marginalisation de rho
        # -------------------------------------------------

        log_predictive_t = torch.logsumexp(
            log_joint_rho,
            dim=-1,
        )

        # [B, C]

        # -------------------------------------------------
        # Accumulation :
        #
        # log p(h_t | theta)
        # -------------------------------------------------

        total_log_prob = (
            total_log_prob
            + log_predictive_t
        )

        # -------------------------------------------------
        # Bayes update :
        #
        # p(rho | theta,h_t)
        # -------------------------------------------------

        log_p_rho = (
            log_joint_rho
            - log_predictive_t.unsqueeze(-1)
        )

    return total_log_prob


def contrastive_bound(
    theta_candidates,
    eta_history,
    r_history,
    log_likelihood_fn,
    log_p_rho_prior,
):
    log_prob = trajectory_log_likelihood(
        theta_candidates,
        eta_history,
        r_history,
        log_likelihood_fn,
        log_p_rho_prior,
    )

    # vrai theta = colonne 0
    log_prob_true = log_prob[:, 0]

    C = theta_candidates.shape[1]

    log_evidence = (
        torch.logsumexp(
            log_prob,
            dim=1,
        )
        - math.log(C) #C=L+1, le +1 vient du fait que theta_true est ajouté à la liste des candidats
    ) 

    g_L = (
        log_prob_true
        - log_evidence
    ) #donc log_prob_true = log p(h_T | theta_true) et log_evidence =log \frac{1}{L+1} \sum_{\ell=1}^L p(h_T | theta_{candidates}) 

    return g_L.mean(), g_L


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
    sigma,
    params,
):
    """
    rho_grid : [R]

    Retourne une likelihood évaluée pour
    TOUS les rho de la grille.
    """

    rho_grid = _to_real_tensor(
        rho_grid,
        s,
    )

    sigma = _to_real_tensor(
        sigma,
        s,
    )

    sigma2 = sigma**2

    def log_likelihood_fn(
        theta,
        eta,
        amp_vec,
    ):
        """
        theta   : [B, C]
        eta     : [B, C, N]
        amp_vec : [B, C, Ns]

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

        # [B, C, N]

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

        # [B, C, 1, Ns]

        return log_amplitude_vector_likelihood(
            amp_vec=amp_vec,
            s=s,
            alpha=alpha,
            sigma2=sigma2,
        )

        # [B, C, R]

    return log_likelihood_fn


def make_observation_fn(
    rho,
    s,
    sigma,
    params,
):
    """
    Ici rho = rho_true.

    C'est le vrai rho du canal simulé.
    """

    rho = _to_real_tensor(
        rho,
        s,
    )

    sigma = _to_real_tensor(
        sigma,
        s,
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