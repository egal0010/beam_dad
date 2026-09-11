import torch

from modules.beam_eig.array_model import beam_from_phases
from modules.beam_eig.likelihood import log_amplitude_vector_likelihood, log_besseli0
from modules.beam_eig.simulator import sigma_from_snr

#remarque, on vectorise ici, c'est à dire que l'on calcule le log likelihood pour toutes les hypothèses theta et rho en même temps, ce qui est plus efficace que de le faire une par une.
def update_posterior(
    eta,
    a_grid,
    amp_vec,
    posterior,
    rho_grid,
    s,
    snr_db=None,
    sigma=None,
    mode=None,
):
    """
    posterior(theta, rho) après une nouvelle observation.

    eta       : [K, 1]
    a_grid    : [K, L]
    posterior : [L, Lrho]
    rho_grid  : [Lrho]
    amp_vec   : [Ns]
    """

    b = beam_from_phases(eta)  # [K, 1]

    # b^H a(theta), pour tous les theta
    alpha_theta = (
        b.conj().T @ a_grid
    ).squeeze(0)  # [L]

    # rho * b^H a(theta), pour tous theta et rho
    alpha_grid = (
        alpha_theta[:, None]
        * rho_grid[None, :]
    )  # [L, Lrho]

    if mode =="sigma_fixed":
        sigma2=sigma**2

    elif mode == "sigma_snr":

        sigma_grid = sigma_from_snr(
        s=s,
        snr_db=snr_db,
        rho_true=rho_grid,
        ) 
         # [R]
        sigma2 = (
            sigma_grid**2
        )[None, :, None]  # [1,R,1]
        
    else:
        raise ValueError(f"Unknown mode: {mode}")    

    # log likelihood pour toutes les hypothèses
    log_likelihood = log_amplitude_vector_likelihood(
        amp_vec,
        s,
        alpha_grid,
        sigma2,
    )  # [L, Lrho]

    tiny = torch.finfo(posterior.dtype).tiny

    log_p = (
        torch.log(posterior + tiny)
        + log_likelihood
    )

    # normalisation
    log_p = log_p - torch.logsumexp(
        log_p.flatten(),
        dim=0
    )

    return torch.exp(log_p)


##Fonctions pour marginaliser, permet d'alléger le code, nécessaire?

def marginal_theta(posterior):

    p_theta = posterior.sum(dim=1)

    return p_theta / p_theta.sum()


def marginal_rho(posterior):

    p_rho = posterior.sum(dim=0)

    return p_rho / p_rho.sum()


def compute_rho_mean(posterior, rho_grid):

    p_rho = marginal_rho(posterior)

    return (
        p_rho[:, None].T
        @ rho_grid[:, None]
    ).squeeze()
