import torch


def log_besseli0(x):
    x = torch.clamp(x.real, min=0.0)

    return x + torch.log(
        torch.special.i0e(x)
    )


def log_amplitude_vector_likelihood(
    amp_vec,
    s,
    alpha,
    sigma2,
):
    eps = torch.finfo(amp_vec.dtype).tiny

    sigma2 = torch.as_tensor(
        sigma2,
        dtype=amp_vec.real.dtype,
        device=amp_vec.device,
    )

    while sigma2.ndim < amp_vec.ndim:
        sigma2 = sigma2.unsqueeze(-1)

    amp_vec = torch.clamp(
        amp_vec.real,
        min=eps
    ) #on fait ça pour éviter les problèmes de log(0) et de division par zéro

    mu_abs = torch.abs(alpha[..., None] * s)

    x = 2 * amp_vec * mu_abs / sigma2

    lp = (
        torch.log(2 * amp_vec)
        - torch.log(sigma2)
        - (amp_vec**2 + mu_abs**2) / sigma2
        + log_besseli0(x)
    )

    return lp.sum(dim=-1)  # somme sur la dimension des symboles, on obtient un log likelihood par hypothèse (theta, rho    )
