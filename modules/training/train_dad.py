import math

import torch

from modules.beam_eig.simulator import (
    sigma_from_snr,
)

from modules.dad.experiment import rollout

from modules.dad.contrastive import (
    contrastive_bound,
    make_log_likelihood_fn,
    make_observation_fn,
)


def sample_theta_prior(
    n,
    device,
    dtype,
):
    """
    theta ~ Uniform(30°, 150°)
    """

    theta_min = math.radians(30.0)
    theta_max = math.radians(150.0)

    return (
        theta_min
        + (theta_max - theta_min)
        * torch.rand(
            n,
            device=device,
            dtype=dtype,
        )
    )


def sample_rho_prior(
    n,
    device,
    dtype,
):
    """
    rho ~ Uniform(0.05, 1)
    """

    rho_min = 0.05
    rho_max = 1.0

    return (
        rho_min
        + (rho_max - rho_min)
        * torch.rand(
            n,
            device=device,
            dtype=dtype,
        )
    )


def train_dad(
    policy,
    params,
    s,
    *,
    num_steps,
    batch_size,
    L,
    n_experiments,
    snr_db,
    rho_grid_size=50,
    learning_rate=1e-3,
    grad_clip=1.0,
    print_every=10,
):
    """
    Train DAD policy by maximizing the contrastive bound.

    Parameters
    ----------
    num_steps : int
        Nombre d'updates Adam.

    batch_size : int
        Nombre de trajectoires Monte-Carlo B.

    L : int
        Nombre d'hypothèses theta contrastives.

    n_experiments : int
        Nombre de beams T par trajectoire.
    """

    device = next(
        policy.parameters()
    ).device

    dtype = next(
        policy.parameters()
    ).dtype

    s = s.to(
        device=device,
        dtype=dtype,
    )


    # ========================================================
    # rho grid
    # ========================================================

    rho_grid = torch.linspace(
        0.05,
        1.0,
        rho_grid_size,
        device=device,
        dtype=dtype,
    )

    R = rho_grid.numel()

    # prior uniforme sur rho
    log_p_rho_prior = torch.full(
        (R,),
        -math.log(R),
        device=device,
        dtype=dtype,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.98,
    )


    policy.train()

    history = {
        "loss": [],
        "bound": [],
        "grad_norm": [],
    }

    ema_bound = None

    # ========================================================
    # Training
    # ========================================================

    for step in range(1, num_steps + 1):
        if step % 1000 == 0:
            scheduler.step()
        # ----------------------------------------------------
        # 1. Sample true latent
        # ----------------------------------------------------

        theta_true = sample_theta_prior(
            batch_size,
            device,
            dtype,
        ) #nous donne vecteur avec batch_size valeurs de theta

        rho_true = sample_rho_prior(
            batch_size,
            device,
            dtype,
        )

        # ----------------------------------------------------
        # Noise level
        # ----------------------------------------------------

        sigma = sigma_from_snr(
            s=s,
            snr_db=snr_db,
            rho_true=rho_true,
        )

        # ----------------------------------------------------
        # 2. Generate trajectory under true theta,rho
        # ----------------------------------------------------

        observation_fn = make_observation_fn(
            rho=rho_true,
            s=s,
            sigma=sigma,
            params=params,
        )

        eta_history, r_history = rollout(
            policy=policy,
            theta=theta_true,
            n_steps=n_experiments,
            observation_fn=observation_fn,
        )

        # eta_history : [B,T,N]
        # r_history   : [B,T,Ns]

        # ----------------------------------------------------
        # 3. Contrastive theta samples
        # ----------------------------------------------------

        theta_contrast = sample_theta_prior(
            batch_size * L,
            device,
            dtype,
        ).reshape(
            batch_size,
            L,
        )

        theta_candidates = torch.cat(
            [
                theta_true.unsqueeze(1),
                theta_contrast,
            ],
            dim=1,
        )

        # [B,L+1]

        # ----------------------------------------------------
        # 4. Likelihood with rho marginalized
        # ----------------------------------------------------

        log_likelihood_fn = make_log_likelihood_fn(
            rho_grid=rho_grid,
            s=s,
            sigma=sigma,
            params=params,
        )

        # ----------------------------------------------------
        # 5. Contrastive bound (bound is the mean of the g_L batches)
        # ----------------------------------------------------

        bound, g_L = contrastive_bound(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
        ) 

        # ----------------------------------------------------
        # 6. We maximize bound
        #
        # Adam minimizes -> loss = -bound
        # ----------------------------------------------------

        loss = -bound

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # 7. Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        ) # this erases the gradients of the previous step (otherwise they would accumulate)

        loss.backward() #this computes the gradient of the loss with respect to the parameters (or anything requiring gradients) using backpropagation

        # ----------------------------------------------------
        # Gradient norm
        # ----------------------------------------------------

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=grad_clip,
                norm_type=float("inf"),
            )
        ) # we clip the gradient to avoid overshooting, which can happen if the gradient is too large

        # ----------------------------------------------------
        # Weight update
        # ----------------------------------------------------

        optimizer.step() #here we update the parameters using the gradients computed in the backward pass and the learning rate specified in the optimizer

        # ====================================================
        # Logging
        # ====================================================

        bound_value = bound.detach().item()
        loss_value = loss.detach().item()
        grad_value = float(grad_norm)

        history["loss"].append(
            loss_value
        )

        history["bound"].append(
            bound_value
        )

        history["grad_norm"].append(
            grad_value
        )

        if ema_bound is None:
            ema_bound = bound_value
        else:
            ema_bound = (
                0.95 * ema_bound
                + 0.05 * bound_value
            )

        if (
            step == 1
            or step % print_every == 0
        ):
            print(
                f"step {step:6d} | "
                f"loss = {loss_value:+.4f} | "
                f"g_L = {bound_value:+.4f} | "
                f"EMA = {ema_bound:+.4f} | "
                f"std(g_L) = "
                f"{g_L.std(unbiased=False).item():.4f} | "
                f"grad = {grad_value:.4f}"
            )

    return history