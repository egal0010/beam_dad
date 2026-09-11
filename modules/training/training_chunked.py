import math
import os

import torch

from modules.dad.experiment import rollout
from modules.dad.contrastive import (
    contrastive_bound,
    make_log_likelihood_fn,
    make_observation_fn,
)


os.makedirs(
    "checkpoints",
    exist_ok=True,
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


def _cuda_memory_gb():
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()

    return {
        "allocated": torch.cuda.memory_allocated() / 2**30,
        "reserved": torch.cuda.memory_reserved() / 2**30,
        "peak": torch.cuda.max_memory_allocated() / 2**30,
    }


def _print_cuda_memory(label):
    values = _cuda_memory_gb()

    if values is None:
        return

    print(
        f"{label:28s} | "
        f"allocated={values['allocated']:.2f} GiB | "
        f"reserved={values['reserved']:.2f} GiB | "
        f"peak={values['peak']:.2f} GiB"
    )


def train_dad_chunked(
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
    candidate_chunk_size=32,
    use_checkpoint=True,
    profile_first_step=False,
):
    """
    Train DAD policy by maximizing the contrastive bound.

    Parameters
    ----------
    batch_size : int
        Monte-Carlo trajectory batch size B.

    L : int
        Number of contrastive theta samples PER trajectory.

    candidate_chunk_size : int or None
        Number of theta candidates evaluated simultaneously inside the Rice
        likelihood. This controls peak memory. Typical starting values:
        16, 32, 64.

        Set to None to disable candidate chunking.

    use_checkpoint : bool
        If True, activation checkpointing is applied to each candidate chunk.
        This lowers GPU memory further, at the cost of extra backward compute.

    profile_first_step : bool
        If True, print CUDA memory after rollout, after contrastive bound,
        and after backward for the first training step.
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

    log_p_rho_prior = torch.full(
        (R,),
        -math.log(R),
        device=device,
        dtype=dtype,
    )

    # ========================================================
    # Likelihood closure
    #
    # This depends only on fixed training settings, so create it
    # ONCE rather than once per optimizer step.
    # ========================================================

    log_likelihood_fn = make_log_likelihood_fn(
        rho_grid=rho_grid,
        s=s,
        snr_db=snr_db,
        params=params,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.8, 0.998),
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
    best_ema = -float("inf")

    # ========================================================
    # Training
    # ========================================================

    for step in range(1, num_steps + 1):

        if (
            profile_first_step
            and step == 1
            and torch.cuda.is_available()
        ):
            torch.cuda.reset_peak_memory_stats()
            _print_cuda_memory("start step")

        # ----------------------------------------------------
        # 1. Sample true latent
        # ----------------------------------------------------

        theta_true = sample_theta_prior(
            batch_size,
            device,
            dtype,
        )

        rho_true = sample_rho_prior(
            batch_size,
            device,
            dtype,
        )

        # ----------------------------------------------------
        # 2. Generate trajectory
        # ----------------------------------------------------

        observation_fn = make_observation_fn(
            rho=rho_true,
            s=s,
            snr_db=snr_db,
            params=params,
        )

        eta_history, r_history = rollout(
            policy=policy,
            theta=theta_true,
            n_steps=n_experiments,
            observation_fn=observation_fn,
        )

        if profile_first_step and step == 1:
            _print_cuda_memory("after rollout")

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
        # [B, L+1]

        # ----------------------------------------------------
        # 4. Contrastive bound
        # ----------------------------------------------------

        bound, g_L = contrastive_bound(
            theta_candidates=theta_candidates,
            eta_history=eta_history,
            r_history=r_history,
            log_likelihood_fn=log_likelihood_fn,
            log_p_rho_prior=log_p_rho_prior,
            candidate_chunk_size=candidate_chunk_size,
            use_checkpoint=use_checkpoint,
        )

        if profile_first_step and step == 1:
            _print_cuda_memory("after contrastive bound")

        # ----------------------------------------------------
        # 5. Maximize bound
        # ----------------------------------------------------

        loss = -bound

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # 6. Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True,
        )

        loss.backward()

        if profile_first_step and step == 1:
            _print_cuda_memory("after backward")

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=grad_clip,
                norm_type=float("inf"),
            )
        )

        optimizer.step()

        # ----------------------------------------------------
        # Scheduler / periodic checkpoint
        # ----------------------------------------------------

        if step % 1000 == 0:
            scheduler.step()

            torch.save(
                {
                    "model_state_dict":
                        policy.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "Nx": params.Nx,
                    "Ny": params.Ny,
                    "Ns": s.numel(),

                    "hidden_dim": policy.hidden_dim,
                    "encoding_dim": policy.encoding_dim,

                    "encoder_type": "mean_var",

                    "step": step,
                    "history": history,
                    "ema_bound": ema_bound,

                    "L": L,
                    "batch_size": batch_size,
                    "candidate_chunk_size": candidate_chunk_size,
                    "use_checkpoint": use_checkpoint,
                },
                f"checkpoints/dad_T{n_experiments}_step_{step}.pt",
            )

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

        if ema_bound > best_ema:
            best_ema = ema_bound

            torch.save(
                {
                    "model_state_dict":
                        policy.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "Nx": params.Nx,
                    "Ny": params.Ny,
                    "Ns": s.numel(),

                    "hidden_dim": policy.hidden_dim,
                    "encoding_dim": policy.encoding_dim,

                    "encoder_type": "mean_var",

                    "step": step,
                    "history": history,
                    "ema_bound": ema_bound,

                    "L": L,
                    "batch_size": batch_size,
                    "candidate_chunk_size": candidate_chunk_size,
                    "use_checkpoint": use_checkpoint,
                },
                f"checkpoints/dad_T{n_experiments}_best.pt",
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
