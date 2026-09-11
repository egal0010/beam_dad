import math
import torch
import matplotlib.pyplot as plt

from modules.dad.rice_estimator import RiceEstimator

from modules.beam_eig.simulator import (
    generate_amplitude_measurement,
)

from modules.beam_eig.pilot import (
    generate_pilot_sequence,
)

from modules.beam_eig.array_model import (
    beam_from_phases,
    steering_vector,
)

from modules.beam_eig.params import Params


# ============================================================
# DEVICE / SETTINGS
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dtype = torch.float32

params = Params()

NUM_AMPLITUDES = 127

nx = 8

batch_size = 512

# 5000 suffit largement pour commencer.
# Ton précédent run avait déjà convergé très tôt.
num_steps = 5000

num_test_samples = 10_000

learning_rate = 1e-3


# ============================================================
# SNR SETTINGS
# ============================================================

SNR_DB_MIN = -10.0
SNR_DB_MAX = 10.0

SNR_TEST_VALUES = [
    -10.0,
    -5.0,
    0.0,
    5.0,
    10.0,
]


print(f"Device: {device}")


# ============================================================
# PRIORS
# ============================================================

def sample_theta_prior(
    n,
    device,
    dtype,
):

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


def sample_reference_snr_db(
    n,
    device,
    dtype,
):

    return (
        SNR_DB_MIN
        + (SNR_DB_MAX - SNR_DB_MIN)
        * torch.rand(
            n,
            device=device,
            dtype=dtype,
        )
    )


# ============================================================
# SNR -> SIGMA
#
# Reference SNR:
#
# SNR_ref = rho² / sigma²
#
# therefore
#
# sigma = rho / sqrt(SNR_linear)
#
#       = rho * 10^(-SNR_dB/20)
# ============================================================

def sigma_from_rho_snr(
    rho,
    snr_db,
):

    return (
        rho
        * torch.pow(
            10.0,
            -snr_db / 20.0,
        )
    )


# ============================================================
# DIRECT RICE SIMULATOR
#
# nu    : [B]
# sigma : [B] or scalar
#
# r : [B, 127]
# ============================================================

def generate_direct_rice(
    nu,
    sigma,
    num_amplitudes=127,
):

    B = nu.shape[0]

    # Allow scalar sigma or [B]
    if not torch.is_tensor(sigma):

        sigma = torch.full_like(
            nu,
            float(sigma),
        )

    elif sigma.ndim == 0:

        sigma = sigma.expand_as(
            nu
        )

    sigma = sigma.to(
        device=nu.device,
        dtype=nu.dtype,
    )

    # CN(0, sigma²)
    #
    # Re/Im std = sigma/sqrt(2)

    noise_std = (
        sigma[:, None]
        / math.sqrt(2.0)
    )

    noise_real = (
        noise_std
        * torch.randn(
            B,
            num_amplitudes,
            device=nu.device,
            dtype=nu.dtype,
        )
    )

    noise_imag = (
        noise_std
        * torch.randn(
            B,
            num_amplitudes,
            device=nu.device,
            dtype=nu.dtype,
        )
    )

    y_real = (
        nu[:, None]
        + noise_real
    )

    y_imag = noise_imag

    r = torch.sqrt(
        y_real**2
        + y_imag**2
    )

    return r


# ============================================================
# MOMENT ESTIMATOR
#
# 2 E[R²]² - E[R⁴] = nu⁴
#
# sigma disappears from the expression.
# ============================================================

def estimate_nu_moments(
    r,
):

    m2 = torch.mean(
        r**2,
        dim=-1,
    )

    m4 = torch.mean(
        r**4,
        dim=-1,
    )

    nu4_hat = torch.clamp(
        2.0 * m2**2 - m4,
        min=0.0,
    )

    return (
        nu4_hat ** 0.25
    )


# ============================================================
# SAMPLE A PHYSICAL-LIKE PURE-RICE BATCH
#
# rho  ~ U(0.05, 1)
# gain ~ U(0,1)
#
# nu = rho * gain
#
# SNR_ref ~ U(-10,10 dB)
#
# sigma = rho * 10^(-SNR_ref/20)
# ============================================================

def generate_training_batch(
    batch_size,
):

    rho = sample_rho_prior(
        batch_size,
        device,
        dtype,
    )

    # Proxy for |b^H a(theta)|
    #
    # Uniform here deliberately:
    # we want coverage of the whole [0,1] range.
    gain = torch.rand(
        batch_size,
        device=device,
        dtype=dtype,
    )

    nu = (
        rho
        * gain
    )

    snr_ref_db = sample_reference_snr_db(
        batch_size,
        device,
        dtype,
    )

    sigma = sigma_from_rho_snr(
        rho,
        snr_ref_db,
    )

    r = generate_direct_rice(
        nu,
        sigma,
        NUM_AMPLITUDES,
    )

    return (
        r,
        nu,
        rho,
        gain,
        sigma,
        snr_ref_db,
    )


# ============================================================
# TRAIN
# ============================================================

def train_deep_sets():

    print("\n")
    print("=" * 60)
    print("TRAINING DEEP SETS")
    print("SNR_ref ~ Uniform(-10, +10 dB)")
    print("=" * 60)

    model = RiceEstimator().to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        num_steps + 1,
    ):

        (
            r,
            nu,
            rho,
            gain,
            sigma,
            snr_ref_db,
        ) = generate_training_batch(
            batch_size
        )

        nu_hat = model(
            r
        ).squeeze(-1)

        loss = torch.mean(
            (
                nu_hat - nu
            ) ** 2
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        if step % 100 == 0:

            mean_loss = (
                running_loss
                / 100
            )

            running_loss = 0.0

            print(
                f"Step {step:5d} | "
                f"loss = {mean_loss:.6f}"
            )

    return model


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(
    model,
):

    print("\n")
    print("=" * 60)
    print("PERMUTATION INVARIANCE")
    print("=" * 60)

    model.eval()

    (
        r,
        nu,
        rho,
        gain,
        sigma,
        snr,
    ) = generate_training_batch(
        512
    )

    permutation = torch.randperm(
        NUM_AMPLITUDES,
        device=device,
    )

    with torch.no_grad():

        pred_1 = model(
            r
        ).squeeze(-1)

        pred_2 = model(
            r[:, permutation]
        ).squeeze(-1)

    difference = torch.abs(
        pred_1 - pred_2
    )

    print(
        f"Maximum difference = "
        f"{difference.max().item():.8e}"
    )

    print(
        f"Mean difference    = "
        f"{difference.mean().item():.8e}"
    )


# ============================================================
# PURE-RICE PERFORMANCE VS REFERENCE SNR
# ============================================================

def evaluate_vs_snr(
    model,
    n_per_snr=10_000,
):

    print("\n")
    print("=" * 60)
    print("PURE RICE PERFORMANCE VS SNR")
    print("=" * 60)

    model.eval()

    ds_maes = []
    moment_maes = []
    constant_maes = []

    ds_rmses = []
    moment_rmses = []

    with torch.no_grad():

        for snr_db_value in SNR_TEST_VALUES:

            # -----------------------------------------------
            # Same physical-like prior at every SNR
            # -----------------------------------------------

            rho = sample_rho_prior(
                n_per_snr,
                device,
                dtype,
            )

            gain = torch.rand(
                n_per_snr,
                device=device,
                dtype=dtype,
            )

            nu = (
                rho * gain
            )

            snr_db = torch.full(
                (n_per_snr,),
                snr_db_value,
                device=device,
                dtype=dtype,
            )

            sigma = sigma_from_rho_snr(
                rho,
                snr_db,
            )

            r = generate_direct_rice(
                nu,
                sigma,
                NUM_AMPLITUDES,
            )

            # -----------------------------------------------
            # Deep Sets
            # -----------------------------------------------

            nu_hat_ds = model(
                r
            ).squeeze(-1)

            # -----------------------------------------------
            # Moments
            # -----------------------------------------------

            nu_hat_mom = (
                estimate_nu_moments(
                    r
                )
            )

            # -----------------------------------------------
            # Constant baseline
            # -----------------------------------------------

            prior_mean = (
                nu.mean()
            )

            nu_hat_const = (
                torch.full_like(
                    nu,
                    prior_mean,
                )
            )

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            error_ds = (
                nu_hat_ds - nu
            )

            error_mom = (
                nu_hat_mom - nu
            )

            error_const = (
                nu_hat_const - nu
            )

            mae_ds = torch.mean(
                torch.abs(error_ds)
            ).item()

            mae_mom = torch.mean(
                torch.abs(error_mom)
            ).item()

            mae_const = torch.mean(
                torch.abs(error_const)
            ).item()

            rmse_ds = torch.sqrt(
                torch.mean(
                    error_ds**2
                )
            ).item()

            rmse_mom = torch.sqrt(
                torch.mean(
                    error_mom**2
                )
            ).item()

            corr_ds = torch.corrcoef(
                torch.stack(
                    [
                        nu,
                        nu_hat_ds,
                    ]
                )
            )[0, 1].item()

            ds_maes.append(
                mae_ds
            )

            moment_maes.append(
                mae_mom
            )

            constant_maes.append(
                mae_const
            )

            ds_rmses.append(
                rmse_ds
            )

            moment_rmses.append(
                rmse_mom
            )

            print(
                f"SNR_ref = {snr_db_value:5.1f} dB | "
                f"DS MAE = {mae_ds:.4f} | "
                f"DS RMSE = {rmse_ds:.4f} | "
                f"corr = {corr_ds:.4f} | "
                f"Moments MAE = {mae_mom:.4f} | "
                f"Constant MAE = {mae_const:.4f}"
            )


    # ========================================================
    # MAE VS SNR PLOT
    # ========================================================

    plt.figure()

    plt.plot(
        SNR_TEST_VALUES,
        ds_maes,
        marker="o",
        label="Deep Sets",
    )

    plt.plot(
        SNR_TEST_VALUES,
        moment_maes,
        marker="o",
        label="Moments",
    )

    plt.plot(
        SNR_TEST_VALUES,
        constant_maes,
        marker="o",
        label="Constant",
    )

    plt.xlabel(
        "Reference SNR [dB]"
    )

    plt.ylabel(
        r"MAE on $\nu$"
    )

    plt.title(
        "Rice estimation performance versus SNR"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "rice_mae_vs_snr.png",
        dpi=200,
    )

    plt.close()


    # ========================================================
    # RMSE VS SNR
    # ========================================================

    plt.figure()

    plt.plot(
        SNR_TEST_VALUES,
        ds_rmses,
        marker="o",
        label="Deep Sets",
    )

    plt.plot(
        SNR_TEST_VALUES,
        moment_rmses,
        marker="o",
        label="Moments",
    )

    plt.xlabel(
        "Reference SNR [dB]"
    )

    plt.ylabel(
        r"RMSE on $\nu$"
    )

    plt.title(
        "Rice estimation RMSE versus SNR"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "rice_rmse_vs_snr.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# PHYSICAL SIMULATOR VS SNR
#
# Here:
#
# nu = rho |b^H a(theta)|
#
# sigma = rho * 10^(-SNR_ref/20)
#
# Effective SNR:
#
# SNR_eff =
# SNR_ref + 20 log10 |b^H a(theta)|
#
# ============================================================

def evaluate_physical_vs_snr(
    model,
    n_per_snr=3000,
):

    print("\n")
    print("=" * 60)
    print("PHYSICAL SIMULATOR VS SNR")
    print("=" * 60)

    model.eval()

    s = generate_pilot_sequence(
        device,
        sequence_type="PSS",
    )

    ds_maes = []
    constant_maes = []

    with torch.no_grad():

        for snr_db_value in SNR_TEST_VALUES:

            true_values = []
            predicted_values = []
            effective_snr_values = []

            for _ in range(
                n_per_snr
            ):

                theta = sample_theta_prior(
                    1,
                    device,
                    dtype,
                )

                rho = sample_rho_prior(
                    1,
                    device,
                    dtype,
                )

                eta = (
                    2.0
                    * torch.pi
                    * torch.rand(
                        nx,
                        device=device,
                        dtype=dtype,
                    )
                )

                a = steering_vector(
                    theta,
                    params,
                )

                b = beam_from_phases(
                    eta
                )

                beam_gain = (
                    b.conj().T
                    @ a
                ).squeeze()

                gain = torch.abs(
                    beam_gain
                )

                target_nu = (
                    rho.squeeze()
                    * gain
                )

                snr_db_tensor = torch.tensor(
                    snr_db_value,
                    device=device,
                    dtype=dtype,
                )

                sigma = (
                    rho.squeeze()
                    * torch.pow(
                        10.0,
                        -snr_db_tensor / 20.0,
                    )
                )

                r = generate_amplitude_measurement(
                    theta,
                    rho,
                    eta,
                    s=s,
                    sigma=sigma,
                    params=params,
                )

                if r.ndim == 1:

                    r = r.unsqueeze(
                        0
                    )

                nu_hat = model(
                    r
                ).squeeze()

                true_values.append(
                    target_nu.item()
                )

                predicted_values.append(
                    nu_hat.item()
                )

                # Actual SNR after beamforming
                effective_snr_db = (
                    snr_db_value
                    + 20.0
                    * math.log10(
                        max(
                            gain.item(),
                            1e-12,
                        )
                    )
                )

                effective_snr_values.append(
                    effective_snr_db
                )


            true = torch.tensor(
                true_values
            )

            pred = torch.tensor(
                predicted_values
            )

            prior_mean = (
                true.mean()
            )

            constant = torch.full_like(
                true,
                prior_mean,
            )

            mae_ds = torch.mean(
                torch.abs(
                    pred - true
                )
            ).item()

            rmse_ds = torch.sqrt(
                torch.mean(
                    (
                        pred - true
                    ) ** 2
                )
            ).item()

            mae_const = torch.mean(
                torch.abs(
                    constant - true
                )
            ).item()

            corr = torch.corrcoef(
                torch.stack(
                    [
                        true,
                        pred,
                    ]
                )
            )[0, 1].item()

            effective_snr = torch.tensor(
                effective_snr_values
            )

            median_effective_snr = (
                effective_snr
                .median()
                .item()
            )

            ds_maes.append(
                mae_ds
            )

            constant_maes.append(
                mae_const
            )

            print(
                f"SNR_ref = {snr_db_value:5.1f} dB | "
                f"median SNR_eff = "
                f"{median_effective_snr:6.2f} dB | "
                f"DS MAE = {mae_ds:.4f} | "
                f"RMSE = {rmse_ds:.4f} | "
                f"corr = {corr:.4f} | "
                f"constant MAE = {mae_const:.4f}"
            )


    # ========================================================
    # PLOT
    # ========================================================

    plt.figure()

    plt.plot(
        SNR_TEST_VALUES,
        ds_maes,
        marker="o",
        label="Deep Sets",
    )

    plt.plot(
        SNR_TEST_VALUES,
        constant_maes,
        marker="o",
        label="Constant",
    )

    plt.xlabel(
        "Reference SNR [dB]"
    )

    plt.ylabel(
        r"MAE on $\nu$"
    )

    plt.title(
        "Deep Sets on physical beam model"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "physical_mae_vs_snr.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Train ONCE across the whole SNR range
    model = train_deep_sets()

    # Verify Deep Sets property
    permutation_test(
        model
    )

    # Controlled Rice experiment
    evaluate_vs_snr(
        model,
        n_per_snr=10_000,
    )

    # Actual beamforming simulator
    evaluate_physical_vs_snr(
        model,
        n_per_snr=3000,
    )

    torch.save(
        model.state_dict(),
        "rice_deepsets_multisnr.pt",
    )

    print("\nDone.")