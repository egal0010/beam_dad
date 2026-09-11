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

# sigma = sqrt(E[|w|^2])
sigma = 1.0

nu_min = 0.0
nu_max =1

batch_size = 512

num_steps = 50000

num_test_samples = 20_000

learning_rate = 1e-3

nx = 8


print(f"Device: {device}")


# ============================================================
# PRIORS FOR PHYSICAL MODEL
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


# ============================================================
# DIRECT RICE SIMULATOR
#
# r_n = |nu + w_n|
#
# with
#
# w ~ CN(0, sigma^2)
#
# => Re(w), Im(w) ~ N(0, sigma^2 / 2)
#
# This corresponds to:
#
# p(r|nu)
# = 2r/sigma^2
#   exp(-(r^2 + nu^2)/sigma^2)
#   I0(2 r nu / sigma^2)
# ============================================================

def generate_direct_rice(
    nu,
    num_amplitudes=127,
    sigma=1.0,
):
    """
    nu : [B]

    returns:
        r : [B, num_amplitudes]
    """

    B = nu.shape[0]

    noise_std = sigma / math.sqrt(2.0)

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
# E[R²] = nu² + sigma²
#
# E[R⁴] = nu⁴ + 4 nu² sigma² + 2 sigma⁴
#
# Hence:
#
# 2 E[R²]^2 - E[R⁴] = nu⁴
# ============================================================

def estimate_nu_moments(r):

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
# 1. BASIC RICE SANITY CHECK
# ============================================================

def sanity_check_direct_rice():

    print("\n")
    print("=" * 60)
    print("DIRECT RICE SANITY CHECK")
    print("=" * 60)

    nu_values = [
        0.0,
        0.2,
        0.5,
        1.0,
        2.0,
        4.0,
    ]

    num_mc = 10_000

    for nu_value in nu_values:

        nu = torch.full(
            (num_mc,),
            nu_value,
            device=device,
            dtype=dtype,
        )

        r = generate_direct_rice(
            nu,
            NUM_AMPLITUDES,
            sigma,
        )

        mean_r = r.mean().item()

        mean_r2 = (
            r**2
        ).mean().item()

        expected_r2 = (
            nu_value**2
            + sigma**2
        )

        print(
            f"nu = {nu_value:4.1f} | "
            f"E[r] ~= {mean_r:.4f} | "
            f"E[r²] ~= {mean_r2:.4f} | "
            f"theory E[r²] = {expected_r2:.4f}"
        )


# ============================================================
# 2. CHECK YOUR PHYSICAL SIMULATOR
#
# If the target_nu is really the Rice non-centrality,
# we should approximately have
#
# mean(r²) = nu² + sigma²
#
# for the PSS amplitudes.
# ============================================================

def sanity_check_physical_simulator(
    num_samples=3000,
):

    print("\n")
    print("=" * 60)
    print("PHYSICAL SIMULATOR CONSISTENCY CHECK")
    print("=" * 60)

    s = generate_pilot_sequence(
        device,
        sequence_type="PSS",
    )

    target_nus = []
    measured_m2s = []
    expected_m2s = []

    for aa in range(num_samples):

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
            eta,
        )

        beam_gain = (
            b.conj().T @ a
        ).squeeze()

        target_nu = (
            rho.squeeze()
            * torch.abs(beam_gain)
        )
        if aa==11 or aa==22:
            wrong_gain = torch.sum(
                torch.conj(b) * a.squeeze()
            )

            correct_gain = (
                b.conj().T @ a
            ).squeeze()

            print(
                "WRONG   =", torch.abs(wrong_gain).item()
            )

            print(
                "CORRECT =", torch.abs(correct_gain).item()
            )

        r = generate_amplitude_measurement(
            theta,
            rho,
            eta,
            s=s,
            sigma=sigma,
            params=params,
        )

        r = r.squeeze()

        m2 = torch.mean(
            r**2
        )

        expected_m2 = (
            target_nu**2
            + sigma**2
        )

        target_nus.append(
            target_nu.item()
        )

        measured_m2s.append(
            m2.item()
        )

        expected_m2s.append(
            expected_m2.item()
        )

    target_nus = torch.tensor(
        target_nus
    )

    measured_m2s = torch.tensor(
        measured_m2s
    )

    expected_m2s = torch.tensor(
        expected_m2s
    )

    mae_m2 = torch.mean(
        torch.abs(
            measured_m2s
            - expected_m2s
        )
    )

    correlation = torch.corrcoef(
        torch.stack(
            [
                measured_m2s,
                expected_m2s,
            ]
        )
    )[0, 1]

    print(
        f"MAE between measured m2 "
        f"and nu² + sigma² = "
        f"{mae_m2.item():.6f}"
    )

    print(
        f"Correlation = "
        f"{correlation.item():.6f}"
    )

    plt.figure()

    plt.scatter(
        expected_m2s.numpy(),
        measured_m2s.numpy(),
        s=8,
        alpha=0.4,
    )

    limit = max(
        expected_m2s.max().item(),
        measured_m2s.max().item(),
    )

    plt.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
    )

    plt.xlabel(
        r"Theoretical $\nu^2+\sigma^2$"
    )

    plt.ylabel(
        r"Measured $\frac{1}{127}\sum r_n^2$"
    )

    plt.title(
        "Physical simulator consistency"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "physical_simulator_consistency.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# 3. TRAIN DEEP SETS ON PURE RICE
# ============================================================

def train_deep_sets():

    print("\n")
    print("=" * 60)
    print("TRAINING DEEP SETS ON PURE RICE")
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

        # ----------------------------------------------------
        # Balanced nu prior
        # ----------------------------------------------------

        nu = (
            nu_min
            + (nu_max - nu_min)
            * torch.rand(
                batch_size,
                device=device,
                dtype=dtype,
            )
        )

        # ----------------------------------------------------
        # Generate Rice observations
        # ----------------------------------------------------

        r = generate_direct_rice(
            nu,
            NUM_AMPLITUDES,
            sigma,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        nu_hat = model(
            r
        ).squeeze(-1)

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        loss = torch.mean(
            (
                nu_hat
                - nu
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
                f"mean loss = "
                f"{mean_loss:.6f}"
            )

    return model


# ============================================================
# 4. PERMUTATION INVARIANCE TEST
# ============================================================

def permutation_test(model):

    print("\n")
    print("=" * 60)
    print("PERMUTATION INVARIANCE TEST")
    print("=" * 60)

    model.eval()

    nu = (
        nu_min
        + (nu_max - nu_min)
        * torch.rand(
            256,
            device=device,
            dtype=dtype,
        )
    )

    r = generate_direct_rice(
        nu,
        NUM_AMPLITUDES,
        sigma,
    )

    permutation = torch.randperm(
        NUM_AMPLITUDES,
        device=device,
    )

    r_permuted = (
        r[:, permutation]
    )

    with torch.no_grad():

        pred_1 = model(
            r
        ).squeeze(-1)

        pred_2 = model(
            r_permuted
        ).squeeze(-1)

    difference = torch.abs(
        pred_1 - pred_2
    )

    print(
        "Maximum prediction difference "
        "after permutation:"
    )

    print(
        difference.max().item()
    )

    print(
        "Mean prediction difference:"
    )

    print(
        difference.mean().item()
    )


# ============================================================
# 5. TEST SET
#
# Compare:
#
# - Deep Sets
# - moment estimator
# - constant prior estimator
#
# For nu ~ Uniform(0,5),
#
# E[nu] = 2.5
#
# so the optimal constant under MSE is 2.5.
# ============================================================

def evaluate_direct_rice(
    model,
):

    print("\n")
    print("=" * 60)
    print("DIRECT RICE TEST SET")
    print("=" * 60)

    model.eval()

    all_true = []
    all_ds = []
    all_moments = []
    all_constant = []

    remaining = (
        num_test_samples
    )

    with torch.no_grad():

        while remaining > 0:

            B = min(
                batch_size,
                remaining,
            )

            remaining -= B

            nu = (
                nu_min
                + (nu_max - nu_min)
                * torch.rand(
                    B,
                    device=device,
                    dtype=dtype,
                )
            )

            r = generate_direct_rice(
                nu,
                NUM_AMPLITUDES,
                sigma,
            )

            nu_hat_ds = model(
                r
            ).squeeze(-1)

            nu_hat_moments = (
                estimate_nu_moments(
                    r
                )
            )

            nu_hat_constant = (
                torch.full_like(
                    nu,
                    (
                        nu_min
                        + nu_max
                    ) / 2.0,
                )
            )

            all_true.append(
                nu.cpu()
            )

            all_ds.append(
                nu_hat_ds.cpu()
            )

            all_moments.append(
                nu_hat_moments.cpu()
            )

            all_constant.append(
                nu_hat_constant.cpu()
            )

    true = torch.cat(
        all_true
    )

    ds = torch.cat(
        all_ds
    )

    moments = torch.cat(
        all_moments
    )

    constant = torch.cat(
        all_constant
    )


    # ========================================================
    # METRICS
    # ========================================================

    def metrics(
        prediction,
    ):

        error = (
            prediction
            - true
        )

        mae = torch.mean(
            torch.abs(error)
        )

        rmse = torch.sqrt(
            torch.mean(
                error**2
            )
        )

        correlation = torch.corrcoef(
            torch.stack(
                [
                    true,
                    prediction,
                ]
            )
        )[0, 1]

        return (
            mae.item(),
            rmse.item(),
            correlation.item(),
        )


    mae_ds, rmse_ds, corr_ds = (
        metrics(ds)
    )

    mae_mom, rmse_mom, corr_mom = (
        metrics(moments)
    )

    mae_const, rmse_const, corr_const = (
        metrics(constant)
    )


    print(
        "\nDeep Sets"
    )

    print(
        f"MAE  = {mae_ds:.6f}"
    )

    print(
        f"RMSE = {rmse_ds:.6f}"
    )

    print(
        f"Corr = {corr_ds:.6f}"
    )


    print(
        "\nMoment estimator"
    )

    print(
        f"MAE  = {mae_mom:.6f}"
    )

    print(
        f"RMSE = {rmse_mom:.6f}"
    )

    print(
        f"Corr = {corr_mom:.6f}"
    )


    print(
        "\nConstant estimator"
    )

    print(
        f"MAE  = {mae_const:.6f}"
    )

    print(
        f"RMSE = {rmse_const:.6f}"
    )

    print(
        f"Corr = {corr_const:.6f}"
    )


    # ========================================================
    # SCATTER
    # ========================================================

    n_plot = min(
        3000,
        len(true),
    )

    plt.figure()

    plt.scatter(
        true[:n_plot],
        ds[:n_plot],
        s=8,
        alpha=0.4,
        label="Deep Sets",
    )

    plt.plot(
        [nu_min, nu_max],
        [nu_min, nu_max],
        linestyle="--",
        label="Ideal",
    )

    plt.xlabel(
        r"True $\nu$"
    )

    plt.ylabel(
        r"Estimated $\hat{\nu}$"
    )

    plt.title(
        "Deep Sets Rice estimation"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "rice_deepsets_scatter.png",
        dpi=200,
    )

    plt.close()


    # ========================================================
    # BINNED MAE
    # ========================================================

    bins = torch.linspace(
        nu_min,
        nu_max,
        11,
    )

    bin_centers = (
        bins[:-1]
        + bins[1:]
    ) / 2

    mae_bins_ds = []
    mae_bins_mom = []

    print("\n")
    print("=" * 60)
    print("MAE BY NU RANGE")
    print("=" * 60)

    for i in range(
        len(bins) - 1
    ):

        lower = bins[i]
        upper = bins[i + 1]

        mask = (
            (true >= lower)
            & (true < upper)
        )

        ds_bin_mae = torch.mean(
            torch.abs(
                ds[mask]
                - true[mask]
            )
        )

        mom_bin_mae = torch.mean(
            torch.abs(
                moments[mask]
                - true[mask]
            )
        )

        mae_bins_ds.append(
            ds_bin_mae.item()
        )

        mae_bins_mom.append(
            mom_bin_mae.item()
        )

        print(
            f"nu in "
            f"[{lower.item():.1f}, "
            f"{upper.item():.1f}) | "
            f"DS MAE = "
            f"{ds_bin_mae.item():.4f} | "
            f"Moments MAE = "
            f"{mom_bin_mae.item():.4f}"
        )


    plt.figure()

    plt.plot(
        bin_centers.numpy(),
        mae_bins_ds,
        marker="o",
        label="Deep Sets",
    )

    plt.plot(
        bin_centers.numpy(),
        mae_bins_mom,
        marker="o",
        label="Moments",
    )

    plt.xlabel(
        r"$\nu$"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "Estimation error versus Rice non-centrality"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "rice_mae_by_nu.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# 6. TEST THE PURE-RICE-TRAINED NETWORK
#    ON YOUR REAL PHYSICAL SIMULATOR
#
# This is extremely useful:
#
# If pure Rice works but physical model fails,
# then there is probably a mismatch between target_nu
# and the simulator.
# ============================================================

def evaluate_on_physical_model(
    model,
    num_samples=5000,
):

    print("\n")
    print("=" * 60)
    print("PURE-RICE NETWORK ON PHYSICAL SIMULATOR")
    print("=" * 60)

    model.eval()

    s = generate_pilot_sequence(
        device,
        sequence_type="PSS",
    )

    true_values = []
    ds_values = []
    moment_values = []

    with torch.no_grad():

        for _ in range(
            num_samples
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
            ).squeeze()

            b = beam_from_phases(
                eta
            )

            beam_gain = (
                b.conj().T @ a
            ).squeeze()

            target_nu = (
                rho.squeeze()
                * torch.abs(beam_gain)
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
                r = r.unsqueeze(0)

            nu_hat_ds = model(
                r
            ).squeeze()

            nu_hat_moment = (
                estimate_nu_moments(
                    r
                ).squeeze()
            )

            true_values.append(
                target_nu.item()
            )

            ds_values.append(
                nu_hat_ds.item()
            )

            moment_values.append(
                nu_hat_moment.item()
            )


    true = torch.tensor(
        true_values
    )

    ds = torch.tensor(
        ds_values
    )

    moments = torch.tensor(
        moment_values
    )


    # --------------------------------------------------------
    # Constant baseline
    #
    # We use empirical mean of the physical prior here only
    # as a diagnostic baseline.
    # --------------------------------------------------------

    prior_mean = true.mean()

    constant = torch.full_like(
        true,
        prior_mean,
    )


    def print_metrics(
        name,
        prediction,
    ):

        error = (
            prediction - true
        )

        mae = torch.mean(
            torch.abs(error)
        )

        rmse = torch.sqrt(
            torch.mean(
                error**2
            )
        )

        corr = torch.corrcoef(
            torch.stack(
                [
                    true,
                    prediction,
                ]
            )
        )[0, 1]

        print(
            f"\n{name}"
        )

        print(
            f"MAE  = "
            f"{mae.item():.6f}"
        )

        print(
            f"RMSE = "
            f"{rmse.item():.6f}"
        )

        print(
            f"Corr = "
            f"{corr.item():.6f}"
        )


    print(
        f"\nPhysical prior mean nu = "
        f"{prior_mean.item():.6f}"
    )

    print_metrics(
        "Deep Sets",
        ds,
    )

    print_metrics(
        "Moments",
        moments,
    )

    print_metrics(
        "Constant",
        constant,
    )


    # --------------------------------------------------------
    # SCATTER
    # --------------------------------------------------------

    plt.figure()

    plt.scatter(
        true,
        ds,
        s=8,
        alpha=0.4,
    )

    max_value = max(
        true.max().item(),
        ds.max().item(),
    )

    plt.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--",
    )

    plt.xlabel(
        r"True $\nu$"
    )

    plt.ylabel(
        r"Deep Sets $\hat{\nu}$"
    )

    plt.title(
        "Pure-Rice-trained Deep Sets on physical simulator"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "physical_deepsets_scatter.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test A:
    # Is our direct Rice generator behaving correctly?
    # --------------------------------------------------------

    sanity_check_direct_rice()


    # --------------------------------------------------------
    # Test B:
    # Does your REAL simulator obey the same Rice model?
    # --------------------------------------------------------

    sanity_check_physical_simulator(
        num_samples=3000,
    )


    # --------------------------------------------------------
    # Test C:
    # Can Deep Sets learn nu from pure Rice observations?
    # --------------------------------------------------------

    model = train_deep_sets()


    # --------------------------------------------------------
    # Test D:
    # Is the architecture truly permutation invariant?
    # --------------------------------------------------------

    permutation_test(
        model
    )


    # --------------------------------------------------------
    # Test E:
    # Deep Sets vs moments vs constant
    # on balanced pure Rice data
    # --------------------------------------------------------

    evaluate_direct_rice(
        model
    )


    # --------------------------------------------------------
    # Test F:
    # Does the SAME network work on your physical simulator?
    # --------------------------------------------------------

    evaluate_on_physical_model(
        model,
        num_samples=5000,
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        "rice_deepsets_validation.pt",
    )

    print("\nDone.")