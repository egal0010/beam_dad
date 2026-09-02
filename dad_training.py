import torch

from modules.beam_eig.params import Params

from modules.beam_eig.pilot import (
    generate_pilot_sequence,
)

from modules.dad.policy import DADPolicy

from modules.training.train_dad import train_dad


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Physical parameters
# ============================================================

params = Params(
    Nx=4,
    Ny=1,
)


# ============================================================
# Pilot
# ============================================================

s = generate_pilot_sequence(
    device=device,
    sequence_type="PSS",
)

Ns = s.numel()

print("N antennas :", params.K)
print("Ns         :", Ns)


# ============================================================
# DAD policy
# ============================================================

policy = DADPolicy(
    design_dim=params.K,
    observation_dim=Ns,
    hidden_dim=256,
    encoding_dim=64,
).to(device)


# ============================================================
# Small smoke training
# ============================================================

history = train_dad(
    policy=policy,
    params=params,
    s=s,

    num_steps=500,

    batch_size=80,

    L=70,

    n_experiments=5,

    snr_db=0.0, #on n'entraînerait pas avec différents snr? 

    rho_grid_size=50,

    learning_rate=1e-4,

    grad_clip=1.0,

    print_every=10,
)


# ============================================================
# Save
# ============================================================

torch.save(
    {
        "model_state_dict":
            policy.state_dict(),

        "history":
            history,

        "Nx":
            params.Nx,

        "Ny":
            params.Ny,

        "Ns":
            Ns,
    },
    "dad_nx4_smoke.pt",
)

print("Model saved.")