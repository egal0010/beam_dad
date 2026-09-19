import argparse
from contextlib import redirect_stdout, redirect_stderr
import sys
from pathlib import Path

import torch

from modules.beam_eig.params import Params
from modules.beam_eig.pilot import generate_pilot_sequence
from modules.dad.policy import DADPolicy
from modules.training.train_dad import train_dad


class Tee:
    def __init__(self, console, log):
        self.console = console
        self.log = log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--T",
        nargs="+",
        type=int,
        default=[5],
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=15000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--L",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0,0.1,0.3,0.4,0.6],
    )

    output_root = (
        Path(__file__).resolve().parents[1]
        / "model_checkpoints/mean_var_loss/checkpoints_nx8"
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        default=output_root / "logs",
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=output_root / "models",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Same pilot for every T
    s = generate_pilot_sequence(
        device=device,
        sequence_type="PSS",
    )

    for alpha in args.alphas:
        alpha_folder = f"alpha_{alpha}"
        log_dir = args.log_root / alpha_folder
        log_dir.mkdir(parents=True, exist_ok=True)

        for T in args.T:
            log_path = log_dir / f"T{T}_L{args.L}_B{args.batch_size}.log"
            with log_path.open("w", buffering=1) as log, \
                    redirect_stdout(Tee(sys.stdout, log)), \
                    redirect_stderr(Tee(sys.stderr, log)):
                # Same initialization convention for each horizon
                torch.manual_seed(42)

                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                params = Params(
                    Nx=8,
                    Ny=1,
                    T=T,
                )

                checkpoint_dir = (
                    args.checkpoint_root
                    / alpha_folder
                    / f"T{T}_L{args.L}_B{args.batch_size}"
                )

                checkpoint_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                print("\n" + "=" * 60)
                print(f"Alpha = {alpha}")
                print(f"T = {T}")
                print(f"L = {args.L}")
                print(f"Batch size = {args.batch_size}")
                print(f"Steps = {args.num_steps}")
                print(f"Checkpoints = {checkpoint_dir}")
                print("=" * 60)

                policy = DADPolicy(
                    design_dim=params.K,
                    observation_dim=s.numel(),
                    hidden_dim=256,
                    encoding_dim=64,
                ).to(device)

                history = train_dad(
                    policy=policy,
                    params=params,
                    s=s,

                    num_steps=args.num_steps,

                    batch_size=args.batch_size,
                    L=args.L,

                    snr_db=0.0,
                    rho_grid_size=50,

                    alpha=alpha,
                    learning_rate=5e-5,
                    grad_clip=1.0,

                    print_every=20,

                    checkpoint_dir=checkpoint_dir,
                )

                # Final checkpoint
                torch.save(
                    {
                        "model_state_dict": policy.state_dict(),
                        "history": history,

                        "Nx": params.Nx,
                        "Ny": params.Ny,
                        "Ns": s.numel(),
                        "T": T,

                        "hidden_dim": policy.hidden_dim,
                        "encoding_dim": policy.encoding_dim,

                        "encoder_type": ("deepsets" if hasattr(policy.encoder, "amp") else "summary_stats"),

                        "snr_db": 0.0,
                        "sigma_mode": "sigma_fixed",

                        "alpha": alpha,
                        "L": args.L,
                        "batch_size": args.batch_size,
                        "num_steps": args.num_steps,
                    },
                    checkpoint_dir / f"dad_T{T}_final.pt",
                )

                print(f"T={T} finished.")

                del policy

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()