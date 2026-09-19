"""Shared configuration for training and evaluation runs."""

import argparse
from pathlib import Path


CHECKPOINT_STEPS = (2000, 4000, 6000, 10000, 15000, 20000)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def default_checkpoint_dir(params):
    array_name = f"nx{params.Nx}"
    if params.Ny != 1:
        array_name += f"_ny{params.Ny}"
    return (
        Path(__file__).resolve().parents[1]
        / "training"
        / f"checkpoints_fixed_sigma_{array_name}_T{params.T}_ds"
    )
