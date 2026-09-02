import itertools
import torch


def generate_quantized_codebook(
    K,
    B,
    device,
):
    """
    Returns eta candidates as a column stack [K, R].
    """

    Q = 2**B

    combinations = list(
        itertools.product(
            range(Q),
            repeat=K - 1,
        )
    )

    eta_grid = torch.zeros(
        K,
        len(combinations),
        dtype=torch.float32,
        device=device,
    )

    for i, combination in enumerate(
        combinations
    ):
        indices = torch.tensor(
            combination,
            dtype=torch.float32,
            device=device,
        )

        eta_grid[1:, i] = (
            2 * torch.pi
            / Q
            * indices
        )

    return eta_grid

