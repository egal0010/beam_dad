import torch
import torch.nn as nn


class Emitter(nn.Module):
    """
    Transforme la représentation de l'historique en prochain design.

    Pour N = 4 :

        encoding : [..., 64]
              |
              v
        Linear(64, 4)
              |
              v
        eta_next : [..., 4]
    """

    def __init__(
        self,
        encoding_dim=64,
        design_dim=4,
    ):
        super().__init__()

        self.design_dim = design_dim

        self.emitter = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, design_dim-1),
        )

    def forward(self, encoding):

        # [..., K-1]
        eta_relative = self.emitter(
            encoding
        )

        # eta_0 = 0
        eta_0 = torch.zeros(
            *eta_relative.shape[:-1],
            1,
            dtype=eta_relative.dtype,
            device=eta_relative.device,
        )

        # [..., K]
        eta = torch.cat(
            [
                eta_0,
                eta_relative,
            ],
            dim=-1,
        )

        return eta