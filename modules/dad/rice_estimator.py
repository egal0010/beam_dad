import torch
import torch.nn as nn
from  modules.dad.encoder_amp import AmplitudeDeepSet

class RiceEstimator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = AmplitudeDeepSet(
            element_hidden_dim=64,
            element_encoding_dim=32,
            output_dim=16,
        )

        self.head = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, r):

        z = self.encoder(r)

        nu_hat = self.head(z).squeeze(-1)

        return nu_hat