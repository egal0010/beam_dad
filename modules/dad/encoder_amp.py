import torch
import torch.nn as nn


class AmplitudeDeepSet(nn.Module):

    def __init__(
        self,
        element_hidden_dim=64,
        element_encoding_dim=32,
        output_dim=16,
    ):
        super().__init__()

        # phi(r_n)
        # Même réseau appliqué à CHAQUE amplitude
        self.phi = nn.Sequential(
            nn.Linear(1, element_hidden_dim),
            nn.LeakyReLU(0.01),

            nn.Linear(element_hidden_dim, element_encoding_dim),
            nn.LeakyReLU(0.01),
        )

        # rho(sum/mean phi(r_n))
        self.rho = nn.Sequential(
            nn.Linear(element_encoding_dim, 64),
            nn.LeakyReLU(0.01),

            nn.Linear(64, output_dim),
        )

    def forward(self, r):
        """
        r : [..., 127]

        Exemple :
            [batch, 127]
        ou
            [batch, T, 127]
        """

        # [..., 127] -> [..., 127, 1]
        r = r.unsqueeze(-1)

        # Le même phi est appliqué aux 127 amplitudes
        # [..., 127, 1] -> [..., 127, element_encoding_dim]
        element_embeddings = self.phi(r)

        # Pooling permutation-invariant
        # [..., 127, element_encoding_dim]
        #               ->
        # [..., element_encoding_dim]
        pooled = element_embeddings.mean(dim=-2)

        # Résumé final de l'observation
        encoding = self.rho(pooled)

        return encoding