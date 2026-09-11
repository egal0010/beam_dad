import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        design_dim=4,
        observation_dim=127,
        hidden_dim=256,
        encoding_dim=64,
    ):
        super().__init__()

        input_dim = 2 * (design_dim - 1) + 6  # cos/sin des phases + amplitudes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, encoding_dim),
        )

    def forward(self, eta, r):

        # La première phase est toujours 0 :
        # on ne garde que les K-1 phases relatives
        eta_relative = eta[..., 1:]

        eta_features = torch.cat(
            [
                torch.cos(eta_relative),
                torch.sin(eta_relative),
            ],
            dim=-1,
        )
        #information redondantes pour les moments d'ordre 2 et 4, mais on les garde pour l'instant
        mean_r = torch.mean(r, dim=-1, keepdim=True)
        var_r = torch.var(r, dim=-1, keepdim=True, correction=False)
        skew_r = torch.mean(torch.pow(r - mean_r, 3), dim=-1, keepdim=True) / (var_r + 1e-8) ** 1.5
        kurt_r = torch.mean(torch.pow(r - mean_r, 4), dim=-1, keepdim=True) / (var_r + 1e-8) ** 2
        m2= torch.mean(r**2, dim=-1, keepdim=True)
        m4= torch.mean(r**4, dim=-1, keepdim=True)
        nu2_hat= torch.sqrt(torch.clamp(2*m2**2 - m4, min=1e-8, max=1e6))
        sigma2_hat= torch.clamp(m2 - nu2_hat, min=1e-8, max=1e6)

        r_features = torch.cat(
            [
                mean_r,
                var_r,
                skew_r,
                kurt_r,
                nu2_hat,
                sigma2_hat,
            ],
            dim=-1,
        )
        x = torch.cat(
            [
                eta_features,
                r_features,
            ],
            dim=-1,
        )

        return self.net(x)