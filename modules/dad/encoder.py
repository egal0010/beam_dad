import torch
import torch.nn as nn

from modules.dad.encoder_amp import AmplitudeDeepSet

class Encoder(nn.Module):

    def __init__(
        self,
        observation_dim=127,
        design_dim=4,
        hidden_dim=128,
        encoding_dim=32,
        amp_output_dim=16,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Deep Sets encoder for the 127 amplitudes
        #
        # r = [r_1, ..., r_127]
        #       ->
        # z_amp in R^amp_output_dim
        # ----------------------------------------------------

        self.amp = AmplitudeDeepSet(
            element_hidden_dim=64,
            element_encoding_dim=32,
            output_dim=amp_output_dim,
        )

        # ----------------------------------------------------
        # Beam representation
        #
        # eta_0 = 0 is fixed.
        #
        # Keep only K-1 relative phases and represent them
        # through cos/sin to avoid the 2*pi discontinuity.
        #
        # dimension = 2 * (K - 1)
        # ----------------------------------------------------

        eta_feature_dim = 2 * (design_dim - 1)

        # ----------------------------------------------------
        # Complete experiment representation:
        #
        # [beam features, amplitude embedding]
        # ----------------------------------------------------

        input_dim = (
            eta_feature_dim
            + amp_output_dim
        )

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.LeakyReLU(0.01),

            nn.Linear(
                hidden_dim,
                encoding_dim,
            ),
        )


    def forward(
        self,
        eta,
        r,
    ):

        # ----------------------------------------------------
        # Relative phases
        # ----------------------------------------------------

        eta_relative = eta[..., 1:]

        # [..., K-1]
        #      ->
        # [..., 2*(K-1)]

        eta_features = torch.cat(
            [
                torch.cos(eta_relative),
                torch.sin(eta_relative),
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Deep Sets amplitude encoding
        #
        # [..., 127]
        #      ->
        # [..., amp_output_dim]
        # ----------------------------------------------------

        z_amp = self.amp(r)

        # ----------------------------------------------------
        # Complete (design, observation) representation
        # ----------------------------------------------------

        x = torch.cat(
            [
                eta_features,
                z_amp,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Experiment embedding
        #
        # This is what will later be summed over the DAD
        # history.
        # ----------------------------------------------------

        return self.net(x)