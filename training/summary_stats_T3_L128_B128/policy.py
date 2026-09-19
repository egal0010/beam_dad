import torch
import torch.nn as nn

from .encoder import Encoder
from .emitter import Emitter


class DADPolicy(nn.Module):
    """
    Deep Adaptive Design policy.

    Pour chaque élément de l'historique :

        (eta_i, r_i)
             |
          Encoder
             |
             v
            e_i

    puis

        h_t = sum_i e_i

    puis

        eta_{t+1} = Emitter(h_t)
    """

    def __init__(
        self,
        design_dim=4,
        observation_dim=127,
        hidden_dim=256,
        encoding_dim=64,
    ):
        super().__init__()

        self.design_dim = design_dim
        self.encoding_dim = encoding_dim
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(
            design_dim=design_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
        )

        self.emitter = Emitter(
            encoding_dim=encoding_dim,
            design_dim=design_dim,
        )

    def forward(
        self,
        eta_history,
        r_history,
        batch_size=None,
    ):
        """
        Parameters
        ----------
        eta_history : Tensor [B, T, N]
            Historique des phases.

        r_history : Tensor [B, T, Ns]
            Historique des observations.

        batch_size : int, optional
            Utilisé uniquement lorsque T = 0.

        Returns
        -------
        eta_next : Tensor [B, N]
            Prochain design proposé par DAD.
        """

        # -----------------------------------------
        # Cas t = 0 :
        # aucun historique disponible
        # -----------------------------------------

        if eta_history is None:

            if batch_size is None:
                raise ValueError(
                    "batch_size doit être fourni lorsque "
                    "eta_history=None."
                )

            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            summary = torch.zeros(
                batch_size,
                self.encoding_dim,
                device=device,
                dtype=dtype,
            )

        else:

            if eta_history.ndim != 3:
                raise ValueError(
                    "eta_history doit avoir la forme [B, T, N]."
                )

            B, T, N = eta_history.shape

            if N != self.design_dim:
                raise ValueError(
                    f"Design dimension attendue : "
                    f"{self.design_dim}, reçue : {N}."
                )

            # -----------------------------------------
            # Historique vide :
            #
            # eta_history.shape = [B, 0, N]
            # -----------------------------------------

            if T == 0:

                summary = torch.zeros(
                    B,
                    self.encoding_dim,
                    device=eta_history.device,
                    dtype=eta_history.dtype,
                )

            else:

                # [B, T, N] + [B, T, Ns]
                #
                #            |
                #          Encoder
                #            |
                #            v
                #
                #       [B, T, encoding_dim]

                encodings = self.encoder(
                    eta_history,
                    r_history,
                )

                # permutation-invariant aggregation
                #
                # [B, T, encoding_dim]
                #          ↓ sum
                # [B, encoding_dim]

                summary = encodings.sum(dim=1)

        # -----------------------------------------
        # Choix du prochain beam
        # -----------------------------------------

        eta_next = self.emitter(summary)

        return eta_next