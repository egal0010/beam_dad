Ceci sont les logs initiaux pour les tests où on change la taille de l'historique/le nombre d'expériences avec les paramètres suivants :

history = train_dad(
    policy=policy,
    params=params,
    s=s,

    num_steps=400,

    batch_size=40,

    L=70,

    n_experiments=T,

    snr_db=0.0, #on n'entraînerait pas avec différents snr? 

    rho_grid_size=50,

    learning_rate=1e-3,

    print_every=10,
)
Avec T = 1,2,3,5,10

L'idée est de voir apd quand on a quelque chose de meilleur que random (si on fait suffisamment d'expérience, la méthode random va être aussi bonne que ce qu'on fait en terme d'approximation de l'angle etc. En effet, avec suffisamment d'échantillons aléatoire, l'inférence bayésienne va faire que notre postérior va devenir de moins en moins incertain. L'idée c'est de voir quand est ce que c'est rentable de passer par deep adaptive design.)

On a aussi ces paramètres là

from dataclasses import dataclass

@dataclass
class Params:
    Nx: int = 4
    Ny: int = 1

    Mx: int = 1
    My: int = 1

    freq: float = 28e9

    M: int = 50
    N: int = 50
    T: int = 1

    B: int = 2

    c: float = 299792458.0

    @property
    def lambda_c(self):
        return self.c / self.freq

    @property
    def d(self):
        return self.lambda_c / 2

    @property
    def K(self):
        return self.Nx * self.Ny

    @property
    def R(self):
        Q = 2** self.B
        return Q ** (self.K-1)

Donc on est toujours dans le cas 1D, avec Nx=4 et Ny=1

Et on a ça dans les tests de comparaison 
# ============================================================
# Configuration
# ============================================================

SNR_DB = 0.0

N_REAL = 200

# Nombre de contrastifs uniquement pour l'EVALUATION finale
# de g_L. Ça n'a rien à voir avec le L utilisé pendant
# l'entraînement.
L_EVAL = 256

SEED = 42



Bon, en règle générale, on voit que ça ne change pas trop, mais je suspecte que c'est parce que le réseau de l'émitteur est trop simple (on a juste une fonction affine pour l'émitteur)

On a cette architecture là grosso modo :

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        design_dim=4,
        observation_dim=127,
        hidden_dim=256,
        encoding_dim=16,
    ):
        super().__init__()

        input_dim = design_dim + observation_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoding_dim),
        )

    def forward(self, eta, r):
        """
        eta : [..., 4]
        r   : [..., 127]

        output : [..., 16]
        """

        x = torch.cat(
            [eta, r],
            dim=-1,
        )

        return self.net(x)


et pour l'emitter

import torch
import torch.nn as nn


class Emitter(nn.Module):
    """
    Transforme la représentation de l'historique en prochain design.

    Pour N = 4 :

        encoding : [..., 16]
              |
              v
        Linear(16, 4)
              |
              v
        eta_next : [..., 4]
    """

    def __init__(
        self,
        encoding_dim=16,
        design_dim=4,
    ):
        super().__init__()

        self.linear = nn.Linear(
            encoding_dim,
            design_dim,
        )

    def forward(self, encoding):
        """
        Parameters
        ----------
        encoding : Tensor [..., encoding_dim]

        Returns
        -------
        eta : Tensor [..., design_dim]
            Phases produites par la policy.
        """

        eta = self.linear(encoding)

        return eta



J'étais parti sur ça parce que c'est ce que faisais location_finding.py, mais je pense que juste une fonction affine pour la stratégie de la policy pour chercher le prochain beam par rapport à l'historique est trop faible, d'où le fait que les résultats sont pas oufs pour l'instant



