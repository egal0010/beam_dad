L'architecture du nn est comme suit :
j'ai choisi N et M = 400.

class Encoder(nn.Module):

    def __init__(
        self,
        design_dim=4,
        observation_dim=127,
        hidden_dim=256,
        encoding_dim=64,
    ):
        super().__init__()

        input_dim = (2*(design_dim-1) + 2) #*2 car on concatene cos et sin et +2 car on concatene mean et var de r

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
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

        mean_r = torch.mean(r, dim=-1, keepdim=True)
        mean_r2= torch.square(torch.mean(r, dim=-1, keepdim=True)) #will try this later on
        var_r = torch.var(r, dim=-1, keepdim=True, correction=False)

        r_features = torch.cat(
            [
                mean_r,
                var_r,
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


et pour l'émetteur: 

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
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
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


les paramètres d'entrianement sont 

history = train_dad(
    policy=policy,
    params=params,
    s=s,

    num_steps=1000,

    batch_size=128,

    L=128,

    n_experiments=10,

    snr_db=0.0, #on n'entraînerait pas avec différents snr? 

    rho_grid_size=50,

    learning_rate=1e-4,

    grad_clip=1.0,

    print_every=25,
)