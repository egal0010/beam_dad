from dataclasses import dataclass

@dataclass
class Params:
    Nx: int = 8
    Ny: int = 1

    Mx: int = 1
    My: int = 1

    freq: float = 28e9

    M: int = 300
    N: int = 300
    T: int = 3

    R: int | None = None
    B: int = 2

    c: float = 299792458.0

    def __post_init__(self):
        if self.R is None or self.R <= 0 or self.R > 2 ** (self.B * (self.K - 1)):
            Q = 2 ** self.B
            self.R = Q ** (self.K - 1)

    @property
    def lambda_c(self):
        return self.c / self.freq

    @property
    def d(self):
        return self.lambda_c / 2

    @property
    def K(self):
        return self.Nx * self.Ny
