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
    T: int = 10

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