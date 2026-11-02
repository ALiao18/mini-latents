from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def m_step(
        self,
        W_new: np.ndarray,
        SB: np.ndarray,
        diagS: np.ndarray,
    ):
        """
        m-step of the EM algorithm

        Params:
        -------
        W_new (d, k):
            updated loading matrix from the M-step
        SB (d, k):
            sum_i x_i Ez_i^T / N
        diagS (d, ):
            diag(S), S = Xc^T Xc / N
        """

    @abstractmethod
    def initialize(self, diagS: np.ndarray, psi=None):
        """
        Initialize the noise model parameters based on X

        diagS: (d, ) ndarray
            diag(S), S = Xc^T Xc / N
        psi: np.ndarray | None
            optional starting value (scalar for isotropic, (d,) for anisotropic
        """

    @abstractmethod
    def noise_as_vec(self):
        """
        returns diagonal as a (d,) vector
        """

    def as_matrix(self):
        """
        Returns dxd noise covariance matrix
        """
        return np.diag(self.noise_as_vec())


class IsotropicNoise(NoiseModel):
    def __init__(self):
        self.d = None
        self.psi = None

    def initialize(self, diagS: np.ndarray, psi=None) -> None:
        self.d = diagS.shape[0]
        self.psi = diagS.mean() if psi is None else psi

    def m_step(self, W_new: np.ndarray, SB: np.ndarray, diagS: np.ndarray) -> None:
        """Noise update for pPCA O(dk)"""
        self.psi = (diagS.sum() - np.sum(W_new * SB)) / self.d

    def noise_as_vec(self):
        return np.full(self.d, self.psi)  # (d,)


class AnisotropicNoise(NoiseModel):
    def __init__(self, min_var_frac: float = 0.01):
        self.min_var_frac = min_var_frac
        self.d = None
        self.psi = None
        self.var_floor = None
        self.floored = None

    def initialize(self, diagS: np.ndarray, psi=None) -> None:
        self.d = diagS.shape[0]
        self.psi = diagS.copy() if psi is None else np.full(self.d, psi, dtype=float)
        self.var_floor = self.min_var_frac * diagS  # (d,), O(d)

    def m_step(
        self,
        W_new: np.ndarray,
        SB: np.ndarray,
        diagS: np.ndarray,
    ) -> None:
        """Noise update for FA. O(dk)"""
        self.psi = diagS - np.sum(W_new * SB, axis=1)  # (d,)
        self.floored = np.any(self.psi < self.var_floor)
        self.psi = np.clip(self.psi, self.var_floor, None)

    def noise_as_vec(self):
        return self.psi
