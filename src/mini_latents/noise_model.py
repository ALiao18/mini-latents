from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def m_step(
        self, 
        W: np.ndarray, 
        sum_Ezz: np.ndarray, 
        N: int,
        X2: np.ndarray,
        XtEz: np.ndarray
    ):
        """
        m-step of the EM algorithm

        Params:
        -------
        W       (d,k): weight matrix
        sum_Ezz (k,k): sum over n posterior second moment of latent variables
        N            : sample count, int
        X2      (d, ): sum_n x_nj^2
        XtEz    (d,k): X^T Ez
        """

    @abstractmethod
    def initialize(self, X: np.ndarray, psi=None):
        """
        Initialize the noise model parameters based on X

        X   (N,d): centered data
        psi      : optional starting value (scalar for isotropic, (d,) for anisotropic
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

    def initialize(self, X, psi=None) -> None:
        self.d = X.shape[1]
        self.psi = np.var(X, axis=0).mean() if psi is None else psi

    def m_step(
        self,
        W: np.ndarray,
        sum_Ezz: np.ndarray,
        N: int,
        X2: np.ndarray,
        XtEz: np.ndarray,
    ) -> None:
        """
        Noise update for pPCA. O(Ndk + dk^2)

        params
        ------
        W       (d, k): loading matrix
        sum_Ezz (k, k): sum of E[z z^T], (N, k, k) posterior second moment over N
        X2      (d,  ): sum_n x_nj^2, optional; computed from X if None
        XtEz    (d, k): X^T Ez, optional; fit() reuses m_step_W's product
        """
        d = W.shape[0]
        recon_term = X2.sum()  # scalar: sums j over (d,)
        cross_term = 2 * np.sum(
            W * XtEz
        )  # scalar: 2 sum_n (W^T x_n)^Ez_n = 2 trace(W^T X^T Ez), O(dk)
        trace_term = np.sum(
            (W.T @ W) * sum_Ezz
        )  # scalar sum_n Ez_n^T W^T W Ez_n = trace(W^T W sum_n Ezz_n), O(dk^2)

        S = recon_term - cross_term + trace_term  # scalar
        self.psi = S / (N * d)  # scalar

    def noise_as_vec(self):
        return np.full(self.d, self.psi)  # (d,)


class AnisotropicNoise(NoiseModel):
    def __init__(self, min_var_frac: float = 0.01):
        self.min_var_frac = min_var_frac
        self.d = None
        self.psi = None
        self.var_floor = None
        self.floored = None

    def initialize(self, X, psi=None) -> None:
        self.d = X.shape[1]
        self.psi = (
            np.var(X, axis=0) if psi is None else np.full(self.d, psi, dtype=float)
        )  # scalar or (d,) -> (d,)
        self.var_floor = self.min_var_frac * np.var(X, axis=0)  # (d,), O(Nd)

    def m_step(
        self,
        W: np.ndarray,
        sum_Ezz: np.ndarray,
        N: int,
        X2: np.ndarray,
        XtEz: np.ndarray,
    ) -> None:
        """
        Noise update for FA. O(Ndk + dk^2)

        psi_j = expected squared residual (R_j) of feature j, summed over samples

        R_j = recon - cross + quad
            recon = sum_n x_nj**2
            cross = 2 w_j^T (sum_n <z_n> x_nj)
            quad  = w_j^T (sum_n < z_n z_n^T>) w_j

        Params
        ------
        X       (N, d): centered data
        W       (d, k): loading matrix
        Ez      (N, k): posterior mean
        sum_Ezz (k, k): sum of E[z z^T], (N, k, k) posterior second moment over N
        X2      (d,  ): sum_n x_nj^2, optional; computed from X if None
        XtEz    (d, k): X^T Ez, optional; fit() reuses m_step_W's product
        """
        recon_term = X2  # (d,): sums n over (N, d)
        cross_term = 2 * np.sum(W * XtEz, axis=1)  # (d,): sums k over (d, k), O(dk)
        quad_term = np.sum((W @ sum_Ezz) * W, axis=1)  # (d,): sums k over (d, k)

        R = recon_term - cross_term + quad_term  # (d,)
        self.psi = R / N  # (d,)
        self.floored = np.any(self.psi < self.var_floor)
        self.psi = np.clip(self.psi, self.var_floor, None)

    def noise_as_vec(self):
        return self.psi
