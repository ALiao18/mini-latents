from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def m_step(self, X: np.ndarray, W, Ez: np.ndarray, Ezz: np.ndarray, X2=None, XtEz=None):
        """
        m-step of the EM algorithm

        Params:
        -------
        X       (n,d): data matrix
        W       (d,k): weight matrix
        Ez      (n,k): expected mean of posterior distribution of latent variables
        sum_Ezz (k,k): sum over n posterior second moment of latent variables
        X2      (d,  ): sum_n x_nj^2, optional; computed from X if None
        XtEz    (d,k): X^T Ez, optional; computed from X, Ez if None
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

    def m_step(self, X: np.ndarray, W: np.ndarray, Ez: np.ndarray, sum_Ezz: np.ndarray, X2=None, XtEz=None) -> None:
        """
        Noise update for pPCA. O(Ndk + dk^2)

        params
        ------
        X       (N, d): centered data
        W       (d, k): loading matrix
        Ez      (N, k): posterior mean
        sum_Ezz (k, k): sum of E[z z^T], (N, k, k) posterior second moment over N
        X2      (d,  ): sum_n x_nj^2, optional; computed from X if None
        XtEz    (d, k): X^T Ez, optional; fit() reuses m_step_W's product
        """
        N, d = X.shape

        if X2 is None:
            X2 = np.sum(X**2, axis=0)  # (d,)
        recon_term = X2.sum()  # scalar: sums j over (d,)
        if XtEz is None:
            XtEz = X.T @ Ez  # (d,k), O(Ndk)
        cross_term = 2 * np.sum(W * XtEz)  # scalar: 2 sum_n (W^T x_n)^T Ez_n = 2 trace(W^T X^T Ez), O(dk)
        trace_term = np.trace(W.T @ W @ sum_Ezz)  # scalar: trace of (k, k)

        S = recon_term - cross_term + trace_term  # scalar
        self.psi = S / (N * d)  # scalar

    def noise_as_vec(self):
        return np.full(self.d, self.psi)  # (d,)


class AnisotropicNoise(NoiseModel):
    def __init__(self):
        self.d = None
        self.psi = None

    def initialize(self, X, psi=None):
        self.d = X.shape[1]
        self.psi = (
            np.var(X, axis=0) if psi is None else np.full(self.d, psi, dtype=float)
        )  # scalar or (d,) -> (d,)

    def m_step(self, X: np.ndarray, W: np.ndarray, Ez: np.ndarray, sum_Ezz: np.ndarray, X2=None, XtEz=None) -> None:
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
        N, self.d = X.shape

        if X2 is None:
            X2 = np.sum(X**2, axis=0)  # (d,)
        recon_term = X2  # (d,): sums n over (N, d)
        if XtEz is None:
            XtEz = X.T @ Ez  # (d,k), O(Ndk)
        cross_term = 2 * np.sum(W * XtEz, axis=1)  # (d,): sums k over (d, k), O(dk)
        quad_term = np.sum((W @ sum_Ezz) * W, axis=1)  # (d,): sums k over (d, k)

        R = recon_term - cross_term + quad_term  # (d,)
        self.psi = R / N  # (d,)

    def noise_as_vec(self):
        return self.psi
