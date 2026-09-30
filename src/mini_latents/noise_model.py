from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    @abstractmethod
    def m_step(self, X: np.ndarray, W, Ez: np.ndarray, Ezz: np.ndarray):
        """
        m-step of the EM algorithm

        Params:
        -------
        X:   data matrix (n_samples, n_features)
        W:   weight matrix (n_features, n_components)
        Ez:  expected mean of posterior distribution of latent variables (n_samples, n_components)
        Ezz: posterior second moment of latent variables (n_samples, n_components, n_components)
        """

    @abstractmethod
    def initialize(self, X: np.ndarray):
        """
        Initialize the noise model parameters based on X
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

    def initialize(self, X) -> None:
        self.d = X.shape[1]
        self.psi = np.var(X, axis=0).mean()

    def m_step(self, X, W, Ez, sum_Ezz) -> None:
        """
        Noise update for pPCA. O(Ndk + dk^2)

        params
        ------
        X       (N, d): centered data
        W       (d, k): loading matrix
        Ez      (N, k): posterior mean
        sum_Ezz (k, k): sum of E[z z^T], (N, k, k) posterior second moment over N
        """
        N, d = X.shape

        recon_term = np.sum(X**2)                   # scalar: sums n over (N,d)
        cross_term = 2 * np.sum((X @ W) * Ez)       # scalar: sums n over (N,k)
        trace_term = np.trace(W.T @ W @ sum_Ezz)    # scalar: trace of (k, k)
        
        S = recon_term - cross_term + trace_term    # scalar
        self.psi = S / (N * d)                      # scalar

    def noise_as_vec(self):
        return np.full(self.d, self.psi)  # (d,)


class AnisotropicNoise(NoiseModel):
    def __init__(self):
        self.d = None
        self.psi = None

    def initialize(self, X):
        self.d = X.shape[1]
        self.psi = np.var(X, axis=0)

    def m_step(self, X, W, Ez, sum_Ezz) -> None:
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
        """
        N, self.d = X.shape

        recon_term = np.sum(X**2, axis=0)                     # scalar: sums n over (N, d)
        cross_term = 2 * np.sum(W * (X.T @ Ez), axis = 1)     # scalar: sums k over (d, k)
        quad_term  = np.sum((W @ sum_Ezz) * W, axis = 1)      # scalar: sums k over (d, k)

        R = recon_term - cross_term + quad_term     # scalar
        self.psi = R / N                            # scalar 

    def noise_as_vec(self):
        return self.psi
