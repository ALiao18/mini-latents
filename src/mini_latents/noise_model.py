from abc import ABC, abstractmethod
import numpy as np

class NoiseModel(ABC):
    @abstractmethod
    def m_step(self, X: np.ndarray, W, Ez: np.ndarray, Ezz: np.ndarray):
        '''
        m-step of the EM algorithm

        Params:
        -------
        X:   data matrix (n_samples, n_features)
        W:   weight matrix (n_features, n_components)
        Ez:  expected mean of posterior distribution of latent variables (n_samples, n_components)
        Ezz: posterior second moment of latent variables (n_samples, n_components, n_components)
        '''

    @abstractmethod
    def as_matrix(self) -> np.ndarray:
        '''
        Return the noise covariance matrix
        '''

    @abstractmethod
    def initialize(self, X: np.ndarray):
        '''
        Initialize the noise model parameters based on X
        '''

class IsotropicNoise(NoiseModel):
    def __init__(self, d):
        self.d = d
        self.psi = np.ones(d)

    def m_step(self, X, W, Ez, Ezz):
        N, d = X.shape
        recon_term = np.sum(X**2)  # sum_i x_i^T x_i
        cross_term = 2 * np.sum((X @ W) * Ez)  # sum_i x_i^T W Ez_i
        WtW = W.T @ W
        trace_term = np.sum(np.einsum('ij,nji->n', WtW, Ezz))  # sum_i tr(WtW @ Ezz_i)
        S = recon_term - cross_term + trace_term
        self.psi = S / (N * d)

    def as_matrix(self) -> np.ndarray:
        return self.psi * np.eye(self.d)

    def initialize(self, X):
        self.psi = np.var(X, axis=0).mean()

class AnisotropicNoise(NoiseModel):
    def __init__(self, d):
        self.d = d
        self.psi = np.ones(d)

    def m_step(self, X, W, Ez, Ezz):
        N, self.d = X.shape
        recon = np.sum(X**2, axis=0)                          # (d,) -- sum_i x_ij^2, per feature
        cross = 2 * np.sum(X * (Ez @ W.T), axis=0)            # (d,) -- sum_i x_ij (w_j^T Ez_i)
        quad = np.einsum('dk,nkl,dl->d', W, Ezz, W)           # (d,) -- sum_i w_j^T Ezz_i w_j
        S = recon - cross + quad
        self.psi = S / N                                      # (d,)

    def as_matrix(self) -> np.ndarray:
        return np.diag(self.psi)

    def initialize(self, X):
        self.psi = np.var(X, axis=0)