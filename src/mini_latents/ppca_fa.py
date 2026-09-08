from .base import LinearLatentModels
from .noise_model import NoiseModel, IsotropicNoise, AnisotropicNoise
from .em_core import e_step, m_step_W, log_likelihood
import numpy as np


class ProbabilisticLinearLatentModels(LinearLatentModels):
    noise_model: NoiseModel

    def transform(self, X):
        Xc = X - self.mean_
        Ez, _ = e_step(Xc, self.components_, self.noise_model.as_matrix())
        return Ez

    def fit(self, X, n_components, max_iter=100, tol=1e-4):
        '''
        Fit the model using the EM algorithm.
        '''
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        N, d = Xc.shape

        W = self._init_W(Xc, n_components)
        self.noise_model.initialize(Xc)

        prev_ll = -np.inf
        for i in range(max_iter):
            Psi = self.noise_model.as_matrix()
            Ez, Ezz = e_step(Xc, W, Psi)
            W = m_step_W(Xc, Ez, Ezz)
            self.noise_model.m_step(Xc, W, Ez, Ezz)

            ll = log_likelihood(Xc, W, self.noise_model.as_matrix())
            if ll - prev_ll < tol:
                break
            elif ll < prev_ll:
                print("Bug! log-likelihood decreased. EM guarantees monotonic increase")
            prev_ll = ll

        self.components_ = W
        self.n_iter_ = i + 1
        self.log_likelihood_ = prev_ll
        return self

    def _init_W(self, Xc, n_components):
        N, d = Xc.shape
        cov = Xc.T @ Xc / N                     # (d, d), symmetric

        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order

        # eigh returns smallest-first; take the top n_components
        idx = np.argsort(eigvals)[::-1][:n_components]
        top_eigvals = eigvals[idx]
        top_eigvecs = eigvecs[:, idx]

        top_eigvals = np.clip(top_eigvals, a_min=1e-8, a_max=None) 
        W = top_eigvecs * np.sqrt(top_eigvals)
        return W


class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = IsotropicNoise()


class FA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = AnisotropicNoise()