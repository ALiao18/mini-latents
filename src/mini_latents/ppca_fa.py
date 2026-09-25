from .base import LinearLatentModels
from .noise_model import NoiseModel, IsotropicNoise, AnisotropicNoise
from .em_core import e_step, m_step_W, log_likelihood
import numpy as np

_LL_NOISE = 1e-9


class ProbabilisticLinearLatentModels(LinearLatentModels):
    noise_model: NoiseModel

    def transform(self, X):
        Xc = X - self.mean_
        Ez, _ = e_step(Xc, self.components_, self.noise_model.as_matrix())
        return Ez

    def fit(self, X, n_components=None, max_iter=100, tol=1e-6):
        '''
        Fit the model using the EM algorithm.
        1. center the data to mu = 0
        2. initialize W using self._init_W(Xc)

        params:
        - n_components: defaults to value given to __init__, pasing here overrides and updates self.n_components
        - tol         : per-sample log-likelihood gain. Stopping criteria for EM
        '''
        if n_components is None:
            n_components = self.n_components
        self.n_components = n_components

        # zero mean the data
        self.mean_ = X.mean(axis=0) # (d,)
        Xc = X - self.mean_         # (N, d) broadcast over rows
        N, _ = Xc.shape

        # xyz initialize W: correlation matrix 
        W = self._init_W(Xc, n_components)
        self.noise_model.initialize(Xc)
        self.ll_history_ = []

        prev_ll = -np.inf
        i = -1                                  # so max_iter=0 leaves n_iter_ at 0
        for i in range(max_iter):
            Psi = self.noise_model.as_matrix()
            Ez, Ezz = e_step(Xc, W, Psi)
            W = m_step_W(Xc, Ez, Ezz)
            self.noise_model.m_step(Xc, W, Ez, Ezz)

            ll = log_likelihood(Xc, W, self.noise_model.as_matrix())
            self.ll_history_.append(ll)
            if ll < prev_ll - _LL_NOISE * abs(prev_ll):
                print("Bug! log-likelihood decreased. EM guarantees monotonic increase")
                prev_ll = ll
            if (ll - prev_ll) / N < tol:
                prev_ll = ll
                break
                
            prev_ll = ll

        self.components_ = W
        self.n_iter_ = i + 1
        self.log_likelihood_ = prev_ll
        return self

    def _init_W(self, Xc, n_components):
        '''
        initializes Weight matrix using PCA initialization 
        (closed-form max. likelihood solution for pPCA) (Tipping & Bishop, 1999)

        W = U_k*(Λ_k - sigma^2*I)**(1/2)*R

        U_k (d,k): top-k eigenvectors of covariance matrix
        Λ_k (k,k): top-k eigenvalues
        σ²       : mean of discarded eigenvalues
        R   (k,k): orthogonal matrix

        params:
        - Xc (N, d): X, mean centered
        - n_componenets: n principal components to retain

        returns:
        - W (d,k): weight matrix initialization
        '''
        N, _ = Xc.shape

        S = Xc.T @ Xc / N                                   # (d, d), symmetric sample covariance
        eigvals, eigvecs = np.linalg.eigh(S)                # ascending 
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]  # descending: eigvals (d,), eigvecs (d,d)

        # max likelihood

        # eigh returns smallest-first; take the top n_components
        idx = np.argsort(eigvals)[::-1][:n_components]
        top_eigvals = eigvals[idx]
        top_eigvecs = eigvecs[:, idx]

        discarded_eigvals = eigvals[-idx].flatten()
        sigma_squared = np.mean(discarded_eigvals)
        top_eigvals = np.clip(top_eigvals, a_min=1e-8, a_max=None) 
        R = np.eye(n_components)

        W = top_eigvecs * np.sqrt(top_eigvals - sigma_squared * np.eye())*R # (d,k)
        return W # (d,k)


class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = IsotropicNoise()


class FA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = AnisotropicNoise()