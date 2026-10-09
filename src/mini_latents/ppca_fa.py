import numpy as np

from .base import LinearLatentModels, fix_signs
from .em_core import e_step, log_likelihood, m_step_W
from .noise_model import AnisotropicNoise, IsotropicNoise, NoiseModel
from .tracking import FitFlags


class ProbabilisticLinearLatentModels(LinearLatentModels):
    noise_model: NoiseModel

    def infer_latents(self, X):
        """
        Infer latent variables for X using current model parameters

        Returns
        ------
        Ez (N,k): posterior mean of latent variables
        """
        Xc = X - self.mean_
        Ez, _ = e_step(Xc, self.components_, self.noise_model.noise_as_vec())
        return Ez

    def fit(
        self,
        X,
        n_components=None,
        max_iter=5000,
        tol=1e-8,
    ):
        """
        Fit the model using the EM algorithm.
        1. center the data to mu = 0
        2. initialize W using self._init_W(Xc)

        Stopping criterion:
        fastfa.m: after 2 baseline iterations, stop when the latest ll gain is less than
        tol * total gain since baseline.

        params
        ------
        - X           : (N, d) data matrix
        - n_components: defaults to value given to __init__, pasing here overrides and updates self.n_components
        - tol         : per-sample log-likelihood gain. Stopping criteria for EM
        """
        if n_components is None:
            n_components = self.n_components

        self.n_components = n_components
        self.flags = FitFlags()

        # zero mean the data
        self.mean_ = X.mean(axis=0)  # (d,)
        Xc = X - self.mean_  # (N, d) broadcast over rows
        X2 = np.sum(Xc**2, axis=0)  # (d,) sum_n x_nj^2, fixed during EM: computed once, reused every iteration
        # N, _ = Xc.shape

        # initialize loading matrix with closed-form PCA solution.
        W, sigma2 = self._init_W(Xc, n_components)  # (d, k), scalar
        self._init_noise(Xc, sigma2)
        self.ll_history_ = []

        ll_curr = -np.inf
        ll_base = -np.inf
        ll_old = -np.inf

        for i in range(max_iter):
            psi = self.noise_model.noise_as_vec()  # (d,)
            Ez, Ezz = e_step(Xc, W, psi)
            XtEz = Xc.T @ Ez  # (d,k), O(Ndk): shared by both M-steps

            W = m_step_W(Xc, Ez, Ezz, XtEz=XtEz)
            self.noise_model.m_step(Xc, W, Ez, Ezz, X2=X2, XtEz=XtEz)  # psi update uses the new W

            ll_curr = log_likelihood(Xc, W, self.noise_model.noise_as_vec(), X2=X2)
            self.ll_history_.append(ll_curr)

            if i <= 1:
                ll_base = ll_curr

            # check for convergence and decreasing log likelihood
            else:
                if ll_curr < ll_old:
                    self.flags.decreasing_ll = True
                elif (ll_curr - ll_base) < (1 + tol) * (ll_old - ll_base):
                    self.flags.converged = True
                    break  # stop EM
            ll_old = ll_curr

        self.components_ = fix_signs(W)  # EM is sign-equivariant: flipping a column of W leaves C unchanged
        self.n_iter_ = len(self.ll_history_)
        self.log_likelihood_ = ll_curr
        return self

    def _init_W(self, Xc, n_components):
        """
        initializes Weight matrix using PCA initialization
        (closed-form max. likelihood solution for pPCA) (Tipping & Bishop, 1999)

        W = U_k*(Λ_k - sigma^2*I)**(1/2)*R

        U_k (d,k): top-k eigenvectors of covariance matrix
        Λ_k (k,k): top-k eigenvalues
        σ²       : mean of discarded eigenvalues
        R   (k,k): orthogonal matrix. We just choose I in this case

        params:
        - Xc (N, d): X, mean centered
        - n_componenets: n principal components to retain

        returns:
        - W      (d,k): weight matrix initialization
        - sigma2 (scalar): isotropic noise variance initialiation
        """
        N, d = Xc.shape

        S = Xc.T @ Xc / N  # (d, d), symmetric sample covariance
        eigvals, eigvecs = np.linalg.eigh(S)  # ascending
        eigvals, eigvecs = (
            eigvals[::-1],
            eigvecs[:, ::-1],
        )  # eigvals (d,), eigvecs (d,d)

        # max likelihood noise variance = mean(discarded eigenvalues)
        sigma2 = eigvals[n_components:].mean() if n_components < d else 0.0  # scalar

        # Λ_k - sigma^2*I
        scale = np.clip(eigvals[:n_components] - sigma2, 1e-12, None)  # (n_components,)

        W = eigvecs[:, :n_components] * np.sqrt(scale)  # (d, k)
        return W, sigma2

    def _init_noise(self, Xc, sigma2):
        """
        Start variance at maximum likelihood value, consistent with W from _init_W

        Used by pPCA, where (W, sigma2) is the exact ML solution. FA overrides this.
        """
        if sigma2 <= 0:
            raise ValueError(
                "need n_components < x_dim: sigma^2_ML is the mean "
                "of the discarded eigenvalues, and there are none"
            )

        self.noise_model.initialize(Xc, psi=sigma2)


class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = IsotropicNoise()
        self.method = "closed_form"
        self.random_seed = None

    def fit(self, X, n_components=None, max_iter=5000, tol=1e-8, method="closed_form", random_seed=None):
        """
        method = "closed_form": ML solution (eigh of the covariance), no EM (default)
        method = "em"         : EM from a random start, W ~ N(0, 1) and sigma^2 = mean feature
                                variance. 
        """

        if method not in ("em", "closed_form"):
            raise ValueError(f"method must be 'em' or 'closed_form', got {method}")
        self.method = method
        self.random_seed = random_seed
        if method == "closed_form":
            max_iter = 0

        super().fit(X, n_components, max_iter=max_iter, tol=tol)

        if method == "closed_form":
            Xc = X - self.mean_
            self.log_likelihood_ = log_likelihood(
                Xc, self.components_, self.noise_model.noise_as_vec()
            )
            self.flags.converged = True

        return self

    def _init_W(self, Xc, n_components):
        """closed_form: the ML solution. em: random W, sigma^2 = mean feature variance."""
        if self.method == "em":
            W = np.random.default_rng(self.random_seed).normal(size=(Xc.shape[1], n_components))  # (d, k)
            return W, Xc.var(axis=0).mean()
        return super()._init_W(Xc, n_components)


class FA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = AnisotropicNoise()

    def _init_noise(self, Xc, sigma2):
        """
        FA: per-feature variance diag(S).
        """
        self.noise_model.initialize(Xc)
