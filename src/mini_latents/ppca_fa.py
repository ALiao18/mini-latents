import numpy as np
from scipy.linalg import eigvalsh

from .base import LinearLatentModels, fix_signs, top_k_eig_cov
from .em_core import e_step_S, e_step, log_likelihood, m_step_W, logdet_from_cholesky
from .noise_model import AnisotropicNoise, IsotropicNoise, NoiseModel
from .tracking import FitFlags

def geometric_nonzero_eig(S, N, TrS):
    """
    Geometric mean of the nonzero eigenvalues of S (d, d) covariance matrix.

    if N > d: 
        S is full rank, cholesky O(d^3/3)
        log|S| = 2 sum log diag(L) where L is cholesky factor of S
    Else:
        S is rank deficient, use eigvalsh O(d^3)
    """
    d = S.shape[0]
    tol_scale = max(N, d) * np.finfo(float).eps

    if N > d:
        try:
            L = np.linalg.cholesky(S)
            # trS >= largest eigenvalue of S, so 
            if (np.min(np.diagonal(L)) ** 2) > (TrS * tol_scale):
                return np.exp(logdet_from_cholesky(L)/d) 
        except np.linalg.LinAlgError:
            pass # not numerically positive definite
    
    eigvals = eigvalsh(S)[::-1]
    rank = int(np.sum(eigvals > eigvals[0] * tol_scale))
    return np.exp(np.log(eigvals[:rank]).mean())

def S_products(S, W, psi):
    """products of S for params (W, psi). O(d^2 k)"""
    B = W / psi[:, None]
    SB_raw = S @ B
    return B.T @ SB_raw, SB_raw

class ProbabilisticLinearLatentModels(LinearLatentModels):
    noise_model: NoiseModel

    def infer_latents(self, X):
        """
        Infer latent variables for X using current model parameters

        Returns
        -------
        Ez (N,k): posterior mean of latent variables
        """
        Xc = X - self.mean_
        Ez, _ = e_step(Xc, self.components_, self.noise_model.noise_as_vec())
        return Ez

    def fit(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        max_iter: int = 5000,
        tol: float = 1e-8,
        random_seed=None,
        min_var_frac=None,
    ):
        """
        Fit the model using the EM algorithm. O(d^2k)

        Stopping criterion:
        fastfa.m: after 2 baseline iterations, stop when the latest ll gain is less than
        tol * total gain since baseline.

        params
        ------
        - X: (N, d) np.ndarray
            data matrix
        - n_components: int,
            denoted as k, defaults to value passed to __init__, passing here overrides and updates self.n_components
        - tol: float
            Stopping criteria for EM
        """
        X = np.asarray(X, dtype=np.float64)
        N, d = X.shape
        if n_components is None:
            n_components = self.n_components
        self.n_components = n_components
        if random_seed is not None:
            self.random_seed = random_seed
        if min_var_frac is not None:
            self.noise_model.min_var_frac = min_var_frac
        if not 0 < self.n_components <= min(N, d):
            msg = f"n_components must be in < min({N}, {d}), got {self.n_components}"
            raise ValueError(msg)
       
        self.flags = FitFlags()

        # zero mean the data
        self.mean_ = X.mean(axis=0)  # (d,)
        Xc = X - self.mean_  # (N, d). O(Nd)
        S = Xc.T @ Xc / N  # (d,d), O(Nd^2) sample covariance
        del Xc
        diagS = np.diag(S).copy() # (d,)
        TrS = diagS.sum()  # (scalar), O(d), total variance
         
        # initialize loading matrix with scaled random vectors.
        W, sigma2 = self._init_W(S = S, TrS = TrS, N = N, n_components = n_components)  # (d, k), scalar
        self._init_noise(diagS, sigma2)
        self.ll_history_ = []

        # cache for (W, psi), used by LL and next E-step
        psi = self.noise_model.noise_as_vec()
        BtSB, SB_raw = S_products(S, W, psi)

        ll_curr = -np.inf
        ll_base = -np.inf
        ll_old = -np.inf

        for i in range(max_iter):
            # 1. e-step from S O(dk^2 + k^3) for FA, O(dk) for pPCA
            SB, G = e_step_S(W, psi, BtSB, SB_raw)

            # 2. m-step 
            W = m_step_W(SB, G) # O(k^3 + dk^2)
            self.noise_model.m_step(W, SB, diagS) # O(dk)
            psi = self.noise_model.noise_as_vec()
            BtSB, SB_raw = S_products(S, W, psi)

            # 3. convergence checks, update flags
            if (
                isinstance(self.noise_model, AnisotropicNoise)
                and self.noise_model.floored
            ):
                self.flags.heywood = True

            ll_curr, BtSB = log_likelihood(W, psi, diagS, SB_raw, N) # O(d + k^3)
            self.ll_history_.append(ll_curr)

            if i <= 1:
                ll_base = ll_curr
            else:
                if ll_curr < ll_old:
                    self.flags.decreasing_ll = True
                elif (ll_curr - ll_base) < (1 + tol) * (ll_old - ll_base):
                    self.flags.converged = True
                    break  # stop EM

            # 4. update log likelihood
            ll_old = ll_curr

        if not self.ll_history_: # max_iter == 0 PCA closed form
            ll_curr, _ = log_likelihood(W, psi, diagS, SB_raw, N)

        # EM is sign-equivariant: flipping a column of W leaves C unchanged
        self.components_ = fix_signs(W)  
        self.n_iter_ = len(self.ll_history_)
        self.log_likelihood_ = ll_curr
        return self

    def _init_W(self, S, TrS, N, n_components):
        """
        PCA initialization (Tipping & Bishop, 1999)

        W = U_k*(Λ_k - sigma^2*I)**(1/2)*R

        U_k (d,k): top-k eigenvectors of covariance matrix
        Λ_k (k,k): top-k eigenvalues
        σ²       : mean of discarded eigenvalues
        R   (k,k): orthogonal matrix. We just choose I in this case

        Parameters:
        -------------
        - S (d,d)
            covariance matrix of the data
        - TrS: scalar
            Tr(S), total variance
        - n_components: n principal components to retain

        Returns:
        -------------
        - W: np.ndarray (d,k) 
            weight matrix initialization
        - sigma2: scalar
            isotropic noise variance initialization
        """
        d = S.shape[0]
        k = n_components

        eigvals, eigvecs = top_k_eig_cov(S, k)  # (k,), (d,k), scalar
        sigma2 = (TrS - eigvals.sum()) / (d-k) if k < d else 0.0
        # ML noise variance = mean of the d - k discarded eigenvalues
        tol = 10 * np.finfo(float).eps * TrS  # roundoff scale of tr(S)
        if sigma2 <= tol:  # k >= rank(S): no residual variance -> _init_noise raises
            sigma2 = 0.0

        # Λ_k - σ²I >= 0 in exact arithmetic (λ_i >= λ_k >= mean of the tail)
        diff = eigvals - sigma2
        if np.any(diff < -tol):
            msg = f"lambda_i < sigma^2 beyond roundoff: {diff}"
            raise FloatingPointError(msg)
        W = eigvecs * np.sqrt(np.clip(diff, 0.0, None))  # clip removes roundoff only
        return W, sigma2

    def _init_noise(self, diagS, sigma2):
        """
        Start variance at maximum likelihood value, consistent with W from _init_W

        Used by pPCA, where (W, sigma2) is the exact ML solution. FA overrides this.
        """
        if sigma2 <= 0:
            msg = "need n_components < min(N, d): sigma^2_ML is the mean of the discarded eigenvalues, and there are none"
            raise ValueError(msg)
        else:
            self.noise_model.initialize(diagS, psi = sigma2)

class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = IsotropicNoise()
        self.method = "closed_form"
        self.random_seed = 0  # default

    def fit(
        self,
        X,
        n_components=None,
        max_iter=5000,
        tol=1e-8,
        method="closed_form",
        random_seed=None,
    ):
        """
        method = "closed_form": ML solution (eigh of the covariance), no EM (default)
        method = "em"         : EM from a random start, W ~ N(0, 1) and sigma^2 = mean feature
                                variance.
        """

        if method not in ("em", "closed_form"):
            msg = f"method must be 'em' or 'closed_form', got {method}"
            raise ValueError(msg)
        self.method = method

        if random_seed is not None:
            self.random_seed = random_seed
        if method == "closed_form":
            max_iter = 0

        super().fit(X, n_components, max_iter=max_iter, tol=tol)

        if method == "closed_form":
            self.flags.converged = True

        return self

    def _init_W(self, S, TrS, N, n_components):
        """closed_form: the ML solution. em: random W, sigma^2 = mean feature variance."""
        if self.method == "em":
            d = S.shape[0]
            k = self.n_components
            W = np.random.default_rng(self.random_seed).normal(size=(d, k))  # (d, k)
            return W, TrS / d
        else:
            return super()._init_W(S, TrS, N, n_components)

class FA(ProbabilisticLinearLatentModels):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.noise_model = AnisotropicNoise()
        self.random_seed = 0  # default

    def fit(
        self,
        X,
        n_components=None,
        max_iter=5000,
        tol=1e-8,
        random_seed=None,
        min_var_frac=None,
    ):
        if random_seed is not None:
            self.random_seed = random_seed
        if min_var_frac is not None:
            self.noise_model.min_var_frac = min_var_frac
        return super().fit(X, n_components, max_iter=max_iter, tol=tol)

    def _init_noise(self, diagS, sigma2):
        """
        FA: per-feature variance diag(S).
        """
        self.noise_model.initialize(diagS, psi=None)

    def _init_W(self, S, TrS, N, n_components):
        """
        W ~ N(0, scale/k)
        scale = geometric mean of nonzero eigenvalues of S
        """
        d = S.shape[0]
        k = self.n_components
        scale = geometric_nonzero_eig(S, N, TrS)
        rng = np.random.default_rng(self.random_seed)
        return (rng.normal(size=(d, k)) * np.sqrt(scale / n_components), None)