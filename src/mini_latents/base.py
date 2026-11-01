from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from scipy.sparse.linalg import svds

from .tracking import FitFlags


class LinearLatentModels(ABC):
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None  # (d,k) loadings of n_components
        self.mean_ = None
        self.explained_variance_ = None  # (k, ) eigenvalues of n_components
        self.noise_variance_ = None
        self.flags = FitFlags()

    @abstractmethod
    def fit(self, X: np.ndarray):
        """
        Fit the model to the data X
        """

    @abstractmethod
    def infer_latents(self, X: np.ndarray):
        """
        Infer the latent variables for the data X
        """

    def inverse_transform(self, Z: np.ndarray):
        """
        Transform the latent representation Z back to the original space
        """
        return Z @ self.components_.T + self.mean_


def fix_signs(W: np.ndarray) -> np.ndarray:
    """
    Flip each column of W (d,k) so its largest |entry| is positive.

    A column's sign is arbitrary (v and -v fit equally well) and the solver may return
    either (depends on LAPACK, solver, feature order); this makes components_ deterministic.
    """
    idx = np.argmax(np.abs(W), axis=0)  # (k,) row of each column's largest |entry|
    return W * np.sign(W[idx, np.arange(W.shape[1])])

def top_k_eig(Xc, k, solver="svds", random_state = 42):
    """
    Top-k eigenpairs of S = Xc^T Xc / N (descending), and tr(S).

    solver="svds": Lanczos on Xc; never forms S. O(Nd) per matvec.
    otherwise: forms S (d, d), then dense symmetric eigh.
        O(Nd^2) to form S + O(d^3) tridiagonal reduction
        (subset_by_index only saves the back-transform).

    Parameters
    ----------
    Xc: ndarray (N, d)
        Centered data matrix
    k: int
        Number of top eigenpairs to compute
    solver: str, optional
        "svds" (default) or anything else for dense eigh
    """
    N, d = Xc.shape

    if solver == "svds" and k < min(N, d):
        _, s, Vt = svds(Xc, k=k, random_state=random_state)  # Lanczos, doesn't form S
        order = np.argsort(s)[::-1]
        return s[order] ** 2 / N, Vt[order].T

    S = Xc.T @ Xc / N  # (d, d) sample covariance
    trS = np.trace(S)
    vals, vecs = sp.linalg.eigh(S, subset_by_index=[d - k, d - 1], overwrite_a=True)
    return vals[::-1], vecs[:, ::-1], trS
