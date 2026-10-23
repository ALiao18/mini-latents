from abc import ABC, abstractmethod
from .tracking import FitFlags
from scipy.sparse.linalg import svds
import scipy as sp
import numpy as np


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


def top_k_eig(Xc, k, solver="svds"):
    """
    Top-k eigenpairs of S = Xc^T Xc / N (descending),
    tr(S)
    """
    N, d = Xc.shape
    total_var = np.einsum("ij,ij->", Xc, Xc) / N  # Tr(S), O(Nd)
    if solver == "svds" and k < min(N, d):
        v0 = np.ones(min(N, d)) / np.sqrt(min(N, d))  # fixed start
        _, s, Vt = svds(Xc, k=k, v0=v0)  # Lanczos solver, iterative, doesn't form S
        order = np.argsort(s)[::-1]
        return s[order] ** 2 / N, Vt[order].T, total_var
    S = Xc.T @ Xc / N
    vals, vecs = sp.linalg.eigh(S, subset_by_index=[d - k, d - 1])
    return vals[::-1], vecs[:, ::-1], total_var
