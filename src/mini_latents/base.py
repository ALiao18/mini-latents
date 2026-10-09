from abc import ABC, abstractmethod
from .tracking import FitFlags
import numpy as np


def fix_signs(W: np.ndarray) -> np.ndarray:
    """
    Flip each column of W (d,k) so its largest |entry| is positive.

    A column's sign is arbitrary (v and -v fit equally well) and the solver may return
    either (depends on LAPACK, solver, feature order); this makes components_ deterministic.
    """
    idx = np.argmax(np.abs(W), axis=0)  # (k,) row of each column's largest |entry|
    return W * np.sign(W[idx, np.arange(W.shape[1])])


class LinearLatentModels(ABC):
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None  # (d,k) eigenvectors of n_components
        self.mean_ = None
        self.explained_variance_ = None  # (k, ) eigenvalues of n_components
        self.noise_variance_ = None
        self.flags = FitFlags()
        self.cov_ = None

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
