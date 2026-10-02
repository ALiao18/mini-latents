from abc import ABC, abstractmethod
from .tracking import FitFlags
import numpy as np


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
