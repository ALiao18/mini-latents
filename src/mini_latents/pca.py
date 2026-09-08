from .base import LinearLatentModels
import numpy as np

class PCA(LinearLatentModels):
    def __init__(self, n_components: int):
        super().__init__(n_components)
        self.Xc = None

    def fit(self, X: np.ndarray):
        '''
        Fit the PCA model to the data X
        '''
        # center the data
        self.mean_   = X.mean(axis=0)
        self.Xc   = X - self.mean_

        # get the covariance matrix
        self.cov_ = np.cov(self.Xc, rowvar = False, ddof=0)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_)

        # sort the eigenvalues and eigenvectors in descending order
        sorted_indices              = np.argsort(eigenvalues)[::-1]
        self.explained_variance_    = eigenvalues[sorted_indices][:self.n_components]
        self.components_            = eigenvectors[:, sorted_indices][:, :self.n_components]

        return self

    def transform(self, X: np.ndarray):
        X_centered = X - self.mean_
        self.Z = X_centered @ self.components_
        return self.Z



