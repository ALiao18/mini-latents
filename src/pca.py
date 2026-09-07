from base import LinearLatentModels
import numpy as np
from scipy.linalg import eigh

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
        self.noise_cov_ = np.cov(self.Xc, rowvar = False)
        eigenvectors, eigenvalues = eigh(self.noise_cov_)

        # sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_indices][:self.n_components]
        self.components = eigenvectors[sorted_indices][:self.n_components]

    def transform(self):
        self.Z = self.Xc @ self.components_



