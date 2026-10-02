import numpy as np

from .base import LinearLatentModels


class PCA(LinearLatentModels):
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the data X
        """
        # center the data
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # get the covariance matrix
        self.cov_ = np.cov(Xc, rowvar=False, ddof=0)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_)
        eigenvalues, eigenvectors = (
            eigenvalues[::-1],
            eigenvectors[:, ::-1],
        )  # eigh is ascending

        # sort the eigenvalues and eigenvectors in descending order
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.components_ = eigenvectors[:, : self.n_components]

        # closed form pPCA solution for noise variance
        discarded = eigenvalues[self.n_components :]
        self.noise_variance_ = (
            float(max(discarded.mean(), 0.0)) if discarded.size else 0.0
        )

        return self

    def infer_latents(self, X: np.ndarray):
        return (X - self.mean_) @ self.components_
