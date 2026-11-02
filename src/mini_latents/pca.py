import numpy as np

from .base import LinearLatentModels, fix_signs, top_k_eig


class PCA(LinearLatentModels):
    """
    Principal Component Analysis via partial eigendecomposition

    Parameters
    ----------
    n_components: int, denoted as k
        Number of principal components

    Attributes
    ----------
    mean_: ndarray (d, )
        Feature means
    components_: ndarray (d, k)
        leading eigenvectors of the sample covariance as columns
    explained_variance_ : ndarray of shape (k)
        corresponding eigenvalues, in descending order.
    noise_variance_ : float
        Mean of discarded eigenvalues
    """

    def __init__(self, n_components: int):
        super().__init__(n_components)

    def fit(self, X: np.ndarray):
        """
        Fit the top n_components (k) principal components to the data X.

        computes top 'n_components' with
        1. default svds (lanczos solver): O(mnd + m^2d) time where k < m < d, doesn't require sample covariance
        2. else scipy.linalg.eigh(subset_by_index=...) O(kd^2) time, requires sample covariance

        Parameters
        ----------
        X: ndarray (N, d)
            Data with rows as samples

        Returns
        ---------
        self: PCA (fitted model)

        Raises
        ----------
        ValueError
            If 'n_components' is not in [1, d]
        """
        X = np.asarray(X, dtype=np.float64)

        N, d = X.shape
        k = self.n_components

        if not 0 < k <= min(N, d):
            msg = f"n_components must be < min({N - 1}, {d}), got {k}"
            raise ValueError(msg)

        # center the data
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        eigvals, eigvecs, total_var = top_k_eig(Xc=Xc, k=k)

        self.explained_variance_ = eigvals
        self.components_ = fix_signs(eigvecs)

        if d == k:
            self.noise_variance_ = 0.0
        else:
            resid = total_var - self.explained_variance_.sum()
            self.noise_variance_ = np.maximum((resid / (d - k)), 0.0)

        return self

    def infer_latents(self, X: np.ndarray):
        """project data onto latent space (principal components)"""
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_
