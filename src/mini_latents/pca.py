import numpy as np
import scipy as sp

from .base import LinearLatentModels, fix_signs


class PCA(LinearLatentModels):
    """
    Principal Component Analysis via partial eigendecomposition

    Parameters
    ----------
    n_components: int 
        Number of principal components

    Attributes
    ----------
    mean_: ndarray (y_dim, )
        Feature means
    components_: ndarray (y_dim, x_dim)
        leading eigenvectors of the sample covariance as columns
    explained_variance_ : ndarray of shape (x_dim, )
        corresponding eigenvalues, in descending order. 
    noise_variance_ : float
        Mean of discarded eigenvalues

    """
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def fit(self, X: np.ndarray):
        """
        Fit the top n_components principal components to the data X.

        Forms (y_dim, y_dim) covariacne and computes top 'n_components'
        eigenpairs with scipy.linalg.eigh(subset_by_index=...)
        
        Parameters
        ----------
        X: ndarray (n_samples, y_dim)
            Data with rows as samples
        
        Returns
        ---------
        PCA
            The fitted model
        
        Raises
        ----------
        ValueError
            If 'n_components' is not in [1, y_dim]

        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, y_dim = X.shape
        x_dim = self.n_components
        if not 0 < x_dim <= y_dim:
            msg = 'n_components must be in [1, y_dim]'
            raise ValueError(msg)
        
        # center the data
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_ # (n_samples, y_dim)
        cov = Xc.T @ Xc / n_samples # (y_dim, y_dim)
        total_var = np.trace(cov)

        eigvals, eigvecs = sp.linalg.eigh(cov, subset_by_index= [y_dim - x_dim, y_dim - 1]) # ascending order
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

        self.explained_variance_ = eigvals
        self.components_ = fix_signs(eigvecs)

        if y_dim > x_dim: 
            resid = total_var - self.explained_variance_.sum()
            self.noise_variance_ = np.maximum((resid / (y_dim - x_dim)), 0.0)
        else:
            self.noise_variance_ = 0.0

        return self

    def infer_latents(self, X: np.ndarray):
        return (X - self.mean_) @ self.components_
