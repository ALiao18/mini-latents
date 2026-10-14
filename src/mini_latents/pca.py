import numpy as np
import sklearn
import torch

from .base import LinearLatentModels, fix_signs


class PCA(LinearLatentModels):
    def __init__(self, n_components: int):
        super().__init__(n_components)

    def fit(self, X: np.ndarray):
        """
        Fit the PCA model to the data X
        
        Assumes sparse matrices in most cases, so uses torch.svd_lowrank (halko et al., 2009). 
        Uses paper recommendation of oversampling by 10. 
        """
        # center the data
        self.mean_ = X.mean(axis=0)
        Xc = torch.from_numpy(X - self.mean_)
        n, d = Xc.shape
        k = self.n_components
        q = min(k + 10, n, d)

        # Algorithm 5.1 from Halko et al., 2009
        # better for sparse matrices
        # for dense matrices, use torch.linalg.svd or np.linalg.svd
        with torch.random.fork_rng():
            torch.manual_seed(42)
            _, S, V = torch.svd_lowrank(Xc, q = q) # V: (d, q)

        # sort the eigenvalues and eigenvectors in descending order
        eigenvalues = (S[:k] ** 2 / n).numpy()
        self.explained_variance_ = eigenvalues
        self.components_ = fix_signs(V[:, :k].numpy())

        total_var = (Xc ** 2).sum().item() / n  # = tr(cov), no dxd matrix
        self.noise_variance_ = max((total_var - eigenvalues.sum()) / (d-k), 0.0) if d > k else 0.0
        return self

    def infer_latents(self, X: np.ndarray):
        return (X - self.mean_) @ self.components_
