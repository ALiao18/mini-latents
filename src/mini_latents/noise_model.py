from abc import ABC, abstractmethod
import numpy as np

class NoiseModel(ABC):
    @abstractmethod
    def m_step(self, X: np.ndarray, W, Ez: np.ndarray, Ezz: np.ndarray):
        '''
        m-step of the EM algorithm

        Params:
        -------
        X:   data matrix (n_samples, n_features)
        W:   weight matrix (n_features, n_components)
        Ez:  expected mean of posterior distribution of latent variables (n_samples, n_components)
        Ezz: posterior second moment of latent variables (n_samples, n_components, n_components)
        '''

    @abstractmethod
    def initialize(self, X: np.ndarray):
        '''
        Initialize the noise model parameters based on X
        '''

    @abstractmethod
    def noise_as_vec(self):
        '''
        returns diagonal as a (d,) vector
        '''

    def as_matrix(self):
        '''
        Returns dxd noise covariance matrix
        '''
        return np.diag(self.noise_as_vec())

class IsotropicNoise(NoiseModel):
    def __init__(self):
        self.d = None
        self.psi = None

    def initialize(self, X):
        self.d = X.shape[1]
        self.psi = np.var(X, axis=0).mean()

    def m_step(self, X, W, Ez, Ezz):
        N, d = X.shape
        recon_term = np.sum(X**2)  # sum_i x_i^T x_i
        cross_term = 2 * np.sum((X @ W) * Ez)  # sum_i x_i^T W Ez_i
        WtW = W.T @ W
        trace_term = np.sum(np.einsum('ij,nji->n', WtW, Ezz))  # sum_i tr(WtW @ Ezz_i)
        S = recon_term - cross_term + trace_term
        self.psi = S / (N * d)

    def noise_as_vec(self):
        return np.full(self.d, self.psi) # (d,)
    

class AnisotropicNoise(NoiseModel):
    def __init__(self):
        self.d = None
        self.psi = None

    def initialize(self, X):

        self.d = X.shape[1]
        self.psi = np.var(X, axis=0)

    def m_step(self, X, W, Ez, Ezz):
        """
        FA noise update. O(dk)
        
        psi_j = expected squared residual (R_j) of feature j, summed over samples

        R_j = recon - cross + quad
            recon = sum_n x_nj**2
            cross = 2 w_j^T (sum_n <z_n> x_nj)
            quad  = w_j^T (sum_n < z_n z_n^T>) w_j

        Params
        ------
        X   (N,d): centered data
        W   (d,k): loading matrix. 
        Ez  (N,k): posterior means <z_n>
        Ezz (N,k,k): posterior second moments <z_n z_n^T>
        """
        N, self.d = X.shape

        # aggregate statistics so each passes over n once
        sum_Ezz = Ezz.sum(axis=0)   # (k,k) = sum_n <z_n, z_n^T> symmetric PSD
        XtEz    = X.T @ Ez          # (d,k) row j = sum_n x_nj <>z_n>^T. one BLAS matmul O(Ndk)

    
        recon = np.sum(X**2, axis=0)            # (d,)
        # cross_j = 2w_j^T (row j of XtEz)
        # elementwise (d, k) * (d, k), then sum over k  ->  one dot product per row
        cross = 2 * np.sum(W * XtEz, axis=1)    # (d,)
        # quad  = w_j^T sum_Ezz w_j
        # W @ sum_Ezz is (d, k), with row j = w_j^T sum_Ezz; then a row-wise dot with W.
        # This equals diag(W sum_Ezz W^T) without forming the (d, d) matrix. O(dk^2)
        quad = np.sum((W @ sum_Ezz) * W, axis=1) # (d,)

        R = recon - cross + quad    # (d,) Expected squared residual per feature
        self.psi = R / N            # (d,) FA: one noise variance per feature

    def noise_as_vec(self):
        return self.psi
