import numpy as np
from scipy.linalg import cho_solve

def cholesky_solve(B: np.ndarray, L: np.ndarray) -> np.ndarray:
     """A^-1 B given L = cholesky(A)."""
     return cho_solve((L, True), B)     # True means L is the lower triangle

def logdet_from_cholesky(L: np.ndarray) -> float:
     """log|A| = 2 sum(log(diag(L))) for A = L L^T"""
     return 2 * np.log(np.diagonal(L)).sum()

def e_step(X: np.ndarray, W: np.ndarray, psi: np.ndarray):
    """
    Compute posterior over latent variables z | x for each sample.

    Params
    ------
    X   (N,d): centered data matrix 
    W   (d,k): current loading matrix 
    psi (d, ): flattened diagonal noise covariance from noise_model.noise_as_vec()

    Returns
    -------
    Ez       (N,k): posterior mean
    sum_Ezz  (k,k): sum of E[z z^T], (N, k, k) posterior second moment over N
    """
    N, _ = X.shape

    if psi.ndim != 1: 
        raise ValueError(f"psi must be the (d, ) diagonal, got shape {psi.shape}")
    
    k = W.shape[1]

    psi_inv   = 1/psi                      # (d,)
    Psi_inv_W = psi_inv[:, None] * W       # (d,k), O(dk)

    M_inv = np.eye(k) + W.T @ Psi_inv_W    # (k,k), symmetric PD
    L     = np.linalg.cholesky(M_inv)      # (k,k)
    M     = cholesky_solve(np.eye(k), L)   # (k,k) posterior cov 

    Ez = X @ Psi_inv_W @ M                 # posterior mean (N, k)
    sum_Ezz = N * M + (Ez.T @ Ez)          # (k, k) m_step_W uses sum

    return Ez, sum_Ezz

def m_step_W(X: np.ndarray, Ez: np.ndarray, sum_Ezz: np.ndarray) -> np.ndarray:
    """
    Closed-form M-step update for W.
    W_new = ( sum_i x_i Ez_i^T ) ( sum_i Ezz_i )^-1

    Params
    ------
    X       (N,d): centered data matrix
    Ez      (N,k): posterior mean
    sum_Ezz (k,k): sum of E[z z^T], (N, k, k) posterior second moment over N

    Returns
    ------
    W_new   (d,k): updated loading matrix
    """
    sum_xEz = X.T @ Ez                              # (d,k)
    L = np.linalg.cholesky(sum_Ezz)                 # (k,k)
    W_new = cholesky_solve(sum_xEz.T, L).T          # (d,k)

    return W_new

def log_likelihood(X: np.ndarray, W: np.ndarray, psi: np.ndarray) -> float:
    """
    Marginal log-likelihood under the model, with z integrated out:
        x ~ N(0, C),   C = W W^T + Psi

    Used for EM convergence monitoring (should increase monotonically
    each iteration).

    Params
    ------
    X   (N,d): centered data matrix
    W   (d,k): current loading matrix
    psi (d, ): flattened diagonal noise covariance from noise_model.noise_as_vec()
    """
    if psi.ndim != 1: 
            raise ValueError(f"psi must be the (d, ) diagonal, got shape {psi.shape}")
    
    Psi = np.diag(psi)
    N, d = X.shape
    C = W @ W.T + Psi

    L         = np.linalg.cholesky(C)   # (d,d)
    logdet    = logdet_from_cholesky(L)
    quad_term = np.sum(X.T * cholesky_solve(X.T, L)) # sum_n * x_n^T C^-1 x_n 

    ll = -0.5 * (N * d * np.log(2 * np.pi) + N * logdet + quad_term)
    return ll
