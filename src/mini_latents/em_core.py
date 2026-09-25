import numpy as np

def _sym_inv_logdet(A: np.ndarray, eps: float = 1e-10):
    """
    Invert a symmetric positive (semi-)definite matrix and compute its
    log-determinant via eigendecomposition

    Returns
    -------
    A_inv:  inverse of A, (n, n)
    logdet: log|A|, computed as sum(log(eigenvalues))
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, a_min=eps, a_max=None)  # guard near-zero/negative eigenvalues

    A_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    logdet = np.sum(np.log(eigvals))
    return A_inv, logdet

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
    Ez:  posterior mean, (N, k)
    Ezz: posterior second moment E[z z^T], (N, k, k)
    """
    if psi.ndim != 1: 
        raise ValueError(f"psi must be the (d, ) diagonal, got shape {psi.shape}")
    
    k = W.shape[1]

    psi_inv   = 1/psi                            # (d,)
    Psi_inv_W = psi_inv[:, None] * W             # (d,k), O(dk)

    M_inv = np.eye(k) + W.T @ Psi_inv_W                # (k, k), symmetric PD
    M, _ = _sym_inv_logdet(M_inv)                      # posterior covariance, shared across samples

    Ez = X @ Psi_inv_W @ M                              # (N, k)
    Ezz = M[None, :, :] + np.einsum('nk,nl->nkl', Ez, Ez)  # (N, k, k)

    return Ez, Ezz

def m_step_W(X: np.ndarray, Ez: np.ndarray, Ezz: np.ndarray) -> np.ndarray:
    """
    Closed-form M-step update for W. Identical for pPCA and FA.

    W_new = ( sum_i x_i Ez_i^T ) ( sum_i Ezz_i )^-1
    """
    sum_xEz = X.T @ Ez                                  # (d, k)
    sum_Ezz = Ezz.sum(axis=0)                           # (k, k), symmetric PD

    sum_Ezz_inv, _ = _sym_inv_logdet(sum_Ezz)
    W_new = sum_xEz @ sum_Ezz_inv
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

    C_inv, logdet = _sym_inv_logdet(C)
    quad = np.einsum('ni,ij,nj->', X, C_inv, X)         # sum_i x_i^T C^-1 x_i

    ll = -0.5 * (N * d * np.log(2 * np.pi) + N * logdet + quad)
    return ll
