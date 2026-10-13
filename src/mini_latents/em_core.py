import numpy as np
from scipy.linalg import cho_solve


def cholesky_solve(B: np.ndarray, L: np.ndarray) -> np.ndarray:
    """A^-1 B given L = cholesky(A)."""
    return cho_solve((L, True), B)  # True means L is the lower triangle


def logdet_from_cholesky(L: np.ndarray) -> float:
    """
    Cholesky factorization of SPD matrix A = L L^T gives log determinant:
    log|A| = 2 sum(log(diag(L))) for A = L L^T"""
    return 2 * np.log(np.diagonal(L)).sum()


def e_step(X: np.ndarray, W: np.ndarray, psi: np.ndarray):
    """
    Compute posterior over latent variables z | x for each sample. O(Ndk)

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

    psi_inv = 1 / psi  # (d,)
    Psi_inv_W = psi_inv[:, None] * W  # (d,k), O(dk)

    M_inv = np.eye(k) + W.T @ Psi_inv_W  # (k,k), symmetric PD
    L = np.linalg.cholesky(M_inv)  # (k,k)
    M = cholesky_solve(np.eye(k), L)  # (k,k) posterior cov

    Ez = X @ Psi_inv_W @ M  # posterior mean (N, k)
    sum_Ezz = N * M + (Ez.T @ Ez)  # (k, k) m_step_W uses sum

    return Ez, sum_Ezz


def m_step_W(X: np.ndarray, Ez: np.ndarray, sum_Ezz: np.ndarray, XtEz=None) -> np.ndarray:
    """
    Closed-form M-step update for W.
    W_new = ( sum_i x_i Ez_i^T ) ( sum_i Ezz_i )^-1

    Params
    ------
    X       (N,d): centered data matrix
    Ez      (N,k): posterior mean
    sum_Ezz (k,k): sum of E[z z^T], (N, k, k) posterior second moment over N
    XtEz    (d,k): X^T Ez = sum_i x_i Ez_i^T, optional; fit() passes it so the noise
                   M-step reuses it instead of a second O(Ndk) product

    Returns
    ------
    W_new   (d,k): updated loading matrix
    """
    if XtEz is None:
        XtEz = X.T @ Ez  # (d,k), O(Ndk)
    L = np.linalg.cholesky(sum_Ezz)  # (k,k)
    W_new = cholesky_solve(XtEz.T, L).T  # (d,k)

    return W_new


def log_likelihood(
        X: np.ndarray, 
        W: np.ndarray, 
        psi: np.ndarray, 
        X2: np.ndarray | None = None
        ) -> float:
    """
    Marginal log-likelihood under the model, with z integrated out:
        x ~ N(0, C),   C = W W^T + Psi

    With M_inv = I + W^T Psi^-1 W (k,k) instead of (d,d) C:
    1. determinant lemma: log|C| = sum_j log psi_j + log|M_inv|
    2. Woodbury: C^-1 = Psi^-1 - Psi^-1 W M W^T Psi^-1, so with B = X Psi^-1 W (N,k)
       sum_n x_n^T C^-1 x_n = sum_n x_n^T Psi^-1 x_n - trace(M B^T B)

    Parameters
    ----------
    X: ndarray of shape (N,d)
        centered data matrix
    W   (d,k): current loading matrix
    psi (d, ): flattened diagonal noise covariance from noise_model.noise_as_vec()
    X2  (d, ): sum_n x_nj^2

    Returns
    ----------
    log_likelihood: float
        marginal log-likelihood of the data under the parameters W, psi over z
    """
    if psi.ndim != 1:
        msg = f"psi must be the (d, ) diagonal, got shape {psi.shape}"
        raise ValueError(msg)
    # C = W W^T + Psi is PD <=> every psi_j > 0; a zero psi_j with
    # rank(W W^T) < d makes C singular 
    if np.any(psi <= 0):
        msg = "C = W W^T + Psi is singular: psi must be positive"
        raise np.linalg.LinAlgError(msg)
    if X2 is None:
        X2 = np.einsum("ij,ij->j", X, X)  # (d,) sum_n x_nj^2

    # setup
    N, d = X.shape
    k = W.shape[1] 
    psi_inv = 1 / psi  # (d,)
    psi_inv_W = psi_inv[:, None] * W # (d,k), O(dk)
    M_inv =  np.eye(k) + W.T @ psi_inv_W  # (k,k), O(dk^2) SPD
    L = np.linalg.cholesky(M_inv)        # (k,k) O(k^3)

    # term 1    determinant lemma: N log|C| = N (log |Psi| + log|M_inv|)
    # log determinant of diag = sum of log of diagonal entries, O(d)
    log_Psi = np.log(psi).sum() 
    # log-determinant from Cholesky factorization
    log_M_inv = logdet_from_cholesky(L)  # log|M_inv|, O(k)
    log_C = log_Psi + log_M_inv
    N_log_C = N * log_C

    # term 2   Woodbury identity: sum_n x_n^T C^-1 x_n = quad_psi - quad_W
    U = X @ psi_inv_W # (N,k), O(Ndk)
    # 1. sum (x^T psi_inv x) 
    quad_psi = psi_inv @ X2 # O(d)
    # 2. tr(M B^T B) with B = X psi_inv W
    quad_W = np.trace(cholesky_solve(U.T @ U, L)) # O(Nk^2 + k^3)
    quad_C_inv = quad_psi - quad_W

    # marginal log-likelihood
    constant = N * d * np.log(2 * np.pi) 

    return -1/2 * (constant + N_log_C + quad_C_inv) 



