"""Tests for the EM primitives shared by pPCA and FA."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from mini_latents.em_core import _sym_inv_logdet, e_step, m_step_W, log_likelihood


@pytest.fixture
def wpsi():
    '''A centered dataset with a matching (W, Psi) parameter pair.'''
    rng = np.random.default_rng(10)
    N, d, k = 40, 6, 3
    X = rng.normal(size=(N, d))
    X = X - X.mean(axis=0)
    W = rng.normal(size=(d, k))
    Psi = np.diag(rng.uniform(0.3, 1.2, size=d))
    return X, W, Psi


# --- _sym_inv_logdet -------------------------------------------------------

def test_sym_inv_logdet_matches_numpy(spd_matrix):
    A_inv, logdet = _sym_inv_logdet(spd_matrix)

    np.testing.assert_allclose(A_inv, np.linalg.inv(spd_matrix), rtol=1e-10, atol=1e-12)
    sign, expected = np.linalg.slogdet(spd_matrix)
    assert sign == 1
    np.testing.assert_allclose(logdet, expected, rtol=1e-10)


def test_sym_inv_logdet_returns_symmetric(spd_matrix):
    A_inv, _ = _sym_inv_logdet(spd_matrix)
    np.testing.assert_allclose(A_inv, A_inv.T, rtol=1e-12, atol=1e-14)


def test_sym_inv_logdet_is_a_true_inverse(spd_matrix):
    A_inv, _ = _sym_inv_logdet(spd_matrix)
    np.testing.assert_allclose(A_inv @ spd_matrix, np.eye(len(spd_matrix)), atol=1e-10)


def test_sym_inv_logdet_clips_singular_matrix():
    '''A rank-deficient input must not blow up: eigenvalues are clipped at eps.'''
    rng = np.random.default_rng(11)
    B = rng.normal(size=(5, 2))
    A = B @ B.T                      # rank 2, so 3 zero eigenvalues

    A_inv, logdet = _sym_inv_logdet(A, eps=1e-10)

    assert np.all(np.isfinite(A_inv))
    assert np.isfinite(logdet)
    # the clipped directions contribute 1/eps, not inf
    assert np.max(np.abs(A_inv)) <= 1 / 1e-10


# --- e_step ----------------------------------------------------------------

def test_e_step_shapes(wpsi):
    X, W, Psi = wpsi
    N, k = X.shape[0], W.shape[1]

    Ez, Ezz = e_step(X, W, Psi)

    assert Ez.shape == (N, k)
    assert Ezz.shape == (N, k, k)


def test_e_step_matches_textbook_form(wpsi):
    '''M = (I + W^T Psi^-1 W)^-1,  E[z_i] = M W^T Psi^-1 x_i.'''
    X, W, Psi = wpsi
    k = W.shape[1]

    Ez, _ = e_step(X, W, Psi)

    Psi_inv = np.linalg.inv(Psi)
    M = np.linalg.inv(np.eye(k) + W.T @ Psi_inv @ W)
    expected = np.array([M @ W.T @ Psi_inv @ x for x in X])
    np.testing.assert_allclose(Ez, expected, rtol=1e-9, atol=1e-11)


def test_e_step_matches_woodbury_form(wpsi):
    '''
    E[z] = X C^-1 W with C = W W^T + Psi.

    Algebraically the same posterior as the textbook form, but reached by a
    different route -- so a transposed or misplaced factor shows up here.
    '''
    X, W, Psi = wpsi

    Ez, _ = e_step(X, W, Psi)

    C_inv = np.linalg.inv(W @ W.T + Psi)
    np.testing.assert_allclose(Ez, X @ C_inv @ W, rtol=1e-9, atol=1e-11)


def test_e_step_second_moment_decomposition(wpsi):
    '''E[z z^T] = Cov(z|x) + E[z] E[z]^T, with the covariance shared by all samples.'''
    X, W, Psi = wpsi
    k = W.shape[1]

    Ez, Ezz = e_step(X, W, Psi)

    Psi_inv = np.linalg.inv(Psi)
    M = np.linalg.inv(np.eye(k) + W.T @ Psi_inv @ W)
    expected = M[None, :, :] + np.einsum('nk,nl->nkl', Ez, Ez)
    np.testing.assert_allclose(Ezz, expected, rtol=1e-9, atol=1e-11)


def test_e_step_second_moments_are_symmetric_psd(wpsi):
    X, W, Psi = wpsi

    _, Ezz = e_step(X, W, Psi)

    for S in Ezz:
        np.testing.assert_allclose(S, S.T, rtol=1e-10, atol=1e-12)
        assert np.linalg.eigvalsh(S).min() > -1e-10


def test_e_step_shrinks_toward_prior():
    '''With huge observation noise the posterior mean is pulled to zero.'''
    rng = np.random.default_rng(12)
    X = rng.normal(size=(30, 4))
    X = X - X.mean(axis=0)
    W = rng.normal(size=(4, 2))

    Ez_loud, _ = e_step(X, W, 1e4 * np.eye(4))
    Ez_quiet, _ = e_step(X, W, 1e-2 * np.eye(4))

    assert np.abs(Ez_loud).max() < np.abs(Ez_quiet).max()
    assert np.abs(Ez_loud).max() < 1e-2


# --- m_step_W --------------------------------------------------------------

def test_m_step_W_solves_normal_equations(wpsi):
    '''W_new (sum_i Ezz_i) = sum_i x_i Ez_i^T is the equation the M-step claims to solve.'''
    X, W, Psi = wpsi
    Ez, Ezz = e_step(X, W, Psi)

    W_new = m_step_W(X, Ez, Ezz)

    np.testing.assert_allclose(W_new @ Ezz.sum(axis=0), X.T @ Ez, rtol=1e-7, atol=1e-9)


def test_m_step_W_shape(wpsi):
    X, W, Psi = wpsi
    Ez, Ezz = e_step(X, W, Psi)

    assert m_step_W(X, Ez, Ezz).shape == W.shape


def test_m_step_W_recovers_exact_loading_in_noiseless_limit():
    '''If x_i = W z_i exactly and the posterior is a point mass at z_i, the M-step returns W.'''
    rng = np.random.default_rng(13)
    W_true = rng.normal(size=(5, 2))
    Z = rng.normal(size=(60, 2))
    X = Z @ W_true.T

    Ez = Z
    Ezz = np.einsum('nk,nl->nkl', Z, Z)

    np.testing.assert_allclose(m_step_W(X, Ez, Ezz), W_true, rtol=1e-8, atol=1e-10)


# --- log_likelihood --------------------------------------------------------

def test_log_likelihood_matches_scipy(wpsi):
    X, W, Psi = wpsi
    C = W @ W.T + Psi

    ll = log_likelihood(X, W, Psi)

    expected = multivariate_normal(mean=np.zeros(len(C)), cov=C).logpdf(X).sum()
    np.testing.assert_allclose(ll, expected, rtol=1e-9)


def test_log_likelihood_prefers_the_true_covariance():
    '''The generating parameters must score higher than a mismatched W.'''
    rng = np.random.default_rng(14)
    W_true = rng.normal(size=(6, 2))
    X = rng.normal(size=(400, 2)) @ W_true.T + 0.3 * rng.normal(size=(400, 6))
    X = X - X.mean(axis=0)
    Psi = 0.09 * np.eye(6)

    ll_true = log_likelihood(X, W_true, Psi)
    ll_wrong = log_likelihood(X, rng.normal(size=(6, 2)), Psi)

    assert ll_true > ll_wrong
