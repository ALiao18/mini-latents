"""Tests for the two noise models plugged into the shared EM loop."""

import numpy as np
import pytest

from mini_latents.em_core import e_step
from mini_latents.noise_model import AnisotropicNoise, IsotropicNoise


@pytest.fixture
def em_state():
    """Centered data plus one E-step's worth of posterior moments."""
    rng = np.random.default_rng(20)
    N, d, k = 200, 5, 2
    X = rng.normal(size=(N, d)) @ rng.normal(size=(d, d))
    X = X - X.mean(axis=0)
    W = rng.normal(size=(d, k))
    psi = rng.uniform(0.4, 1.1, size=d)
    Ez, Ezz = e_step(X, W, psi)
    return X, W, Ez, Ezz


# --- initialize / as_matrix ------------------------------------------------


def test_isotropic_initialize_uses_mean_feature_variance():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 4)) * np.array([1.0, 2.0, 3.0, 4.0])

    noise = IsotropicNoise()
    noise.initialize(X)

    assert noise.d == 4
    np.testing.assert_allclose(noise.psi, np.var(X, axis=0).mean())


def test_anisotropic_initialize_uses_per_feature_variance():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 4)) * np.array([1.0, 2.0, 3.0, 4.0])

    noise = AnisotropicNoise()
    noise.initialize(X)

    assert noise.d == 4
    np.testing.assert_allclose(noise.psi, np.var(X, axis=0))


def test_isotropic_as_matrix_is_scaled_identity():
    rng = np.random.default_rng(23)
    noise = IsotropicNoise()
    noise.initialize(rng.normal(size=(50, 4)))

    Psi = noise.as_matrix()

    assert Psi.shape == (4, 4)
    np.testing.assert_allclose(Psi, noise.psi * np.eye(4))


def test_anisotropic_as_matrix_is_diagonal():
    rng = np.random.default_rng(24)
    noise = AnisotropicNoise()
    noise.initialize(rng.normal(size=(50, 4)))

    Psi = noise.as_matrix()

    assert Psi.shape == (4, 4)
    np.testing.assert_allclose(Psi, np.diag(np.diag(Psi)))
    np.testing.assert_allclose(np.diag(Psi), noise.psi)


# --- m_step ----------------------------------------------------------------


def test_isotropic_m_step_is_the_mean_of_the_anisotropic_m_step(em_state):
    """
    Both M-steps average the same expected residual -- the isotropic one just
    pools across features. This cross-checks the two independent einsums
    (trace_term vs quad) against each other.
    """
    X, W, Ez, Ezz = em_state

    iso = IsotropicNoise()
    iso.initialize(X)
    iso.m_step(X, W, Ez, Ezz)

    aniso = AnisotropicNoise()
    aniso.initialize(X)
    aniso.m_step(X, W, Ez, Ezz)

    np.testing.assert_allclose(iso.psi, aniso.psi.mean(), rtol=1e-10)


def test_anisotropic_m_step_matches_naive_loop(em_state):
    """Reference implementation of sum_i (x_i - W z_i)(x_i - W z_i)^T, per feature."""
    X, W, Ez, Ezz = em_state
    N, d = X.shape

    aniso = AnisotropicNoise()
    aniso.initialize(X)
    aniso.m_step(X, W, Ez, Ezz)

    # e_step returns only the sum over samples; rebuild the shared posterior
    # covariance M from it so each E[z_i z_i^T] = M + Ez_i Ez_i^T is explicit.
    M = (Ezz - Ez.T @ Ez) / N

    expected = np.zeros(d)
    for i in range(N):
        Ezz_i = M + np.outer(Ez[i], Ez[i])
        expected += X[i] ** 2
        expected -= 2 * X[i] * (W @ Ez[i])
        expected += np.diag(W @ Ezz_i @ W.T)
    expected /= N

    np.testing.assert_allclose(aniso.psi, expected, rtol=1e-9, atol=1e-11)


def test_isotropic_m_step_matches_naive_loop(em_state):
    X, W, Ez, Ezz = em_state
    N, d = X.shape

    iso = IsotropicNoise()
    iso.initialize(X)
    iso.m_step(X, W, Ez, Ezz)

    M = (Ezz - Ez.T @ Ez) / N  # shared posterior covariance, see anisotropic test

    total = 0.0
    for i in range(N):
        Ezz_i = M + np.outer(Ez[i], Ez[i])
        total += X[i] @ X[i]
        total -= 2 * X[i] @ (W @ Ez[i])
        total += np.trace(W.T @ W @ Ezz_i)

    np.testing.assert_allclose(iso.psi, total / (N * d), rtol=1e-9)


@pytest.mark.parametrize("cls", [IsotropicNoise, AnisotropicNoise])
def test_m_step_keeps_variances_positive(cls, em_state):
    """The residual is a sum of squares, so psi can never go negative."""
    X, W, Ez, Ezz = em_state

    noise = cls()
    noise.initialize(X)
    noise.m_step(X, W, Ez, Ezz)

    assert np.all(noise.psi > 0)


@pytest.mark.parametrize("cls", [IsotropicNoise, AnisotropicNoise])
def test_m_step_with_precomputed_X2_matches_without(cls, em_state):
    """fit() passes X2 = sum_n x_n^2 to skip a pass over X; the update must not change."""
    X, W, Ez, Ezz = em_state

    with_X2, without = cls(), cls()
    with_X2.initialize(X)
    without.initialize(X)
    with_X2.m_step(X, W, Ez, Ezz, X2=np.sum(X**2, axis=0))
    without.m_step(X, W, Ez, Ezz)

    np.testing.assert_allclose(with_X2.psi, without.psi)
