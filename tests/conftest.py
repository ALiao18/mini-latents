"""Shared data fixtures for the mini-latents test suite.

Every fixture is seeded, so the whole suite is deterministic.
"""
import numpy as np
import pytest


def _make_data(rng, N, d, k, noise_std):
    '''X = Z W^T + mu + eps, with eps ~ N(0, diag(noise_std**2)).'''
    W_true = rng.normal(size=(d, k))
    mu_true = rng.normal(size=d)
    Z = rng.normal(size=(N, k))
    X = Z @ W_true.T + mu_true + rng.normal(size=(N, d)) * noise_std
    return X, W_true, mu_true, noise_std ** 2


@pytest.fixture
def iso_data():
    '''Isotropic noise -- pPCA's generative assumption holds exactly.'''
    rng = np.random.default_rng(0)
    X, W_true, mu_true, psi_true = _make_data(rng, 500, 8, 3, 0.5)
    return dict(X=X, W_true=W_true, mu_true=mu_true, psi_true=psi_true, k=3)


@pytest.fixture
def aniso_data():
    '''
    Heteroscedastic per-feature noise -- FA's assumption holds exactly.

    The noise floor starts at 0.5 rather than near zero on purpose: a feature
    with a near-zero uniqueness sits against the Heywood boundary, where the
    ML estimate of that one variance is barely identified and its relative
    error stays large no matter how much data you add.
    '''
    rng = np.random.default_rng(1)
    noise_std = np.linspace(0.5, 1.5, 8)
    X, W_true, mu_true, psi_true = _make_data(rng, 2000, 8, 3, noise_std)
    return dict(X=X, W_true=W_true, mu_true=mu_true, psi_true=psi_true, k=3)


@pytest.fixture
def tiny_noise_data():
    '''Near-zero noise -- the regime where pPCA must collapse onto PCA.'''
    rng = np.random.default_rng(2)
    X, W_true, mu_true, psi_true = _make_data(rng, 300, 6, 2, 1e-6)
    return dict(X=X, W_true=W_true, mu_true=mu_true, psi_true=psi_true, k=2)


@pytest.fixture
def spd_matrix():
    '''A well-conditioned symmetric positive definite matrix.'''
    rng = np.random.default_rng(3)
    A = rng.normal(size=(6, 6))
    return A @ A.T + 6 * np.eye(6)
