"""Tests for probabilistic PCA fitted by EM.

pPCA has a closed-form ML solution (Tipping & Bishop 1999), so most of these
compare the EM fixed point against it directly.
"""
import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from mini_latents.em_core import log_likelihood
from mini_latents.noise_model import IsotropicNoise
from mini_latents.pca import PCA
from mini_latents.ppca_fa import pPCA

from helpers import em_ll_trace, subspace_dist

TIGHT = dict(max_iter=2000, tol=1e-12)


def _spectrum(X):
    '''Eigenvalues of the ddof=0 sample covariance, descending.'''
    Xc = X - X.mean(axis=0)
    return np.sort(np.linalg.eigvalsh(np.cov(Xc, rowvar=False, ddof=0)))[::-1]


# --- the closed-form ML solution -------------------------------------------

def test_noise_variance_is_the_mean_discarded_eigenvalue(iso_data):
    '''sigma^2_ML = (1 / (d - k)) * sum_{j>k} lambda_j.'''
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)

    np.testing.assert_allclose(model.noise_model.psi, _spectrum(X)[k:].mean(), rtol=1e-6)


def test_loading_singular_values_match_closed_form(iso_data):
    '''The singular values of W_ML are sqrt(lambda_i - sigma^2).'''
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)

    eigvals = _spectrum(X)
    sigma2 = eigvals[k:].mean()
    np.testing.assert_allclose(
        np.linalg.svd(model.components_, compute_uv=False),
        np.sqrt(eigvals[:k] - sigma2),
        rtol=1e-5,
    )


def test_subspace_matches_pca(iso_data):
    '''W_ML spans the same subspace as the top-k principal components.'''
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)

    assert subspace_dist(model.components_, PCA(k).fit(X).components_) < 1e-8


def test_implied_covariance_matches_sklearn_pca(iso_data):
    '''
    C = W W^T + sigma^2 I is rotation-invariant, so it is directly comparable
    to sklearn's probabilistic-PCA covariance -- after the same ddof=0 vs
    ddof=1 rescale that test_pca.py pins down.
    '''
    X, k = iso_data['X'], iso_data['k']
    N = len(X)

    model = pPCA(k).fit(X, k, **TIGHT)
    C = model.components_ @ model.components_.T + model.noise_model.as_matrix()

    C_sk = SklearnPCA(n_components=k).fit(X).get_covariance()
    assert np.linalg.norm(C * N / (N - 1) - C_sk) / np.linalg.norm(C_sk) < 1e-6


# --- EM behaviour ----------------------------------------------------------

def test_em_log_likelihood_is_monotone(iso_data):
    '''EM guarantees the marginal likelihood never decreases.'''
    X, k = iso_data['X'], iso_data['k']
    Xc = X - X.mean(axis=0)

    model = pPCA(k)
    lls = em_ll_trace(IsotropicNoise(), Xc, model._init_W(Xc, k), n_iter=100)

    assert np.all(np.diff(lls) >= -1e-8)


def test_em_improves_on_its_initialization(iso_data):
    X, k = iso_data['X'], iso_data['k']
    Xc = X - X.mean(axis=0)

    model = pPCA(k)
    W0 = model._init_W(Xc, k)
    noise = IsotropicNoise()
    noise.initialize(Xc)
    ll_start = log_likelihood(Xc, W0, noise.as_matrix())

    model.fit(X, k, **TIGHT)
    ll_end = log_likelihood(Xc, model.components_, model.noise_model.as_matrix())

    assert ll_end > ll_start


def test_reported_log_likelihood_matches_final_parameters(iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)

    recomputed = log_likelihood(
        X - model.mean_, model.components_, model.noise_model.as_matrix()
    )
    np.testing.assert_allclose(model.log_likelihood_, recomputed, rtol=1e-9)


def test_n_iter_respects_max_iter(iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, max_iter=7, tol=1e-15)

    assert model.n_iter_ == 7


def test_stops_early_once_converged(iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, max_iter=5000, tol=1e-9)

    assert model.n_iter_ < 5000


# --- recovery and limits ---------------------------------------------------

def test_recovers_the_generating_subspace(iso_data):
    X, k, W_true = iso_data['X'], iso_data['k'], iso_data['W_true']

    model = pPCA(k).fit(X, k, **TIGHT)

    assert subspace_dist(model.components_, W_true) < 0.1


def test_recovers_the_generating_noise_variance(iso_data):
    X, k, psi_true = iso_data['X'], iso_data['k'], iso_data['psi_true']

    model = pPCA(k).fit(X, k, **TIGHT)

    np.testing.assert_allclose(model.noise_model.psi, psi_true, rtol=0.15)


def test_collapses_onto_pca_as_noise_vanishes(tiny_noise_data):
    '''
    As sigma -> 0 the pPCA and PCA subspaces must coincide. Regression test for
    the ddof mismatch between the two covariance estimates.
    '''
    X, k = tiny_noise_data['X'], tiny_noise_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)

    assert subspace_dist(model.components_, PCA(k).fit(X).components_) < 1e-6
    assert model.noise_model.psi < 1e-8


@pytest.mark.parametrize('k', [1, 2, 5, 7])
def test_converges_across_ranks(iso_data, k):
    '''Including the boundaries k=1 and k=d-1.'''
    X = iso_data['X']

    model = pPCA(k).fit(X, k, **TIGHT)

    assert model.components_.shape == (X.shape[1], k)
    assert np.all(np.isfinite(model.components_))
    np.testing.assert_allclose(model.noise_model.psi, _spectrum(X)[k:].mean(), rtol=1e-4)


def test_transform_and_inverse_transform_shapes(iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)
    Z = model.transform(X)

    assert Z.shape == (len(X), k)
    assert model.inverse_transform(Z).shape == X.shape


def test_reconstruction_beats_the_mean_only_baseline(iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = pPCA(k).fit(X, k, **TIGHT)
    recon_err = np.sum((X - model.inverse_transform(model.transform(X))) ** 2)

    assert recon_err < np.sum((X - X.mean(axis=0)) ** 2)
