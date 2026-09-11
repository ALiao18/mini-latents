"""Tests for the eigendecomposition-based PCA."""
import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from mini_latents.pca import PCA
from mini_latents.ppca_fa import pPCA


def test_components_shape_and_orthonormality(iso_data):
    X, k = iso_data['X'], iso_data['k']
    d = X.shape[1]

    pca = PCA(k).fit(X)

    assert pca.components_.shape == (d, k)
    np.testing.assert_allclose(pca.components_.T @ pca.components_, np.eye(k), atol=1e-10)


def test_fit_returns_self(iso_data):
    pca = PCA(3)
    assert pca.fit(iso_data['X']) is pca


def test_mean_is_the_column_mean(iso_data):
    X = iso_data['X']
    pca = PCA(3).fit(X)
    np.testing.assert_allclose(pca.mean_, X.mean(axis=0))


def test_explained_variance_is_positive_and_descending(iso_data):
    pca = PCA(5).fit(iso_data['X'])

    ev = pca.explained_variance_
    assert np.all(ev > 0)
    assert np.all(np.diff(ev) < 0)


def test_explained_variance_equals_score_variance(iso_data):
    '''explained_variance_[j] is the variance of the j-th score, with ddof=0.'''
    X, k = iso_data['X'], iso_data['k']

    pca = PCA(k).fit(X)
    Z = pca.transform(X)

    np.testing.assert_allclose(Z.var(axis=0, ddof=0), pca.explained_variance_, rtol=1e-10)


def test_scores_are_centered_and_uncorrelated(iso_data):
    X, k = iso_data['X'], iso_data['k']

    Z = PCA(k).fit(X).transform(X)

    np.testing.assert_allclose(Z.mean(axis=0), np.zeros(k), atol=1e-10)
    cov = np.cov(Z, rowvar=False, ddof=0)
    np.testing.assert_allclose(cov - np.diag(np.diag(cov)), np.zeros((k, k)), atol=1e-10)


def test_transform_reuses_stored_mean_on_held_out_data(iso_data):
    '''New data must be centered with the training mean, not its own.'''
    X = iso_data['X']
    train, test = X[:400], X[400:]

    pca = PCA(3).fit(train)

    np.testing.assert_allclose(
        pca.transform(test), (test - pca.mean_) @ pca.components_, rtol=1e-12
    )


# --- against sklearn -------------------------------------------------------

def test_components_match_sklearn_up_to_sign(iso_data):
    X, k = iso_data['X'], iso_data['k']

    ours = PCA(k).fit(X)
    theirs = SklearnPCA(n_components=k).fit(X)

    cosines = np.abs(np.sum(ours.components_.T * theirs.components_, axis=1))
    np.testing.assert_allclose(cosines, np.ones(k), atol=1e-8)


def test_explained_variance_matches_sklearn_after_ddof_rescale(iso_data):
    '''
    This PCA uses ddof=0 (so it agrees with pPCA's covariance); sklearn uses
    ddof=1. The two differ by exactly N/(N-1).
    '''
    X, k = iso_data['X'], iso_data['k']
    N = len(X)

    ours = PCA(k).fit(X)
    theirs = SklearnPCA(n_components=k).fit(X)

    np.testing.assert_allclose(
        ours.explained_variance_ * N / (N - 1), theirs.explained_variance_, rtol=1e-10
    )


# --- reconstruction --------------------------------------------------------

def test_full_rank_roundtrip_is_the_identity(iso_data):
    X = iso_data['X']
    d = X.shape[1]

    pca = PCA(d).fit(X)

    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-10)


def test_reconstruction_error_equals_discarded_eigenvalues(iso_data):
    '''Eckart-Young: the residual is exactly the tail of the spectrum.'''
    X, k = iso_data['X'], iso_data['k']
    N = len(X)

    pca = PCA(k).fit(X)
    residual = X - pca.inverse_transform(pca.transform(X))

    eigvals = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0)))[::-1]
    np.testing.assert_allclose(
        np.sum(residual ** 2), N * eigvals[k:].sum(), rtol=1e-8
    )


def test_reconstruction_beats_random_subspaces(iso_data):
    '''No other k-dimensional basis can reconstruct better.'''
    X, k = iso_data['X'], iso_data['k']
    rng = np.random.default_rng(30)

    pca = PCA(k).fit(X)
    Xc = X - pca.mean_
    best = np.sum((Xc - Xc @ pca.components_ @ pca.components_.T) ** 2)

    for _ in range(20):
        Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], k)))
        assert best <= np.sum((Xc - Xc @ Q @ Q.T) ** 2) + 1e-8


def test_reconstruction_error_decreases_with_more_components(iso_data):
    X = iso_data['X']

    errors = []
    for k in range(1, X.shape[1] + 1):
        pca = PCA(k).fit(X)
        errors.append(np.sum((X - pca.inverse_transform(pca.transform(X))) ** 2))

    assert np.all(np.diff(errors) < 1e-9)


# --- residual noise --------------------------------------------------------

def test_noise_variance_is_the_mean_discarded_eigenvalue(iso_data):
    X, k = iso_data['X'], iso_data['k']

    pca = PCA(k).fit(X)

    eigvals = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0)))[::-1]
    np.testing.assert_allclose(pca.noise_variance_, eigvals[k:].mean(), rtol=1e-10)


def test_noise_cov_is_the_isotropic_residual(iso_data):
    X, k = iso_data['X'], iso_data['k']
    d = X.shape[1]

    pca = PCA(k).fit(X)

    np.testing.assert_allclose(pca.noise_cov_, pca.noise_variance_ * np.eye(d))


def test_noise_variance_is_zero_at_full_rank(iso_data):
    '''Nothing is discarded, so there is no residual left to spread.'''
    X = iso_data['X']

    pca = PCA(X.shape[1]).fit(X)

    assert pca.noise_variance_ == 0.0


def test_noise_variance_matches_ppca(iso_data):
    '''
    PCA's residual variance is exactly the sigma^2 that pPCA's EM converges to;
    the two agree because both are built on a ddof=0 covariance.
    '''
    X, k = iso_data['X'], iso_data['k']

    pca = PCA(k).fit(X)
    ppca = pPCA(k).fit(X, max_iter=2000, tol=1e-12)

    np.testing.assert_allclose(pca.noise_variance_, ppca.noise_model.psi, rtol=1e-6)
