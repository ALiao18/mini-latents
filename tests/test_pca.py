"""Tests for the eigendecomposition-based PCA."""

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from mini_latents.pca import PCA


def test_components_shape_and_orthonormality(iso_data):
    X, k = iso_data["X"], iso_data["k"]
    d = X.shape[1]

    pca = PCA(k).fit(X)

    assert pca.components_.shape == (d, k)
    np.testing.assert_allclose(
        pca.components_.T @ pca.components_, np.eye(k), atol=1e-10
    )


def test_explained_variance_equals_score_variance(iso_data):
    """explained_variance_[j] is the variance of the j-th score, with ddof=0."""
    X, k = iso_data["X"], iso_data["k"]

    pca = PCA(k).fit(X)
    Z = pca.infer_latents(X)

    np.testing.assert_allclose(
        Z.var(axis=0, ddof=0), pca.explained_variance_[:k], rtol=1e-10
    )


def test_infer_latents_reuses_stored_mean_on_held_out_data(iso_data):
    """New data must be centered with the training mean, not its own."""
    X = iso_data["X"]
    train, test = X[:400], X[400:]

    pca = PCA(3).fit(train)

    np.testing.assert_allclose(
        pca.infer_latents(test), (test - pca.mean_) @ pca.components_, rtol=1e-12
    )


# --- against sklearn -------------------------------------------------------


def test_components_match_sklearn_up_to_sign(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    ours = PCA(k).fit(X)
    theirs = SklearnPCA(n_components=k).fit(X)

    cosines = np.abs(np.sum(ours.components_.T * theirs.components_, axis=1))
    np.testing.assert_allclose(cosines, np.ones(k), atol=1e-8)


# --- reconstruction --------------------------------------------------------


def test_reconstruction_error_equals_discarded_eigenvalues(iso_data):
    """Eckart-Young: the residual is exactly the tail of the spectrum."""
    X, k = iso_data["X"], iso_data["k"]
    N = len(X)

    pca = PCA(k).fit(X)
    residual = X - pca.inverse_transform(pca.infer_latents(X))

    eigvals = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0)))[::-1]
    np.testing.assert_allclose(np.sum(residual**2), N * eigvals[k:].sum(), rtol=1e-8)


# --- residual noise --------------------------------------------------------


def test_noise_variance_is_the_mean_discarded_eigenvalue(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    pca = PCA(k).fit(X)

    eigvals = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False, ddof=0)))[::-1]
    np.testing.assert_allclose(pca.noise_variance_, eigvals[k:].mean(), rtol=1e-10)
