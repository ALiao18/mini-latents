"""Tests for probabilistic PCA fitted by EM.

pPCA has a closed-form ML solution (Tipping & Bishop 1999), so most of these
compare the EM fixed point against it directly.
"""

import numpy as np
import pytest
from helpers import subspace_dist
from sklearn.decomposition import PCA as SklearnPCA

from mini_latents.em_core import log_likelihood
from mini_latents.pca import PCA
from mini_latents.ppca_fa import pPCA

# These tests compare a fit against a closed form, against sklearn, or against
# another fit, so EM has to get close to the optimum. A tight relative tol stops
# with ll error ~1e-12 and parameter error ~sqrt(tol) = 1e-6.
CONVERGED = {"max_iter": 100_000, "tol": 1e-12}


def _spectrum(X):
    """Eigenvalues of the ddof=0 sample covariance, descending."""
    Xc = X - X.mean(axis=0)
    return np.sort(np.linalg.eigvalsh(np.cov(Xc, rowvar=False, ddof=0)))[::-1]


# --- the closed-form ML solution -------------------------------------------


def test_noise_variance_is_the_mean_discarded_eigenvalue(iso_data):
    """sigma^2_ML = (1 / (d - k)) * sum_{j>k} lambda_j."""
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    np.testing.assert_allclose(
        model.noise_model.psi, _spectrum(X)[k:].mean(), rtol=1e-6
    )


def test_loading_singular_values_match_closed_form(iso_data):
    """The singular values of W_ML are sqrt(lambda_i - sigma^2)."""
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    eigvals = _spectrum(X)
    sigma2 = eigvals[k:].mean()
    np.testing.assert_allclose(
        np.linalg.svd(model.components_, compute_uv=False),
        np.sqrt(eigvals[:k] - sigma2),
        rtol=1e-5,
    )


def test_subspace_matches_pca(iso_data):
    """W_ML spans the same subspace as the top-k principal components."""
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    assert subspace_dist(model.components_, PCA(k).fit(X).components_) < 1e-8


def test_implied_covariance_matches_sklearn_pca(iso_data):
    """
    C = W W^T + sigma^2 I is rotation-invariant, so it is directly comparable
    to sklearn's probabilistic-PCA covariance -- after the same ddof=0 vs
    ddof=1 rescale that test_pca.py pins down.
    """
    X, k = iso_data["X"], iso_data["k"]
    N = len(X)

    model = pPCA(k).fit(X, k, **CONVERGED)
    C = model.components_ @ model.components_.T + model.noise_model.as_matrix()

    C_sk = SklearnPCA(n_components=k).fit(X).get_covariance()
    assert np.linalg.norm(C * N / (N - 1) - C_sk) / np.linalg.norm(C_sk) < 1e-6


# --- EM behaviour ----------------------------------------------------------


@pytest.mark.fit
def test_ll_monotonicity(ppca_fitted_converged):
    """Test that the log-likelihood from fit() is monotonically non-decreasing.

    Uses sqrt(machine epsilon) as tolerance to account for floating-point
    accumulation errors while remaining precision-aware.
    """
    ll = np.array(ppca_fitted_converged.ll_history_)

    # sqrt(machine epsilon) as tolerance for floating-point accumulation
    tol = float(np.sqrt(np.finfo(ll.dtype).eps))

    ll_diff = np.diff(ll)
    assert np.all(ll_diff >= -tol), (
        f"Log-likelihood decreased by more than tolerance. "
        f"Min diff: {ll_diff.min():.2e}, tolerance: {-tol:.2e}"
    )


def test_reported_log_likelihood_matches_final_parameters(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    Xc = X - model.mean_
    recomputed = log_likelihood(
        model.components_,
        model.noise_model.noise_as_vec(),
        Xc.T @ Xc / len(Xc),
        len(Xc),
    )
    np.testing.assert_allclose(model.log_likelihood_, recomputed, rtol=1e-9)

# --- recovery and limits ---------------------------------------------------


def test_collapses_onto_pca_as_noise_vanishes(tiny_noise_data):
    """
    As sigma -> 0 the pPCA and PCA subspaces must coincide. Regression test for
    the ddof mismatch between the two covariance estimates.
    """
    X, k = tiny_noise_data["X"], tiny_noise_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    assert subspace_dist(model.components_, PCA(k).fit(X).components_) < 1e-6
    assert model.noise_model.psi < 1e-8


@pytest.mark.parametrize("k", [1, 2, 5, 7])
def test_converges_across_ranks(iso_data, k):
    """Including the boundaries k=1 and k=d-1."""
    X = iso_data["X"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    assert model.components_.shape == (X.shape[1], k)
    assert np.all(np.isfinite(model.components_))
    np.testing.assert_allclose(
        model.noise_model.psi, _spectrum(X)[k:].mean(), rtol=1e-4
    )
