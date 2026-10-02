"""Tests for probabilistic PCA fitted by EM.

pPCA has a closed-form ML solution (Tipping & Bishop 1999), so most of these
compare the EM fixed point against it directly.
"""

import numpy as np
import pytest
from helpers import em_ll_trace, subspace_dist
from sklearn.decomposition import PCA as SklearnPCA

from mini_latents.em_core import log_likelihood
from mini_latents.noise_model import IsotropicNoise
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


def test_em_log_likelihood_is_monotone(iso_data):
    """EM guarantees the marginal likelihood never decreases."""
    X, k = iso_data["X"], iso_data["k"]
    Xc = X - X.mean(axis=0)

    model = pPCA(k)
    lls = em_ll_trace(IsotropicNoise(), Xc, model._init_W(Xc, k)[0], n_iter=100)

    assert np.all(np.diff(lls) >= -1e-8)


def test_em_improves_on_its_initialization(iso_data):
    X, k = iso_data["X"], iso_data["k"]
    Xc = X - X.mean(axis=0)

    model = pPCA(k)
    W0, _ = model._init_W(Xc, k)
    noise = IsotropicNoise()
    noise.initialize(Xc)
    psi = noise.noise_as_vec()
    ll_start = log_likelihood(Xc, W0, psi)

    model.fit(X, k, **CONVERGED)
    ll_end = log_likelihood(Xc, model.components_, model.noise_model.noise_as_vec())

    assert ll_end > ll_start


def test_reported_log_likelihood_matches_final_parameters(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    recomputed = log_likelihood(
        X - model.mean_, model.components_, model.noise_model.noise_as_vec()
    )
    np.testing.assert_allclose(model.log_likelihood_, recomputed, rtol=1e-9)


def test_n_iter_respects_max_iter(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, max_iter=7, tol=0, method="em")

    assert model.n_iter_ == 7


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_em_from_random_start_reaches_the_closed_form(iso_data, monkeypatch, seed):
    """
    EM started from a random (W, sigma^2) converges to the closed-form ML solution
    and stops on its own.

    W is identified only up to rotation (W R gives the same model), so W itself is
    not compared; the implied covariance W W^T + sigma^2 I is.

    Tolerances: near the maximum the ll is quadratic in the parameters, so stopping
    at relative ll error ~tol leaves parameter errors ~sqrt(tol). At tol=1e-12 this
    data measures ll ~1e-12, sigma^2 ~3e-8, covariance ~6e-6 (relative); the bounds
    below leave about 100x headroom.
    """
    X, k = iso_data["X"], iso_data["k"]
    d = X.shape[1]

    closed = pPCA(k).fit(X)
    C_closed = closed.components_ @ closed.components_.T + closed.noise_model.psi * np.eye(d)

    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(d, k))
    sigma2_0 = rng.uniform(0.5, 2.0)
    model = pPCA(k)
    monkeypatch.setattr(model, "_init_W", lambda Xc, n_components: (W0, sigma2_0))
    model.fit(X, method="em", tol=1e-12, max_iter=100_000)
    C_em = model.components_ @ model.components_.T + model.noise_model.psi * np.eye(d)

    assert model.flags.converged
    assert not model.flags.decreasing_ll
    assert model.n_iter_ > 2
    # the closed form is the global maximum: EM can approach it, not exceed it
    assert model.log_likelihood_ <= closed.log_likelihood_ + 1e-12 * abs(closed.log_likelihood_)
    np.testing.assert_allclose(model.log_likelihood_, closed.log_likelihood_, rtol=1e-10)
    np.testing.assert_allclose(model.noise_model.psi, closed.noise_model.psi, rtol=1e-6)
    np.testing.assert_allclose(C_em, C_closed, rtol=0, atol=1e-4 * np.abs(C_closed).max())


# --- recovery and limits ---------------------------------------------------


def test_recovers_the_generating_subspace(iso_data):
    X, k, W_true = iso_data["X"], iso_data["k"], iso_data["W_true"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    assert subspace_dist(model.components_, W_true) < 0.1


def test_recovers_the_generating_noise_variance(iso_data):
    X, k, psi_true = iso_data["X"], iso_data["k"], iso_data["psi_true"]

    model = pPCA(k).fit(X, k, **CONVERGED)

    np.testing.assert_allclose(model.noise_model.psi, psi_true, rtol=0.15)


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


def test_infer_latents_and_inverse_transform_shapes(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)
    Z = model.infer_latents(X)

    assert Z.shape == (len(X), k)
    assert model.inverse_transform(Z).shape == X.shape


def test_reconstruction_beats_the_mean_only_baseline(iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = pPCA(k).fit(X, k, **CONVERGED)
    recon_err = np.sum((X - model.inverse_transform(model.infer_latents(X))) ** 2)

    assert recon_err < np.sum((X - X.mean(axis=0)) ** 2)
