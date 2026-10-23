"""Tests for factor analysis fitted by EM.

FA has no closed form, so it is checked against sklearn's independent
implementation plus the invariances the model is defined by.
"""

import numpy as np
import pytest
from sklearn.decomposition import FactorAnalysis as SklearnFA

from mini_latents.em_core import log_likelihood
from mini_latents.ppca_fa import FA, pPCA

# These tests compare a fit against a closed form, against sklearn, or against
# another fit, so EM has to get close to the optimum. A tight relative tol stops
# with ll error ~1e-12 and parameter error ~sqrt(tol) = 1e-6.
CONVERGED = {"max_iter": 100_000, "tol": 1e-12}


def _implied_cov(model):
    return model.components_ @ model.components_.T + model.noise_model.as_matrix()


# --- against sklearn -------------------------------------------------------


def test_implied_covariance_matches_sklearn(aniso_data):
    """
    sklearn's loadings are an arbitrary rotation of ours, but W W^T + Psi is
    rotation-invariant and so directly comparable.
    """
    X, k = aniso_data["X"], aniso_data["k"]

    ours = FA(k).fit(X, k, **CONVERGED)
    theirs = SklearnFA(n_components=k, max_iter=3000, tol=1e-11).fit(X)

    C, C_sk = _implied_cov(ours), theirs.get_covariance()
    assert np.linalg.norm(C - C_sk) / np.linalg.norm(C_sk) < 1e-6


def test_uniquenesses_match_sklearn(aniso_data):
    X, k = aniso_data["X"], aniso_data["k"]

    ours = FA(k).fit(X, k, **CONVERGED)
    theirs = SklearnFA(n_components=k, max_iter=3000, tol=1e-11).fit(X)

    np.testing.assert_allclose(ours.noise_model.psi, theirs.noise_variance_, rtol=1e-4)


# --- EM behaviour ----------------------------------------------------------


@pytest.mark.fit
def test_ll_monotonicity(fa_fitted_converged):
    """Test that the log-likelihood from fit() is monotonically non-decreasing.

    Uses sqrt(machine epsilon) as tolerance to account for floating-point
    accumulation errors while remaining precision-aware.
    """
    ll = np.array(fa_fitted_converged.ll_history_)

    # sqrt(machine epsilon) as tolerance for floating-point accumulation
    tol = float(np.sqrt(np.finfo(ll.dtype).eps))

    ll_diff = np.diff(ll)
    assert np.all(ll_diff >= -tol), (
        f"Log-likelihood decreased by more than tolerance. "
        f"Min diff: {ll_diff.min():.2e}, tolerance: {-tol:.2e}"
    )


def test_fa_fits_at_least_as_well_as_ppca(aniso_data):
    """
    pPCA is FA with Psi constrained to be isotropic, so FA's optimum can never
    be worse on the same data.
    """
    X, k = aniso_data["X"], aniso_data["k"]
    Xc = X - X.mean(axis=0)

    fa = FA(k).fit(X, k, **CONVERGED)
    pp = pPCA(k).fit(X, k, **CONVERGED)

    S, N = Xc.T @ Xc / len(Xc), len(Xc)

    ll_fa = log_likelihood(fa.components_, fa.noise_model.noise_as_vec(), S, N)
    ll_pp = log_likelihood(pp.components_, pp.noise_model.noise_as_vec(), S, N)
    assert ll_fa > ll_pp


# --- recovery --------------------------------------------------------------


# --- the invariance that defines FA ----------------------------------------


def test_is_equivariant_under_per_feature_rescaling(aniso_data):
    """
    Rescaling feature j by c_j must rescale the fitted covariance to
    diag(c) C diag(c). FA is scale-equivariant per feature; pPCA is not, which
    is exactly what separates the two models.
    """
    X, k = aniso_data["X"], aniso_data["k"]
    c = np.array([0.5, 2.0, 1.0, 3.0, 0.25, 1.5, 4.0, 0.75])

    base = FA(k).fit(X, **CONVERGED)
    scaled = FA(k).fit(X * c, **CONVERGED)

    # Each fit stops at its own iteration (rounding moves the stop by a few), so the
    # two agree to the stopping accuracy, sqrt(tol) = 1e-6, not to rounding.
    expected = c[:, None] * _implied_cov(base) * c[None, :]
    assert (
        np.linalg.norm(_implied_cov(scaled) - expected) / np.linalg.norm(expected)
        < 1e-6
    )
