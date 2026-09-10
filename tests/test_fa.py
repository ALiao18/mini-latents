"""Tests for factor analysis fitted by EM.

FA has no closed form, so it is checked against sklearn's independent
implementation plus the invariances the model is defined by.
"""
import numpy as np
from sklearn.decomposition import FactorAnalysis as SklearnFA

from mini_latents.em_core import log_likelihood
from mini_latents.noise_model import AnisotropicNoise
from mini_latents.ppca_fa import FA, pPCA

from helpers import em_ll_trace, subspace_dist

TIGHT = dict(max_iter=3000, tol=1e-11)


def _implied_cov(model):
    return model.components_ @ model.components_.T + model.noise_model.as_matrix()


# --- against sklearn -------------------------------------------------------

def test_implied_covariance_matches_sklearn(aniso_data):
    '''
    sklearn's loadings are an arbitrary rotation of ours, but W W^T + Psi is
    rotation-invariant and so directly comparable.
    '''
    X, k = aniso_data['X'], aniso_data['k']

    ours = FA(k).fit(X, k, **TIGHT)
    theirs = SklearnFA(n_components=k, max_iter=3000, tol=1e-11).fit(X)

    C, C_sk = _implied_cov(ours), theirs.get_covariance()
    assert np.linalg.norm(C - C_sk) / np.linalg.norm(C_sk) < 1e-6


def test_log_likelihood_matches_sklearn(aniso_data):
    X, k = aniso_data['X'], aniso_data['k']

    ours = FA(k).fit(X, k, **TIGHT)
    theirs = SklearnFA(n_components=k, max_iter=3000, tol=1e-11).fit(X)

    ll = log_likelihood(X - ours.mean_, ours.components_, ours.noise_model.as_matrix())
    np.testing.assert_allclose(ll, theirs.score(X) * len(X), rtol=1e-8)


def test_uniquenesses_match_sklearn(aniso_data):
    X, k = aniso_data['X'], aniso_data['k']

    ours = FA(k).fit(X, k, **TIGHT)
    theirs = SklearnFA(n_components=k, max_iter=3000, tol=1e-11).fit(X)

    np.testing.assert_allclose(ours.noise_model.psi, theirs.noise_variance_, rtol=1e-4)


# --- EM behaviour ----------------------------------------------------------

def test_em_log_likelihood_is_monotone(aniso_data):
    X, k = aniso_data['X'], aniso_data['k']
    Xc = X - X.mean(axis=0)

    model = FA(k)
    lls = em_ll_trace(AnisotropicNoise(), Xc, model._init_W(Xc, k), n_iter=150)

    assert np.all(np.diff(lls) >= -1e-8)


def test_fa_fits_at_least_as_well_as_ppca(aniso_data):
    '''
    pPCA is FA with Psi constrained to be isotropic, so FA's optimum can never
    be worse on the same data.
    '''
    X, k = aniso_data['X'], aniso_data['k']
    Xc = X - X.mean(axis=0)

    fa = FA(k).fit(X, k, **TIGHT)
    pp = pPCA(k).fit(X, k, **TIGHT)

    ll_fa = log_likelihood(Xc, fa.components_, fa.noise_model.as_matrix())
    ll_pp = log_likelihood(Xc, pp.components_, pp.noise_model.as_matrix())
    assert ll_fa > ll_pp


# --- recovery --------------------------------------------------------------

def test_recovers_the_generating_subspace(aniso_data):
    X, k, W_true = aniso_data['X'], aniso_data['k'], aniso_data['W_true']

    model = FA(k).fit(X, k, **TIGHT)

    # ||P_A - P_B|| maxes out at sqrt(2k) ~ 2.45 for k=3, so this is a tight
    # subspace match; the residual is finite-sample error and shrinks with N.
    assert subspace_dist(model.components_, W_true) < 0.4


def test_recovers_heteroscedastic_noise(aniso_data):
    '''The whole point of FA: per-feature noise variances, not one shared value.'''
    X, k, psi_true = aniso_data['X'], aniso_data['k'], aniso_data['psi_true']

    model = FA(k).fit(X, k, **TIGHT)

    np.testing.assert_allclose(model.noise_model.psi, psi_true, rtol=0.25, atol=0.02)


# --- the invariance that defines FA ----------------------------------------

def test_is_equivariant_under_per_feature_rescaling(aniso_data):
    '''
    Rescaling feature j by c_j must rescale the fitted covariance to
    diag(c) C diag(c). FA is scale-equivariant per feature; pPCA is not, which
    is exactly what separates the two models.
    '''
    X, k = aniso_data['X'], aniso_data['k']
    c = np.array([0.5, 2.0, 1.0, 3.0, 0.25, 1.5, 4.0, 0.75])

    base = FA(k).fit(X, k, **TIGHT)
    scaled = FA(k).fit(X * c, k, **TIGHT)

    expected = c[:, None] * _implied_cov(base) * c[None, :]
    assert np.linalg.norm(_implied_cov(scaled) - expected) / np.linalg.norm(expected) < 1e-6


def test_ppca_is_not_scale_equivariant(aniso_data):
    '''The contrast case: an isotropic Psi cannot absorb per-feature rescaling.'''
    X, k = aniso_data['X'], aniso_data['k']
    c = np.array([0.5, 2.0, 1.0, 3.0, 0.25, 1.5, 4.0, 0.75])

    base = pPCA(k).fit(X, k, **TIGHT)
    scaled = pPCA(k).fit(X * c, k, **TIGHT)

    expected = c[:, None] * _implied_cov(base) * c[None, :]
    assert np.linalg.norm(_implied_cov(scaled) - expected) / np.linalg.norm(expected) > 1e-2


def test_reduces_to_ppca_when_noise_is_isotropic(iso_data, aniso_data):
    '''
    Given genuinely isotropic noise, FA should land on pPCA's subspace and on a
    far flatter Psi than it finds on genuinely heteroscedastic data. The
    dispersion is compared against the anisotropic fit rather than a bare
    constant, so the test calibrates itself instead of encoding a magic number.
    '''
    X, k = iso_data['X'], iso_data['k']

    fa = FA(k).fit(X, k, **TIGHT)
    pp = pPCA(k).fit(X, k, **TIGHT)

    assert subspace_dist(fa.components_, pp.components_) < 0.15

    def dispersion(psi):
        return psi.std() / psi.mean()

    fa_aniso = FA(k).fit(aniso_data['X'], k, **TIGHT)
    assert dispersion(fa.noise_model.psi) < 0.5 * dispersion(fa_aniso.noise_model.psi)
