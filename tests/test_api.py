"""Contract-level tests shared across PCA, pPCA and FA, plus known API defects."""
import numpy as np
import pytest

from mini_latents.pca import PCA
from mini_latents.ppca_fa import pPCA, FA


def _fit(cls, X, k):
    '''PCA takes k from the constructor; pPCA/FA require it again on fit().'''
    model = cls(k)
    return model.fit(X) if cls is PCA else model.fit(X, k, max_iter=50)


PROBABILISTIC = [pPCA, FA]
ALL_MODELS = [PCA] + PROBABILISTIC


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_fit_returns_self(cls, iso_data):
    model = cls(iso_data['k'])
    X, k = iso_data['X'], iso_data['k']

    returned = model.fit(X) if cls is PCA else model.fit(X, k, max_iter=50)

    assert returned is model


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_fit_stores_the_training_mean(cls, iso_data):
    X = iso_data['X']

    model = _fit(cls, X, iso_data['k'])

    np.testing.assert_allclose(model.mean_, X.mean(axis=0))


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_transform_and_inverse_transform_shapes(cls, iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = _fit(cls, X, k)
    Z = model.transform(X)

    assert Z.shape == (len(X), k)
    assert model.inverse_transform(Z).shape == X.shape


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_components_are_finite(cls, iso_data):
    model = _fit(cls, iso_data['X'], iso_data['k'])

    assert np.all(np.isfinite(model.components_))


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_fitting_twice_is_deterministic(cls, iso_data):
    '''No hidden RNG anywhere in fit().'''
    X, k = iso_data['X'], iso_data['k']

    a = _fit(cls, X, k)
    b = _fit(cls, X, k)

    np.testing.assert_array_equal(a.components_, b.components_)


@pytest.mark.parametrize('cls', ALL_MODELS)
def test_inverse_transform_accepts_latents_it_did_not_produce(cls, iso_data):
    '''inverse_transform takes Z as an argument, so transform() need not run first.'''
    X, k = iso_data['X'], iso_data['k']
    model = _fit(cls, X, k)

    Z = np.zeros((5, k))

    np.testing.assert_allclose(model.inverse_transform(Z), np.tile(model.mean_, (5, 1)))


@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_reconstruction_improves_with_more_components(cls, iso_data):
    '''
    Not asserted as an exact roundtrip: the E-step shrinks the posterior mean
    toward the prior, so reconstruction is biased by design -- unlike PCA's
    orthogonal projection.
    '''
    X = iso_data['X']

    errors = []
    for k in (1, 2, 3, 4):
        model = cls(k).fit(X, k, max_iter=200, tol=1e-8)
        errors.append(np.sum((X - model.inverse_transform(model.transform(X))) ** 2))

    assert np.all(np.diff(errors) < 0)


# --- n_components handling -------------------------------------------------

@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_fit_uses_n_components_from_the_constructor(cls, iso_data):
    X = iso_data['X']

    model = cls(3).fit(X)

    assert model.components_.shape == (X.shape[1], 3)


@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_max_iter_zero_does_not_crash(cls, iso_data):
    X, k = iso_data['X'], iso_data['k']

    model = cls(k).fit(X, k, max_iter=0)

    assert model.n_iter_ == 0


@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_n_components_attribute_stays_consistent_with_components(cls, iso_data):
    X = iso_data['X']

    model = cls(2).fit(X, 4, max_iter=20)

    assert model.n_components == model.components_.shape[1]


# --- EM convergence contract -----------------------------------------------

@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_negative_tol_runs_every_iteration(cls, iso_data):
    '''A negative tol disables the early break, for tests that need the optimum.'''
    X, k = iso_data['X'], iso_data['k']

    model = cls(k).fit(X, max_iter=50, tol=-np.inf)

    assert model.n_iter_ == 50


@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_tol_is_per_sample(cls, iso_data):
    '''
    Tiling the data leaves the sample covariance unchanged, so EM follows an
    identical trajectory. A per-sample tol therefore stops at the same
    iteration whatever N is; a raw total would stop later and later.
    '''
    X, k = iso_data['X'], iso_data['k']

    counts = {
        len(Xr): cls(k).fit(Xr, max_iter=5000, tol=1e-9).n_iter_
        for Xr in (X, np.tile(X, (2, 1)), np.tile(X, (4, 1)))
    }

    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize('cls', PROBABILISTIC)
def test_converged_run_reports_no_monotonicity_violation(cls, iso_data, capsys):
    '''
    Once EM is at the optimum the log-likelihood wobbles by a few ulps. That is
    rounding noise, not a violation, and must not be reported as one.
    '''
    X, k = iso_data['X'], iso_data['k']

    cls(k).fit(X, max_iter=600, tol=-np.inf)

    assert 'Bug!' not in capsys.readouterr().out
