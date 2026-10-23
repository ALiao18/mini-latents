"""Contract-level tests shared across PCA, pPCA and FA, plus known API defects."""

import numpy as np
import pytest

from mini_latents.pca import PCA
from mini_latents.ppca_fa import FA, pPCA


def _fit(cls, X, k):
    """PCA takes k from the constructor; pPCA/FA require it again on fit()."""
    model = cls(k)
    return model.fit(X) if cls is PCA else model.fit(X, k, max_iter=50)


PROBABILISTIC = [pPCA, FA]
ALL_MODELS = [PCA] + PROBABILISTIC


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_fit_returns_self(cls, iso_data):
    model = cls(iso_data["k"])
    X, k = iso_data["X"], iso_data["k"]

    returned = model.fit(X) if cls is PCA else model.fit(X, k, max_iter=50)

    assert returned is model


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_fit_stores_the_training_mean(cls, iso_data):
    X = iso_data["X"]

    model = _fit(cls, X, iso_data["k"])

    np.testing.assert_allclose(model.mean_, X.mean(axis=0))


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_infer_latents_and_inverse_transform_shapes(cls, iso_data):
    X, k = iso_data["X"], iso_data["k"]

    model = _fit(cls, X, k)
    Z = model.infer_latents(X)

    assert Z.shape == (len(X), k)
    assert model.inverse_transform(Z).shape == X.shape


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_components_largest_entry_is_positive(cls, iso_data):
    """Each column's sign is arbitrary (v and -v fit equally well); fit picks the one whose largest |entry| is positive.

    k = d - 1 (the most pPCA allows): with few columns all signs can come out positive by chance.
    """
    X = iso_data["X"]
    W = _fit(cls, X, X.shape[1] - 1).components_

    largest = W[np.argmax(np.abs(W), axis=0), np.arange(W.shape[1])]
    assert np.all(largest > 0)


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_fitting_twice_is_deterministic(cls, iso_data):
    """No hidden RNG anywhere in fit()."""
    X, k = iso_data["X"], iso_data["k"]

    a = _fit(cls, X, k)
    b = _fit(cls, X, k)

    np.testing.assert_array_equal(a.components_, b.components_)


# --- n_components handling -------------------------------------------------


@pytest.mark.parametrize("cls", PROBABILISTIC)
def test_n_components_attribute_stays_consistent_with_components(cls, iso_data):
    X = iso_data["X"]

    model = cls(2).fit(X, 4, max_iter=20)

    assert model.n_components == model.components_.shape[1]


# --- EM convergence contract -----------------------------------------------
