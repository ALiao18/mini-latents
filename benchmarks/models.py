"""One fitting interface for every benchmark: same models, same settings everywhere."""

import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import FactorAnalysis as SkFA

from benchmarks.config import FIT, FitSettings
from mini_latents.pca import PCA
from mini_latents.ppca_fa import FA, pPCA

REFERENCE_MODELS = ("sk_pca", "sk_fa")  # runtime comparison only, not recovery


@dataclass
class FitResult:
    model: str
    W: np.ndarray  # (d, k) loadings (components_)
    mu: np.ndarray  # (d,) fitted mean
    psi: np.ndarray  # (d,) noise variances (PCA: mean discarded eigenvalue, broadcast)
    Z: np.ndarray | None  # (N, k) inferred latents (posterior mean / PCA scores)
    n_iter: int  # 0 for closed-form solutions
    converged: bool
    decreasing_ll: bool
    heywood: bool
    total_time: float  # seconds, fit() only (excludes inference of Z)


def _build_and_fit(name, X, k, seed, fit: FitSettings):
    """Construct the model and run fit(). Returns the fitted estimator."""
    if name == "pca":
        return PCA(k).fit(X)
    if name == "ppca_cf":
        return pPCA(k).fit(X, method="closed_form")
    if name == "ppca_em":
        return pPCA(k).fit(X, method="em", random_seed=seed, max_iter=fit.max_iter, tol=fit.tol)
    if name == "fa":
        return FA(k).fit(
            X, max_iter=fit.max_iter, tol=fit.tol, random_seed=seed, min_var_frac=fit.min_var_frac
        )
    if name == "sk_pca":
        return SkPCA(n_components=k).fit(X)  # svd_solver="auto", out-of-the-box
    if name == "sk_fa":
        # sklearn defaults (tol=1e-2 on absolute ll change, randomized SVD): different
        # stopping rule from ours, so compare iterations as well as time.
        return SkFA(n_components=k, max_iter=fit.max_iter, random_state=_int_seed(seed)).fit(X)
    msg = f"unknown model '{name}'"
    raise ValueError(msg)


def _int_seed(seed) -> int | None:
    """sklearn needs an int; derive one from a SeedSequence-style list."""
    if seed is None:
        return None
    return int(np.random.SeedSequence(seed).generate_state(1)[0])


def fit_model(
    name: str,
    X: np.ndarray,
    k: int,
    seed: int | Sequence[int] | None,
    fit: FitSettings = FIT,
    infer: bool = True,
) -> FitResult:
    """Fit one model and return parameters, latents, flags, and wall-clock fit time."""
    t0 = time.perf_counter()
    m = _build_and_fit(name, X, k, seed, fit)
    total_time = time.perf_counter() - t0

    d = X.shape[1]
    if name in ("sk_pca", "sk_fa"):
        W = m.components_.T  # sklearn: (k, d)
        psi = np.full(d, m.noise_variance_) if name == "sk_pca" else m.noise_variance_
        n_iter = getattr(m, "n_iter_", 0)
        return FitResult(name, W, m.mean_, psi, None, n_iter, True, False, False, total_time)

    psi = np.full(d, m.noise_variance_) if name == "pca" else m.noise_model.noise_as_vec()
    flags = m.flags
    return FitResult(
        model=name,
        W=m.components_,
        mu=m.mean_,
        psi=np.asarray(psi, dtype=float),
        Z=m.infer_latents(X) if infer else None,
        n_iter=getattr(m, "n_iter_", 0),
        converged=bool(flags.converged) or name == "pca",
        decreasing_ll=bool(flags.decreasing_ll),
        heywood=bool(flags.heywood),
        total_time=total_time,
    )


def peak_memory(name: str, X: np.ndarray, k: int, seed, fit: FitSettings = FIT) -> float:
    """
    Peak Python-tracked allocation (MB) during fit(), in a SEPARATE fit from timing:
    tracemalloc slows every allocation. NumPy arrays are tracked; LAPACK's internal
    workspace is not, so this is a lower bound dominated by the arrays we create.
    """
    tracemalloc.start()
    try:
        _build_and_fit(name, X, k, seed, fit)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024**2