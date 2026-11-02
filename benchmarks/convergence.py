"""FA EM vs sklearn FA on equal terms: same data, same starting psi, one likelihood.

Iteration counts are not comparable: sklearn solves W exactly by SVD given psi each
iteration, then stops on absolute ll change; we run EM and stop on relative ll gain.
So both are scored on what they achieve:
  - true log-likelihood per sample of every iterate, from one evaluator (sklearn's
    loglike_ is wrong when the randomized SVD is inexact);
  - wall-clock time of a fit stopped at the first iterate within `gap` nats/sample of
    the best likelihood any solver reached on that dataset.
"""

import json
import time
import warnings

import numpy as np
from sklearn.decomposition import FactorAnalysis as SkFA

from benchmarks.config import FIT, RUNTIME_CONFIG, BenchmarkConfig
from benchmarks.models import _int_seed
from benchmarks.runtime import sweep_points, sweep_tasks
from mini_latents.em_core import log_likelihood
from mini_latents.ppca_fa import FA, S_products

SK_SVD = {"sk_fa": "randomized", "sk_fa_lapack": "lapack"}
CONVERGENCE_MODELS = ("fa", *SK_SVD)
TARGET_GAPS = (1e-1, 1e-2, 1e-3, 1e-4)  # nats/sample below the best likelihood
SK_MAX_ITER = 300  # sklearn iterates are scored by refitting, O(n_iter^2) iterations

tasks = sweep_tasks


def _fit(model, X, k, seed, max_iter, tol):
    """Fit and return (estimator, W, psi). sklearn starts from psi = diag(S), like FA."""
    if model == "fa":
        m = FA(k).fit(
            X,
            max_iter=max_iter,
            tol=tol,
            random_seed=seed,
            min_var_frac=FIT.min_var_frac,
        )
        return m, m.components_, m.noise_model.noise_as_vec()
    m = SkFA(
        n_components=k,
        tol=tol,
        max_iter=max_iter,
        svd_method=SK_SVD[model],
        noise_variance_init=X.var(axis=0),
        random_state=_int_seed(seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # stopping at max_iter is intended
        m.fit(X)
    return m, m.components_.T, m.noise_variance_


def _trace(model, X, S, k, seed) -> tuple[np.ndarray, float]:
    """(true ll per sample of each iterate, min psi_j / var_j at the end)."""
    N = X.shape[0]
    diagS = np.diag(S)
    if model == "fa":  # ll_history_ already is the true likelihood of each iterate
        m, _, psi = _fit(model, X, k, seed, FIT.max_iter, FIT.tol)
        return np.asarray(m.ll_history_) / N, float(np.min(psi / diagS))

    # tol=0 runs until the likelihood stops rising. Stopping earlier with the same seed
    # replays the same iterates, so max_iter=i recovers the parameters at iteration i.
    m, _, psi = _fit(model, X, k, seed, SK_MAX_ITER, 0.0)
    trace = []
    for i in range(1, m.n_iter_ + 1):
        _, W, psi = _fit(model, X, k, seed, i, 0.0)
        BtSB, _ = S_products(S, W, psi)
        trace.append(log_likelihood(W, psi, diagS, BtSB, N) / N)
    return np.asarray(trace), float(np.min(psi / diagS))


def _timed_fit(model, X, k, seed, n_iter) -> float:
    t0 = time.perf_counter()
    _fit(model, X, k, seed, n_iter, 0.0)
    return time.perf_counter() - t0


def run_task(
    sweep: str, run_idx: int, config: BenchmarkConfig = RUNTIME_CONFIG
) -> list[dict]:
    rows = []
    for value_idx, X, point, snr_used in sweep_points(sweep, run_idx, config):
        k = point["k"]
        seed = config.get_fit_seed(sweep, run_idx, value_idx)
        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / X.shape[0]
        del Xc
        traces = {m: _trace(m, X, S, k, seed) for m in CONVERGENCE_MODELS}
        best = max(float(np.max(t)) for t, _ in traces.values())

        for model, (trace, min_psi_frac) in traces.items():
            row = {
                "sweep": sweep,
                "run_idx": run_idx,
                **point,
                "snr": snr_used,
                "model": model,
                "n_iter": len(trace),
                "best_ll": best,
                "final_gap": best - float(trace[-1]),
                "min_psi_frac": min_psi_frac,
            }
            times = {}  # n_iter -> seconds; tight targets often land on the same iterate
            for gap in TARGET_GAPS:
                hits = np.flatnonzero(trace >= best - gap)
                n = int(hits[0]) + 1 if hits.size else None
                if n is not None and n not in times:
                    times[n] = _timed_fit(model, X, k, seed, n)
                row[f"iters_gap_{gap:.0e}"] = n if n is not None else float("nan")
                row[f"time_gap_{gap:.0e}"] = times[n] if n is not None else float("nan")
            row["ll_trace"] = json.dumps(np.round(trace, 8).tolist())
            rows.append(row)
    return rows
