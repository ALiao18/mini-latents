"""
KEEP IN SYNC with ppca_fa.py: if fit() changes, the loop below needs to change.
"""

import time
from collections import defaultdict

import numpy as np

from benchmarks.config import FIT, MODELS, STAGE_CONFIG, SUBSETTING_SWEEPS, BenchmarkConfig
from benchmarks.data import generate_ground_truth, subset_by_dims, subset_by_samples
from benchmarks.models import fit_model
from mini_latents.base import top_k_eig
from mini_latents.em_core import e_step, log_likelihood, m_step_W
from mini_latents.noise_model import AnisotropicNoise
from mini_latents.ppca_fa import FA, pPCA

STAGES = (
    "center", "X2", "form_S", "init", "e_step", "XtEz",
    "m_step_W", "m_step_noise", "log_likelihood", "final_ll", "eig",
)


class _Timer:
    def __init__(self):
        self.t = defaultdict(float)

    def __call__(self, stage, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self.t[stage] += time.perf_counter() - t0
        return out


def _profile_em(model_name, X, k, seed) -> tuple[dict, int]:
    """Mirror of ProbabilisticLinearLatentModels.fit() (and pPCA closed form)."""
    T = _Timer()
    if model_name == "fa":
        m = FA(k)
        m.noise_model.min_var_frac = FIT.min_var_frac
    else:
        m = pPCA(k)
        m.method = "em" if model_name == "ppca_em" else "closed_form"
    m.random_seed = seed
    max_iter = 0 if model_name == "ppca_cf" else FIT.max_iter
    tol = FIT.tol

    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    mean = T("center", np.mean, X, axis=0)
    Xc = T("center", np.subtract, X, mean)
    X2 = T("X2", np.einsum, "ij,ij->j", Xc, Xc)
    S = T("form_S", lambda A: A.T @ A / N, Xc)

    def _init():
        W, sigma2 = m._init_W(Xc, k)
        m._init_noise(Xc, sigma2)
        return W

    W = T("init", _init)
    ll_base = ll_old = -np.inf
    n_iter = 0
    for i in range(max_iter):
        psi = m.noise_model.noise_as_vec()
        Ez, sum_Ezz = T("e_step", e_step, Xc, W, psi)
        XtEz = T("XtEz", lambda: Xc.T @ Ez)
        W = T("m_step_W", m_step_W, Ez, sum_Ezz, XtEz=XtEz)
        T("m_step_noise", m.noise_model.m_step, W, sum_Ezz, N=N, X2=X2, XtEz=XtEz)
        ll = T("log_likelihood", log_likelihood, W, m.noise_model.noise_as_vec(), S, N)
        n_iter += 1
        if i <= 1:
            ll_base = ll
        elif ll >= ll_old and (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break
        ll_old = ll

    if model_name == "ppca_cf":  # pPCA.fit recomputes S for its reported log-likelihood
        T("final_ll", lambda: log_likelihood(W, m.noise_model.noise_as_vec(), Xc.T @ Xc / N, N))
    return dict(T.t), n_iter


def _profile_pca(X, k) -> tuple[dict, int]:
    """Mirror of PCA.fit()."""
    T = _Timer()
    X = np.asarray(X, dtype=np.float64)
    mean = T("center", np.mean, X, axis=0)
    Xc = T("center", np.subtract, X, mean)
    T("eig", top_k_eig, Xc, k)
    return dict(T.t), 0


def tasks(config: BenchmarkConfig = STAGE_CONFIG) -> list[tuple]:
    return [(s, i) for s in config.get_active_sweeps() for i in range(config.n_runs)]


def run_task(sweep: str, run_idx: int, config: BenchmarkConfig = STAGE_CONFIG) -> list[dict]:
    if sweep not in SUBSETTING_SWEEPS:
        msg = f"stage profiling supports N and d sweeps, got '{sweep}'"
        raise ValueError(msg)
    N, d, k = int(config.N.default), int(config.d.default), int(config.k.default)
    N_gen = int(config.N.max_value) if sweep == "N" else N
    d_gen = int(config.d.max_value) if sweep == "d" else d
    sim = generate_ground_truth(N_gen, d_gen, k, config.snr, isotropic=False,
                                seed=config.get_data_seed(sweep, run_idx))
    rows = []
    for value_idx, value in enumerate(config.get_sweep_config(sweep).values):
        if sweep == "N":
            X, _ = subset_by_samples(sim, value)
            point = {"N": value, "d": d, "k": k}
        else:
            X, _ = subset_by_dims(sim, value)
            point = {"N": N, "d": value, "k": k}
        seed = config.get_fit_seed(sweep, run_idx, value_idx)
        for model in MODELS:
            stage_t, n_iter = _profile_pca(X, k) if model == "pca" else _profile_em(model, X, k, seed)
            fit_time = fit_model(model, X, k, seed, infer=False).total_time
            stage_sum = sum(stage_t.values())
            rows.append({
                "sweep": sweep, "run_idx": run_idx, **point, "model": model,
                "n_iter": n_iter, "fit_time": fit_time, "stage_sum": stage_sum,
                "stage_sum_over_fit": stage_sum / fit_time,
                **{f"t_{s}": stage_t.get(s, 0.0) for s in STAGES},
            })
    return rows