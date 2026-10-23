import numpy as np

from benchmarks.config import MODELS, RECOVERY_CONFIG, BenchmarkConfig
from benchmarks.data import generate_ground_truth, subset_by_samples
from benchmarks.metrics import denoised_r2, relative_l2_error, subspace_error
from benchmarks.models import fit_model

NOISE_TYPES = ("iso", "fa")  # iso: pPCA's model holds; fa: FA's model holds


def tasks(config: BenchmarkConfig = RECOVERY_CONFIG) -> list[tuple]:
    """One task per (noise type, run)."""
    return [(noise, run_idx) for noise in NOISE_TYPES for run_idx in range(config.n_runs)]


def run_task(noise: str, run_idx: int, config: BenchmarkConfig = RECOVERY_CONFIG) -> list[dict]:
    N_max = int(config.N.max_value)
    d, k = int(config.d.default), int(config.k.default)

    # Same data seed for iso and fa: W and mu are identical across the pair
    # (they are drawn before psi); only the noise structure differs.
    sim = generate_ground_truth(
        N_max, d, k, config.snr, isotropic=(noise == "iso"),
        seed=config.get_data_seed("N", run_idx),
    )
    p = sim.params
    sigma2_true = float(np.mean(p.psi))  # iso: the common noise variance

    rows = []
    for value_idx, N in enumerate(config.N.values):
        X, Z_true = subset_by_samples(sim, N)
        fit_seed = config.get_fit_seed("N", run_idx, value_idx)
        for model in MODELS:
            r = fit_model(model, X, k, fit_seed)
            rows.append({
                "noise": noise,
                "run_idx": run_idx,
                "N": N,
                "d": d,
                "k": k,
                "model": model,
                "W_subspace_error": subspace_error(p.W, r.W),
                "mu_error": relative_l2_error(p.mu, r.mu),
                "psi_error": relative_l2_error(p.psi, r.psi),
                # FA -> pPCA check (meaningful on iso data): every psi_j -> sigma^2
                "psi_max_rel_dev": float(np.max(np.abs(r.psi / sigma2_true - 1.0))),
                "psi_cv": float(np.std(r.psi) / np.mean(r.psi)),
                "denoised_r2": denoised_r2(p.W, Z_true.T, p.mu, r.W, r.Z.T, r.mu),
                "n_iter": r.n_iter,
                "converged": r.converged,
                "decreasing_ll": r.decreasing_ll,
                "heywood": r.heywood,
                "total_time": r.total_time,
            })
    return rows