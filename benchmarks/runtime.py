"""Runtime and peak memory: one-factor sweeps over N, d, k, and the (N, d) grid
for pPCA closed form vs EM.

Timing and memory come from separate fits (tracemalloc distorts timing).
Data generation is not timed.
"""

from benchmarks.config import (
    BOUNDARY_CONFIG,
    MODELS,
    RUNTIME_CONFIG,
    SUBSETTING_SWEEPS,
    BenchmarkConfig,
    GridConfig,
)
from benchmarks.data import generate_ground_truth, realized_snr, subset_by_dims, subset_by_samples
from benchmarks.models import REFERENCE_MODELS, fit_model, peak_memory

RUNTIME_MODELS = MODELS + REFERENCE_MODELS
GRID_MODELS = ("ppca_cf", "ppca_em")
MEMORY_RUNS = 2  # peak memory barely varies across runs; measure it in the first 2 only


def _measure(model, X, k, seed, memory: bool) -> dict:
    r = fit_model(model, X, k, seed, infer=False)
    return {
        "model": model,
        "total_time": r.total_time,
        "n_iter": r.n_iter,
        "time_per_iter": r.total_time / r.n_iter if r.n_iter > 0 else float("nan"),
        "converged": r.converged,
        "decreasing_ll": r.decreasing_ll,
        "heywood": r.heywood,
        "peak_mem_mb": peak_memory(model, X, k, seed) if memory else float("nan"),
    }


# --- one-factor sweeps -----------------------------------------------------------


def sweep_tasks(config: BenchmarkConfig = RUNTIME_CONFIG) -> list[tuple]:
    """One task per (sweep, run)."""
    return [(s, i) for s in config.get_active_sweeps() for i in range(config.n_runs)]


def run_sweep_task(
    sweep: str,
    run_idx: int,
    config: BenchmarkConfig = RUNTIME_CONFIG,
    models: tuple[str, ...] = RUNTIME_MODELS,
) -> list[dict]:
    memory = run_idx < MEMORY_RUNS
    values = config.get_sweep_config(sweep).values
    N, d, k = int(config.N.default), int(config.d.default), int(config.k.default)
    rows = []

    if sweep in SUBSETTING_SWEEPS:  # generate once at max, slice
        N_gen = int(config.N.max_value) if sweep == "N" else N
        d_gen = int(config.d.max_value) if sweep == "d" else d
        sim = generate_ground_truth(N_gen, d_gen, k, config.snr, isotropic=False,
                                    seed=config.get_data_seed(sweep, run_idx))

    for value_idx, value in enumerate(values):
        if sweep == "N":
            X, _ = subset_by_samples(sim, value)
            point, snr_used = {"N": value, "d": d, "k": k}, config.snr
        elif sweep == "d":
            X, params = subset_by_dims(sim, value)  # re-standardized to config.snr
            point, snr_used = {"N": N, "d": value, "k": k}, realized_snr(params)
        else:  # k: fresh ground truth per value
            sim_k = generate_ground_truth(N, d, value, config.snr, isotropic=False,
                                          seed=config.get_data_seed(sweep, run_idx, value_idx))
            X = sim_k.X
            point, snr_used = {"N": N, "d": d, "k": value}, config.snr

        seed = config.get_fit_seed(sweep, run_idx, value_idx)
        for model in models:
            rows.append({"sweep": sweep, "run_idx": run_idx, **point, "snr": snr_used,
                         **_measure(model, X, point["k"], seed, memory)})
    return rows


# --- (N, d) grid: pPCA closed form vs EM ------------------------------------------


def grid_tasks(config: GridConfig = BOUNDARY_CONFIG) -> list[tuple]:
    """One task per (run, d): each task slices its own d from a max-size generation."""
    return [(i, j) for i in range(config.n_runs) for j in range(len(config.d_values))]


def run_grid_task(run_idx: int, d_idx: int, config: GridConfig = BOUNDARY_CONFIG) -> list[dict]:
    memory = run_idx < MEMORY_RUNS
    # Same data seed for every d_idx in a run -> identical ground truth, sliced.
    # Generating at (max_N, d) rather than (max_N, max_d) and slicing would give
    # different W per d; we keep the latents design (generate at max, slice).
    sim = generate_ground_truth(config.max_N, config.max_d, config.k, config.snr,
                                isotropic=True, seed=config.get_data_seed(run_idx))
    d = config.d_values[d_idx]
    X_d, params = subset_by_dims(sim, d)
    del sim  # free the max-size copy before fitting
    rows = []
    for N_idx, N in enumerate(config.N_values):
        if config.k >= min(N, d):
            continue
        X = X_d[:N].copy()
        seed = config.get_fit_seed(run_idx, N_idx, d_idx)
        for model in GRID_MODELS:
            rows.append({"run_idx": run_idx, "N": N, "d": d, "k": config.k,
                         "snr": realized_snr(params), **_measure(model, X, config.k, seed, memory)})
    return rows