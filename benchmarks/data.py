"""
observation data containers for latent variable models
"""

import numpy as np
from mini_latents.simulation import SimConfig, SimParams, SimulationResult, simulate, adjust_snr

def generate_ground_truth(N, d, k, snr, isotropic, seed) -> SimulationResult:
    """wrapper to build SimConfig and call simulate()"""
    config = SimConfig(N=N, d=d, k=k, snr=snr, isotropic=isotropic, random_seed=seed)
    return simulate(config)

def subset_by_samples(result, N) -> tuple[np.ndarray, np.ndarray]:
    """first N samples"""
    if N > result.X.shape[0]:
        msg = f"N={N} exceeds generate N={result.X.shape[0]}"
        raise ValueError(msg)
    return result.X[:N].copy(), result.Z[:N].copy()

def subset_by_dims(result, d, restandardize=True) -> tuple[np.ndarray, SimParams]:
    """First d features (contiguous copy) + matching W, mu, psi; rescaled to config.snr."""
    if d > result.X.shape[1]:
        raise ValueError(f"d={d} exceeds generated d={result.X.shape[1]}")
    p = result.params
    params = SimParams(W=p.W[:d].copy(), mu=p.mu[:d].copy(), psi=p.psi[:d].copy())
    X = np.ascontiguousarray(result.X[:, :d])                 # (N, d)
    if not restandardize:
        return X, params
    target = adjust_snr(params, result.config.snr)           # uniform rescale of psi
    c = target.psi[0] / params.psi[0]                        # same factor for every psi_j
    signal = result.Z @ params.W.T + params.mu               # (N, d)
    return signal + np.sqrt(c) * (X - signal), target        # rescale the same noise draws

def realized_snr(params) -> float:
    """tr(WW^T)/tr(psi) of params used"""
    return float(np.sum(params.W**2)/np.sum(params.psi))