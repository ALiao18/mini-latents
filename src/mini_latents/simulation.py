"""
simulate data from pPCA / FA generative model

z_n ~ N(0, I_k)
x_n = W z_n + mu + eps_n
eps_n ~ N(0, Psi)
Psi = diag(psi)

X (N, d), Z (N, k), W (d, k).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

@dataclass(frozen = True)
class SimConfig:
    N: int                      # samples
    d: int                      # observed dimensions
    k: int                      # latent dimensions
    snr: float = 1.0            # tr(WW ^T)/tr(psi)
    isotropic: bool = False     # True: pPCA noise, False: FA noise
    random_seed: int | Sequence[int] | None = None

@dataclass
class SimParams:
    W: np.ndarray    # (d, k)
    mu: np.ndarray   # (d, )
    psi: np.ndarray # (d, ) noise variances

@dataclass
class SimulationResult:
    config: SimConfig
    params: SimParams
    Z: np.ndarray   # (N, k)
    X: np.ndarray   # (N, d)

def sample_params(d: int, k: int, isotropic: bool, rng: np.random.Generator) -> SimParams:
    """
    W_ij ~ N(0, 1), mu_j ~ N(0, 1)
    psi: 
        isotropic -> all ones
        anisotropic -> psi_j ~ Uniform(0.5, 1.5)
    scale of psi is arbitrarily set by adjust_snr

    set it as milder than latents GFA, which uses phi_j ~ Gamma(1, 1)
    Gamma(1, 1) psi_j = 1/phi_j has infinite mean
    """
    W = rng.standard_normal((d, k)) # (d, k)
    mu = rng.standard_normal(d) # (d, )
    if isotropic:
        psi = np.ones(d) # (d, )
    else:
        psi = rng.uniform(0.5, 1.5, size=d) # (d, )
    return SimParams(W=W, mu=mu, psi=psi)

def adjust_snr(params: SimParams, snr: float) -> SimParams:
    """
    Rescale psi so that tr(W W^T) / tr(Psi) = snr
    W and mu are constant, so doesn't break isotropic
    """
    if snr <= 0:
        msg = f"snr must be positive, got {snr}"
        raise ValueError(msg)
    signal_var = np.sum(params.W**2) # tr(WW^T)
    noise_var = np.sum(params.psi) # tr(Psi)
    psi = params.psi * (signal_var / snr) / noise_var # (d,)
    return SimParams(W=params.W.copy(), mu=params.mu.copy(), psi=psi)

def sample_latents(N: int, k: int, rng: np.random.Generator) -> np.ndarray:                  
    """
    Z (N, k), z_n ~ N(0, I_k)
    """
    return rng.standard_normal((N, k))

def sample_observations(Z: np.ndarray, params: SimParams, rng: np.random.Generator) -> np.ndarray:
    """
    X (N, d) = Z W^T + mu + eps, eps_nj ~ N(0, psi_j). 
    """
    N = Z.shape[0] 
    d = params.W.shape[0]
    eps = rng.standard_normal((N, d)) * np.sqrt(params.psi) # (N, d)
    return Z @ params.W.T + params.mu + eps

def simulate(config: SimConfig) -> SimulationResult:                          
    """
    Sample ground-truth parameters, latents and observations
    """
    rng = np.random.default_rng(config.random_seed)
    params = sample_params(config.d, config.k, config.isotropic, rng)
    params = adjust_snr(params, config.snr)
    Z = sample_latents(config.N, config.k, rng)
    X = sample_observations(Z, params, rng)
    return SimulationResult(config=config, params=params, Z=Z, X=X)
