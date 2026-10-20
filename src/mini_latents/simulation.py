from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimConfig:
    n_samples: int
    y_dim: int
    x_dim: int
    snr: float = 1.0
    isotropic: bool = False  # True: pPCA noise, False: FA noise
    random_seed: int | None = None


@dataclass
class ObsParams:
    C: np.ndarray  # (y_dim, x_dim) loadings
    d: np.ndarray  # (y_dim,) mean
    phi: np.ndarray  # (y_dim,) noise PRECISIONS, psi = 1 / phi


@dataclass
class SimulationResult:
    config: SimConfig
    obs_params: ObsParams
    latents: np.ndarray  # (n_samples, x_dim)
    observations: np.ndarray  # (n_samples, y_dim)


def sample_obs_params(config, rng) -> ObsParams:
    raise NotImplementedError


def adjust_snr(obs_params, snr) -> ObsParams:
    raise NotImplementedError


def sample_latents(x_dim, n_samples, rng) -> np.ndarray:
    raise NotImplementedError


def sample_observations(latents, obs_params, rng) -> np.ndarray:
    raise NotImplementedError


def simulate(config) -> SimulationResult:
    raise NotImplementedError
