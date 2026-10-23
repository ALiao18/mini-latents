"""
Benchmark configuratoin: sweep values, defaults, models, seeds
"""

from dataclasses import dataclass

_STREAM_DATA = 0 # W, mu, psi, Z, eps
_STREAM_FIT  = 1 # init for pPCA EM and FA

_SWEEP_NAMES = ("N", "d", "k")

SUBSETTING_SWEEPS = {"N", "d"}
STRUCTURAL_SWEEPS = {"k"} # fresh ground truth per value
MODELS = ("pca", "ppca_cf", "ppca_em", "fa")

@dataclass(frozen=True)
class FitSettings:
    tol: float=1e-8
    max_iter: int = 50000
    min_var_frac: float = 0.01

FIT = FitSettings()

@dataclass(frozen=True)
class SweepConfig:
    default: int | float
    values: tuple[int, ...] | tuple[float, ...] | None = None

    @property
    def is_swept(self) -> bool:
        return self.values is not None

    @property
    def max_value(self) -> int | float:
        return self.default if self.values is None else max(self.values)

@dataclass(frozen=True)
class BenchmarkConfig:
    N: SweepConfig
    d: SweepConfig
    k: SweepConfig
    snr: float=1.0
    n_runs: int =10
    base_seed: int=42

    def get_sweep_config(self, sweep_name: str) -> SweepConfig:
        return getattr(self, sweep_name) if sweep_name in _SWEEP_NAMES else self._bad(sweep_name)

    def get_active_sweeps(self) -> list[str]:
        return [s for s in _SWEEP_NAMES if self.get_sweep_config(s).is_swept]

    def get_data_seed(
            self, sweep_name: str, run_idx: int, value_idx: int | None = None
    ) -> list[int]:
        """
        Subsetting sweeps (N, d): omit value_idx, all values in a run share one ground truth
        """
        if sweep_name in STRUCTURAL_SWEEPS and value_idx is None:
            msg = f"structural sweep '{sweep_name}' needs value_idx"
            raise ValueError(msg)
        parts = [self._sweep_idx(sweep_name), run_idx]
        if value_idx is not None:
            parts.append(value_idx)
        return [*parts, _STREAM_DATA, self.base_seed]

    def get_fit_seed(self, sweep_name: str, run_idx: int, value_idx: int) -> list[int]:
        return [self._sweep_idx(sweep_name), run_idx, value_idx, _STREAM_FIT, self.base_seed]

    def _sweep_idx(self, sweep_name: str) -> int:
        if sweep_name not in _SWEEP_NAMES:
            self._bad(sweep_name)
        return _SWEEP_NAMES.index(sweep_name)

    @staticmethod
    def _bad(sweep_name: str):
        msg = f"Unknown sweep '{sweep_name}'. Valid: {list(_SWEEP_NAMES)}"
        raise ValueError(msg)

@dataclass(frozen=True)
class GridConfig:
    """
    full factorial (N x d) grid for closed-form vs EM boundary
    Ground truth is generated once per run at (max N, max d), and sliced both ways
    Cells with k >= min(N, d) are skipped
    """

    N_values: tuple[int, ...]
    d_values: tuple[int, ...]
    k: int
    snr: float=1.0
    n_runs: int =5
    base_seed: int=42

    @property
    def max_N(self) -> int:
        return max(self.N_values)
    
    @property
    def max_d(self) -> int:
        return max(self.d_values)

    def get_data_seed(self, run_idx: int) -> list[int]:
        return [run_idx, _STREAM_DATA, self.base_seed]

    def get_fit_seed(self, run_idx: int, N_idx: int, d_idx: int) -> list[int]:
        return [run_idx, N_idx, d_idx, _STREAM_FIT, self.base_seed]
    

RECOVERY_CONFIG = BenchmarkConfig(
    N=SweepConfig(default=1_000, values=(100, 1_000, 10_000, 100_000)),
    d=SweepConfig(default=50),
    k=SweepConfig(default=3),
)
"""Recovery vs N only (latents' values). Run on isotropic and FA data."""

RUNTIME_CONFIG = BenchmarkConfig(
    N=SweepConfig(default=1_000, values=(100, 1_000, 10_000, 100_000)),
    d=SweepConfig(default=100, values=(10, 30, 100, 300, 1_000)),
    k=SweepConfig(default=3, values=(2, 5, 10, 20, 50)),
)
"""Runtime/memory vs N, d, k. k sweep runs at the default d=100."""

BOUNDARY_CONFIG = GridConfig(
    N_values=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
    d_values=(10, 30, 100, 300, 1_000, 3_000),
    k=5,
)

STAGE_CONFIG = BenchmarkConfig(
    N=SweepConfig(default=10_000, values=(1_000, 10_000, 100_000)),
    d=SweepConfig(default=500, values=(30, 100, 300, 1_000, 3_000)),
    k=SweepConfig(default=3),
    n_runs=3,
)
