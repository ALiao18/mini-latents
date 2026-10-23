# Adapted from latents/benchmarks/system_info.py
# Copyright (c) 2024 Evren Gokcen, MIT License (https://github.com/egokcen/latents)
# Changes: mini_latents instead of latents; no psutil dependency; records BLAS
# backend and thread settings, which decide whether runtime numbers are comparable.
"""System information collection for benchmark reproducibility."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import scipy
import sklearn

_BYTES_PER_GB = 1024**3
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def get_system_info() -> dict:
    """Collect hardware, software, OS, and BLAS/threading information."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "hardware": _get_hardware_info(),
        "software": _get_software_info(),
        "os": _get_os_info(),
        "blas": _get_blas_info(),
    }


def _get_hardware_info() -> dict:
    return {
        "cpu": _get_cpu_model(),
        "cpu_cores_logical": os.cpu_count(),
        "ram_gb": _get_ram_gb(),
    }


def _get_cpu_model() -> str:
    """On Linux read /proc/cpuinfo (platform.processor() often gives only the arch)."""
    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "Unknown"


def _get_ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _BYTES_PER_GB, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _get_software_info() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def _get_os_info() -> dict:
    info = {"system": platform.system(), "release": platform.release()}
    try:
        info["distro"] = platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        pass
    return info


def _get_blas_info() -> dict:
    """Thread env vars plus what the BLAS libraries actually loaded with."""
    info: dict = {"env": {v: os.environ.get(v) for v in _THREAD_VARS}}
    try:
        from threadpoolctl import threadpool_info  # installed with scikit-learn

        info["threadpools"] = [
            {k: p.get(k) for k in ("internal_api", "num_threads", "version", "filepath")}
            for p in threadpool_info()
        ]
    except ImportError:
        info["threadpools"] = None
    return info


def save_system_info(path: str | os.PathLike[str]) -> None:
    with open(path, "w") as f:
        json.dump(get_system_info(), f, indent=2)
        f.write("\n")


def load_system_info(path: str | os.PathLike[str]) -> dict:
    with open(path) as f:
        return json.load(f)