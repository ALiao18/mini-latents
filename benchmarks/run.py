"""Entry point: run benchmarks in parallel single-threaded processes.

    uv run python -m benchmarks.run all --workers 32 --out benchmarks/results
    uv run python -m benchmarks.run grid --workers 30
    uv run python -m benchmarks.run all --quick          # tiny smoke test, ~1 min

Each task writes independent CSV (one row per fit) under <out>/<benchmark>/parts/;
existing parts are skipped, benchmark results are concatenated into <out>/<benchmark>.csv.
"""

import os

# BLAS threads must be pinned BEFORE numpy is imported (here and in every worker,
# which inherits the environment). One thread per process, parallelism across
# processes: timings stay comparable and the 128 cores are actually used.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from benchmarks import config as C  # noqa: E402
from benchmarks import recovery, runtime, stages  # noqa: E402
from benchmarks.system_info import save_system_info  # noqa: E402

QUICK = {  # tiny configs for a smoke test
    "recovery": replace(C.RECOVERY_CONFIG, N=C.SweepConfig(200, (100, 400)), d=C.SweepConfig(10), n_runs=2),
    "runtime": replace(C.RUNTIME_CONFIG, N=C.SweepConfig(200, (100, 400)),
                       d=C.SweepConfig(20, (10, 20)), k=C.SweepConfig(2, (2, 4)), n_runs=1),
    "grid": replace(C.BOUNDARY_CONFIG, N_values=(100, 400), d_values=(10, 30), k=3, n_runs=1),
    "stages": replace(C.STAGE_CONFIG, N=C.SweepConfig(400, (200, 400)),
                      d=C.SweepConfig(20, (10, 20)), n_runs=1),
}


def _benchmarks(quick: bool) -> dict:
    """name -> (list of task args, task function, config)."""
    cfg = QUICK if quick else {
        "recovery": C.RECOVERY_CONFIG, "runtime": C.RUNTIME_CONFIG,
        "grid": C.BOUNDARY_CONFIG, "stages": C.STAGE_CONFIG,
    }
    return {
        "recovery": (recovery.tasks(cfg["recovery"]), recovery.run_task, cfg["recovery"]),
        "runtime": (runtime.sweep_tasks(cfg["runtime"]), runtime.run_sweep_task, cfg["runtime"]),
        "grid": (runtime.grid_tasks(cfg["grid"]), runtime.run_grid_task, cfg["grid"]),
        "stages": (stages.tasks(cfg["stages"]), stages.run_task, cfg["stages"]),
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)  # atomic: a killed run never leaves a half-written part


def _run_one(fn, args, cfg, part_path: str) -> tuple[str, float, str | None]:
    """Worker: run one task, write its part file. Errors are returned, not raised."""
    t0 = time.perf_counter()
    try:
        rows = fn(*args, config=cfg)
        _write_csv(rows, Path(part_path))
        return part_path, time.perf_counter() - t0, None
    except Exception:  # keep the pool alive; report at the end
        return part_path, time.perf_counter() - t0, traceback.format_exc()


def _concat(parts_dir: Path, out_path: Path) -> int:
    rows = []
    for p in sorted(parts_dir.glob("*.csv")):
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    _write_csv(rows, out_path)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["recovery", "runtime", "grid", "stages", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--quick", action="store_true", help="tiny configs (smoke test)")
    args = ap.parse_args()

    out = args.out / "quick" if args.quick else args.out
    out.mkdir(parents=True, exist_ok=True)
    save_system_info(out / "system_info.json")

    benches = _benchmarks(args.quick)
    # Slowest first, so the long tail starts early.
    order = ["grid", "runtime", "stages", "recovery"]
    names = order if args.benchmark == "all" else [args.benchmark]

    jobs = []
    for name in names:
        task_args, fn, cfg = benches[name]
        parts = out / name / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for a in task_args:
            part = parts / ("_".join(map(str, a)) + ".csv")
            if not part.exists():
                jobs.append((name, fn, a, cfg, str(part)))

    print(f"{len(jobs)} tasks to run ({args.workers} workers) -> {out}", flush=True)
    failures = []
    ctx = mp.get_context("spawn")  # clean BLAS state in every worker
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(_run_one, fn, a, cfg, part): name for name, fn, a, cfg, part in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            part, secs, err = fut.result()
            status = "FAILED" if err else "ok"
            print(f"[{i}/{len(jobs)}] {futs[fut]:9s} {Path(part).name:24s} {secs:8.1f}s {status}", flush=True)
            if err:
                failures.append((part, err))

    for name in names:
        n = _concat(out / name / "parts", out / f"{name}.csv")
        print(f"{name}: {n} rows -> {out / f'{name}.csv'}", flush=True)
    for part, err in failures:
        print(f"\n--- FAILED {part}\n{err}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()