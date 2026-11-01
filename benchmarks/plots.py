"""Figures from benchmark CSVs.

    uv run --with matplotlib --with pandas python -m benchmarks.plots benchmarks/results/dendrite

Writes PNGs to <results>/figures/ and prints the numbers behind each figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

# ---- style: fixed categorical slots per entity, thin marks, recessive chrome ----
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MODEL_STYLE = {  # color follows the algorithm; dashed = sklearn implementation
    "pca": dict(color="#2a78d6", marker="o", ls="-", label="PCA"),
    "ppca_cf": dict(color="#eb6834", marker="s", ls="-", label="pPCA closed form"),
    "ppca_em": dict(color="#1baf7a", marker="^", ls="-", label="pPCA EM"),
    "fa": dict(color="#eda100", marker="D", ls="-", label="FA EM"),
    "sk_pca": dict(color="#2a78d6", marker="o", ls="--", label="sklearn PCA", mfc="white"),
    "sk_fa": dict(color="#eda100", marker="D", ls="--", label="sklearn FA", mfc="white"),
}
STAGE_GROUPS = {  # <= 8 stacked segments, fixed order (categorical slots 1..8)
    "setup (center, X2)": ["t_center", "t_X2"],
    "form S": ["t_form_S"],
    "init": ["t_init"],
    "E-step": ["t_e_step"],
    "X^T E[z]": ["t_XtEz"],
    "M-steps": ["t_m_step_W", "t_m_step_noise"],
    "log-likelihood": ["t_log_likelihood"],
    "final LL (recomputes S)": ["t_final_ll"],
}
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#b3302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "figure.dpi": 150,
})


def mean_sem(df, by, col):
    g = df.groupby(by)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out


def plot_line(ax, x, y, sem, model):
    st = MODEL_STYLE[model]
    ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
            mfc=st.get("mfc", st["color"]), label=st["label"])
    ax.fill_between(x, y - sem, y + sem, color=st["color"], alpha=0.15, lw=0)


def ref_slope(ax, x, y_anchor, slope, label):
    """Reference line y = y_anchor * (x / x0)^slope through the first point."""
    x = np.asarray(x, dtype=float)
    ax.plot(x, y_anchor * (x / x[0]) ** slope, color=INK2, ls=":", lw=1.2, zorder=0)
    ax.annotate(label, (x[-1], y_anchor * (x[-1] / x[0]) ** slope), color=INK2,
                fontsize=8, xytext=(4, 0), textcoords="offset points", va="center")


def fitted_slope(x, y) -> float:
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


# ---- Fig 1: recovery vs N ------------------------------------------------------


def fig_recovery(rec: pd.DataFrame, out: Path) -> None:
    models = ["ppca_cf", "fa", "pca"]  # pPCA EM coincides with pPCA closed form
    metrics = [("W_subspace_error", "Loading subspace error"), ("psi_error", "Noise-variance error (rel. L2)")]
    noises = [("iso", "Isotropic noise (pPCA model true)"), ("fa", "Heteroscedastic noise (FA model true)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    print("\n[recovery] fitted log-log slope vs N (theory: -0.5 when the model is correct)")
    for i, (metric, mlabel) in enumerate(metrics):
        for j, (noise, nlabel) in enumerate(noises):
            ax = axes[i, j]
            sub = rec[rec.noise == noise]
            for m in models:
                s = mean_sem(sub[sub.model == m], "N", metric)
                plot_line(ax, s.N, s["mean"], s["sem"], m)
                print(f"  {noise:3s} {metric:17s} {m:8s} slope={fitted_slope(s.N, s['mean']):+.2f}  "
                      f"values={np.round(s['mean'].to_numpy(), 4).tolist()}")
            anchor = mean_sem(sub[sub.model == "fa"], "N", metric)["mean"].iloc[0]
            ref_slope(ax, sorted(sub.N.unique()), anchor, -0.5, r"$N^{-1/2}$")
            ax.set(xscale="log", yscale="log")
            if i == 0:
                ax.set_title(nlabel)
            if j == 0:
                ax.set_ylabel(mlabel)
            if i == 1:
                ax.set_xlabel("Number of samples N")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Parameter recovery vs N  (d = 50, k = 3, SNR = 1; 10 runs, mean ± SEM)\n"
                 "PCA and pPCA EM give the same subspace as pPCA closed form", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "1_recovery_vs_N.png")
    plt.close(fig)


# ---- Fig 2: FA -> pPCA on isotropic data ----------------------------------------


def fig_fa_to_ppca(rec: pd.DataFrame, out: Path) -> None:
    sub = rec[(rec.noise == "iso") & (rec.model == "fa")]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    print("\n[FA -> pPCA] FA fitted to isotropic data: per-feature psi_j vs the true sigma^2")
    for col, label, color, marker in [
        ("psi_max_rel_dev", r"max$_j\,|\hat\psi_j/\sigma^2 - 1|$  (worst feature)", "#eda100", "D"),
        ("psi_cv", r"std$(\hat\psi)$ / mean$(\hat\psi)$  (spread across features)", "#6250d6", "o"),
    ]:
        s = mean_sem(sub, "N", col)
        ax.plot(s.N, s["mean"], color=color, marker=marker, label=label)
        ax.fill_between(s.N, s["mean"] - s["sem"], s["mean"] + s["sem"], color=color, alpha=0.15, lw=0)
        print(f"  {col:16s} slope={fitted_slope(s.N, s['mean']):+.2f} values={np.round(s['mean'].to_numpy(), 4).tolist()}")
    s0 = mean_sem(sub, "N", "psi_max_rel_dev")
    ref_slope(ax, s0.N, s0["mean"].iloc[0], -0.5, r"$N^{-1/2}$")
    ax.set(xscale="log", yscale="log", xlabel="Number of samples N", ylabel="Relative deviation")
    ax.set_title("FA → pPCA: every noise variance converges to σ² as N grows\n"
                 "(d = 50, k = 3, isotropic data; 10 runs, mean ± SEM)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "2_fa_to_ppca.png")
    plt.close(fig)


# ---- Fig 3: runtime / iterations / memory vs N, d, k ----------------------------


def fig_runtime(rt: pd.DataFrame, out: Path) -> None:
    sweeps = [("N", "Number of samples N  (d=100, k=3)"), ("d", "Observed dims d  (N=1000, k=3)"),
              ("k", "Latent dims k  (N=1000, d=100)")]
    cols = [("total_time", "Total fit time (s)"), ("n_iter", "EM iterations"),
            ("peak_mem_mb", "Peak memory (MB, tracemalloc)")]
    models = ["pca", "ppca_cf", "ppca_em", "fa", "sk_pca", "sk_fa"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    print("\n[runtime] fitted log-log slope of total time")
    for i, (sweep, xlabel) in enumerate(sweeps):
        sub = rt[rt.sweep == sweep]
        for j, (col, ylabel) in enumerate(cols):
            ax = axes[i, j]
            for m in models:
                if col == "n_iter" and m in ("pca", "ppca_cf", "sk_pca"):
                    continue  # closed form: no iterations
                s = mean_sem(sub[sub.model == m].dropna(subset=[col]), sweep, col)
                if s.empty:
                    continue
                plot_line(ax, s[sweep], s["mean"], s["sem"], m)
                if col == "total_time":
                    print(f"  vs {sweep}: {m:8s} slope={fitted_slope(s[sweep], s['mean']):+.2f}")
            xs = sorted(sub[sweep].unique())
            if col == "total_time":
                base = mean_sem(sub[sub.model == "ppca_cf"], sweep, col)["mean"].iloc[0]
                ref_slope(ax, xs, base, 1, "∝x")
                ref_slope(ax, xs, base, 2, "∝x²")
            ax.set(xscale="log", yscale="log")
            ax.set_xticks(xs, [f"{v:,}" for v in xs])  # label only the swept values
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(xlabel)
            if i == 0:
                ax.set_title(ylabel)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Runtime, EM iterations and peak memory (one-factor sweeps, SNR = 1; "
                 "10 runs for time, 2 for memory; single-threaded BLAS)\n"
                 "sklearn FA uses its own SVD-based iteration and stopping rule (tol=1e-2): "
                 "iteration counts are not comparable", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "3_runtime_scaling.png")
    plt.close(fig)


# ---- Fig 4: where the time goes ------------------------------------------------


def fig_stages(stg: pd.DataFrame, out: Path) -> None:
    models = [("ppca_cf", "pPCA closed form"), ("ppca_em", "pPCA EM"), ("fa", "FA EM")]
    sweeps = [("d", "d  (N = 10,000, k = 3)"), ("N", "N  (d = 500, k = 3)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    print("\n[stages] share of fit time (%) per stage; stage_sum/fit_time sanity in brackets")
    for i, (sweep, xlabel) in enumerate(sweeps):
        for j, (m, title) in enumerate(models):
            ax = axes[i, j]
            sub = stg[(stg.sweep == sweep) & (stg.model == m)]
            g = sub.groupby(sweep)
            t = pd.DataFrame({name: g[cs].mean().sum(axis=1) for name, cs in STAGE_GROUPS.items()})
            share = t.div(t.sum(axis=1), axis=0)
            fit_time = g["fit_time"].mean()
            ratio = g["stage_sum_over_fit"].mean()
            xpos = np.arange(len(share))
            bottom = np.zeros(len(share))
            for (name, color) in zip(STAGE_GROUPS, STAGE_COLORS):
                ax.bar(xpos, share[name], bottom=bottom, color=color, width=0.7,
                       edgecolor=SURFACE, linewidth=1.5, label=name)
                bottom += share[name].to_numpy()
            for x, ft in zip(xpos, fit_time):
                ax.annotate(f"{ft:.0f}s" if ft >= 10 else f"{ft:.2g}s", (x, 1.0), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=INK2)
            ax.set_xticks(xpos, [f"{v:,}" for v in share.index])
            ax.set_ylim(0, 1.08)
            ax.grid(axis="x", visible=False)
            ax.set_xlabel(xlabel)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel("Share of fit time")
            print(f"  {m:8s} vs {sweep}:")
            for v in share.index:
                top = share.loc[v].sort_values(ascending=False)
                top = ", ".join(f"{k} {p:.0%}" for k, p in top.items() if p >= 0.03)
                print(f"     {sweep}={v:>7,}: {top}  [fit {fit_time[v]:.3g}s, sum/fit {ratio[v]:.2f}]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Where fit() spends its time (stage timers around mini-latents' own functions; "
                 "labels = total fit time; 3 runs)", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "4_stage_breakdown.png")
    plt.close(fig)


# ---- Fig 5: closed form vs EM over (N, d) ---------------------------------------


def fig_boundary(grid: pd.DataFrame, out: Path) -> None:
    p = grid.groupby(["model", "d", "N"]).agg(t=("total_time", "mean"), it=("n_iter", "mean"),
                                              mem=("peak_mem_mb", "mean")).reset_index()
    t = p.pivot_table(index=["d", "N"], columns="model", values="t")
    ratio = (t["ppca_em"] / t["ppca_cf"]).unstack("N")  # rows d, cols N
    iters = p[p.model == "ppca_em"].pivot(index="d", columns="N", values="it")
    mem = p.pivot_table(index=["d", "N"], columns="model", values="mem")
    mem_ratio = (mem["ppca_em"] / mem["ppca_cf"]).unstack("N")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    L = np.log10(ratio.to_numpy())
    lim = max(abs(np.nanmin(L)), abs(np.nanmax(L)))
    im = ax1.imshow(L, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), origin="lower", aspect="auto")
    for (r, c), v in np.ndenumerate(ratio.to_numpy()):
        ax1.text(c, r, f"{v:.0f}×\n{iters.to_numpy()[r, c]:.0f} it", ha="center", va="center", fontsize=7.5,
                 color="white" if abs(L[r, c]) > 0.6 * lim else INK)
    ax1.set_xticks(range(ratio.shape[1]), [f"{n:,}" for n in ratio.columns])
    ax1.set_yticks(range(ratio.shape[0]), [f"{d:,}" for d in ratio.index])
    ax1.set(xlabel="N", ylabel="d")
    ax1.grid(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046)
    cb.set_label("log10( EM time / closed-form time )\n> 0: closed form faster")
    ax1.set_title("pPCA: EM time ÷ closed-form time (cell: ratio, EM iterations)", fontsize=10)

    for N in iters.columns:
        alpha = 0.35 + 0.65 * (np.log10(N) - 2) / 3
        ax2.plot(iters.index, iters[N], color="#1baf7a", alpha=alpha, marker="^", lw=1.5,
                 label=f"N = {N:,}")
    d = np.asarray(iters.index, dtype=float)
    ax2.plot(d, iters.iloc[2, -1] * d / d[2], color=INK2, ls=":", lw=1.2)
    ax2.annotate("∝ d", (d[-1], iters.iloc[2, -1] * d[-1] / d[2]), xytext=(4, 0),
                 textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax2.set(xscale="log", yscale="log", xlabel="d", ylabel="EM iterations to converge (tol 1e-8)")
    ax2.set_title("EM iterations grow ∝ d and do not depend on N", fontsize=10)
    ax2.legend(fontsize=7.5, ncol=2)
    fig.suptitle("Closed form vs EM for pPCA (k = 5, SNR = 1, isotropic data; 5 runs). "
                 f"Peak-memory ratio EM/closed form: {np.nanmin(mem_ratio.to_numpy()):.2f}–"
                 f"{np.nanmax(mem_ratio.to_numpy()):.2f} in every cell", fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "5_closed_form_vs_em.png")
    plt.close(fig)

    print("\n[boundary] EM/closed-form time ratio (rows d, cols N)")
    print(ratio.round(1).to_string())
    print(f"  min ratio {np.nanmin(ratio.to_numpy()):.1f}, max {np.nanmax(ratio.to_numpy()):.1f}; "
          f"memory ratio {np.nanmin(mem_ratio.to_numpy()):.3f}–{np.nanmax(mem_ratio.to_numpy()):.3f}")
    print("[boundary] EM iterations / d:")
    print((iters.div(iters.index.to_series(), axis=0)).round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="folder with recovery.csv, runtime.csv, stages.csv, grid.csv")
    args = ap.parse_args()
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    rec = pd.read_csv(args.results / "recovery.csv")
    rt = pd.read_csv(args.results / "runtime.csv")
    stg = pd.read_csv(args.results / "stages.csv")
    grid = pd.read_csv(args.results / "grid.csv")
    fig_recovery(rec, out)
    fig_fa_to_ppca(rec, out)
    fig_runtime(rt, out)
    fig_stages(stg, out)
    fig_boundary(grid, out)
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()