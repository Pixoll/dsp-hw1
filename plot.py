"""
DSP Tarea 1 – Measurement Visualisation
Generates all graphs required by the assignment rubric:
  1. Sequential times vs array size
  2. Parallel times vs threads (per size)
  3. Speedup vs threads (per size)
  4. Efficiency vs threads (per size)
  5. Effect of granularity threshold
  6. Effect of k in k-way variants
  7. Profiling metrics (perf.log)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ── output directory ──────────────────────────────────────────────────────────
OUT = Path("plots")
OUT.mkdir(exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    }
)

PALETTE = {
    "sequential_mergesort": "#2196F3",
    "sequential_k_way_mergesort_k4": "#4CAF50",
    "sequential_k_way_mergesort_k8": "#8BC34A",
    "parallel_mergesort": "#FF5722",
    "parallel_k_way_mergesort_k4": "#FF9800",
    "parallel_k_way_mergesort_k8": "#FFC107",
    "parallel_ranks_mergesort": "#9C27B0",
    "parallel_ranks_k_way_mergesort_k4": "#E91E63",
    "parallel_ranks_k_way_mergesort_k8": "#F48FB1",
}

LABELS = {
    "sequential_mergesort": "Mergesort",
    "sequential_k_way_mergesort_k4": "K-way (k=4)",
    "sequential_k_way_mergesort_k8": "K-way (k=8)",
    "parallel_mergesort": "Parallel mergesort",
    "parallel_k_way_mergesort_k4": "Parallel k-way (k=4)",
    "parallel_k_way_mergesort_k8": "Parallel k-way (k=8)",
    "parallel_ranks_mergesort": "Ranks mergesort",
    "parallel_ranks_k_way_mergesort_k4": "Ranks k-way (k=4)",
    "parallel_ranks_k_way_mergesort_k8": "Ranks k-way (k=8)",
}

N_SIZES = [2 ** 20, 2 ** 22, 2 ** 24, 2 ** 26]
N_LABELS = ["$2^{20}$", "$2^{22}$", "$2^{24}$", "$2^{26}$"]
N_MB = [n * 4 / 1e6 for n in N_SIZES]  # int32 → bytes → MB
THREADS = [1, 2, 4, 8]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & LABEL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # The 'n' column is bugged (always 4).  Detect the four size-groups by
    # finding sequential_mergesort baseline rows (no k, p, or g_threshold)
    # and assigning a group_id to every row that follows each one.
    baseline_mask = (
        (df["type"] == "sequential_mergesort")
        & df["k"].isna()
        & df["p"].isna()
        & df["g_threshold"].isna()
    )
    baseline_indices = df.index[baseline_mask].tolist()

    group_ids = pd.Series(-1, index=df.index)
    for gid, start in enumerate(baseline_indices):
        end = baseline_indices[gid + 1] if gid + 1 < len(baseline_indices) else len(df)
        group_ids.iloc[start:end] = gid

    df["size_group"] = group_ids
    df["n_actual"] = df["size_group"].map(dict(enumerate(N_SIZES)))
    df["n_label"] = df["size_group"].map(dict(enumerate(N_LABELS)))
    df["n_mb"] = df["size_group"].map(dict(enumerate(N_MB)))

    # Convenience: fill k as integer where present
    df["k_int"] = df["k"].fillna(0).astype(int)

    # Build a canonical key for each algorithm variant
    def make_key(row):
        t = row["type"]
        if "k_way" in t:
            return f"{t}_k{row['k_int']}"
        return t

    df["algo_key"] = df.apply(make_key, axis=1)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SEQUENTIAL TIMES vs SIZE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sequential_times(df: pd.DataFrame):
    seq = df[df["type"].str.startswith("sequential") & df["p"].isna()].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in [
        "sequential_mergesort",
        "sequential_k_way_mergesort_k4",
        "sequential_k_way_mergesort_k8"
    ]:
        sub = seq[seq["algo_key"] == key].sort_values("size_group")
        if sub.empty:
            continue
        ax.plot(
            sub["n_actual"], sub["t_mean"],
            marker="o", linewidth=2, markersize=7,
            color=PALETTE[key], label=LABELS[key],
            zorder=3,
        )
        ax.fill_between(
            sub["n_actual"],
            sub["t_mean"] - sub["t_stdev"],
            sub["t_mean"] + sub["t_stdev"],
            alpha=0.12, color=PALETTE[key],
        )

    # reference O(n log n) curve
    x = np.array(N_SIZES, dtype=float)
    ref = x * np.log2(x)
    ref = ref / ref[0] * seq[seq["algo_key"] == "sequential_mergesort"] \
        .sort_values("size_group")["t_mean"].iloc[0]
    ax.plot(
        x, ref, "--", color="gray", linewidth=1.2,
        alpha=0.6, label="O(n log n) ref", zorder=2
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(N_SIZES)
    ax.set_xticklabels(N_LABELS)
    ax.set_xlabel("Array size  n")
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Sequential algorithms – mean execution time vs array size")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "1_sequential_times.png", dpi=150)
    plt.close(fig)
    print("✔  1_sequential_times.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PARALLEL TIMES vs THREADS  (one subplot per size)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_parallel_times(df: pd.DataFrame, threshold: int = 1024):
    par = df[
        df["type"].str.startswith("parallel")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
    ].copy()

    par_keys = [
        "parallel_mergesort",
        "parallel_k_way_mergesort_k4",
        "parallel_k_way_mergesort_k8",
        "parallel_ranks_mergesort",
        "parallel_ranks_k_way_mergesort_k4",
        "parallel_ranks_k_way_mergesort_k8",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()

    for gid, ax in enumerate(axes):
        sub = par[par["size_group"] == gid]
        for key in par_keys:
            d = sub[sub["algo_key"] == key].sort_values("p")
            if d.empty:
                continue
            ax.plot(
                d["p"], d["t_mean"],
                marker="o", linewidth=2, markersize=6,
                color=PALETTE[key], label=LABELS[key], zorder=3,
            )
            ax.fill_between(
                d["p"],
                d["t_mean"] - d["t_stdev"],
                d["t_mean"] + d["t_stdev"],
                alpha=0.1, color=PALETTE[key],
            )
        ax.set_title(f"n = {N_LABELS[gid]}")
        ax.set_xticks(THREADS)
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Mean time (s)")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True
    )
    fig.suptitle(
        f"Parallel algorithms – mean time vs threads  (threshold={threshold})",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT / "2_parallel_times.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✔  2_parallel_times.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SPEEDUP vs THREADS  (one subplot per size)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speedup(df: pd.DataFrame, threshold: int = 1024):
    par = df[
        df["type"].str.startswith("parallel")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
    ].copy()

    par_keys = [
        "parallel_mergesort",
        "parallel_k_way_mergesort_k4",
        "parallel_k_way_mergesort_k8",
        "parallel_ranks_mergesort",
        "parallel_ranks_k_way_mergesort_k4",
        "parallel_ranks_k_way_mergesort_k8",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    axes = axes.flatten()

    for gid, ax in enumerate(axes):
        sub = par[par["size_group"] == gid]

        # ideal speedup reference
        ax.plot(
            THREADS, THREADS, "--", color="gray", linewidth=1.2,
            alpha=0.6, label="Ideal (linear)", zorder=2
        )

        for key in par_keys:
            d = sub[sub["algo_key"] == key].sort_values("p")
            if d.empty:
                continue
            # Use pre-computed 's' column when available; fall back to
            # T(p=1) / T(p) computed from the data itself.
            t1_row = d[d["p"] == 1]
            if t1_row.empty:
                continue
            t1 = t1_row["t_mean"].values[0]
            d = d.copy()
            d["speedup_calc"] = t1 / d["t_mean"]
            ax.plot(
                d["p"], d["speedup_calc"],
                marker="s", linewidth=2, markersize=6,
                color=PALETTE[key], label=LABELS[key], zorder=3,
            )

        ax.set_title(f"n = {N_LABELS[gid]}")
        ax.set_xticks(THREADS)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Speedup  S(p) = T₁ / Tₚ")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True
    )
    fig.suptitle(
        f"Speedup vs threads  (threshold={threshold})",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT / "3_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✔  3_speedup.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EFFICIENCY vs THREADS  (one subplot per size)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficiency(df: pd.DataFrame, threshold: int = 1024):
    par = df[
        df["type"].str.startswith("parallel")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
    ].copy()

    par_keys = [
        "parallel_mergesort",
        "parallel_k_way_mergesort_k4",
        "parallel_k_way_mergesort_k8",
        "parallel_ranks_mergesort",
        "parallel_ranks_k_way_mergesort_k4",
        "parallel_ranks_k_way_mergesort_k8",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    axes = axes.flatten()

    for gid, ax in enumerate(axes):
        sub = par[par["size_group"] == gid]

        ax.axhline(
            1.0, linestyle="--", color="gray", linewidth=1.2,
            alpha=0.6, label="Ideal (E=1)", zorder=2
        )

        for key in par_keys:
            d = sub[sub["algo_key"] == key].sort_values("p")
            if d.empty:
                continue
            t1_row = d[d["p"] == 1]
            if t1_row.empty:
                continue
            t1 = t1_row["t_mean"].values[0]
            d = d.copy()
            d["efficiency"] = (t1 / d["t_mean"]) / d["p"]
            ax.plot(
                d["p"], d["efficiency"],
                marker="^", linewidth=2, markersize=6,
                color=PALETTE[key], label=LABELS[key], zorder=3,
            )

        ax.set_title(f"n = {N_LABELS[gid]}")
        ax.set_xticks(THREADS)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Efficiency  E(p) = S(p) / p")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=True
    )
    fig.suptitle(
        f"Efficiency vs threads  (threshold={threshold})",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(OUT / "4_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✔  4_efficiency.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  EFFECT OF GRANULARITY THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════

def plot_threshold_effect(df: pd.DataFrame):
    par = df[df["type"].str.startswith("parallel") & df["p"].notna()].copy()

    thresholds = [1024, 4096]
    thresh_colors = {1024: "#1976D2", 4096: "#E64A19"}
    thresh_labels = {1024: "threshold=1024", 4096: "threshold=4096"}

    algo_map = {
        "parallel_mergesort": "Parallel mergesort",
        "parallel_k_way_mergesort_k4": "Parallel k-way k=4",
        "parallel_k_way_mergesort_k8": "Parallel k-way k=8",
        "parallel_ranks_mergesort": "Ranks mergesort",
    }

    # Use the largest size group (gid=3) for clearest effect
    sub = par[par["size_group"] == 3]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, algo_map.items()):
        for thr in thresholds:
            d = sub[(sub["algo_key"] == key) & (sub["g_threshold"] == thr)].sort_values("p")
            if d.empty:
                continue
            ax.plot(
                d["p"], d["t_mean"],
                marker="o", linewidth=2, markersize=6,
                color=thresh_colors[thr], label=thresh_labels[thr], zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks(THREADS)
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Mean time (s)")
        ax.legend()

    fig.suptitle(
        r"Effect of granularity threshold on mean time  (n = $2^{26}$)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "5_threshold_effect.png", dpi=150)
    plt.close(fig)
    print("✔  5_threshold_effect.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  EFFECT OF k  (k=4 vs k=8)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_k_effect(df: pd.DataFrame, threshold: int = 1024):
    kway = df[
        df["type"].str.contains("k_way")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
    ].copy()

    k_colors = {4: "#388E3C", 8: "#8BC34A"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, base_key, title in zip(
        axes,
        ["parallel_k_way_mergesort", "parallel_ranks_k_way_mergesort"],
        ["Parallel k-way mergesort", "Parallel ranks k-way mergesort"],
    ):
        for gid, ls in zip([1, 3], ["-", "--"]):  # n=2^22 and n=2^26
            for k_val in [4, 8]:
                algo_key_target = f"{base_key}_k{k_val}"
                d = kway[
                    (kway["algo_key"] == algo_key_target)
                    & (kway["size_group"] == gid)
                ].sort_values("p")
                if d.empty:
                    continue
                lbl = f"k={k_val}, n={N_LABELS[gid]}"
                ax.plot(
                    d["p"], d["t_mean"],
                    marker="o", linewidth=2, markersize=6,
                    color=k_colors[k_val], linestyle=ls,
                    label=lbl, zorder=3,
                )
        ax.set_title(title)
        ax.set_xticks(THREADS)
        ax.set_xlabel("Threads (p)")
        ax.set_ylabel("Mean time (s)")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Effect of k on k-way mergesort  (threshold={threshold})",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "6_k_effect.png", dpi=150)
    plt.close(fig)
    print("✔  6_k_effect.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7b. SEQUENTIAL vs PARALLEL – best of each family, large n
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_vs_threads(df: pd.DataFrame, threshold: int = 1024):
    """Bar chart comparing all algorithms at p=8, for the two largest sizes."""
    par = df[
        df["type"].str.startswith("parallel")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
        & (df["p"] == 8)
    ].copy()

    seq = df[
        df["type"].str.startswith("sequential")
        & df["p"].isna()
        & df["k"].isna()
    ].copy()
    seq["p"] = 1

    focus_keys = [
        "sequential_mergesort",
        "parallel_mergesort",
        "parallel_k_way_mergesort_k8",
        "parallel_ranks_mergesort",
        "parallel_ranks_k_way_mergesort_k8",
    ]
    focus_labels = [
        "Sequential\nmergesort",
        "Parallel\nmergesort\np=8",
        "Parallel k-way\nk=8, p=8",
        "Ranks\nmergesort\np=8",
        "Ranks k-way\nk=8, p=8",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, gid in zip(axes, [2, 3]):
        means, errors = [], []
        for key in focus_keys:
            if key.startswith("sequential"):
                row = seq[(seq["algo_key"] == key) & (seq["size_group"] == gid)]
            else:
                row = par[(par["algo_key"] == key) & (par["size_group"] == gid)]
            if row.empty:
                means.append(0);
                errors.append(0)
            else:
                means.append(row["t_mean"].values[0])
                errors.append(row["t_stdev"].values[0])

        xs = np.arange(len(focus_keys))
        colors = [PALETTE.get(k, "#607D8B") for k in focus_keys]
        bars = ax.bar(
            xs, means, yerr=errors, color=colors, capsize=4,
            edgecolor="white", linewidth=0.8, zorder=3
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(focus_labels, fontsize=8)
        ax.set_ylabel("Mean time (s)")
        ax.set_title(f"n = {N_LABELS[gid]}")

        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
                    f"{val:.2f}s", ha="center", va="bottom", fontsize=7.5, fontweight="bold"
                )

    fig.suptitle(
        "All algorithm families – mean time comparison",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "7_algorithm_comparison.png", dpi=150)
    plt.close(fig)
    print("✔  7_algorithm_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  PROFILING (perf.log – hardcoded from the provided file)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_profiling():
    algos = [
        "sequential\nmergesort",
        "sequential\nk-way",
        "parallel\nmergesort",
        "parallel\nk-way",
        "parallel\nranks",
        "ranks\nk-way",
    ]

    cycles = [
        27_895_979_136, 31_693_755_363, 32_372_630_577,
        45_181_561_183, 35_194_837_199, 114_916_156_209
    ]
    instructions = [
        34_187_258_071, 121_172_698_996, 33_630_841_308,
        121_310_265_239, 36_818_887_007, 182_430_427_161
    ]
    cache_refs = [
        518_223_131, 347_003_424, 547_878_197,
        413_132_313, 620_409_659, 796_439_705
    ]
    cache_miss = [
        32_335_037, 18_612_082, 29_532_340,
        30_353_457, 61_190_916, 97_674_403
    ]
    elapsed_s = [6.000, 6.817, 2.237, 2.546, 2.129, 3.900]
    ipc = [ins / cy for ins, cy in zip(instructions, cycles)]
    miss_rate = [m / r * 100 for m, r in zip(cache_miss, cache_refs)]

    bar_color = "#1565C0"
    accent = "#F57C00"
    xs = np.arange(len(algos))
    width = 0.55

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))

    # ── panel helpers ──────────────────────────────────────────────────────────
    def bar_panel(ax, values, title, ylabel, unit_scale=1, color=bar_color, fmt=".2f"):
        vals = [v / unit_scale for v in values]
        bars = ax.bar(
            xs, vals, width=width, color=color,
            edgecolor="white", linewidth=0.8, zorder=3
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(algos, fontsize=8.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=8
            )
        return ax

    # 1. cycles
    bar_panel(
        axes[0, 0], cycles, "CPU Cycles", "Cycles (× 10⁹)",
        unit_scale=1e9, color="#1565C0", fmt=".1f"
    )

    # 2. instructions
    bar_panel(
        axes[0, 1], instructions, "Instructions executed", "Instructions (× 10⁹)",
        unit_scale=1e9, color="#2E7D32", fmt=".1f"
    )

    # 3. IPC
    axes[1, 0].bar(
        xs, ipc, width=width, color="#6A1B9A",
        edgecolor="white", linewidth=0.8, zorder=3
    )
    axes[1, 0].set_xticks(xs);
    axes[1, 0].set_xticklabels(algos, fontsize=8.5)
    axes[1, 0].set_title("Instructions per Cycle (IPC)")
    axes[1, 0].set_ylabel("IPC")
    for i, v in enumerate(ipc):
        axes[1, 0].text(
            i, v + max(ipc) * 0.01, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8
        )

    # 4. cache references
    bar_panel(
        axes[1, 1], cache_refs, "Cache References", "Refs (× 10⁸)",
        unit_scale=1e8, color="#00695C", fmt=".2f"
    )

    # 5. cache misses
    bar_panel(
        axes[2, 0], cache_miss, "Cache Misses", "Misses (× 10⁶)",
        unit_scale=1e6, color="#BF360C", fmt=".1f"
    )

    # 6. cache miss rate  (bar) + elapsed time (line)
    ax6 = axes[2, 1]
    bars = ax6.bar(
        xs, miss_rate, width=width, color=accent,
        edgecolor="white", linewidth=0.8, zorder=3,
        label="Cache miss rate (%)"
    )
    for bar, v in zip(bars, miss_rate):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(miss_rate) * 0.015,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=8
        )
    ax6.set_xticks(xs);
    ax6.set_xticklabels(algos, fontsize=8.5)
    ax6.set_title("Cache miss rate  vs  Elapsed time")
    ax6.set_ylabel("Cache miss rate (%)")
    ax6_r = ax6.twinx()
    ax6_r.plot(
        xs, elapsed_s, "D--", color="#D32F2F", linewidth=2,
        markersize=7, label="Elapsed time (s)", zorder=4
    )
    ax6_r.set_ylabel("Elapsed time (s)", color="#D32F2F")
    ax6_r.tick_params(axis="y", colors="#D32F2F")
    ax6_r.spines["right"].set_visible(True)
    ax6_r.spines["right"].set_color("#D32F2F")

    lines1, lbl1 = ax6.get_legend_handles_labels()
    lines2, lbl2 = ax6_r.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=8)

    fig.suptitle(
        "Profiling – perf stat results  (n = 2²⁶, 8 threads)",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "8_profiling.png", dpi=150)
    plt.close(fig)
    print("✔  8_profiling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SPEEDUP TABLE  (text as heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_speedup_heatmap(df: pd.DataFrame, gid: int = 3, threshold: int = 1024):
    """Heatmap of speedup values: algorithms × threads for the largest size."""
    par = df[
        df["type"].str.startswith("parallel")
        & df["p"].notna()
        & (df["g_threshold"] == threshold)
        & (df["size_group"] == gid)
    ].copy()

    par_keys = [
        "parallel_mergesort",
        "parallel_k_way_mergesort_k4",
        "parallel_k_way_mergesort_k8",
        "parallel_ranks_mergesort",
        "parallel_ranks_k_way_mergesort_k4",
        "parallel_ranks_k_way_mergesort_k8",
    ]

    matrix = []
    for key in par_keys:
        d = par[par["algo_key"] == key].sort_values("p")
        t1_row = d[d["p"] == 1]
        if t1_row.empty:
            matrix.append([np.nan] * len(THREADS))
            continue
        t1 = t1_row["t_mean"].values[0]
        row_vals = []
        for p in THREADS:
            tp_row = d[d["p"] == p]
            row_vals.append(t1 / tp_row["t_mean"].values[0] if not tp_row.empty else np.nan)
        matrix.append(row_vals)

    mat = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=8)

    ax.set_xticks(range(len(THREADS)))
    ax.set_xticklabels([f"p={p}" for p in THREADS])
    ax.set_yticks(range(len(par_keys)))
    ax.set_yticklabels([LABELS[k] for k in par_keys], fontsize=9)

    for i in range(len(par_keys)):
        for j in range(len(THREADS)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="black" if v < 5 else "white"
                )

    plt.colorbar(im, ax=ax, label="Speedup S(p)")
    ax.set_title(
        f"Speedup heatmap  —  n = {N_LABELS[gid]},  threshold={threshold}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "9_speedup_heatmap.png", dpi=150)
    plt.close(fig)
    print("✔  9_speedup_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    csv_file = "data/measurements.csv"

    print(f"Loading {csv_file} …")
    df = load_data(csv_file)

    n_groups = df["size_group"].nunique()
    print(
        f"Detected {n_groups} size group(s): "
        + ", ".join(N_LABELS[:n_groups])
    )
    print(f"Saving plots to  '{OUT}/' …\n")

    plot_sequential_times(df)
    plot_parallel_times(df)
    plot_speedup(df)
    plot_efficiency(df)
    plot_threshold_effect(df)
    plot_k_effect(df)
    plot_all_vs_threads(df)
    plot_profiling()
    plot_speedup_heatmap(df)

    print(f"\n✅  All plots saved to '{OUT}/'")
