#!/usr/bin/env python3
"""Draw Figures 9 and 11 from the recorded generator reserve comparison."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import DATA, OUTPUT
from method_colors import METHOD_COLORS
from plot_style import apply_style, finalize_figure
from plot_figures8_10_12 import style_axis, style_legend
from reserve_data import read_results
RISKS = (
    (0.03, 0.06, "r23_118_003_006_full_01"),
    (0.05, 0.10, "r23_118_005_010_full_01"),
    (0.08, 0.12, "r23_118_full_adaptive_01"),
    (0.10, 0.15, "r23_118_010_015_full_01"),
)
STYLES = {
    "direct": ("Direct (paired)", METHOD_COLORS["Direct EIFICA"], "s"),
    "reserve": ("Reserve adaptation [31]", METHOD_COLORS["Reserve adaptation"], "^"),
}
STATUS_COLOR = "#C00000"
TARGET_COLOR = "red"
MARKER_SIZE = 7
LINE_WIDTH = 2.5


def load_grids():
    grids = []
    for epsilon, theta, run_id in RISKS:
        run_dir = DATA / "reserve" / run_id
        manifest, sizes, seeds, paired, status13 = read_results(run_dir)
        if (not np.isclose(float(manifest["epsilon"]), epsilon)
                or not np.isclose(float(manifest["theta"]), theta)):
            raise ValueError(f"Risk parameters disagree with {run_id}")
        if sizes != [50, 80, 100, 150, 200, 250, 300, 350] or len(seeds) != 5:
            raise ValueError(f"Unexpected grid in {run_id}")
        with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as fh:
            direct = {(int(row["N_WDR"]), int(row["seed"])): row
                      for row in csv.DictReader(fh) if row["case"] == "direct"}
        if (set(direct) != {(n, seed) for n in sizes for seed in seeds}
                or any(row["status"] != "OPTIMAL" for row in direct.values())):
            raise ValueError(f"Direct five-seed grid incomplete in {run_id}")
        direct_all = {n: [direct[(n, seed)] for seed in seeds] for n in sizes}
        grids.append((epsilon, theta, sizes, seeds, paired, status13, direct_all))
    return grids


def paired_values(pairs, index: int, field: str, scale: float = 1.0) -> np.ndarray:
    return np.asarray([float(pair[index][field]) * scale for pair in pairs], dtype=float)


def draw_direct_reference(ax, sizes, direct_all, field: str,
                          scale: float = 1.0) -> None:
    means = [np.mean([float(row[field]) * scale for row in direct_all[n]])
             for n in sizes]
    ax.plot(sizes, means, color=METHOD_COLORS["Direct EIFICA"],
            linestyle=(0, (2.5, 1.6)), linewidth=1.8, alpha=0.9, zorder=2)


def draw_series(ax, sizes, seeds, paired, field: str, scale: float = 1.0,
                show_band: bool = True, *, reliability: bool = False,
                compact: bool = False,
                dash_incomplete_reserve: bool = False) -> None:
    for case, (label, color, marker) in STYLES.items():
        index = 0 if case == "direct" else 1
        values = [paired_values(paired[n], index, field, scale) for n in sizes]
        means = np.asarray([v.mean() if len(v) else np.nan for v in values])
        full = np.asarray([len(v) == len(seeds) for v in values])
        line_style = (0, (5.0, 2.0)) if reliability else "-"
        if case == "reserve" and dash_incomplete_reserve:
            for j in range(len(sizes) - 1):
                if not (np.isfinite(means[j]) and np.isfinite(means[j + 1])):
                    continue
                segment_style = "-" if full[j] and full[j + 1] else (0, (4.0, 2.0))
                ax.plot(sizes[j:j + 2], means[j:j + 2], color=color,
                        linewidth=LINE_WIDTH, linestyle=segment_style)
            line_style = "None"
        ax.plot(sizes, means, color=color, marker=marker, linewidth=LINE_WIDTH,
                linestyle=line_style, markersize=6 if compact else MARKER_SIZE,
                markerfacecolor="white", markeredgewidth=1.8)
        for start in range(len(sizes)):
            if not full[start] or (start > 0 and full[start - 1]):
                continue
            stop = start + 1
            while stop < len(sizes) and full[stop]:
                stop += 1
            local_x = sizes[start:stop]
            if show_band:
                if reliability:
                    lower = [v.min() for v in values[start:stop]]
                    upper = [v.max() for v in values[start:stop]]
                else:
                    lower = [np.quantile(v, 0.05) for v in values[start:stop]]
                    upper = [np.quantile(v, 0.95) for v in values[start:stop]]
                ax.fill_between(local_x, lower, upper, color=color,
                                alpha=0.20 if reliability else 0.16,
                                linewidth=0)
        partial = [i for i, v in enumerate(values) if 0 < len(v) < len(seeds)]
        if partial:
            ax.scatter([sizes[i] for i in partial], means[partial], s=50,
                       marker=marker, facecolors="white", edgecolors=color,
                       linewidths=1.8, zorder=6)


def style_panel(ax, sizes, seeds, paired, epsilon: float, theta: float,
                ylabel: str, *, counts: bool = True) -> None:
    ax.set_title(rf"$\epsilon={epsilon:g},\ \theta={theta:g}$")
    ax.set_xlabel(r"$N$ (Training Scenarios)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sizes)
    ax.set_xticklabels([
        str(n) if not counts or len(paired[n]) == len(seeds)
        else f"{n}\n{len(paired[n])}/{len(seeds)}" for n in sizes
    ])
    style_axis(ax)


def method_handles(*, reliability: bool, reference: bool = True):
    handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=LINE_WIDTH,
               linestyle=(0, (5.0, 2.0)) if reliability else "-",
               markerfacecolor="white", markeredgewidth=1.8, label=label)
        for label, color, marker in STYLES.values()
    ]
    if reference:
        handles.append(Line2D([0], [0], color=METHOD_COLORS["Direct EIFICA"],
                              linestyle=(0, (2.5, 1.6)), linewidth=1.8,
                              label="Direct (all 5)"))
    return handles


def plot_runtime(grids, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, (epsilon, theta, sizes, seeds, paired, status13, _direct_all)) in enumerate(zip(axes.flat, grids)):
        draw_series(ax, sizes, seeds, paired, "core_time_s",
                    show_band=False, dash_incomplete_reserve=True)
        for n in sizes:
            if not paired[n]:
                continue
            ratio = np.mean([
                float(reserve["core_time_s"]) / float(direct["core_time_s"])
                for direct, reserve in paired[n]
            ])
            reserve_mean = paired_values(paired[n], 1, "core_time_s").mean()
            y = reserve_mean * (1.20 if n <= 100 else 1.13)
            ax.annotate(f"{ratio:.2f}x", (n, y), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=12, fontweight="bold", color="darkred")
            failures = status13[n]
            if failures:
                x = n + np.linspace(-5, 5, len(failures))
                ax.scatter(x, failures, marker="x", color=STATUS_COLOR,
                           linewidths=1.5, s=48, zorder=7)
        ax.set_yscale("log")
        ax.set_ylim(50, 28800 * 3)
        for seconds, label in [(28800, "8h limit"), (14400, "4h"),
                               (7200, "2h")]:
            ax.axhline(seconds, color="black", linestyle="--",
                       alpha=0.55, linewidth=2.0)
            ax.text(50, seconds * 1.1, label, fontsize=12,
                    color="black", alpha=0.75)
        style_panel(ax, sizes, seeds, paired, epsilon, theta,
                    "Time (s)" if i % 2 == 0 else "")
        handles = method_handles(reliability=False, reference=False) + [
            Line2D([0], [0], color=STYLES["reserve"][1], linewidth=LINE_WIDTH,
                   linestyle=(0, (4.0, 2.0)), label="Reserve: <5 pairs"),
            Line2D([0], [0], linestyle="None", marker="x", color=STATUS_COLOR,
                   markeredgewidth=1.5, label="Reserve: not certified")
        ]
        legend = ax.legend(handles=handles, loc="upper left",
                           bbox_to_anchor=(0.02, 0.65), framealpha=0.9)
        style_legend(legend)
    fig.tight_layout()
    finalize_figure(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(grids, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, (epsilon, theta, sizes, seeds, paired, _status13, direct_all)) in enumerate(zip(axes.flat, grids)):
        draw_direct_reference(ax, sizes, direct_all, "joint_satisfied_rate", 100.0)
        draw_series(ax, sizes, seeds, paired, "joint_satisfied_rate", 100.0,
                    reliability=True)
        target = 100 * (1 - epsilon)
        ax.axhline(target, color=TARGET_COLOR, linestyle="--", linewidth=2.0)
        ax.set_ylim(20, 102)
        style_panel(ax, sizes, seeds, paired, epsilon, theta,
                    "JCC Satisfaction (%)" if i % 2 == 0 else "")

        inset = ax.inset_axes([0.50, 0.46, 0.44, 0.30])
        inset.set_facecolor((1, 1, 1, 0.94))
        draw_direct_reference(inset, sizes, direct_all,
                              "joint_satisfied_rate", 100.0)
        draw_series(inset, sizes, seeds, paired, "joint_satisfied_rate",
                    100.0, show_band=False, reliability=True, compact=True)
        inset.axhline(target, color=TARGET_COLOR, linestyle="--",
                      linewidth=2.0, alpha=0.8)
        inset.set_xlim(75, 355)
        inset.set_ylim(97.0, 100.2)
        inset.set_xticks([80, 150, 250, 350])
        inset.set_yticks([97, 98, 99, 100])
        inset.tick_params(axis="both", labelsize=9, pad=1)
        style_axis(inset, compact=True)
        for spine in inset.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.35")
        handles = method_handles(reliability=True) + [
            Line2D([0], [0], color=TARGET_COLOR, linestyle="--",
                   linewidth=2.0, label=f"Target {target:.0f}%")
        ]
        legend = ax.legend(handles=handles, loc="lower right",
                           bbox_to_anchor=(0.98, 0.02), framealpha=0.9)
        style_legend(legend)
    fig.tight_layout()
    finalize_figure(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style(base_size=16, verbose=False)
    grids=load_grids()
    plot_runtime(grids, OUTPUT/"Figure9.pdf")
    plot_reliability(grids, OUTPUT/"Figure11.pdf")

if __name__ == "__main__":
    main()
