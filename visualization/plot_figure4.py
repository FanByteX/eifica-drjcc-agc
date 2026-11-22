#!/usr/bin/env python3
"""Replot manuscript Figure 4 as three matched, fixed-risk comparison panels.

The original FICA and Direct records and the paired Exact initialized/empty
reruns share RTS-24, (epsilon, theta)=(0.08, 0.12), N, and five seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scaling_data import (
    COLORS, DEFAULT_ABLATION, DEFAULT_ORIGINAL, LABELS, MARKERS,
    METHODS, ROOT, SIZES, load_data,
)


from paths import OUTPUT
OUTPUTS = (OUTPUT / "Figure4.pdf",)

def draw(ax: plt.Axes, x, y, method: str) -> None:
    ax.plot(x, y, color=COLORS[method], marker=MARKERS[method], linewidth=2.5,
            markersize=6.5, markerfacecolor="white", markeredgewidth=1.5,
            label=LABELS[method])


def style(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"$N$ (Training Scenarios)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(SIZES)
    ax.grid(True, linestyle="--", alpha=0.32, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=13, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, nargs="+", default=OUTPUTS)
    args = parser.parse_args()
    data = load_data(args.original, args.ablation)

    sys.path.insert(0, str(ROOT))
    from plot_style import apply_style, finalize_figure

    apply_style(base_size=14, verbose=False)
    fig, (time_ax, speed_ax, jcc_ax) = plt.subplots(1, 3, figsize=(13.0, 3.8))
    grouped = data.groupby(["N", "method"], as_index=False).agg(
        time=("time_s", "mean"), time_std=("time_s", "std"), jcc=("jcc", "mean"),
    )
    for method in METHODS:
        part = grouped.loc[grouped["method"].eq(method)].sort_values("N")
        x = part["N"].to_numpy(dtype=float)
        mean = part["time"].to_numpy(dtype=float)
        std = part["time_std"].to_numpy(dtype=float)
        draw(time_ax, x, mean, method)
        time_ax.fill_between(x, np.maximum(1e-3, mean - std), mean + std,
                             color=COLORS[method], alpha=0.14, linewidth=0)
        draw(jcc_ax, x, 100 * part["jcc"].to_numpy(dtype=float), method)

    paired = data.pivot(index=["N", "seed"], columns="method", values="time_s")
    for method in METHODS[1:]:
        ratio = (paired["FICA"] / paired[method]).groupby(level="N").agg(["mean", "std"])
        ratio = ratio.reindex(SIZES)
        x = np.asarray(SIZES, dtype=float)
        mean = ratio["mean"].to_numpy(dtype=float)
        std = ratio["std"].to_numpy(dtype=float)
        draw(speed_ax, x, mean, method)
        speed_ax.fill_between(x, np.maximum(0, mean - std), mean + std,
                              color=COLORS[method], alpha=0.14, linewidth=0)
        for n, value in zip(SIZES, mean):
            if method == "EIFICA_ONESHOT":
                offset, valign = 8, "bottom"
            elif method == "EIFICA_EXACT":
                offset, valign = (3, "bottom") if n <= 200 else (-9, "top")
            else:
                offset, valign = (-10, "top") if n <= 200 else (8, "bottom")
            speed_ax.annotate(f"{value:.2f}x", (n, value), xytext=(0, offset),
                              textcoords="offset points", ha="center", va=valign,
                              color=COLORS[method], fontsize=11.5, fontweight="bold")
    speed_ax.axhline(1, color="#555555", linestyle="--", linewidth=1.5)
    speed_ax.set_ylim(-0.04, 2.62)

    time_ax.set_yscale("log")
    style(time_ax, "Solve time (s; log scale)")
    style(speed_ax, "Speedup over FICA")
    style(jcc_ax, "Out-of-sample JCC (%)")
    handles, labels = time_ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                        ncol=4, frameon=True, framealpha=1.0,
                        facecolor="white", edgecolor="#C8C8C8",
                        fontsize=11, columnspacing=1.8,
                        handlelength=2.0, handletextpad=0.5)
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.3)

    # Resolve satisfaction differences hidden by the main JCC scale.
    difference_ax = jcc_ax.inset_axes([0.44, 0.13, 0.49, 0.40])
    difference_ax._paper_style_compact = True
    difference_ax.set_facecolor("white")
    fica_jcc = grouped.loc[grouped["method"].eq("FICA")].set_index("N")["jcc"]
    for method in METHODS[1:]:
        part = grouped.loc[
            grouped["method"].eq(method) & grouped["N"].ge(300)
        ].sort_values("N")
        difference_pp = 100 * (part["jcc"] - part["N"].map(fica_jcc))
        difference_ax.plot(part["N"], difference_pp, color=COLORS[method],
                           marker=MARKERS[method], linewidth=1.4, markersize=4,
                           markerfacecolor="white", markeredgewidth=1)
    difference_ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    difference_ax.set_xlim(285, 615)
    difference_ax.set_ylim(-0.105, 0.105)
    difference_ax.set_xticks((300, 400, 500, 600))
    difference_ax.set_yticks((-0.1, 0.0, 0.1))
    difference_ax.tick_params(axis="both", labelsize=7, pad=1)
    difference_ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.5)
    difference_ax.set_title("Difference vs FICA (pp)", fontsize=8, pad=2)

    finalize_figure(fig)
    for label in legend.get_texts():
        label.set_fontfamily("STIXGeneral")
        label.set_fontweight("bold")
        label.set_fontstyle("italic")
    for ax in (time_ax, speed_ax, jcc_ax):
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis="both", labelsize=12)
    for output in args.output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
