"""Draw Figures 8, 10 and 12 from the recorded IEEE 118 and ExactLHS runs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import apply_style, finalize_figure


from paths import DATA, OUTPUT
OUT_DIR=OUTPUT
CASE118_FILE=DATA/"ieee118.csv"
CVAR_FILE=DATA/"ieee118_cvar.csv"
EXACT_FILE=DATA/"exactlhs.csv"

METHODS = ["FICA", "EIFICA", "CVAR"]
RISKS = [(0.03, 0.06), (0.05, 0.10), (0.08, 0.12), (0.10, 0.15)]
N_CASE118 = [50, 80, 100, 150, 200, 250, 300, 350]
COLORS = {"FICA": "#1F77B4", "EIFICA": "#FF7F0E", "CVAR": "#2CA02C"}
MARKERS = {"FICA": "D", "EIFICA": "s", "CVAR": "o"}
LABELS = {"FICA": "FICA [32]", "EIFICA": "Direct EIFICA", "CVAR": "CVaR [36]"}


def style_axis(ax: plt.Axes, *, compact: bool = False) -> None:
    size = 9 if compact else 16
    ax.tick_params(axis="both", which="major", labelsize=size, width=0.8)
    plt.setp(ax.get_xticklabels(), fontfamily="STIXGeneral", fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontfamily="STIXGeneral", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    if compact:
        ax._paper_style_compact = True


def style_legend(legend, *, compact: bool = False) -> None:
    for item in legend.get_texts():
        item.set_fontfamily("STIXGeneral")
        item.set_fontsize(9 if compact else 12)
        item.set_fontweight("bold")
        item.set_fontstyle("italic")


def save(fig: plt.Figure, number: int) -> None:
    finalize_figure(fig)
    fig.savefig(OUT_DIR / {6:"Figure8.pdf",7:"Figure10.pdf",8:"Figure12.pdf"}[number], dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_case118() -> pd.DataFrame:
    base = pd.read_csv(CASE118_FILE)
    base = base.loc[base.status.eq("OK")].copy()
    cvar = pd.read_csv(CVAR_FILE)
    cvar = cvar.loc[
        cvar.method.eq("CVAR")
        & cvar.status.eq("OK")
        & cvar.N_WDR.isin(N_CASE118[:-1])
    ].copy()
    data = pd.concat([base, cvar], ignore_index=True, sort=False)
    data = data.drop_duplicates(
        ["method", "epsilon", "theta", "N_WDR", "seed"], keep="last"
    )
    data["satisfied_rate"] *= 100.0
    return data


def plot_runtime(data: pd.DataFrame) -> None:
    timelimit = 28800
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, (ax, (eps, theta)) in enumerate(zip(axes.ravel(), RISKS)):
        part = data.loc[data.epsilon.eq(eps) & data.theta.eq(theta)]
        for method in METHODS:
            method_data = part.loc[part.method.eq(method)]
            if method_data.empty:
                continue
            sns.lineplot(
                x="N_WDR",
                y="solve_time",
                data=method_data,
                ax=ax,
                color=COLORS[method],
                estimator="mean",
                errorbar=None,
                marker=MARKERS[method],
                markersize=7,
                markerfacecolor="white",
                markeredgecolor=COLORS[method],
                markeredgewidth=1.8,
                linewidth=2.5,
            )

        for seconds, label in [(28800, "8h limit"), (14400, "4h limit"), (7200, "2h limit")]:
            ax.axhline(seconds, color="black", linestyle="--", alpha=0.55, linewidth=2.0)
            ax.text(50, seconds * 1.1, label, fontsize=12, color="black", alpha=0.75)

        cvar_time = part.loc[part.method.eq("CVAR")].groupby("N_WDR").solve_time.mean()
        if 300 in cvar_time.index:
            y_oom = cvar_time.loc[300]
            ax.scatter([350], [y_oom], marker="x", s=90, linewidths=2.5,
                       color=COLORS["CVAR"], zorder=5)
            ax.text(350, y_oom * 1.18, "OOM", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=COLORS["CVAR"])

        means = {
            method: part.loc[part.method.eq(method)].groupby("N_WDR").solve_time.mean()
            for method in METHODS
        }
        for n_value in means["FICA"].index.intersection(means["EIFICA"].index):
            ratio = means["FICA"][n_value] / means["EIFICA"][n_value]
            if int(n_value) == 80:
                ax.text(n_value, means["FICA"][n_value] * 0.82, f"{ratio:.2f}×",
                        ha="center", va="top", fontsize=12, fontweight="bold",
                        color="darkred")
            else:
                ax.text(n_value, means["FICA"][n_value] * 1.3, f"{ratio:.2f}×",
                        ha="center", va="bottom", fontsize=12, fontweight="bold",
                        color="darkred")
        for n_value in means["CVAR"].index.intersection(means["EIFICA"].index):
            ratio = means["CVAR"][n_value] / means["EIFICA"][n_value]
            y_factor = 1.50 if int(n_value) < 100 else (1.35 if int(n_value) == 100 else 1.16)
            x_shift = {50: -3, 80: 2, 100: 4}.get(int(n_value), 0)
            ax.text(n_value + x_shift, means["CVAR"][n_value] * y_factor,
                    f"{ratio:.2f}×", ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="darkgreen")

        ax.set_title(rf"$\epsilon={eps},\ \theta={theta}$")
        ax.set_xlabel(r"$N$ (Training Scenarios)")
        ax.set_ylabel("Time (s)" if i % 2 == 0 else "")
        ax.set_xticks(N_CASE118)
        ax.set_xticklabels([str(n) for n in N_CASE118])
        ax.set_yscale("log")
        ax.set_ylim(50, timelimit * 3)
        style_axis(ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        handles = [
            Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linewidth=2.5,
                   markerfacecolor="white", markeredgewidth=1.8, label=LABELS[m])
            for m in METHODS
        ]
        legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.65),
                           framealpha=0.9)
        style_legend(legend)

    fig.tight_layout()
    save(fig, 6)


def reliability_summary(part: pd.DataFrame, method: str) -> pd.DataFrame:
    return (
        part.loc[part.method.eq(method)]
        .groupby("N_WDR").satisfied_rate.agg(["mean", "min", "max"])
        .reset_index()
    )


def plot_reliability_curve(ax: plt.Axes, part: pd.DataFrame, method: str,
                           *, band: bool, compact: bool = False) -> None:
    summary = reliability_summary(part, method)
    if summary.empty:
        return
    x = summary.N_WDR.to_numpy(float)
    if band:
        ax.fill_between(x, summary["min"], summary["max"],
                        color=COLORS[method], alpha=0.20, linewidth=0)
    dash = {"FICA": (0, (2.5, 1.6)), "EIFICA": (0, (5.0, 2.0)), "CVAR": (0, (7.0, 2.2))}
    ax.plot(x, summary["mean"], color=COLORS[method], marker=MARKERS[method],
            linestyle=dash[method], linewidth=2.5,
            markersize=(6 if compact else 7), markerfacecolor="white",
            markeredgecolor=COLORS[method], markeredgewidth=1.8)


def plot_reliability(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, (eps, theta)) in enumerate(zip(axes.ravel(), RISKS)):
        part = data.loc[data.epsilon.eq(eps) & data.theta.eq(theta)]
        for method in METHODS:
            plot_reliability_curve(ax, part, method, band=True)
        target = 100.0 * (1.0 - eps)
        ax.axhline(target, color="red", linestyle="--", linewidth=2.0)
        ax.set_title(rf"$\epsilon={eps},\ \theta={theta}$")
        ax.set_xlabel(r"$N$ (Training Scenarios)")
        ax.set_ylabel("JCC Satisfaction (%)" if i % 2 == 0 else "")
        ax.set_xticks(N_CASE118)
        ax.set_xticklabels([str(n) for n in N_CASE118])
        ax.set_ylim(20, 101)
        ax.set_yticks([20, 40, 60, 80, 100])
        style_axis(ax)

        inset = ax.inset_axes([0.50, 0.46, 0.44, 0.30])
        inset.set_facecolor((1, 1, 1, 0.94))
        zoom = part.loc[part.N_WDR.ge(80)]
        for method in METHODS:
            plot_reliability_curve(inset, zoom, method, band=False, compact=True)
        inset.axhline(target, color="red", linestyle="--", linewidth=2.0, alpha=0.8)
        inset.set_xlim(75, 355)
        inset.set_ylim(97.0, 100.2)
        inset.set_xticks([80, 150, 250, 350])
        inset.set_yticks([97, 98, 99, 100])
        inset.tick_params(axis="both", labelsize=9, pad=1)
        style_axis(inset, compact=True)
        for spine in inset.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.35")

        handles = [
            Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linewidth=2.5,
                   linestyle={"FICA": (0, (2.5, 1.6)), "EIFICA": (0, (5.0, 2.0)),
                              "CVAR": (0, (7.0, 2.2))}[m],
                   markerfacecolor="white", markeredgewidth=1.8, label=LABELS[m])
            for m in METHODS
        ]
        handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=2.0,
                              label=f"Target {target:.0f}%"))
        legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.02),
                           framealpha=0.9)
        style_legend(legend)

    fig.tight_layout()
    save(fig, 7)


def plot_exact_gap() -> None:
    data = pd.read_csv(EXACT_FILE).dropna(subset=["obj"]).drop_duplicates()
    paired = data.pivot_table(
        index=["sf", "epsilon", "theta", "n_wdr", "seed"],
        columns="method",
        values="obj",
    ).reset_index()
    paired = paired.dropna(subset=["EIFICA", "ExactLHS"])
    paired["diff_pct"] = (
        (paired.EIFICA - paired.ExactLHS) / paired.ExactLHS.abs() * 100.0
    )
    sf_values = sorted(paired.sf.unique())
    eps_values = sorted(paired.epsilon.unique())
    n_values = sorted(paired.n_wdr.unique())
    palette = ["#E68685", "#9986B1", "#F3983C", "#6DAB81", "#7EC8C8"]

    low = paired.diff_pct.min()
    high = paired.diff_pct.max()
    margin = (high - low) * 0.10
    limits = (low - margin, high + margin)
    span = limits[1] - limits[0]
    interval = 0.5 if span <= 3 else (0.75 if span <= 6 else (1.5 if span <= 12 else 3.0))

    fig, axes = plt.subplots(len(sf_values), len(eps_values),
                             figsize=(7 * len(eps_values), 2.7 * len(sf_values)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, sf in enumerate(sf_values):
        for j, eps in enumerate(eps_values):
            ax = axes[i, j]
            subset = paired.loc[paired.sf.eq(sf) & paired.epsilon.eq(eps)]
            sns.boxplot(x="n_wdr", y="diff_pct", data=subset, ax=ax,
                        palette={n: palette[k % len(palette)] for k, n in enumerate(n_values)},
                        hue="n_wdr", legend=False, order=n_values, width=0.6,
                        showfliers=False, linewidth=1.2)
            sns.stripplot(x="n_wdr", y="diff_pct", data=subset, ax=ax,
                          order=n_values, color="none", edgecolor="red", linewidth=1.0,
                          alpha=0.7, jitter=True, size=4.5)
            ax.axhline(0, color="blue", linestyle="-", linewidth=2.0)
            ax.set_ylim(limits)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
            ax.set_title(rf"$SF={sf},\ \epsilon={eps}$")
            ax.set_xlabel(r"$N$ (Sample Size)")
            ax.set_ylabel("Obj. Diff. (%)" if j == 0 else "")
            style_axis(ax)
    fig.tight_layout()
    save(fig, 8)


def main():
    apply_style(base_size=16, verbose=False)
    data=load_case118()
    plot_runtime(data)
    plot_reliability(data)
    np.random.seed(0)
    plot_exact_gap()

if __name__ == "__main__":
    main()
