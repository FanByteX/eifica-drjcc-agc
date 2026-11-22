"""Draw Figure 6 from the bundled plotting arrays."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from plot_style import (
    FIG_WIDTH,
    PALETTE,
    ROW_HEIGHT,
    apply_style,
    finalize_figure,
    style_axis,
)
from paths import DATA, OUTPUT
OUT_DIR=OUTPUT
DIAG_DIR=OUTPUT

T = 24
TSTART = 0
NUM_GEN = 38
NUM_WT = 10
NUM_SOLAR = 5
NUM_ESS = 6
N_SAMPLES_TRAIN = 1000
N_SAMPLES_TEST = 5000
SCENARIO_INDEX = 0
EPSILON = 0.08
THETA = 0.12


def set_style() -> None:
    apply_style(base_size=16, verbose=True)


def style_ax(ax, ylabel: str, title: str | None = None) -> None:
    if title:
        ax.set_title(title)
    style_axis(ax, ylabel, T=T, grid_alpha=0.3)


def style_soc_axis(ax, ylabel: str, title: str | None = None) -> None:
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, T + 0.5)
    ax.set_xticks(np.arange(T + 1))
    ax.set_xticklabels(np.arange(T + 1))
    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", labelsize=16, width=0.8)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)


def save_panel(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    fig.savefig(OUT_DIR / f"{name.replace('Figure_4', 'Figure6')}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_generators(result: dict, data: dict, xi: np.ndarray) -> np.ndarray:
    """Choose two visually informative generators for the dispatch figure.

    The paper figure should make ramp feasibility visible, so we rank by actual
    ramp activity under the selected scenario while requiring the displayed
    actual ramp to stay within the unit ramp limit.
    """
    scores = []
    gen_power = result["gen_power_all"]
    gen_alpha = result["gen_alpha_all"]
    for g in range(gen_power.shape[1]):
        actual = gen_power[:, g] - gen_alpha[:, g] * xi
        actual_ramp = np.diff(actual)
        limit = float(data["gen_ramp"][g])
        if np.max(np.abs(actual_ramp)) <= limit + 1e-8:
            activity = float(np.sum(np.abs(actual_ramp)))
            nonzero_hours = int(np.sum(np.abs(actual_ramp) > 0.5))
            alpha_mean = float(np.mean(np.abs(gen_alpha[:, g])))
            scores.append((activity, nonzero_hours, alpha_mean, g))
    scores.sort(reverse=True)
    return np.array([scores[0][3], scores[1][3]], dtype=int)


def actual_soc_paths(
    ess_soc: np.ndarray,
    ess_beta: np.ndarray,
    xi: np.ndarray,
    ess_capacity: np.ndarray,
) -> np.ndarray:
    actual = np.array(ess_soc, copy=True)
    for e in range(ess_beta.shape[1]):
        shift = np.zeros(T + 1)
        shift[1:] = np.cumsum(ess_beta[:, e] * xi / ess_capacity[e])
        actual[:, e] = ess_soc[:, e] - shift
    return actual


def plot_figure_4a(result: dict, data: dict, gens: np.ndarray, xi: np.ndarray) -> None:
    x = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(FIG_WIDTH, ROW_HEIGHT * 4), sharex=True)

    wt_fore = data["wt_pred"].sum(axis=1)
    wt_actual = wt_fore + data["wt_error_test"][SCENARIO_INDEX].sum(axis=1)
    axes[0].step(x, wt_fore, where="post", linewidth=2.5, color=PALETTE["plan"], label="forecast")
    axes[0].step(x, wt_actual, where="post", linewidth=2.5, color=PALETTE["actual"], label="actual")
    style_ax(axes[0], "Wind (MW)", "Wind Power")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.43, 0.0), ncol=2, framealpha=0.85)

    solar_fore = data["solar_pred"].sum(axis=1)
    solar_actual = solar_fore + data["solar_error_test"][SCENARIO_INDEX].sum(axis=1)
    axes[1].step(x, solar_fore, where="post", linewidth=2.5, color=PALETTE["plan"], label="forecast")
    axes[1].step(x, solar_actual, where="post", linewidth=2.5, color=PALETTE["actual"], label="actual")
    style_ax(axes[1], "Solar (MW)", "Solar Power")
    axes[1].legend(loc="upper left", ncol=1, framealpha=0.85)

    for row, gen in enumerate([int(gens[0]), int(gens[1])], start=2):
        gen_power = result["gen_power_all"][:, gen]
        gen_alpha = result["gen_alpha_all"][:, gen]
        gen_actual = gen_power - gen_alpha * xi
        axes[row].step(
            x,
            gen_power,
            where="post",
            linewidth=2.5,
            color=PALETTE["plan"],
            label="first-stage",
        )
        axes[row].step(
            x,
            gen_actual,
            where="post",
            linewidth=2.5,
            color=PALETTE["actual"],
            label="actual",
        )
        axes[row].axhline(data["gen_cap"][gen], color=PALETTE["bound"], linestyle="--", linewidth=2.0)
        axes[row].axhline(data["gen_pmin"][gen], color=PALETTE["bound"], linestyle="--", linewidth=2.0)
        ax2 = axes[row].twinx()
        ax2.bar(
            x + 0.5,
            gen_alpha,
            width=0.8,
            color=PALETTE["agc"],
            alpha=0.35,
            label=r"$\alpha$",
        )
        ax2.set_ylabel(r"AGC factor $\alpha$")
        ax2.set_ylim(0, max(0.12, float(gen_alpha.max()) * 1.35))
        style_ax(axes[row], "Gen. output (MW)", f"Representative Generator {gen}")
        h1, l1 = axes[row].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axes[row].legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.85)

    fig.suptitle(
        "Representative Cumulative SOC Dispatch",
        y=0.978,
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.045, hspace=0.35)
    save_panel(fig, "Figure_4a")


def plot_figure_4b(result: dict, data: dict, gens: np.ndarray, xi: np.ndarray) -> None:
    x = np.arange(1, T) + 0.5
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, ROW_HEIGHT * 2), sharex=True)
    width = 0.35

    for ax, gen in zip(axes, gens):
        gen = int(gen)
        scheduled = result["gen_power_all"][:, gen]
        actual = scheduled - result["gen_alpha_all"][:, gen] * xi
        ramp_sched = np.diff(scheduled)
        ramp_actual = np.diff(actual)
        limit = float(data["gen_ramp"][gen])
        ax.bar(x - width / 2, ramp_sched, width=width, color=PALETTE["plan"], alpha=0.7, label="First-stage")
        ax.bar(x + width / 2, ramp_actual, width=width, color=PALETTE["actual"], alpha=0.7, label="Actual")
        ax.axhline(limit, color=PALETTE["limit"], linestyle="--", linewidth=2.0, label=f"Limit (±{limit:.1f} MW)")
        ax.axhline(-limit, color=PALETTE["limit"], linestyle="--", linewidth=2.0)
        ax.set_ylim(-1.2 * limit, 1.2 * limit)
        style_ax(ax, "Ramp (MW/h)", f"Generator {gen} Ramping")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.85))

    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.35)
    save_panel(fig, "Figure_4b")


def plot_figure_4c(result: dict, data: dict, xi: np.ndarray) -> None:
    x = np.arange(T)
    xsoc = np.arange(T + 1)
    ess_beta = result["ess_beta_all"]
    ess_plan = result["ess_plan_all"]
    ess_soc = result["ess_soc"]
    ess_actual_soc = actual_soc_paths(ess_soc, ess_beta, xi, data["ess_capacity"])
    colors = plt.cm.tab10(np.linspace(0, 0.9, NUM_ESS))

    fig = plt.figure(figsize=(FIG_WIDTH, ROW_HEIGHT * 4.55))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.0, 1.0, 1.55, 1.15],
        hspace=0.42,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]

    width = 0.72 / NUM_ESS
    for e in range(NUM_ESS):
        offset = (e - (NUM_ESS - 1) / 2) * width
        axes[0].bar(x + offset, ess_beta[:, e], width=width, color=colors[e], alpha=0.88, label=f"ESS {e}")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylim(-0.12, max(0.30, float(np.max(ess_beta)) * 1.25))
    style_ax(axes[0], r"AGC factor $\beta$", "ESS AGC Participation")
    axes[0].legend(ncol=3, loc="upper left", framealpha=0.85)

    for e in range(NUM_ESS):
        axes[1].step(x, ess_plan[:, e], where="post", color=colors[e], lw=2.5, label=f"ESS {e}")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    style_ax(axes[1], "Plan (MW)", "ESS Scheduled Output (+ discharge, - charge)")

    for e in range(NUM_ESS):
        axes[2].plot(
            xsoc,
            ess_soc[:, e],
            color=colors[e],
            lw=2.0,
            ls=":",
            alpha=0.95,
        )
        axes[2].plot(
            xsoc,
            ess_actual_soc[:, e],
            color=colors[e],
            lw=2.5,
            marker="o",
            ms=6.0,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
    axes[2].axhline(0.9, color=PALETTE["limit"], linestyle="--", linewidth=2.0, label="SOC max")
    axes[2].axhline(0.1, color="#1F77B4", ls="--", lw=2.0, label="SOC min")
    axes[2].set_ylim(0.05, 0.95)
    style_soc_axis(axes[2], "SOC", "Actual Cumulative SOC Paths")
    soc_legend = [
        Line2D([0], [0], color="0.2", lw=2.0, ls=":", label="planned SOC"),
        Line2D([0], [0], color=PALETTE["limit"], lw=2.0, ls="--", label="SOC max"),
        Line2D([0], [0], color="0.2", lw=2.5, marker="o", ms=6.0, markerfacecolor="white", label="actual SOC"),
        Line2D([0], [0], color="#1F77B4", lw=2.0, ls="--", label="SOC min"),
    ]
    axes[2].legend(
        handles=soc_legend,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.90),
        framealpha=0.88,
        fontsize=12,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=0.5,
    )

    axins = inset_axes(
        axes[2],
        width="25%",
        height="30%",
        loc="upper center",
        borderpad=1.35,
    )
    axins._paper_style_compact = True
    for e in range(NUM_ESS):
        axins.plot(xsoc, ess_soc[:, e], color=colors[e], lw=1.5, ls=":", alpha=0.78)
        axins.plot(xsoc, ess_actual_soc[:, e], color=colors[e], lw=2.0, marker="o", ms=3.0)
    axins.set_xlim(19, 21)
    axins.set_ylim(0.42, 0.70)
    axins.set_xticks([19, 20, 21])
    axins.set_yticks([0.45, 0.55, 0.65])
    axins.grid(True, linestyle="--", alpha=0.28)
    axins.tick_params(labelsize=9)
    mark_inset(axes[2], axins, loc1=3, loc2=4, fc="none", ec="0.35", lw=0.8)

    for e in range(NUM_ESS):
        actual_output = ess_plan[:, e] + ess_beta[:, e] * xi
        axes[3].step(x, actual_output, where="post", color=colors[e], lw=2.5, label=f"ESS {e}")
    axes[3].axhline(0, color="black", lw=0.8, ls="--")
    style_ax(axes[3], "Actual (MW)", "ESS Actual Regulation Output")

    plt.subplots_adjust(left=0.08, right=0.95, top=0.98, bottom=0.045)
    save_panel(fig, "Figure_4c")


def load_panel_data(filename):
    with np.load(DATA / filename) as a:
        result={k[7:]:a[k] for k in a.files if k.startswith("result_")}
        data={k[5:]:a[k] for k in a.files if k.startswith("data_")}
        return result, data, a["generators"], a["xi"]


def main():
    set_style()
    result,data,gens,xi=load_panel_data("trajectory_6.npz")
    plot_figure_4a(result,data,gens,xi)
    plot_figure_4b(result,data,gens,xi)
    plot_figure_4c(result,data,xi)

if __name__ == "__main__":
    main()
