"""Draw Figures 5 and 7 from the recorded RTS-24 runs."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import apply_style, finalize_figure
from paths import DATA, OUTPUT
from method_colors import METHOD_COLORS
FIG_DIR=OUTPUT
TABLE_DIR=OUTPUT
METHODS = ["FICA", "EIFICA_EXACT", "EIFICA_ONESHOT"]
LABELS = {
    "FICA": "FICA",
    "EIFICA_EXACT": "Exact EIFICA",
    "EIFICA_ONESHOT": "Direct EIFICA",
}
COLORS = {
    "FICA": "#1F77B4",
    "EIFICA_EXACT": "#FF7F0E",
    "EIFICA_ONESHOT": "#2CA02C",
}
MARKERS = {"FICA": "D", "EIFICA_EXACT": "^", "EIFICA_ONESHOT": "s"}
RISK_COLORS = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]


def load_ok(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[df["status"].eq("OK")].copy()

def style_axis(ax: plt.Axes, *, closed: bool = False) -> None:
    ax.grid(True, linestyle="--", alpha=0.32, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=16, width=0.8)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(closed)
    if closed:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

def line(ax: plt.Axes, x, y, method: str, label: str | None = None) -> None:
    ax.plot(
        x,
        y,
        color=COLORS[method],
        marker=MARKERS[method],
        linewidth=2.5,
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=label or LABELS[method],
    )

def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    if name == "Figure_5":
        # This narrower source figure is enlarged more in the manuscript.
        # Compensate so its axis typography matches the surrounding figures.
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            ax.tick_params(axis="both", which="major", labelsize=10)
    if name == "Figure_8":
        # This plot is now Figure 5 in a single manuscript column.
        for ax in fig.axes:
            ax.tick_params(axis="both", which="major", labelsize=9)
            for label in (ax.xaxis.label, ax.yaxis.label):
                label.set_fontsize(9)
            legend = ax.get_legend()
            if legend is not None:
                for legend_text in legend.get_texts():
                    legend_text.set_fontsize(9)
    fig.savefig(FIG_DIR / {"Figure_5": "Figure7.pdf", "Figure_8": "Figure5.pdf"}[name], dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_storage_value(ess: pd.DataFrame, no_ess: pd.DataFrame) -> None:
    keys = ["epsilon", "theta", "N_WDR", "seed", "method_label"]
    e = ess.loc[ess.method_label.eq("EIFICA_EXACT")].copy()
    n = no_ess.loc[no_ess.method_label.eq("EIFICA_EXACT")].copy()
    paired = e.merge(n, on=keys, suffixes=("_ess", "_noess"), validate="one_to_one")
    paired["saving_usd"] = paired["obj_value_noess"] - paired["obj_value_ess"]
    paired["saving_pct"] = 100.0 * paired["saving_usd"] / paired["obj_value_noess"]
    paired["jcc_gain_pp"] = 100.0 * (
        paired["satisfied_rate_ess"] - paired["satisfied_rate_noess"]
    )
    paired.to_csv(TABLE_DIR / "case24_storage_value_paired.csv", index=False)

    risks = paired[["epsilon", "theta"]].drop_duplicates().sort_values(["epsilon", "theta"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for color, row in zip(RISK_COLORS, risks.itertuples(index=False)):
        part = paired.loc[
            paired.epsilon.eq(row.epsilon) & paired.theta.eq(row.theta)
        ]
        agg = part.groupby("N_WDR").agg(
            saving=("saving_usd", "mean"),
            saving_std=("saving_usd", "std"),
            jcc=("jcc_gain_pp", "mean"),
            jcc_std=("jcc_gain_pp", "std"),
        )
        label = rf"$(\epsilon,\theta)=({row.epsilon:.2f},{row.theta:.2f})$"
        axes[0].plot(
            agg.index,
            agg.saving,
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=label,
        )
        axes[0].fill_between(
            agg.index,
            agg.saving - agg.saving_std,
            agg.saving + agg.saving_std,
            color=color,
            alpha=0.20,
        )
        axes[1].plot(
            agg.index,
            agg.jcc,
            color=color,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=label,
        )
        axes[1].fill_between(
            agg.index,
            agg.jcc - agg.jcc_std,
            agg.jcc + agg.jcc_std,
            color=color,
            alpha=0.20,
        )

    axes[0].set_xlabel(r"$N$ (Training Scenarios)")
    axes[0].set_ylabel("Operating-cost reduction (USD/day)")
    axes[1].set_xlabel(r"$N$ (Training Scenarios)")
    axes[1].set_ylabel("JCC improvement (percentage points)")
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=2.0)
    for ax in axes:
        ax.set_xticks((200, 400, 600))
    axes[0].set_yticks((2500, 2600, 2700, 2800))
    axes[1].set_yticks((0, 1))
    for ax in axes:
        style_axis(ax, closed=True)
    handles, labels = axes[0].get_legend_handles_labels()
    for subset, center in ((slice(0, 2), 0.30), (slice(2, 4), 0.78)):
        legend = fig.legend(
            handles[subset],
            labels[subset],
            loc="upper center",
            bbox_to_anchor=(center, 0.90),
            ncol=2,
            frameon=True,
            columnspacing=0.7,
            handlelength=1.2,
            handletextpad=0.3,
        )
        for text in legend.get_texts():
            text.set_fontfamily("STIXGeneral")
            text.set_fontsize(10)
            text.set_fontweight("bold")
            text.set_fontstyle("italic")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "Figure_5")

def plot_constraint_reduction(ess: pd.DataFrame) -> None:
    exact = ess.loc[ess.method_label.eq("EIFICA_EXACT")]
    fica = ess.loc[ess.method_label.eq("FICA")]
    e = exact.groupby("N_WDR").agg(
        constraints=("cumulative_soc_constr_count", "mean"),
        paths=("soc_paths", "mean"),
        iterations=("iterations", "mean"),
    )
    f = fica.groupby("N_WDR").agg(
        constraints=("cumulative_soc_constr_count", "mean")
    )
    out = e.join(f, lsuffix="_exact", rsuffix="_fica")
    removed_count = out.constraints_fica - out.constraints_exact
    if (removed_count < 0).any():
        raise ValueError("Exact EIFICA active SOC block exceeds the full block")
    out["retained_pct"] = 100.0 * out.constraints_exact / out.constraints_fica
    out["removed_pct"] = 100.0 - out.retained_pct
    out.to_csv(TABLE_DIR / "case24_constraint_reduction.csv")

    fig, ax = plt.subplots(figsize=(4.7, 3.3))
    bar_width = 56
    ax.bar(
        out.index,
        out.constraints_exact / 1000.0,
        color=COLORS["EIFICA_EXACT"],
        width=bar_width,
        edgecolor="#8A3A00",
        linewidth=0.8,
        label="Retained",
    )
    ax.bar(
        out.index,
        removed_count / 1000.0,
        bottom=out.constraints_exact / 1000.0,
        color="#D9D9D9",
        width=bar_width,
        edgecolor="#666666",
        linewidth=0.8,
        label="Not retained",
    )
    for n_value, retained, removed, share in zip(
        out.index, out.constraints_exact, removed_count, out.removed_pct
    ):
        ax.text(
            n_value,
            (retained + 0.55 * removed) / 1000.0,
            f"{share:.1f}%",
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )
    ax.set_xlabel(r"$N$ (Training Scenarios)")
    ax.set_ylabel("SOC inequalities (thousands)")
    ax.set_xlim(45, 655)
    ax.set_ylim(0, out.constraints_fica.max() / 1000.0 * 1.08)
    ax.set_xticks(out.index)
    ax.set_yticks((0, 50, 100, 150))
    style_axis(ax, closed=True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#C8C8C8",
        columnspacing=0.7,
        handletextpad=0.4,
    )
    fig.tight_layout()
    save(fig, "Figure_8")

def main():
    apply_style(base_size=16, verbose=False)
    COLORS["EIFICA_EXACT"] = METHOD_COLORS["Exact EIFICA"]
    ess=load_ok(DATA/"rts24_soc.csv")
    plot_constraint_reduction(ess)
    plot_storage_value(ess,load_ok(DATA/"rts24_no_storage.csv"))

if __name__ == "__main__":
    main()
