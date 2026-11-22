"""Draw Figure 3 from the historical and generated error correlation matrices."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from paths import DATA, OUTPUT
OUTPUT_DIRS=[OUTPUT]

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def wind_correlations():
    d=np.load(DATA/"error_correlations.npz")
    return d["wind_original"],d["wind_generated"]

def solar_correlations():
    d=np.load(DATA/"error_correlations.npz")
    return d["solar_original"],d["solar_generated"]

def plot_pair(
    original: np.ndarray,
    generated: np.ndarray,
    output_name: str,
    *,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    tick_labels = np.arange(0, 48, 6)
    tick_positions = tick_labels + 0.5

    heatmap = dict(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": 0.86, "pad": 0.025},
    )
    sns.heatmap(original, ax=axes[0], **heatmap)
    sns.heatmap(generated, ax=axes[1], **heatmap)

    for ax, title in zip(axes, ("Original samples", "Generated samples")):
        ax.set_title(title, fontsize=28, fontweight="bold", fontstyle="italic", pad=10)
        ax.set_xlabel("Time", fontsize=24, fontweight="bold", fontstyle="italic")
        ax.set_ylabel("Time", fontsize=24, fontweight="bold", fontstyle="italic")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, rotation=0)
        ax.tick_params(axis="both", which="major", labelsize=22, width=0.8)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        colorbar = ax.collections[0].colorbar
        colorbar.ax.tick_params(labelsize=20, width=0.8)
        plt.setp(colorbar.ax.get_yticklabels(), fontweight="bold")

    plt.tight_layout(w_pad=2.0)
    for directory in OUTPUT_DIRS:
        fig.savefig(directory / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Preserve the original figure-specific color maps and normalization.
    plot_pair(*wind_correlations(), "Figure3a.pdf", cmap="coolwarm")
    plot_pair(
        *solar_correlations(),
        "Figure3b.pdf",
        cmap="RdBu_r",
        vmin=-0.1,
        vmax=0.9,
    )


if __name__ == "__main__":
    main()
