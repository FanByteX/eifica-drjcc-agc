# Figure reproduction

The scripts generate numbered figure PDFs from the bundled plotting data.

Run from the repository root:

```bash
python visualization/plot_manuscript.py
```

The command reads `data/figures/` and writes Figures 3–12 to `figures/reproduced/`.
To draw selected figures, use `--figures 4,5,6`.

| Figure | Source | Data |
|---|---|---|
| 1 | `figure1.tex` | Diagram from the final manuscript, including resolved equation numbers |
| 2 | `artwork/Figure2.ai`, `plot_topology.py` | IEEE 118 bus network tables and resource buses |
| 3a, 3b | `plot_figure3.py` | `error_correlations.npz` |
| 4 | `plot_figure4.py` | `rts24_soc.csv`, `rts24_initialization.csv` |
| 5, 7 | `plot_figures5_7.py` | `rts24_soc.csv`, `rts24_no_storage.csv` |
| 6a–6c | `plot_figure6.py` | `trajectory_6.npz` |
| 8, 10 | `plot_figures8_10_12.py` | `ieee118.csv`, `ieee118_cvar.csv` |
| 9, 11 | `plot_figures9_11.py` | `reserve/*/summary.csv` and grid settings |
| 12 | `plot_figures8_10_12.py` | `exactlhs.csv` |

Compile Figure 1 with XeLaTeX:

```bash
xelatex -output-directory=figures/reproduced visualization/figure1.tex
```

Figure 2 was arranged in Illustrator. Its editable source is provided. `plot_topology.py` draws the network and resource placement before manual layout.

The Figure 12 jitter seed is fixed at 0. The individual values and box statistics are unchanged; horizontal scatter positions may differ from the manuscript export.

See [data notes](../data/figures/README.md) for array definitions and trajectory provenance.
