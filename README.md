# EIFICA

**Distributionally robust automatic generation control (AGC) scheduling with cumulative state of charge constraints**

Companion repository for *“Data Driven Distributionally Robust Scheduling of AGC Participation Factors with Cumulative State of Charge Constraints: A Constraint Geometry Guided Convex Approximation.”*

Envelope informed fast inner convex approximation (EIFICA) extends the fast inner convex approximation (FICA) computational framework to AGC adjusted generator ramps and affine cumulative deployed state of charge (SOC). Temporal envelopes initialize reduced ramp and SOC constraint sets. Exact EIFICA iteratively adds violated constraints until omitted temporal constraint residuals satisfy the prescribed tolerance.

[![Constraint geometry and EIFICA workflow](figures/method_overview.png)](figures/manuscript/Figure1.pdf)

*Constraint geometry and solution workflow from Figure 1 of the manuscript. Select the diagram to view the full PDF.*

## Quick start

```bash
conda env create -f environment.yml
conda activate eifica-agc
python examples/run_cumulative_soc.py --method Exact --n 100 --seed 0
python visualization/plot_manuscript.py
```

Run these commands from the repository root. The optimization example requires a valid Gurobi license. The plotting command runs without a solver license and writes Figures 3–12 to `figures/reproduced/`; the [figure guide](visualization/README.md) covers the individual data sources and Figures 1 and 2.

## Repository contents

- **Optimization:** [`src/eifica_optimizer.py`](src/eifica_optimizer.py) and the [RTS-24 cumulative SOC example](examples/run_cumulative_soc.py)
- **Renewable trajectories:** [`src/renewable_scenarios.py`](src/renewable_scenarios.py) and [processed forecast and error data](data/processed/)
- **Figures:** [plotting data](data/figures/), [scripts](visualization/), and [manuscript PDFs](figures/manuscript/)

## Reference environment

The computations reported in the manuscript used **Ubuntu 22.04.5** and **two 3.00 GHz Intel Xeon Platinum 8378A processors**. The environment specifies **Python 3.9.19**, **Gurobi 12.0.3**, and **pandapower 2.14.6**. Full package versions are pinned in [`environment.yml`](environment.yml).

## Data provenance

The processed renewable data draw on the [GEFCom2012 wind forecasting benchmark](https://www.kaggle.com/competitions/GEF2012-wind-forecasting) and the Elia Open Data portals for [historical wind](https://opendata.elia.be/explore/dataset/ods031/) and [photovoltaic data](https://opendata.elia.be/explore/dataset/ods032/). Original datasets remain available from their providers under the applicable terms.

## Citation and license

Please cite the associated manuscript when using these materials. Repository metadata are in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE).
