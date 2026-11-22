# EIFICA for AGC Scheduling

Research materials accompanying **“Data Driven Distributionally Robust Scheduling of AGC Participation Factors with Cumulative State of Charge Constraints: A Constraint Geometry Guided Convex Approximation.”**

This repository provides the EIFICA optimization implementation, processed renewable forecast error data, and the figures and plotting materials used in the revised manuscript. The optimization model includes AGC adjusted generator ramp limits and affine cumulative deployed state of charge (SOC) constraints.

## Repository guide

| Material | Location |
| --- | --- |
| EIFICA and FICA optimization | [`src/eifica_optimizer.py`](src/eifica_optimizer.py) |
| Renewable trajectory generation | [`src/renewable_scenarios.py`](src/renewable_scenarios.py) |
| RTS-24 cumulative SOC example | [`examples/run_cumulative_soc.py`](examples/run_cumulative_soc.py) |
| Processed forecast and error data | [`data/processed/`](data/processed/) |
| Figure data and plotting code | [`data/figures/`](data/figures/), [`visualization/`](visualization/) |
| Manuscript figure PDFs | [`figures/manuscript/`](figures/manuscript/) |

## Software environment

The software environment is specified in [`environment.yml`](environment.yml) and uses Python 3.9.19, pandapower 2.14.6, and Gurobi 12.0.3. The numerical experiments reported in the manuscript were run on Ubuntu 22.04.5 with two 3.00 GHz Intel Xeon Platinum 8378A CPUs.

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate eifica-agc
```

The optimization example requires a valid Gurobi license. Figure generation does not require a solver license.

## Run the code

Run the optimization example or generate Figures 3–12 from the included figure data:

```bash
python examples/run_cumulative_soc.py --method Exact --n 100 --seed 0
python visualization/plot_manuscript.py
```

The figure command writes PDFs to `figures/reproduced/`; [figure notes](visualization/README.md) describe the individual sources and the preparation of Figures 1 and 2.

## Data sources

The processed data draw on the [GEFCom2012 wind forecasting benchmark](https://www.kaggle.com/competitions/GEF2012-wind-forecasting) and the Elia Open Data portals for [historical wind](https://opendata.elia.be/explore/dataset/ods031/) and [photovoltaic data](https://opendata.elia.be/explore/dataset/ods032/). The original datasets remain available from their providers under the applicable terms.

## Citation and license

Please cite the associated manuscript if you use these materials. Repository metadata are provided in [`CITATION.cff`](CITATION.cff). The code is released under the [MIT License](LICENSE).
