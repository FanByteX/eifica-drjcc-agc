# ESS DRJCC AGC

Storage-aware DRJCC AGC scheduling experiments under wind and solar uncertainty.

Code for
"A Constraint Family Based Inner Convex Approximation for Data-Driven Distributionally Robust AGC Scheduling with Energy Storage under Coupled Wind and Solar Uncertainty".

All experimental data in this repository are publicly available. The main implementation file `Ess.py` will be released upon formal publication of the paper. If you need research access prior to publication, please feel free to open an issue or leave a message.

All paths below are relative to the repository root unless stated otherwise.

## What Is Here

- optimization code for ESS and wind+solar scheduling
- batch experiment scripts for `case24` and `case118`
- notebook-based result visualization
- reduced optimality benchmark outputs
- input data and generated figures

## Main Entry Points

- `run_case24_ess_experiment.py`  
  Main batch experiment for case24 ESS.

- `run_case118_ess_experiment.py`  
  Main batch experiment for case118 ESS.

- `run_case24_ess_optimality.py`  
  Reduced optimality benchmark for case24 ESS.

- `run_case24_wt_solar_experiment.py`  
  Wind+solar batch experiment.

- `run_case24_wt_solar_optimality.py`  
  Wind+solar reduced optimality benchmark.

- `Ess_plot.py`  
  Representative dispatch plotting.

## Environment

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate eifica
```

Python in the original `fica` environment:

- Python `3.9.19`
- server platform: `Ubuntu 22.04.5`
- CPU: `Intel Xeon Platinum 8378A @ 3.00 GHz`

## Quick Start

From the repository root:

```powershell
conda env create -f environment.yml
conda activate eifica
```

For a quick first dispatch run, start with `Ess.py`.

`Ess.py` currently supports four methods:

- `EIFICA`
- `FICA`
- `CVAR`
- `ExactLHS`

Before running it, edit the method and core parameters in `Ess.py`, then run:

```powershell
python Ess_plot.py
```

If you want a fast first result, it is usually better to keep `N_WDR` below `100`.

After that, run the batch experiments if needed:

```powershell
python run_case24_ess_experiment.py
python run_case118_ess_experiment.py
```

Open the main visualization notebooks:

- `case_study_ess_results/case24/case24_ess_result_vis.ipynb`
- `case_study_ess_results/case118/case118_result_vis.ipynb`

Common output locations:

- case24 ESS figures: `figure/case24_ess/`
- case118 ESS figures: `figure/case118_ess/`
- case24 ESS optimality outputs: `case24_ess_optimality_results/`
- raw input data: `data/raw/`
- processed scenario data: `data/processed/`

Notes for reproduction:

- run commands from the repository root
- a working Gurobi license is required for optimization runs
- some experiments use long time limits and may take hours to finish
- for the closest reproducibility across machines, use the same server model (`Intel Xeon Platinum 8378A @ 3.00 GHz`) and a matching software environment

## Experiment Setup

The experiment settings currently aligned with the manuscript are:

- candidate training trajectories: `1000`
- out-of-sample test trajectories: `5000`
- case24 training sample sizes: `N = 50, 80, 100, 150, 200, 250, 300`
- case118 training sample sizes: `N = 50, 80, 100, 150, 200, 250, 300, 350`
- risk pairs: `(0.03, 0.06)`, `(0.05, 0.10)`, `(0.08, 0.12)`, `(0.10, 0.15)`
- repeated seeds per configuration: `5`
- case24 time limit: `4 hours`
- case24 Gurobi threads: `6`
- case118 time limit: `8 hours`
- case118 Gurobi threads: `4`

## Common Commands

```powershell
python run_case24_ess_experiment.py
python run_case118_ess_experiment.py
python run_case24_ess_optimality.py
python run_case24_wt_solar_experiment.py
python run_case24_wt_solar_optimality.py
python Ess_plot.py
```

## Important Folders

- `case_study_ess_results/`  
  Case-specific experiment outputs and notebooks.

- `case24_ess_optimality_results/`  
  Case24 reduced optimality CSV and figures.

- `case24_wt_solar_results/`  
  Wind+solar experiment outputs.

- `case24_wt_solar_optimality_results/`  
  Wind+solar reduced optimality outputs.

- `figure/`  
  Consolidated figure outputs.

- `data/`  
  Input and processed data.

## Visualization

- `case_study_ess_results/case24/case24_ess_result_vis.ipynb`
- `case_study_ess_results/case118/case118_result_vis.ipynb`

Current figure output conventions:

- case24 ESS figures: `figure/case24_ess/`
- case118 ESS figures: `figure/case118_ess/`

## Notes

- `__pycache__/` is ignored via `.gitignore`.
- The repository currently includes generated results and large `.npy` files.
- If the repo keeps growing, Git LFS is worth considering for large binary data.
