# Figure data

The CSV files contain the recorded runs used by the plotting scripts. Runtime, objective, satisfaction, seed, and solver-status fields are preserved. `sources.json` records the source files and SHA-256 checksums.

- `error_correlations.npz`: four 48 × 48 matrices, keyed by `wind_original`, `wind_generated`, `solar_original`, and `solar_generated`, computed from the historical and generated error samples.
- `trajectory_6a.npz` and `trajectory_6bc.npz`: scheduled generation, AGC factors, planned storage output and SOC, renewable forecasts and errors, capacities, selected generator indices, and aggregate error `xi`. `result_` prefixes identify saved dispatch arrays and `data_` prefixes identify system/scenario inputs. Time is the first axis except for the single retained error trajectory, whose first axis has length 1. Generator indices are zero based.
- `ieee118_*.csv` and `ieee118_resources.json`: buses, coordinates, lines, transformers, and resource connections. Resource bus numbers in JSON are one based.

## Figure 6

`trajectory_6a.npz` contains the plotting inputs for final Figure 6a. `trajectory_6bc.npz` contains the plotting inputs for final Figures 6b and 6c. The script reads each panel's inputs to reproduce the corresponding manuscript illustration. Source records are listed in `sources.json`.

All figure data are extracted from the recorded results used for the final manuscript illustrations.
