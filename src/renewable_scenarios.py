"""Selected renewable scenario utilities used by the manuscript experiments.

The module assembles wind and solar forecasts and error trajectories from the
processed libraries distributed with this repository. It contains no dispatch
or optimization implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _assemble(
    forecast_file: str,
    error_file: str,
    units: int,
    samples: int,
    data_dir: Path,
    error_rescale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if units <= 0 or samples <= 0:
        raise ValueError("units and samples must be positive")

    forecast = np.load(data_dir / forecast_file).reshape(1, -1, 1)
    forecast = np.tile(forecast, (1, 1, units))

    error_library = np.load(data_dir / error_file)
    required = samples * units
    if error_library.shape[0] < required:
        raise ValueError(
            f"{error_file} contains {error_library.shape[0]} paths, "
            f"but {required} are required"
        )

    errors = error_library[:required].reshape(samples, units, -1)
    errors = np.transpose(errors, (0, 2, 1))
    if errors.shape[1] != forecast.shape[1]:
        raise ValueError("forecast and error libraries have different horizons")

    if error_rescale:
        errors = errors * np.std(forecast)

    full = np.maximum(forecast + errors, 0.0)
    errors = full - forecast

    scale = np.max(full)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scenario normalization factor must be positive")

    forecast = forecast / scale
    errors = errors / scale
    full = full / scale

    if not np.allclose(full, forecast + errors):
        raise RuntimeError("scenario identity check failed")
    return forecast[0], errors, full


def wind_scenarios(
    units: int,
    samples: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized wind forecast, errors, and complete trajectories."""
    return _assemble(
        "GEFC2012_wind_forecast.npy",
        "GEFC2012_temporal_correlated_error_samples.npy",
        units,
        samples,
        Path(data_dir),
    )


def solar_scenarios(
    units: int,
    samples: int,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized solar forecast, errors, and complete trajectories."""
    return _assemble(
        "solar_forecast_prototype.npy",
        "solar_temporal_correlated_error_samples.npy",
        units,
        samples,
        Path(data_dir),
        error_rescale=True,
    )


def dispatch_window(
    forecast: np.ndarray,
    errors: np.ndarray,
    full: np.ndarray,
    start_hour: int = 0,
    horizon: int = 24,
    solar: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the scheduling window and apply the 06:00 to 18:00 PV mask."""
    stop = start_hour + horizon
    selected_forecast = forecast[start_hour:stop].copy()
    selected_errors = errors[:, start_hour:stop].copy()
    selected_full = full[:, start_hour:stop].copy()

    if selected_forecast.shape[0] != horizon:
        raise ValueError("requested window exceeds the processed library horizon")

    if solar:
        hours = (start_hour + np.arange(horizon)) % 24
        night = (hours < 6) | (hours >= 18)
        selected_forecast[night] = 0.0
        selected_errors[:, night] = 0.0
        selected_full[:, night] = 0.0

    return selected_forecast, selected_errors, selected_full
