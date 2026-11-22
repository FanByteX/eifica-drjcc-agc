from pathlib import Path
import numpy as np
import pandas as pd
from method_colors import METHOD_COLORS
from paths import DATA, OUTPUT, ROOT
DEFAULT_ORIGINAL=DATA/"rts24_soc.csv"
DEFAULT_ABLATION=DATA/"rts24_initialization.csv"
SIZES = (100, 200, 300, 400, 500, 600)
SEEDS = (0, 10000, 20000, 30000, 40000)
METHODS = ("FICA", "EIFICA_EXACT", "EIFICA_ONESHOT", "EMPTY_EXACT")
LABELS = {
    "FICA": "FICA [32]",
    "EIFICA_EXACT": "Exact EIFICA",
    "EIFICA_ONESHOT": "Direct EIFICA",
    "EMPTY_EXACT": "Exact EIFICA (empty init.)",
}
COLORS = {
    "FICA": METHOD_COLORS["FICA"],
    "EIFICA_EXACT": METHOD_COLORS["Exact EIFICA"],
    "EIFICA_ONESHOT": METHOD_COLORS["Direct EIFICA"],
    "EMPTY_EXACT": METHOD_COLORS["Exact EIFICA (empty init.)"],
}
MARKERS = {"FICA": "D", "EIFICA_EXACT": "^", "EIFICA_ONESHOT": "s", "EMPTY_EXACT": "o"}


def load_data(original_path: Path, ablation_path: Path) -> pd.DataFrame:
    original = pd.read_csv(original_path)
    original = original.loc[
        np.isclose(original["epsilon"], 0.08)
        & np.isclose(original["theta"], 0.12)
        & original["N_WDR"].isin(SIZES)
        & original["method_label"].isin(("FICA", "EIFICA_ONESHOT"))
    ].copy()
    if not original["status"].eq("OK").all():
        raise ValueError("Original FICA/Direct grid includes incomplete runs")
    original = original[["N_WDR", "seed", "method_label", "solve_time", "satisfied_rate"]].rename(columns={
        "N_WDR": "N", "method_label": "method", "solve_time": "time_s",
        "satisfied_rate": "jcc",
    })

    ablation = pd.read_csv(ablation_path)
    if not (ablation["status"].eq(2).all()
            and ablation["complete"].astype(str).str.lower().eq("true").all()
            and ablation["recovered_full_sample"].astype(str).str.lower().eq("true").all()):
        raise ValueError("Ablation grid includes incomplete or nonoptimal runs")
    ablation["method"] = ablation["variant"].map({
        "envelope": "EIFICA_EXACT", "empty": "EMPTY_EXACT",
    })
    ablation = ablation.rename(columns={
        "solve_time_s": "time_s", "joint_reliability": "jcc",
    })[["N", "seed", "method", "time_s", "jcc"]]
    data = pd.concat((original, ablation), ignore_index=True)
    expected = {(n, seed, method) for n in SIZES for seed in SEEDS for method in METHODS}
    actual = set(zip(data["N"], data["seed"], data["method"]))
    if len(data) != len(expected) or actual != expected:
        raise ValueError("The fixed-risk four-method grid is incomplete or duplicated")
    if (data["time_s"] <= 0).any():
        raise ValueError("Nonpositive solve time")
    return data
