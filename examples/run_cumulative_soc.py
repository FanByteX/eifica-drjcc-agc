"""Run one RTS-24 cumulative deployed SOC case from the manuscript setup.

Run each method in a separate process so their timing and memory use do not
affect one another. The numerical inputs are the processed libraries in
``data/processed``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eifica_optimizer import solve_PD_instance


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("FICA", "Direct", "Exact"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.method == "Direct":
        os.environ["EIFICA_MAX_ITER"] = "0"
    else:
        os.environ.pop("EIFICA_MAX_ITER", None)

    inputs = {
        path.name: sha256(path)
        for path in sorted((ROOT / "data" / "processed").glob("*.npy"))
    }
    configuration = {
        "network_name": "case24_ieee_rts",
        "num_gen": 38,
        "num_WT": 10,
        "num_Solar": 5,
        "num_ESS": 6,
        "T": 24,
        "method": "FICA" if args.method == "FICA" else "EIFICA",
        "N_WDR": args.n,
        "epsilon": 0.08,
        "theta": 0.12,
        "seed": args.seed,
        "ESS_power_ratio": 0.1,
        "ESS_capacity_ratio": 4.0,
        "ESS_eta_c": 0.95,
        "ESS_eta_d": 0.95,
        "ESS_SOC_init": 0.5,
        "ESS_SOC_min": 0.1,
        "ESS_SOC_max": 0.9,
        "ESS_c_charge": 50.0,
        "ESS_c_discharge": 50.0,
        "ESS_lambda_AGC": 0.0,
        "ESS_terminal_soc": "minimum",
        "enforce_cumulative_soc": True,
        "soc_constraint_generation": args.method != "FICA",
        "initialization_mode": "envelope",
        "error_scale": 1.0,
        "show_plot": False,
        "thread": 6,
        "time_limit": 3600,
        "MIPGap": 0.001,
    }

    start = time.monotonic()
    try:
        result = solve_PD_instance(**configuration)
        iterations = result.get("iteration_log", [])
        summary = {
            "variant": args.method,
            "status": result.get("status"),
            "objective": result.get("obj_value"),
            "solve_time_s": result.get("solve_time"),
            "joint_satisfaction": result.get("satisfied_rate"),
            "master_solves": len(iterations) if iterations else 1,
            "last_residual_check": {
                key: iterations[-1].get(key)
                for key in ("max_rho", "violated_count", "soc_max_violation", "soc_violated_count", "any_added")
            } if iterations else None,
            "retained_soc_paths": (
                args.n if args.method == "FICA" else
                len(result["initial_soc_indices"]) if args.method == "Direct" else
                len(result["soc_key_set"])
            ),
            "cumulative_soc_constraint_count": result.get("cumulative_soc_constr_count"),
        }
    except Exception as exc:
        summary = {"variant": args.method, "error": repr(exc)}
    summary.update({
        "wall_time_s": time.monotonic() - start,
        "configuration": configuration,
        "input_sha256": inputs,
        "optimizer_sha256": sha256(ROOT / "src" / "eifica_optimizer.py"),
        "host": platform.node(),
        "python": sys.version.split()[0],
    })
    encoded = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print("CUMULATIVE_SOC_RESULT " + json.dumps(summary), flush=True)
    return 0 if summary.get("status") == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
