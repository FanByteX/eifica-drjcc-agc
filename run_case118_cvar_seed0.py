"""
Single-seed legacy-data timing rerun for the case118 experiment.

Purpose
-------
Run CVaR, EIFICA, and FICA on the original case118 data snapshot used by
the April case118 experiments, using the same one-seed, no outer parallelism
protocol. Execute this script from a workspace whose data/ directory points
to the legacy data snapshot.

Default grid
------------
method      : CVAR, EIFICA, FICA
seed        : 0
risk pairs  : (0.03, 0.06), (0.05, 0.10), (0.08, 0.12), (0.10, 0.15)
N_WDR       : 50, 80, 100, 150, 200, 250, 300
total runs  : 84

Outputs
-------
case_study_ess_results/case118_legacy_seed0/
  progress_case118_legacy_seed0.csv
  legacy_seed0_exp.log
  result_<stem>.npy
  case118_seed0_legacy_solve_time_comparison.csv

Usage
-----
  python run_case118_cvar_seed0.py --dry-run
  python run_case118_cvar_seed0.py
"""

import argparse
import os
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


NUM_ESS = 6
METHODS = ["CVAR", "EIFICA", "FICA"]
SEED_LIST = [0]
EPS_THETA_LIST = [
    (0.03, 0.06),
    (0.05, 0.10),
    (0.08, 0.12),
    (0.10, 0.15),
]
# Kept aligned with the CVaR supplement; N=350 exhausted the 503 GiB machine
# during the full-constraint CVaR solve.
N_WDR_LIST = [50, 80, 100, 150, 200, 250, 300]
NUM_GEN_118 = 38

DEFAULT_GUROBI_THREADS = 8
DEFAULT_N_PARALLEL = 1

RESULT_DIR = os.path.join(os.getcwd(), "case_study_ess_results", "case118_legacy_seed0")
CSV_FILE = os.path.join(RESULT_DIR, "progress_case118_legacy_seed0.csv")
LOG_FILE = os.path.join(RESULT_DIR, "legacy_seed0_exp.log")
COMPARISON_CSV = os.path.join(RESULT_DIR, "case118_seed0_legacy_solve_time_comparison.csv")
LOCAL_GUROBI_LICENSE = os.path.join(os.getcwd(), "gurobi.lic")
if os.path.exists(LOCAL_GUROBI_LICENSE):
    os.environ["GRB_LICENSE_FILE"] = LOCAL_GUROBI_LICENSE

FIXED = dict(
    num_gen=NUM_GEN_118,
    num_WT=10,
    num_Solar=5,
    T=24,
    norm_ord=1,
    show_plot=False,
    time_limit=28800,
    MIPGap=0.001,
    load_scaling_factor=1.0,
    network_name="case118",
    thread=DEFAULT_GUROBI_THREADS,
    num_ESS=NUM_ESS,
    error_scale=1.0,
    ESS_power_ratio=0.1,
    ESS_eta_c=0.95,
    ESS_eta_d=0.95,
    ESS_SOC_init=0.5,
    ESS_SOC_min=0.1,
    ESS_SOC_max=0.9,
    ESS_c_charge=5.0,
    ESS_c_discharge=5.0,
    ESS_lambda_AGC=10.0,
)


def log(msg):
    os.makedirs(RESULT_DIR, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_stem(method, eps, theta, n_wdr, seed):
    return (
        f"case118_theta{theta}_epsilon{eps}"
        f"_gurobi_seed{seed}_num_gen{NUM_GEN_118}"
        f"_N_WDR{n_wdr}_load_scaling_factor1"
        f"_{method}_T24_num_Solar5_num_WT10_num_ESS{NUM_ESS}"
    )


def _read_proc_status_kib():
    values = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    key, rest = line.split(":", 1)
                    values[key] = float(rest.strip().split()[0])
    except OSError:
        pass
    return values


def _read_mem_available_kib():
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:")):
                    key, rest = line.split(":", 1)
                    values[key] = float(rest.strip().split()[0])
    except OSError:
        pass
    return values


def start_memory_monitor(interval_s):
    stats = {
        "peak_rss_gib": 0.0,
        "peak_hwm_gib": 0.0,
        "peak_vmsize_gib": 0.0,
        "min_mem_available_gib": None,
        "samples": 0,
    }
    stop_event = threading.Event()

    def sample_once():
        proc = _read_proc_status_kib()
        mem = _read_mem_available_kib()
        if proc:
            stats["peak_rss_gib"] = max(stats["peak_rss_gib"], proc.get("VmRSS", 0.0) / 1024**2)
            stats["peak_hwm_gib"] = max(stats["peak_hwm_gib"], proc.get("VmHWM", 0.0) / 1024**2)
            stats["peak_vmsize_gib"] = max(stats["peak_vmsize_gib"], proc.get("VmSize", 0.0) / 1024**2)
        if mem and "MemAvailable" in mem:
            avail = mem["MemAvailable"] / 1024**2
            stats["min_mem_available_gib"] = (
                avail if stats["min_mem_available_gib"] is None
                else min(stats["min_mem_available_gib"], avail)
            )
        stats["samples"] += 1

    def loop():
        sample_once()
        while not stop_event.wait(interval_s):
            sample_once()

    thread = threading.Thread(target=loop, name="memory-monitor", daemon=True)
    thread.start()
    return stop_event, thread, stats, sample_once


def build_tasks():
    all_tasks = [
        (method, eps, theta, n_wdr, seed)
        for method in METHODS
        for (eps, theta) in EPS_THETA_LIST
        for n_wdr in N_WDR_LIST
        for seed in SEED_LIST
    ]

    remaining = []
    skipped = 0
    for task in all_tasks:
        stem = make_stem(*task)
        npy_path = os.path.join(RESULT_DIR, f"result_{stem}.npy")
        if os.path.exists(npy_path):
            skipped += 1
        else:
            remaining.append(task)

    log(f"completed={skipped}, remaining={len(remaining)} / {len(all_tasks)}")
    return all_tasks, remaining


def run_one(idx, total, method, eps, theta, n_wdr, seed, monitor_interval):
    from Ess import solve_PD_instance

    t_wall_start = time.time()
    status_str = "OK"
    obj_val = solve_time = sat_rate = float("nan")
    stop_mem, mem_thread, mem_stats, sample_mem = start_memory_monitor(monitor_interval)

    stem = make_stem(method, eps, theta, n_wdr, seed)
    npy_path = os.path.join(RESULT_DIR, f"result_{stem}.npy")
    gurobi_log = os.path.join(RESULT_DIR, f"gurobi_{stem}.txt")

    log(
        f"[{idx:>2}/{total}] START {method:<5} N={n_wdr:>3} "
        f"eps={eps} theta={theta} seed={seed} [case118]"
    )

    try:
        res = solve_PD_instance(
            method=method,
            N_WDR=n_wdr,
            epsilon=eps,
            theta=theta,
            seed=seed,
            log_file_name=gurobi_log,
            **FIXED,
        )
        obj_val = res["obj_value"]
        solve_time = res["solve_time"]
        sat_rate = res["satisfied_rate"]
        if res["status"] == 9:
            status_str = "TL"
        sample_mem()

        result_dict = {
            "min_cost (USD)": obj_val,
            "reliability_test (%)": sat_rate * 100,
            "t_solve (s)": solve_time,
            "status": res["status"],
            "ess_beta_all": res.get("ess_beta_all"),
            "ess_plan_all": res.get("ess_plan_all"),
            "ess_soc": res.get("ess_soc"),
            "gen_power_all": res.get("gen_power_all"),
            "gen_alpha_all": res.get("gen_alpha_all"),
            "method": method,
            "epsilon": eps,
            "theta": theta,
            "N_WDR": n_wdr,
            "seed": seed,
            "num_ESS": NUM_ESS,
            "network_name": "case118",
            "num_gen": NUM_GEN_118,
            "peak_rss_gib": mem_stats["peak_rss_gib"],
            "peak_hwm_gib": mem_stats["peak_hwm_gib"],
            "peak_vmsize_gib": mem_stats["peak_vmsize_gib"],
            "min_mem_available_gib": mem_stats["min_mem_available_gib"],
            "memory_samples": mem_stats["samples"],
        }
        np.save(npy_path, result_dict, allow_pickle=True)

    except Exception as exc:
        status_str = "ERR"
        log(
            f"[{idx:>2}/{total}] ERROR {method} N={n_wdr} eps={eps} "
            f"theta={theta} seed={seed}: {str(exc)[:200]}"
        )
        traceback.print_exc()

    finally:
        sample_mem()
        stop_mem.set()
        mem_thread.join(timeout=2.0)

    wall = time.time() - t_wall_start
    log(
        f"[{idx:>2}/{total}] DONE  {method:<5} N={n_wdr:>3} "
        f"eps={eps} theta={theta} seed={seed} {status_str:<3} "
        f"solve={solve_time:>8.1f}s sat={sat_rate:.1%} wall={wall:.0f}s "
        f"peak_rss={mem_stats['peak_rss_gib']:.2f}GiB "
        f"peak_hwm={mem_stats['peak_hwm_gib']:.2f}GiB "
        f"min_avail={mem_stats['min_mem_available_gib']:.1f}GiB"
    )

    row = dict(
        method=method,
        num_ESS=NUM_ESS,
        epsilon=eps,
        theta=theta,
        N_WDR=n_wdr,
        seed=seed,
        status=status_str,
        obj_value=obj_val,
        solve_time=solve_time,
        wall_time=wall,
        satisfied_rate=sat_rate,
        peak_rss_gib=mem_stats["peak_rss_gib"],
        peak_hwm_gib=mem_stats["peak_hwm_gib"],
        peak_vmsize_gib=mem_stats["peak_vmsize_gib"],
        min_mem_available_gib=mem_stats["min_mem_available_gib"],
        memory_samples=mem_stats["samples"],
        network="case118",
        num_gen=NUM_GEN_118,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    pd.DataFrame([row]).to_csv(
        CSV_FILE,
        mode="a",
        header=not os.path.exists(CSV_FILE),
        index=False,
    )
    return row


def write_comparison_csv():
    frames = []

    if os.path.exists(CSV_FILE):
        legacy = pd.read_csv(CSV_FILE)
        legacy = legacy[(legacy["seed"] == 0) & (legacy["method"].isin(METHODS))]
        legacy = legacy[legacy["status"] == "OK"]
        legacy = legacy[legacy["N_WDR"].isin(N_WDR_LIST)]
        frames.append(legacy)

    if not frames:
        log("comparison skipped: no progress CSV files found")
        return

    cols = [
        "method", "epsilon", "theta", "N_WDR", "seed", "status",
        "solve_time", "wall_time", "satisfied_rate", "obj_value",
        "peak_rss_gib", "peak_hwm_gib", "peak_vmsize_gib",
        "min_mem_available_gib", "memory_samples",
    ]
    df = pd.concat(frames, ignore_index=True)
    keep_cols = [c for c in cols if c in df.columns]
    df = df[keep_cols].sort_values(["epsilon", "theta", "N_WDR", "method"])
    df.to_csv(COMPARISON_CSV, index=False)
    log(f"comparison CSV written: {COMPARISON_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List task count without solving.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_N_PARALLEL,
        help="Number of case118 instances to run concurrently.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_GUROBI_THREADS,
        help="Gurobi threads per instance.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=FIXED["time_limit"],
        help="Per-instance Gurobi time limit in seconds.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=30.0,
        help="Memory sampling interval in seconds for each solve process.",
    )
    args = parser.parse_args()

    FIXED["thread"] = args.threads
    FIXED["time_limit"] = args.time_limit

    os.makedirs(RESULT_DIR, exist_ok=True)
    log("=" * 70)
    log("case118 CVaR/EIFICA/FICA legacy-data single-seed timing rerun")
    log(
        f"methods={METHODS}, seeds={SEED_LIST}, parallel={args.parallel}, "
        f"threads={args.threads}, time_limit={args.time_limit}s, "
        f"monitor_interval={args.monitor_interval}s"
    )
    log(f"result_dir={RESULT_DIR}")
    log("=" * 70)

    all_tasks, tasks = build_tasks()
    if args.dry_run:
        log(f"dry run: total={len(all_tasks)}, remaining={len(tasks)}")
        for task in tasks[:10]:
            log(f"  remaining task: {task}")
        if len(tasks) > 10:
            log(f"  ... {len(tasks) - 10} more")
        write_comparison_csv()
        return

    if not tasks:
        log("all legacy-data tasks already completed")
        write_comparison_csv()
        return

    Parallel(n_jobs=args.parallel, backend="loky")(
        delayed(run_one)(i + 1, len(tasks), *task, args.monitor_interval)
        for i, task in enumerate(tasks)
    )
    write_comparison_csv()


if __name__ == "__main__":
    main()
