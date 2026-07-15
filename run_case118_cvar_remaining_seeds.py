"""
Legacy-data CVaR supplement for the case118 experiment.

Runs the four remaining CVaR seeds so the CVaR baseline can be averaged over
the same five seeds used by the original FICA/EIFICA case118 figures.

Execute from the repository root so data/ resolves to the snapshot used by
the case118 experiments.
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
METHODS = ["CVAR"]
SEED_LIST = [10000, 20000, 30000, 40000]
EPS_THETA_LIST = [
    (0.03, 0.06),
    (0.05, 0.10),
    (0.08, 0.12),
    (0.10, 0.15),
]
N_WDR_LIST = [50, 80, 100, 150, 200, 250, 300]
NUM_GEN_118 = 38

DEFAULT_GUROBI_THREADS = 8
DEFAULT_N_PARALLEL = 1

RESULT_DIR = os.path.join(os.getcwd(), "case_study_ess_results", "case118_legacy_cvar_5seed")
CSV_FILE = os.path.join(RESULT_DIR, "progress_case118_legacy_cvar_5seed.csv")
LOG_FILE = os.path.join(RESULT_DIR, "legacy_cvar_5seed_exp.log")
SUMMARY_CSV = os.path.join(RESULT_DIR, "case118_legacy_cvar_5seed_summary.csv")

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


def completed_keys_from_csv():
    if not os.path.exists(CSV_FILE):
        return set()
    df = pd.read_csv(CSV_FILE)
    if "status" in df.columns:
        df = df[df["status"].eq("OK")]
    return set(zip(df["method"], df["epsilon"], df["theta"], df["N_WDR"], df["seed"]))


def build_tasks():
    all_tasks = [
        (method, eps, theta, n_wdr, seed)
        for method in METHODS
        for (eps, theta) in EPS_THETA_LIST
        for n_wdr in N_WDR_LIST
        for seed in SEED_LIST
    ]

    completed = completed_keys_from_csv()
    remaining = []
    skipped = 0
    for task in all_tasks:
        stem = make_stem(*task)
        npy_path = os.path.join(RESULT_DIR, f"result_{stem}.npy")
        if task in completed and os.path.exists(npy_path):
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
        f"[{idx:>3}/{total}] START {method:<5} N={n_wdr:>3} "
        f"eps={eps} theta={theta} seed={seed} [case118 legacy]"
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
            f"[{idx:>3}/{total}] ERROR {method} N={n_wdr} eps={eps} "
            f"theta={theta} seed={seed}: {str(exc)[:200]}"
        )
        traceback.print_exc()

    finally:
        sample_mem()
        stop_mem.set()
        mem_thread.join(timeout=2.0)

    wall = time.time() - t_wall_start
    log(
        f"[{idx:>3}/{total}] DONE  {method:<5} N={n_wdr:>3} "
        f"eps={eps} theta={theta} seed={seed} {status_str:<3} "
        f"solve={solve_time:>8.1f}s sat={sat_rate:.1%} wall={wall:.0f}s "
        f"peak_rss={mem_stats['peak_rss_gib']:.2f}GiB "
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


def write_summary_csv():
    frames = []
    seed0_csv = os.path.join(os.getcwd(), "case_study_ess_results", "case118_legacy_seed0", "progress_case118_legacy_seed0.csv")
    if os.path.exists(seed0_csv):
        seed0 = pd.read_csv(seed0_csv)
        seed0 = seed0[
            seed0["method"].eq("CVAR")
            & seed0["seed"].eq(0)
            & seed0["status"].eq("OK")
            & seed0["N_WDR"].isin(N_WDR_LIST)
        ]
        frames.append(seed0)
    if os.path.exists(CSV_FILE):
        rest = pd.read_csv(CSV_FILE)
        rest = rest[
            rest["method"].eq("CVAR")
            & rest["status"].eq("OK")
            & rest["N_WDR"].isin(N_WDR_LIST)
        ]
        frames.append(rest)

    if not frames:
        log("summary skipped: no CVaR progress rows found")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["method", "epsilon", "theta", "N_WDR", "seed"], keep="last")
    df = df.sort_values(["epsilon", "theta", "N_WDR", "seed"])
    df.to_csv(SUMMARY_CSV, index=False)
    counts = df.groupby(["epsilon", "theta", "N_WDR"]).size()
    log(f"summary CSV written: {SUMMARY_CSV}; rows={len(df)}, min_seeds_per_point={counts.min()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List task count without solving.")
    parser.add_argument("--parallel", type=int, default=DEFAULT_N_PARALLEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_GUROBI_THREADS)
    parser.add_argument("--time-limit", type=int, default=FIXED["time_limit"])
    parser.add_argument("--monitor-interval", type=float, default=30.0)
    args = parser.parse_args()

    FIXED["thread"] = args.threads
    FIXED["time_limit"] = args.time_limit

    os.makedirs(RESULT_DIR, exist_ok=True)
    log("=" * 70)
    log("case118 legacy-data CVaR remaining-seed timing supplement")
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
        write_summary_csv()
        return

    if not tasks:
        log("all remaining CVaR seed tasks already completed")
        write_summary_csv()
        return

    Parallel(n_jobs=args.parallel, backend="loky")(
        delayed(run_one)(i + 1, len(tasks), *task, args.monitor_interval)
        for i, task in enumerate(tasks)
    )
    write_summary_csv()


if __name__ == "__main__":
    main()
