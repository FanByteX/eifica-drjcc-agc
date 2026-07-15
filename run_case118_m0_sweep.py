"""
Ramp budget (m0) sensitivity sweep for the case118 ESS study.

Sweeps the EIFICA ramp budget m0 on case118 at the strict risk setting
(eps, theta) = (0.03, 0.06), where the ramp screening gap is most visible,
for N in {50, 200} and 5 seeds. For each N the sweep also includes the
default budget m0 = max(2*floor(N*eps), ceil(0.05*N)); m0 = N recovers the
full sample ramp treatment and serves as the upper endpoint.

Outputs
-------
case_study_ess_results/case118_m0_sweep/
  case118_m0_sweep.csv
  case118_m0_sweep.log

Completed (N, m0, seed) combinations are skipped on restart.
"""
import os, sys, math
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Ess import solve_PD_instance

# ---------------- Experiment parameters ----------------
EPS, THETA = 0.03, 0.06
N_LIST     = [50, 200]
SEED_LIST  = [0, 10000, 20000, 30000, 40000]
# m0 values swept as fractions of N; m0 = N (full sample ramp treatment) is the upper endpoint
M0_FRACS   = [0.05, 0.10, 0.20, 0.40, 1.00]

# Fixed parameters aligned with the main case118 experiment (excluding method/N/eps/theta/seed/m0)
FIXED = dict(
    num_gen=38, num_WT=10, num_Solar=5, T=24, norm_ord=1,
    show_plot=False, time_limit=28800, MIPGap=0.001,
    load_scaling_factor=1.0, network_name='case118', thread=4,
    num_ESS=6, error_scale=1.0,
    ESS_power_ratio=0.1, ESS_eta_c=0.95, ESS_eta_d=0.95,
    ESS_SOC_init=0.5, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
    ESS_c_charge=5.0, ESS_c_discharge=5.0, ESS_lambda_AGC=10.0,
)

OUT_DIR = os.path.join(os.getcwd(), 'case_study_ess_results', 'case118_m0_sweep')
os.makedirs(OUT_DIR, exist_ok=True)
CSV = os.path.join(OUT_DIR, 'case118_m0_sweep.csv')
LOG = os.path.join(OUT_DIR, 'case118_m0_sweep.log')


def log(m):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def done_keys():
    if not os.path.exists(CSV):
        return set()
    df = pd.read_csv(CSV)
    return set(zip(df.N.astype(int), df.m0.astype(int), df.seed.astype(int)))


def append_row(row):
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)


def default_m0(N):
    k = math.floor(N * EPS)
    return max(2 * k, math.ceil(0.05 * N))


def main():
    tasks = []
    default_by_N = {}
    for N in N_LIST:
        dm0 = default_m0(N)
        default_by_N[N] = dm0
        m0s = sorted({max(1, math.ceil(f * N)) for f in M0_FRACS} | {N, dm0})
        for m0 in m0s:
            for seed in SEED_LIST:
                tasks.append((N, m0, seed))
    dk = done_keys()
    log(f"total {len(tasks)} runs, {len(dk)} already done, {len(tasks) - len(dk)} remaining")

    for i, (N, m0, seed) in enumerate(tasks, 1):
        if (N, m0, seed) in dk:
            continue
        log(f"[{i}/{len(tasks)}] N={N} m0={m0} (frac={m0/N:.2f}) seed={seed}")
        row = dict(network='case118', eps=EPS, theta=THETA, N=N, m0=m0,
                   m0_frac=round(m0 / N, 3), is_default=(m0 == default_by_N[N]),
                   seed=seed, method='EIFICA',
                   solve_time=float('nan'), objective=float('nan'),
                   satisfied_rate=float('nan'), status='ERR')
        try:
            res = solve_PD_instance(method='EIFICA', N_WDR=N, epsilon=EPS, theta=THETA,
                                    seed=seed, m0_override=m0, log_file_name=None, **FIXED)
            row.update(solve_time=res['solve_time'], objective=res['obj_value'],
                       satisfied_rate=res['satisfied_rate'], status=res['status'])
            log(f"      -> t={res['solve_time']:.1f}s obj={res['obj_value']:.3f} "
                f"sat={res['satisfied_rate']*100:.1f}%")
        except Exception as e:
            log(f"      ERROR: {str(e)[:200]}")
        append_row(row)
    log("done")


if __name__ == '__main__':
    main()
