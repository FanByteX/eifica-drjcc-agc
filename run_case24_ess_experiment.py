"""
Main parameter sweep for the case24 ESS study.

This script compares EIFICA and FICA with a fixed six-unit ESS portfolio.
Its parameter grid is aligned with the archived wind+solar benchmark so the
ESS study can be compared directly with earlier baseline runs.
"""


import os, sys, time, traceback
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# ============================================================
GUROBI_THREADS = 6
N_PARALLEL     = 3

# ============================================================
# ============================================================
NUM_ESS       = 6
METHODS       = ['EIFICA', 'FICA']
EPS_THETA_LIST = [
    (0.03, 0.06),
    (0.05, 0.10),
    (0.08, 0.12),
    (0.10, 0.15),
]
N_WDR_LIST   = [50, 80, 100, 150, 200, 250, 300]
SEED_LIST    = [0, 10000, 20000, 30000, 40000]

FIXED = dict(
    num_WT=10,
    num_Solar=5,
    T=24,
    norm_ord=1,
    show_plot=False,
    time_limit=14400,
    MIPGap=0.001,
    load_scaling_factor=1.0,
    network_name='case24_ieee_rts',
    thread=GUROBI_THREADS,
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

# ============================================================
# ============================================================
RESULT_DIR = os.path.join(os.getcwd(), 'case_study_ess_results', 'case24')
os.makedirs(RESULT_DIR, exist_ok=True)
CSV_FILE = os.path.join(RESULT_DIR, 'progress.csv')
LOG_FILE = os.path.join(RESULT_DIR, 'main_exp.log')

def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def make_stem(method, eps, theta, n_wdr, seed):
    """Build a result filename stem that follows the existing convention."""
    return (f'case24_ieee_rts_theta{theta}_epsilon{eps}'
            f'_gurobi_seed{seed}_num_gen38'
            f'_N_WDR{n_wdr}_load_scaling_factor1'
            f'_{method}_T24_num_Solar5_num_WT10_num_ESS{NUM_ESS}')

# ============================================================
# ============================================================
def build_tasks():
    all_tasks = [
        (method, eps, theta, n_wdr, seed)
        for method       in METHODS
        for (eps, theta) in EPS_THETA_LIST
        for n_wdr        in N_WDR_LIST
        for seed         in SEED_LIST
    ]
    remaining = []
    skipped   = 0
    for t in all_tasks:
        stem = make_stem(*t)
        npy_path = os.path.join(RESULT_DIR, f'result_{stem}.npy')
        if os.path.exists(npy_path):
            skipped += 1
        else:
            remaining.append(t)
    log(f"Completed {skipped} runs already present as .npy files; "
        f"{len(remaining)} / {len(all_tasks)} runs remain")
    return remaining

# ============================================================
# ============================================================
def run_one(idx, total, method, eps, theta, n_wdr, seed):
    from Ess import solve_PD_instance
    t_wall_start = time.time()
    status_str   = 'OK'
    obj_val = solve_time = sat_rate = float('nan')

    stem         = make_stem(method, eps, theta, n_wdr, seed)
    npy_path     = os.path.join(RESULT_DIR, f'result_{stem}.npy')
    gurobi_log   = os.path.join(RESULT_DIR, f'gurobi_{stem}.txt')

    log(f"[{idx:>3}/{total}] START {method:<8} N={n_wdr:>3} "
        f"ε={eps} θ={theta} seed={seed}")
    try:
        res = solve_PD_instance(
            method=method, N_WDR=n_wdr,
            epsilon=eps, theta=theta,
            seed=seed,
            **FIXED
        )
        obj_val    = res['obj_value']
        solve_time = res['solve_time']
        sat_rate   = res['satisfied_rate']
        if res['status'] == 9:
            status_str = 'TL'

        result_dict = {
            'min_cost (USD)':        obj_val,
            'reliability_test (%)':  sat_rate * 100,
            't_solve (s)':           solve_time,
            'status':                res['status'],
            'ess_beta_all':   res.get('ess_beta_all'),
            'ess_plan_all':   res.get('ess_plan_all'),
            'ess_soc':        res.get('ess_soc'),
            'gen_power_all':  res.get('gen_power_all'),   # (T, num_gen)
            'gen_alpha_all':  res.get('gen_alpha_all'),   # (T, num_gen)
            'method':  method, 'epsilon': eps, 'theta': theta,
            'N_WDR':   n_wdr,  'seed':    seed, 'num_ESS': NUM_ESS,
        }
        np.save(npy_path, result_dict, allow_pickle=True)

    except Exception as e:
        status_str = 'ERR'
        log(f"[{idx:>3}/{total}] ERROR {method} N={n_wdr} ε={eps} "
            f"seed={seed}: {str(e)[:120]}")
        traceback.print_exc()

    wall = time.time() - t_wall_start
    log(f"[{idx:>3}/{total}] DONE  {method:<8} N={n_wdr:>3} "
        f"ε={eps} θ={theta} seed={seed}  "
        f"{status_str:<4}  solve={solve_time:>7.1f}s  "
        f"sat={sat_rate:.1%}  wall={wall:.0f}s")

    row = dict(
        method=method, num_ESS=NUM_ESS,
        epsilon=eps, theta=theta,
        N_WDR=n_wdr, seed=seed,
        status=status_str,
        obj_value=obj_val,
        solve_time=solve_time,
        wall_time=wall,
        satisfied_rate=sat_rate,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    pd.DataFrame([row]).to_csv(
        CSV_FILE, mode='a',
        header=not os.path.exists(CSV_FILE),
        index=False
    )
    return row

# ============================================================
# ============================================================
if __name__ == '__main__':
    log('=' * 70)
    log(f'ESS main-parameter experiment  num_ESS={NUM_ESS}  '
        f'parallel={N_PARALLEL}  Gurobi threads/job={GUROBI_THREADS}')
    log('Parameters aligned with case24_wt_solar_results '
        '(280 no-ESS baseline runs already available)')
    log('=' * 70)

    tasks = build_tasks()
    if not tasks:
        log('All tasks are already completed.')
        sys.exit(0)

    Parallel(n_jobs=N_PARALLEL, backend='loky')(
        delayed(run_one)(i+1, len(tasks), *t)
        for i, t in enumerate(tasks)
    )

    log('=' * 70)
    log('All runs finished. Summary statistics:')
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        for metric, label in [('solve_time', 'Mean solve time (s)'),
                               ('satisfied_rate', 'Mean out-of-sample satisfaction rate')]:
            pivot = df.pivot_table(
                values=metric,
                index='method',
                columns='N_WDR',
                aggfunc='mean'
            )
            log(f'\n{label}:\n' + pivot.to_string())
