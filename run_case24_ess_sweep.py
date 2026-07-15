"""
Storage parameter sensitivity sweep for the case24 ESS study.

Sweeps the storage rated power ratio, the round trip efficiency, and the
regulating reserve margin rho on case24 at the representative risk setting
(eps, theta) = (0.08, 0.12) with N = 200, 3 seeds, method EIFICA. Each
storage configuration (num_ESS = 6) is paired with a no storage baseline
(num_ESS = 0) under the same seed, and
cost_reduction% = (obj_no - obj_with) / obj_no * 100.
Power and efficiency use existing solve_PD_instance parameters; rho uses
reserve_fraction_override.

Outputs
-------
case_study_ess_results/case24_ess_sweep/
  case24_ess_sweep.csv
  case24_ess_sweep.log

Completed (sweep_dim, value, seed) combinations are skipped on restart.
"""
import os, sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Ess import solve_PD_instance

# ---------------- Experiment parameters ----------------
EPS, THETA, N = 0.08, 0.12, 200
SEED_LIST     = [0, 10000, 20000]

# Default storage configuration (aligned with the main experiment)
DEF = dict(ESS_power_ratio=0.10, ESS_eta_c=0.95, ESS_eta_d=0.95,
           reserve_fraction_override=None)
DEFAULT_RHO_LABEL = 1.0 / 6.0  # label only; true model default is reserve_fraction_override=None

# Common fixed parameters (excluding method/N/eps/theta/seed/num_ESS and the swept storage items)
BASE = dict(
    num_gen=38, num_WT=10, num_Solar=5, T=24, norm_ord=1,
    show_plot=False, time_limit=14400, MIPGap=0.001,
    load_scaling_factor=1.0, network_name='case24_ieee_rts', thread=6,
    error_scale=1.0,
    ESS_SOC_init=0.5, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
    ESS_c_charge=5.0, ESS_c_discharge=5.0, ESS_lambda_AGC=10.0,
)


def build_configs():
    """Return the list of (sweep_dim, value, storage_overrides, is_default).

    The default rho point keeps reserve_fraction_override=None so the model
    default logic is reproduced; value is only a table label that lets the
    postprocessing identify the default point.
    """
    cfgs = []
    for pr in [0.05, 0.10, 0.15, 0.20]:
        cfgs.append(('power_ratio', pr, {**DEF, 'ESS_power_ratio': pr}, pr == DEF['ESS_power_ratio']))
    for eta in [0.90, 0.95, 0.98]:
        cfgs.append(('efficiency', eta, {**DEF, 'ESS_eta_c': eta, 'ESS_eta_d': eta}, eta == DEF['ESS_eta_c']))
    for rho in [0.1, DEFAULT_RHO_LABEL, 0.3, 0.5]:
        ov = DEF if rho == DEFAULT_RHO_LABEL else {**DEF, 'reserve_fraction_override': rho}
        cfgs.append(('rho', rho, ov, rho == DEFAULT_RHO_LABEL))
    return cfgs


OUT_DIR = os.path.join(os.getcwd(), 'case_study_ess_results', 'case24_ess_sweep')
os.makedirs(OUT_DIR, exist_ok=True)
CSV = os.path.join(OUT_DIR, 'case24_ess_sweep.csv')
LOG = os.path.join(OUT_DIR, 'case24_ess_sweep.log')


def log(m):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def done_keys():
    if not os.path.exists(CSV):
        return set()
    df = pd.read_csv(CSV)
    return set(zip(df.sweep_dim.astype(str), df.value.astype(float), df.seed.astype(int)))


def append_row(row):
    pd.DataFrame([row]).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)


def agc_share(res):
    beta = res.get('ess_beta_all')
    alpha = res.get('gen_alpha_all')
    if beta is None or alpha is None:
        return float('nan')
    b = float(np.abs(beta).sum())
    a = float(np.abs(alpha).sum())
    return b / (a + b + 1e-12)


def run(num_ESS, seed, ov):
    return solve_PD_instance(method='EIFICA', N_WDR=N, epsilon=EPS, theta=THETA,
                             seed=seed, num_ESS=num_ESS, log_file_name=None,
                             **{**BASE, **ov})


def main():
    cfgs = build_configs()
    dk = done_keys()
    log(f"{len(cfgs)} configs x {len(SEED_LIST)} seeds; {len(dk)} rows already done")

    # The no storage baseline is solved once per seed and cached
    baseline = {}  # seed -> obj_no
    for seed in SEED_LIST:
        try:
            res0 = run(num_ESS=0, seed=seed, ov=DEF)
            baseline[seed] = res0['obj_value']
            log(f"baseline (no storage) seed={seed}: obj={res0['obj_value']:.3f} "
                f"sat={res0['satisfied_rate']*100:.1f}%")
        except Exception as e:
            baseline[seed] = float('nan')
            log(f"baseline seed={seed} ERROR: {str(e)[:200]}")

    for dim, val, ov, is_default in cfgs:
        for seed in SEED_LIST:
            if (dim, float(val), seed) in dk:
                continue
            log(f"[{dim}={val}] seed={seed}")
            row = dict(sweep_dim=dim, value=val, is_default=is_default, seed=seed,
                       obj_no=baseline.get(seed, float('nan')),
                       obj_with=float('nan'), cost_reduction_pct=float('nan'),
                       satisfied_rate_with=float('nan'), ess_agc_share=float('nan'),
                       status_with='ERR')
            try:
                res = run(num_ESS=6, seed=seed, ov=ov)
                obj_no = baseline.get(seed, float('nan'))
                cr = (obj_no - res['obj_value']) / obj_no * 100 if obj_no == obj_no else float('nan')
                row.update(obj_with=res['obj_value'], cost_reduction_pct=cr,
                           satisfied_rate_with=res['satisfied_rate'],
                           ess_agc_share=agc_share(res), status_with=res['status'])
                log(f"      -> cost_red={cr:.3f}% sat={res['satisfied_rate']*100:.1f}% "
                    f"agc_share={row['ess_agc_share']*100:.1f}%")
            except Exception as e:
                log(f"      ERROR: {str(e)[:200]}")
            append_row(row)
    log("done")


if __name__ == '__main__':
    main()
