"""
Batch parameter evaluation for the wind+solar study based on ``wind_solar.py``.

The script compares FICA and EIFICA on mixed wind+solar scenarios and saves
full result dictionaries together with Gurobi log files for downstream analysis.
"""
import os
import sys
import numpy as np
import pandas as pd
import itertools
import time
import traceback
from joblib import Parallel, delayed
from gurobipy import GRB
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# ============================================================
# ============================================================
LOG_FILE = os.path.join(os.getcwd(), 'WT_Solar_all_param_eval.log')
CSV_FILE = os.path.join(os.getcwd(), 'WT_Solar_results', 'progress_main.csv')
CSV_COLUMNS = ['method', 'epsilon', 'theta', 'n_wdr', 'seed',
               'num_WT', 'num_Solar', 'obj', 'solve_time',
               'total_duration', 'satisfied_rate']

def log_msg(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()

def append_csv(row: dict):
    """Append one row to the CSV progress file."""
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    df = pd.DataFrame([row], columns=CSV_COLUMNS)
    df.to_csv(CSV_FILE, mode='a',
              header=not os.path.exists(CSV_FILE),
              index=False)


# ============================================================
# ============================================================
def solve_one_instance(param, save_path_root, bigM, thread, time_limit, total, current_idx):
    """
    Solve one wind+solar instance and save the ``.npy`` result and Gurobi log.

    The saved result format matches ``Solar_all_param_eval.py`` and remains
    compatible with the visualization notebook.
    """
    (network_name, load_scaling_factor, (epsilon, theta), T, num_gen,
     N_WDR, gurobi_seed, method, norm_ord, num_Solar, num_WT) = param

    N_samples_train = 1000
    N_samples_test  = 5000
    MIPGap  = 0.001
    Tstart  = 0

    file_stem = (f'{network_name}_theta{theta}_epsilon{epsilon}'
                 f'_gurobi_seed{gurobi_seed}_num_gen{num_gen}'
                 f'_N_WDR{N_WDR}_load_scaling_factor{load_scaling_factor}'
                 f'_{method}_T{T}_num_Solar{num_Solar}_num_WT{num_WT}')

    log_file_name    = os.path.join(save_path_root, file_stem + '.txt')
    result_dict_path = os.path.join(save_path_root, 'result_' + file_stem + '.npy')

    if os.path.exists(result_dict_path):
        log_msg(f'[{current_idx+1}/{total}] [SKIP] {method} seed={gurobi_seed} '
                f'N_WDR={N_WDR} eps={epsilon} theta={theta} - result already exists')
        return None

    log_msg(f'[{current_idx+1}/{total}] [START] {method} seed={gurobi_seed} '
            f'N_WDR={N_WDR} eps={epsilon} theta={theta} '
            f'num_WT={num_WT} num_Solar={num_Solar}')

    t_wall_start = time.time()

    for fpath in [log_file_name, result_dict_path]:
        if os.path.exists(fpath):
            os.remove(fpath)

    try:
        import importlib
        import wind_solar
        importlib.reload(wind_solar)

        import pandapower as pp
        import pandapower.networks as ppnw
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pd2ppc import _pd2ppc
        from WT_error_gen import WT_sce_gen
        from Solar_error_gen import Solar_sce_gen

        network_dict = {
            'case24_ieee_rts': ppnw.case24_ieee_rts(),
            'case118':         ppnw.case118(),
            'case_ieee30':     ppnw.case_ieee30(),
        }

        seed      = gurobi_seed
        rng       = np.random.RandomState(seed)
        rng_fixed = np.random.RandomState(0)

        network = network_dict[network_name]

        load_location = os.path.join(os.getcwd(), 'data', 'processed', 'UK_norm_load_curve_highest.npy')
        network_load  = np.load(load_location)
        network_load  = np.mean(np.vstack([network_load[::2], network_load[1::2]]), axis=0)
        network_load  = np.tile(network_load, 2)
        network_load  = network_load[Tstart:Tstart+T]

        pp.rundcpp(network)
        _, ppci      = _pd2ppc(network)
        bus_info     = ppci['bus']
        branch_info  = ppci['branch']
        PTDF         = makePTDF(ppci["baseMVA"], bus_info, branch_info, using_sparse_solver=False)
        num_branch   = len(branch_info)

        load_bus_size = bus_info[:, 2] * load_scaling_factor
        load_total    = np.sum(load_bus_size)
        load_bus_all  = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

        gen_cap_total      = load_total
        gen_cap_individual = gen_cap_total / num_gen
        gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
        gen_pmin_individual = 0.1 * gen_cap_individual
        gen_ramp_rate       = 0.6 * gen_cap_individual

        gen_cost        = rng.uniform(23.13, 57.03, num_gen)
        gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)

        bus_list        = np.arange(bus_info.shape[0])
        gen_bus_list    = rng_fixed.choice(bus_list, num_gen,   replace=True)
        WT_bus_list_arr = rng_fixed.choice(bus_list, num_WT,    replace=True)
        Solar_bus_list_arr = (rng_fixed.choice(bus_list, num_Solar, replace=True)
                              if num_Solar > 0 else None)

        P_line_limit = np.abs(ppci['branch'][:, 5])
        P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

        WT_total      = 0.4 * load_total
        WT_individual = WT_total / num_WT
        WT_pred, WT_error_scenarios, _ = WT_sce_gen(num_WT, N_samples_train + N_samples_test)
        WT_pred              = WT_pred[Tstart:Tstart+T] * WT_individual
        WT_error_scenarios   = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual
        WT_error_train       = WT_error_scenarios[:N_samples_train]
        WT_error_test        = WT_error_scenarios[N_samples_train:]

        Solar_total      = 0.45 * load_total
        Solar_individual = Solar_total / num_Solar
        Solar_pred, Solar_error_scenarios, _ = Solar_sce_gen(num_Solar, N_samples_train + N_samples_test)
        Solar_pred            = Solar_pred[Tstart:Tstart+T] * Solar_individual
        Solar_error_scenarios = Solar_error_scenarios[:, Tstart:Tstart+T] * Solar_individual
        for t in range(T):
            hour_of_day = (Tstart + t) % 24
            if hour_of_day < 6 or hour_of_day >= 18:
                Solar_pred[t, :]              = 0
                Solar_error_scenarios[:, t, :] = 0
        Solar_error_train = Solar_error_scenarios[:N_samples_train]
        Solar_error_test  = Solar_error_scenarios[N_samples_train:]

        input_param_dict = {
            'T':          T,
            'num_gen':    num_gen,
            'num_WT':     num_WT,
            'num_branch': num_branch,
            'load_bus_all':              load_bus_all,
            'PTDF':                      PTDF,
            'gen_cap_individual':        gen_cap_individual,
            'gen_pmin_individual':       gen_pmin_individual,
            'WT_pred':                   WT_pred,
            'WT_error_scenarios_train':  WT_error_train,
            'P_line_limit':              P_line_limit,
            'gen_bus_list':              gen_bus_list,
            'WT_bus_list':               WT_bus_list_arr,
            'N_WDR':    N_WDR,
            'epsilon':  epsilon,
            'thread':   thread,
            'theta':    theta,
            'method':   method,
            'MIPGap':   MIPGap,
            'gen_cost':         gen_cost,
            'gen_cost_quadra':  gen_cost_quadra,
            'bigM':             bigM,
            'gurobi_seed':      gurobi_seed,
            'log_file_name':    log_file_name,
            'rng':              rng,
            'norm_ord':         norm_ord,
            'num_Solar':        num_Solar,
            'Solar_pred':       Solar_pred,
            'Solar_error_scenarios_train': Solar_error_train,
            'Solar_bus_list':   Solar_bus_list_arr,
            'gen_ramp_rate':    gen_ramp_rate,
            'time_limit':       time_limit,
        }

        solve_result = wind_solar.solve_PD(**input_param_dict)

        prob             = solve_result['prob']
        gen_power_all_val = solve_result['gen_power_all']
        gen_alpha_all_val = solve_result['gen_alpha_all']
        t_solve           = solve_result.get('solve_time', prob.Runtime if prob else np.nan)

        if (prob is None
                or prob.status not in [GRB.Status.OPTIMAL,
                                       GRB.Status.TIME_LIMIT,
                                       GRB.Status.SUBOPTIMAL]
                or prob.SolCount == 0):
            min_cost        = np.nan
            t_solve         = np.nan if prob is None else time_limit
            reliability_test = np.nan
        else:
            min_cost = prob.objVal

            reliability_test = wind_solar.check_JCC(
                T, num_gen, num_branch,
                gen_power_all_val, gen_alpha_all_val,
                load_bus_all, PTDF,
                gen_cap_individual, gen_pmin_individual,
                WT_pred, WT_error_test,
                P_line_limit, gen_bus_list, WT_bus_list_arr,
                Solar_pred, Solar_error_test, Solar_bus_list_arr,
                gen_ramp_rate,
            ) * 100

            t_wall = time.time() - t_wall_start
            log_msg(f'[{current_idx+1}/{total}] [DONE] {method} seed={gurobi_seed} '
                    f'N_WDR={N_WDR} eps={epsilon} theta={theta}: '
                    f'cost={min_cost:.2f}, JCC={reliability_test:.1f}%, '
                    f'solver_time={t_solve:.1f}s, wall_time={t_wall:.1f}s')

        t_wall = time.time() - t_wall_start
        append_csv({
            'method':          method,
            'epsilon':         epsilon,
            'theta':           theta,
            'n_wdr':           N_WDR,
            'seed':            gurobi_seed,
            'num_WT':          num_WT,
            'num_Solar':       num_Solar,
            'obj':             min_cost,
            'solve_time':      t_solve,
            'total_duration':  t_wall,
            'satisfied_rate':  reliability_test / 100 if not np.isnan(reliability_test) else np.nan,
        })

        result_dict = {
            'min_cost (USD)':       min_cost,
            'reliability_test (%)': reliability_test,
            't_solve (s)':          t_solve,
        }
        np.save(result_dict_path, result_dict, allow_pickle=True)

    except Exception as e:
        t_wall = time.time() - t_wall_start
        log_msg(f'[{current_idx+1}/{total}] [ERROR] {method} seed={gurobi_seed} '
                f'N_WDR={N_WDR} eps={epsilon} theta={theta}: {e} (wall={t_wall:.1f}s)')
        traceback.print_exc()
        append_csv({
            'method':          method,
            'epsilon':         epsilon,
            'theta':           theta,
            'n_wdr':           N_WDR,
            'seed':            gurobi_seed,
            'num_WT':          num_WT,
            'num_Solar':       num_Solar,
            'obj':             np.nan,
            'solve_time':      np.nan,
            'total_duration':  t_wall,
            'satisfied_rate':  np.nan,
        })
        result_dict = {
            'min_cost (USD)':       np.nan,
            'reliability_test (%)': np.nan,
            't_solve (s)':          np.nan,
        }
        np.save(result_dict_path, result_dict, allow_pickle=True)
        if not os.path.exists(log_file_name):
            with open(log_file_name, 'w') as f:
                f.write(f'ERROR: {e}\n')


# ============================================================
# ============================================================
def run_all_param():
    bigM       = 1e5
    thread     = 4
    n_jobs     = 3
    time_limit = 14400

    network_name_list        = ['case24_ieee_rts']
    load_scaling_factor_list = [1]
    eps_theta_pair_list      = [(0.08, 0.12), (0.05, 0.10), (0.10, 0.15), (0.03, 0.06)]
    T_list                   = [24]
    num_gen_list             = [38]
    N_WDR_list               = [50, 80, 100, 150, 200, 250, 300]
    gurobi_seed_list         = [i for i in range(0, 10000 * 5, 10000)]  # 5 seeds
    method_list              = ['FICA', 'EIFICA']
    norm_ord_list            = [1]
    num_Solar_list           = [5]
    num_WT_list              = [10]

    param_comb = list(itertools.product(
        network_name_list, load_scaling_factor_list, eps_theta_pair_list,
        T_list, num_gen_list, N_WDR_list, gurobi_seed_list,
        method_list, norm_ord_list, num_Solar_list, num_WT_list,
    ))

    save_path_root = os.path.join(
        os.getcwd(), f'WT_Solar_results_bigM{int(bigM)}_thread{int(thread)}')
    os.makedirs(save_path_root, exist_ok=True)

    global CSV_FILE
    CSV_FILE = os.path.join(save_path_root, 'progress_main.csv')

    log_msg('=' * 60)
    log_msg('Wind+Solar batch parameter evaluation (wind_solar.py)')
    log_msg(f'Total parameter combinations: {len(param_comb)}')
    log_msg(f'  eps/theta pairs: {eps_theta_pair_list}')
    log_msg(f'  N_WDR:          {N_WDR_list}')
    log_msg(f'  Seeds:          {gurobi_seed_list}')
    log_msg(f'  Methods:        {method_list}')
    log_msg(f'  num_WT:         {num_WT_list},  num_Solar: {num_Solar_list}')
    log_msg(f'Output path:   {save_path_root}')
    log_msg(f'Parallel jobs: {n_jobs}')
    log_msg(f'Time limit:    {time_limit}s = {time_limit/3600:.1f}h')
    log_msg(f'Log file:      {LOG_FILE}')
    log_msg(f'Progress CSV:  {CSV_FILE}')
    log_msg('=' * 60)

    param_comb_with_idx = list(enumerate(param_comb))

    Parallel(n_jobs=n_jobs)(
        delayed(solve_one_instance)(
            param, save_path_root, bigM, thread, time_limit,
            len(param_comb), idx
        )
        for idx, param in param_comb_with_idx
    )

    log_msg('=' * 60)
    log_msg('All parameter combinations finished.')
    log_msg(f'Result directory: {save_path_root}')
    log_msg('=' * 60)


if __name__ == '__main__':
    run_all_param()
