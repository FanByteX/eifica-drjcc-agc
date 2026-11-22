"""Main FICA and EIFICA optimization program for AGC scheduling.

This module builds and solves the scheduling model, including AGC adjusted
generator ramp constraints and affine cumulative deployed SOC constraints.
The released ``examples/run_cumulative_soc.py`` script supplies the RTS-24
data and settings for the "SOC bounds only" row of the revised manuscript's
terminal SOC comparison table (N=100, epsilon=0.08, theta=0.12). Its Exact
mode runs the corresponding Exact EIFICA case.

Other manuscript experiments use their corresponding data and settings.
Reproducing them with this release may require adapting the input processing,
experiment scripts, solver configuration, and parameters.
"""

import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc

# EIFICA and FICA implementation with cumulative SOC constraints.
import os
import gurobipy as gp
from gurobipy import GRB
# gp.setParam("licensefile", "C:\\Gurobi\\gurobi.lic")
# --------------------------------------------------------------------------
try:
    from .renewable_scenarios import DEFAULT_DATA_DIR
    from .renewable_scenarios import solar_scenarios as Solar_sce_gen
    from .renewable_scenarios import wind_scenarios as WT_sce_gen
except ImportError:
    from renewable_scenarios import DEFAULT_DATA_DIR
    from renewable_scenarios import solar_scenarios as Solar_sce_gen
    from renewable_scenarios import wind_scenarios as WT_sce_gen
from scipy.linalg import norm
import time
from datetime import datetime
# import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'mathtext.fontset': 'cm',
})

def select_envelope_samples(total_RE_error_train, t_g_list, m0):
    """Select the top m0 ramp-stress samples for each time and generator pair."""
    N_WDR, T = total_RE_error_train.shape
    key_sets = {}
    # The endpoint envelope score only depends on the time pair (t-1, t),
    # not on generator g.  Compute the ranking once per t and reuse it.
    key_by_t = {}
    for t in sorted({t for (t, _) in t_g_list}):
        a = total_RE_error_train[:, t]
        b = total_RE_error_train[:, t-1]

        s1 = np.abs(a - b)
        s2 = np.abs(a)
        s3 = np.abs(b)
        scores = np.maximum.reduce([s1, s2, s3])

        m = min(m0, N_WDR)
        if m < N_WDR:
            idx_top = np.argpartition(-scores, m-1)[:m]
            idx_top = idx_top[np.argsort(-scores[idx_top])]
        else:
            idx_top = np.arange(N_WDR, dtype=int)
        key_by_t[t] = np.asarray(idx_top, dtype=int)
    for (t, g) in t_g_list:
        key_sets[(t, g)] = key_by_t[t].copy()
    return key_sets

def select_soc_envelope_samples(total_RE_error_train, m0_soc, lam=0.2):
    """
    Path-level SOC initial screening. Each training scenario is a full error
    trajectory; the score captures cumulative energy drift and total regulation
    energy without using beta, which is unknown before optimization.
    """
    N_WDR = total_RE_error_train.shape[0]
    cumulative_error = np.cumsum(total_RE_error_train, axis=1)
    cumulative_envelope = np.max(np.abs(cumulative_error), axis=1)
    absolute_energy = np.sum(np.abs(total_RE_error_train), axis=1)
    scores = cumulative_envelope + float(lam) * absolute_energy

    m = min(max(int(m0_soc), 1), N_WDR)
    if m < N_WDR:
        idx_top = np.argpartition(-scores, m - 1)[:m]
        idx_top = idx_top[np.argsort(-scores[idx_top])]
    else:
        idx_top = np.arange(N_WDR, dtype=int)
    return np.asarray(idx_top, dtype=int), scores
def check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test,
              P_line_limit, gen_bus_list, WT_bus_list, Solar_pred=None, Solar_error_scenarios_test=None, Solar_bus_list=None,
              gen_ramp_rate=None,
              num_ESS=0, ESS_power_max=None, ESS_beta_all=None, ESS_plan_all=None,
              ESS_capacity=None, ESS_SOC_init=0.5, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
              ESS_SOC_all=None, ESS_bus_list=None, return_details=False):
    # WT_error_scenarios_test has the shape of (N_samples_test, T, num_WT)
    # Solar_error_scenarios_test has the shape of (N_samples_test, T, num_Solar)

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0
    PTDF_gen = PTDF[:, gen_bus_list].T
    PTDF_wind = PTDF[:, WT_bus_list].T
    PTDF_solar = PTDF[:, Solar_bus_list].T if Solar_bus_list is not None else None
    PTDF_ess = PTDF[:, ESS_bus_list].T if ESS_bus_list is not None else None
    PTDF_load = PTDF.T  # the load_bus_all is the load at all buses, with shape (T, num_bus)

    # Pmax min constraints
    P_res = []
    # Total renewable error (wind + solar)
    total_RE_error = WT_error_scenarios_test.sum(axis=-1)
    if Solar_error_scenarios_test is not None:
        total_RE_error = total_RE_error + Solar_error_scenarios_test.sum(axis=-1)

    for t in range(T):
        for g in range(num_gen):
            gen_power_adjusted = gen_power_all[t, g] - total_RE_error[:, t] * gen_alpha_all[t, g]
            P_res.append(gen_power_adjusted <= gen_cap_individual[g])
            P_res.append(gen_power_adjusted >= gen_pmin_individual[g])
    # Line flow constraints
    L_res = []
    for t in range(T):
        for l in range(num_branch):
            total_error_expanded = total_RE_error[:, t]  # (N_WDR,)

            line_flow_gen = (gen_power_all[t] @ PTDF_gen[:, l]
                            - gen_alpha_all[t] @ PTDF_gen[:, l] * total_error_expanded)

            line_flow_wind = (WT_pred[t] + WT_error_scenarios_test[:, t]) @ PTDF_wind[:, l]
            line_flow = line_flow_gen + line_flow_wind

            # Add solar contribution if present
            if Solar_pred is not None and PTDF_solar is not None:
                line_flow_solar = (Solar_pred[t] + Solar_error_scenarios_test[:, t]) @ PTDF_solar[:, l]
                line_flow = line_flow + line_flow_solar

            if (PTDF_ess is not None and ESS_plan_all is not None
                    and ESS_beta_all is not None):
                ess_actual = (
                    ESS_plan_all[t].reshape(1, -1)
                    - total_error_expanded.reshape(-1, 1) * ESS_beta_all[t].reshape(1, -1)
                )
                line_flow = line_flow + ess_actual @ PTDF_ess[:, l]

            line_flow_load = load_bus_all[t] @ PTDF_load[:, l]
            line_flow = line_flow - line_flow_load

            L_res.append(line_flow <= P_line_limit[l])
            L_res.append(line_flow >= -P_line_limit[l])

    # Ramping constraints (if gen_ramp_rate is provided)
    # |[P(t) - α(t)*ξ(t)] - [P(t-1) - α(t-1)*ξ(t-1)]| <= ramp_rate
    R_res = []
    if gen_ramp_rate is not None:
        for t in range(1, T):
            for g in range(num_gen):
                # Actual power at time t: P(t) - α(t)*ξ(t)
                actual_power_t = gen_power_all[t, g] - gen_alpha_all[t, g] * total_RE_error[:, t]
                # Actual power at time t-1: P(t-1) - α(t-1)*ξ(t-1)
                actual_power_t_1 = gen_power_all[t-1, g] - gen_alpha_all[t-1, g] * total_RE_error[:, t-1]
                # Ramping amount
                ramp = actual_power_t - actual_power_t_1
                # Ramp-up and ramp-down constraints
                R_res.append(ramp <= gen_ramp_rate[g])
                R_res.append(ramp >= -gen_ramp_rate[g])

    # Positive ESS power denotes discharge/injection; actual output is P_plan-beta*xi.
    ESS_res = []
    if num_ESS > 0 and ESS_power_max is not None and ESS_beta_all is not None:
        total_RE_error = WT_error_scenarios_test.sum(axis=-1)
        if Solar_error_scenarios_test is not None:
            total_RE_error = total_RE_error + Solar_error_scenarios_test.sum(axis=-1)
        for t in range(T):
            for e in range(num_ESS):
                P_plan_te = ESS_plan_all[t, e] if ESS_plan_all is not None else 0.0
                ess_actual = P_plan_te - ESS_beta_all[t, e] * total_RE_error[:, t]
                ESS_res.append(ess_actual <= ESS_power_max[e])
                ESS_res.append(ess_actual >= -ESS_power_max[e])

    # Actual SOC equals SOC_plan + sum beta*xi/E in the affine AGC model.
    ESS_SOC_res = []
    if (num_ESS > 0 and ESS_beta_all is not None and ESS_capacity is not None
            and (ESS_SOC_all is not None or ESS_plan_all is not None)):
        total_RE_error = WT_error_scenarios_test.sum(axis=-1)
        if Solar_error_scenarios_test is not None:
            total_RE_error = total_RE_error + Solar_error_scenarios_test.sum(axis=-1)

        if ESS_SOC_all is not None:
            soc_plan = ESS_SOC_all
        else:
            # Fallback only used for diagnostics when planned SOC was not returned.
            soc_plan = np.full((T + 1, num_ESS), ESS_SOC_init, dtype=float)
            for t in range(T):
                soc_plan[t + 1, :] = soc_plan[t, :] - ESS_plan_all[t, :] / np.asarray(ESS_capacity)

        for e in range(num_ESS):
            cumulative_shift = np.zeros(total_RE_error.shape[0], dtype=float)
            for t in range(1, T + 1):
                cumulative_shift = cumulative_shift + (
                    ESS_beta_all[t - 1, e] * total_RE_error[:, t - 1] / ESS_capacity[e]
                )
                soc_actual = soc_plan[t, e] + cumulative_shift
                ESS_SOC_res.append(soc_actual >= ESS_SOC_min)
                ESS_SOC_res.append(soc_actual <= ESS_SOC_max)

    if len(ESS_res) > 0 or len(ESS_SOC_res) > 0:
        res = np.vstack(P_res + L_res + (R_res if len(R_res) > 0 else []) + ESS_res + ESS_SOC_res).T
    elif len(R_res) > 0:
        res = np.vstack(P_res + L_res + R_res).T
    else:
        res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))

    P_res_arr = np.vstack(P_res).T
    L_res_arr = np.vstack(L_res).T
    P_satisfy = np.mean(np.all(P_res_arr, axis=1))
    L_satisfy = np.mean(np.all(L_res_arr, axis=1))
    details = {
        'generator': float(P_satisfy),
        'line': float(L_satisfy),
    }
    print(f"[check_JCC] Generator output satisfaction: {P_satisfy*100:.1f}%, line flow satisfaction: {L_satisfy*100:.1f}%", end="")
    if len(R_res) > 0:
        R_res_arr = np.vstack(R_res).T
        R_satisfy = np.mean(np.all(R_res_arr, axis=1))
        details['ramp'] = float(R_satisfy)
        print(f", ramp satisfaction: {R_satisfy*100:.1f}%", end="")
    if len(ESS_res) > 0:
        ESS_res_arr = np.vstack(ESS_res).T
        ESS_satisfy = np.mean(np.all(ESS_res_arr, axis=1))
        details['ess_power'] = float(ESS_satisfy)
        print(f", ESS power satisfaction: {ESS_satisfy*100:.1f}%", end="")
    if len(ESS_SOC_res) > 0:
        ESS_SOC_res_arr = np.vstack(ESS_SOC_res).T
        ESS_SOC_satisfy = np.mean(np.all(ESS_SOC_res_arr, axis=1))
        details['ess_soc'] = float(ESS_SOC_satisfy)
        print(f", cumulative ESS SOC satisfaction: {ESS_SOC_satisfy*100:.1f}%", end="")
    if len(R_res) > 0 or len(ESS_res) > 0 or len(ESS_SOC_res) > 0:
        print("")
    else:
        print("")

    details['joint'] = float(satisfied_rate)
    return (satisfied_rate, details) if return_details else satisfied_rate

def dual_norm_constr(prob, lhs, rhs, norm_ord=2):
    # this is for lhs >= ||rhs||_norm*
    if norm_ord == 1:
        # return inf-norm
        return [lhs >= rhs, lhs >= -rhs]
    elif norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= rhs @ rhs, lhs_anc == lhs]
    elif norm_ord == np.inf:
        # return 1-norm
        rhs_anc = prob.addMVar(rhs.shape, lb=0, ub=GRB.INFINITY)
        return [lhs >= rhs_anc.sum(), rhs_anc >= rhs, rhs_anc >= -rhs]

def dual_norm_constr_exact_method(prob, lhs, rhs_list, norm_ord=2):
    # this is for lhs >= ||rhs||_norm*
    # only implemented the 2-norm
    if norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= gp.quicksum([exp @ exp for exp in rhs_list]), lhs_anc == lhs]
    elif norm_ord == 1:
        # return inf-norm
        return [lhs >= rhs for rhs in rhs_list] + [lhs >= -rhs for rhs in rhs_list]
    else:
        raise NotImplementedError(f'Only 2-norm is implemented, but got {norm_ord}.')
def solve_PD(T, num_gen, num_WT, num_branch, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_train,
              P_line_limit, gen_bus_list, WT_bus_list, N_WDR, epsilon, theta, MIPGap, rng, bigM,
              gen_cost, gen_cost_quadra, gurobi_seed, method="EIFICA",
              njobs = 1, log_file_name = None, thread = 16, norm_ord = 2,
              num_Solar=0, Solar_pred=None, Solar_error_scenarios_train=None, Solar_bus_list=None,
              gen_ramp_rate=None, time_limit=14400,
              debug_log=False,
              num_ESS=0, ESS_power_max=None, ESS_bus_list=None,
              ESS_capacity=None, ESS_eta_c=0.95, ESS_eta_d=0.95,
              ESS_SOC_init=0.5, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
              ESS_c_charge=0.0, ESS_c_discharge=0.0, ESS_lambda_AGC=0.0,
              ESS_terminal_soc='minimum',
              m0_override=None, reserve_fraction_override=None,
              enforce_cumulative_soc=True,
              soc_constraint_generation=True, soc_m0_override=None, soc_m1_override=None,
              soc_score_lambda=0.2, initialization_mode='envelope'):
    """Build and solve the sampled scheduling model for the selected method."""
    if abs(float(ESS_lambda_AGC)) > 1e-12:
        raise ValueError(
            "The manuscript objective contains no AGC participation reward or "
            "penalty; ESS_lambda_AGC must be zero."
        )
    PTDF[np.abs(PTDF) < 1e-5] = 0
    t_start_total = time.time()

    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N_WDR, replace=False)
    WT_error_scenarios_train = WT_error_scenarios_train[random_var_scenario_index, :, :]

    total_RE_error_train = WT_error_scenarios_train.sum(axis=-1)
    if Solar_error_scenarios_train is not None:
        Solar_error_scenarios_train = Solar_error_scenarios_train[random_var_scenario_index, :, :]
        total_RE_error_train = total_RE_error_train + Solar_error_scenarios_train.sum(axis=-1)

    N_WDR_indices = np.arange(N_WDR)

    t_g_list = [(t, g) for t in range(1, T) for g in range(num_gen)]

    k = int(np.floor(N_WDR * epsilon))
    m0 = max(2 * k, int(np.ceil(0.05 * N_WDR)))
    if os.environ.get('EIFICA_M0') is not None:
        m0 = int(os.environ['EIFICA_M0'])
    if m0_override is not None:
        m0 = int(min(max(int(m0_override), 1), N_WDR))
    m0 = int(min(max(m0, 1), N_WDR))
    m1 = max(5, int(np.ceil(0.01 * N_WDR)))
    if os.environ.get('EIFICA_M1') is not None:
        m1 = int(os.environ['EIFICA_M1'])
    m1 = int(min(max(m1, 1), N_WDR))
    max_iter = int(os.environ.get('EIFICA_MAX_ITER', 30))
    reopt_method = os.environ.get('EIFICA_REOPT_METHOD')
    reopt_method = int(reopt_method) if reopt_method is not None and reopt_method != '' else None
    reopt_lp_warm_start = os.environ.get('EIFICA_REOPT_LPWARMSTART')
    reopt_lp_warm_start = int(reopt_lp_warm_start) if reopt_lp_warm_start is not None and reopt_lp_warm_start != '' else None
    tol = 1e-6
    if initialization_mode not in ('envelope', 'empty'):
        raise ValueError("initialization_mode must be 'envelope' or 'empty'")

    if method == 'EIFICA':
        print(f"[EIFICA] N_WDR={N_WDR}, epsilon={epsilon}, k={k}, m0={m0}, m1={m1}, max_iter={max_iter}")
        if reopt_method is not None:
            print(f"[EIFICA] reopt Method={reopt_method}" + (
                f", LPWarmStart={reopt_lp_warm_start}" if reopt_lp_warm_start is not None else ""
            ))
        key_sets = (select_envelope_samples(total_RE_error_train, t_g_list, m0)
                    if initialization_mode == 'envelope' else
                    {key: np.empty(0, dtype=int) for key in t_g_list})
    elif method == 'FICA':
        print(f"[FICA] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}
    else:
        print(f"[{method}] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}

    soc_constraint_generation = bool(soc_constraint_generation)
    if os.environ.get('SOC_CG') is not None:
        soc_constraint_generation = os.environ.get('SOC_CG', '').strip().lower() in ('1', 'true', 'yes', 'on')
    m0_soc = max(2 * k, int(np.ceil(0.05 * N_WDR)))
    if os.environ.get('SOC_M0') is not None:
        m0_soc = int(os.environ['SOC_M0'])
    if soc_m0_override is not None:
        m0_soc = int(soc_m0_override)
    m0_soc = int(min(max(m0_soc, 1), N_WDR))
    if os.environ.get('SOC_SCORE_LAMBDA') is not None:
        soc_score_lambda = float(os.environ['SOC_SCORE_LAMBDA'])
    m1_soc = max(5, int(np.ceil(0.01 * N_WDR)))
    if os.environ.get('SOC_M1') is not None:
        m1_soc = int(os.environ['SOC_M1'])
    if soc_m1_override is not None:
        m1_soc = int(soc_m1_override)
    m1_soc = int(min(max(m1_soc, 1), N_WDR))
    soc_key_set = set()
    soc_scores = None
    if (enforce_cumulative_soc and num_ESS > 0 and soc_constraint_generation
            and initialization_mode == 'envelope'):
        soc_init_idx, soc_scores = select_soc_envelope_samples(total_RE_error_train, m0_soc, lam=soc_score_lambda)
        soc_key_set = set(soc_init_idx.tolist())
        print(
            f"[SOC-CG] enabled: m0_soc={m0_soc}, lambda={soc_score_lambda}, "
            f"m1_soc={m1_soc}, initial_paths={len(soc_key_set)}/{N_WDR}"
        )
        if method == 'EIFICA' and len(soc_key_set) > 0:
            soc_idx_arr = np.asarray(sorted(soc_key_set), dtype=int)
            ramp_key_count_before = sum(len(v) for v in key_sets.values())
            for key, idx in key_sets.items():
                key_sets[key] = np.union1d(np.asarray(idx, dtype=int), soc_idx_arr).astype(int)
            ramp_key_count_after = sum(len(v) for v in key_sets.values())
            print(
                f"[SOC-CG] merged SOC initial paths into every ramp key set: "
                f"ramp_keys {ramp_key_count_before}->{ramp_key_count_after}"
            )
    initial_soc_key_set = np.asarray(sorted(soc_key_set), dtype=int)
    initial_ramp_key_count = sum(len(v) for v in key_sets.values())

    ramp_active_masks = {}
    if method == 'EIFICA':
        for key, idx in key_sets.items():
            mask = np.zeros(N_WDR, dtype=bool)
            mask[np.asarray(idx, dtype=int)] = True
            ramp_active_masks[key] = mask

    iteration_log = []
    line_block_build_total_s = 0.0

    final_solution = None
    final_prob = None

    def find_soc_violated_paths(sol_ess_beta, sol_ess_soc):
        if (not enforce_cumulative_soc or not soc_constraint_generation or num_ESS <= 0
                or sol_ess_beta is None or sol_ess_soc is None):
            return np.array([], dtype=int), 0, 0.0
        omitted_idx = np.asarray([i for i in N_WDR_indices if int(i) not in soc_key_set], dtype=int)
        if omitted_idx.size == 0:
            return np.array([], dtype=int), 0, 0.0

        severity = np.zeros(omitted_idx.size)
        for e in range(num_ESS):
            cap_e = float(ESS_capacity[e])
            # (num_omitted, T): cumulative beta-weighted energy shift in SOC units
            shifts = np.cumsum(
                total_RE_error_train[omitted_idx, :] * sol_ess_beta[:, e].reshape(1, -1),
                axis=1
            ) * (1.0 / cap_e)
            actual_soc = sol_ess_soc[1:, e].reshape(1, -1) + shifts
            lower_violation = ESS_SOC_min - actual_soc
            upper_violation = actual_soc - ESS_SOC_max
            severity = np.maximum(severity, np.max(np.maximum(lower_violation, upper_violation), axis=1))

        violated_local = np.where(severity > tol)[0]
        violated_count = int(violated_local.size)
        max_violation = float(np.max(severity)) if severity.size > 0 else 0.0
        if violated_count == 0:
            return np.array([], dtype=int), 0, max_violation

        order = violated_local[np.argsort(-severity[violated_local])]
        idx_add = omitted_idx[order[:m1_soc]]
        return np.asarray(idx_add, dtype=int), violated_count, max_violation

    for it in range(max_iter + 1):
        iter_time_start = time.time()
        _env = gp.Env(empty=True)
        if log_file_name is not None:
            _env.setParam('LogFile', log_file_name)
        _env.start()
        prob = gp.Model('ED', env=_env)
        gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
        gen_alpha_all = prob.addMVar((T, num_gen), lb=0, ub=1)

        if num_ESS > 0:
            ESS_beta_all    = prob.addMVar((T, num_ESS), lb=-1.0, ub=1.0)
            ESS_discharge   = prob.addMVar((T, num_ESS), lb=0.0, ub=GRB.INFINITY)
            ESS_charge      = prob.addMVar((T, num_ESS), lb=0.0, ub=GRB.INFINITY)
            ESS_plan_all    = prob.addMVar((T, num_ESS), lb=-GRB.INFINITY, ub=GRB.INFINITY)
            ESS_SOC         = prob.addMVar((T + 1, num_ESS), lb=ESS_SOC_min, ub=ESS_SOC_max)
        else:
            ESS_beta_all = ESS_discharge = ESS_charge = ESS_plan_all = ESS_SOC = None

        if num_ESS > 0 and ESS_capacity is not None:
            dt = 1.0
            for e in range(num_ESS):
                prob.addConstr(ESS_SOC[0, e] == ESS_SOC_init)
                if ESS_terminal_soc == 'cyclic':
                    prob.addConstr(ESS_SOC[T, e] == ESS_SOC_init)
                elif ESS_terminal_soc == 'minimum':
                    prob.addConstr(ESS_SOC[T, e] >= ESS_SOC_min)
                else:
                    raise ValueError(
                        "ESS_terminal_soc must be 'cyclic' or 'minimum', "
                        f"got {ESS_terminal_soc!r}"
                    )
                for t in range(T):
                    prob.addConstr(ESS_discharge[t, e] <= ESS_power_max[e])
                    prob.addConstr(ESS_charge[t, e]    <= ESS_power_max[e])
                    prob.addConstr(ESS_plan_all[t, e] == ESS_discharge[t, e] - ESS_charge[t, e])
                    prob.addConstr(
                        ESS_SOC[t + 1, e] == ESS_SOC[t, e]
                        - (ESS_discharge[t, e] / ESS_eta_d
                           - ESS_eta_c * ESS_charge[t, e]) * dt / ESS_capacity[e]
                    )
                    beta_ub_e = np.minimum(1.0, ESS_power_max / (ESS_power_max.sum() + 1e-9))[e]
                    reserve_fraction = (min(beta_ub_e, 0.3) if reserve_fraction_override is None
                                        else float(reserve_fraction_override))
                    prob.addConstr(
                        ESS_discharge[t, e] <= ESS_power_max[e] * (1.0 - reserve_fraction)
                    )
                    prob.addConstr(
                        ESS_charge[t, e] <= ESS_power_max[e] * (1.0 - reserve_fraction)
                    )

        for t in range(T):
            total_renewable = WT_pred[t, :].sum()
            if Solar_pred is not None:
                total_renewable = total_renewable + Solar_pred[t, :].sum()
            # Positive ESS power denotes discharge/injection.
            if num_ESS > 0 and ESS_plan_all is not None:
                prob.addConstr(
                    gen_power_all[t, :].sum() + total_renewable
                    + ESS_plan_all[t, :].sum() == load_bus_all[t, :].sum()
                )
            else:
                prob.addConstr(gen_power_all[t, :].sum() + total_renewable == load_bus_all[t, :].sum())

            if num_WT > 0:
                if num_ESS > 0:
                    prob.addConstr(gen_alpha_all[t, :].sum() + ESS_beta_all[t, :].sum() == 1)
                else:
                    prob.addConstr(gen_alpha_all[t, :].sum() == 1)
            else:
                has_solar = (Solar_pred is not None) and (Solar_pred[t, :].sum() > 1e-6)
                if has_solar:
                    if num_ESS > 0:
                        prob.addConstr(gen_alpha_all[t, :].sum() + ESS_beta_all[t, :].sum() == 1)
                    else:
                        prob.addConstr(gen_alpha_all[t, :].sum() == 1)
                else:
                    prob.addConstr(gen_alpha_all[t, :] == 0)
                    if num_ESS > 0:
                        prob.addConstr(ESS_beta_all[t, :] == 0)

            prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)
            prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)

            if num_ESS > 0 and ESS_power_max is not None:
                beta_ub = np.minimum(1.0, ESS_power_max / (ESS_power_max.sum() + 1e-9))
                for e in range(num_ESS):
                    prob.addConstr(ESS_beta_all[t, e] <= beta_ub[e])
                    prob.addConstr(ESS_beta_all[t, e] >= -beta_ub[e])

        if gen_ramp_rate is not None:
            for t in range(1, T):
                prob.addConstr(gen_power_all[t, :] - gen_power_all[t-1, :] <= gen_ramp_rate)
                prob.addConstr(gen_power_all[t-1, :] - gen_power_all[t, :] <= gen_ramp_rate)

        s = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        r = prob.addMVar(N_WDR, lb=0, ub=GRB.INFINITY)

        bAx_list = [] if method == 'ExactLHS' else None

        if method == 'ExactLHS':
            z = prob.addMVar(N_WDR, vtype=GRB.BINARY)
            prob.addConstr(bigM * (1 - z) >= s - r)
            prob.addConstr(gp.quicksum(z) <= N_WDR * epsilon)

        PTDF[np.abs(PTDF) < 1e-5] = 0
        PTDF_gen = PTDF[:, gen_bus_list].T
        PTDF_wind = PTDF[:, WT_bus_list].T
        PTDF_solar = PTDF[:, Solar_bus_list].T if Solar_bus_list is not None else None
        PTDF_load = PTDF.T

        for t in range(T):
            for g in range(num_gen):
                b_Ax = gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))

                if method == 'CVAR':
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] >= s - r[N_WDR_indices])
                elif method == 'ExactLHS':
                    bAx_list.append(b_Ax)
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] + bigM * z[N_WDR_indices] >= 0)
                elif method in ['EIFICA', 'FICA']:
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    k_local = int(np.floor(N_WDR * epsilon))
                    random_elements = total_RE_error_train[:,t]
                    q_p_plus_base = np.sort(random_elements)[k_local]
                    q_p_minus_base = np.sort(random_elements)[N_WDR-k_local-1]
                    N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                    N_p_minus = np.where(random_elements > q_p_minus_base)[0]
                    if len(N_p_plus) > 0:
                        prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_p_plus,t:t+1].T - gen_power_all[t, g] >= s - r[N_p_plus])
                    if len(N_p_minus) > 0:
                        prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_p_minus,t:t+1].T - gen_power_all[t, g] >= s - r[N_p_minus])
                    prob.addConstr(q_p_plus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)
                    prob.addConstr(q_p_minus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)

        for t in range(T):
            for g in range(num_gen):
                b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))

                if method == 'CVAR':
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] >= s - r[N_WDR_indices])
                elif method == 'ExactLHS':
                    bAx_list.append(b_Ax)
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] + bigM * z[N_WDR_indices] >= 0)
                elif method in ['EIFICA', 'FICA']:
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    k_local = int(np.floor(N_WDR * epsilon))
                    random_elements = total_RE_error_train[:,t]
                    q_p_plus_base = np.sort(random_elements)[k_local]
                    q_p_minus_base = np.sort(random_elements)[N_WDR-k_local-1]
                    N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                    N_p_minus = np.where(random_elements > q_p_minus_base)[0]
                    if len(N_p_plus) > 0:
                        prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_p_plus,t:t+1].T + gen_power_all[t, g] >= s - r[N_p_plus])
                    if len(N_p_minus) > 0:
                        prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_p_minus,t:t+1].T + gen_power_all[t, g] >= s - r[N_p_minus])
                    prob.addConstr(-q_p_plus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)
                    prob.addConstr(-q_p_minus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)

        if method == 'ExactLHS':
            bAx_list = [bAx * N_WDR * theta for bAx in bAx_list]
            prob.addConstrs(
                constr for constr in dual_norm_constr_exact_method(
                    prob, epsilon * N_WDR * s - gp.quicksum(r), bAx_list, norm_ord=norm_ord
                )
            )

        PTDF_gen = PTDF[:, gen_bus_list]
        PTDF_wind = PTDF[:, WT_bus_list]
        PTDF_solar = PTDF[:, Solar_bus_list] if Solar_bus_list is not None else None
        PTDF_load = PTDF
        PTDF_ess = PTDF[:, ESS_bus_list] if (num_ESS > 0 and ESS_bus_list is not None) else None

        line_block_build_start = time.perf_counter()
        t_l_list = [(t, l) for t in range(T) for l in range(num_branch)]

        # 3.1) Line Max Flow Constraints
        for t, l in t_l_list:
            num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
            b_Ax_wind = -PTDF_wind[l]
            b_Ax_solar = -PTDF_solar[l] if (PTDF_solar is not None and num_Solar > 0) else np.zeros(0)
            b_Ax_combined = np.concatenate([b_Ax_wind, b_Ax_solar])
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_RE) + b_Ax_combined
            if PTDF_ess is not None and ESS_beta_all is not None:
                # P_limit-flow has +PTDF_ess*beta in each renewable-error dimension.
                b_Ax = b_Ax + (PTDF_ess[l] @ ESS_beta_all[t]) * np.ones(num_RE)

            line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T
            line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T
            line_flow_load = PTDF_load[l] @ load_bus_all[t]
            if PTDF_solar is not None and Solar_error_scenarios_train is not None:
                line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T
                total_line_flow = line_flow_gen + line_flow_wind + line_flow_solar - line_flow_load
            else:
                total_line_flow = line_flow_gen + line_flow_wind - line_flow_load
            if PTDF_ess is not None and ESS_plan_all is not None:
                line_flow_ess_plan = PTDF_ess[l] @ ESS_plan_all[t]
                total_line_flow = total_line_flow + line_flow_ess_plan
            if PTDF_ess is not None and ESS_beta_all is not None:
                line_flow_ess_agc = PTDF_ess[l] @ ESS_beta_all[t] * total_RE_error_train[N_WDR_indices, t]
                total_line_flow = total_line_flow - line_flow_ess_agc

            if method == 'ExactLHS':
                bAx_list.append(b_Ax)
                prob.addConstr(P_line_limit[l] - total_line_flow + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                prob.addConstr(P_line_limit[l] - total_line_flow + bigM * z[N_WDR_indices] >= 0)
            else:
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                prob.addConstr(P_line_limit[l] - total_line_flow >= s - r[N_WDR_indices])

        # 3.2) Line Min Flow Constraints (P_line_min = -P_line_limit)
        P_line_min = -P_line_limit
        for t, l in t_l_list:
            num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
            b_Ax_wind = PTDF_wind[l]
            b_Ax_solar = PTDF_solar[l] if (PTDF_solar is not None and num_Solar > 0) else np.zeros(0)
            b_Ax_combined = np.concatenate([b_Ax_wind, b_Ax_solar])
            b_Ax = -PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_RE) + b_Ax_combined
            if PTDF_ess is not None and ESS_beta_all is not None:
                # flow-P_min has -PTDF_ess*beta in each renewable-error dimension.
                b_Ax = b_Ax - (PTDF_ess[l] @ ESS_beta_all[t]) * np.ones(num_RE)

            line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T
            line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T
            line_flow_load = PTDF_load[l] @ load_bus_all[t]
            if PTDF_solar is not None and Solar_error_scenarios_train is not None:
                line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T
                total_line_flow = line_flow_gen + line_flow_wind + line_flow_solar - line_flow_load
            else:
                total_line_flow = line_flow_gen + line_flow_wind - line_flow_load
            if PTDF_ess is not None and ESS_plan_all is not None:
                total_line_flow = total_line_flow + PTDF_ess[l] @ ESS_plan_all[t]
            if PTDF_ess is not None and ESS_beta_all is not None:
                line_flow_ess_agc = PTDF_ess[l] @ ESS_beta_all[t] * total_RE_error_train[N_WDR_indices, t]
                total_line_flow = total_line_flow - line_flow_ess_agc

            if method == 'ExactLHS':
                bAx_list.append(b_Ax)
                prob.addConstr(-P_line_min[l] + total_line_flow + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                prob.addConstr(-P_line_min[l] + total_line_flow + bigM * z[N_WDR_indices] >= 0)
            else:
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                prob.addConstr(-P_line_min[l] + total_line_flow >= s - r[N_WDR_indices])

        line_block_build_time_s = time.perf_counter() - line_block_build_start
        line_block_build_total_s += line_block_build_time_s
        if gen_ramp_rate is not None:
            ramp_constr_count = 0
            for (t, g) in t_g_list:
                if method == 'EIFICA':
                    sel_idx = key_sets.get((t, g), None)
                    if sel_idx is None:
                        sel_idx = N_WDR_indices
                    sel_idx = np.asarray(sel_idx, dtype=int)
                    if sel_idx.size == 0:
                        continue
                else:
                    sel_idx = N_WDR_indices

                alpha_t = gen_alpha_all[t, g]
                alpha_t_minus_1 = gen_alpha_all[t-1, g]
                delta_P = gen_power_all[t, g] - gen_power_all[t-1, g]

                if method == 'ExactLHS':
                    bAx_list.append(alpha_t * np.ones(num_RE))
                    bAx_list.append(-alpha_t_minus_1 * np.ones(num_RE))

                    prob.addConstr(gen_ramp_rate[g] - delta_P
                                   + alpha_t * total_RE_error_train[sel_idx, t]
                                   - alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   + bigM * z[sel_idx] >= s - r[sel_idx])
                    prob.addConstr(gen_ramp_rate[g] - delta_P
                                   + alpha_t * total_RE_error_train[sel_idx, t]
                                   - alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   + bigM * z[sel_idx] >= 0)
                    prob.addConstr(gen_ramp_rate[g] + delta_P
                                   - alpha_t * total_RE_error_train[sel_idx, t]
                                   + alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   + bigM * z[sel_idx] >= s - r[sel_idx])
                    prob.addConstr(gen_ramp_rate[g] + delta_P
                                   - alpha_t * total_RE_error_train[sel_idx, t]
                                   + alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   + bigM * z[sel_idx] >= 0)
                    ramp_constr_count += 4 * len(sel_idx)
                else:
                    if method in ['FICA', 'CVAR']:
                        num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
                        lhs_expr = epsilon * N_WDR * s - r.sum()
                        if norm_ord == 1:
                            prob.addConstr(lhs_expr >= theta * N_WDR * alpha_t)
                            prob.addConstr(lhs_expr >= -theta * N_WDR * alpha_t)
                            prob.addConstr(lhs_expr >= theta * N_WDR * alpha_t_minus_1)
                            prob.addConstr(lhs_expr >= -theta * N_WDR * alpha_t_minus_1)
                        elif norm_ord == 2:
                            lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
                            prob.addConstr(lhs_anc * lhs_anc >= num_RE * (alpha_t * alpha_t + alpha_t_minus_1 * alpha_t_minus_1))
                            prob.addConstr(lhs_anc == lhs_expr / (theta * N_WDR))

                    prob.addConstr(gen_ramp_rate[g] - delta_P
                                   + alpha_t * total_RE_error_train[sel_idx, t]
                                   - alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   >= s - r[sel_idx])
                    prob.addConstr(gen_ramp_rate[g] + delta_P
                                   - alpha_t * total_RE_error_train[sel_idx, t]
                                   + alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                                   >= s - r[sel_idx])
                    ramp_constr_count += 2 * len(sel_idx)
            if debug_log:
                print(f"Iter {it}: added ~{ramp_constr_count} ramp constraints (method={method})")

        # Final step for ExactLHS: Add the combined dual norm constraint
        if method == 'ExactLHS':
            bAx_list = [bAx * N_WDR * theta for bAx in bAx_list]
            prob.addConstrs(
                constr for constr in dual_norm_constr_exact_method(
                    prob, epsilon * N_WDR * s - gp.quicksum(r), bAx_list, norm_ord=norm_ord
                )
            )

        if num_ESS > 0 and ESS_beta_all is not None and ESS_power_max is not None:
            num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
            for t in range(T):
                random_elements = total_RE_error_train[:, t]
                k_local = int(np.floor(N_WDR * epsilon))
                q_plus  = np.sort(random_elements)[k_local]
                q_minus = np.sort(random_elements)[N_WDR - k_local - 1]
                N_p_plus  = np.where(random_elements < q_plus)[0]
                N_p_minus = np.where(random_elements > q_minus)[0]

                for e in range(num_ESS):
                    P_e_max = ESS_power_max[e]
                    beta_te = ESS_beta_all[t, e]

                    b_Ax_ess = beta_te * np.ones(num_RE)
                    if method in ['FICA', 'EIFICA', 'CVAR']:
                        prob.addConstrs(
                            constr for constr in dual_norm_constr(
                                prob, epsilon * N_WDR * s - r.sum(),
                                theta * N_WDR * b_Ax_ess, norm_ord=norm_ord
                            )
                        )

                    if method == 'CVAR':
                        P_plan_te = ESS_plan_all[t, e] if ESS_plan_all is not None else 0.0
                        # P_actual=P_plan-beta*xi.
                        prob.addConstr(
                            P_e_max - P_plan_te + beta_te * total_RE_error_train[N_WDR_indices, t]
                            >= s - r[N_WDR_indices])
                        prob.addConstr(
                            P_e_max + P_plan_te - beta_te * total_RE_error_train[N_WDR_indices, t]
                            >= s - r[N_WDR_indices])

                    elif method in ['FICA', 'EIFICA']:
                        P_plan_te = ESS_plan_all[t, e] if ESS_plan_all is not None else 0.0

                        if len(N_p_plus) > 0:
                            prob.addConstr(
                                P_e_max - P_plan_te + beta_te * total_RE_error_train[N_p_plus, t]
                                >= s - r[N_p_plus])
                        if len(N_p_minus) > 0:
                            prob.addConstr(
                                P_e_max - P_plan_te + beta_te * total_RE_error_train[N_p_minus, t]
                                >= s - r[N_p_minus])
                        prob.addConstr(P_e_max - P_plan_te + q_plus  * beta_te >= s)
                        prob.addConstr(P_e_max - P_plan_te + q_minus * beta_te >= s)

                        if len(N_p_plus) > 0:
                            prob.addConstr(
                                P_e_max + P_plan_te - beta_te * total_RE_error_train[N_p_plus, t]
                                >= s - r[N_p_plus])
                        if len(N_p_minus) > 0:
                            prob.addConstr(
                                P_e_max + P_plan_te - beta_te * total_RE_error_train[N_p_minus, t]
                                >= s - r[N_p_minus])
                        prob.addConstr(P_e_max + P_plan_te - q_plus  * beta_te >= s)
                        prob.addConstr(P_e_max + P_plan_te - q_minus * beta_te >= s)

        cumulative_soc_constr_count = 0
        if enforce_cumulative_soc and num_ESS > 0:
            if ESS_capacity is None:
                raise ValueError("ESS_capacity must be provided when enforce_cumulative_soc=True")
            dt = 1.0
            if soc_constraint_generation:
                soc_initial_indices = np.asarray(sorted(soc_key_set), dtype=int)
            else:
                soc_initial_indices = N_WDR_indices
            for e in range(num_ESS):
                cap_e = float(ESS_capacity[e])
                for sample_idx in soc_initial_indices:
                    agc_soc_shift = 0.0
                    for t in range(1, T + 1):
                        agc_soc_shift = agc_soc_shift + (
                            ESS_beta_all[t - 1, e]
                            * float(total_RE_error_train[sample_idx, t - 1])
                            * dt / cap_e
                        )
                        soc_actual = ESS_SOC[t, e] + agc_soc_shift
                        prob.addConstr(soc_actual >= ESS_SOC_min)
                        prob.addConstr(soc_actual <= ESS_SOC_max)
                        cumulative_soc_constr_count += 2
            if debug_log:
                print(f"Added {cumulative_soc_constr_count} cumulative ESS SOC sample constraints")
            if soc_constraint_generation:
                print(
                    f"[SOC-CG] initial SOC paths added: {len(soc_initial_indices)}/{N_WDR} "
                    f"({cumulative_soc_constr_count} constraints)"
                )

        # objective and solver params
        FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2

        obj_expr = FC.sum()

        if num_ESS > 0 and ESS_discharge is not None:
            if ESS_c_discharge > 0:
                obj_expr = obj_expr + ESS_c_discharge * ESS_discharge.sum()
            if ESS_c_charge > 0:
                obj_expr = obj_expr + ESS_c_charge * ESS_charge.sum()

        prob.setObjective(obj_expr, GRB.MINIMIZE)

        print(f"Setting Gurobi Params: TimeLimit={time_limit}, MIPGap={MIPGap}")

        prob.setParam('MIPGap', MIPGap)
        prob.setParam('Seed', gurobi_seed)
        prob.setParam('Threads', thread)
        prob.setParam('InfUnbdInfo', 1)
        diagnostic_params = {
            'SOC_GUROBI_METHOD': 'Method',
            'SOC_GUROBI_NUMERIC_FOCUS': 'NumericFocus',
            'SOC_GUROBI_BAR_HOMOGENEOUS': 'BarHomogeneous',
            'SOC_GUROBI_CROSSOVER': 'Crossover',
        }
        for env_key, param_name in diagnostic_params.items():
            if os.environ.get(env_key) not in (None, ''):
                param_value = int(os.environ[env_key])
                prob.setParam(param_name, param_value)
                print(f"Diagnostic Gurobi param: {param_name}={param_value}")
        if time_limit is not None and time_limit > 0:
            prob.setParam('TimeLimit', time_limit)
        if log_file_name is not None:
            prob.setParam('LogFile', log_file_name)
            for tt in range(T):
                for gg in range(num_gen):
                    try:
                        gen_power_all[tt, gg].start = prev_p[tt, gg]
                        gen_alpha_all[tt, gg].start = prev_a[tt, gg]
                    except Exception:
                        pass

        prob.optimize()

        if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
            if prob.status == GRB.Status.INFEASIBLE:
                print('[DEBUG] INFEASIBLE: computing IIS...')
                prob.computeIIS()
                iis_constrs = [c.ConstrName for c in prob.getConstrs() if c.IISConstr]
                iis_lb = [v.VarName for v in prob.getVars() if v.IISLB]
                iis_ub = [v.VarName for v in prob.getVars() if v.IISUB]
                print(f'  IIS constraints ({len(iis_constrs)}): {iis_constrs[:20]}')
                print(f'  IIS LB violations ({len(iis_lb)}): {iis_lb[:10]}')
                print(f'  IIS UB violations ({len(iis_ub)}): {iis_ub[:10]}')
            raise ValueError(f'Iter {it}: Solver failed with status {prob.status}')

        sol_gen_power = gen_power_all.X.copy()
        sol_gen_alpha = gen_alpha_all.X.copy()
        sol_s = float(np.asarray(s.X).reshape(-1)[0])
        sol_ess_beta    = ESS_beta_all.X.copy()  if (num_ESS > 0 and ESS_beta_all  is not None) else None
        sol_ess_plan    = ESS_plan_all.X.copy()   if (num_ESS > 0 and ESS_plan_all  is not None) else None
        sol_ess_discharge = ESS_discharge.X.copy() if (num_ESS > 0 and ESS_discharge is not None) else None
        sol_ess_charge = ESS_charge.X.copy() if (num_ESS > 0 and ESS_charge is not None) else None
        sol_ess_soc     = ESS_SOC.X.copy()        if (num_ESS > 0 and ESS_SOC       is not None) else None
        final_solution = {'gen_power_all': sol_gen_power, 'gen_alpha_all': sol_gen_alpha,
                          'ess_beta_all': sol_ess_beta, 'ess_plan_all': sol_ess_plan,
                          'ess_discharge_all': sol_ess_discharge,
                          'ess_charge_all': sol_ess_charge,
                          'ess_soc': sol_ess_soc, 'obj': prob.objVal, 's_value': sol_s}
        final_prob = prob

        if method != 'EIFICA':
            print(f"[{method}] optimization finished in one pass.")
            break

        any_added = False
        per_constraint_added = {}
        added_indices_by_constraint = {}
        max_rho = 0.0
        violated_count = 0
        total_key_samples_before = sum(int(mask.sum()) for mask in ramp_active_masks.values())
        if gen_ramp_rate is not None:
            for (t, g) in t_g_list:
                active_mask = ramp_active_masks.get((t, g))
                if active_mask is None:
                    active_mask = np.zeros(N_WDR, dtype=bool)
                    ramp_active_masks[(t, g)] = active_mask
                omitted_idx = N_WDR_indices[~active_mask]
                if omitted_idx.size == 0:
                    per_constraint_added[(t, g)] = 0
                    continue

                delta_P = sol_gen_power[t, g] - sol_gen_power[t - 1, g]
                ramp_term = (
                    sol_gen_alpha[t, g] * total_RE_error_train[omitted_idx, t]
                    - sol_gen_alpha[t - 1, g] * total_RE_error_train[omitted_idx, t - 1]
                )
                lhs_upper = gen_ramp_rate[g] - delta_P + ramp_term
                lhs_lower = gen_ramp_rate[g] + delta_P - ramp_term
                rho = sol_s - np.minimum(lhs_upper, lhs_lower)
                if rho.size > 0:
                    max_rho = max(max_rho, float(np.max(rho)))

                violated_local = np.where(rho > tol)[0]
                violated_count += int(violated_local.size)
                if violated_local.size == 0:
                    per_constraint_added[(t, g)] = 0
                    continue

                order = violated_local[np.argsort(-rho[violated_local])]
                idx_add = omitted_idx[order[:m1]]
                idx_add = idx_add[~active_mask[idx_add]]
                added_count = int(idx_add.size)
                per_constraint_added[(t, g)] = added_count
                if added_count > 0:
                    any_added = True
                    added_indices_by_constraint[(t, g)] = np.asarray(idx_add, dtype=int)
                    active_mask[idx_add] = True
                    key_sets[(t, g)] = np.flatnonzero(active_mask)
        total_key_samples_after = sum(int(mask.sum()) for mask in ramp_active_masks.values())

        soc_paths_before = len(soc_key_set)
        added_soc_indices = np.array([], dtype=int)
        soc_violated_count = 0
        soc_max_violation = 0.0
        if soc_constraint_generation:
            added_soc_indices, soc_violated_count, soc_max_violation = find_soc_violated_paths(sol_ess_beta, sol_ess_soc)
            if len(added_soc_indices) > 0:
                any_added = True
                soc_key_set.update(int(i) for i in added_soc_indices.tolist())
        soc_paths_after = len(soc_key_set)

        iter_time = time.time() - iter_time_start
        iter_log_entry = {
            'iter': it,
            'obj': final_solution['obj'],
            's_value': float(sol_s),
            'max_rho': float(max_rho),
            'violated_count': int(violated_count),
            'any_added': any_added,
            'total_key_samples_before': int(total_key_samples_before),
            'total_key_samples_after': int(total_key_samples_after),
            'soc_paths_before': int(soc_paths_before),
            'soc_paths_after': int(soc_paths_after),
            'soc_violated_count': int(soc_violated_count),
            'soc_max_violation': float(soc_max_violation),
            'soc_added_paths': int(len(added_soc_indices)),
            'soc_added_indices': [int(i) for i in added_soc_indices],
            'ramp_added_indices': {
                f'{t}:{g}': [int(i) for i in indices]
                for (t, g), indices in added_indices_by_constraint.items()
            },
            'iter_time_s': float(iter_time)
        }
        if debug_log:
            iter_log_entry['per_constraint_added_sample_counts'] = per_constraint_added
        iteration_log.append(iter_log_entry)

        print(
            f"[Iter {it}] obj={final_solution['obj']:.4f} "
            f"max_rho={max_rho:.3e} ramp_violated={violated_count} "
            f"ramp_added={total_key_samples_after-total_key_samples_before} "
            f"soc_max={soc_max_violation:.3e} soc_violated={soc_violated_count} "
            f"soc_added={len(added_soc_indices)} time={iter_time:.1f}s"
        )

        if not any_added:
            print(f"[EIFICA] converged at iter {it}: no omitted ramp/SOC trajectory violation.")
            break
        if it >= max_iter:
            print(f"[EIFICA] WARNING: NOT fully converged after max_iter={max_iter}; remaining violations may exist.")
            break

        # Constraint generation: keep the same model and append only newly violated ramp scenarios.
        # SOC, line-flow, Pmax/Pmin, variables, and objective remain in the model and are not rebuilt.
        while any_added and it < max_iter:
            appended_samples = sum(len(v) for v in added_indices_by_constraint.values())
            appended_constraints = 0
            for (t, g), idx_add in added_indices_by_constraint.items():
                if len(idx_add) == 0:
                    continue
                alpha_t = gen_alpha_all[t, g]
                alpha_t_minus_1 = gen_alpha_all[t - 1, g]
                delta_P = gen_power_all[t, g] - gen_power_all[t - 1, g]
                prob.addConstr(
                    gen_ramp_rate[g] - delta_P
                    + alpha_t * total_RE_error_train[idx_add, t]
                    - alpha_t_minus_1 * total_RE_error_train[idx_add, t - 1]
                    >= s - r[idx_add]
                )
                prob.addConstr(
                    gen_ramp_rate[g] + delta_P
                    - alpha_t * total_RE_error_train[idx_add, t]
                    + alpha_t_minus_1 * total_RE_error_train[idx_add, t - 1]
                    >= s - r[idx_add]
                )
                appended_constraints += 2 * len(idx_add)

            it += 1
            iter_time_start = time.time()
            print(
                f"[EIFICA] appended {appended_samples} ramp samples "
                f"({appended_constraints} constraints)"
            )
            appended_soc_paths = len(added_soc_indices)
            appended_soc_constraints = 0
            if soc_constraint_generation and appended_soc_paths > 0:
                for sample_idx in added_soc_indices:
                    sample_idx = int(sample_idx)
                    for e in range(num_ESS):
                        cap_e = float(ESS_capacity[e])
                        agc_soc_shift = 0.0
                        for tt in range(1, T + 1):
                            agc_soc_shift = agc_soc_shift + (
                                ESS_beta_all[tt - 1, e]
                                * float(total_RE_error_train[sample_idx, tt - 1])
                                / cap_e
                            )
                            soc_actual = ESS_SOC[tt, e] + agc_soc_shift
                            prob.addConstr(soc_actual >= ESS_SOC_min)
                            prob.addConstr(soc_actual <= ESS_SOC_max)
                            cumulative_soc_constr_count += 2
                            appended_soc_constraints += 2
            print(
                f"[EIFICA] appended {appended_soc_paths} SOC paths "
                f"({appended_soc_constraints} constraints); re-optimizing iter {it}"
            )
            if reopt_method is not None:
                prob.setParam('Method', reopt_method)
            if reopt_lp_warm_start is not None:
                prob.setParam('LPWarmStart', reopt_lp_warm_start)
            prob.optimize()

            if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
                raise ValueError(f'Iter {it}: Solver failed with status {prob.status}')

            sol_gen_power = gen_power_all.X.copy()
            sol_gen_alpha = gen_alpha_all.X.copy()
            sol_s = float(np.asarray(s.X).reshape(-1)[0])
            sol_ess_beta = ESS_beta_all.X.copy() if (num_ESS > 0 and ESS_beta_all is not None) else None
            sol_ess_plan = ESS_plan_all.X.copy() if (num_ESS > 0 and ESS_plan_all is not None) else None
            sol_ess_discharge = ESS_discharge.X.copy() if (num_ESS > 0 and ESS_discharge is not None) else None
            sol_ess_charge = ESS_charge.X.copy() if (num_ESS > 0 and ESS_charge is not None) else None
            sol_ess_soc = ESS_SOC.X.copy() if (num_ESS > 0 and ESS_SOC is not None) else None
            final_solution = {'gen_power_all': sol_gen_power, 'gen_alpha_all': sol_gen_alpha,
                              'ess_beta_all': sol_ess_beta, 'ess_plan_all': sol_ess_plan,
                              'ess_discharge_all': sol_ess_discharge,
                              'ess_charge_all': sol_ess_charge,
                              'ess_soc': sol_ess_soc, 'obj': prob.objVal, 's_value': sol_s}
            final_prob = prob

            any_added = False
            per_constraint_added = {}
            added_indices_by_constraint = {}
            max_rho = 0.0
            violated_count = 0
            total_key_samples_before = sum(int(mask.sum()) for mask in ramp_active_masks.values())
            if gen_ramp_rate is not None:
                for (t, g) in t_g_list:
                    active_mask = ramp_active_masks.get((t, g))
                    if active_mask is None:
                        active_mask = np.zeros(N_WDR, dtype=bool)
                        ramp_active_masks[(t, g)] = active_mask
                    omitted_idx = N_WDR_indices[~active_mask]
                    if omitted_idx.size == 0:
                        per_constraint_added[(t, g)] = 0
                        continue

                    delta_P = sol_gen_power[t, g] - sol_gen_power[t - 1, g]
                    ramp_term = (
                        sol_gen_alpha[t, g] * total_RE_error_train[omitted_idx, t]
                        - sol_gen_alpha[t - 1, g] * total_RE_error_train[omitted_idx, t - 1]
                    )
                    lhs_upper = gen_ramp_rate[g] - delta_P + ramp_term
                    lhs_lower = gen_ramp_rate[g] + delta_P - ramp_term
                    rho = sol_s - np.minimum(lhs_upper, lhs_lower)
                    if rho.size > 0:
                        max_rho = max(max_rho, float(np.max(rho)))

                    violated_local = np.where(rho > tol)[0]
                    violated_count += int(violated_local.size)
                    if violated_local.size == 0:
                        per_constraint_added[(t, g)] = 0
                        continue

                    order = violated_local[np.argsort(-rho[violated_local])]
                    idx_add = omitted_idx[order[:m1]]
                    idx_add = idx_add[~active_mask[idx_add]]
                    added_count = int(idx_add.size)
                    per_constraint_added[(t, g)] = added_count
                    if added_count > 0:
                        any_added = True
                        added_indices_by_constraint[(t, g)] = np.asarray(idx_add, dtype=int)
                        active_mask[idx_add] = True
                        key_sets[(t, g)] = np.flatnonzero(active_mask)
            total_key_samples_after = sum(int(mask.sum()) for mask in ramp_active_masks.values())

            soc_paths_before = len(soc_key_set)
            added_soc_indices = np.array([], dtype=int)
            soc_violated_count = 0
            soc_max_violation = 0.0
            if soc_constraint_generation:
                added_soc_indices, soc_violated_count, soc_max_violation = find_soc_violated_paths(sol_ess_beta, sol_ess_soc)
                if len(added_soc_indices) > 0:
                    any_added = True
                    soc_key_set.update(int(i) for i in added_soc_indices.tolist())
            soc_paths_after = len(soc_key_set)

            iter_time = time.time() - iter_time_start
            iter_log_entry = {
                'iter': it,
                'obj': final_solution['obj'],
                's_value': float(sol_s),
                'max_rho': float(max_rho),
                'violated_count': int(violated_count),
                'any_added': any_added,
                'total_key_samples_before': int(total_key_samples_before),
                'total_key_samples_after': int(total_key_samples_after),
                'soc_paths_before': int(soc_paths_before),
                'soc_paths_after': int(soc_paths_after),
                'soc_violated_count': int(soc_violated_count),
                'soc_max_violation': float(soc_max_violation),
                'soc_added_paths': int(len(added_soc_indices)),
                'soc_added_indices': [int(i) for i in added_soc_indices],
                'ramp_added_indices': {
                    f'{t}:{g}': [int(i) for i in indices]
                    for (t, g), indices in added_indices_by_constraint.items()
                },
                'iter_time_s': float(iter_time),
                'appended_ramp_samples': int(appended_samples),
                'appended_ramp_constraints': int(appended_constraints),
                'appended_soc_paths': int(appended_soc_paths),
                'appended_soc_constraints': int(appended_soc_constraints),
            }
            if debug_log:
                iter_log_entry['per_constraint_added_sample_counts'] = per_constraint_added
            iteration_log.append(iter_log_entry)
            print(
                f"[Iter {it}] obj={final_solution['obj']:.4f} "
                f"max_rho={max_rho:.3e} ramp_violated={violated_count} "
                f"ramp_added={total_key_samples_after-total_key_samples_before} "
                f"soc_max={soc_max_violation:.3e} soc_violated={soc_violated_count} "
                f"soc_added={len(added_soc_indices)} time={iter_time:.1f}s"
            )

            if not any_added:
                print(f"[EIFICA] converged at iter {it}: no omitted ramp/SOC trajectory violation.")
                break

        if any_added and it >= max_iter:
            print(f"[EIFICA] WARNING: NOT fully converged after max_iter={max_iter}; remaining violations may exist.")
        break

    solve_time = time.time() - t_start_total
    results = {
        'prob': final_prob,
        'gen_power_all': final_solution['gen_power_all'],
        'gen_alpha_all': final_solution['gen_alpha_all'],
        'ess_beta_all':  final_solution['ess_beta_all'],
        'ess_plan_all':  final_solution['ess_plan_all'],
        'ess_discharge_all': final_solution['ess_discharge_all'],
        'ess_charge_all': final_solution['ess_charge_all'],
        'ess_soc':       final_solution['ess_soc'],
        's_value': float(final_solution['s_value']),
        'obj_value': float(final_solution['obj']),
        'solve_time': float(solve_time),
        'iteration_log': iteration_log,
        'key_sets': key_sets,
        'status': final_prob.status if final_prob is not None else None,
        'method': method,
        'epsilon': epsilon,
        'theta': theta,
        'N_WDR': N_WDR,
        'enforce_cumulative_soc': bool(enforce_cumulative_soc),
        'soc_constraint_generation': bool(soc_constraint_generation),
        'soc_key_set': np.asarray(sorted(soc_key_set), dtype=int) if soc_constraint_generation else None,
        'soc_m0': int(m0_soc),
        'soc_m1': int(m1_soc),
        'soc_score_lambda': float(soc_score_lambda),
        'initialization_mode': initialization_mode,
        'initial_soc_indices': initial_soc_key_set,
        'soc_initial_scores': soc_scores,
        'initial_ramp_key_count': int(initial_ramp_key_count),
        'training_sample_indices': random_var_scenario_index,
        'line_block_build_time_s': float(line_block_build_time_s),
        'line_block_build_total_s': float(line_block_build_total_s),
        'ess_terminal_soc': ESS_terminal_soc,
        'cumulative_soc_constr_count': int(cumulative_soc_constr_count) if 'cumulative_soc_constr_count' in locals() else 0
    }
    return results

def solve_PD_instance(num_gen=38, num_WT=10, num_Solar=0, Tstart=0, norm_ord=1, T=24,
                     method='EIFICA', N_WDR=100, epsilon=0.05, theta=1.5e-1,
                     load_scaling_factor=1, solar_mode='auto', show_plot=True, seed=0,
                     time_limit=14400, MIPGap=0.01,
                     num_ESS=0, ESS_power_ratio=0.1, ESS_capacity_ratio=2.0,
                     ESS_eta_c=0.95, ESS_eta_d=0.95,
                     ESS_SOC_init=0.5, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
                     ESS_c_charge=5.0, ESS_c_discharge=5.0, ESS_lambda_AGC=0.0,
                     ESS_terminal_soc='minimum',
                     error_scale=1.0,
                     thread=4,
                     network_name='case24_ieee_rts',
                     m0_override=None, reserve_fraction_override=None,
                     log_file_name=None,
                     enforce_cumulative_soc=True,
                     soc_constraint_generation=True,
                     soc_m0_override=None, soc_m1_override=None,
                     soc_score_lambda=0.2,
                     plot_output_dir=None, plot_name_suffix='',
                     initialization_mode='envelope', reserve_ess_bus_draws=0):
    """
    Solve power dispatch problem with wind and solar integration.

    Parameters:
    -----------
    solar_mode : str, options: 'auto', 'night', 'day', 'full'
        Control solar integration behavior:
        - 'auto' (default): T=24 uses full cycle; T<24 uses nighttime window
        - 'night': Force nighttime window (minimal/no solar), good for T<24
        - 'day': Force daytime window (maximum solar), good for T<24
        - 'full': No adjustment, use Tstart as-is (for custom scenarios)
    show_plot : bool, default True
        Whether to display plots (set False for server/batch runs)

    RECOMMENDATION: For solar integration (num_Solar > 0):
        - Best: T=24 with solar_mode='auto' for complete day-night cycle
        - Alternative: T<24 with solar_mode='night' for pure wind baseline
                      T<24 with solar_mode='day' for peak solar scenarios
    """
    if abs(float(ESS_lambda_AGC)) > 1e-12:
        raise ValueError(
            "The manuscript objective contains no AGC participation reward or "
            "penalty; ESS_lambda_AGC must be zero."
        )
    print(f"--- Calling solve_PD_instance with method={method}, N_WDR={N_WDR}, epsilon={epsilon}, time_limit={time_limit}, MIPGap={MIPGap} ---")
    N_samples_train = 1000 # the number of wind power scenarios used for training
    N_samples_test = 5000 # the number of wind power scenarios used for testing

    gurobi_seed = seed

    gen_cap_total_prop = 1  # scale the total generation capacity of the network data

    bigM =1e5 # this is only for "exact"
    # A caller may provide a per-case Gurobi log. None keeps normal runs quiet.
    #------------------

    network_dict = {'case118': ppnw.case118(),
                    'case300': ppnw.case300(),
                    'case24_ieee_rts': ppnw.case24_ieee_rts(),
                    'case5': ppnw.case5(),
                    'case4gs': ppnw.case4gs(),
                    'case_ieee30': ppnw.case_ieee30()}

    seed = gurobi_seed
    rng_system = np.random.RandomState(0)
    rng_fixed = np.random.RandomState(0)  # this is to avoid too much randomness that requires too many runs to have stable results
    rng_sample = np.random.RandomState(seed)

    # load network model
    network = network_dict[network_name]

    # load network load data
    load_location = DEFAULT_DATA_DIR / 'UK_norm_load_curve_highest.npy'
    network_load = np.load(load_location)
    # the network load is at half-hourly resolution, we need to average the consceutive time steps to get hourly resolution
    network_load = np.mean(np.vstack([network_load[::2],
                                      network_load[1::2]]), axis=0)
    # duplicate the network load to make it two days
    network_load = np.tile(network_load, 2)
    network_load = network_load[Tstart:Tstart+T]

    # -------------------------------------
    pp.rundcpp(network)
    _, ppci = _pd2ppc(network)
    bus_info = ppci['bus']
    branch_info = ppci['branch']
    PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                    using_sparse_solver=False)

    num_branch = len(branch_info)

    # get load info
    load_bus_size = bus_info[:, 2] * load_scaling_factor

    load_total = np.sum(load_bus_size)
    # we then get the load curves at all buses, using the network_load curve
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    ###### set generator capacity
    gen_cap_total = load_total * gen_cap_total_prop  # the total generation capacity
    gen_cap_individual = gen_cap_total / num_gen  # the individual generation capacity
    # add some randomness when assigning the generation capacity to each generator
    gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
    gen_pmin_individual = 0.1 * gen_cap_individual  # the individual minimum generation capacity.

    # Ramping rate: 50% of capacity per hour (can be adjusted)
    gen_ramp_rate = 0.6*gen_cap_individual  # 60% of capacity per hour

    # generator cost parameters
    gen_cost = rng_system.uniform(23.13, 57.03, num_gen)  # the cost of gas generators (USD/MWh)
    gen_cost_quadra = rng_system.uniform(0.002, 0.008, num_gen)  # the quadratic cost of gas generators (USD/MWh^2)
    # print("---------------------------------------------------------------------")



    # get generator locations
    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)

    if num_ESS > 0:
        ESS_total_power = load_total * ESS_power_ratio
        base_weights = np.array([0.35, 0.15, 0.25, 0.10, 0.15,
                                  0.30, 0.20, 0.25, 0.12, 0.18],
                                 dtype=float)[:num_ESS]
        raw_weights = base_weights / base_weights.sum()
        ESS_power_max = raw_weights * ESS_total_power

        cap_ratios = np.array([1.0, 4.0, 2.0, 3.0, 1.5,
                                2.5, 1.0, 3.5, 2.0, 4.0],
                               dtype=float)[:num_ESS]
        cap_ratios = cap_ratios * (float(ESS_capacity_ratio) / 2.0)
        ESS_capacity = ESS_power_max * cap_ratios  # (MWh)
        ESS_bus_list = rng_fixed.choice(bus_list, num_ESS, replace=True)
        print(f"\n=== ESS Integration ===")
        print(f"ESS units: {num_ESS}, Total power: {ESS_total_power:.2f} MW, "
              f"eta_c={ESS_eta_c}, eta_d={ESS_eta_d}")
        for e in range(num_ESS):
            print(f"  ESS {e}: P={ESS_power_max[e]:.2f} MW, "
                  f"E={ESS_capacity[e]:.2f} MWh ({raw_weights[e]*100:.1f}%, "
                  f"{cap_ratios[e]:.1f}h)")
    else:
        ESS_power_max = None
        ESS_capacity  = None
        ESS_bus_list  = None
        # Optional paired no-ESS control: consume the same placement draws as
        # the ESS case so wind/solar bus locations remain identical.
        if reserve_ess_bus_draws:
            rng_fixed.choice(bus_list, int(reserve_ess_bus_draws), replace=True)
    WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)
    Solar_bus_list = rng_fixed.choice(bus_list, num_Solar, replace=True) if num_Solar > 0 else None
    # if Solar_bus_list is not None:

    # get line info
    P_line_limit = np.abs(ppci['branch'][:, 5])  # the line flow limit
    # clip on 2 times of the total load to avoid numerical issues
    P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

    # === Wind scenario generation (or zero if num_WT=0) ===
    if num_WT > 0:
        WT_total = 0.4 * load_total  # Wind capacity (40% of total load)
        WT_individual = WT_total / num_WT
        # load the wind power scenarios, which is decomposed into prediction and error scenarios
        WT_pred, WT_error_scenarios, WT_full_scenarios = WT_sce_gen(num_WT, N_samples_train + N_samples_test)
        WT_pred = WT_pred[Tstart:Tstart+T] * WT_individual  # scale
        WT_error_scenarios = WT_error_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
        WT_full_scenarios = WT_full_scenarios[:, Tstart:Tstart+T] * WT_individual  # scale
        # generate training and testing scenarios
        WT_error_scenarios_train = WT_error_scenarios[:N_samples_train] * error_scale
        WT_error_scenarios_test  = WT_error_scenarios[N_samples_train:] * error_scale
        if error_scale != 1.0:
            print(f"[error_scale={error_scale}] WT error amplified x{error_scale}")
    else:
        # No wind: create zero arrays with minimal dimension for compatibility
        print("\n=== Wind Power Disabled (num_WT=0) ===")
        num_WT = 1  # Use 1 dummy wind bus for array compatibility
        WT_bus_list = np.array([0])  # Dummy bus
        WT_pred = np.zeros((T, num_WT))
        WT_error_scenarios_train = np.zeros((N_samples_train, T, num_WT))
        WT_error_scenarios_test = np.zeros((N_samples_test, T, num_WT))

    # === Solar scenario generation ===
    if num_Solar > 0:
        Solar_total = 0.45 * load_total  # Solar capacity (45% of total load)
        Solar_individual = Solar_total / num_Solar

        # Generate solar scenarios
        Solar_pred, Solar_error_scenarios, Solar_full_scenarios = Solar_sce_gen(num_Solar, N_samples_train + N_samples_test)
        Solar_pred = Solar_pred[Tstart:Tstart+T] * Solar_individual  # scale
        Solar_error_scenarios = Solar_error_scenarios[:, Tstart:Tstart+T] * Solar_individual  # scale
        Solar_full_scenarios = Solar_full_scenarios[:, Tstart:Tstart+T] * Solar_individual  # scale

        # Apply daylight mask: only 6:00-18:00 have solar generation
        for t in range(T):
            hour_of_day = (Tstart + t) % 24  # Calculate actual hour
            if hour_of_day < 6 or hour_of_day >= 18:  # Nighttime
                Solar_pred[t, :] = 0
                Solar_error_scenarios[:, t, :] = 0
                Solar_full_scenarios[:, t, :] = 0

        # Generate training and testing scenarios
        Solar_error_scenarios_train = Solar_error_scenarios[:N_samples_train] * error_scale
        Solar_error_scenarios_test  = Solar_error_scenarios[N_samples_train:] * error_scale
        if error_scale != 1.0:
            print(f"[error_scale={error_scale}] Solar error amplified x{error_scale}")

        print(f"\n=== Solar Integration ===")
        print(f"Solar total capacity: {Solar_total:.2f} MW ({0.45*100:.0f}% of load)")
        print(f"Individual solar station: {Solar_individual:.2f} MW")
        print(f"Daylight hours: 6:00-18:00 (time window: {Tstart}-{Tstart+T})")
    else:
        Solar_pred = None
        Solar_error_scenarios_train = None
        Solar_error_scenarios_test = None
        Solar_bus_list = None
    # perform SUC
    input_param_dict = {'T': T, 'num_gen': num_gen, 'num_WT': num_WT, 'num_branch': num_branch,
                        'load_bus_all': load_bus_all, 'PTDF': PTDF, 'gen_cap_individual': gen_cap_individual,
                        'gen_pmin_individual': gen_pmin_individual, 'WT_pred': WT_pred,
                        'WT_error_scenarios_train': WT_error_scenarios_train, 'P_line_limit': P_line_limit,
                        'gen_bus_list': gen_bus_list, 'WT_bus_list': WT_bus_list, 'N_WDR': N_WDR, 'epsilon': epsilon,
                        'thread': thread,
                        'theta': theta, 'method': method, 'MIPGap': MIPGap, 'gen_cost': gen_cost,
                        'gen_cost_quadra': gen_cost_quadra, 'bigM': bigM, 'gurobi_seed': gurobi_seed,
                        'log_file_name': log_file_name, 'rng': rng_sample, "norm_ord": norm_ord,
                        'num_Solar': num_Solar, 'Solar_pred': Solar_pred,
                        'Solar_error_scenarios_train': Solar_error_scenarios_train, 'Solar_bus_list': Solar_bus_list,
                        'gen_ramp_rate': gen_ramp_rate, 'time_limit': time_limit,
                        'num_ESS': num_ESS, 'ESS_power_max': ESS_power_max,
                        'ESS_bus_list': ESS_bus_list,
                        'ESS_capacity': ESS_capacity,
                        'ESS_eta_c': ESS_eta_c, 'ESS_eta_d': ESS_eta_d,
                        'ESS_SOC_init': ESS_SOC_init,
                        'ESS_SOC_min': ESS_SOC_min, 'ESS_SOC_max': ESS_SOC_max,
                        'ESS_c_charge': ESS_c_charge, 'ESS_c_discharge': ESS_c_discharge,
                        'ESS_lambda_AGC': ESS_lambda_AGC,
                        'ESS_terminal_soc': ESS_terminal_soc,
                        'm0_override': m0_override, 'reserve_fraction_override': reserve_fraction_override,
                        'enforce_cumulative_soc': enforce_cumulative_soc,
                        'soc_constraint_generation': soc_constraint_generation,
                        'soc_m0_override': soc_m0_override,
                        'soc_m1_override': soc_m1_override,
                        'soc_score_lambda': soc_score_lambda,
                        'initialization_mode': initialization_mode}
    solve_results = solve_PD(**input_param_dict)

    prob = solve_results['prob']
    gen_power_all = solve_results['gen_power_all']
    gen_alpha_all = solve_results['gen_alpha_all']
    ess_beta_all  = solve_results.get('ess_beta_all', None)
    ess_plan_all  = solve_results.get('ess_plan_all', None)
    ess_discharge_all = solve_results.get('ess_discharge_all', None)
    ess_charge_all = solve_results.get('ess_charge_all', None)
    ess_soc       = solve_results.get('ess_soc', None)

    if num_ESS > 0 and ess_beta_all is not None:
        beta_sum  = np.abs(ess_beta_all).sum(axis=-1)
        alpha_sum = np.abs(gen_alpha_all).sum(axis=-1)
        print(f"\n=== AGC participation ===")
        print(f"Mean ESS |beta|: {beta_sum.mean():.4f}, mean generator |alpha|: {alpha_sum.mean():.4f}")
        print(f"ESS share of absolute participation: {beta_sum.mean() / (beta_sum.mean() + alpha_sum.mean()) * 100:.1f}%")
    if num_ESS > 0 and ess_plan_all is not None:
        print(f"\n=== Planned ESS output (discharge > 0, charge < 0) ===")
        for e in range(num_ESS):
            print(f"  ESS {e}: mean={ess_plan_all[:,e].mean():.2f} MW, "
                  f"range=[{ess_plan_all[:,e].min():.2f}, {ess_plan_all[:,e].max():.2f}]")
    if num_ESS > 0 and ess_soc is not None:
        print(f"\n=== ESS SOC ===" )
        for e in range(num_ESS):
            print(f"  ESS {e}: init={ess_soc[0,e]:.3f}, "
                  f"min={ess_soc[:,e].min():.3f}, max={ess_soc[:,e].max():.3f}, "
                  f"final={ess_soc[-1,e]:.3f}")

    # Check the status of the solution
    if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
        raise ValueError('The problem does not have a feasible solution.')

    t_solve = solve_results.get('solve_time', prob.Runtime)

    # test JCC satisfaction rate
    satisfied_rate, jcc_components = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list,
              Solar_pred, Solar_error_scenarios_test, Solar_bus_list, gen_ramp_rate,
              num_ESS=num_ESS, ESS_power_max=ESS_power_max, ESS_beta_all=ess_beta_all,
              ESS_plan_all=ess_plan_all, ESS_capacity=ESS_capacity,
              ESS_SOC_init=ESS_SOC_init, ESS_SOC_min=ESS_SOC_min, ESS_SOC_max=ESS_SOC_max,
              ESS_SOC_all=ess_soc, ESS_bus_list=ESS_bus_list, return_details=True)

    planned_renewable = WT_pred.sum(axis=1)
    if Solar_pred is not None:
        planned_renewable = planned_renewable + Solar_pred.sum(axis=1)
    planned_ess = np.zeros(T) if ess_plan_all is None else ess_plan_all.sum(axis=1)
    planned_balance_residual = (
        gen_power_all.sum(axis=1) + planned_renewable + planned_ess
        - load_bus_all.sum(axis=1)
    )
    beta_sum = np.zeros(T) if ess_beta_all is None else ess_beta_all.sum(axis=1)
    agc_balance_residual = gen_alpha_all.sum(axis=1) + beta_sum - 1.0
    balance_diagnostics = {
        'planned_max_abs_mw': float(np.max(np.abs(planned_balance_residual))),
        'agc_factor_max_abs': float(np.max(np.abs(agc_balance_residual))),
    }

    # print the total power from generators, total reserve from generators, total load
    print('------------------------------------')
    print(f'{network_name}, {num_gen} generators, {T}-step horizon')
    print(f'Risk level {epsilon}, radius {theta}, N_WDR {N_WDR}')
    print('')
    print(f'the objective value is {prob.objVal}, the out-of-sample JCC rate is {satisfied_rate*100}%')
    print(f'The method used is {method}')
    print(f'The computing time for solving the dispatch is {t_solve} seconds')
    print('')
    print('------------------------------------')
    # plot the results
    if show_plot:
        plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                      WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost,
                      Solar_pred, Solar_error_scenarios_test, gen_ramp_rate,
                      ess_beta_all=ess_beta_all, ess_plan_all=ess_plan_all, ess_soc=ess_soc,
                      ESS_capacity=ESS_capacity, ESS_SOC_min=ESS_SOC_min, ESS_SOC_max=ESS_SOC_max,
                      plot_output_dir=plot_output_dir, plot_name_suffix=plot_name_suffix)

    # Return comprehensive results dictionary
    results = {
        'prob': prob,
        'gen_power_all': gen_power_all,
        'gen_alpha_all': gen_alpha_all,
        'obj_value': prob.objVal,
        'solve_time': t_solve,  # Gurobi Runtime (seconds)
        'satisfied_rate': satisfied_rate,  # Out-of-sample JCC satisfaction rate
        'jcc_components': jcc_components,
        'balance_diagnostics': balance_diagnostics,
        'status': prob.status,
        'network_name': network_name,
        'gen_bus_list': gen_bus_list,
        'WT_bus_list': WT_bus_list,
        'Solar_bus_list': Solar_bus_list,
        'ESS_bus_list': ESS_bus_list,
        'num_gen': num_gen,
        'T': T,
        'method': method,
        'epsilon': epsilon,
        'theta': theta,
        'N_WDR': N_WDR,
        'ess_beta_all': ess_beta_all,
        'ess_plan_all': ess_plan_all,
        'ess_discharge_all': ess_discharge_all,
        'ess_charge_all': ess_charge_all,
        'ess_soc':      ess_soc,
        's_value': solve_results.get('s_value'),
        'iteration_log': solve_results.get('iteration_log', []),
        'key_sets': solve_results.get('key_sets', {}),
        'enforce_cumulative_soc': solve_results.get('enforce_cumulative_soc', False),
        'soc_constraint_generation': solve_results.get('soc_constraint_generation', False),
        'model_version': 'soc_physical_v2',
        'ess_terminal_soc': solve_results.get('ess_terminal_soc', ESS_terminal_soc),
        'soc_key_set': solve_results.get('soc_key_set'),
        'soc_m0': solve_results.get('soc_m0'),
        'soc_m1': solve_results.get('soc_m1'),
        'soc_score_lambda': solve_results.get('soc_score_lambda'),
        'initialization_mode': solve_results.get('initialization_mode'),
        'initial_soc_indices': solve_results.get('initial_soc_indices'),
        'soc_initial_scores': solve_results.get('soc_initial_scores'),
        'initial_ramp_key_count': solve_results.get('initial_ramp_key_count'),
        'training_sample_indices': solve_results.get('training_sample_indices'),
        'line_block_build_time_s': solve_results.get('line_block_build_time_s'),
        'line_block_build_total_s': solve_results.get('line_block_build_total_s'),
        'cumulative_soc_constr_count': solve_results.get('cumulative_soc_constr_count', 0)
    }
    return results

def plot_all_gen(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost):
    rng = np.random.RandomState(0)  # fixed random seed for reproducibility
    # pick 5 generators to plot, unless there are less than 5 generators
    num_plot_gen = min(5, num_gen)
    # pick random num_plot_gen from 60% generators with the smallest cost, unless there are less than num_plot_gen generators
    top_pick = max(int(0.6 * num_gen), num_plot_gen)
    plot_gen_index = rng.choice(np.argsort(gen_cap_individual)[:top_pick], num_plot_gen, replace=False)
    # make plot for three out-of-sample scenarios
    num_plot_sce = 3
    fig, axs = plt.subplots(num_plot_gen, num_plot_sce, figsize=(5*num_plot_sce, 2 * num_plot_gen))
    for i in range(3):
        ax = axs[:, i]
        for ig, g in enumerate(plot_gen_index):
            # plot the first-stage power output
            x = np.arange(T)
            ax[ig].step(x, gen_power_all[:, g], label='first-stage')
            # plot the actual power output
            ax[ig].step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * WT_error_scenarios_test[i].sum(axis=-1), label='actual')
            # set x-axis label
            ax[ig].set_xlabel('hour')
            # plot Pmin and Pmax as dashed lines
            ax[ig].axhline(gen_pmin_individual[g], color='black', linestyle='--')
            ax[ig].axhline(gen_cap_individual[g], color='black', linestyle='--')
            ax[ig].legend()
            ax[ig].set_title(f'scenario {i}, {method}, generator {g}, eps {epsilon}, theta {theta}')
    plt.tight_layout()
    # save figure to figure/test folder
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}.png')
    plt.savefig(save_name, dpi=300)
    plt.show()

def plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost,
                  Solar_pred=None, Solar_error_scenarios_test=None, gen_ramp_rate=None,
                  ess_beta_all=None, ess_plan_all=None, ess_soc=None,
                  ESS_capacity=None, ESS_SOC_min=0.1, ESS_SOC_max=0.9,
                  plot_output_dir=None, plot_name_suffix=''):
    # Select generators with largest AGC variation (by alpha std across time)
    alpha_std = np.std(gen_alpha_all, axis=0)  # std of alpha for each generator
    alpha_mean = np.mean(np.abs(gen_alpha_all), axis=0)

    print(f"\n=== All Generators AGC Statistics (sorted by |alpha| mean) ===")
    sorted_by_mean = np.argsort(alpha_mean)[::-1]
    print(f"{'Gen':>5} {'|alpha| mean':>14} {'alpha std':>12} {'alpha max':>12} {'nonzero hours':>15}")
    for g in sorted_by_mean:
        nonzero_hours = np.sum(np.abs(gen_alpha_all[:, g]) > 1e-4)
        print(f"{g:>5} {alpha_mean[g]:>14.6f} {alpha_std[g]:>12.6f} {np.max(np.abs(gen_alpha_all[:, g])):>12.6f} {nonzero_hours:>15}/{gen_alpha_all.shape[0]}")

    night_hours = list(range(0, 7))
    candidate_mask = np.array([
        np.sum(np.abs(gen_alpha_all[night_hours, g]) > 1e-4) >= 3
        for g in range(num_gen)
    ])
    candidates = np.where(candidate_mask)[0]
    if len(candidates) >= 2:
        plot_gen_index = candidates[np.argsort(alpha_mean[candidates])[-2:][::-1]]
        print(f"\n=== Selected generators with night-time AGC (t=0~6) ===")
    else:
        plot_gen_index = np.argsort(alpha_mean)[-2:][::-1]
        print(f"\n=== Selected generators with highest |alpha| mean (fallback) ===")
    print(f"Selected: Gen {plot_gen_index[0]} (|alpha| mean={alpha_mean[plot_gen_index[0]]:.6f}), "
          f"Gen {plot_gen_index[1]} (|alpha| mean={alpha_mean[plot_gen_index[1]]:.6f})")
    print(f"Gen {plot_gen_index[0]} alpha per hour: {np.round(gen_alpha_all[:, plot_gen_index[0]], 4).tolist()}")
    print(f"Gen {plot_gen_index[1]} alpha per hour: {np.round(gen_alpha_all[:, plot_gen_index[1]], 4).tolist()}")

    # Randomly select test scenario
    num_plot_sce = 1
    rng = np.random.RandomState(10)  # fixed random seed for reproducibility
    scenario_set = rng.choice(WT_error_scenarios_test.shape[0], num_plot_sce, replace=False)

    # Debug: Print scenario error statistics
    print(f"\n=== Selected Scenario (Random) ===")
    sce_i = scenario_set[0]
    print(f"Selected scenario index: {sce_i}")

    wt_error = WT_error_scenarios_test.sum(axis=-1)[sce_i]
    print(f"WT error in this scenario: mean={wt_error.mean():.4f}, std={wt_error.std():.4f}, range=[{wt_error.min():.4f}, {wt_error.max():.4f}]")

    # Calculate total error (wind only or wind+solar)
    total_error = wt_error
    if Solar_error_scenarios_test is not None:
        solar_error = Solar_error_scenarios_test.sum(axis=-1)[sce_i]
        print(f"Solar error in this scenario: mean={solar_error.mean():.4f}, std={solar_error.std():.4f}, range=[{solar_error.min():.4f}, {solar_error.max():.4f}]")
        total_error = total_error + solar_error
    print(f"Total RE error in this scenario: mean={total_error.mean():.4f}, std={total_error.std():.4f}, range=[{total_error.min():.4f}, {total_error.max():.4f}]")

    # print(f"\n=== Alpha Values for Plotted Generators ===")
    # for g in plot_gen_index:
    #     alpha_vals = gen_alpha_all[:, g]
    #     print(f"Gen {g} (Cost {gen_cost[g]:.2f}): alpha mean={alpha_vals.mean():.6f}, std={alpha_vals.std():.6f}, range=[{alpha_vals.min():.6f}, {alpha_vals.max():.6f}]")
    #     # Calculate actual adjustment
    #     adjustment = alpha_vals * total_error
    #     print(f"  -> Power adjustment: mean={adjustment.mean():.4f} MW, range=[{adjustment.min():.4f}, {adjustment.max():.4f}] MW")

    # print(f"\n=== All Generators Alpha Statistics ===")
    alpha_abs_sum = np.abs(gen_alpha_all).sum(axis=0)  # Sum of |alpha| for each generator
    alpha_std_all = np.std(gen_alpha_all, axis=0)  # Std of alpha for each generator
    non_zero_gens = np.where(alpha_abs_sum > 1e-6)[0]
    # print(f"Number of generators with non-zero alpha: {len(non_zero_gens)}/{num_gen}")

    # Print top 10 by std (to see variation)
    # print(f"\nGenerators ranked by alpha STD (top 10):")
    top_std_gens = np.argsort(alpha_std_all)[-10:][::-1]
    # for g in top_std_gens:
    #     alpha_vals = gen_alpha_all[:, g]
    #     print(f"  Gen {g} (Cost {gen_cost[g]:.2f}): std={alpha_std_all[g]:.6f}, mean={alpha_vals.mean():.6f}, range=[{alpha_vals.min():.4f}, {alpha_vals.max():.4f}]")

    # if len(non_zero_gens) > 0:
    #     print(f"\nGenerators with AGC assignment (top 10 by |alpha| sum):")
    #     top_agc_gens = np.argsort(alpha_abs_sum)[-10:][::-1]
    #     for g in top_agc_gens:
    #         if alpha_abs_sum[g] > 1e-6:
    #             alpha_vals = gen_alpha_all[:, g]
    #             print(f"  Gen {g} (Cost {gen_cost[g]:.2f}): |alpha| sum={alpha_abs_sum[g]:.6f}, mean={alpha_vals.mean():.6f}")

    # Determine number of rows: Wind + Solar (if present) + 3*generators (power + alpha + ramping) + ESS
    num_rows = 1  # Wind plot (always shown)
    if Solar_pred is not None:
        num_rows += 1  # Solar
    num_rows += len(plot_gen_index) * 3  # Generators (power + alpha + ramping)
    num_ESS_plot = ess_beta_all.shape[1] if ess_beta_all is not None else 0
    if num_ESS_plot > 0:
        num_rows += 1  # ESS β subplot
    if ess_plan_all is not None and num_ESS_plot > 0:
        num_rows += 1  # ESS planned output subplot
    if ess_soc is not None and num_ESS_plot > 0:
        num_rows += 1  # ESS SOC subplot
    if ess_plan_all is not None and ess_beta_all is not None and num_ESS_plot > 0:
        num_rows += 1  # ESS actual output (2nd-stage) subplot

    fig, axs = plt.subplots(num_rows, num_plot_sce, figsize=(8*num_plot_sce, 2.5* num_rows))
    if num_plot_sce <= 1:
        axs = axs[..., None]

    def _style_ax(ax, ylabel, title, T, ylim=None, grid_alpha=0.5):
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels(np.arange(T), fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=grid_alpha, color='lightgrey')
        if ylim is not None:
            ax.set_ylim(ylim)

    row_idx = 0

    # Plot wind forecast and actual wind power output
    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        ax_w = axs[row_idx, i]
        x = np.arange(T)
        ax_w.step(x, WT_pred.sum(axis=-1), label='forecast', color='steelblue', linewidth=1.5, where='post')
        ax_w.step(x, WT_pred.sum(axis=-1) + WT_error_scenarios_test.sum(axis=-1)[sce_i],
                  label='actual', color='darkorange', linewidth=1.5, where='post')
        _style_ax(ax_w, 'Wind (MW)', 'Wind Farm', T, grid_alpha=0.4)
        ax_w.legend(bbox_to_anchor=(1.0, 0.3), loc='lower right', fontsize=7, framealpha=0.8, prop={'weight': 'bold', 'size': 7})
    row_idx += 1

    # Plot solar forecast and actual solar power output (if present)
    if Solar_pred is not None and Solar_error_scenarios_test is not None:
        for i in range(num_plot_sce):
            sce_i = scenario_set[i]
            ax_s = axs[row_idx, i]
            x = np.arange(T)
            ax_s.step(x, Solar_pred.sum(axis=-1), label='forecast', color='steelblue', linewidth=1.5, where='post')
            ax_s.step(x, Solar_pred.sum(axis=-1) + Solar_error_scenarios_test.sum(axis=-1)[sce_i],
                      label='actual', color='darkorange', linewidth=1.5, where='post')
            _style_ax(ax_s, 'Solar (MW)', 'Solar Farm', T, grid_alpha=0.4)
            ax_s.legend(bbox_to_anchor=(1.0, 0.3), loc='lower right', fontsize=7, framealpha=0.8, prop={'weight': 'bold', 'size': 7})
        row_idx += 1

    # Plot generator power output
    gen_start_row = row_idx
    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3, i]
            x = np.arange(T)
            ax.step(x, gen_power_all[:, g], label='first-stage', color='steelblue', linewidth=1.5, where='post')
            total_RE_error_test = WT_error_scenarios_test.sum(axis=-1)[sce_i]
            if Solar_error_scenarios_test is not None:
                total_RE_error_test = total_RE_error_test + Solar_error_scenarios_test.sum(axis=-1)[sce_i]
            ax.step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * total_RE_error_test,
                    label='actual', color='darkorange', linewidth=1.5, where='post')
            ax.axhline(gen_pmin_individual[g], color='black', linestyle='--', linewidth=1.2)
            ax.axhline(gen_cap_individual[g], color='black', linestyle='--', linewidth=1.2)
            _style_ax(ax, 'Gen (MW)', f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh', T, grid_alpha=0.4)
            ax.legend(bbox_to_anchor=(0.95, 0.15), loc='lower right', fontsize=7, framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot AGC alpha bars
    for i in range(num_plot_sce):
        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3 + 1, i]
            x = np.arange(T) + 0.5
            ax.bar(x, gen_alpha_all[:, g], width=0.75, label='AGC',
                   color='#E879F9', align='center', edgecolor='none')
            ax.axhline(0, color='black', linewidth=0.8)
            _style_ax(ax, 'AGC Factor', f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh', T,
                      ylim=(-1, 1), grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(1.0, 0.05), loc='lower right', fontsize=7, framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot ramping bars
    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        total_RE_error_test = WT_error_scenarios_test.sum(axis=-1)[sce_i]
        if Solar_error_scenarios_test is not None:
            total_RE_error_test = total_RE_error_test + Solar_error_scenarios_test.sum(axis=-1)[sce_i]
        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3 + 2, i]
            first_stage_ramp = np.diff(gen_power_all[:, g])
            actual_power = gen_power_all[:, g] - gen_alpha_all[:, g] * total_RE_error_test
            second_stage_ramp = np.diff(actual_power)
            x = np.arange(1, T) - 0.5
            width = 0.38
            ax.bar(x - width/2, first_stage_ramp, width, label='First-stage',
                   color='steelblue', alpha=0.85, edgecolor='none')
            ax.bar(x + width/2, second_stage_ramp, width, label='Actual',
                   color='darkorange', alpha=0.85, edgecolor='none')
            ramp_limit = gen_ramp_rate[g] if gen_ramp_rate is not None else gen_cap_individual[g]
            ax.axhline(ramp_limit, color='red', linestyle='--', linewidth=1.5,
                       label=f'Limit (\u00b1{ramp_limit:.1f} MW)')
            ax.axhline(-ramp_limit, color='red', linestyle='--', linewidth=1.5)
            _style_ax(ax, 'Ramp (MW/h)',
                      f'Gen {g} Ramping, Cap={gen_cap_individual[g]:.1f} MW', T,
                      ylim=(-ramp_limit * 1.25, ramp_limit * 1.25), grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(0.85, 0.08), loc='lower right', fontsize=7, framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot ESS AGC beta bars/lines (all ESS units on one subplot)
    if ess_beta_all is not None and num_ESS_plot > 0:
        ess_row = gen_start_row + len(plot_gen_index) * 3
        colors_ess = plt.cm.tab10(np.linspace(0, 0.9, num_ESS_plot))
        for i in range(num_plot_sce):
            ax = axs[ess_row, i]
            x = np.arange(T)
            if num_ESS_plot <= 6:
                width_ess = 0.7 / num_ESS_plot
                for e in range(num_ESS_plot):
                    offset = (e - (num_ESS_plot - 1) / 2) * width_ess
                    ax.bar(x + 0.5 + offset, ess_beta_all[:, e], width_ess,
                           label=f'ESS {e}', color=colors_ess[e], alpha=0.85, edgecolor='none')
            else:
                for e in range(num_ESS_plot):
                    ax.step(x, ess_beta_all[:, e], where='post',
                            label=f'ESS {e}', color=colors_ess[e], linewidth=1.2, alpha=0.85)
            ax.axhline(0, color='black', linewidth=0.8)
            _style_ax(ax, 'ESS β', f'ESS AGC Factor ({num_ESS_plot} units)', T,
                      ylim=(-1.1, 1.1), grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(1.0, 0.05), loc='lower right', fontsize=7,
                      framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot ESS planned output
    if ess_plan_all is not None and num_ESS_plot > 0:
        plan_row = ess_row + 1 if ess_beta_all is not None else gen_start_row + len(plot_gen_index) * 3
        colors_ess = plt.cm.tab10(np.linspace(0, 0.9, num_ESS_plot))
        for i in range(num_plot_sce):
            ax = axs[plan_row, i]
            x = np.arange(T)
            for e in range(num_ESS_plot):
                ax.step(x, ess_plan_all[:, e], where='post',
                        label=f'ESS {e}', color=colors_ess[e], linewidth=1.5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            _style_ax(ax, 'Plan (MW)', f'ESS Planned Output ({num_ESS_plot} units, +discharge -charge)', T,
                      grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(1.0, 0.05), loc='lower right', fontsize=7,
                      framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot ESS SOC
    if ess_soc is not None and num_ESS_plot > 0:
        soc_row = plan_row + 1 if ess_plan_all is not None else (
                  ess_row + 1 if ess_beta_all is not None else gen_start_row + len(plot_gen_index) * 3)
        colors_ess = plt.cm.tab10(np.linspace(0, 0.9, num_ESS_plot))
        for i in range(num_plot_sce):
            sce_i = scenario_set[i]
            ax = axs[soc_row, i]
            x = np.arange(T + 1)
            actual_soc = None
            if ess_beta_all is not None and ESS_capacity is not None:
                xi = WT_error_scenarios_test.sum(axis=-1)[sce_i]
                if Solar_error_scenarios_test is not None:
                    xi = xi + Solar_error_scenarios_test.sum(axis=-1)[sce_i]
                actual_soc = np.array(ess_soc, copy=True)
                for e in range(num_ESS_plot):
                    cumulative_shift = np.zeros(T + 1)
                    cumulative_shift[1:] = np.cumsum(
                        ess_beta_all[:, e] * xi / ESS_capacity[e]
                    )
                    actual_soc[:, e] = ess_soc[:, e] + cumulative_shift

            for e in range(num_ESS_plot):
                ax.plot(x, ess_soc[:, e], color=colors_ess[e],
                        linewidth=1.5, marker='o', markersize=2, label=f'ESS {e} plan')
                if actual_soc is not None:
                    ax.plot(x, actual_soc[:, e], color=colors_ess[e],
                            linewidth=1.2, linestyle='--', alpha=0.9, label=f'ESS {e} actual')
            ax.axhline(ESS_SOC_max, color='red',  linestyle='--', linewidth=1.0, alpha=0.6, label='SOC max')
            ax.axhline(ESS_SOC_min, color='blue', linestyle='--', linewidth=1.0, alpha=0.6, label='SOC min')
            if actual_soc is not None:
                actual_min = float(np.min(actual_soc))
                actual_max = float(np.max(actual_soc))
                actual_ok = (actual_min >= ESS_SOC_min - 1e-8) and (actual_max <= ESS_SOC_max + 1e-8)
                print(f"ESS actual SOC in scenario {sce_i}: range=[{actual_min:.4f}, {actual_max:.4f}], feasible={actual_ok}")
            _style_ax(ax, 'SOC', f'ESS SOC Plan vs Actual ({num_ESS_plot} units, scenario {sce_i})', T,
                      ylim=(-0.05, 1.05), grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(1.0, 0.05), loc='lower right', fontsize=5,
                      framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # Plot ESS 2nd-stage actual output for the selected scenario
    # actual = P_plan - beta * xi_t
    if ess_plan_all is not None and ess_beta_all is not None and num_ESS_plot > 0:
        actual_row = soc_row + 1
        colors_ess = plt.cm.tab10(np.linspace(0, 0.9, num_ESS_plot))
        for i in range(num_plot_sce):
            sce_i = scenario_set[i]
            # total RE error for this scenario (T,)
            xi = WT_error_scenarios_test.sum(axis=-1)[sce_i]  # (T,)
            if Solar_error_scenarios_test is not None:
                xi = xi + Solar_error_scenarios_test.sum(axis=-1)[sce_i]
            ax = axs[actual_row, i]
            x = np.arange(T)
            for e in range(num_ESS_plot):
                actual = ess_plan_all[:, e] - ess_beta_all[:, e] * xi  # (T,)
                ax.step(x, actual, where='post',
                        label=f'ESS {e}', color=colors_ess[e], linewidth=1.5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            _style_ax(ax, 'Actual (MW)',
                      f'ESS 2nd-Stage Actual Output ({num_ESS_plot} units, scenario {sce_i})', T,
                      grid_alpha=0.5)
            ax.legend(bbox_to_anchor=(1.0, 0.05), loc='lower right', fontsize=7,
                      framealpha=0.8, prop={'weight': 'bold', 'size': 7})

    # adjust margins
    plt.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.03, hspace=0.39, wspace=0.3)
    # save figure to output folder
    save_dir = plot_output_dir or os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_{plot_name_suffix}" if plot_name_suffix else ""
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}{suffix}_{timestamp}.pdf')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f'[plot_paper] Figure saved to: {save_name}')
    plt.show()

if __name__ == '__main__':
    # ===== Configuration =====
    # Method: EIFICA, FICA, CVAR, or ExactLHS
    method = 'EIFICA'
    #method = 'FICA'
    #method = 'CVAR'
    #method = 'ExactLHS'

    # Scenario parameters
    N_WDR =80 # Number of training scenarios
    epsilon = 0.03  # Risk level (violation probability)
    theta = 0.06  # Wasserstein radius

    # Network
    network_name = 'case24_ieee_rts'

    # System parameters
    num_gen = 38  # Number of generators
    num_WT = 10  # Number of wind turbines (set to 0 for solar-only)
    num_Solar = 5  # Number of solar stations (set to 0 for wind-only)
    num_ESS = 6         # Number of ESS units (set to 0 to disable)
    ESS_power_ratio = 0.1     # Total ESS power as fraction of total load
    ESS_capacity_ratio = 2.0  # Energy capacity = power * ratio (h), e.g. 2.0 = 2-hour storage
    ESS_eta_c = 0.95          # Charging efficiency
    ESS_eta_d = 0.95          # Discharging efficiency
    ESS_SOC_init = 0.5        # Initial SOC (50%)
    ESS_SOC_min  = 0.1        # Minimum SOC
    ESS_SOC_max  = 0.9        # Maximum SOC
    ESS_c_charge    = 5.0
    ESS_c_discharge = 5.0
    # The manuscript objective includes generation and storage throughput costs.
    ESS_lambda_AGC  = 0.0
    error_scale     = 1.0     # Error amplification factor (1.0=original, >1=larger RE errors)

    # Time parameters
    Tstart = 0  # Start time (will be auto-adjusted based on solar_mode)
    T = 24  # Time horizon in hours

    # Solar integration mode (only used when num_Solar > 0)
    # Options: 'auto', 'night', 'day', 'full'
    solar_mode = 'auto'
    # - 'auto': Recommended. T=24 uses full cycle; T<24 uses nighttime
    # - 'night': Force nighttime (pure wind baseline, no solar)
    # - 'day': Force daytime (maximum solar exposure)
    # - 'full': Use Tstart as-is (for custom scenarios)


    norm_ord = 1  # Norm order for Wasserstein distance
    load_scaling_factor = 1  # Load scaling factor
    seed = 0  # Random seed
    time_limit = 6000  # Gurobi time limit in seconds (e.g., 600 for 10 min)
    MIPGap = 0.01  # Gurobi MIPGap (e.g., 0.01 for 1%)
    show_plot = True  # Whether to display results plots
    # Run optimization (no log file will be generated)
    solve_PD_instance(
        num_gen=num_gen,
        num_WT=num_WT,
        num_Solar=num_Solar,
        Tstart=Tstart,
        norm_ord=norm_ord,
        T=T,
        method=method,
        N_WDR=N_WDR,
        epsilon=epsilon,
        theta=theta,
        load_scaling_factor=load_scaling_factor,
        solar_mode=solar_mode,
        seed=seed,
        time_limit=time_limit,
        MIPGap=MIPGap,
        show_plot=show_plot,
        num_ESS=num_ESS,
        ESS_power_ratio=ESS_power_ratio,
        ESS_capacity_ratio=ESS_capacity_ratio,
        ESS_eta_c=ESS_eta_c,
        ESS_eta_d=ESS_eta_d,
        ESS_SOC_init=ESS_SOC_init,
        ESS_SOC_min=ESS_SOC_min,
        ESS_SOC_max=ESS_SOC_max,
        ESS_c_charge=ESS_c_charge,
        ESS_c_discharge=ESS_c_discharge,
        ESS_lambda_AGC=ESS_lambda_AGC,
        error_scale=error_scale,
        network_name=network_name
    )
