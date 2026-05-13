import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
import os
import gurobipy as gp
from gurobipy import GRB
# --------------------------------------------------------------------------
from WT_error_gen import WT_sce_gen
from Solar_error_gen import Solar_sce_gen
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
    """  
    Endpoint-envelope pre-screening (revised version).

    For each ramping constraint ``(t, g)``, compute the cross-period extremum
    score under the endpoint envelope correctly, then choose the top ``m0``
    sample indices as the initial critical-scenario set.
    """  
    N_WDR, T = total_RE_error_train.shape  
    key_sets = {}  
    for (t, g) in t_g_list:  
        a = total_RE_error_train[:, t]
        b = total_RE_error_train[:, t-1]
        
        
        s1 = np.abs(a - b)     # α(t)=+A, α(t-1)=-A
        s2 = np.abs(a + b)     # α(t)=+A, α(t-1)=+A
        s3 = np.abs(-a - b)    # α(t)=-A, α(t-1)=-A
        s4 = np.abs(-a + b)    # α(t)=-A, α(t-1)=+A
        
        scores = np.maximum.reduce([s1, s2, s3, s4])  
        
        m = min(m0, N_WDR)  
        if m < N_WDR:  
            idx_top = np.argpartition(-scores, m-1)[:m]  
            idx_top = idx_top[np.argsort(-scores[idx_top])]  
        else:  
            idx_top = np.arange(N_WDR, dtype=int)  
        key_sets[(t, g)] = np.asarray(idx_top, dtype=int)  
    return key_sets  
def check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test,
              P_line_limit, gen_bus_list, WT_bus_list, Solar_pred=None, Solar_error_scenarios_test=None, Solar_bus_list=None,
              gen_ramp_rate=None):
    # WT_error_scenarios_test has the shape of (N_samples_test, T, num_WT)
    # Solar_error_scenarios_test has the shape of (N_samples_test, T, num_Solar)

    # set small PTDF to zero to avoid numerical issues
    PTDF[np.abs(PTDF) < 1e-5] = 0
    PTDF_gen = PTDF[:, gen_bus_list].T
    PTDF_wind = PTDF[:, WT_bus_list].T
    PTDF_solar = PTDF[:, Solar_bus_list].T if Solar_bus_list is not None else None
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

    # Combine all constraints
    if len(R_res) > 0:
        res = np.vstack(P_res + L_res + R_res).T
    else:
        res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))
    
    P_res_arr = np.vstack(P_res).T
    L_res_arr = np.vstack(L_res).T
    P_satisfy = np.mean(np.all(P_res_arr, axis=1))
    L_satisfy = np.mean(np.all(L_res_arr, axis=1))
    print(f"[check_JCC diagnostics] Pmax/Pmin satisfaction: {P_satisfy*100:.1f}%, line-flow satisfaction: {L_satisfy*100:.1f}%", end="")
    if len(R_res) > 0:
        R_res_arr = np.vstack(R_res).T
        R_satisfy = np.mean(np.all(R_res_arr, axis=1))
        print(f", ramping satisfaction: {R_satisfy*100:.1f}%")
    else:
        print("")
    
    return satisfied_rate

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
              debug_log=False):  
    """  
    solve_PD with integrated envelope+iterative ramp scenario selection (EIFICA style)  
    Returns a result dictionary containing the solver object and iteration logs.
    Parameters are largely consistent with the original function; ``debug_log``
    enables more verbose diagnostic output.
    """  
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
    m1 = max(5, int(np.ceil(0.01 * N_WDR)))
    max_iter = 0
    tol = 1e-3  

    if method == 'EIFICA':
        print(f"[EIFICA] N_WDR={N_WDR}, epsilon={epsilon}, k={k}, m0={m0}, m1={m1}, max_iter={max_iter}")  
        key_sets = select_envelope_samples(total_RE_error_train, t_g_list, m0)  
    elif method == 'FICA':
        print(f"[FICA] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}
    else:
        print(f"[{method}] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}
  
    iteration_log = []  
  
    prev_alpha = None  
    final_solution = None  
    final_prob = None  
  
    for it in range(max_iter + 1):  
        iter_time_start = time.time()  
        prob = gp.Model('ED')  
        gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)  
        gen_alpha_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)  
  
        for t in range(T):  
            total_renewable = WT_pred[t, :].sum()  
            if Solar_pred is not None:  
                total_renewable = total_renewable + Solar_pred[t, :].sum()  
            prob.addConstr(gen_power_all[t, :].sum() + total_renewable == load_bus_all[t, :].sum())  
  
            if num_WT > 0:
                prob.addConstr(gen_alpha_all[t, :].sum() == 1)
            else:
                has_solar = (Solar_pred is not None) and (Solar_pred[t, :].sum() > 1e-6)
                if has_solar:
                    prob.addConstr(gen_alpha_all[t, :].sum() == 1)
                else:
                    prob.addConstr(gen_alpha_all[t, :] == 0)  
  
            prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)  
            prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)  
  
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
  
        t_l_list = [(t, l) for t in range(T) for l in range(num_branch)]  
        
        # 3.1) Line Max Flow Constraints
        for t, l in t_l_list:  
            num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
            b_Ax_wind = -PTDF_wind[l]  
            b_Ax_solar = -PTDF_solar[l] if (PTDF_solar is not None and num_Solar > 0) else np.zeros(0)  
            b_Ax_combined = np.concatenate([b_Ax_wind, b_Ax_solar])  
            b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_RE) + b_Ax_combined  

            line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T  
            line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T  
            line_flow_load = PTDF_load[l] @ load_bus_all[t]  
            if PTDF_solar is not None and Solar_error_scenarios_train is not None:  
                line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T  
                total_line_flow = line_flow_gen + line_flow_wind + line_flow_solar - line_flow_load
            else:
                total_line_flow = line_flow_gen + line_flow_wind - line_flow_load

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

            line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T  
            line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T  
            line_flow_load = PTDF_load[l] @ load_bus_all[t]  
            if PTDF_solar is not None and Solar_error_scenarios_train is not None:  
                line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T  
                total_line_flow = line_flow_gen + line_flow_wind + line_flow_solar - line_flow_load
            else:
                total_line_flow = line_flow_gen + line_flow_wind - line_flow_load

            if method == 'ExactLHS':
                bAx_list.append(b_Ax)
                prob.addConstr(-P_line_min[l] + total_line_flow + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                prob.addConstr(-P_line_min[l] + total_line_flow + bigM * z[N_WDR_indices] >= 0)
            else:
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))  
                prob.addConstr(-P_line_min[l] + total_line_flow >= s - r[N_WDR_indices])

        if gen_ramp_rate is not None:  
            ramp_constr_count = 0  
            for (t, g) in t_g_list:  
                if method == 'EIFICA':
                    sel_idx = key_sets.get((t, g), None)  
                    if sel_idx is None or len(sel_idx) == 0:  
                        sel_idx = N_WDR_indices  
                    sel_idx = np.asarray(sel_idx, dtype=int)
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

        # objective and solver params  
        FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2  
        prob.setObjective(FC.sum(), GRB.MINIMIZE)  
        
        print(f"Setting Gurobi Params: TimeLimit={time_limit}, MIPGap={MIPGap}")
        
        prob.setParam('MIPGap', MIPGap)  
        prob.setParam('Seed', gurobi_seed)  
        prob.setParam('Threads', thread)  
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
            raise ValueError(f'Iter {it}: Solver failed with status {prob.status}')  
  
        sol_gen_power = gen_power_all.X.copy()  
        sol_gen_alpha = gen_alpha_all.X.copy()  
        final_solution = {'gen_power_all': sol_gen_power, 'gen_alpha_all': sol_gen_alpha, 'obj': prob.objVal}  
        final_prob = prob
  
        if method != 'EIFICA':
            print(f"[{method}] optimization finished in one pass.")
            break

        if prev_alpha is None:  
            alpha_diff = np.inf  
        else:  
            alpha_diff = np.max(np.abs(sol_gen_alpha - prev_alpha))  
        prev_alpha = sol_gen_alpha.copy()  
  
        any_added = False  
        per_constraint_added = {}  
        total_key_samples_before = sum(len(v) for v in key_sets.values())  
        for (t, g) in t_g_list:  
            a = total_RE_error_train[:, t]  
            b = total_RE_error_train[:, t-1]  
            alpha_t_val = sol_gen_alpha[t, g]  
            alpha_t1_val = sol_gen_alpha[t-1, g]  
            s_all = alpha_t_val * a - alpha_t1_val * b  
            metric = np.abs(s_all)  
            m_add = min(m1, N_WDR)  
            if m_add < N_WDR:  
                idx_add = np.argpartition(-metric, m_add-1)[:m_add]  
            else:  
                idx_add = np.arange(N_WDR, dtype=int)  
            old_set = set(key_sets.get((t, g), np.array([], dtype=int)).tolist())  
            new_set = old_set.union(set(idx_add.tolist()))  
            added_count = len(new_set) - len(old_set)  
            per_constraint_added[(t, g)] = added_count  
            if added_count > 0:  
                any_added = True  
                key_sets[(t, g)] = np.array(sorted(new_set), dtype=int)  
        total_key_samples_after = sum(len(v) for v in key_sets.values())  
  
        iter_time = time.time() - iter_time_start  
        iter_log_entry = {  
            'iter': it,  
            'obj': final_solution['obj'],  
            'alpha_diff': float(alpha_diff),  
            'any_added': any_added,  
            'total_key_samples_before': int(total_key_samples_before),  
            'total_key_samples_after': int(total_key_samples_after),  
            'per_constraint_added_sample_counts': per_constraint_added,  
            'iter_time_s': float(iter_time)  
        }  
        iteration_log.append(iter_log_entry)  
  
        print(f"[Iter {it}] obj={final_solution['obj']:.4f} alpha_diff={alpha_diff:.3e} added_total={total_key_samples_after-total_key_samples_before} time={iter_time:.1f}s")  
  
        if (not any_added) or (alpha_diff < tol):  
            print(f"[EIFICA] converged at iter {it}")  
            break  
  
    solve_time = time.time() - t_start_total  
    results = {  
        'prob': final_prob,  
        'gen_power_all': final_solution['gen_power_all'],  
        'gen_alpha_all': final_solution['gen_alpha_all'],  
        'obj_value': float(final_solution['obj']),  
        'solve_time': float(solve_time),  
        'iteration_log': iteration_log,  
        'key_sets': key_sets,  
        'status': final_prob.status if final_prob is not None else None,  
        'method': method,  
        'epsilon': epsilon,  
        'theta': theta,  
        'N_WDR': N_WDR  
    }  
    return results

def solve_PD_instance(num_gen=38, num_WT=10, num_Solar=0, Tstart=0, norm_ord=1, T=24, 
                     method='EIFICA', N_WDR=100, epsilon=0.05, theta=1.5e-1, 
                     load_scaling_factor=1, solar_mode='auto', show_plot=True, seed=0,
                     time_limit=14400, MIPGap=0.01):
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
    print(f"--- Calling solve_PD_instance with method={method}, N_WDR={N_WDR}, epsilon={epsilon}, time_limit={time_limit}, MIPGap={MIPGap} ---")
    N_samples_train = 1000 # the number of wind power scenarios used for training
    N_samples_test = 5000 # the number of wind power scenarios used for testing
    thread = 4

    gurobi_seed = seed

    network_name = 'case24_ieee_rts' 
    gen_cap_total_prop = 1 # scale the total generation capacity of the network data

    bigM =1e5 # this is only for "exact"
    # Set log file name to None to disable logging
    log_file_name = None  # Disable logging for normal runs
    #------------------

    network_dict = {'case118': ppnw.case118(),
                    'case300': ppnw.case300(),
                    'case24_ieee_rts': ppnw.case24_ieee_rts(),
                    'case5': ppnw.case5(),
                    'case4gs': ppnw.case4gs(),
                    'case_ieee30': ppnw.case_ieee30()}

    seed = gurobi_seed
    rng = np.random.RandomState(seed)
    rng_fixed = np.random.RandomState(0)  # this is to avoid too much randomness that requires too many runs to have stable results

    # load network model
    network = network_dict[network_name]

    # load network load data
    load_location = os.path.join(os.getcwd(), 'data', 'processed', 'UK_norm_load_curve_highest.npy')
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
    gen_cost = rng.uniform(23.13, 57.03, num_gen)  # the cost of gas generators (USD/MWh)
    gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)  # the quadratic cost of gas generators (USD/MWh^2)
    # print("---------------------------------------------------------------------")
        
    
    
    # get generator locations
    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
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
        WT_error_scenarios_train = WT_error_scenarios[:N_samples_train]
        WT_error_scenarios_test = WT_error_scenarios[N_samples_train:]
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
        Solar_error_scenarios_train = Solar_error_scenarios[:N_samples_train]
        Solar_error_scenarios_test = Solar_error_scenarios[N_samples_train:]
        
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
                        'log_file_name': log_file_name, 'rng': rng, "norm_ord": norm_ord,
                        'num_Solar': num_Solar, 'Solar_pred': Solar_pred,
                        'Solar_error_scenarios_train': Solar_error_scenarios_train, 'Solar_bus_list': Solar_bus_list,
                        'gen_ramp_rate': gen_ramp_rate, 'time_limit': time_limit}
    solve_results = solve_PD(**input_param_dict)
    
    prob = solve_results['prob']
    gen_power_all = solve_results['gen_power_all']
    gen_alpha_all = solve_results['gen_alpha_all']
    
    # Extract the values if they are Gurobi variables
    if hasattr(gen_power_all, 'X'):
        gen_power_all = gen_power_all.X
    if hasattr(gen_alpha_all, 'X'):
        gen_alpha_all = gen_alpha_all.X
    
    # Check the status of the solution
    if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
        raise ValueError('The problem does not have a feasible solution.')

    t_solve = solve_results.get('solve_time', prob.Runtime)

    # test JCC satisfaction rate
    satisfied_rate = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list,
              Solar_pred, Solar_error_scenarios_test, Solar_bus_list, gen_ramp_rate)

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
                      Solar_pred, Solar_error_scenarios_test, gen_ramp_rate)

    # Return comprehensive results dictionary
    results = {
        'prob': prob,
        'gen_power_all': gen_power_all,
        'gen_alpha_all': gen_alpha_all,
        'obj_value': prob.objVal,
        'solve_time': t_solve,  # Gurobi Runtime (seconds)
        'satisfied_rate': satisfied_rate,  # Out-of-sample JCC satisfaction rate
        'status': prob.status,
        'network_name': network_name,
        'num_gen': num_gen,
        'T': T,
        'method': method,
        'epsilon': epsilon,
        'theta': theta,
        'N_WDR': N_WDR
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
                  Solar_pred=None, Solar_error_scenarios_test=None, gen_ramp_rate=None):
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
    
    # Determine number of rows: Wind + Solar (if present) + 3*generators (power + alpha + ramping)
    num_rows = 1  # Wind plot (always shown)
    if Solar_pred is not None:
        num_rows += 1  # Solar
    num_rows += len(plot_gen_index) * 3  # Generators (power + alpha + ramping)
    
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

    # adjust margins
    plt.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.03, hspace=0.39, wspace=0.3)
    # save figure to figure/test folder
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}_{timestamp}.pdf')
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f'[plot_paper] Figure saved to: {save_name}')
    plt.show()

if __name__ == '__main__':
    # ===== Configuration =====
    # Method: EIFICA, FICA, CVAR, or ExactLHS
    #method = 'EIFICA'
    method = 'FICA'
    #method = 'CVAR'
    #method = 'ExactLHS'
    
    # Scenario parameters
    N_WDR = 100 # Number of training scenarios
    epsilon = 0.03  # Risk level (violation probability)
    theta = 0.06  # Wasserstein radius
    
    # System parameters
    num_gen = 38  # Number of generators
    num_WT = 10  # Number of wind turbines (set to 0 for solar-only)
    num_Solar = 5# Number of solar stations (set to 0 for wind-only)
    
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
    time_limit = 600  # Gurobi time limit in seconds (e.g., 600 for 10 min)
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
        show_plot=show_plot
    )
