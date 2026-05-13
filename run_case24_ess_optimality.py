"""
Reduced case24 ESS optimality benchmark: EIFICA vs ExactLHS.

Compared with the wind-only optimality script, this version enables ESS,
uses T=3, a longer time limit, and writes outputs to
``case24_ess_optimality_results``.
"""
import pandas as pd
import numpy as np
import time
import os
import sys
from Ess import solve_PD_instance
from joblib import Parallel, delayed


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def worker(sf, eps, theta, n, seed, method, current_task, total_tasks, output_file):
    """Execute one parameter configuration for the benchmark."""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] [{current_task}/{total_tasks}] STARTING {method}: "
          f"SF={sf}, eps={eps}, theta={theta}, N={n}, Seed={seed}")

    start_time = time.time()
    try:
        res = solve_PD_instance(
            method=method,
            N_WDR=n,
            epsilon=eps,
            theta=theta,
            load_scaling_factor=sf,
            show_plot=False,
            num_WT=10,
            num_Solar=5,
            T=3,
            time_limit=7200,
            MIPGap=0.02,
            num_ESS=6,
            thread=4,
            seed=seed,
        )
        duration = time.time() - start_time
        result = {
            'method':         method,
            'sf':             sf,
            'epsilon':        eps,
            'theta':          theta,
            'n_wdr':          n,
            'seed':           seed,
            'obj':            res.get('obj_value',    res.get('min_cost (USD)', float('nan'))),
            'solve_time':     res.get('solve_time',   res.get('t_solve (s)',    float('nan'))),
            'total_duration': duration,
            'satisfied_rate': res.get('satisfied_rate', res.get('reliability_test (%)', float('nan'))),
            'status':         res.get('status', 'unknown'),
        }

        pd.DataFrame([result]).to_csv(
            output_file, mode='a',
            header=not os.path.exists(output_file),
            index=False
        )

        finish_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{finish_time}] [{current_task}/{total_tasks}] FINISHED {method} (Seed {seed}) "
              f"in {duration:.2f}s.  Obj: {result['obj']:.2f}, "
              f"Rate: {result['satisfied_rate']}")
        return result

    except Exception as e:
        error_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{error_time}] [{current_task}/{total_tasks}] !!! ERROR in {method} "
              f"(Seed {seed}): {str(e)}")
        err_log = os.path.join('case24_ess_optimality_results', 'error_details.log')
        with open(err_log, "a") as ef:
            ef.write(f"[{error_time}] SF={sf}, eps={eps}, theta={theta}, "
                     f"N={n}, method={method}, seed={seed} -> {str(e)}\n")
        return None


def run_experiment():
    SF_list    = [1.0, 1.5, 2.0]
    EPS_THETA  = [(0.03, 0.06), (0.06, 0.06)]
    N_list     = [50, 80, 100, 150, 200]
    METHODS    = ['EIFICA', 'ExactLHS']
    NUM_RUNS   = 10

    N_PARALLEL = 4

    output_dir  = 'case24_ess_optimality_results'
    output_file = os.path.join(output_dir, 'optimality_gap_main.csv')
    log_file    = os.path.join(output_dir, 'main_experiment.log')

    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = Logger(log_file)

    all_tasks = []
    for sf in SF_list:
        for (eps, theta) in EPS_THETA:
            for n in N_list:
                for run in range(NUM_RUNS):
                    for method in METHODS:
                        all_tasks.append((sf, eps, theta, n, run * 100, method))

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        print(f"Loaded {len(existing_df)} existing records; duplicate tasks will be skipped.")
        tasks_to_run = []
        for task in all_tasks:
            sf, eps, theta, n, seed, method = task
            mask = (
                (existing_df['sf']      == sf)     &
                (existing_df['epsilon'] == eps)    &
                (existing_df['theta']   == theta)  &
                (existing_df['n_wdr']   == n)      &
                (existing_df['seed']    == seed)   &
                (existing_df['method']  == method)
            )
            if not mask.any():
                tasks_to_run.append(task)
    else:
        tasks_to_run = all_tasks

    total_tasks = len(tasks_to_run)
    print(f"\n{'='*60}")
    print("ESS optimality comparison experiment  EIFICA vs ExactLHS")
    print(f"T=3  time_limit=7200s  num_ESS=6")
    print(f"N_list={N_list}  EPS_THETA={EPS_THETA}")
    print(f"Tasks to run: {total_tasks}  (parallel={N_PARALLEL})")
    print(f"{'='*60}\n")

    Parallel(n_jobs=N_PARALLEL)(
        delayed(worker)(sf, eps, theta, n, seed, method,
                        i + 1, total_tasks, output_file)
        for i, (sf, eps, theta, n, seed, method) in enumerate(tasks_to_run)
    )

    print(f"\nBatch experiment finished. {time.ctime()}")


if __name__ == "__main__":
    run_experiment()
