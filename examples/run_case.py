"""Run an RTS-24 cumulative SOC instance."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eifica_optimizer import solve_PD_instance


if __name__ == "__main__":
    result = solve_PD_instance(
        network_name="case24_ieee_rts",
        num_gen=38,
        num_WT=10,
        num_Solar=5,
        num_ESS=6,
        T=24,
        method="EIFICA",
        N_WDR=100,
        epsilon=0.08,
        theta=0.12,
        seed=0,
        ESS_power_ratio=0.1,
        ESS_capacity_ratio=4.0,
        ESS_eta_c=0.95,
        ESS_eta_d=0.95,
        ESS_SOC_init=0.5,
        ESS_SOC_min=0.1,
        ESS_SOC_max=0.9,
        ESS_c_charge=50.0,
        ESS_c_discharge=50.0,
        ESS_lambda_AGC=0.0,
        ESS_terminal_soc="minimum",
        enforce_cumulative_soc=True,
        soc_constraint_generation=True,
        initialization_mode="envelope",
        error_scale=1.0,
        show_plot=False,
        thread=6,
        time_limit=3600,
        MIPGap=0.001,
    )
    print("status:", result.get("status"))
    print("objective:", result.get("obj_value"))
    print("solve time:", result.get("solve_time"))
