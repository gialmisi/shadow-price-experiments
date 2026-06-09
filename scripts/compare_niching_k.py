"""Sweep niching_k for the kth-nearest niching mechanism.

Compares best_so_far, HV, and shadow price error across different values of k
on a handful of representative problems.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from compare_niching import build_hv_context, compute_metrics, run_once
from utils import PROBLEM_BUILDERS

CONFIG_PATH = Path("experiment_config.yaml")
THRESHOLDS_DIR = Path("results/thresholds")
FRONTS_DIR = Path("results/fronts")


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH))
    operator_cfg = {
        "xover_probability": cfg["xover_probability"],
        "xover_distribution": cfg.get("experiment_xover_distribution", cfg["xover_distribution"]),
        "distribution_index": cfg.get("experiment_distribution_index", cfg["distribution_index"]),
        "tournament_size": cfg["tournament_size"],
        "mutation_probability": None,
    }
    mut_factor = float(cfg.get("experiment_mutation_probability_factor", 1))
    eps_percent = float(cfg["hv_eps_percent"])

    # Representative problems: 2 where kth helped, 2 where it didn't
    test_problems = ["g6", "g9", "g8", "branin", "pressure_vessel"]
    ct_level = "med"
    pop_size = 60
    n_generations = 200
    n_runs = 20
    base_seed = int(cfg["base_seed"])

    k_values = [1, 2, 3, 5, 7, 10]

    print(f"Config: ct_level={ct_level}, pop_size={pop_size}, n_gen={n_generations}, n_runs={n_runs}")
    print(f"Sweeping k = {k_values} for niching='kth'\n")

    for prob_name in test_problems:
        print(f"\n========== {prob_name} ==========")
        problem = PROBLEM_BUILDERS[prob_name]()
        p_cfg = next(p for p in cfg["problems"] if p["name"] == prob_name)
        constraint_symbols = p_cfg["constraint_symbols"]
        objective_symbol = p_cfg["objective_symbol"]
        f_opt = p_cfg["objective_optimum"]

        with open(THRESHOLDS_DIR / f"{prob_name}.yaml") as f:
            thr_doc = yaml.safe_load(f)
        thresholds = {c: float(thr_doc["levels"][ct_level].get(c, 0.0)) for c in constraint_symbols}
        shadow_opt_lvl = thr_doc.get("objective_shadow_optima", {}).get(ct_level)
        true_sp = (f_opt - shadow_opt_lvl) if shadow_opt_lvl is not None else None

        operator_cfg["mutation_probability"] = min(1.0, mut_factor / len(problem.variables))
        front_path = (
            FRONTS_DIR / f"{prob_name}_gen{cfg['n_generations_front']}_psize{cfg['population_size_front']}.parquet"
        )
        hv_ctx = build_hv_context(front_path, ct_level, thr_doc, objective_symbol, constraint_symbols, eps_percent)

        print(f"  true optimum: {f_opt}, true shadow price: {true_sp}")
        print(
            f"  {'k':>3s} | {'best mean':>12s} {'std':>8s} | {'HV mean':>8s} {'std':>7s} | "
            f"{'|SP err|':>10s} {'std':>8s} | time(s)"
        )

        for k in k_values:
            t0 = time.perf_counter()
            bests, hvs, sp_errs = [], [], []
            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                sols = run_once(
                    problem,
                    thresholds,
                    pop_size,
                    n_generations,
                    seed,
                    objective_symbol,
                    niching="kth",
                    niching_k=k,
                    operator_cfg=operator_cfg,
                )
                bf, bs, hv = compute_metrics(sols, objective_symbol, constraint_symbols, thresholds, hv_ctx)
                if bf is not None:
                    bests.append(bf)
                hvs.append(hv)
                if bf is not None and bs is not None and true_sp is not None:
                    sp_errs.append(abs((bf - bs) - true_sp))
            elapsed = time.perf_counter() - t0

            best_arr = np.array(bests)
            hv_arr = np.array(hvs)
            sp_arr = np.array(sp_errs)
            best_str = (
                f"{best_arr.mean():>12.4f} {best_arr.std():>8.4f}" if len(best_arr) else f"{'--':>12s} {'--':>8s}"
            )
            hv_str = f"{hv_arr.mean():>8.4f} {hv_arr.std():>7.4f}"
            sp_str = f"{sp_arr.mean():>10.4f} {sp_arr.std():>8.4f}" if len(sp_arr) else f"{'--':>10s} {'--':>8s}"
            print(f"  {k:>3d} | {best_str} | {hv_str} | {sp_str} | {elapsed:.1f}")


if __name__ == "__main__":
    main()
