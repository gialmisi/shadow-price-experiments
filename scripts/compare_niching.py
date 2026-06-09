"""Quick comparison of niching mechanisms in the ranking mode.

Compares crowding, knearest (mean of k nearest), and kth (SPEA2-style)
on a handful of problems. Reports final-generation best_so_far, HV,
and shadow-price error metrics.

Not a Snakemake job.
"""

import sys
import time
from pathlib import Path

import moocore
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import PROBLEM_BUILDERS

from desdeo.emo import (
    algorithms,
    crossover,
    generator,
    mutation,
    scalar_selection,
    selection,
    termination,
)
from desdeo.emo.hooks.archivers import Archive
from desdeo.tools.non_dominated_sorting import non_dominated_merge

CONFIG_PATH = Path("experiment_config.yaml")
THRESHOLDS_DIR = Path("results/thresholds")
FRONTS_DIR = Path("results/fronts")


def run_once(problem, constraints, pop_size, n_generations, seed, objective_symbol, niching, niching_k, operator_cfg):
    nsga2_options = algorithms.nsga2_options()
    nsga2_options.template.seed = seed
    nsga2_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=operator_cfg["xover_probability"],
        xover_distribution=operator_cfg["xover_distribution"],
    )
    nsga2_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=operator_cfg["mutation_probability"],
        distribution_index=operator_cfg["distribution_index"],
    )
    nsga2_options.template.mate_selection = scalar_selection.TournamentSelectionOptions(
        name="TournamentSelection",
        tournament_size=operator_cfg["tournament_size"],
        winner_size=pop_size,
    )
    nsga2_options.template.selection = selection.SingleObjectiveConstrainedRankingSelectorOptions(
        target_objective_symbol=objective_symbol,
        constraints=constraints,
        population_size=pop_size,
        mode="ranking",
        niching=niching,
        niching_k=niching_k,
    )
    nsga2_options.template.generator = generator.LHSGeneratorOptions(n_points=pop_size)
    nsga2_options.template.termination = termination.MaxGenerationsTerminatorOptions(max_generations=n_generations)

    solver, extras = algorithms.emo_constructor(emo_options=nsga2_options, problem=problem)
    archive = Archive(problem=problem, publisher=extras.publisher)
    extras.publisher.auto_subscribe(archive)
    extras.publisher.register_topics(archive.provided_topics[archive.verbosity], archive.__class__.__name__)
    solver()
    return archive.solutions


def compute_metrics(sols_df, objective_symbol, constraint_symbols, thresholds, hv_ctx):
    """Return (best_feasible, shadow_best_feasible, hv) for one run."""
    f_col = f"{objective_symbol}_min"

    # Best strictly feasible objective in the run
    strict_feas = sols_df.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in constraint_symbols]))
    best_feas = float(strict_feas[f_col].min()) if strict_feas.height > 0 else None

    # Best threshold-feasible objective in the run
    shadow_feas = sols_df.filter(pl.all_horizontal([pl.col(c) <= thresholds[c] for c in constraint_symbols]))
    best_shadow = float(shadow_feas[f_col].min()) if shadow_feas.height > 0 else None

    # HV: cumulative non-dominated archive in (obj, active constraints) space, same logic as summary_statistics
    dim_cols = [f_col] + list(constraint_symbols)
    active_mask = hv_ctx["active_mask"]
    ideal_active = hv_ctx["ideal_active"]
    range_active = hv_ctx["range_active"]
    ref_norm = hv_ctx["ref_norm"]
    hv_ind = hv_ctx["hv_ind"]

    # Build archive generation-by-generation
    archive = np.empty((0, int(active_mask.sum())), dtype=float)
    prev_hv = 0.0
    gens = sorted(sols_df["generation"].unique().to_list())

    for g in gens:
        sub = sols_df.filter(pl.col("generation") == g)
        pts = sub.select(dim_cols).to_numpy()[:, active_mask]
        pts = (pts - ideal_active) / range_active
        pts = pts[(pts <= ref_norm).all(axis=1)]
        if pts.shape[0] == 0:
            continue
        nd_mask = moocore.is_nondominated(pts, maximise=False)
        new_nd = pts[nd_mask]
        if archive.shape[0] == 0:
            archive = new_nd
        else:
            mask_old, mask_new = non_dominated_merge(archive, new_nd)
            if not mask_new.any():
                continue
            archive = np.vstack([archive[mask_old], new_nd[mask_new]])
        prev_hv = float(hv_ind(archive))

    return best_feas, best_shadow, prev_hv


def build_hv_context(front_path, ct_level, thresholds_doc, objective_symbol, constraint_symbols, eps_percent):
    f_col = f"{objective_symbol}_min"
    dim_cols = [f_col] + list(constraint_symbols)
    df_front = pl.read_parquet(front_path)

    # Shadow-feasible front: fully feasible + single-constraint relaxations
    def filter_relax_only(relaxed):
        terms = []
        for c in constraint_symbols:
            if relaxed is None or c != relaxed:
                terms.append(pl.col(c) <= 0.0)
            else:
                terms.append(pl.col(c) <= float(thresholds_doc["levels"][ct_level].get(c, 0.0)))
        return pl.all_horizontal(terms)

    parts = [df_front.filter(filter_relax_only(None))]
    for ci in constraint_symbols:
        parts.append(df_front.filter(filter_relax_only(ci)))
    shadow_feasible = pl.concat(parts).unique()

    ideal_vals = np.array(shadow_feasible.select([pl.col(x).min().alias(x) for x in dim_cols]).row(0), dtype=float)
    nadir_vals = np.array(shadow_feasible.select([pl.col(x).max().alias(x) for x in dim_cols]).row(0), dtype=float)
    sf_ranges = nadir_vals - ideal_vals

    evidence = thresholds_doc.get("evidence", {})
    active_mask = sf_ranges > 1e-12
    active_mask[0] = True
    for i, c in enumerate(constraint_symbols):
        ev = evidence.get(c, {})
        if ev.get("source", "") == "random_sampling" or ev.get("n", 0) < 10:
            active_mask[i + 1] = False

    ideal_active = ideal_vals[active_mask]
    range_active = sf_ranges[active_mask]
    range_active = np.where(range_active > 0, range_active, 1.0)
    ref_norm = np.ones(int(active_mask.sum())) * (1.0 + eps_percent)
    hv_ind = moocore.Hypervolume(ref=ref_norm, maximise=False)

    return {
        "active_mask": active_mask,
        "ideal_active": ideal_active,
        "range_active": range_active,
        "ref_norm": ref_norm,
        "hv_ind": hv_ind,
    }


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

    # Sample across constraint counts: 1, 2, 4
    test_problems = [
        "branin",  # 1 constraint (2-D rank vector)
        "mystery",  # 1 constraint
        "cantilevered_beam",  # 2 constraints (3-D rank vector)
        "g6",  # 2 constraints
        "g24",  # 2 constraints
        "g8",  # 2 constraints
        "pressure_vessel",  # 4 constraints (5-D rank vector)
        "g9",  # 4 constraints
    ]
    ct_level = "med"
    pop_size = 60
    n_generations = 200
    n_runs = 30
    base_seed = int(cfg["base_seed"])
    niching_methods = ["crowding", "knearest", "kth"]

    print(f"Config: ct_level={ct_level}, pop_size={pop_size}, n_gen={n_generations}, n_runs={n_runs}")
    print(
        f"Operators: η_c={operator_cfg['xover_distribution']}, η_m={operator_cfg['distribution_index']}, "
        f"p_mut={mut_factor}/n_vars\n"
    )

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

        metrics = {m: {"best": [], "hv": [], "sp_diff": []} for m in niching_methods}
        timings = {m: 0.0 for m in niching_methods}

        for niching in niching_methods:
            print(f"\n  --- niching = {niching} ---")
            t0 = time.perf_counter()
            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                sols = run_once(
                    problem,
                    thresholds,
                    pop_size,
                    n_generations,
                    seed,
                    objective_symbol,
                    niching,
                    niching_k=3,
                    operator_cfg=operator_cfg,
                )
                bf, bs, hv = compute_metrics(sols, objective_symbol, constraint_symbols, thresholds, hv_ctx)
                metrics[niching]["best"].append(bf)
                metrics[niching]["hv"].append(hv)
                metrics[niching]["sp_diff"].append((bf - bs) if (bf is not None and bs is not None) else None)
                if (run_idx + 1) % 10 == 0:
                    print(f"    run {run_idx + 1}/{n_runs} done")
            timings[niching] = time.perf_counter() - t0

        # Summary
        print(f"\n  SUMMARY ({n_runs} runs):")
        print(
            f"  {'niching':<10s} | {'best mean':>10s} {'std':>8s} | {'HV mean':>8s} {'std':>8s} | "
            f"{'SP err|mean|':>12s} {'std':>8s} | time(s)"
        )
        for niching in niching_methods:
            m = metrics[niching]
            best_arr = np.array([v for v in m["best"] if v is not None])
            hv_arr = np.array(m["hv"])
            sp_err_arr = (
                np.array([abs((sp - true_sp)) for sp in m["sp_diff"] if sp is not None])
                if true_sp is not None
                else np.array([])
            )

            best_str = (
                f"{best_arr.mean():>10.4f} {best_arr.std():>8.4f}" if len(best_arr) else f"{'--':>10s} {'--':>8s}"
            )
            hv_str = f"{hv_arr.mean():>8.4f} {hv_arr.std():>8.4f}"
            sp_str = (
                f"{sp_err_arr.mean():>12.4f} {sp_err_arr.std():>8.4f}" if len(sp_err_arr) else f"{'--':>12s} {'--':>8s}"
            )
            print(f"  {niching:<10s} | {best_str} | {hv_str} | {sp_str} | {timings[niching]:.1f}")

        # Paired comparisons (vs crowding)
        print("\n  Paired comparisons vs crowding (best_so_far; lower is better):")
        cr = np.array(metrics["crowding"]["best"], dtype=float)
        for niching in ["knearest", "kth"]:
            alt = np.array(metrics[niching]["best"], dtype=float)
            valid = ~(np.isnan(cr) | np.isnan(alt))
            if valid.sum() > 5:
                diff = cr[valid] - alt[valid]
                print(
                    f"    {niching}: mean_diff={diff.mean():.4f}, "
                    f"crowding_wins={(diff < 0).sum()}, {niching}_wins={(diff > 0).sum()}"
                )

        print("\n  Paired comparisons vs crowding (HV; higher is better):")
        cr_hv = np.array(metrics["crowding"]["hv"], dtype=float)
        for niching in ["knearest", "kth"]:
            alt_hv = np.array(metrics[niching]["hv"], dtype=float)
            diff = cr_hv - alt_hv  # positive = crowding better (HV is maximized)
            print(
                f"    {niching}: mean_diff={diff.mean():.4f}, "
                f"crowding_wins={(diff > 0).sum()}, {niching}_wins={(diff < 0).sum()}"
            )

        if true_sp is not None:
            print(f"\n  Paired comparisons vs crowding (|SP error|; lower is better):")
            cr_sp = np.array([abs(v - true_sp) if v is not None else np.nan for v in metrics["crowding"]["sp_diff"]])
            for niching in ["knearest", "kth"]:
                alt_sp = np.array([abs(v - true_sp) if v is not None else np.nan for v in metrics[niching]["sp_diff"]])
                valid = ~(np.isnan(cr_sp) | np.isnan(alt_sp))
                if valid.sum() > 5:
                    diff = cr_sp[valid] - alt_sp[valid]
                    print(
                        f"    {niching}: mean_diff={diff.mean():.4f}, "
                        f"crowding_wins={(diff < 0).sum()}, {niching}_wins={(diff > 0).sum()}"
                    )


if __name__ == "__main__":
    main()
