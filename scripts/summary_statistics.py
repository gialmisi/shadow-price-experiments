"""Compute per-generation summary statistics (HV mean/SE/CI) for one experiment file."""

import moocore
import numpy as np
import polars as pl
import yaml
from scipy.stats import t
from snakemake.script import snakemake

from desdeo.tools.non_dominated_sorting import non_dominated


def snakemake_main() -> None:  # noqa: D103
    data_path = str(snakemake.input["data"])
    front_path = str(snakemake.input["front"])
    out_path = str(snakemake.output[0])

    objective_symbol = str(snakemake.params.objective_symbol)
    ct_level = str(snakemake.params.ct_level)

    thresholds_path = str(snakemake.input["thresholds"])
    with open(thresholds_path, encoding="utf-8") as f:  # noqa: PTH123
        thresholds_doc = yaml.safe_load(f)

    constraint_symbols = list(snakemake.params.constraint_symbols)
    level_constraints = dict(thresholds_doc["levels"].get(ct_level, {}))
    thresholds = {c: float(level_constraints.get(c, 0.0)) for c in constraint_symbols}

    f_col = f"{objective_symbol}_min"
    c_cols = constraint_symbols
    dim_cols = [f_col, *c_cols]

    eps_percent = float(snakemake.config["hv_eps_percent"])

    df = pl.read_parquet(data_path)
    df_front = pl.read_parquet(front_path)

    # Best feasible objective
    feasible_expr = pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols])
    per_run_gen = df.group_by(["generation", "run"]).agg(
        pl.when(feasible_expr).then(pl.col(f_col)).otherwise(None).min().alias("gen_best_feasible")
    )

    inf = 1_000_000_000
    per_run_best_so_far = (
        per_run_gen.sort(["run", "generation"])
        .with_columns(pl.col("gen_best_feasible").fill_null(inf).cum_min().over("run").alias("run_best_so_far_raw"))
        .with_columns(
            pl.when(pl.col("run_best_so_far_raw") == inf)
            .then(None)
            .otherwise(pl.col("run_best_so_far_raw"))
            .alias("run_best_so_far")
        )
    ).drop("run_best_so_far_raw")

    summary = (
        per_run_best_so_far.group_by("generation")
        .agg(
            pl.col("run_best_so_far").mean().alias("run_best_so_far_mean"),
            pl.col("run_best_so_far").std().alias("run_best_so_far_std"),
            pl.col("run_best_so_far").count().alias("best_n_feasible_runs"),
        )
        .with_columns(
            (pl.col("run_best_so_far_std") / pl.col("best_n_feasible_runs").sqrt()).alias("run_best_so_far_stderr")
        )
    ).sort("generation")

    summary = summary.with_columns(
        pl.when(pl.col("best_n_feasible_runs") > 1)
        .then(pl.Series("best_t_crit", t.ppf(0.975, summary["best_n_feasible_runs"] - 1)))
        .otherwise(None)
    )

    summary = summary.with_columns(
        (pl.col("run_best_so_far_mean") + pl.col("best_t_crit") * pl.col("run_best_so_far_stderr")).alias(
            "best_ci_upper"
        ),
        (pl.col("run_best_so_far_mean") - pl.col("best_t_crit") * pl.col("run_best_so_far_stderr")).alias(
            "best_ci_lower"
        ),
    ).sort("generation")

    # Shadow price: best objective in each generation using THRESHOLD-feasibility
    # A point is "shadow-feasible" if all constraints satisfy c <= threshold[c]
    shadow_feasible_expr = pl.all_horizontal([pl.col(c) <= float(thresholds[c]) for c in c_cols])

    # Per (run, generation): best objective among shadow-feasible points in that generation
    per_run_shadow_gen = df.group_by(["generation", "run"]).agg(
        pl.when(shadow_feasible_expr).then(pl.col(f_col)).otherwise(None).min().alias("shadow_gen_best")
    )

    per_run_shadow_best_so_far = (
        per_run_shadow_gen.sort(["run", "generation"])
        .with_columns(pl.col("shadow_gen_best").fill_null(inf).cum_min().over("run").alias("shadow_best_so_far_raw"))
        .with_columns(
            pl.when(pl.col("shadow_best_so_far_raw") == inf)
            .then(None)
            .otherwise(pl.col("shadow_best_so_far_raw"))
            .alias("shadow_best_so_far")
        )
        .drop("shadow_best_so_far_raw")
    )

    # Generation-wise summary for the "in a generation" shadow price
    shadow_gen_summary = (
        per_run_shadow_best_so_far.group_by("generation")
        .agg(
            pl.col("shadow_gen_best").mean().alias("shadow_gen_best_mean"),
            pl.col("shadow_gen_best").std().alias("shadow_gen_best_std"),
            pl.col("shadow_gen_best").count().alias("shadow_gen_best_n_runs"),
        )
        .with_columns(
            (pl.col("shadow_gen_best_std") / pl.col("shadow_gen_best_n_runs").sqrt()).alias("shadow_gen_best_stderr")
        )
        .sort("generation")
    )

    shadow_gen_summary = shadow_gen_summary.with_columns(
        pl.when(pl.col("shadow_gen_best_n_runs") > 1)
        .then(pl.Series("shadow_gen_best_t_crit", t.ppf(0.975, shadow_gen_summary["shadow_gen_best_n_runs"] - 1)))
        .otherwise(None)
    ).with_columns(
        (pl.col("shadow_gen_best_mean") + pl.col("shadow_gen_best_t_crit") * pl.col("shadow_gen_best_stderr")).alias(
            "shadow_gen_best_ci_upper"
        ),
        (pl.col("shadow_gen_best_mean") - pl.col("shadow_gen_best_t_crit") * pl.col("shadow_gen_best_stderr")).alias(
            "shadow_gen_best_ci_lower"
        ),
    )

    # Summary for threshold-feasible best-so-far
    shadow_best_summary = (
        per_run_shadow_best_so_far.group_by("generation")
        .agg(
            pl.col("shadow_best_so_far").mean().alias("shadow_best_so_far_mean"),
            pl.col("shadow_best_so_far").std().alias("shadow_best_so_far_std"),
            pl.col("shadow_best_so_far").count().alias("shadow_best_so_far_n_runs"),
        )
        .with_columns(
            (pl.col("shadow_best_so_far_std") / pl.col("shadow_best_so_far_n_runs").sqrt()).alias(
                "shadow_best_so_far_stderr"
            )
        )
        .sort("generation")
    )

    shadow_best_summary = shadow_best_summary.with_columns(
        pl.when(pl.col("shadow_best_so_far_n_runs") > 1)
        .then(pl.Series("shadow_best_t_crit", t.ppf(0.975, shadow_best_summary["shadow_best_so_far_n_runs"] - 1)))
        .otherwise(None)
    ).with_columns(
        (pl.col("shadow_best_so_far_mean") + pl.col("shadow_best_t_crit") * pl.col("shadow_best_so_far_stderr")).alias(
            "shadow_best_ci_upper"
        ),
        (pl.col("shadow_best_so_far_mean") - pl.col("shadow_best_t_crit") * pl.col("shadow_best_so_far_stderr")).alias(
            "shadow_best_ci_lower"
        ),
    )

    # Hypervolume reference point from reference front
    def filter_relax_only(relaxed: str | None) -> pl.Expr:
        terms = []
        for c in c_cols:
            if relaxed is None:
                terms.append(pl.col(c) <= 0.0)
            elif c == relaxed:
                terms.append(pl.col(c) <= float(thresholds.get(c, 0.0)))
            else:
                terms.append(pl.col(c) <= 0.0)
        return pl.all_horizontal(terms)

    # Shadow front nadir: collect all shadow-feasible points from the reference
    # front (fully feasible + each single-constraint relaxation) and take the
    # per-dimension maximum (worst) as the reference point for HV.
    shadow_feasible_parts = [df_front.filter(filter_relax_only(relaxed=None))]
    for ci in c_cols:
        shadow_feasible_parts.append(df_front.filter(filter_relax_only(relaxed=ci)))

    shadow_feasible = pl.concat(shadow_feasible_parts).unique()
    if shadow_feasible.height == 0:
        raise ValueError("No shadow-feasible points on the reference front.")

    # Ideal (min) and nadir (max) per dimension on the shadow-feasible front
    ideal_vals = np.array(shadow_feasible.select([pl.col(x).min().alias(x) for x in dim_cols]).row(0), dtype=float)
    nadir_vals = np.array(shadow_feasible.select([pl.col(x).max().alias(x) for x in dim_cols]).row(0), dtype=float)
    sf_ranges = nadir_vals - ideal_vals

    # Detect inactive dimensions: near-zero range in the shadow-feasible region
    active_mask = sf_ranges > 1e-12
    # Always keep the objective (first column)
    active_mask[0] = True

    # Normalize to [0, 1] using ideal/nadir so all dimensions contribute equally.
    # ref is set to 1 + eps in the normalized space.
    ideal_active = ideal_vals[active_mask]
    range_active = sf_ranges[active_mask]
    # Guard against zero range (shouldn't happen after active_mask, but be safe)
    range_active = np.where(range_active > 0, range_active, 1.0)

    ref_norm = np.ones(int(active_mask.sum())) * (1.0 + eps_percent)
    hv_ind = moocore.Hypervolume(ref=ref_norm, maximise=False)

    def normalize(pts: np.ndarray) -> np.ndarray:
        return (pts - ideal_active) / range_active

    # Cumulative best HV with non-dominated archive per run
    # Pre-partition data into {(run, gen): numpy_array} for fast lookup
    generations = sorted(df["generation"].unique().to_list())
    runs = sorted(df["run"].unique().to_list())

    grouped = df.group_by(["run", "generation"], maintain_order=True)
    gen_data: dict[tuple[int, int], np.ndarray] = {}
    for (run_id, gen), sub in grouped:
        pts = normalize(sub.select(dim_cols).to_numpy()[:, active_mask])
        gen_data[(int(run_id), int(gen))] = pts

    n_active = int(active_mask.sum())
    hv_rows = []
    for run_id in runs:
        archive = np.empty((0, n_active), dtype=float)

        for gen in generations:
            new_pts = gen_data.get((int(run_id), int(gen)))

            if new_pts is None or new_pts.shape[0] == 0:
                hv_val = 0.0 if archive.shape[0] == 0 else float(hv_ind(archive))
                hv_rows.append((int(run_id), int(gen), hv_val))
                continue

            if archive.shape[0] == 0:
                combined = new_pts
            else:
                combined = np.vstack([archive, new_pts])

            # ref-box filter (in normalized space)
            combined = combined[(combined <= ref_norm).all(axis=1)]

            if combined.shape[0] == 0:
                archive = np.empty((0, n_active), dtype=float)
                hv_rows.append((int(run_id), int(gen), 0.0))
                continue

            # Filter to non-dominated
            nd_mask = non_dominated(combined)
            archive = combined[nd_mask]

            hv_val = float(hv_ind(archive))
            hv_rows.append((int(run_id), int(gen), hv_val))

    hv_df = pl.DataFrame(hv_rows, schema=["run", "generation", "hv"]).sort(["run", "generation"])

    hv_summary = (
        hv_df.group_by("generation")
        .agg(
            pl.col("hv").mean().alias("hv_mean"),
            pl.col("hv").std().alias("hv_std"),
            pl.col("hv").count().alias("hv_n_runs"),
        )
        .with_columns((pl.col("hv_std") / pl.col("hv_n_runs").sqrt()).alias("hv_stderr"))
        .sort("generation")
    )

    hv_summary = hv_summary.with_columns(
        pl.when(pl.col("hv_n_runs") > 1)
        .then(pl.Series("hv_t_crit", t.ppf(0.975, hv_summary["hv_n_runs"] - 1)))
        .otherwise(None)
    )

    hv_summary = hv_summary.with_columns(
        (pl.col("hv_mean") + pl.col("hv_t_crit") * pl.col("hv_stderr")).alias("hv_ci_upper"),
        (pl.col("hv_mean") - pl.col("hv_t_crit") * pl.col("hv_stderr")).alias("hv_ci_lower"),
    )

    # Shadow price difference: run_best_so_far - shadow_best_so_far (per run, per generation)
    shadow_diff_per_run = (
        per_run_best_so_far.select(["run", "generation", "run_best_so_far"])
        .join(
            per_run_shadow_best_so_far.select(["run", "generation", "shadow_best_so_far"]),
            on=["run", "generation"],
        )
        .with_columns((pl.col("run_best_so_far") - pl.col("shadow_best_so_far")).alias("shadow_price_diff"))
    )

    shadow_diff_summary = (
        shadow_diff_per_run.group_by("generation")
        .agg(
            pl.col("shadow_price_diff").mean().alias("shadow_price_diff_mean"),
            pl.col("shadow_price_diff").std().alias("shadow_price_diff_std"),
            pl.col("shadow_price_diff").count().alias("shadow_price_diff_n_runs"),
        )
        .with_columns(
            (pl.col("shadow_price_diff_std") / pl.col("shadow_price_diff_n_runs").sqrt()).alias(
                "shadow_price_diff_stderr"
            )
        )
        .sort("generation")
    )

    shadow_diff_summary = shadow_diff_summary.with_columns(
        pl.when(pl.col("shadow_price_diff_n_runs") > 1)
        .then(
            pl.Series(
                "shadow_price_diff_t_crit",
                t.ppf(0.975, shadow_diff_summary["shadow_price_diff_n_runs"] - 1),
            )
        )
        .otherwise(None)
    ).with_columns(
        (
            pl.col("shadow_price_diff_mean") + pl.col("shadow_price_diff_t_crit") * pl.col("shadow_price_diff_stderr")
        ).alias("shadow_price_diff_ci_upper"),
        (
            pl.col("shadow_price_diff_mean") - pl.col("shadow_price_diff_t_crit") * pl.col("shadow_price_diff_stderr")
        ).alias("shadow_price_diff_ci_lower"),
    )

    # Collate and save
    summary_all = (
        summary.join(shadow_gen_summary, on="generation")
        .join(shadow_best_summary, on="generation")
        .join(hv_summary, on="generation")
        .join(shadow_diff_summary, on="generation")
    )
    summary_all.write_parquet(out_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err

    snakemake_main()
