"""Compute per-generation summary statistics (HV mean/SE/CI) for one experiment file."""

import numpy as np
import polars as pl
from scipy.stats import t
from snakemake.script import snakemake

from desdeo.tools.indicators_unary import hv


def hv_from_front(front: list[dict] | None, dim_keys: list[str], ref: float = 1.0) -> float | None:
    """Compute hypervolume from a list of structs converted to python dicts by Polars."""
    if len(front) == 0:  # [] or None
        return None

    pts = np.asarray([[p[k] for k in dim_keys] for p in front], dtype=float)
    return float(hv(pts, ref))


def snakemake_main() -> None:
    data_path = str(snakemake.input["data"])
    front_path = str(snakemake.input["front"])
    out_path = str(snakemake.output[0])

    objective_symbol = str(snakemake.params.objective_symbol)
    constraint_symbols = list(snakemake.params.constraint_symbols)
    constraint_thresholds = dict(snakemake.params.constraint_thresholds)

    f_col = f"{objective_symbol}_min"
    c_cols = constraint_symbols

    df = pl.read_parquet(data_path)
    df_front = pl.read_parquet(front_path)

    # best optimal stuff

    ## filter feasible solutions
    feasible_expr = pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols])
    per_run_gen = df.group_by(["generation", "run"]).agg(
        pl.when(feasible_expr).then(pl.col(f_col)).otherwise(None).min().alias("gen_best_feasible")
    )

    ## take cumulative minimum over all generations in each run
    inf = 1_000_000
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

    ## compute standard error
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

    ## compute 95% confidence intervals
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

    # hypervolume stuff
    ref_filter = pl.all_horizontal(
        [(pl.col(c) >= 0.0) & (pl.col(c) <= float(constraint_thresholds[c])) for c in c_cols]
    )
    reference_front = df_front.filter(ref_filter)

    if reference_front.height == 0:
        raise ValueError(
            f"Reference front is empty after filtering by thresholds: {constraint_thresholds}. "
            "Cannot normalize or compute HV."
        )

    f_lo, f_hi = reference_front[f_col].min(), reference_front[f_col].max()
    c_los = {c: reference_front[c].min() for c in c_cols}
    c_his = {c: reference_front[c].max() for c in c_cols}

    mask_terms = [pl.col(f_col).is_between(f_lo, f_hi, closed="both")]
    mask_terms += [pl.col(c).is_between(c_los[c], c_his[c], closed="both") for c in c_cols]
    mask = pl.all_horizontal(mask_terms)

    norm_fields = [((pl.col(f_col) - f_lo) / (f_hi - f_lo)).clip(0.0, 1.0).alias("f_norm")]
    for c in c_cols:
        denom = c_his[c] - c_los[c]
        norm_fields.append(((pl.col(c) - c_los[c]) / denom).clip(0.0, 1.0).alias(f"{c}_norm"))

    filtered = (
        df.group_by(["run", "generation"])
        .agg(
            pl.struct([pl.col(f_col).alias("f"), *[pl.col(c).alias(c) for c in c_cols]])
            .filter(mask)
            .alias("shadow_front"),
            pl.struct(norm_fields).filter(mask).alias("shadow_front_norm"),
        )
        .sort(["run", "generation"])
    )

    hv_keys = ["f_norm"] + [f"{c}_norm" for c in c_cols]
    filtered = filtered.with_columns(
        pl.col("shadow_front_norm")
        .map_elements(lambda front: hv_from_front(front, hv_keys), return_dtype=pl.Float64)
        .alias("hv")
    )

    hv_summary = (
        filtered.group_by("generation")
        .agg(
            pl.col("hv").mean().alias("hv_mean"),
            pl.col("hv").std().alias("hv_std"),
            pl.col("hv").count().alias("hv_n_supporting_runs"),
        )
        .with_columns(
            pl.when(pl.col("hv_n_supporting_runs") > 1)
            .then(pl.col("hv_std") / pl.col("hv_n_supporting_runs").sqrt())
            .otherwise(None)
            .alias("hv_stderr")
        )
        .sort("generation")
    )

    hv_summary = hv_summary.with_columns(
        pl.when(pl.col("hv_n_supporting_runs") > 1)
        .then(
            pl.Series(
                "hv_t_crit",
                t.ppf(0.975, hv_summary["hv_n_supporting_runs"] - 1),
            )
        )
        .otherwise(None)
    )

    hv_summary = hv_summary.with_columns(
        (pl.col("hv_mean") + pl.col("hv_t_crit") * pl.col("hv_stderr")).alias("hv_ci_upper"),
        (pl.col("hv_mean") - pl.col("hv_t_crit") * pl.col("hv_stderr")).alias("hv_ci_lower"),
    )

    # collate and save
    summary_all = summary.join(hv_summary, on="generation")

    summary_all.write_parquet(out_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit(
            "This script is intended to be run via Snakemake's `script:` directive, which injects a `snakemake` object."
        ) from err

    snakemake_main()
