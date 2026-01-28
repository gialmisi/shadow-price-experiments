"""Compute per-generation summary statistics (HV mean/SE/CI) for one experiment file."""

import moocore
import numpy as np
import polars as pl
import yaml
from scipy.stats import t
from snakemake.script import snakemake


def snakemake_main() -> None:  # noqa: D103
    data_path = str(snakemake.input["data"])
    front_path = str(snakemake.input["front"])
    out_path = str(snakemake.output[0])

    objective_symbol = str(snakemake.params.objective_symbol)
    ct_level = str(snakemake.params.ct_level)

    thresholds_path = str(snakemake.input["thresholds"])
    with open(thresholds_path, "r", encoding="utf-8") as f:  # noqa: PTH123
        thresholds_doc = yaml.safe_load(f)

    constraint_symbols = list(snakemake.params.constraint_symbols)
    level_constraints = dict(thresholds_doc["levels"].get(ct_level, {}))
    thresholds = {c: float(level_constraints.get(c, 0.0)) for c in constraint_symbols}

    f_col = f"{objective_symbol}_min"
    c_cols = constraint_symbols
    dim_cols = [f_col, *c_cols]

    eps_percent = float(snakemake.params.hv_eps_percent)

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

    # Hypervolume reference point from reference front
    def filter_relax_only(relaxed: str | None) -> pl.Expr:
        terms = []
        for c in c_cols:
            if relaxed is None:
                terms.append(pl.col(c) <= 0.0)
            else:
                if c == relaxed:
                    terms.append(pl.col(c) <= float(thresholds.get(c, 0.0)))
                else:
                    terms.append(pl.col(c) <= 0.0)
        return pl.all_horizontal(terms)

    # shadow front nadir approximation
    # best fully feasible objective components
    cand_all = df_front.filter(filter_relax_only(relaxed=None))
    if cand_all.height == 0:
        raise ValueError("No points on df_front with all constraints enforced (c <= 0).")

    best_all = cand_all.sort(f_col).head(1)

    # relax each constraint one by one
    selected_rows = [best_all]
    best_relax = {}

    for ci in c_cols:
        cand = df_front.filter(filter_relax_only(relaxed=ci))
        if cand.height == 0:
            raise ValueError(
                f"No points when relaxing {ci} to <= {thresholds.get(ci, 0.0)} while enforcing others as <= 0."
            )
        row = cand.sort(f_col).head(1)
        best_relax[ci] = row
        selected_rows.append(row)

    sel = pl.concat(selected_rows).unique()

    ref_f = float(best_all[f_col][0])
    ref_cs = [float(best_relax[ci][ci][0]) for ci in c_cols]
    ref = np.array([ref_f, *ref_cs], dtype=float)

    # epsilon padding
    max_vals = sel.select([pl.col(x).max().alias(x) for x in dim_cols]).row(0)
    min_vals = sel.select([pl.col(x).min().alias(x) for x in dim_cols]).row(0)
    ranges = np.maximum(np.array(max_vals, float) - np.array(min_vals, float), 0.0)
    ref = ref + eps_percent * ranges

    hv_ind = moocore.Hypervolume(ref=ref, maximise=False)

    # Hypervolume per (run, generation)
    rows = []
    for (run, gen), sub in df.group_by(["run", "generation"], maintain_order=True):
        pts = sub.select(dim_cols).to_numpy()
        pts = pts[(pts <= ref).all(axis=1)]  # ref-box filter

        hv_val = 0.0 if pts.shape[0] == 0 else float(hv_ind(pts))
        rows.append((int(run), int(gen), hv_val))

    hv_df = pl.DataFrame(rows, schema=["run", "generation", "hv"]).sort(["run", "generation"])

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

    # Collate and save
    summary_all = summary.join(hv_summary, on="generation")
    summary_all.write_parquet(out_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err

    snakemake_main()
