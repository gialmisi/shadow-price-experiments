"""Compute per-generation summary statistics (HV mean/SE/CI) for one experiment file."""

import moocore
import numpy as np
import polars as pl
import yaml
from scipy.stats import t

try:
    from snakemake.script import snakemake  # snakemake <= 8.x
except ImportError:
    pass  # snakemake >= 9.x injects `snakemake` via the script preamble

from desdeo.tools.non_dominated_sorting import non_dominated_merge


def snakemake_main() -> None:  # noqa: D103
    data_path = str(snakemake.input["data"])
    front_path = str(snakemake.input["front"])
    out_path = str(snakemake.output[0])
    out_meta_path = str(snakemake.output[1])
    out_per_run_path = str(snakemake.output[2])

    objective_symbol = str(snakemake.params.objective_symbol)
    ct_level = str(snakemake.params.ct_level)

    # Build a label for progress messages
    job_label = (
        f"{snakemake.wildcards.problem_name}"
        f"/{snakemake.wildcards.mode}"
        f"/ct{snakemake.wildcards.ctlevel}"
        f"/psize{snakemake.wildcards.psize}"
    )

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

    print(f"[{job_label}] Loaded {df.height} rows, {df_front.height} front points", flush=True)

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

    print(f"[{job_label}] Best-feasible summary done", flush=True)

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

    print(f"[{job_label}] Shadow-price summaries done", flush=True)

    # Relaxation-gain HV box (with fallback to the original shadow-feasible-front box):
    #   - objective axis: [f*_relaxed, f*_strict]
    #       f*_strict  = best objective on reference front with all c <= 0
    #       f*_relaxed = best objective on reference front with all c <= threshold
    #   - each constraint axis: [0, threshold_c]
    # Strict-feasible coordinates (c <= 0) and below-reference objective (f < f*_relaxed)
    # clip to the box floor, giving full width on the corresponding axis. Points outside
    # the box (c > threshold or f > f*_strict) are dropped.
    # Fallback to the original box when no strict-feasible reference set exists or
    # relaxation provides no objective gain (f*_strict <= f*_relaxed).
    joint_threshold_expr = pl.all_horizontal([pl.col(c) <= float(thresholds.get(c, 0.0)) for c in c_cols])
    shadow_feasible = df_front.filter(joint_threshold_expr).unique()
    if shadow_feasible.height == 0:
        raise ValueError("No shadow-feasible points on the reference front.")

    strict_feasible = df_front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols]))
    f_strict = float(strict_feasible[f_col].min()) if strict_feasible.height > 0 else None
    f_relaxed = float(shadow_feasible[f_col].min())
    use_relax_box = f_strict is not None and f_strict > f_relaxed

    # Evidence-based dimension filtering: drop constraints with weak evidence
    evidence = thresholds_doc.get("evidence", {})
    active_mask = np.ones(len(dim_cols), dtype=bool)
    dropped_meta = {}
    for i, c in enumerate(c_cols):
        col_idx = i + 1
        ev = evidence.get(c, {})
        n_ev = ev.get("n", 0)
        source = ev.get("source", "")
        if source == "random_sampling":
            active_mask[col_idx] = False
            dropped_meta[c] = {"reason": "random_sampling", "n": n_ev}
        elif source != "manual" and n_ev < 10:
            # Hand-picked manual thresholds are deliberate and never dropped, even though
            # compute_thresholds.py records n=1 for them. Without this exemption,
            # single-constraint problems end up with n_active=1 and segfault moocore's
            # 1D Hypervolume.
            active_mask[col_idx] = False
            dropped_meta[c] = {"reason": "insufficient_evidence", "n": n_ev, "threshold": 10}

    if use_relax_box:
        # Drop constraint dims whose threshold has no positive width
        for i, c in enumerate(c_cols):
            if float(thresholds[c]) <= 0.0 and active_mask[i + 1]:
                active_mask[i + 1] = False
                dropped_meta.setdefault(c, {"reason": "threshold_nonpositive"})
        ideal_vals = np.zeros(len(dim_cols))
        ideal_vals[0] = f_relaxed
        range_full = np.empty(len(dim_cols))
        range_full[0] = f_strict - f_relaxed
        for i, c in enumerate(c_cols):
            range_full[i + 1] = float(thresholds[c])
        range_full = np.where(range_full > 0, range_full, 1.0)
        ref_full = np.ones(len(dim_cols))  # nadir on each axis; eps margin added below
        hv_box_kind = "relax"
    else:
        # Fallback: shadow-feasible-front-based box (the original definition)
        ideal_vals = np.array(shadow_feasible.select([pl.col(x).min().alias(x) for x in dim_cols]).row(0), dtype=float)
        nadir_vals = np.array(shadow_feasible.select([pl.col(x).max().alias(x) for x in dim_cols]).row(0), dtype=float)
        sf_ranges = nadir_vals - ideal_vals
        for i in range(1, len(dim_cols)):
            if sf_ranges[i] <= 1e-12 and active_mask[i]:
                active_mask[i] = False
                dropped_meta.setdefault(c_cols[i - 1], {"reason": "zero_range_on_shadow_front"})
        range_full = np.where(sf_ranges > 0, sf_ranges, 1.0)
        ref_full = np.empty(len(dim_cols))
        for i, dname in enumerate(dim_cols):
            if dname == f_col:
                ref_full[i] = 1.0
            else:
                ref_full[i] = (float(thresholds[dname]) - ideal_vals[i]) / range_full[i]
        hv_box_kind = "current_fallback"

    # Expand the nadir reference outward by eps on every (active) axis so that solutions lying
    # exactly on a nadir boundary still contribute hypervolume (otherwise the strict and threshold
    # optima would each contribute zero volume). Applied uniformly to both the relax and fallback
    # boxes.
    ref_full = ref_full + eps_percent

    active_dim_names = [dim_cols[i] for i in range(len(dim_cols)) if active_mask[i]]
    ideal_active = ideal_vals[active_mask]
    range_active = range_full[active_mask]
    ref_norm = ref_full[active_mask]
    hv_ind = moocore.Hypervolume(ref=ref_norm, maximise=False)

    ref_orig = ideal_active + ref_norm * range_active
    hv_meta = {
        "hv_box_kind": hv_box_kind,
        "hv_dimensions": {
            "all": dim_cols,
            "active": active_dim_names,
            "dropped": dropped_meta if dropped_meta else None,
        },
        "hv_reference_point": {
            active_dim_names[k]: {
                "normalized": float(ref_norm[k]),
                "original": float(ref_orig[k]),
            }
            for k in range(len(active_dim_names))
        },
        "hv_relax_anchors": {
            "f_strict": f_strict,
            "f_relaxed": f_relaxed,
            "objective_gain": (f_strict - f_relaxed) if f_strict is not None else None,
        },
    }

    def normalize(pts: np.ndarray) -> np.ndarray:
        return (pts - ideal_active) / range_active

    # Cumulative best HV with incremental non-dominated merge per run
    # Pre-partition data into {(run, gen): numpy_array} for fast lookup
    generations = sorted(df["generation"].unique().to_list())
    runs = sorted(df["run"].unique().to_list())

    grouped = df.group_by(["run", "generation"], maintain_order=True)
    gen_data: dict[tuple[int, int], np.ndarray] = {}
    for (run_id, gen), sub in grouped:
        pts = normalize(sub.select(dim_cols).to_numpy()[:, active_mask])
        gen_data[(int(run_id), int(gen))] = pts

    n_active = int(active_mask.sum())
    print(
        f"[{job_label}] HV loop: {len(runs)} runs \u00d7 {len(generations)} generations, {n_active} active dims",
        flush=True,
    )

    hv_rows = []
    for run_idx, run_id in enumerate(runs):
        archive = np.empty((0, n_active), dtype=float)
        prev_hv = 0.0

        for gen in generations:
            new_pts = gen_data.get((int(run_id), int(gen)))

            if new_pts is None or new_pts.shape[0] == 0:
                hv_rows.append((int(run_id), int(gen), prev_hv))
                continue

            # Clip strict-feasible / below-reference coordinates to the box floor.
            # No-op for the fallback box (its ideal is the shadow-feasible front min,
            # so normalized coords are non-negative by construction).
            if hv_box_kind == "relax":
                new_pts = np.maximum(new_pts, 0.0)

            # Ref-box filter new points only (archive already passed)
            new_pts = new_pts[(new_pts <= ref_norm).all(axis=1)]
            if new_pts.shape[0] == 0:
                hv_rows.append((int(run_id), int(gen), prev_hv))
                continue

            # Filter new points to non-dominated among themselves
            nd_new_mask = moocore.is_nondominated(new_pts, maximise=False)
            new_nd = new_pts[nd_new_mask]

            if archive.shape[0] == 0:
                archive = new_nd
            else:
                # Incremental merge: only compare archive × new, not archive × archive
                mask_old, mask_new = non_dominated_merge(archive, new_nd)
                if not mask_new.any():
                    # No new points survived → archive unchanged, reuse previous HV
                    hv_rows.append((int(run_id), int(gen), prev_hv))
                    continue
                archive = np.vstack([archive[mask_old], new_nd[mask_new]])

            prev_hv = float(hv_ind(archive))
            hv_rows.append((int(run_id), int(gen), prev_hv))

        if (run_idx + 1) % 50 == 0:
            print(f"[{job_label}] HV: {run_idx + 1}/{len(runs)} runs done", flush=True)

    print(f"[{job_label}] HV loop done", flush=True)

    hv_df = pl.DataFrame(hv_rows, schema=["run", "generation", "hv"], orient="row").sort(["run", "generation"])

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

    # Per-run final-generation values (for statistical testing)
    max_gen = max(generations)
    per_run_final = (
        per_run_best_so_far.filter(pl.col("generation") == max_gen)
        .select(["run", "run_best_so_far"])
        .join(
            per_run_shadow_best_so_far.filter(pl.col("generation") == max_gen).select(
                ["run", "shadow_gen_best", "shadow_best_so_far"]
            ),
            on="run",
        )
        .join(
            hv_df.filter(pl.col("generation") == max_gen).select(["run", "hv"]),
            on="run",
        )
        .join(
            shadow_diff_per_run.filter(pl.col("generation") == max_gen).select(["run", "shadow_price_diff"]),
            on="run",
        )
        .sort("run")
    )
    per_run_final.write_parquet(out_per_run_path)

    # Write sidecar metadata
    with open(out_meta_path, "w", encoding="utf-8") as f:  # noqa: PTH123
        yaml.dump(hv_meta, f, default_flow_style=False, sort_keys=False)

    print(f"[{job_label}] Written {out_path}", flush=True)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err

    snakemake_main()
