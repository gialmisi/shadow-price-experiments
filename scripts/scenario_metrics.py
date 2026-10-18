"""Scenario metrics for one experiment file: the 2x2 grid of relaxation scenarios.

Scenarios (see the manuscript, Table "scenarios"): end points vs. whole curve, under
single vs. joint relaxation. The joint/curve cell is the existing relaxation-gain
hypervolume computed in ``summary_statistics.py``; this script computes the other three
cells from the same data, without re-running any experiment:

- ``joint_gap``   end points, joint relaxation. Relative optimality gap of the best-so-far
                  threshold-feasible objective (all ``c_j <= t_j``) against the reference
                  threshold optimum ``f*(t)``, normalized by the reference joint gain
                  ``pi(t) = f*(0) - f*(t)`` and clipped at zero.
- ``single_gap``  end points, single relaxation. For each constraint ``j`` with a positive
                  reference single gain, the relative gap of the best-so-far objective among
                  solutions with ``c_j <= t_j`` and every other constraint strictly satisfied,
                  against the reference single-relaxed optimum ``f*(t_j e_j)``; averaged
                  over the contributing constraints.
- ``single_hv``   whole curve, single relaxation. For each contributing constraint ``j``, the
                  hypervolume of the non-dominated archive of the same solutions in the
                  2-D box ``[f*(t_j e_j), f*(0)] x [0, t_j]``, each axis normalized to
                  ``[0, 1]``, reference point ``1 + eps`` on both axes, strictly satisfied
                  coordinates clipped to the floor (as in the joint hypervolume); averaged
                  over the contributing constraints.

Reference values: ``f*(0)`` is the known optimum from the experiment config; the relaxed
optima are the best-known values computed by ``compute_reference_optima.py``. A constraint contributes to the single
metrics only if its threshold is positive and its single gain exceeds the tolerance ``eps``
of the thresholds YAML. Problems without any contributing constraint, or without a positive
joint gain, get null values (shown as dashes in the tables).

Outputs (Snakemake): per-generation summary parquet (mean, std, n, stderr, t-CI per metric),
per-run final-generation parquet (for the Wilcoxon tests) and a metadata YAML with the
reference values and the contributing constraints.
"""

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

METRICS = ["joint_gap", "single_gap", "single_hv"]
INF = 1_000_000_000.0


def _best_so_far(df: pl.DataFrame, mask: pl.Expr, f_col: str, name: str) -> pl.DataFrame:
    """Per (run, generation) cumulative minimum of ``f_col`` over rows satisfying ``mask``."""
    per_gen = df.group_by(["run", "generation"]).agg(
        pl.when(mask).then(pl.col(f_col)).otherwise(None).min().alias("gen_best")
    )
    return (
        per_gen.sort(["run", "generation"])
        .with_columns(pl.col("gen_best").fill_null(INF).cum_min().over("run").alias("raw"))
        .with_columns(pl.when(pl.col("raw") >= INF).then(None).otherwise(pl.col("raw")).alias(name))
        .select(["run", "generation", name])
    )


def _hv_per_run_gen(
    df: pl.DataFrame,
    mask: pl.Expr,
    f_col: str,
    c_col: str,
    f_floor: float,
    f_range: float,
    c_range: float,
    ref: float,
    name: str,
) -> pl.DataFrame:
    """Cumulative 2-D hypervolume of the non-dominated archive per run and generation."""
    pts_df = (
        df.filter(mask)
        .select(
            pl.col("run"),
            pl.col("generation"),
            ((pl.col(f_col) - f_floor) / f_range).clip(lower_bound=0.0).alias("f_norm"),
            (pl.col(c_col).clip(lower_bound=0.0) / c_range).alias("c_norm"),
        )
        .filter((pl.col("f_norm") <= ref) & (pl.col("c_norm") <= ref))
    )
    gen_data: dict[tuple[int, int], np.ndarray] = {}
    for (run_id, gen), sub in pts_df.group_by(["run", "generation"], maintain_order=True):
        gen_data[(int(run_id), int(gen))] = sub.select(["f_norm", "c_norm"]).to_numpy()

    hv_ind = moocore.Hypervolume(ref=np.array([ref, ref]), maximise=False)
    generations = sorted(df["generation"].unique().to_list())
    runs = sorted(df["run"].unique().to_list())

    rows = []
    for run_id in runs:
        archive = np.empty((0, 2), dtype=float)
        prev_hv = 0.0
        for gen in generations:
            new_pts = gen_data.get((int(run_id), int(gen)))
            if new_pts is None or new_pts.shape[0] == 0:
                rows.append((int(run_id), int(gen), prev_hv))
                continue
            new_nd = new_pts[moocore.is_nondominated(new_pts, maximise=False)]
            if archive.shape[0] == 0:
                archive = new_nd
            else:
                mask_old, mask_new = non_dominated_merge(archive, new_nd)
                if not mask_new.any():
                    rows.append((int(run_id), int(gen), prev_hv))
                    continue
                archive = np.vstack([archive[mask_old], new_nd[mask_new]])
            prev_hv = float(hv_ind(archive))
            rows.append((int(run_id), int(gen), prev_hv))
    return pl.DataFrame(rows, schema=["run", "generation", name], orient="row")


def _summarize(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Per-generation mean, std, n, standard error and 95% t confidence interval of ``col``."""
    out = (
        df.group_by("generation")
        .agg(
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).count().alias(f"{col}_n_runs"),
        )
        .with_columns((pl.col(f"{col}_std") / pl.col(f"{col}_n_runs").sqrt()).alias(f"{col}_stderr"))
        .sort("generation")
    )
    n = out[f"{col}_n_runs"].to_numpy().astype(float)
    t_crit = np.where(n > 1, t.ppf(0.975, np.maximum(n - 1, 1)), np.nan)
    out = out.with_columns(pl.Series(f"{col}_t_crit", t_crit))
    return out.with_columns(
        (pl.col(f"{col}_mean") + pl.col(f"{col}_t_crit") * pl.col(f"{col}_stderr")).alias(f"{col}_ci_upper"),
        (pl.col(f"{col}_mean") - pl.col(f"{col}_t_crit") * pl.col(f"{col}_stderr")).alias(f"{col}_ci_lower"),
    )


def snakemake_main() -> None:  # noqa: D103
    data_path = str(snakemake.input["data"])
    reference_path = str(snakemake.input["reference"])
    thresholds_path = str(snakemake.input["thresholds"])
    out_summary = str(snakemake.output["summary"])
    out_per_run = str(snakemake.output["per_run"])
    out_meta = str(snakemake.output["meta"])

    objective_symbol = str(snakemake.params.objective_symbol)
    constraint_symbols = list(snakemake.params.constraint_symbols)
    ct_level = str(snakemake.params.ct_level)
    eps = float(snakemake.config["hv_eps_percent"])
    ref = 1.0 + eps

    job_label = (
        f"{snakemake.wildcards.problem_name}/{snakemake.wildcards.mode}"
        f"/ct{snakemake.wildcards.ctlevel}/psize{snakemake.wildcards.psize}"
    )

    with open(thresholds_path, encoding="utf-8") as fh:  # noqa: PTH123
        thr_doc = yaml.safe_load(fh)
    thresholds = {c: float(thr_doc["levels"][ct_level].get(c, 0.0)) for c in constraint_symbols}
    f_strict = float(thr_doc["objective_optimum"])
    tol = float(thr_doc.get("eps", 1e-9))
    f_col = f"{objective_symbol}_min"

    # Best-known reference optima
    with open(reference_path, encoding="utf-8") as fh:  # noqa: PTH123
        ref_doc = yaml.safe_load(fh)["levels"][ct_level]
    f_joint = ref_doc["joint"]["f"]
    gain_joint = None if f_joint is None else f_strict - f_joint
    joint_ok = gain_joint is not None and gain_joint > tol

    single_ref: dict[str, dict] = {}
    for j in constraint_symbols:
        f_j = ref_doc["single"][j]["f"]
        gain_j = None if f_j is None else f_strict - f_j
        contributes = thresholds[j] > 0.0 and gain_j is not None and gain_j > tol
        single_ref[j] = {
            "threshold": thresholds[j],
            "f_relaxed": f_j,
            "gain": gain_j,
            "source": ref_doc["single"][j].get("source"),
            "contributes": bool(contributes),
        }
    contributing = [j for j in constraint_symbols if single_ref[j]["contributes"]]

    print(
        f"[{job_label}] f*(0)={f_strict:.6g}, joint gain={gain_joint}, single contributing={contributing}",
        flush=True,
    )

    df = pl.read_parquet(data_path).select(["run", "generation", f_col, *constraint_symbols])
    generations = sorted(df["generation"].unique().to_list())
    max_gen = max(generations)
    base = df.select(["run", "generation"]).unique().sort(["run", "generation"])

    # Joint end points
    joint_mask = pl.all_horizontal([pl.col(c) <= thresholds[c] for c in constraint_symbols])
    joint_bsf = _best_so_far(df, joint_mask, f_col, "joint_best_so_far")
    per = base.join(joint_bsf, on=["run", "generation"], how="left")
    if joint_ok:
        per = per.with_columns(
            ((pl.col("joint_best_so_far") - f_joint) / gain_joint).clip(lower_bound=0.0).alias("joint_gap")
        )
    else:
        per = per.with_columns(pl.lit(None, dtype=pl.Float64).alias("joint_gap"))

    # Single end points and single curve, per contributing constraint
    gap_cols: list[str] = []
    hv_cols: list[str] = []
    for j in contributing:
        mask_j = pl.all_horizontal([pl.col(c) <= (thresholds[c] if c == j else 0.0) for c in constraint_symbols])
        f_j = single_ref[j]["f_relaxed"]
        gain_j = single_ref[j]["gain"]

        bsf_j = _best_so_far(df, mask_j, f_col, f"best_so_far_{j}")
        per = per.join(bsf_j, on=["run", "generation"], how="left").with_columns(
            ((pl.col(f"best_so_far_{j}") - f_j) / gain_j).clip(lower_bound=0.0).alias(f"gap_{j}")
        )
        gap_cols.append(f"gap_{j}")

        hv_j = _hv_per_run_gen(df, mask_j, f_col, j, f_j, gain_j, thresholds[j], ref, f"hv_{j}")
        per = per.join(hv_j, on=["run", "generation"], how="left")
        hv_cols.append(f"hv_{j}")
        print(f"[{job_label}] constraint {j} done", flush=True)

    if contributing:
        # Null propagates: a run contributes only once it has a value for every constraint.
        gap_expr = pl.col(gap_cols[0])
        hv_expr = pl.col(hv_cols[0])
        for g, h in zip(gap_cols[1:], hv_cols[1:], strict=True):
            gap_expr = gap_expr + pl.col(g)
            hv_expr = hv_expr + pl.col(h)
        per = per.with_columns(
            (gap_expr / len(contributing)).alias("single_gap"),
            (hv_expr / len(contributing)).alias("single_hv"),
        )
    else:
        per = per.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("single_gap"),
            pl.lit(None, dtype=pl.Float64).alias("single_hv"),
        )

    # Per-generation summary
    summary = pl.DataFrame({"generation": generations})
    for m in [*METRICS, *gap_cols, *hv_cols]:
        summary = summary.join(_summarize(per, m), on="generation", how="left")
    summary.sort("generation").write_parquet(out_summary)

    # Per-run final generation
    per.filter(pl.col("generation") == max_gen).select(
        ["run", *METRICS, "joint_best_so_far", *gap_cols, *hv_cols]
    ).sort("run").write_parquet(out_per_run)

    meta = {
        "ct_level": ct_level,
        "eps": eps,
        "tolerance": tol,
        "f_strict": f_strict,
        "joint": {
            "f_relaxed": f_joint,
            "gain": gain_joint,
            "contributes": bool(joint_ok),
            "source": ref_doc["joint"].get("source"),
        },
        "single": single_ref,
        "contributing_constraints": contributing,
    }
    with open(out_meta, "w", encoding="utf-8") as fh:  # noqa: PTH123
        yaml.safe_dump(meta, fh, sort_keys=False)
    print(f"[{job_label}] written {out_summary}", flush=True)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err
    snakemake_main()
