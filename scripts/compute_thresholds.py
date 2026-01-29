"""Compute constraint thresholds (low/med/high) from a Pareto/front parquet.

Rule:
- Use known true optimum f_opt (from config).
- Consider points on the front with objective better than optimum (f < f_opt - eps).
- For each constraint c_j:
    if there exist such points with c_j > 0, define max_violation_j = max(c_j) over that subset
    thresholds for low/med/high are p * max_violation_j for p in percents (e.g. 0.01/0.05/0.10)
  else:
    omit that constraint (no thresholds produced for it).
"""

from typing import Any

import polars as pl
import yaml
from snakemake.script import snakemake

LEVELS = ["low", "med", "high"]


def snakemake_main() -> None:
    front_path = str(snakemake.input["front"])
    out_path = str(snakemake.output[0])

    prob: dict[str, Any] = dict(snakemake.params.problem)
    percents = list(snakemake.params.percents)

    if len(percents) != 3:
        raise ValueError(f"Expected exactly 3 percents for low/med/high, got: {percents}")

    problem_name = prob["name"]
    objective_symbol = prob["objective_symbol"]
    constraint_symbols = list(prob["constraint_symbols"])

    f_col = f"{objective_symbol}_min"

    if "objective_optimum" not in prob:
        raise KeyError(
            f"Problem '{problem_name}' missing 'objective_optimum'. This threshold rule requires the known optimum."
        )
    f_opt = float(prob["objective_optimum"])

    # avoid treating floating noise as "better than optimum"
    eps = max(1e-12, 1e-6 * max(1.0, abs(f_opt)))

    df = pl.read_parquet(front_path)

    # Require needed columns
    cols_needed = [f_col, *constraint_symbols]
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(
                f"Column '{c}' missing from front parquet for '{problem_name}'. Available columns: {df.columns}"
            )

    front = df.select(cols_needed).drop_nulls()
    front = front.filter(pl.col(f_col).is_finite())
    for c in constraint_symbols:
        front = front.filter(pl.col(c).is_finite())

    if front.height == 0:
        raise ValueError(f"Front parquet '{front_path}' has no usable finite rows after filtering.")

    # Approximated optimum from reference front: best strictly feasible objective (c <= 0)
    feasible_front = front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in constraint_symbols]))

    if feasible_front.height == 0:
        objective_approximated_optimum = None
    else:
        objective_approximated_optimum = float(feasible_front.select(pl.min(f_col)).item())

    # Improvement-eligible subset, better than optimum
    eligible = front.filter(pl.col(f_col) < (pl.lit(f_opt) - pl.lit(eps)))

    # constraints may be omitted if no evidence
    levels_out: dict[str, dict[str, float]] = {lvl: {} for lvl in LEVELS}
    evidence: dict[str, dict[str, Any]] = {}

    for c in constraint_symbols:
        # Evidence for this constraint, eligible points with positive violation
        ev = eligible.filter(pl.col(c) > 0.0)

        n = ev.height
        if n == 0:
            # No positive violations for the current constraint, all levels set to 0.0 (default eligibility)
            evidence[c] = {"n": 0, "max_violation": None}

            for lvl in LEVELS:
                levels_out[lvl][c] = 0.0

            continue

        max_v = float(ev.select(pl.max(c)).item())
        if max_v <= 0.0:
            evidence[c] = {"n": n, "max_violation": max_v}

            continue

        evidence[c] = {"n": n, "max_violation": max_v}

        for lvl, p in zip(LEVELS, percents, strict=True):
            t = float(p) * max_v

            if t <= 0.0:
                continue

            levels_out[lvl][c] = t

    # Objective shadow optimum from the reference front
    def threshold_feasible_expr(level: str) -> pl.Expr:
        # If a constraint has no level threshold produced, we treat it as 0.0 (normal feasibility)
        return pl.all_horizontal(
            [pl.col(c) <= pl.lit(float(levels_out.get(level, {}).get(c, 0.0))) for c in constraint_symbols]
        )

    shadow_optimum_by_level: dict[str, float | None] = {}
    for lvl in LEVELS:
        cand = front.filter(threshold_feasible_expr(lvl))
        if cand.height == 0:
            shadow_optimum_by_level[lvl] = None
        else:
            shadow_optimum_by_level[lvl] = float(cand.select(pl.min(f_col)).item())

    payload = {
        "problem": problem_name,
        "objective_symbol": objective_symbol,
        "f_col": f_col,
        "objective_optimum": f_opt,
        "objective_approximated_optimum": objective_approximated_optimum,
        "objective_shadow_optima": shadow_optimum_by_level,
        "eps": eps,
        "percents": [float(p) for p in percents],
        "rule": "thresholds_from_positive_violations_on_optimum_improvers",
        "evidence": evidence,
        "levels": levels_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:  # noqa: PTH123
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"[{problem_name}] wrote thresholds to {out_path}: {levels_out}")


if __name__ == "__main__":
    snakemake_main()
