"""Emit per-problem threshold YAML from hand-picked values in the experiment config.

Constraint thresholds (low/med/high) are read directly from `manual_thresholds`
on each problem entry in experiment_config.yaml. The reference front is still
read here, but only to compute:

- objective_approximated_optimum: best objective on the strictly feasible front
- objective_shadow_optima:        best objective at each ct level's thresholds

Both are consumed by downstream plotting/statistics scripts. The threshold
values themselves no longer come from the front (rationale: see
results/figures/threshold_picker/hand_picked.md).
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
    problem_name = prob["name"]
    objective_symbol = prob["objective_symbol"]
    constraint_symbols = list(prob["constraint_symbols"])
    f_col = f"{objective_symbol}_min"

    if "objective_optimum" not in prob:
        raise KeyError(f"Problem '{problem_name}' missing 'objective_optimum'. Required to anchor the shadow optima.")
    f_opt = float(prob["objective_optimum"])
    eps = max(1e-12, 1e-6 * max(1.0, abs(f_opt)))

    manual = prob.get("manual_thresholds")
    if manual is None:
        raise KeyError(
            f"Problem '{problem_name}' missing 'manual_thresholds' in experiment_config.yaml. "
            "Hand-picked thresholds are required; see results/figures/threshold_picker/hand_picked.md."
        )
    missing_levels = [lvl for lvl in LEVELS if lvl not in manual]
    if missing_levels:
        raise KeyError(f"Problem '{problem_name}' manual_thresholds missing levels: {missing_levels} (need {LEVELS}).")

    levels_out: dict[str, dict[str, float]] = {}
    for lvl in LEVELS:
        level_dict = dict(manual[lvl])
        missing_cs = [c for c in constraint_symbols if c not in level_dict]
        if missing_cs:
            raise KeyError(f"Problem '{problem_name}' manual_thresholds[{lvl}] missing constraints: {missing_cs}.")
        levels_out[lvl] = {c: float(level_dict[c]) for c in constraint_symbols}

    # Reference front: still used for the approximated optimum and shadow optima
    df = pl.read_parquet(front_path)
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

    feasible_front = front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in constraint_symbols]))
    objective_approximated_optimum = (
        None if feasible_front.height == 0 else float(feasible_front.select(pl.min(f_col)).item())
    )

    def threshold_feasible_expr(level: str) -> pl.Expr:
        return pl.all_horizontal([pl.col(c) <= pl.lit(float(levels_out[level][c])) for c in constraint_symbols])

    shadow_optimum_by_level: dict[str, float | None] = {}
    for lvl in LEVELS:
        cand = front.filter(threshold_feasible_expr(lvl))
        shadow_optimum_by_level[lvl] = None if cand.height == 0 else float(cand.select(pl.min(f_col)).item())

    # Mark every constraint as "manual" so summary_statistics' evidence-based
    # active-mask filter (which drops only random_sampling sources with n<10) passes.
    evidence = {c: {"n": 1, "max_violation": None, "source": "manual"} for c in constraint_symbols}

    payload = {
        "problem": problem_name,
        "objective_symbol": objective_symbol,
        "f_col": f_col,
        "objective_optimum": f_opt,
        "objective_approximated_optimum": objective_approximated_optimum,
        "objective_shadow_optima": shadow_optimum_by_level,
        "eps": eps,
        "rule": "hand_picked_from_config",
        "evidence": evidence,
        "levels": levels_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:  # noqa: PTH123
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"[{problem_name}] wrote thresholds to {out_path}: {levels_out}")


if __name__ == "__main__":
    snakemake_main()
