"""Compute constraint thresholds (low/med/high) from a Pareto/front parquet."""

import math
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
            f"Problem '{problem_name}' missing 'objective_optimum'. "
            "Add it to the YAML (recommended for reproducibility)."
        )
    optimal_value = float(prob["objective_optimum"])

    df_front = pl.read_parquet(front_path)

    # Keep only finite objective + constraint rows
    cols_needed = [f_col] + constraint_symbols
    for c in cols_needed:
        if c not in df_front.columns:
            raise KeyError(
                f"Column '{c}' missing from front parquet for '{problem_name}'. Available columns: {df_front.columns}"
            )

    front = df_front.select(cols_needed).drop_nulls().filter(pl.col(f_col).is_finite())
    for c in constraint_symbols:
        front = front.filter(pl.col(c).is_finite())

    if front.height == 0:
        raise ValueError(f"Front parquet '{front_path}' has no usable finite rows after filtering.")

    # Normalize objective based on (ideal, nadir) of the front objective column
    # minimization is assumed
    f_ideal = float(front.select(pl.min(f_col)).item())
    f_nadir = float(front.select(pl.max(f_col)).item())
    if math.isclose(f_nadir, f_ideal):
        raise ValueError(f"Cannot normalize: f_nadir == f_ideal == {f_ideal} for '{problem_name}'.")

    front_norm = front.with_columns(((pl.col(f_col) - f_ideal) / (f_nadir - f_ideal)).alias("f_norm"))

    norm_opt = (optimal_value - f_ideal) / (f_nadir - f_ideal)

    # Targets: percentage (i.e., 1%, 5%, 10%) improvement relative to optimum
    targets = [norm_opt * (1.0 - float(p)) for p in percents]

    # Find closest front row per target
    levels_out: dict[str, dict[str, float]] = {}
    for level, target in zip(LEVELS, targets, strict=True):
        closest = (
            front_norm.with_columns((pl.col("f_norm") - pl.lit(target)).abs().alias("delta")).sort("delta").head(1)
        )
        if closest.height != 1:
            raise RuntimeError("Unexpected: failed to select closest row.")

        # Threshold per constraint = constraint value at this row (aligned across constraints)
        level_thresholds = {c: float(closest.select(pl.col(c)).item()) for c in constraint_symbols}
        levels_out[level] = level_thresholds

    payload = {
        "problem": problem_name,
        "objective_symbol": objective_symbol,
        "f_col": f_col,
        "objective_optimum": optimal_value,
        "percents": percents,
        "levels": levels_out,
    }

    # Write out as yaml
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    print(f"[{problem_name}] wrote thresholds to {out_path}: {levels_out}")


if __name__ == "__main__":
    snakemake_main()
