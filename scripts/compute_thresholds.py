"""Emit per-problem threshold YAML from hand-picked values in the experiment config.

Constraint thresholds (low/med/high) are read directly from `manual_thresholds`
on each problem entry in experiment_config.yaml (rationale: see
results/figures/threshold_picker/hand_picked.md). The reference optima for the
relaxed problems are computed separately by scripts/compute_reference_optima.py.
"""

from typing import Any

import yaml

try:
    from snakemake.script import snakemake  # snakemake <= 8.x
except ImportError:
    pass  # snakemake >= 9.x injects `snakemake` via the script preamble

LEVELS = ["low", "med", "high"]


def snakemake_main() -> None:
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

    # Mark every constraint as "manual" so summary_statistics' evidence-based
    # active-mask filter (which drops only random_sampling sources with n<10) passes.
    evidence = {c: {"n": 1, "max_violation": None, "source": "manual"} for c in constraint_symbols}

    payload = {
        "problem": problem_name,
        "objective_symbol": objective_symbol,
        "f_col": f_col,
        "objective_optimum": f_opt,
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
