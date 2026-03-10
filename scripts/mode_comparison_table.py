"""Produce a mode-comparison parquet and LaTeX table.

For each (problem, population_size, threshold_level) cell the table shows
the final-generation metric mean for each mode (Baseline / Relaxed / Ranking),
with \\textbf{bold} for best and \\textit{italic} for worst.
"""

from math import floor, log10
from pathlib import Path

import polars as pl
import yaml
from snakemake.script import snakemake

# Metrics where higher is better (all others: lower is better)
_HIGHER_IS_BETTER = {"hv"}
# Metrics where closer to zero is better
_CLOSER_TO_ZERO = {"shadow_price_diff"}

_METRIC_COL = {
    "best_so_far": "run_best_so_far_mean",
    "shadow_best_so_far": "shadow_best_so_far_mean",
    "shadow_gen_best": "shadow_gen_best_mean",
    "hv": "hv_mean",
    "shadow_price_diff": "shadow_price_diff_mean",
}

_PROBLEM_LABEL = {
    "branin": "Branin",
    "mystery": "Mystery",
    "cantilevered_beam": "C.\\ beam",
    "pressure_vessel": "P.\\ vessel",
    "townsend": "Townsend",
    "g2": "G02",
    "g6": "G06",
    "g8": "G08",
    "g9": "G09",
    "g24": "G24",
}


def _sigfig(val: float, n: int = 3, max_decimals: int | None = None) -> str:
    """Format *val* to *n* significant figures, optionally capping decimals."""
    if val == 0:
        return "0"
    mag = floor(log10(abs(val)))
    prec = max(n - 1 - mag, 0)
    if max_decimals is not None:
        prec = min(prec, max_decimals)
    return f"{val:.{prec}f}"


def _shared_exponent(values: list[float | None]) -> int:
    """Compute a shared base-10 exponent for a group of values.

    Returns 0 when no scaling is needed.
    """
    abs_vals = [abs(v) for v in values if v is not None and v != 0]
    if not abs_vals:
        return 0
    max_abs = max(abs_vals)
    return floor(log10(max_abs))


def _fmt_triple(
    values: list[float | None],
    higher_is_better: bool,
    closer_to_zero: bool = False,
    sig: int = 3,
    max_decimals: int | None = None,
) -> list[str]:
    """Return a list of formatted strings (one per mode), bold best, italic worst."""
    valid = [v for v in values if v is not None]
    if not valid:
        return ["--"] * len(values)

    if closer_to_zero:
        abs_valid = [abs(v) for v in valid]
        best_abs = min(abs_valid)
        worst_abs = max(abs_valid)
        ref = max(worst_abs, 1e-15)
        all_same = abs(best_abs - worst_abs) <= 1e-10 * ref
    else:
        best = max(valid) if higher_is_better else min(valid)
        worst = min(valid) if higher_is_better else max(valid)
        ref = max(abs(best), abs(worst), 1e-15)
        all_same = abs(best - worst) <= 1e-10 * ref

    parts: list[str] = []
    for v in values:
        if v is None:
            parts.append("--")
            continue
        s = _sigfig(v, sig, max_decimals=max_decimals)
        if not all_same:
            if closer_to_zero:
                if abs(abs(v) - best_abs) <= 1e-10 * ref:
                    s = f"\\textbf{{{s}}}"
                elif abs(abs(v) - worst_abs) <= 1e-10 * ref:
                    s = f"\\textit{{{s}}}"
            elif abs(v - best) <= 1e-10 * ref:
                s = f"\\textbf{{{s}}}"
            elif abs(v - worst) <= 1e-10 * ref:
                s = f"\\textit{{{s}}}"
        parts.append(s)
    return parts


def snakemake_main() -> None:  # noqa: D103
    summaries_meta = list(snakemake.params.summaries_meta)
    modes = list(snakemake.params.modes)
    population_sizes = sorted(snakemake.params.population_sizes)
    ct_levels = list(snakemake.params.ct_levels)
    problems = list(snakemake.params.problems)
    n_generations = int(snakemake.params.n_generations)
    metric_key = str(snakemake.params.metric_key)

    out_parquet = str(snakemake.output["parquet"])
    out_latex = str(snakemake.output["latex"])

    col_name = _METRIC_COL[metric_key]
    higher_is_better = metric_key in _HIGHER_IS_BETTER
    closer_to_zero = metric_key in _CLOSER_TO_ZERO

    # ------------------------------------------------------------------
    # Load true shadow prices from thresholds YAMLs (for error computation)
    # ------------------------------------------------------------------
    true_shadow_price: dict[tuple[str, str], float] = {}
    if closer_to_zero:
        thresholds_dir = Path(str(snakemake.params.thresholds_dir))
        for prob in problems:
            path = thresholds_dir / f"{prob}.yaml"
            with open(path, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            f_opt = doc.get("objective_optimum")
            shadow_optima = doc.get("objective_shadow_optima", {})
            for ct in ct_levels:
                s_opt = shadow_optima.get(ct)
                if f_opt is not None and s_opt is not None:
                    true_shadow_price[(prob, ct)] = float(f_opt) - float(s_opt)

    # ------------------------------------------------------------------
    # 1. Read final-generation value from every summary parquet
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for meta in summaries_meta:
        df = pl.read_parquet(meta["path"])
        final = df.filter(pl.col("generation") == n_generations)
        val = float(final[col_name][0]) if final.height > 0 and final[col_name][0] is not None else None
        # For shadow_price_diff: convert to error (distance from true shadow price)
        if closer_to_zero and val is not None:
            ref = true_shadow_price.get((meta["problem"], meta["ctlevel"]))
            if ref is not None:
                val = val - ref
        rows.append(
            {
                "problem": meta["problem"],
                "population_size": meta["psize"],
                "ct_level": meta["ctlevel"],
                "mode": meta["mode"],
                "value": val,
            }
        )

    result_df = pl.DataFrame(rows)
    result_df.write_parquet(out_parquet)

    # ------------------------------------------------------------------
    # 2. Build LaTeX table
    # ------------------------------------------------------------------
    # Index: {(problem, psize, ct_level) -> {mode -> value}}
    idx: dict[tuple[str, int, str], dict[str, float | None]] = {}
    for r in rows:
        key = (r["problem"], r["population_size"], r["ct_level"])
        idx.setdefault(key, {})[r["mode"]] = r["value"]

    n_ct = len(ct_levels)
    n_modes = len(modes)
    _MODE_ABBREV = {"baseline": "Bs", "relaxed": "Rx", "ranking": "Rk"}
    mode_abbrevs = [_MODE_ABBREV.get(m, m[:2].capitalize()) for m in modes]
    _CT_LABEL = {"low": "low", "med": "medium", "high": "high"}
    n_psizes = len(population_sizes)

    lines: list[str] = []
    lines.append("% Auto-generated by mode_comparison_table.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    abbrev_legend = ", ".join(f"{a} = {m.capitalize()}" for a, m in zip(mode_abbrevs, modes))
    if closer_to_zero:
        caption_metric = "shadow price error (approx.\\ $-$ true)"
        caption_note = "Closer to zero is better."
    else:
        caption_metric = "\\texttt{" + metric_key.replace("_", "\\_") + "}"
        caption_note = ""
    lines.append(
        "\\caption{Final-generation mean "
        + caption_metric
        + " by mode ("
        + abbrev_legend
        + ").  \\textbf{Bold} = best, \\textit{italic} = worst."
        + (" " + caption_note if caption_note else "")
        + "}"
    )
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    # columns: problem_name | n_pop | (n_modes per ct_level) * n_ct
    total_data_cols = 1 + n_ct * n_modes  # npop + mode columns
    total_cols = 1 + total_data_cols  # problem col + data cols
    _MODE_SHADES = ["gray!0", "gray!10", "gray!20"]
    mode_col_spec = " ".join(
        ">{" + "\\columncolor{" + _MODE_SHADES[i % len(_MODE_SHADES)] + "}}c" for i in range(n_modes)
    )
    col_spec = "p{0.7cm} @{}r@{\\hspace{4pt}} " + " ".join([mode_col_spec] * n_ct)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    # Header: threshold level labels and mode abbreviations
    lines.append("\\toprule")

    # --- low --- | --- medium --- | --- high --- (each spanning its mode group)
    parts = ["", ""]  # empty for problem col and npop col
    for ct in ct_levels:
        ct_label = _CT_LABEL.get(ct, ct)
        parts.append(f"\\multicolumn{{{n_modes}}}{{c}}{{\\hrulefill\\;{ct_label}\\;\\hrulefill}}")
    lines.append(" & ".join(parts) + " \\\\")

    # Sub-header: rotated mode abbreviations
    rotated = [f"\\rotatebox{{90}}{{\\small {a}}}" for a in mode_abbrevs]
    parts = ["", "$n$"]
    for _ in ct_levels:
        parts.extend(rotated)
    lines.append(" & ".join(parts) + " \\\\")
    lines.append("\\midrule")

    # Compute shared exponent per problem for scientific notation
    problem_exponent: dict[str, int] = {}
    for prob in problems:
        all_vals: list[float | None] = []
        for ps in population_sizes:
            for ct in ct_levels:
                all_vals.extend(idx.get((prob, ps, ct), {}).values())
        problem_exponent[prob] = _shared_exponent(all_vals)

    # Data rows: rotated problem name in leftmost col, then sub-rows per psize
    for prob_idx, prob in enumerate(problems):
        label = _PROBLEM_LABEL.get(prob, prob)
        exp = problem_exponent[prob]
        # Separator between problem blocks
        if prob_idx > 0:
            lines.append(f"\\cmidrule{{2-{total_data_cols + 1}}}")
        scale = 10.0 ** (-exp) if exp != 0 else 1.0
        mdp = 3 if exp != 0 else None  # cap mantissa decimals
        for row_in_block, ps in enumerate(population_sizes):
            # Problem name cell: multirow on first row, empty otherwise
            if row_in_block == 0:
                if exp != 0:
                    rot_content = f"\\shortstack[c]{{\\small {label} \\\\ \\scriptsize $(\\times 10^{{{exp}}})$}}"
                else:
                    rot_content = f"\\small {label}"
                prob_cell = f"\\multirow{{{n_psizes}}}{{*}}{{\\rotatebox{{90}}{{{rot_content}}}}}"
            else:
                prob_cell = ""
            data_cells: list[str] = [str(ps)]
            for ct in ct_levels:
                mode_vals = idx.get((prob, ps, ct), {})
                vals = [mode_vals.get(m) for m in modes]
                if exp != 0:
                    vals = [v * scale if v is not None else None for v in vals]
                data_cells.extend(_fmt_triple(vals, higher_is_better, closer_to_zero, max_decimals=mdp))
            parts: list[str] = [prob_cell] + data_cells
            lines.append(" & ".join(parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_latex, "w", encoding="utf-8") as f:  # noqa: PTH123
        f.write("\n".join(lines) + "\n")

    print(f"Written {out_parquet} and {out_latex}", flush=True)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err
    snakemake_main()
