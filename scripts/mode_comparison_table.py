r"""Produce mode-comparison tables with Wilcoxon signed-rank statistical testing.

For each (problem, population_size, threshold_level) cell the table shows
the final-generation metric mean for each mode (Baseline / Relaxed / Ranking).
Values significantly better than Baseline (paired Wilcoxon, alpha=0.025 after
Bonferroni correction) are set in \\textbf{bold}.

A separate detailed statistical table reports W, p-value, rank-biserial r,
and significance for each comparison.
"""

from math import floor, log10
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import wilcoxon

try:
    from snakemake.script import snakemake  # snakemake <= 8.x
except ImportError:
    pass  # snakemake >= 9.x injects `snakemake` via the script preamble

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

_PER_RUN_COL = {
    "best_so_far": "run_best_so_far",
    "shadow_best_so_far": "shadow_best_so_far",
    "shadow_gen_best": "shadow_gen_best",
    "hv": "hv",
    "shadow_price_diff": "shadow_price_diff",
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

ALPHA = 0.10
# Three pairwise comparisons among the three modes (baseline/relaxed/ranking) per cell.
BONFERRONI_TESTS = 3
ALPHA_CORRECTED = ALPHA / BONFERRONI_TESTS


def _sigfig(val: float, n: int = 3, max_decimals: int | None = None) -> str:
    """Format *val* to *n* significant figures, optionally capping decimals."""
    if np.isnan(val):
        return "--"
    if val == 0:
        return "0"
    mag = floor(log10(abs(val)))
    prec = max(n - 1 - mag, 0)
    if max_decimals is not None:
        prec = min(prec, max_decimals)
    return f"{val:.{prec}f}"


def _shared_exponent(values: list[float | None]) -> int:
    """Compute a shared base-10 exponent for a group of values."""
    abs_vals = [abs(v) for v in values if v is not None and v != 0 and not np.isnan(v)]
    if not abs_vals:
        return 0
    max_abs = max(abs_vals)
    return floor(log10(max_abs))


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation as effect size for paired Wilcoxon.

    r = 1 - (2T) / (n(n+1)/2)  where T = W+ (sum of positive ranks).
    """
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    t_plus = float(ranks[d > 0].sum())
    return 1.0 - (2.0 * t_plus) / (n * (n + 1) / 2.0)


def _wilcoxon_test(
    baseline_vals: np.ndarray,
    alt_vals: np.ndarray,
    higher_is_better: bool,
    closer_to_zero: bool,
) -> dict:
    """Run paired Wilcoxon signed-rank test: is alt significantly better than baseline?

    Returns dict with W, p_value, significant, rank_biserial, direction.
    """
    if closer_to_zero:
        # Better = closer to zero, so compare |alt| vs |baseline|
        d = np.abs(baseline_vals) - np.abs(alt_vals)  # positive if alt is closer to zero
    elif higher_is_better:
        d = alt_vals - baseline_vals  # positive if alt is higher (better)
    else:
        d = baseline_vals - alt_vals  # positive if alt is lower (better)

    # Drop zero differences (Wilcoxon requirement)
    d_nonzero = d[d != 0]
    n_nonzero = len(d_nonzero)

    if n_nonzero < 10:
        return {
            "W": None,
            "p_value": None,
            "significant": False,
            "rank_biserial": None,
            "direction": "insufficient_data",
            "n_nonzero": n_nonzero,
        }

    # Two-sided: significance can go either way; the `direction` field tells which mode is better.
    try:
        stat, p_value = wilcoxon(d_nonzero, alternative="two-sided")
    except ValueError:
        return {
            "W": None,
            "p_value": None,
            "significant": False,
            "rank_biserial": None,
            "direction": "test_error",
            "n_nonzero": n_nonzero,
        }

    # Effect size
    r = _rank_biserial(
        np.abs(baseline_vals) if closer_to_zero else alt_vals,
        np.abs(alt_vals) if closer_to_zero else baseline_vals,
    )

    # Direction: which mode is better on average?
    if closer_to_zero:
        direction = "alt_better" if np.abs(alt_vals).mean() < np.abs(baseline_vals).mean() else "baseline_better"
    elif higher_is_better:
        direction = "alt_better" if alt_vals.mean() > baseline_vals.mean() else "baseline_better"
    else:
        direction = "alt_better" if alt_vals.mean() < baseline_vals.mean() else "baseline_better"

    return {
        "W": float(stat),
        "p_value": float(p_value),
        "significant": p_value < ALPHA_CORRECTED,
        "rank_biserial": float(r),
        "direction": direction,
        "n_nonzero": n_nonzero,
    }


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
    out_stats_parquet = str(snakemake.output["stats_parquet"])
    out_stats_latex = str(snakemake.output["stats_latex"])

    col_name = _METRIC_COL[metric_key]
    per_run_col = _PER_RUN_COL[metric_key]
    higher_is_better = metric_key in _HIGHER_IS_BETTER
    closer_to_zero = metric_key in _CLOSER_TO_ZERO

    # Load true shadow prices from thresholds YAMLs (for error computation)
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

    # 1. Read final-generation aggregated values and per-run values
    rows: list[dict] = []
    per_run_data: dict[tuple[str, int, str, str], np.ndarray] = {}

    for meta in summaries_meta:
        # Aggregated summary value
        df = pl.read_parquet(meta["path"])
        final = df.filter(pl.col("generation") == n_generations)
        val = float(final[col_name][0]) if final.height > 0 and final[col_name][0] is not None else None

        # Per-run values
        pr_path = meta.get("per_run_path")
        run_vals = None
        if pr_path:
            pr_df = pl.read_parquet(pr_path)
            run_vals = pr_df[per_run_col].to_numpy().astype(float)
            if closer_to_zero:
                ref = true_shadow_price.get((meta["problem"], meta["ctlevel"]))
                if ref is not None:
                    run_vals = run_vals - ref
            per_run_data[(meta["problem"], meta["psize"], meta["ctlevel"], meta["mode"])] = run_vals

        # For closer_to_zero metrics, display mean absolute error
        if closer_to_zero and run_vals is not None:
            val = float(np.nanmean(np.abs(run_vals)))
        elif closer_to_zero and val is not None:
            ref = true_shadow_price.get((meta["problem"], meta["ctlevel"]))
            if ref is not None:
                val = abs(val - ref)

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

    # 2. Statistical testing: all pairwise comparisons, then determine the winner set per cell.
    #
    # Winner rule: take the mode with the best mean. If it is significantly better than at
    # least one other mode, it is the winner; additionally mark every mode that is NOT
    # significantly worse than it (i.e. tied-for-best). If the best-mean mode is not
    # significantly better than any other mode, there is no winner and nothing is marked.
    from itertools import combinations

    stats_rows: list[dict] = []
    # winners[(problem, psize, ct_level)] = set of winning modes (possibly empty)
    winners: dict[tuple[str, int, str], set[str]] = {}

    for prob in problems:
        for ps in population_sizes:
            for ct in ct_levels:
                cell_vals = {m: per_run_data.get((prob, ps, ct, m)) for m in modes}

                # Per-mode metric mean (mean absolute error for closer-to-zero metrics).
                cell_means: dict[str, float | None] = {}
                for m in modes:
                    v = cell_vals[m]
                    cell_means[m] = float(np.nanmean(np.abs(v) if closer_to_zero else v)) if v is not None else None

                # Pairwise Wilcoxon tests. pair_better[(a, b)] = True iff a is significantly better than b.
                pair_better: dict[tuple[str, str], bool] = {}
                for a, b in combinations(modes, 2):
                    av, bv = cell_vals[a], cell_vals[b]
                    if av is None or bv is None:
                        stats_rows.append(
                            {
                                "problem": prob,
                                "population_size": ps,
                                "ct_level": ct,
                                "comparison": f"{b}_vs_{a}",
                                "W": None,
                                "p_value": None,
                                "significant": False,
                                "rank_biserial": None,
                                "direction": "missing_data",
                                "n_nonzero": 0,
                            }
                        )
                        pair_better[(a, b)] = pair_better[(b, a)] = False
                        continue

                    # Paired alignment; drop pairs where either value is NaN.
                    n = min(len(av), len(bv))
                    a2, b2 = av[:n], bv[:n]
                    valid = ~(np.isnan(a2) | np.isnan(b2))
                    # _wilcoxon_test(reference, candidate): direction "alt_better" => candidate (b) is better.
                    result = _wilcoxon_test(
                        a2[valid],
                        b2[valid],
                        higher_is_better=higher_is_better,
                        closer_to_zero=closer_to_zero,
                    )
                    stats_rows.append(
                        {
                            "problem": prob,
                            "population_size": ps,
                            "ct_level": ct,
                            "comparison": f"{b}_vs_{a}",
                            **result,
                        }
                    )
                    pair_better[(a, b)] = result["significant"] and result["direction"] == "baseline_better"
                    pair_better[(b, a)] = result["significant"] and result["direction"] == "alt_better"

                # Winner set.
                valid_modes = [m for m in modes if cell_means[m] is not None]
                if not valid_modes:
                    winners[(prob, ps, ct)] = set()
                    continue
                best = (max if higher_is_better else min)(valid_modes, key=lambda m: cell_means[m])
                others = [m for m in valid_modes if m != best]
                best_beats = {m: pair_better.get((best, m), False) for m in others}
                if any(best_beats.values()):
                    # A significant winner exists: best + every mode not significantly worse than it.
                    winners[(prob, ps, ct)] = {best} | {m for m in others if not best_beats[m]}
                else:
                    winners[(prob, ps, ct)] = set()

    stats_df = pl.DataFrame(stats_rows)
    stats_df.write_parquet(out_stats_parquet)

    # 3. Build main LaTeX table (bold = winner; see winner rule above).
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

    # Count winning cells per (ct_level, mode) column.
    win_counts: dict[tuple[str, str], int] = {(ct, m): 0 for ct in ct_levels for m in modes}
    for (prob, ps, ct), wset in winners.items():
        for m in wset:
            win_counts[(ct, m)] = win_counts.get((ct, m), 0) + 1

    lines: list[str] = []
    lines.append("% Auto-generated by mode_comparison_table.py")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    abbrev_legend = ", ".join(f"{a} = {m.capitalize()}" for a, m in zip(mode_abbrevs, modes))
    if closer_to_zero:
        caption_metric = "shadow price mean absolute error"
        caption_note = "Lower is better."
    else:
        caption_metric = "\\texttt{" + metric_key.replace("_", "\\_") + "}"
        caption_note = ""
    lines.append(
        "\\caption{Final-generation mean "
        + caption_metric
        + " by mode ("
        + abbrev_legend
        + ").  \\textbf{Bold}: the winning mode(s), i.e. the best-mean mode when it is"
        + " significantly better than at least one other mode, together with any mode not"
        + " significantly worse than it; no mark when no significant winner exists"
        + f" (paired two-sided Wilcoxon, $\\alpha={ALPHA}$, Bonferroni-corrected)."
        + (" " + caption_note if caption_note else "")
        + "}"
    )
    lines.append("\\normalsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    total_data_cols = 1 + n_ct * n_modes
    _MODE_SHADES = ["gray!0", "gray!10", "gray!20"]
    mode_col_spec = " ".join(
        ">{" + "\\columncolor{" + _MODE_SHADES[i % len(_MODE_SHADES)] + "}}c" for i in range(n_modes)
    )
    col_spec = "p{0.7cm} @{}r@{\\hspace{8pt}} " + " ".join([mode_col_spec] * n_ct)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")

    lines.append("\\toprule")
    parts = ["", ""]
    for ct in ct_levels:
        ct_label = _CT_LABEL.get(ct, ct)
        parts.append(f"\\multicolumn{{{n_modes}}}{{c}}{{\\hrulefill\\;{ct_label}\\;\\hrulefill}}")
    lines.append(" & ".join(parts) + " \\\\")

    rotated = [f"\\rotatebox{{90}}{{\\small {a}}}" for a in mode_abbrevs]
    parts = ["", "$n$"]
    for _ in ct_levels:
        parts.extend(rotated)
    lines.append(" & ".join(parts) + " \\\\")
    lines.append("\\midrule")

    # Shared exponent per problem
    problem_exponent: dict[str, int] = {}
    for prob in problems:
        all_vals: list[float | None] = []
        for ps in population_sizes:
            for ct in ct_levels:
                all_vals.extend(idx.get((prob, ps, ct), {}).values())
        problem_exponent[prob] = _shared_exponent(all_vals)

    # Data rows
    for prob_idx, prob in enumerate(problems):
        label = _PROBLEM_LABEL.get(prob, prob)
        exp = problem_exponent[prob]
        if prob_idx > 0:
            lines.append(f"\\cmidrule{{2-{total_data_cols + 1}}}")
        scale = 10.0 ** (-exp) if exp != 0 else 1.0
        mdp = 3
        for row_in_block, ps in enumerate(population_sizes):
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
                cell_winners = winners.get((prob, ps, ct), set())
                for m in modes:
                    v = mode_vals.get(m)
                    if v is None:
                        data_cells.append("--")
                        continue
                    display_v = v * scale if exp != 0 else v
                    s = _sigfig(display_v, 3, max_decimals=mdp)
                    # Bold if this mode is a winner in this cell.
                    if m in cell_winners:
                        s = f"\\textbf{{{s}}}"
                    data_cells.append(s)
            parts_row: list[str] = [prob_cell] + data_cells
            lines.append(" & ".join(parts_row) + " \\\\")

    # Bottom row: significant win counts per column.
    # Per (ct level, mode): number of cells in which that mode is a winner.
    lines.append("\\midrule")
    count_cells: list[str] = ["\\multicolumn{2}{r}{\\small Wins}"]
    for ct in ct_levels:
        for m in modes:
            count_cells.append(f"\\textbf{{{win_counts.get((ct, m), 0)}}}")
    lines.append(" & ".join(count_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_latex, "w", encoding="utf-8") as f:  # noqa: PTH123
        f.write("\n".join(lines) + "\n")

    # 4. Detailed statistical results LaTeX table
    stats_lines: list[str] = []
    stats_lines.append("% Auto-generated statistical details by mode_comparison_table.py")
    stats_lines.append("\\begin{longtable}{llrlllrrrl}")
    stats_lines.append(
        "\\caption{Wilcoxon signed-rank test results: all pairwise mode comparisons"
        " ($\\alpha=" + str(ALPHA) + "$, Bonferroni-corrected to $" + f"{ALPHA_CORRECTED:.4g}" + "$). "
        "Direction names the better mode.}\\\\"
    )
    stats_lines.append("\\toprule")
    stats_lines.append("Problem & $n$ & $th$ & Comparison & $W$ & $p$ & $r$ & Sig. & Direction \\\\")
    stats_lines.append("\\midrule")
    stats_lines.append("\\endfirsthead")
    stats_lines.append("\\toprule")
    stats_lines.append("Problem & $n$ & $th$ & Comparison & $W$ & $p$ & $r$ & Sig. & Direction \\\\")
    stats_lines.append("\\midrule")
    stats_lines.append("\\endhead")

    _COMP_LABEL = {
        f"{b}_vs_{a}": f"{_MODE_ABBREV.get(b, b)} vs {_MODE_ABBREV.get(a, a)}" for a, b in combinations(modes, 2)
    }

    for row in stats_rows:
        prob_label = _PROBLEM_LABEL.get(row["problem"], row["problem"])
        comp_label = _COMP_LABEL.get(row["comparison"], row["comparison"])
        w_str = f"{row['W']:.0f}" if row["W"] is not None else "--"
        p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "--"
        r_str = f"{row['rank_biserial']:.3f}" if row["rank_biserial"] is not None else "--"
        sig_str = "Yes" if row["significant"] else "No"
        # The comparison "{b}_vs_{a}" maps "alt_better" -> b, "baseline_better" -> a.
        cand, _, ref = row["comparison"].partition("_vs_")
        if row["direction"] == "alt_better":
            dir_str = f"{_MODE_ABBREV.get(cand, cand)} better"
        elif row["direction"] == "baseline_better":
            dir_str = f"{_MODE_ABBREV.get(ref, ref)} better"
        else:
            dir_str = row["direction"].replace("_", " ")
        stats_lines.append(
            f"{prob_label} & {row['population_size']} & {row['ct_level']} & "
            f"{comp_label} & {w_str} & {p_str} & {r_str} & {sig_str} & {dir_str} \\\\"
        )

    stats_lines.append("\\bottomrule")
    stats_lines.append("\\end{longtable}")

    with open(out_stats_latex, "w", encoding="utf-8") as f:  # noqa: PTH123
        f.write("\n".join(stats_lines) + "\n")

    print(f"Written {out_parquet}, {out_latex}, {out_stats_parquet}, {out_stats_latex}", flush=True)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err
    snakemake_main()
