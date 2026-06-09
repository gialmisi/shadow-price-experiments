"""Plot per-generation summary statistics for one problem as a psize x ct grid."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.ticker import FixedLocator, FuncFormatter
from snakemake.script import snakemake

METRICS: dict[str, dict[str, str]] = {
    "best_so_far": {
        "mean": "run_best_so_far_mean",
        "lower": "best_ci_lower",
        "upper": "best_ci_upper",
        "ytitle": "Best-so-far feasible objective",
        "title": "Best-so-far (feasible) objective",
        "slug": "best_so_far",
        "optimum_key": "objective_optimum",  # scalar
        "count_col": "best_n_feasible_runs",
    },
    "hv": {
        "mean": "hv_mean",
        "lower": "hv_ci_lower",
        "upper": "hv_ci_upper",
        "ytitle": "Relaxation-gain hypervolume",
        "title": "Relaxation-gain hypervolume",
        "slug": "hv",
        "optimum_key": "",  # no optimum line
        "count_col": "hv_n_runs",
    },
    "shadow_gen_best": {
        "mean": "shadow_gen_best_mean",
        "lower": "shadow_gen_best_ci_lower",
        "upper": "shadow_gen_best_ci_upper",
        "ytitle": "Best (threshold-feasible) objective in generation",
        "title": "Shadow price per generation (threshold-feasible)",
        "slug": "shadow_gen_best",
        "optimum_key": "objective_shadow_optima",  # dict by ct level
        "count_col": "shadow_gen_best_n_runs",
    },
    "shadow_best_so_far": {
        "mean": "shadow_best_so_far_mean",
        "lower": "shadow_best_ci_lower",
        "upper": "shadow_best_ci_upper",
        "ytitle": "Best-so-far (threshold-feasible) objective",
        "title": "Shadow price best-so-far (threshold-feasible)",
        "slug": "shadow_best_so_far",
        "optimum_key": "objective_shadow_optima",  # dict by ct level
        "count_col": "shadow_best_so_far_n_runs",
    },
    "shadow_price_diff": {
        "mean": "shadow_price_diff_mean",
        "lower": "shadow_price_diff_ci_lower",
        "upper": "shadow_price_diff_ci_upper",
        "ytitle": "Shadow price (difference)",
        "title": "Shadow price difference (feasible - threshold-feasible)",
        "slug": "shadow_price_diff",
        "optimum_key": "true_shadow_price",
        "count_col": "shadow_price_diff_n_runs",
    },
}


def _read_series(file_path: str, metric_spec: dict[str, str]) -> pl.DataFrame:
    cols = ["generation", metric_spec["mean"], metric_spec["lower"], metric_spec["upper"]]
    if metric_spec.get("count_col"):
        cols.append(metric_spec["count_col"])
    return pl.read_parquet(file_path).select(cols).sort("generation")


def _load_thresholds_doc(thresholds_yaml: str) -> dict:
    with open(thresholds_yaml, encoding="utf-8") as f:  # noqa: PTH123
        return yaml.safe_load(f)


def _get_optimum(doc: dict, metric_spec: dict[str, str], ctlevel: str) -> float | None:
    """Return optimum value for the given metric and ctlevel, or None if not applicable/missing."""
    key = str(metric_spec.get("optimum_key", "")).strip()
    if not key:
        return None

    if key == "objective_optimum":
        v = doc.get("objective_optimum")
        return None if v is None else float(v)

    if key == "objective_shadow_optima":
        d = doc.get("objective_shadow_optima")
        if not isinstance(d, dict):
            return None
        v = d.get(ctlevel, None)
        return None if v is None else float(v)

    if key == "true_shadow_price":
        f_opt = doc.get("objective_optimum")
        d = doc.get("objective_shadow_optima")
        if f_opt is None or not isinstance(d, dict):
            return None
        shadow_opt = d.get(ctlevel, None)
        if shadow_opt is None:
            return None
        return float(f_opt) - float(shadow_opt)

    raise KeyError(f"Unknown optimum_key='{key}' in metric spec.")


def snakemake_main() -> None:
    out_path = str(snakemake.output[0])

    thresholds_yaml = str(snakemake.input["thresholds"])
    # Keep for Snakemake tracking:
    summary_paths = list(map(str, snakemake.input["summaries"]))

    problem = str(snakemake.params["problem"])
    metric_key = str(snakemake.params["metric_key"])

    psizes = list(map(int, snakemake.params["population_sizes"]))
    ct_levels = list(map(str, snakemake.params["ct_levels"]))
    modes = list(map(str, snakemake.params["modes"]))

    n_generations = int(snakemake.params["n_generations"])
    n_runs = int(snakemake.params["n_runs"])

    plot_cfg: dict[str, Any] = dict(snakemake.params.get("plotting", {}))
    metric_spec = METRICS[metric_key]

    # Per-(problem, metric) y-scale override (e.g. "log" for g9:best_so_far)
    yscale_overrides = dict(plot_cfg.get("yscale_overrides", {}))
    yscale = str(yscale_overrides.get(f"{problem}:{metric_key}", "linear"))

    # Per-(problem, metric) inset zoom config
    inset_overrides = dict(plot_cfg.get("inset_overrides", {}))
    inset_cfg = inset_overrides.get(f"{problem}:{metric_key}")

    # Load thresholds doc once (for optimum lines)
    thr_doc = _load_thresholds_doc(thresholds_yaml)

    # Apply rcParams from config
    rcparams = dict(plot_cfg.get("rcparams", {}))
    if rcparams:
        plt.rcParams.update(rcparams)

    # Styles from config (with sensible defaults)
    palette = dict(plot_cfg.get("palette", {})) or {
        "black": "#000000",
        "blue": "#0072B2",
        "orange": "#E69F00",
        "red": "#D55E00",
    }
    mode_style = plot_cfg.get("mode_style", {}) or {
        "baseline": {"color": palette["black"], "linestyle": "-", "linewidth": 2.0, "label": "Baseline"},
        "relaxed": {"color": palette["blue"], "linestyle": "--", "linewidth": 2.0, "label": "Relaxed"},
        "ranking": {"color": palette["orange"], "linestyle": ":", "linewidth": 2.2, "label": "Ranking"},
    }

    ci_alpha = float(plot_cfg.get("ci_alpha", 0.18))
    gen_ticks_n = int(plot_cfg.get("gen_ticks", 7))

    fig_width = float(plot_cfg.get("fig_width", 11.0))
    row_height = float(plot_cfg.get("row_height", 3.2))

    # Build (psize, ctlevel, mode) -> path from Snakemake-provided metadata
    summaries_meta = list(snakemake.params["summaries_meta"])
    index: dict[tuple[int, str, str], str] = {}
    for item in summaries_meta:
        p = str(item["path"])
        # Safety: ensure Snakemake actually declared this file as input (avoid mismatches)
        if p not in summary_paths:
            continue
        index[(int(item["psize"]), str(item["ctlevel"]), str(item["mode"]))] = p

    opt_style = plot_cfg.get("optimum_line", {})
    opt_color = str(opt_style.get("color", palette.get("red", "#D55E00")))
    opt_ls = str(opt_style.get("linestyle", "-"))
    opt_lw = float(opt_style.get("linewidth", 1.6))
    opt_label = str(opt_style.get("label", "Optimum"))

    # Compute x_start per column (ct_level): first generation where all modes/psizes
    # in that column have count_col >= 10% of n_runs. Columns share x via sharex="col".
    count_col = metric_spec.get("count_col", "")
    min_count = max(2, int(n_runs * 0.1))
    x_start_by_col: dict[str, int] = dict.fromkeys(ct_levels, 0)
    if count_col:
        for ct in ct_levels:
            col_start = 0
            for psize in psizes:
                for mode in modes:
                    path = index.get((psize, ct, mode))
                    if path is None:
                        continue
                    sdf = pl.read_parquet(path).select(["generation", count_col]).sort("generation")
                    full = sdf.filter(pl.col(count_col) >= min_count)
                    if full.height > 0:
                        col_start = max(col_start, int(full["generation"][0]))
            x_start_by_col[ct] = col_start

    nrows, ncols = len(psizes), len(ct_levels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, row_height * nrows),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    if nrows == 1:
        axes = axes.reshape(1, -1)

    missing: list[tuple[str, int, str, str]] = []

    for i, psize in enumerate(psizes):
        for j, ct in enumerate(ct_levels):
            ax = axes[i, j]

            if i == 0:
                ax.set_title(f"$th = \\mathrm{{{ct}}}$")

            if yscale != "linear":
                ax.set_yscale(yscale)

            ax.grid(
                True,
                which="major",
                alpha=float(plot_cfg.get("grid_alpha", 0.25)),
                linewidth=float(plot_cfg.get("grid_lw", 0.6)),
            )

            x_start = x_start_by_col[ct]

            # Compute optimum early so it can serve as CI clip floor for log scale
            opt = _get_optimum(thr_doc, metric_spec, ct)

            for mode in modes:
                path = index.get((psize, ct, mode))
                if path is None:
                    missing.append((problem, psize, ct, mode))
                    continue
                df = _read_series(path, metric_spec).filter(pl.col("generation") >= x_start)

                ms = mode_style[mode]
                ax.plot(
                    df["generation"].to_numpy(),
                    df[metric_spec["mean"]].to_numpy(),
                    label=str(ms.get("label", mode)),
                    color=str(ms["color"]),
                    linestyle=str(ms["linestyle"]),
                    linewidth=float(ms["linewidth"]),
                    zorder=3,
                )
                ci_lo = df[metric_spec["lower"]].to_numpy()
                ci_hi = df[metric_spec["upper"]].to_numpy()
                if yscale == "log" and opt is not None and opt > 0:
                    ci_lo = np.clip(ci_lo, opt * 0.9, None)
                ax.fill_between(
                    df["generation"].to_numpy(),
                    ci_lo,
                    ci_hi,
                    color=str(ms["color"]),
                    alpha=ci_alpha,
                    linewidth=0,
                    zorder=2,
                )

            if opt is not None:
                ax.axhline(opt, color=opt_color, linestyle=opt_ls, linewidth=opt_lw, zorder=1, label=opt_label)

            gen_ticks = np.unique(np.round(np.linspace(x_start, n_generations, gen_ticks_n)).astype(int))
            eval_ticks = gen_ticks * psize

            # Inset zoom subplot (x-axis in evaluations, matching the outer plot)
            if inset_cfg is not None:
                ins_pos = list(map(float, inset_cfg["position"]))
                ins_ylim = list(map(float, inset_cfg["ylim"]))
                ins_eval_start = float(inset_cfg.get("xlim_evals_start", x_start * psize))
                ins_eval_end = n_generations * psize

                axins = ax.inset_axes(ins_pos)
                for mode in modes:
                    path = index.get((psize, ct, mode))
                    if path is None:
                        continue
                    ins_df = _read_series(path, metric_spec).filter(pl.col("generation") >= x_start)
                    ins_evals = ins_df["generation"].to_numpy() * psize
                    ms = mode_style[mode]
                    axins.plot(
                        ins_evals,
                        ins_df[metric_spec["mean"]].to_numpy(),
                        color=str(ms["color"]),
                        linestyle=str(ms["linestyle"]),
                        linewidth=float(ms["linewidth"]) * 0.8,
                    )
                    ins_ci_lo = np.clip(ins_df[metric_spec["lower"]].to_numpy(), ins_ylim[0], ins_ylim[1])
                    ins_ci_hi = np.clip(ins_df[metric_spec["upper"]].to_numpy(), ins_ylim[0], ins_ylim[1])
                    axins.fill_between(
                        ins_evals,
                        ins_ci_lo,
                        ins_ci_hi,
                        color=str(ms["color"]),
                        alpha=ci_alpha,
                        linewidth=0,
                    )
                if opt is not None:
                    axins.axhline(opt, color=opt_color, linestyle=opt_ls, linewidth=opt_lw * 0.8)
                axins.set_xlim(ins_eval_start, ins_eval_end)
                axins.set_ylim(ins_ylim)
                # Use outer plot's eval ticks filtered to inset range
                ins_ticks = eval_ticks[(eval_ticks >= ins_eval_start) & (eval_ticks <= ins_eval_end)]
                axins.xaxis.set_major_locator(FixedLocator(ins_ticks))
                axins.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
                axins.tick_params(labelsize=7)
                axins.grid(True, alpha=0.2, linewidth=0.4)
            ax.set_xlim(x_start, n_generations)
            ax.xaxis.set_major_locator(FixedLocator(gen_ticks))
            ax.tick_params(axis="x", which="both", labelbottom=False)

            secax = ax.secondary_xaxis(
                "bottom",
                functions=(lambda g, p=psize: g * p, lambda e, p=psize: e / p),
            )
            secax.xaxis.set_major_locator(FixedLocator(eval_ticks))
            secax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))
            secax.tick_params(axis="x", which="both", labelbottom=True)

            if j == 0:
                ax.set_ylabel(f"$N_\\mathrm{{pop}}$={psize}")

    fig.canvas.draw()
    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    x_center = (left + right) / 2

    suptitle_y = float(plot_cfg.get("suptitle_y", 1.05))
    fig.suptitle(
        f"{metric_spec['title']} for {problem} (generations: {n_generations}; runs: {n_runs})",
        x=x_center,
        ha="center",
        y=suptitle_y,
    )

    xlabel_y = float(plot_cfg.get("xlabel_y", -0.02))
    fig.text(
        x_center,
        xlabel_y,
        "Evaluations",
        ha="center",
        va="bottom",
        fontsize=plt.rcParams.get("figure.titlesize", plt.rcParams.get("axes.titlesize", 10)),
    )
    fig.supylabel(metric_spec["ytitle"])

    # Legend: deduplicate
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    handles_u, labels_u = [], []
    for h, lab in zip(handles, labels, strict=True):
        if lab in seen:
            continue
        seen.add(lab)
        handles_u.append(h)
        labels_u.append(lab)

    legend_cfg = dict(plot_cfg.get("legend", {}))
    legend_y = float(legend_cfg.get("y", 1.03))
    legend = fig.legend(
        handles_u,
        labels_u,
        loc="upper center",
        ncol=int(legend_cfg.get("ncol", len(labels_u))),
        frameon=bool(legend_cfg.get("frameon", True)),
        fancybox=bool(legend_cfg.get("fancybox", False)),
        bbox_to_anchor=(x_center, legend_y),
        bbox_transform=fig.transFigure,
    )

    if bool(legend_cfg.get("frameon", True)):
        frame = legend.get_frame()
        frame.set_facecolor(str(legend_cfg.get("facecolor", "white")))
        frame.set_edgecolor(str(legend_cfg.get("edgecolor", "black")))
        frame.set_alpha(float(legend_cfg.get("framealpha", 1.0)))
        frame.set_linewidth(float(legend_cfg.get("linewidth", 0.8)))

    if missing:
        print("Missing series (problem, psize, ctlevel, mode):")
        for m in missing:
            print("  ", m)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    snakemake_main()
