"""Generate LaTeX appendix sections for experiment figures.

Reads experiment_config.yaml and emits LaTeX figure blocks for each
(metric, problem) combination.  Output goes to stdout so it can be
redirected to a .tex file::

    uv run python scripts/generate_appendix_tex.py > appendix_figures.tex
"""

import yaml

CONFIG_PATH = "experiment_config.yaml"

METRICS = [
    {
        "key": "best_so_far",
        "section_title": "Best found value per generation",
        "label_prefix": "appendix_best_value",
        "caption": (
            "Best feasible objective found so far for {prob}."
            " Solid/dashed/dotted lines show the mean across runs;"
            " shaded bands show the 95\\% confidence interval."
        ),
    },
    {
        "key": "hv",
        "section_title": "Hypervolume per generation",
        "label_prefix": "appendix_hypervolume",
        "caption": (
            "Cumulative hypervolume of the non-dominated threshold-feasible archive"
            " for {prob}, computed over the relaxation-gain box: the objective axis"
            " spans $[f^{{*}}_{{\\text{{relaxed}}}}, f^{{*}}_{{\\text{{strict}}}}]$"
            " and each active constraint axis spans $[0, \\tau_c]$, with strictly"
            " feasible coordinates clipped to the box floor. Higher is better."
            " (Falls back to the shadow-feasible-front box when no strict-feasible"
            " reference set exists.)"
        ),
    },
    {
        "key": "shadow_price_diff",
        "section_title": "Shadow price difference per generation",
        "label_prefix": "appendix_shadow_price_diff",
        "caption": (
            "Shadow price (difference between best feasible and best"
            " threshold-feasible objective) for {prob}."
            " The horizontal line shows the approximate shadow price"
            " computed from the known optima."
            " The other lines show how well each mode matches this approximation."
        ),
    },
]


def _display_name(problem: str) -> str:
    """Pretty problem name for captions: g2 -> G02, pressure_vessel -> Pressure Vessel."""
    import re

    m = re.fullmatch(r"g(\d+)", problem)
    if m:
        return f"G{int(m.group(1)):02d}"
    return problem.replace("_", " ").title()


def _fig_label(problem: str, metric_key: str) -> str:
    return f"fig:{problem}_{metric_key}"


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    problems = cfg["summary_problems"]
    n_gen = cfg["n_generations"]
    n_runs = cfg["n_runs"]
    psizes_str = "-".join(map(str, cfg["population_sizes"]))

    lines: list[str] = []

    for metric in METRICS:
        key = metric["key"]
        lines.append(f"\\section{{{metric['section_title']}}}")
        lines.append(f"\\label{{sec:{metric['label_prefix']}}}")

        for prob in problems:
            fname = f"figures/{prob}__{key}_gen{n_gen}_runs{n_runs}_psizes{psizes_str}.pdf"
            label = _fig_label(prob, key)
            caption = metric["caption"].format(prob=_display_name(prob))
            lines.append("\\begin{figure}")
            lines.append("    \\centering")
            lines.append(f"    \\includegraphics[width=1.0\\linewidth]{{{fname}}}")
            lines.append(f"    \\caption{{{caption}}}")
            lines.append(f"    \\label{{{label}}}")
            lines.append("\\end{figure}")
            lines.append("")

        lines.append("\\FloatBarrier")
        lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
