"""Generate the figure sections of the supplementary material.

Reads experiment_config.yaml and emits one LaTeX section per metric, with one figure
per problem, for the per-generation convergence plots produced by the pipeline. Output
goes to stdout so it can be redirected to a .tex file that the manuscript's
supplement.tex inputs::

    uv run python scripts/generate_supplement_tex.py > supplement_figures.tex
"""

import yaml

CONFIG_PATH = "experiment_config.yaml"

_LINES = (
    " Lines show the mean across runs; shaded bands show the 95\\% confidence interval."
    " Rows are population sizes, columns are threshold levels."
)

METRICS = [
    {
        "key": "best_so_far",
        "section_title": "Best-so-far feasible objective",
        "label": "supp_best_value",
        "caption": (
            "Best strictly feasible objective found so far for {prob}."
            " The horizontal line marks the known optimum." + _LINES
        ),
    },
    {
        "key": "single_gap",
        "section_title": "End points under single relaxation:\\ relative gap",
        "label": "supp_single_gap",
        "caption": (
            "Relative optimality gap of the best-so-far single-relaxed objective for {prob},"
            " averaged over the constraints with a positive single-relaxation gain."
            " Lower is better." + _LINES
        ),
    },
    {
        "key": "single_hv",
        "section_title": "Whole curve under single relaxation:\\ hypervolume",
        "label": "supp_single_hv",
        "caption": (
            "Hypervolume of the non-dominated archive under single relaxation for {prob},"
            " averaged over the constraints with a positive single-relaxation gain."
            " Higher is better." + _LINES
        ),
    },
    {
        "key": "joint_gap",
        "section_title": "End points under joint relaxation:\\ relative gap",
        "label": "supp_joint_gap",
        "caption": (
            "Relative optimality gap of the best-so-far threshold-feasible objective"
            " under joint relaxation for {prob}. Lower is better." + _LINES
        ),
    },
    {
        "key": "hv",
        "section_title": "Whole curve under joint relaxation:\\ hypervolume",
        "label": "supp_joint_hv",
        "caption": (
            "Hypervolume of the non-dominated threshold-feasible archive under joint"
            " relaxation for {prob}. Higher is better." + _LINES
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
    return f"fig:supp_{problem}_{metric_key}"


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
        lines.append(f"\\label{{sec:{metric['label']}}}")

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
