configfile: "experiment_config.yaml"

import os

RESULTS_DIR = config["results_dir"]
DATA_GEN_SCRIPT = config["data_gen_script"]
FRONT_GEN_SCRIPT = config["front_gen_script"]
SUMMARY_PROBLEMS = config["summary_problems"]
SUMMARY_STATS_SCRIPT = config["summary_stats_script"]
THRESHOLDS_SCRIPT = config["thresholds_script"]
THRESH_PCTS = config["threshold_improvement_percentages"]
EXT = config["result_extension"]
PLOT_SCRIPT = config["plot_script"]
MODE_TABLE_SCRIPT = config["mode_table_script"]

SCENARIO_SCRIPT = config["scenario_script"]
REFERENCE_SCRIPT = config["reference_script"]
SUPPLEMENT_SCRIPT = config["supplement_script"]

# Metrics computed by summary_statistics.py (per-generation summary + per-run final values)
SUMMARY_METRIC_KEYS = ["best_so_far", "hv", "shadow_gen_best", "shadow_best_so_far", "shadow_price_diff"]
# Metrics computed by scenario_metrics.py (the other three cells of the 2x2 scenario grid)
SCENARIO_METRIC_KEYS = ["joint_gap", "single_gap", "single_hv"]
# Metrics with figures and comparison tables (one per cell of the scenario grid + optimizer performance)
METRIC_KEYS = ["best_so_far", "hv"] + SCENARIO_METRIC_KEYS
# Figures collected in the supplementary material, in order
SUPPLEMENT_METRIC_KEYS = ["best_so_far", "single_gap", "single_hv", "joint_gap", "hv"]
PROBLEM_BY_NAME = {p["name"]: p for p in config["problems"]}
CT_LEVELS = ["low", "med", "high"]

wildcard_constraints:
    problem_name="(" + "|".join(SUMMARY_PROBLEMS) + ")",
    metric_key="(" + "|".join(METRIC_KEYS) + ")",
    mode="(" + "|".join(config["modes"]) + ")",
    ctlevel="(" + "|".join(CT_LEVELS) + ")",
    psize="[0-9]+",
    n_generations="[0-9]+",
    n_runs="[0-9]+"


def thresholds_for_level(prob: dict, level: str) -> dict:
    key = f"constraint_thresholds_{level}"
    try:
        return prob[key]
    except KeyError as err:
        raise KeyError(f"Problem '{prob.get('name', '<unknown>')}' missing key {key}") from err


def param_combinations():
    """Cartesian sum of the wanted parameters."""
    for prob in config["problems"]:
        pname = prob["name"]
        for mode in config["modes"]:
            for level in CT_LEVELS:
                    for pop_size in config["population_sizes"]:
                        yield {
                            "problem_name": pname,
                            "mode": mode,
                            "ct_level": level,
                            "population_size": pop_size,
                        }


def summary_stem(problem_name: str, mode: str, ct: str, psize) -> str:
    return os.path.join(
        RESULTS_DIR,
        "summary",
        f"{problem_name}_{mode}_gen{config['n_generations']}_runs{config['n_runs']}_ct{ct}_psize{psize}",
    )


def metric_files(problem_name: str, mode: str, ct: str, psize, metric_key: str) -> tuple[str, str]:
    """(per-generation summary, per-run final values) holding `metric_key`."""
    stem = summary_stem(problem_name, mode, ct, psize)
    if metric_key in SCENARIO_METRIC_KEYS:
        return f"{stem}_scenario_summary.{EXT}", f"{stem}_scenario_per_run.{EXT}"
    return f"{stem}_summary.{EXT}", f"{stem}_per_run.{EXT}"


def summary_inputs_and_meta(problem_name: str, metric_key: str):
    meta = []
    paths = []
    for psize in config["population_sizes"]:
        for ct in CT_LEVELS:
            for mode in config["modes"]:
                path, _ = metric_files(problem_name, mode, ct, psize, metric_key)
                paths.append(path)
                meta.append({"path": path, "psize": psize, "ctlevel": ct, "mode": mode})
    return paths, meta


def all_summary_inputs_and_meta(metric_key: str):
    """All summary paths and metadata across all problems (for mode comparison tables)."""
    paths, meta = [], []
    for pname in SUMMARY_PROBLEMS:
        for psize in config["population_sizes"]:
            for ct in CT_LEVELS:
                for mode in config["modes"]:
                    summary_path, per_run_path = metric_files(pname, mode, ct, psize, metric_key)
                    paths.extend([summary_path, per_run_path])
                    meta.append({
                        "path": summary_path,
                        "per_run_path": per_run_path,
                        "problem": pname,
                        "psize": psize,
                        "ctlevel": ct,
                        "mode": mode,
                    })
    return paths, meta


rule all:
    input:
        # experiment data
        [
        os.path.join(
                RESULTS_DIR,
                "data",
                f"{combo["problem_name"]}_{combo['mode']}_gen{config['n_generations']}_runs{config['n_runs']}_ct{combo['ct_level']}_psize{combo['population_size']}.{EXT}",
            )
            for combo in param_combinations()
        ] +
        # Pareto front data
        [
        os.path.join(
                RESULTS_DIR,
                "fronts",
                f"{p["name"]}_gen{config['n_generations_front']}_psize{config['population_size_front']}.{EXT}",
            )
            for p in config["problems"]
        ] +
        # Compute constraint thresholds
        [
            os.path.join(
                RESULTS_DIR, "thresholds", f"{p['name']}.yaml"
            ) for p in config["problems"]
        ] +
        # Best-known reference optima for every relaxation
        [
            os.path.join(RESULTS_DIR, "reference", f"{p}.yaml") for p in SUMMARY_PROBLEMS
        ] +
        # summary statistics
        [
            os.path.join(
                RESULTS_DIR,
                "summary",
                f"{combo['problem_name']}_{combo['mode']}_gen{config['n_generations']}_runs{config['n_runs']}_ct{combo["ct_level"]}_psize{combo['population_size']}_summary.{EXT}",
            )
            for combo in param_combinations()
            if combo["problem_name"] in SUMMARY_PROBLEMS
        ] +
        # scenario metrics (joint gap, single gap, single hypervolume)
        [
            summary_stem(combo['problem_name'], combo['mode'], combo['ct_level'], combo['population_size'])
            + f"_scenario_summary.{EXT}"
            for combo in param_combinations()
            if combo["problem_name"] in SUMMARY_PROBLEMS
        ] +
        # plots (one PDF per problem per metric)
        [
            os.path.join(
                RESULTS_DIR, "figures",
                f"{p['name']}__{m}_gen{config['n_generations']}_runs{config['n_runs']}_psizes{'-'.join(map(str, config['population_sizes']))}.pdf",
            )
            for p in config["problems"]
            for m in METRIC_KEYS
            if p["name"] in SUMMARY_PROBLEMS
        ] +
        # mode comparison tables
        [
            os.path.join(RESULTS_DIR, "tables", f"mode_comparison_{m}.tex")
            for m in METRIC_KEYS
        ] +
        # supplementary material LaTeX (figure pages)
        [os.path.join(RESULTS_DIR, "tables", "supplement_figures.tex")]


rule generate_data:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml")
    output:
        os.path.join(
            RESULTS_DIR,
            "data",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}." + EXT,
        )
    log:
        os.path.join(
            RESULTS_DIR,
            "log",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}.log"
        )
    params:
        n_generations=lambda wc: config["n_generations"],
        n_runs=lambda wc: config["n_runs"],
        constraint_symbols=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbols"],
        objective_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["objective_symbol"],
        ct_level=lambda wc: wc.ctlevel
    script:
        DATA_GEN_SCRIPT

rule generate_pareto_front:
    output:
        os.path.join(
                RESULTS_DIR,
                "fronts",
                "{problem_name}_gen" + str(config["n_generations_front"]) + "_psize" + str(config["population_size_front"]) + '.' + EXT,
            )
    params:
        constraint_symbols=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbols"],
        objective_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["objective_symbol"]

    script:
        FRONT_GEN_SCRIPT

rule compute_constraint_thresholds:
    output:
        os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml")
    params:
        problem=lambda wc: PROBLEM_BY_NAME[wc.problem_name],
        percents=lambda wc: config["threshold_improvement_percentages"],
    log:
        os.path.join(RESULTS_DIR, "log", "{problem_name}_thresholds.log")
    script:
        THRESHOLDS_SCRIPT

rule summary_statistics:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml"),
        reference=os.path.join(RESULTS_DIR, "reference", "{problem_name}.yaml"),
        data=os.path.join(
            RESULTS_DIR,
            "data",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}." + EXT,
        )
    output:
        os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_summary." + EXT,
        ),
        os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_summary_meta.yaml",
        ),
        os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_per_run." + EXT,
        )
    log:
        os.path.join(
            RESULTS_DIR,
            "log",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_summary.log"
        )
    params:
        n_generations=lambda wc: config["n_generations"],
        n_runs=lambda wc: config["n_runs"],
        constraint_symbols=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbols"],
        objective_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["objective_symbol"],
        ct_level=lambda wc: wc.ctlevel
    script:
        SUMMARY_STATS_SCRIPT

rule compute_reference_optima:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml"),
        front=os.path.join(
            RESULTS_DIR,
            "fronts",
            "{problem_name}_gen" + str(config["n_generations_front"]) + "_psize" + str(config["population_size_front"]) + "." + EXT,
        ),
        data=lambda wc: [
            os.path.join(
                RESULTS_DIR,
                "data",
                f"{wc.problem_name}_{mode}_gen{config['n_generations']}_runs{config['n_runs']}_ct{ct}_psize{max(config['population_sizes'])}.{EXT}",
            )
            for mode in config["modes"]
            for ct in CT_LEVELS
        ]
    output:
        os.path.join(RESULTS_DIR, "reference", "{problem_name}.yaml")
    log:
        os.path.join(RESULTS_DIR, "log", "{problem_name}_reference.log")
    params:
        problem=lambda wc: PROBLEM_BY_NAME[wc.problem_name],
        n_run_starts=lambda wc: config["reference_n_run_starts"],
        n_random_starts=lambda wc: config["reference_n_random_starts"],
        maxiter=lambda wc: config["reference_maxiter"],
        feasibility_tol=lambda wc: config["reference_feasibility_tol"],
        seed=lambda wc: config["base_seed"],
    script:
        REFERENCE_SCRIPT

rule scenario_metrics:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml"),
        reference=os.path.join(RESULTS_DIR, "reference", "{problem_name}.yaml"),
        data=os.path.join(
            RESULTS_DIR,
            "data",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}." + EXT,
        )
    output:
        summary=os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_scenario_summary." + EXT,
        ),
        per_run=os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_scenario_per_run." + EXT,
        ),
        meta=os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_scenario_meta.yaml",
        )
    log:
        os.path.join(
            RESULTS_DIR,
            "log",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_scenario.log"
        )
    params:
        constraint_symbols=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbols"],
        objective_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["objective_symbol"],
        ct_level=lambda wc: wc.ctlevel
    script:
        SCENARIO_SCRIPT

rule plot_summary_figures:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml"),
        reference=os.path.join(RESULTS_DIR, "reference", "{problem_name}.yaml"),
        summaries=lambda wc: summary_inputs_and_meta(wc.problem_name, wc.metric_key)[0],
    output:
        os.path.join(
            RESULTS_DIR,
            "figures",
            "{problem_name}__{metric_key}_gen"
            + str(config["n_generations"])
            + "_runs"
            + str(config["n_runs"])
            + "_psizes"
            + "-".join(map(str, config["population_sizes"]))
            + ".pdf",
        )
    params:
        problem=lambda wc: wc.problem_name,
        metric_key=lambda wc: wc.metric_key,
        population_sizes=lambda wc: config["population_sizes"],
        ct_levels=lambda wc: CT_LEVELS,
        modes=lambda wc: config["modes"],
        n_generations=lambda wc: config["n_generations"],
        n_runs=lambda wc: config["n_runs"],
        plotting=lambda wc: config.get("plotting", {}),
        summaries_meta=lambda wc: summary_inputs_and_meta(wc.problem_name, wc.metric_key)[1],
    script:
        PLOT_SCRIPT


rule mode_comparison_table:
    input:
        summaries=lambda wc: all_summary_inputs_and_meta(wc.metric_key)[0],
        references=[os.path.join(RESULTS_DIR, "reference", f"{p}.yaml") for p in SUMMARY_PROBLEMS],
    output:
        parquet=os.path.join(RESULTS_DIR, "tables", "mode_comparison_{metric_key}.parquet"),
        latex=os.path.join(RESULTS_DIR, "tables", "mode_comparison_{metric_key}.tex"),
        stats_parquet=os.path.join(RESULTS_DIR, "tables", "mode_comparison_{metric_key}_stats.parquet"),
        stats_latex=os.path.join(RESULTS_DIR, "tables", "mode_comparison_{metric_key}_stats.tex"),
    params:
        summaries_meta=lambda wc: all_summary_inputs_and_meta(wc.metric_key)[1],
        modes=lambda wc: config["modes"],
        population_sizes=lambda wc: config["population_sizes"],
        ct_levels=lambda wc: CT_LEVELS,
        problems=lambda wc: SUMMARY_PROBLEMS,
        n_generations=lambda wc: config["n_generations"],
        metric_key=lambda wc: wc.metric_key,
        reference_dir=os.path.join(RESULTS_DIR, "reference"),
    script:
        MODE_TABLE_SCRIPT

rule generate_supplement:
    input:
        figures=[
            os.path.join(
                RESULTS_DIR, "figures",
                f"{p}__{m}_gen{config['n_generations']}_runs{config['n_runs']}_psizes{'-'.join(map(str, config['population_sizes']))}.pdf",
            )
            for p in SUMMARY_PROBLEMS
            for m in SUPPLEMENT_METRIC_KEYS
        ]
    output:
        os.path.join(RESULTS_DIR, "tables", "supplement_figures.tex")
    shell:
        "uv run python {SUPPLEMENT_SCRIPT} > {output}"
