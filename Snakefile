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

METRIC_KEYS = ["best_so_far", "hv"]
PROBLEM_BY_NAME = {p["name"]: p for p in config["problems"]}
CT_LEVELS = ["low", "med", "high"]

wildcard_constraints:
    problem_name="(" + "|".join(SUMMARY_PROBLEMS) + ")",
    metric_key="(" + "|".join(METRIC_KEYS) + ")"


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


def summary_inputs_and_meta(problem_name: str, metric_key: str):
    # metric_key isn't needed for file paths, but keeps interface symmetric
    meta = []
    paths = []
    for psize in config["population_sizes"]:
        for ct in CT_LEVELS:
            for mode in config["modes"]:
                path = os.path.join(
                    RESULTS_DIR,
                    "summary",
                    f"{problem_name}_{mode}_gen{config['n_generations']}_runs{config['n_runs']}_ct{ct}_psize{psize}_summary.{EXT}",
                )
                paths.append(path)
                meta.append({"path": path, "psize": psize, "ctlevel": ct, "mode": mode})
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
        # plots (one PDF per problem per metric)
        [
            os.path.join(
                RESULTS_DIR, "figures",
                f"{p['name']}__{m}_gen{config['n_generations']}_runs{config['n_runs']}_psizes{'-'.join(map(str, config['population_sizes']))}.pdf",
            )
            for p in config["problems"]
            for m in METRIC_KEYS
            if p["name"] in SUMMARY_PROBLEMS
        ]


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
    input:
        front=os.path.join(
            RESULTS_DIR,
            "fronts",
            "{problem_name}_gen" + str(config["n_generations_front"]) +
            "_psize" + str(config["population_size_front"]) + "." + EXT,
        )
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
        data=os.path.join(
            RESULTS_DIR,
            "data",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}." + EXT,
        ),
        front=os.path.join(
            RESULTS_DIR,
            "fronts",
            "{problem_name}_gen" + str(config["n_generations_front"]) + "_psize" + str(config["population_size_front"]) + "." + EXT,
        )
    output:
        os.path.join(
            RESULTS_DIR,
            "summary",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ctlevel}_psize{psize}_summary." + EXT,
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

rule plot_summary_figures:
    input:
        thresholds=os.path.join(RESULTS_DIR, "thresholds", "{problem_name}.yaml"),
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
