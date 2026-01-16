configfile: "experiment_config.yaml"

import os

RESULTS_DIR = config["results_dir"]
DATA_GEN_SCRIPT = config["data_gen_script"]
FRONT_GEN_SCRIPT = config["front_gen_script"]
SUMMARY_PROBLEMS = config["summary_problems"]
SUMMARY_STATS_SCRIPT = config["summary_stats_script"]
EXT = config["result_extension"]

PROBLEM_BY_NAME = {p["name"]: p for p in config["problems"]}

CT_LEVELS = ["low", "med", "high"]

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
        # summary statistics
        [
            os.path.join(
                RESULTS_DIR,
                "summary",
                f"{combo['problem_name']}_{combo['mode']}_gen{config['n_generations']}_runs{config['n_runs']}_ct{combo["ct_level"]}_psize{combo['population_size']}_summary.{EXT}",
            )
            for combo in param_combinations()
            if combo["problem_name"] in SUMMARY_PROBLEMS
        ]

rule generate_data:
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
        constraint_thresholds=lambda wc: thresholds_for_level(PROBLEM_BY_NAME[wc.problem_name], wc.ctlevel),
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


rule summary_statistics:
    input:
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
        constraint_thresholds=lambda wc: thresholds_for_level(PROBLEM_BY_NAME[wc.problem_name], wc.ctlevel),
        ct_level=lambda wc: wc.ctlevel
    script:
        SUMMARY_STATS_SCRIPT