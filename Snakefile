configfile: "experiment_config.yaml"

import os

RESULTS_DIR = config["results_dir"]
DATA_GEN_SCRIPT = config["data_gen_script"]
FRONT_GEN_SCRIPT = config["front_gen_script"]
EXT = config["result_extension"]

PROBLEM_BY_NAME = {p["name"]: p for p in config["problems"]}

def param_combinations():
    """Cartesian sum of the wanted parameters."""
    for prob in config["problems"]:
        pname = prob["name"]
        for mode in config["modes"]:
            for ct in prob["constraint_thresholds"]:
                for pop_size in config["population_sizes"]:
                    yield {
                        "problem_name": pname,
                        "mode": mode,
                        "constraint_threshold": ct,
                        "population_size": pop_size,
                    }

def constraint_threshold_str(ct: float) -> str:
    """Constraint threshold to file name friendly str.
    
    Replaces dashes '-' with 'm' and dots '.' with 'p'.
    """
    return f"{ct}".replace("-", "m").replace(".", "p")


rule all:
    input:
        [
        os.path.join(
                RESULTS_DIR,
                f"{combo["problem_name"]}_{combo['mode']}_gen{config['n_generations']}_runs{config['n_runs']}_ct{constraint_threshold_str(combo['constraint_threshold'])}_psize{combo['population_size']}.{EXT}",
            )
            for combo in param_combinations()
        ] +
        [
        os.path.join(
                RESULTS_DIR,
                "fronts",
                f"{p["name"]}_gen{config['n_generations_front']}_psize{config['population_size_front']}.{EXT}",
            )
            for p in config["problems"]
        ]

rule generate_data:
    output:
        os.path.join(
            RESULTS_DIR,
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ct}_psize{psize}." + EXT,
        )
    log:
        os.path.join(
            RESULTS_DIR,
            "log",
            "{problem_name}_{mode}_gen{n_generations}_runs{n_runs}_ct{ct}_psize{psize}.log"
        )
    params:
        n_generations=lambda wc: config["n_generations"],
        n_runs=lambda wc: config["n_runs"],
        constraint_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbol"],
        objective_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["objective_symbol"],
        constraint_threshold=lambda wc: next(
            ct
            for ct in PROBLEM_BY_NAME[wc.problem_name]["constraint_thresholds"]
            if constraint_threshold_str(ct) == wc.ct
        ),
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
        constraint_symbol=lambda wc: PROBLEM_BY_NAME[wc.problem_name]["constraint_symbol"],

    script:
        FRONT_GEN_SCRIPT