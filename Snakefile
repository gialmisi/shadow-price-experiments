configfile: "experiment_config.yaml"

import os

RESULTS_DIR = config["results_dir"]
SCRIPT = config["experiment_script"]
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
                combo["problem_name"],
                f"{combo["problem_name"]}_{combo['mode']}_ct{constraint_threshold_str(combo['constraint_threshold'])}_psize{combo['population_size']}.{EXT}",
            )
            for combo in param_combinations()
        ]

rule run_experiment:
    output:
        os.path.join(
            RESULTS_DIR,
            "{problem_name}",
            "{mode}_ct{ct}_psize{psize}." + EXT,
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
        SCRIPT