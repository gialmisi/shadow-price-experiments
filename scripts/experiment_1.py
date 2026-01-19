"""Experiment script, called by Snakemake."""

from datetime import datetime

import polars as pl
import yaml
from snakemake.script import snakemake
from utils import PROBLEM_BUILDERS

from desdeo.emo import (
    algorithms,
    crossover,
    generator,
    mutation,
    scalar_selection,
    selection,
    termination,
)
from desdeo.emo.hooks.archivers import Archive
from desdeo.problem import Problem


def run_nsga2_with_mode(  # noqa: PLR0913
    problem: Problem,
    mode: str,
    constraints: dict[str, float],
    pop_size: int,
    n_generations: int,
    objective_symbol: str = "f_1",
) -> Archive:
    """Run the NSGA-II style EA once for a given constraint-handling mode."""
    nsga2_options = algorithms.nsga2_options()

    nsga2_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=snakemake.config["xover_probability"],
        xover_distribution=snakemake.config["xover_distribution"],
    )
    nsga2_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=1.0 / len(problem.variables),
        distribution_index=snakemake.config["distribution_index"],
    )
    nsga2_options.template.mate_selection = scalar_selection.TournamentSelectionOptions(
        name="TournamentSelection",
        tournament_size=snakemake.config["tournament_size"],
        winner_size=pop_size,
    )
    nsga2_options.template.selection = selection.SingleObjectiveConstrainedRankingSelectorOptions(
        target_objective_symbol=objective_symbol,
        constraints=constraints,
        population_size=pop_size,
        mode=mode,
    )
    nsga2_options.template.generator = generator.LHSGeneratorOptions(n_points=pop_size)
    nsga2_options.template.termination = termination.MaxGenerationsTerminatorOptions(max_generations=n_generations)

    solver, extras = algorithms.emo_constructor(emo_options=nsga2_options, problem=problem)

    archive = Archive(problem=problem, publisher=extras.publisher)
    extras.publisher.auto_subscribe(archive)
    extras.publisher.register_topics(archive.provided_topics[archive.verbosity], archive.__class__.__name__)

    # Run optimization
    _ = solver()

    return archive


def single_run(  # noqa: PLR0913
    problem: Problem,
    mode: str,
    constraints: dict[str, float],
    pop_size: int,
    n_generations: int,
    objective_symbol: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run one EA and return (best_so_far_per_generation, all_solutions)."""
    archive = run_nsga2_with_mode(
        problem=problem,
        mode=mode,
        constraints=constraints,
        pop_size=pop_size,
        n_generations=n_generations,
        objective_symbol=objective_symbol,
    )

    return archive.solutions


def snakemake_main():
    """Entry point when called from Snakemake via `script:`."""
    problem_name = snakemake.wildcards.problem_name
    mode = snakemake.wildcards.mode
    pop_size = int(snakemake.wildcards.psize)

    ct_level = str(snakemake.params.ct_level)

    n_generations = snakemake.params.n_generations
    n_runs = snakemake.params.n_runs
    objective_symbol = snakemake.params.objective_symbol
    constraint_symbols = list(snakemake.params.constraint_symbols)

    thresholds_path = str(snakemake.input["thresholds"])
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_doc = yaml.safe_load(f)

    level_constraints = dict(thresholds_doc["levels"][ct_level])
    constraints = {sym: float(level_constraints[sym]) for sym in constraint_symbols if sym in level_constraints}

    out_path = str(snakemake.output[0])

    try:
        problem_fun = PROBLEM_BUILDERS[problem_name]
    except KeyError as err:
        raise ValueError(f"Unknown problem '{problem_name}'. Add it to PROBLEM_BUILDERS in run_experiment.py.") from err
    problem = problem_fun()

    solutions_results: pl.DataFrame | None = None

    for run_idx in range(n_runs):
        if run_idx % max(1, n_runs // 10) == 0:
            print(
                f"[{problem_name}] mode={mode}, ct_levels={constraints}({ct_level}), pop_size={pop_size}: run {run_idx + 1}/{n_runs}"
            )

        solutions = single_run(
            problem=problem,
            mode=mode,
            constraints=constraints,
            pop_size=pop_size,
            n_generations=n_generations,
            objective_symbol=objective_symbol,
        ).with_columns(pl.lit(run_idx).alias("run"))

        solutions_results = solutions if solutions_results is None else pl.concat([solutions_results, solutions])

    now_str = datetime.now().isoformat()

    parameters = f"""
        problem: {problem_name}
        mode: {mode}
        ct_level: {ct_level}
        pop_size: {pop_size}
        n_generations: {n_generations}
        n_runs: {n_runs}
        constraints: {constraints}
        objective_symbol: {objective_symbol}
        """

    solutions_results.write_parquet(
        out_path,
        metadata={
            "notes": parameters,
            "created_at": now_str,
        },
    )

    print(f"[{problem_name}] mode={mode}, ct_level={ct_level}, pop_size={pop_size}: written {out_path}")


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit(
            "This script is intended to be run via Snakemake's `script:` directive, which injects a `snakemake` object."
        ) from err

    snakemake_main()
