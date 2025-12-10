"""Experiment script, called by Snakemake."""

from collections.abc import Callable
from datetime import datetime

import polars as pl
from snakemake.script import snakemake

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
from desdeo.problem.testproblems.single_objective import (
    mishras_bird_constrained,
    mystery_function,
    new_branin_function,
)

# Map problem names from config / Snakefile to constructors
PROBLEM_BUILDERS: dict[str, Callable[[], Problem]] = {
    "branin": new_branin_function,
    "mystery": mystery_function,
    "mishrasbird": mishras_bird_constrained,
}


def run_nsga2_with_mode(  # noqa: PLR0913
    problem: Problem,
    mode: str,
    constraint_threshold: float,
    pop_size: int,
    n_generations: int,
    constraint_symbol: str = "c_1",
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
        target_constraint_symbol=constraint_symbol,
        constraint_threshold=constraint_threshold,
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
    constraint_threshold: float,
    pop_size: int,
    n_generations: int,
    constraint_symbol: str,
    objective_symbol: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run one EA and return (best_so_far_per_generation, all_solutions)."""
    archive = run_nsga2_with_mode(
        problem=problem,
        mode=mode,
        constraint_threshold=constraint_threshold,
        pop_size=pop_size,
        n_generations=n_generations,
        constraint_symbol=constraint_symbol,
        objective_symbol=objective_symbol,
    )

    solutions = archive.solutions

    feasible = solutions.with_columns(
        pl.when(pl.col(constraint_symbol) <= 0)
        .then(pl.col(f"{objective_symbol}_min"))
        .otherwise(float("inf"))
        .alias(f"feasible_{objective_symbol}")
    )

    best_per_gen = (
        feasible.group_by("generation")
        .agg(pl.col(f"feasible_{objective_symbol}").min().alias(f"best_{objective_symbol}_this_gen"))
        .sort("generation")
    )

    best_so_far = best_per_gen.with_columns(best_per_gen[f"best_{objective_symbol}_this_gen"].cum_min().alias("best"))

    return best_so_far, solutions


def snakemake_main():
    """Entry point when called from Snakemake via `script:`."""
    problem_name = snakemake.wildcards.problem_name
    mode = snakemake.wildcards.mode
    pop_size = int(snakemake.wildcards.psize)

    ct_str = snakemake.wildcards.ct  # e.g. "0p5", mainly for logging if needed

    n_generations = snakemake.params.n_generations
    n_runs = snakemake.params.n_runs
    constraint_symbol = snakemake.params.constraint_symbol
    objective_symbol = snakemake.params.objective_symbol
    constraint_threshold = snakemake.params.constraint_threshold

    out_path = str(snakemake.output[0])

    try:
        problem_fun = PROBLEM_BUILDERS[problem_name]
    except KeyError as err:
        raise ValueError(f"Unknown problem '{problem_name}'. Add it to PROBLEM_BUILDERS in run_experiment.py.") from err
    problem = problem_fun()

    best_results: pl.DataFrame | None = None
    solutions_results: pl.DataFrame | None = None

    for run_idx in range(n_runs):
        if run_idx % max(1, n_runs // 10) == 0:
            print(
                f"[{problem_name}] mode={mode}, ct={constraint_threshold}, pop_size={pop_size}: run {run_idx + 1}/{n_runs}"
            )

        best, solutions = single_run(
            problem=problem,
            mode=mode,
            constraint_threshold=constraint_threshold,
            pop_size=pop_size,
            n_generations=n_generations,
            constraint_symbol=constraint_symbol,
            objective_symbol=objective_symbol,
        )

        best_results = best if best_results is None else pl.concat([best_results, best])
        solutions_results = solutions if solutions_results is None else pl.concat([solutions_results, solutions])

    now_str = datetime.now().isoformat()

    parameters = f"""
        problem: {problem_name}
        mode: {mode}
        constraint_threshold: {constraint_threshold} (encoded: {ct_str})
        pop_size: {pop_size}
        n_generations: {n_generations}
        n_runs: {n_runs}
        constraint_symbol: {constraint_symbol}
        objective_symbol: {objective_symbol}
        """

    combined = best_results

    stats = (
        combined.filter(pl.col("best").is_finite())
        .group_by("generation")
        .agg(
            [
                pl.col("best").mean().alias("best_mean"),
                pl.col("best").std().alias("best_std"),
            ]
        )
        .with_columns(
            [
                (pl.col("best_mean") + pl.col("best_std")).alias("best_upper"),
                (pl.col("best_mean") - pl.col("best_std")).alias("best_lower"),
            ]
        )
        .sort("generation")
    ).filter(
        pl.all_horizontal(
            pl.col("best_mean").is_finite(),
            pl.col("best_upper").is_finite(),
            pl.col("best_lower").is_finite(),
        )
    )

    solutions_and_stats = solutions_results.join(stats, on="generation", how="left")

    solutions_and_stats.write_parquet(
        out_path,
        metadata={
            "notes": parameters,
            "created_at": now_str,
        },
    )

    print(f"[{problem_name}] mode={mode}, ct={constraint_threshold}, pop_size={pop_size}: written {out_path}")


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit(
            "This script is intended to be run via Snakemake's `script:` directive, which injects a `snakemake` object."
        ) from err

    snakemake_main()
