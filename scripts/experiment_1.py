"""Experiment 1: Baseline vs relaxed."""

from datetime import datetime

import polars as pl

from desdeo.emo import algorithms, crossover, generator, mutation, scalar_selection, selection, termination
from desdeo.emo.hooks.archivers import Archive
from desdeo.problem import Problem
from desdeo.problem.testproblems.single_objective import mystery_function, new_branin_function


def run_nsga2_with_mode(
    problem: Problem,
    mode: str,
    constraint_threshold: float,
    pop_size: int,
    n_generations: int,
    constraint_symbol="c_1",
):
    """Run the NSGA-II style EA once for a given constraint-handling mode."""
    nsga2_options = algorithms.nsga2_options()

    nsga2_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=0.9, xover_distribution=20
    )
    nsga2_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=1.0 / len(problem.variables), distribution_index=20
    )
    nsga2_options.template.mate_selection = scalar_selection.TournamentSelectionOptions(
        name="TournamentSelection", tournament_size=2, winner_size=pop_size
    )
    nsga2_options.template.selection = selection.SingleObjectiveConstrainedRankingSelectorOptions(
        target_objective_symbol="f_1",
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

    # ---- Run optimization ----
    _ = solver()  # result object not strictly needed; archive holds all solutions

    # Full history of solutions as Polars DataFrame, last population
    return archive


def run(
    problem: Problem,
    mode: str,
    constraint_threshold: float,
    pop_size: int,
    n_generations: int,
    constraint_symbol: str,
    objective_symbol: str,
):
    results = run_nsga2_with_mode(
        problem,
        mode,
        constraint_threshold,
        pop_size=pop_size,
        n_generations=n_generations,
        constraint_symbol=constraint_symbol,
    )

    solutions = results.solutions

    feasible_baseline = solutions.with_columns(
        pl.when(pl.col(constraint_symbol) <= 0)
        .then(pl.col(f"{objective_symbol}_min"))
        .otherwise(float("inf"))
        .alias(f"feasible_{objective_symbol}")
    )

    best_baseline = (
        feasible_baseline.group_by("generation")
        .agg(pl.col(f"feasible_{objective_symbol}").min().alias(f"best_{objective_symbol}_this_gen"))
        .sort("generation")
    )
    best_so_far_baseline = best_baseline.with_columns(
        best_baseline[f"best_{objective_symbol}_this_gen"].cum_min().alias("best")
    )

    return best_so_far_baseline, solutions


def main():
    problem_fun = mystery_function
    problem = problem_fun()
    mode = "relaxed"
    constraint_thresholds = [0.2, 0.4, 0.6]
    pop_sizes = [6, 12, 18]
    n_generations = 100
    n_runs = 500
    constraint_symbol = "c_1"
    objective_symbol = "f_1"

    # do runs

    for pop_size in pop_sizes:
        for constraint_threshold in constraint_thresholds:
            print(
                f"Computing mode {mode} with constraint threshold {constraint_threshold} and population size {pop_size}."
            )

            best_results = None
            solutions_results = None

            for step_i in range(n_runs):
                if step_i % 10 == 0:
                    print(f"{step_i}/{n_runs}")

                best, solutions = run(
                    problem=problem,
                    mode=mode,
                    constraint_threshold=constraint_threshold,
                    pop_size=pop_size,
                    n_generations=n_generations,
                    constraint_symbol=constraint_symbol,
                    objective_symbol=objective_symbol,
                )

                best_results = best if best_results is None else pl.concat([best_results, best])

                solutions_results = (
                    solutions if solutions_results is None else pl.concat([solutions_results, solutions])
                )

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

            now_str = datetime.now().isoformat()

            parameters = f"""
                problem: {problem_fun.__name__}
                mode: {mode}
                constraint_threshold: {constraint_threshold}
                pop_size: {pop_size}
                n_generations: {n_generations}
                n_runs: {n_runs}
                constraint_symbol: {constraint_symbol}
                objective_symbol: {objective_symbol}
                """

            solutions_results.write_parquet(
                f"{problem_fun.__name__}_{mode}_{now_str}_solutions_{constraint_threshold}_{pop_size}_{n_generations}_{n_runs}_{constraint_symbol}_{objective_symbol}.parquet",
                metadata={
                    "notes": parameters,
                    "created_at": now_str,
                },
            )
            combined.write_parquet(
                f"{problem_fun.__name__}_{mode}_{now_str}_combined_{constraint_threshold}_{pop_size}_{n_generations}_{n_runs}_{constraint_symbol}_{objective_symbol}.parquet",
                metadata={
                    "notes": parameters,
                    "created_at": now_str,
                },
            )
            stats.write_parquet(
                f"{problem_fun.__name__}_{mode}_{now_str}_stats_{constraint_threshold}_{pop_size}_{n_generations}_{n_runs}_{constraint_symbol}_{objective_symbol}.parquet",
                metadata={
                    "notes": parameters,
                    "created_at": now_str,
                },
            )

            print("ok")


if __name__ == "__main__":
    main()
