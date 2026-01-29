"""Script to run with Snakemake to generate Pareto fronts for problems."""

import numpy as np
import polars as pl
from snakemake.script import snakemake
from utils import PROBLEM_BUILDERS

from desdeo.emo import algorithms, crossover, mutation, selection, termination
from desdeo.emo.hooks.archivers import NonDominatedArchive
from desdeo.emo.operators.selection import ReferenceVectorOptions
from desdeo.emo.options.generator import SeededHybridGeneratorOptions
from desdeo.problem import Objective, Problem


def objective_from_constraint(problem: Problem, constraint_symbol: str) -> Objective:
    """Create an Objective that evaluates exactly like the given constraint."""
    cons = problem.get_constraint(constraint_symbol)

    if cons.func is not None:
        return Objective(
            name=f"Constraint_{constraint_symbol}",
            symbol=constraint_symbol,
            func=cons.func,
        )

    if getattr(cons, "simulator_path", None) is not None:
        return Objective(
            name=f"Constraint_{constraint_symbol}",
            symbol=constraint_symbol,
            func=None,
            simulator_path=cons.simulator_path,
            objective_type="simulator",
        )

    raise ValueError(
        f"Constraint '{constraint_symbol}' has neither 'func' nor 'simulator_path'. Cannot promote to objective."
    )


def setup_problem(problem: Problem, constraint_symbols: list[str]) -> Problem:
    """Promote selected constraints to objectives and drop constraints."""
    extra_objectives = [objective_from_constraint(problem, sym) for sym in constraint_symbols]
    return problem.model_copy(
        update={
            "constraints": None,
            "objectives": [*problem.objectives, *extra_objectives],
        }
    )


def _seed_solution_from_config(problem: Problem, problem_name: str) -> pl.DataFrame:
    """Get the seed solution (1 row) for this problem from the experiment config."""
    probs = snakemake.config.get("problems", [])
    for p in probs:
        if p.get("name") == problem_name:
            seed_solution_ = p.get("optimal_solution", None)
            if seed_solution_ is None:
                raise ValueError(f"Missing 'optimal_solution' for problem '{problem_name}' in config.")
            return pl.DataFrame(
                np.atleast_2d(seed_solution_),
                schema=[v.symbol for v in problem.variables],
            )
    raise ValueError(f"Problem '{problem_name}' not found in config['problems'].")


def generate_front(
    problem: Problem,
    seed_solution: pl.DataFrame,
    xover_probability: float,
    xover_distribution: float,
    distribution_index: float,
    population_size: int,
    n_generations: int,
    perturb_fraction: float = 0.2,
    sigma: float = 0.02,
    flip_prob: float = 0.1,
) -> NonDominatedArchive:
    """Run NSGA-III to generate a (seeded) reference front for a given problem."""
    nsga3_options = algorithms.nsga3_options()

    nsga3_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=xover_probability,
        xover_distribution=xover_distribution,
    )
    nsga3_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=1.0 / len(problem.variables),
        distribution_index=distribution_index,
    )
    nsga3_options.template.selection = selection.NSGA3SelectorOptions(
        reference_vector_options=ReferenceVectorOptions(number_of_vectors=population_size)
    )

    # Seeded hybrid initial population
    nsga3_options.template.generator = SeededHybridGeneratorOptions(
        n_points=population_size,
        seed_solution=seed_solution,
        perturb_fraction=perturb_fraction,
        sigma=sigma,
        flip_prob=flip_prob,
    )

    nsga3_options.template.termination = termination.MaxGenerationsTerminatorOptions(max_generations=n_generations)

    solver, extras = algorithms.emo_constructor(emo_options=nsga3_options, problem=problem)
    _ = solver()
    return extras.archive


def snakemake_main():
    problem_name = snakemake.wildcards.problem_name
    constraint_symbols = list(snakemake.params.constraint_symbols)
    out_path = str(snakemake.output[0])

    try:
        problem_fun = PROBLEM_BUILDERS[problem_name]
    except KeyError as err:
        raise ValueError(f"Unknown problem '{problem_name}'. Add it to PROBLEM_BUILDERS in run_experiment.py.") from err

    problem = problem_fun()
    seed_solution = _seed_solution_from_config(problem, problem_name)

    multi_problem = setup_problem(problem, constraint_symbols)

    pop_size = snakemake.config["population_size_front"]
    n_generations = snakemake.config["n_generations_front"]
    xover_distribution = snakemake.config["xover_distribution"]
    xover_probability = snakemake.config["xover_probability"]
    distribution_index = snakemake.config["distribution_index"]
    perturb_fraction = snakemake.config["seed_perturb_fraction"]
    sigma = snakemake.config["seed_sigma"]
    flip_prob = snakemake.config["seed_flip_prob"]

    archive = generate_front(
        multi_problem,
        seed_solution=seed_solution,
        xover_probability=xover_probability,
        xover_distribution=xover_distribution,
        distribution_index=distribution_index,
        population_size=pop_size,
        n_generations=n_generations,
        perturb_fraction=perturb_fraction,
        sigma=sigma,
        flip_prob=flip_prob,
    )

    archive.solutions.write_parquet(out_path)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit(
            "This script is intended to be run via Snakemake's `script:` directive, which injects a `snakemake` object."
        ) from err

    snakemake_main()
