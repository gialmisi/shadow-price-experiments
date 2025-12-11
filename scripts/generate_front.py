"""Script to run with Snakemake to generate Pareto fronts for problems."""

from snakemake.script import snakemake
from utils import PROBLEM_BUILDERS

from desdeo.emo import algorithms, crossover, generator, mutation, scalar_selection, termination
from desdeo.emo.hooks.archivers import NonDominatedArchive
from desdeo.problem import Objective, Problem


def setup_problem(problem: Problem, constraint_symbol: str) -> Problem:
    """Takes a single-objective optimization problem and setups one...

    Takes a single-objective optimization problem and setups one of its constraints as its second objective function.
    """
    return problem.model_copy(
        update={
            "constraints": None,
            "objectives": [
                *problem.objectives,
                Objective(
                    name="Constraint",
                    symbol=constraint_symbol,
                    func=problem.get_constraint(constraint_symbol).func,
                ),
            ],
        }
    )


def generate_front(
    problem: Problem,
    xover_probability: float,
    xover_distribution: float,
    distribution_index: float,
    tournament_size: int,
    population_size: int,
    n_generations: int,
) -> NonDominatedArchive:
    """Run NSGA2 to generate a front for a given problem."""
    # setup
    nsga2_options = algorithms.nsga2_options()

    nsga2_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=xover_probability, xover_distribution=xover_distribution
    )
    nsga2_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=1.0 / len(problem.variables), distribution_index=distribution_index
    )
    nsga2_options.template.mate_selection = scalar_selection.TournamentSelectionOptions(
        name="TournamentSelection", tournament_size=tournament_size, winner_size=population_size
    )

    nsga2_options.template.generator = generator.LHSGeneratorOptions(n_points=population_size)
    nsga2_options.template.termination = termination.MaxGenerationsTerminatorOptions(max_generations=n_generations)

    solver, extras = algorithms.emo_constructor(emo_options=nsga2_options, problem=problem)

    _ = solver()

    return extras.archive


def snakemake_main():
    problem_name = snakemake.wildcards.problem_name
    constraint_symbol = snakemake.params.constraint_symbol

    out_path = str(snakemake.output[0])

    try:
        problem_fun = PROBLEM_BUILDERS[problem_name]
    except KeyError as err:
        raise ValueError(f"Unknown problem '{problem_name}'. Add it to PROBLEM_BUILDERS in run_experiment.py.") from err

    problem = problem_fun()

    bi_problem = setup_problem(problem, constraint_symbol)

    pop_size = snakemake.config["population_size_front"]
    n_generations = snakemake.config["n_generations_front"]
    xover_distribution = snakemake.config["xover_distribution"]
    xover_probability = snakemake.config["xover_probability"]
    distribution_index = snakemake.config["distribution_index"]
    tournament_size = snakemake.config["tournament_size"]

    archive = generate_front(
        bi_problem,
        xover_probability=xover_probability,
        xover_distribution=xover_distribution,
        distribution_index=distribution_index,
        tournament_size=tournament_size,
        population_size=pop_size,
        n_generations=n_generations,
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
