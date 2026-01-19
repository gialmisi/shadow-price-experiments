"""Script to run with Snakemake to generate Pareto fronts for problems."""

from snakemake.script import snakemake
from utils import PROBLEM_BUILDERS

from desdeo.emo import algorithms, crossover, generator, mutation, scalar_selection, selection, termination
from desdeo.emo.hooks.archivers import NonDominatedArchive
from desdeo.emo.operators.selection import ReferenceVectorOptions
from desdeo.problem import Objective, Problem


def objective_from_constraint(problem: Problem, constraint_symbol: str) -> Objective:
    """Create an Objective that evaluates exactly like the given constraint.

    Works for both analytic constraints (func != None) and simulator-backed constraints
    (func is None but simulator_path is set).
    """
    cons = problem.get_constraint(constraint_symbol)

    if cons.func is not None:
        return Objective(
            name=f"Constraint_{constraint_symbol}",
            symbol=constraint_symbol,
            func=cons.func,
        )

    # Simulator-backed constraint (e.g., pymoo external problems)
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
    """Takes a single-objective optimization problem and setups one...

    Takes a single-objective optimization problem and setups one of its constraints as its second objective function.
    """
    extra_objectives = [objective_from_constraint(problem, sym) for sym in constraint_symbols]
    return problem.model_copy(
        update={
            "constraints": None,
            "objectives": [*problem.objectives, *extra_objectives],
        }
    )


def _generate_front(
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


def generate_front(
    problem: Problem,
    xover_probability: float,
    xover_distribution: float,
    distribution_index: float,
    tournament_size: int,
    population_size: int,
    n_generations: int,
) -> NonDominatedArchive:
    """Run NSGA3 to generate a front for a given problem."""
    # setup
    nsga3_options = algorithms.nsga3_options()

    nsga3_options.template.crossover = crossover.SimulatedBinaryCrossoverOptions(
        xover_probability=xover_probability, xover_distribution=xover_distribution
    )
    nsga3_options.template.mutation = mutation.BoundedPolynomialMutationOptions(
        mutation_probability=1.0 / len(problem.variables), distribution_index=distribution_index
    )
    nsga3_options.template.selection = selection.NSGA3SelectorOptions(
        reference_vector_options=ReferenceVectorOptions(
            number_of_vectors=population_size,
        )
    )

    nsga3_options.template.generator = generator.LHSGeneratorOptions(n_points=population_size)
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

    multi_problem = setup_problem(problem, constraint_symbols)

    pop_size = snakemake.config["population_size_front"]
    n_generations = snakemake.config["n_generations_front"]
    xover_distribution = snakemake.config["xover_distribution"]
    xover_probability = snakemake.config["xover_probability"]
    distribution_index = snakemake.config["distribution_index"]
    tournament_size = snakemake.config["tournament_size"]

    archive = generate_front(
        multi_problem,
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
