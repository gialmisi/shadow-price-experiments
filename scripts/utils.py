"""General stuff useful generally."""

from collections.abc import Callable

from desdeo.problem import Problem
from desdeo.problem.external.pymoo_provider import PymooProblemParams, create_pymoo_problem
from desdeo.problem.testproblems.single_objective import (
    mishras_bird_constrained,
    mystery_function,
    new_branin_function,
    townsend_modified,
)

PROBLEM_BUILDERS: dict[str, Callable[[], Problem]] = {
    "branin": new_branin_function,
    "mystery": mystery_function,
    "mishrasbird": mishras_bird_constrained,
    "townsend": townsend_modified,
    "cantilevered_beam": lambda: create_pymoo_problem(PymooProblemParams(name="cantilevered_beam")),
    "pressure_vessel": lambda: create_pymoo_problem(PymooProblemParams(name="pressure_vessel")),
    "g1": lambda: create_pymoo_problem(PymooProblemParams(name="g1")),
    "g2": lambda: create_pymoo_problem(PymooProblemParams(name="g2")),
    "g6": lambda: create_pymoo_problem(PymooProblemParams(name="g6")),
    "g8": lambda: create_pymoo_problem(PymooProblemParams(name="g8")),
    "g9": lambda: create_pymoo_problem(PymooProblemParams(name="g9")),
    "g10": lambda: create_pymoo_problem(PymooProblemParams(name="g10")),
    "g24": lambda: create_pymoo_problem(PymooProblemParams(name="g24")),
}
