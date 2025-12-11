"""General stuff useful generally."""

from collections.abc import Callable

from desdeo.problem import Problem
from desdeo.problem.testproblems.single_objective import mishras_bird_constrained, mystery_function, new_branin_function

PROBLEM_BUILDERS: dict[str, Callable[[], Problem]] = {
    "branin": new_branin_function,
    "mystery": mystery_function,
    "mishrasbird": mishras_bird_constrained,
}
