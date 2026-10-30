"""General stuff useful generally."""

from collections.abc import Callable
from typing import Any

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


OPERATOR_FIELDS = ("crossover", "mutation", "repair", "mate_selection")


def _flatten(options: dict, prefix: str = "") -> dict[str, Any]:
    """Nested options as {"a.b": value}, so that nested option models are compared option by option."""
    flat = {}
    for key, value in options.items():
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _merge(base: dict, update: dict) -> dict:
    """`base` with the (possibly nested) values of `update`."""
    merged = dict(base)
    for key, value in update.items():
        merged[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return merged


def set_operators(template: Any, operators: dict[str, dict], fixed: dict[str, dict] | None = None) -> None:
    """Set the template's operators from the config, and stop if they differ from the preset.

    `template` is the template of a DESDEO preset (e.g. `nsga2_options().template`), `operators` the
    config's operator block (it may also hold the selector, e.g. NSGA-III's reference vectors). The
    config must state every option of every operator the preset sets, and each value must equal the
    preset's, so that the config documents exactly what runs and a changed preset (e.g. after a
    DESDEO upgrade) cannot change the experiments silently. `fixed` holds options set by the
    experiment rather than the config (the tournament's winner_size, the number of reference
    vectors), nested like the options; they are excluded from the comparison.
    """
    fixed = fixed or {}
    problems = [
        f"{field}: set by the preset but not stated in the config"
        for field in OPERATOR_FIELDS
        if getattr(template, field, None) is not None and field not in operators
    ]
    for field, stated in operators.items():
        preset = getattr(template, field, None)
        if preset is None:
            problems.append(f"{field}: stated in the config but not set by the preset")
            continue
        preset_values = _flatten(preset.model_dump())
        own = _flatten(fixed.get(field, {}))
        missing = set(preset_values) - set(_flatten(stated)) - set(own)
        if missing:
            problems.append(f"{field}: options not stated in the config: {sorted(missing)}")
        built = type(preset).model_validate(_merge(stated, fixed.get(field, {})))
        for key, value in _flatten(built.model_dump()).items():
            if key not in own and value != preset_values.get(key):
                problems.append(f"{field}.{key}: config {value!r}, preset {preset_values.get(key)!r}")
        setattr(template, field, built)
    if problems:
        raise ValueError("The config's operators differ from the DESDEO preset:\n  " + "\n  ".join(problems))
