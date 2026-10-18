"""Best-known reference optima for every relaxation of one problem.

For each threshold level and each relaxation (all constraints jointly at their thresholds,
and each constraint alone with the others strict), the reference relaxed optimum is the
best of three sources:

1. the best threshold-feasible point of the NSGA-III reference front,
2. the best threshold-feasible point of the stored experiment runs (all modes), and
3. multistart SLSQP (scipy) on the shifted problem ``min f s.t. c_j <= t_j``, started
   from the best stored run points and from random points in the box.

Every candidate is evaluated through DESDEO's own evaluator, so all sources are in the same
units as the experiments. The problems are nonconvex, so the result is a best-known value,
not a certified global optimum; the sources and the solver settings are recorded in the
output so that the computation is reproducible. The strict optimum is also re-solved as a
sanity check against the known optimum in the config, which stays the reference f*(0).

Output YAML (per problem)::

    objective_optimum: <known optimum from config>
    strict: {f: ..., source: ...}            # sanity check only
    levels:
      low:
        joint:  {f, x, source, f_front, f_runs, f_solver, n_solver_feasible}
        single: {c_1: {...}, c_2: {...}}
      ...
"""

from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.optimize import minimize

try:
    from snakemake.script import snakemake  # snakemake <= 8.x
except ImportError:
    pass  # snakemake >= 9.x injects `snakemake` via the script preamble

from desdeo.problem import SimulatorEvaluator
from utils import PROBLEM_BUILDERS

LEVELS = ["low", "med", "high"]


class Shifted:
    """Objective and constraint callables of a DESDEO problem for scipy."""

    def __init__(self, problem, f_col: str, c_cols: list[str]):
        self.ev = SimulatorEvaluator(problem)
        self.xs = [v.symbol for v in problem.variables]
        self.lb = np.array([float(v.lowerbound) for v in problem.variables])
        self.ub = np.array([float(v.upperbound) for v in problem.variables])
        self.f_col = f_col
        self.c_cols = c_cols
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        self.n_evals += 1
        out = self.ev.evaluate({s: [float(v)] for s, v in zip(self.xs, x, strict=True)})
        row = out.row(0, named=True)
        return float(row[self.f_col]), np.array([float(row[c]) for c in self.c_cols])


def _feasible(c: np.ndarray, tv: np.ndarray, tol: float) -> bool:
    return bool(np.all(c <= tv + tol * np.maximum(1.0, np.abs(tv))))


def solve_relaxation(
    sh: Shifted,
    tv: np.ndarray,
    starts: list[np.ndarray],
    tol: float,
    maxiter: int,
) -> tuple[float | None, np.ndarray | None, int]:
    """Multistart SLSQP; returns best feasible (f, x) and the number of feasible finishes."""
    k = len(tv)
    cons = [{"type": "ineq", "fun": (lambda x, i=i: tv[i] - sh(x)[1][i])} for i in range(k)]
    best_f, best_x, n_ok = None, None, 0
    for x0 in starts:
        x0 = np.clip(x0, sh.lb, sh.ub)
        try:
            res = minimize(
                lambda x: sh(x)[0],
                x0,
                method="SLSQP",
                bounds=list(zip(sh.lb, sh.ub, strict=True)),
                constraints=cons,
                options={"maxiter": maxiter, "ftol": 1e-12},
            )
        except Exception:  # noqa: BLE001
            continue
        x = np.clip(res.x, sh.lb, sh.ub)
        f, c = sh(x)
        if _feasible(c, tv, tol):
            n_ok += 1
            if best_f is None or f < best_f:
                best_f, best_x = f, x
    return best_f, best_x, n_ok


def snakemake_main() -> None:  # noqa: D103, PLR0915
    prob_cfg = dict(snakemake.params.problem)
    name = prob_cfg["name"]
    c_cols = list(prob_cfg["constraint_symbols"])
    f_col = f"{prob_cfg['objective_symbol']}_min"
    f_known = float(prob_cfg["objective_optimum"])
    n_run_starts = int(snakemake.params.n_run_starts)
    n_random_starts = int(snakemake.params.n_random_starts)
    maxiter = int(snakemake.params.maxiter)
    seed = int(snakemake.params.seed)
    tol = float(snakemake.params.feasibility_tol)

    with open(str(snakemake.input["thresholds"]), encoding="utf-8") as fh:  # noqa: PTH123
        thr = yaml.safe_load(fh)
    levels = {lvl: {c: float(thr["levels"][lvl].get(c, 0.0)) for c in c_cols} for lvl in LEVELS}

    problem = PROBLEM_BUILDERS[name]()
    sh = Shifted(problem, f_col, c_cols)
    x_cols = sh.xs
    rng = np.random.default_rng(seed)

    def best_of(df: pl.DataFrame, tv: np.ndarray, n: int) -> pl.DataFrame:
        mask = pl.all_horizontal([pl.col(c) <= float(tv[i]) for i, c in enumerate(c_cols)])
        return df.filter(mask).sort(f_col).unique(subset=x_cols, maintain_order=True).head(n)

    # All relaxations to solve: strict, and per level the joint and each single relaxation
    relaxations: dict[str, np.ndarray] = {"strict": np.zeros(len(c_cols))}
    for lvl in LEVELS:
        tv_joint = np.array([levels[lvl][c] for c in c_cols])
        relaxations[f"{lvl}/joint"] = tv_joint
        for j, cj in enumerate(c_cols):
            if tv_joint[j] > 0.0:
                tv = np.zeros(len(c_cols))
                tv[j] = tv_joint[j]
                relaxations[f"{lvl}/{cj}"] = tv

    # Candidate points, pooled file by file with streaming scans so that only the best few
    # points of every relaxation are ever held in memory
    def scan_best(path: str, tv: np.ndarray, n: int) -> pl.DataFrame:
        mask = pl.all_horizontal([pl.col(c) <= float(tv[i]) for i, c in enumerate(c_cols)])
        return (
            pl.scan_parquet(path)
            .select([*x_cols, f_col, *c_cols])
            .filter(mask)
            .sort(f_col)
            .head(n)
            .collect(engine="streaming")
        )

    front_path = str(snakemake.input["front"])
    cand_front = {key: scan_best(front_path, tv, 1) for key, tv in relaxations.items()}
    pooled: dict[str, list[pl.DataFrame]] = {key: [] for key in relaxations}
    data_paths = list(map(str, snakemake.input["data"]))
    for path in data_paths:
        for key, tv in relaxations.items():
            pooled[key].append(scan_best(path, tv, n_run_starts))
    cand_runs = {
        key: pl.concat(parts).sort(f_col).unique(subset=x_cols, maintain_order=True).head(n_run_starts)
        for key, parts in pooled.items()
    }
    n_rows = len(data_paths)
    print(f"[{name}] pooled candidates from {n_rows} run files", flush=True)

    def reference(key: str) -> dict:
        tv = relaxations[key]
        c_front = cand_front[key]
        c_runs = cand_runs[key]
        entries: list[tuple[float, np.ndarray, str]] = []
        # Re-evaluate the stored candidates through the evaluator (same units, guards the data)
        for df, src in ((c_front, "front"), (c_runs.head(1), "runs")):
            if df.height:
                x = np.array(df.select(x_cols).row(0), dtype=float)
                f, c = sh(x)
                if _feasible(c, tv, tol):
                    entries.append((f, x, src))
        starts = [np.array(r, dtype=float) for r in c_runs.select(x_cols).rows()]
        starts += list(sh.lb + rng.random((n_random_starts, len(sh.lb))) * (sh.ub - sh.lb))
        f_solver, x_solver, n_ok = solve_relaxation(sh, tv, starts, tol, maxiter)
        if f_solver is not None:
            entries.append((f_solver, x_solver, "solver"))
        if not entries:
            return {"f": None, "x": None, "source": None, "f_front": None, "f_runs": None, "f_solver": None}
        f_best, x_best, src = min(entries, key=lambda e: e[0])
        by_src = {s: f for f, _, s in entries}
        return {
            "f": float(f_best),
            "x": [float(v) for v in x_best],
            "source": src,
            "f_front": by_src.get("front"),
            "f_runs": by_src.get("runs"),
            "f_solver": by_src.get("solver"),
            "n_solver_feasible": int(n_ok),
            "n_starts": len(starts),
        }

    out: dict = {
        "problem": name,
        "objective_symbol": prob_cfg["objective_symbol"],
        "f_col": f_col,
        "objective_optimum": f_known,
        "method": {
            "solver": "scipy SLSQP, multistart",
            "n_run_starts": n_run_starts,
            "n_random_starts": n_random_starts,
            "maxiter": maxiter,
            "seed": seed,
            "feasibility_tol": tol,
            "run_files": [Path(p).name for p in map(str, snakemake.input["data"])],
        },
        "levels": {},
    }

    strict = reference("strict")
    out["strict"] = strict
    if strict["f"] is not None and strict["f"] < f_known - tol * max(1.0, abs(f_known)):
        print(
            f"[{name}] WARNING: strictly feasible value {strict['f']:.8g} beats the known optimum {f_known:.8g}",
            flush=True,
        )
    print(f"[{name}] strict: {strict['f']} ({strict['source']}), known {f_known}", flush=True)

    for lvl in LEVELS:
        joint = reference(f"{lvl}/joint")
        single = {}
        for cj in c_cols:
            key = f"{lvl}/{cj}"
            single[cj] = reference(key) if key in relaxations else {"f": None, "x": None, "source": "not_relaxed"}
        out["levels"][lvl] = {"thresholds": levels[lvl], "joint": joint, "single": single}
        print(
            f"[{name}] {lvl}: joint {joint['f']} ({joint['source']}); "
            + ", ".join(f"{c}: {single[c]['f']} ({single[c]['source']})" for c in c_cols),
            flush=True,
        )

    out["n_evaluations"] = int(sh.n_evals)
    with open(str(snakemake.output[0]), "w", encoding="utf-8") as fh:  # noqa: PTH123
        yaml.safe_dump(out, fh, sort_keys=False)
    print(f"[{name}] wrote {snakemake.output[0]} after {sh.n_evals} evaluations", flush=True)


if __name__ == "__main__":
    try:
        snakemake  # noqa: B018
    except Exception as err:
        raise SystemExit("This script is intended to be run via Snakemake's `script:` directive.") from err
    snakemake_main()
