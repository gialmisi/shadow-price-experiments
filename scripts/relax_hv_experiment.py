"""Quick experiment: relaxation-gain HV comparison across modes.

For each problem x ct_level, computes a HV metric whose reference box
spans only the relaxation region:

  - objective axis: [f*_relaxed, f*_strict]
      (f*_strict   = best objective on the reference front with all c <= 0
       f*_relaxed  = best objective on the reference front with all c <= threshold)
  - each constraint axis: [0, threshold_c]

Strictly-feasible coordinates (c <= 0) clip to 0 -> full width on that axis.
Below-reference objective values (f < f*_relaxed) clip to f*_relaxed -> full
width on the objective axis. Points outside the box on any axis (c above
threshold, f above f*_strict) are dropped.

Falls back to the current HV definition (see summary_statistics.py) when no
strict-feasible reference set exists or relaxation yields no objective gain.

Reports final HV per run aggregated by mode, with the current HV alongside
for comparison. Intended for standalone use; not a Snakemake job.
"""

import argparse
from pathlib import Path

import moocore
import numpy as np
import polars as pl
import yaml

DEFAULT_PROBLEMS = ["mystery", "cantilevered_beam", "g9", "pressure_vessel"]


def f_strict_relaxed(df_front: pl.DataFrame, f_col: str, c_cols: list[str], thresholds: dict[str, float]):
    strict = df_front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols]))
    relaxed = df_front.filter(pl.all_horizontal([pl.col(c) <= float(thresholds[c]) for c in c_cols]))
    f_strict = float(strict[f_col].min()) if strict.height > 0 else None
    f_relaxed = float(relaxed[f_col].min()) if relaxed.height > 0 else None
    return f_strict, f_relaxed


def evidence_active(c_cols: list[str], evidence: dict) -> np.ndarray:
    mask = np.ones(1 + len(c_cols), dtype=bool)
    for i, c in enumerate(c_cols):
        ev = evidence.get(c, {})
        if ev.get("source", "") == "random_sampling" or ev.get("n", 0) < 10:
            mask[i + 1] = False
    return mask


def build_relax_ctx(f_strict, f_relaxed, c_cols, thresholds, evidence):
    if f_strict is None or f_relaxed is None or f_strict <= f_relaxed:
        return None
    active = evidence_active(c_cols, evidence)
    for i, c in enumerate(c_cols):
        if float(thresholds[c]) <= 0.0:
            active[i + 1] = False
    ideal = np.zeros(1 + len(c_cols))
    ideal[0] = f_relaxed
    rng = np.empty(1 + len(c_cols))
    rng[0] = f_strict - f_relaxed
    for i, c in enumerate(c_cols):
        rng[i + 1] = float(thresholds[c])
    rng = np.where(rng > 0, rng, 1.0)
    n_active = int(active.sum())
    ref_norm = np.ones(n_active)
    return {
        "kind": "relax",
        "active": active,
        "ideal": ideal[active],
        "range": rng[active],
        "ref_norm": ref_norm,
        "hv_ind": moocore.Hypervolume(ref=ref_norm, maximise=False),
        "f_strict": f_strict,
        "f_relaxed": f_relaxed,
    }


def build_current_ctx(df_front, f_col, c_cols, thresholds, evidence, eps_percent):
    """Mirror of summary_statistics.py HV setup."""
    dim_cols = [f_col, *c_cols]
    shadow = df_front.filter(pl.all_horizontal([pl.col(c) <= float(thresholds[c]) for c in c_cols])).unique()
    if shadow.height == 0:
        return None
    ideal = np.array(shadow.select([pl.col(x).min() for x in dim_cols]).row(0), dtype=float)
    nadir = np.array(shadow.select([pl.col(x).max() for x in dim_cols]).row(0), dtype=float)
    sf_range = nadir - ideal
    active = sf_range > 1e-12
    active[0] = True
    for i, c in enumerate(c_cols):
        ev = evidence.get(c, {})
        if ev.get("source", "") == "random_sampling" or ev.get("n", 0) < 10:
            active[i + 1] = False
    rng = np.where(sf_range > 0, sf_range, 1.0)
    active_names = [dim_cols[i] for i in range(len(dim_cols)) if active[i]]
    ref_norm = np.empty(int(active.sum()))
    for k, name in enumerate(active_names):
        full_idx = dim_cols.index(name)
        if name == f_col:
            ref_norm[k] = 1.0 + eps_percent
        else:
            ref_norm[k] = (float(thresholds[name]) - ideal[full_idx]) / rng[full_idx]
    return {
        "kind": "current",
        "active": active,
        "ideal": ideal[active],
        "range": rng[active],
        "ref_norm": ref_norm,
        "hv_ind": moocore.Hypervolume(ref=ref_norm, maximise=False),
    }


def hv_one_run(pts_full: np.ndarray, ctx: dict) -> float:
    pts = pts_full[:, ctx["active"]]
    pts = (pts - ctx["ideal"]) / ctx["range"]
    if ctx["kind"] == "relax":
        pts = np.maximum(pts, 0.0)
    pts = pts[(pts <= ctx["ref_norm"]).all(axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    nd = moocore.is_nondominated(pts, maximise=False)
    return float(ctx["hv_ind"](pts[nd]))


def find_front(fronts_dir: Path, problem: str, psize: int) -> Path | None:
    cands = sorted(fronts_dir.glob(f"{problem}_gen*_psize{psize}.parquet"))
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiment_config.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--problems", nargs="+", default=DEFAULT_PROBLEMS)
    ap.add_argument("--ct-levels", nargs="+", default=["med"])
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--psize", type=int, default=36)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    modes = args.modes or cfg["modes"]
    problems_cfg = {p["name"]: p for p in cfg["problems"]}
    eps_percent = float(cfg["hv_eps_percent"])
    front_psize = int(cfg["population_size_front"])

    results_dir = Path(args.results_dir)
    fronts_dir = results_dir / "fronts"
    thresholds_dir = results_dir / "thresholds"
    data_dir = results_dir / "data"

    for problem in args.problems:
        pc = problems_cfg[problem]
        f_col = f"{pc['objective_symbol']}_min"
        c_cols = list(pc["constraint_symbols"])
        dim_cols = [f_col, *c_cols]

        thr_doc = yaml.safe_load((thresholds_dir / f"{problem}.yaml").read_text())
        evidence = thr_doc.get("evidence", {})

        front_path = find_front(fronts_dir, problem, front_psize)
        if front_path is None:
            print(f"[{problem}] no front file at psize={front_psize}, skipping")
            continue
        df_front = pl.read_parquet(front_path)

        print(f"\n=== {problem}  ({len(c_cols)} constraints) ===")
        for ct in args.ct_levels:
            thresholds = {c: float(thr_doc["levels"][ct].get(c, 0.0)) for c in c_cols}
            f_strict, f_relaxed = f_strict_relaxed(df_front, f_col, c_cols, thresholds)

            relax_ctx = build_relax_ctx(f_strict, f_relaxed, c_cols, thresholds, evidence)
            curr_ctx = build_current_ctx(df_front, f_col, c_cols, thresholds, evidence, eps_percent)
            if curr_ctx is None:
                print(f"  [{ct}] no shadow-feasible reference set; skipping")
                continue

            fallback = relax_ctx is None
            new_ctx = curr_ctx if fallback else relax_ctx
            new_label = "relax" if not fallback else "relax->current (fallback)"

            head = f"  ct={ct}  thresholds={ {c: round(v, 5) for c, v in thresholds.items()} }"
            if fallback:
                reason = (
                    "no strict-feasible front"
                    if f_strict is None
                    else f"f*_strict<=f*_relaxed ({f_strict}<={f_relaxed})"
                )
                head += f"  [fallback: {reason}]"
            else:
                head += f"  f*_strict={f_strict:.5g}  f*_relaxed={f_relaxed:.5g}  gain={f_strict - f_relaxed:.4g}"
            print(head)

            print(
                f"    {'mode':<10s} {'new mean':>12s} {'new std':>11s}  |  {'curr mean':>10s} {'curr std':>11s}  n_runs   metric="
                + new_label
            )
            for mode in modes:
                data_path = (
                    data_dir / f"{problem}_{mode}_gen{args.n_gen}_runs{args.runs}_ct{ct}_psize{args.psize}.parquet"
                )
                if not data_path.exists():
                    print(f"    {mode:<10s} MISSING ({data_path.name})")
                    continue
                df = pl.read_parquet(data_path).select(["run", *dim_cols])
                new_vals, curr_vals = [], []
                for sub in df.partition_by("run"):
                    pts = sub.select(dim_cols).to_numpy()
                    new_vals.append(hv_one_run(pts, new_ctx))
                    curr_vals.append(hv_one_run(pts, curr_ctx))
                new_arr = np.asarray(new_vals)
                curr_arr = np.asarray(curr_vals)
                print(
                    f"    {mode:<10s} {new_arr.mean():>12.5f} {new_arr.std():>11.5f}  |  "
                    f"{curr_arr.mean():>10.5f} {curr_arr.std():>11.5f}  {len(new_arr):>5d}"
                )


if __name__ == "__main__":
    main()
