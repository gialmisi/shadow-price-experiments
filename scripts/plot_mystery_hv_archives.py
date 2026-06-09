"""Visualize the cumulative non-dominated archives used for HV on the mystery problem.

mystery is the only 2D (1 objective + 1 constraint) test problem where
baseline nominally dominates relaxed/ranking on hypervolume. This script
plots the actual archives that drive the HV computation, for one
representative run, across the three modes — illustrating why baseline's
spread along the feasibility boundary translates into higher HV.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import moocore
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from desdeo.tools.non_dominated_sorting import non_dominated_merge

PROBLEM = "mystery"
CT_LEVEL = "high"  # picked because Bs - Rx is largest here (~0.0023)
PSIZE = 12  # smaller pop accentuates the differences
N_RUNS = 500
N_GEN = 200
EPS_PERCENT = 0.01

OBJECTIVE_SYMBOL = "f_1"
CONSTRAINT_SYMBOLS = ["c_1"]
F_COL = f"{OBJECTIVE_SYMBOL}_min"
DIM_COLS = [F_COL] + CONSTRAINT_SYMBOLS

OUT_PATH = Path(f"results/figures/mystery_hv_archives_{CT_LEVEL}_psize{PSIZE}.pdf")


def build_hv_context(front_path, ct_level, thr_doc):
    df_front = pl.read_parquet(front_path)

    def filter_relax_only(relaxed):
        terms = []
        for c in CONSTRAINT_SYMBOLS:
            if relaxed is None or c != relaxed:
                terms.append(pl.col(c) <= 0.0)
            else:
                terms.append(pl.col(c) <= float(thr_doc["levels"][ct_level].get(c, 0.0)))
        return pl.all_horizontal(terms)

    parts = [df_front.filter(filter_relax_only(None))]
    for ci in CONSTRAINT_SYMBOLS:
        parts.append(df_front.filter(filter_relax_only(ci)))
    sf = pl.concat(parts).unique()

    ideal = np.array(sf.select([pl.col(x).min().alias(x) for x in DIM_COLS]).row(0), dtype=float)
    nadir = np.array(sf.select([pl.col(x).max().alias(x) for x in DIM_COLS]).row(0), dtype=float)
    sf_ranges = nadir - ideal

    evidence = thr_doc.get("evidence", {})
    active_mask = sf_ranges > 1e-12
    active_mask[0] = True
    for i, c in enumerate(CONSTRAINT_SYMBOLS):
        ev = evidence.get(c, {})
        if ev.get("source", "") == "random_sampling" or ev.get("n", 0) < 10:
            active_mask[i + 1] = False

    range_active = sf_ranges[active_mask]
    range_active = np.where(range_active > 0, range_active, 1.0)
    ref_norm = np.ones(int(active_mask.sum())) * (1.0 + EPS_PERCENT)

    return {
        "active_mask": active_mask,
        "ideal_active": ideal[active_mask],
        "range_active": range_active,
        "ref_norm": ref_norm,
        "hv_ind": moocore.Hypervolume(ref=ref_norm, maximise=False),
        "ideal": ideal,
        "nadir": nadir,
    }


def cumulative_archive(df_run: pl.DataFrame, hv_ctx) -> tuple[np.ndarray, float]:
    """Replay the HV-archive build and return the final archive (normalized) and its HV."""
    active_mask = hv_ctx["active_mask"]
    ideal_active = hv_ctx["ideal_active"]
    range_active = hv_ctx["range_active"]
    ref_norm = hv_ctx["ref_norm"]
    hv_ind = hv_ctx["hv_ind"]

    archive = np.empty((0, int(active_mask.sum())), dtype=float)
    prev_hv = 0.0
    gens = sorted(df_run["generation"].unique().to_list())

    for g in gens:
        sub = df_run.filter(pl.col("generation") == g)
        pts = sub.select(DIM_COLS).to_numpy()[:, active_mask]
        pts = (pts - ideal_active) / range_active
        pts = pts[(pts <= ref_norm).all(axis=1)]
        if pts.shape[0] == 0:
            continue
        nd_mask = moocore.is_nondominated(pts, maximise=False)
        new_nd = pts[nd_mask]
        if archive.shape[0] == 0:
            archive = new_nd
        else:
            mask_old, mask_new = non_dominated_merge(archive, new_nd)
            if not mask_new.any():
                continue
            archive = np.vstack([archive[mask_old], new_nd[mask_new]])
        prev_hv = float(hv_ind(archive))

    return archive, prev_hv


def main():
    thr_doc = yaml.safe_load(open(f"results/thresholds/{PROBLEM}.yaml"))
    front_path = Path("results/fronts") / f"{PROBLEM}_gen10000_psize200.parquet"
    hv_ctx = build_hv_context(front_path, CT_LEVEL, thr_doc)

    threshold = float(thr_doc["levels"][CT_LEVEL].get("c_1", 0.0))
    f_opt = thr_doc["objective_optimum"]
    shadow_opt = thr_doc.get("objective_shadow_optima", {}).get(CT_LEVEL)

    # Convert ref-box back to original coordinates for axis labels
    ideal = hv_ctx["ideal"]
    nadir = hv_ctx["nadir"]
    range_full = nadir - ideal

    modes = ["baseline", "relaxed", "ranking"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    palette = {"baseline": "#000000", "relaxed": "#0072B2", "ranking": "#E69F00"}

    # Use a single representative run (e.g., median final HV) for each mode
    for ax, mode in zip(axes, modes):
        path = f"results/data/{PROBLEM}_{mode}_gen{N_GEN}_runs{N_RUNS}_ct{CT_LEVEL}_psize{PSIZE}.parquet"
        df = pl.read_parquet(path)

        # Per-run HVs (using cumulative archive). Pick the median run for plotting.
        run_ids = sorted(df["run"].unique().to_list())
        hvs = []
        for r in run_ids[:50]:  # first 50 for speed
            sub = df.filter(pl.col("run") == r)
            _, hv = cumulative_archive(sub, hv_ctx)
            hvs.append((r, hv))
        hvs.sort(key=lambda x: x[1])
        median_run = hvs[len(hvs) // 2][0]
        median_hv = hvs[len(hvs) // 2][1]

        # Recompute the median run's archive to draw it
        sub = df.filter(pl.col("run") == median_run)
        archive_norm, hv_val = cumulative_archive(sub, hv_ctx)

        # Convert archive back to original coordinates for plotting
        active = hv_ctx["active_mask"]
        archive_orig = archive_norm * range_full[active] + ideal[active]

        # Final-generation full population in original coordinates
        last_gen = sub.filter(pl.col("generation") == sub["generation"].max())
        x_pop = last_gen[F_COL].to_numpy()
        y_pop = last_gen["c_1"].to_numpy()

        # Plot full final population
        ax.scatter(x_pop, y_pop, s=15, alpha=0.4, c=palette[mode], label="final population")
        # Plot non-dominated archive (the points that contribute to HV)
        ax.scatter(
            archive_orig[:, 0],
            archive_orig[:, 1],
            s=40,
            edgecolor=palette[mode],
            facecolor="none",
            linewidth=1.5,
            label=f"HV archive ({len(archive_orig)} pts)",
        )

        # Reference lines
        ax.axhline(0, color="red", linestyle="-", linewidth=1, alpha=0.7, label="strict feasibility (c=0)")
        ax.axhline(
            threshold, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"threshold (c={threshold:.3f})"
        )
        ax.axvline(f_opt, color="green", linestyle=":", linewidth=1, alpha=0.7, label=f"f* = {f_opt:.4f}")
        if shadow_opt is not None:
            ax.axvline(
                shadow_opt, color="purple", linestyle=":", linewidth=1, alpha=0.7, label=f"shadow f* = {shadow_opt:.4f}"
            )

        ax.set_title(f"{mode.capitalize()} (HV = {hv_val:.4f}, run {median_run})")
        ax.set_xlabel("$f_1$")
        if ax is axes[0]:
            ax.set_ylabel("$c_1$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Mystery problem ($n_\\mathrm{{pop}}={PSIZE}$, ct=$\\mathrm{{{CT_LEVEL}}}$): cumulative non-dominated archives that drive HV"
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Written {OUT_PATH}")


if __name__ == "__main__":
    main()
