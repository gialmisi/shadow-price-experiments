"""Show how each selection-operator mode ranks a shared set of random solutions.

For a two-constraint problem, this samples a small set of random candidate solutions in
(c_1, c_2) constraint space, assigns each the best objective observed at its location (read
off a min-objective contour surface), and ranks all candidates with each mode's selection
operator (Baseline / Relaxed / Ranking; rank 1 = best / first to be selected). Candidates
are coloured and id-lettered by feasibility category (strict-feasible: c <= 0 on all
constraints; threshold-feasible only: c <= threshold but not strict; infeasible: some
c > threshold).

Two layouts (``--layout``):

  * ``combo`` (default): a constraint-space scatter (with the contour backdrop, for spatial
    and objective context) beside a rank-flow bump chart. In the bump chart each candidate
    is a line across the three modes' rank axes; lines that cross are exactly the solutions
    the modes rank differently. This is the clearest view of *how* the modes disagree.
  * ``contour``: one contour panel per mode, each candidate annotated with its integer rank
    in that mode.

This is a standalone analysis/illustration tool, not part of the Snakemake pipeline.

Example:
    uv run python scripts/plot_mode_ranking_contour.py --problem cantilevered_beam --ct med
"""

import argparse
import string
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from desdeo.emo.operators.selection import SingleObjectiveConstrainedRankingSelector
from desdeo.tools.patterns import Publisher

# Reuse the exact backdrop construction from the contour script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_constraint_objective_contour import bin_min, smooth_with_nan
from utils import PROBLEM_BUILDERS

# Okabe-Ito colour-blind-safe, print-friendly qualitative colours.
_FEAS_STYLE = {
    "strict": {"color": "#009E73", "label": "strict-feasible ($c\\leq 0$)"},
    "threshold": {"color": "#E69F00", "label": "threshold-feasible only"},
    "infeasible": {"color": "#D55E00", "label": "infeasible ($c>$ threshold)"},
}


def feasibility_category(c1: float, c2: float, tau1: float, tau2: float) -> str:
    """Classify a point by strict / threshold-only / infeasible feasibility."""
    if c1 <= 0.0 and c2 <= 0.0:
        return "strict"
    if c1 <= tau1 and c2 <= tau2:
        return "threshold"
    return "infeasible"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiment_config.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--problem", default="cantilevered_beam")
    ap.add_argument("--ct", default="med")
    ap.add_argument("--psize", type=int, default=36)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--smooth", type=float, default=1.2, help="Gaussian sigma in bins; 0 disables")
    ap.add_argument("--n-points", type=int, default=12, help="number of random candidate solutions to rank")
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--niching", default=None, help="ranking-mode niching; default: from config")
    ap.add_argument("--niching-k", type=int, default=None)
    ap.add_argument("--clip-pct", nargs=2, type=float, default=[1.0, 99.0])
    ap.add_argument(
        "--layout",
        choices=["combo", "contour", "contourrank"],
        default="combo",
        help=(
            "combo: constraint-space scatter + rank-flow bump chart; "
            "contour: 3-panel objective contour with rank numbers; "
            "contourrank: contour layout with a second row showing each mode's ranking field"
        ),
    )
    ap.add_argument("--rank-nbins", type=int, default=70, help="grid resolution for the ranking field (contourrank)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    problem_cfg = next(p for p in cfg["problems"] if p["name"] == args.problem)
    c_cols = list(problem_cfg["constraint_symbols"])
    if len(c_cols) != 2:
        raise SystemExit(f"{args.problem} has {len(c_cols)} constraints; this script requires exactly 2.")
    obj_sym = problem_cfg["objective_symbol"]
    f_col = f"{obj_sym}_min"

    thr_doc = yaml.safe_load((Path(args.results_dir) / "thresholds" / f"{args.problem}.yaml").read_text())
    thresholds = {c: float(thr_doc["levels"][args.ct].get(c, 0.0)) for c in c_cols}
    tau1, tau2 = thresholds[c_cols[0]], thresholds[c_cols[1]]

    modes = cfg["modes"]
    niching = args.niching or str(cfg.get("ranking_niching", "kth"))
    niching_k = args.niching_k if args.niching_k is not None else int(cfg.get("ranking_niching_k", 3))

    # Backdrop: min objective per (c_1, c_2) bin over the combined cloud of all modes (final gen).
    data_dir = Path(args.results_dir) / "data"
    frames = []
    for mode in modes:
        path = data_dir / f"{args.problem}_{mode}_gen{args.n_gen}_runs{args.runs}_ct{args.ct}_psize{args.psize}.parquet"
        if not path.exists():
            raise SystemExit(f"missing data file: {path}")
        full = pl.read_parquet(path)
        full = full.filter(pl.col("generation") == int(full["generation"].max()))
        frames.append(full.select([f_col, *c_cols]))
    combined = pl.concat(frames)
    c1_all = combined[c_cols[0]].to_numpy()
    c2_all = combined[c_cols[1]].to_numpy()
    f_all = combined[f_col].to_numpy()

    xlim = (-tau1 * 1.2 if tau1 > 0 else -1.0, tau1 * 1.4 if tau1 > 0 else 1.0)
    ylim = (-tau2 * 1.2 if tau2 > 0 else -1.0, tau2 * 1.4 if tau2 > 0 else 1.0)

    in_window = (c1_all >= xlim[0]) & (c1_all <= xlim[1]) & (c2_all >= ylim[0]) & (c2_all <= ylim[1])
    grid = bin_min(c1_all[in_window], c2_all[in_window], f_all[in_window], xlim, ylim, args.nbins)
    if args.smooth > 0:
        grid = smooth_with_nan(grid, sigma=args.smooth)

    f_in = f_all[in_window]
    vmin, vmax = np.percentile(f_in, args.clip_pct)

    xedges = np.linspace(xlim[0], xlim[1], args.nbins + 1)
    yedges = np.linspace(ylim[0], ylim[1], args.nbins + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")

    def grid_lookup(px: float, py: float) -> float:
        ix = int(np.clip(np.searchsorted(xedges, px) - 1, 0, args.nbins - 1))
        iy = int(np.clip(np.searchsorted(yedges, py) - 1, 0, args.nbins - 1))
        return float(grid[ix, iy])

    # Generate random candidate solutions inside the window where the surface is defined.
    rng = np.random.default_rng(args.seed)
    pts_c1, pts_c2, pts_f = [], [], []
    attempts = 0
    while len(pts_c1) < args.n_points and attempts < 100000:
        attempts += 1
        px = rng.uniform(*xlim)
        py = rng.uniform(*ylim)
        fv = grid_lookup(px, py)
        if not np.isfinite(fv):
            continue
        pts_c1.append(px)
        pts_c2.append(py)
        pts_f.append(fv)
    if len(pts_c1) < args.n_points:
        raise SystemExit("could not sample enough points with a defined objective; lower --n-points or widen window")

    pts_c1 = np.asarray(pts_c1)
    pts_c2 = np.asarray(pts_c2)
    pts_f = np.asarray(pts_f)
    c_mat = np.column_stack([pts_c1, pts_c2])
    cats = [feasibility_category(pts_c1[i], pts_c2[i], tau1, tau2) for i in range(args.n_points)]

    # Rank the candidates with each mode using the production selection operator.
    problem = PROBLEM_BUILDERS[args.problem]()
    publisher = Publisher()
    ranks_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        selector = SingleObjectiveConstrainedRankingSelector(
            problem=problem,
            verbosity=0,
            publisher=publisher,
            population_size=args.n_points,
            target_objective_symbol=obj_sym,
            mode=mode,
            constraints=thresholds,
            niching=niching,
            niching_k=niching_k,
        )
        order = selector.compute_order(pts_f.copy(), c_mat.copy())
        rank = np.empty(args.n_points, dtype=int)
        rank[order] = np.arange(1, args.n_points + 1)
        ranks_by_mode[mode] = rank

    # Reference-front optima (same on every panel).
    fronts_dir = Path(args.results_dir) / "fronts"
    front_psize = int(cfg["population_size_front"])
    front_cands = sorted(fronts_dir.glob(f"{args.problem}_gen*_psize{front_psize}.parquet"))
    ref_strict_pt = ref_relax_pt = None
    if front_cands:
        df_front = pl.read_parquet(front_cands[-1]).select([f_col, *c_cols])
        sf = df_front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols]))
        if sf.height > 0:
            r = sf.sort(f_col).row(0, named=True)
            ref_strict_pt = (r[c_cols[0]], r[c_cols[1]])
        rf = df_front.filter(pl.all_horizontal([pl.col(c) <= thresholds[c] for c in c_cols]))
        if rf.height > 0:
            r = rf.sort(f_col).row(0, named=True)
            ref_relax_pt = (r[c_cols[0]], r[c_cols[1]])

    # Plot.
    n_pts = args.n_points
    ids = list(string.ascii_uppercase)[:n_pts]
    contour_levels = np.linspace(vmin, vmax, 12)
    cx_lab = f"${c_cols[0]}$"
    cy_lab = f"${c_cols[1]}$"
    obj_lab = f"min objective (${obj_sym}$)"
    suptitle = (
        f"{args.problem}  ct={args.ct}  thresholds: ${c_cols[0]} \\leq {tau1:.4g}$, ${c_cols[1]} \\leq {tau2:.4g}$  "
        f"-- {n_pts} random candidates ranked by each mode"
    )

    feas_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markerfacecolor=s["color"], markeredgecolor="black", label=s["label"]
        )
        for s in _FEAS_STYLE.values()
    ]
    star_handles = [
        plt.Line2D(
            [], [], marker="*", linestyle="", markerfacecolor="gold", markeredgecolor="black", label="strict optimum"
        ),
        plt.Line2D(
            [],
            [],
            marker="*",
            linestyle="",
            markerfacecolor="magenta",
            markeredgecolor="black",
            label="threshold optimum",
        ),
    ]

    def add_reference_lines(ax: plt.Axes) -> None:
        """Draw the feasibility (c=0) and threshold reference lines and set axis limits/label."""
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(tau1, color="red", linestyle=":", linewidth=1.1)
        ax.axhline(tau2, color="red", linestyle=":", linewidth=1.1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(cx_lab)

    def add_overlays(ax: plt.Axes, point_labels: list[str]) -> None:
        """Reference lines plus the reference optima and the candidate markers with labels."""
        add_reference_lines(ax)
        if ref_strict_pt is not None:
            ax.scatter(*ref_strict_pt, marker="*", s=160, facecolor="gold", edgecolor="black", linewidths=1.0, zorder=5)
        if ref_relax_pt is not None:
            ax.scatter(
                *ref_relax_pt, marker="*", s=160, facecolor="magenta", edgecolor="black", linewidths=1.0, zorder=5
            )
        for i in range(n_pts):
            ax.scatter(
                pts_c1[i],
                pts_c2[i],
                marker="o",
                s=80,
                facecolor=_FEAS_STYLE[cats[i]]["color"],
                edgecolor="black",
                linewidths=0.8,
                zorder=6,
            )
            # Anchor the label's lower-left corner at the marker centre so it overlaps the circle a little.
            ax.annotate(
                point_labels[i],
                (pts_c1[i], pts_c2[i]),
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="black",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

    def obj_backdrop(ax: plt.Axes) -> "plt.cm.ScalarMappable":
        # Clip into the colour range so the colorbar has flat ends (no extend triangles),
        # matching the ranking-field colorbar. NaNs (no data) are preserved and stay unfilled.
        clipped = np.clip(grid, vmin, vmax)
        return ax.contourf(Xc, Yc, clipped, levels=contour_levels, cmap="viridis_r", extend="neither")

    def draw_scatter(ax: plt.Axes, *, with_ids: bool, rank: np.ndarray | None = None) -> "plt.cm.ScalarMappable":
        cf = obj_backdrop(ax)
        add_overlays(ax, list(ids) if with_ids else [str(r) for r in rank])
        return cf

    def compute_rank_fields(nb: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Per mode, the normalized rank (0 = best) a probe solution would get at each grid cell.

        A probe is placed at every grid-cell centre with objective read from the same backdrop
        surface; all probes are ranked together by the mode's selection operator. The resulting
        field exposes the decision boundaries each mode induces over constraint space.
        """
        xe_r = np.linspace(xlim[0], xlim[1], nb + 1)
        ye_r = np.linspace(ylim[0], ylim[1], nb + 1)
        xc_r = 0.5 * (xe_r[:-1] + xe_r[1:])
        yc_r = 0.5 * (ye_r[:-1] + ye_r[1:])
        Xr, Yr = np.meshgrid(xc_r, yc_r, indexing="ij")
        idxs, probe_f = [], []
        for ix in range(nb):
            for iy in range(nb):
                fv = grid_lookup(xc_r[ix], yc_r[iy])
                if np.isfinite(fv):
                    idxs.append((ix, iy))
                    probe_f.append(fv)
        probe_c = np.array([[xc_r[ix], yc_r[iy]] for ix, iy in idxs], dtype=float)
        probe_f = np.asarray(probe_f, dtype=float)
        denom = max(1, len(probe_f) - 1)
        fields: dict[str, np.ndarray] = {}
        for mode in modes:
            sel = SingleObjectiveConstrainedRankingSelector(
                problem=problem,
                verbosity=0,
                publisher=publisher,
                population_size=len(probe_f),
                target_objective_symbol=obj_sym,
                mode=mode,
                constraints=thresholds,
                niching=niching,
                niching_k=niching_k,
            )
            order = sel.compute_order(probe_f.copy(), probe_c.copy())
            rk = np.empty(len(probe_f))
            rk[order] = np.arange(len(probe_f))
            field = np.full((nb, nb), np.nan)
            for k, (ix, iy) in enumerate(idxs):
                field[ix, iy] = rk[k] / denom
            fields[mode] = field
        return Xr, Yr, fields

    if args.layout == "contour":
        n_modes = len(modes)
        fig, axes = plt.subplots(1, n_modes, figsize=(5.2 * n_modes, 4.8), sharex=True, sharey=True)
        axes = list(np.atleast_1d(axes))
        cf = None
        for ax, mode in zip(axes, modes, strict=True):
            cf = draw_scatter(ax, with_ids=False, rank=ranks_by_mode[mode])
            ax.set_title(f"{mode.capitalize()}  (rank: 1 = best)")
        axes[0].set_ylabel(cy_lab)
        # Colorbar pushed to the top of the right margin, leaving room for the legend beneath it.
        cbar = fig.colorbar(cf, ax=axes, shrink=0.66, pad=0.02, anchor=(0.0, 1.0), panchor=(0.0, 1.0))
        cbar.set_label(obj_lab)
        cpos = cbar.ax.get_position()
        fig.legend(
            handles=feas_handles + star_handles,
            loc="upper center",
            bbox_to_anchor=(cpos.x0 + cpos.width / 2.0, cpos.y0 - 0.05),
            bbox_transform=fig.transFigure,
            fontsize=7,
            framealpha=0.92,
            ncol=1,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.suptitle(suptitle, y=1.02)
    elif args.layout == "contourrank":
        n_modes = len(modes)
        fig, axes = plt.subplots(2, n_modes, figsize=(5.2 * n_modes, 9.4), sharex=True, sharey=True)
        cf_obj = None
        for i, mode in enumerate(modes):
            cf_obj = draw_scatter(axes[0][i], with_ids=False, rank=ranks_by_mode[mode])
            axes[0][i].set_title(f"{mode.capitalize()}  (rank: 1 = best)")
        axes[0][0].set_ylabel(cy_lab)

        Xr, Yr, fields = compute_rank_fields(args.rank_nbins)
        rank_levels = np.linspace(0.0, 1.0, 11)
        cf_rank = None
        for i, mode in enumerate(modes):
            # Field row: clean field with only the reference lines (no candidate markers).
            # Distinct colourmap from the objective row (still perceptually uniform / CVD-safe).
            cf_rank = axes[1][i].contourf(Xr, Yr, fields[mode], levels=rank_levels, cmap="plasma_r")
            add_reference_lines(axes[1][i])
            axes[1][i].set_title(f"{mode.capitalize()}  ranking field")
        axes[1][0].set_ylabel(cy_lab)

        # Matching gradient scales: same colormap and geometry for both rows, anchored to the top
        # of each row's right margin (leaving room for the legend beneath the lower one).
        cb1 = fig.colorbar(cf_obj, ax=list(axes[0]), shrink=0.7, pad=0.02, anchor=(0.0, 1.0), panchor=(0.0, 1.0))
        cb1.set_label(obj_lab)
        cb2 = fig.colorbar(cf_rank, ax=list(axes[1]), shrink=0.7, pad=0.02, anchor=(0.0, 1.0), panchor=(0.0, 1.0))
        cb2.set_label("ranking field (normalized rank, 0 = best)")
        cpos = cb2.ax.get_position()
        fig.legend(
            handles=feas_handles + star_handles,
            loc="upper center",
            bbox_to_anchor=(cpos.x0 + cpos.width / 2.0, cpos.y0 - 0.04),
            bbox_transform=fig.transFigure,
            fontsize=7,
            framealpha=0.92,
            ncol=1,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        fig.suptitle(suptitle, y=1.01)
    else:  # combo: constraint-space scatter + rank-flow bump chart
        from matplotlib import gridspec

        fig = plt.figure(figsize=(13.0, 5.6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.22, 1.0], wspace=0.52)
        ax_sc = fig.add_subplot(gs[0, 0])
        ax_bp = fig.add_subplot(gs[0, 1])

        cf = draw_scatter(ax_sc, with_ids=True)
        ax_sc.set_ylabel(cy_lab)
        ax_sc.set_title("candidate solutions (constraint space)")
        ax_sc.legend(handles=feas_handles + star_handles, loc="best", fontsize=7, framealpha=0.92)
        cbar = fig.colorbar(cf, ax=ax_sc, fraction=0.046, pad=0.04)
        cbar.set_label(obj_lab)

        # Rank-flow bump chart: one line per candidate across the three modes' rank axes.
        xcols = np.arange(len(modes))
        rank_matrix = np.array([ranks_by_mode[m] for m in modes])  # (n_modes, n_pts)
        for i in range(n_pts):
            ys = rank_matrix[:, i]
            color = _FEAS_STYLE[cats[i]]["color"]
            ax_bp.plot(
                xcols, ys, "-o", color=color, markeredgecolor="black", markersize=6, linewidth=1.6, alpha=0.85, zorder=3
            )
            ax_bp.annotate(
                ids[i],
                (xcols[0], ys[0]),
                xytext=(-13, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
                ha="right",
                fontweight="bold",
            )
            ax_bp.annotate(
                ids[i],
                (xcols[-1], ys[-1]),
                xytext=(13, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
                ha="left",
                fontweight="bold",
            )
        ax_bp.set_xticks(xcols)
        ax_bp.set_xticklabels([m.capitalize() for m in modes])
        ax_bp.set_yticks(range(1, n_pts + 1))
        ax_bp.set_ylim(n_pts + 0.6, 0.4)  # invert so rank 1 is on top
        ax_bp.set_xlim(-0.6, len(modes) - 0.4)
        ax_bp.set_ylabel("rank (1 = best / first selected)")
        ax_bp.set_title("how each mode ranks them (crossing lines = disagreement)")
        ax_bp.grid(axis="y", linestyle=":", alpha=0.4)
        fig.suptitle(suptitle, y=1.01)

    out_path = args.output or f"results/figures/{args.problem}_ct{args.ct}_mode_ranking_{args.layout}.pdf"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
