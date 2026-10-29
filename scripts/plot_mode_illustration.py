"""Illustrative, problem-agnostic figure of how each selection-operator mode ranks solutions.

Instead of a specific benchmark problem, this uses a synthetic landscape on the unit square:
the two constraints ``c_1, c_2`` and the objective are all scaled to ``[0, 1]``, and the
objective improves along a clean 45-degree diagonal (worst at the bottom-left corner, best at
the top-right). The strict-feasibility boundary is drawn at ``1/3`` and the relaxation threshold
at ``2/3``, so each axis has three equal bands (strict / relaxed-band / beyond-threshold) and the
constraint space splits into a 3x3 grid of regions. An equal number of solutions is placed in
every region so that each mode's ranking can be read off at a glance.

The ranks are produced by the *real* production selection operator
(``SingleObjectiveConstrainedRankingSelector.compute_order``), not a hand-drawn approximation:
the display coordinates are shifted by ``1/3`` so that the operator's built-in
strict-feasibility boundary (``c <= 0``) lands on the ``1/3`` line and its threshold on ``2/3``.

Layout mirrors the contourrank figure: an "Objective field" row with the numbered, ranked
solutions over the objective gradient, and a "Ranking field" row showing each mode's induced
normalized-rank field over the whole constraint space.

Example:
    uv run python scripts/plot_mode_illustration.py --output results/figures/mode_illustration.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from desdeo.emo.operators.selection import SingleObjectiveConstrainedRankingSelector
from desdeo.tools.patterns import Publisher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROBLEM_BUILDERS

# Okabe-Ito colour-blind-safe markers, matching the contour plots.
_FEAS_STYLE = {
    "strict": {"color": "#009E73", "label": "strict-feasible ($c\\leq 0$)"},
    "threshold": {"color": "#E69F00", "label": "threshold-feasible only"},
    "infeasible": {"color": "#D55E00", "label": "infeasible ($c>$ threshold)"},
}

# Band boundaries in display coordinates (unit square). The beyond-threshold band is deliberately
# narrower than the strict and relaxed bands so the interesting in-box trade-off dominates.
STRICT_LINE = 0.4  # strict-feasibility boundary (operator sees this as c = 0)
THRESH_LINE = 0.8  # relaxation threshold (operator sees this as c = t)


def objective(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """45-degree gradient on the unit square: 1 (worst) at (0,0), 0 (best) at (1,1)."""
    return 1.0 - 0.5 * (c1 + c2)


def feas_category(c1: float, c2: float) -> str:
    """Classify a display-coordinate point into the three feasibility bands."""
    if c1 <= STRICT_LINE and c2 <= STRICT_LINE:
        return "strict"
    if c1 <= THRESH_LINE and c2 <= THRESH_LINE:
        return "threshold"
    return "infeasible"


def rank_with_mode(selector: SingleObjectiveConstrainedRankingSelector, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Return the 1-based rank (1 = best) each (c1, c2) solution receives under the operator."""
    f = objective(c1, c2)
    # Shift display coords so the operator's strict boundary (c <= 0) sits at STRICT_LINE.
    c_op = np.column_stack([c1 - STRICT_LINE, c2 - STRICT_LINE])
    order = selector.compute_order(f.copy(), c_op.copy())
    rank = np.empty(len(f), dtype=int)
    rank[order] = np.arange(1, len(f) + 1)
    return rank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiment_config.yaml")
    ap.add_argument("--n-feasible", type=int, default=2, help="solutions per strict/threshold-feasible region")
    ap.add_argument("--n-infeasible", type=int, default=1, help="solutions per beyond-threshold region")
    ap.add_argument("--rank-nbins", type=int, default=70, help="grid resolution for the ranking field")
    ap.add_argument(
        "--field-smooth",
        type=float,
        default=0.8,
        help="Gaussian sigma (grid cells) applied only to the hypervolume ranking field to de-speckle it; 0 disables",
    )
    ap.add_argument("--modes", nargs="*", default=["baseline", "relaxed", "ranking", "hypervolume"])
    ap.add_argument("--output", default="results/figures/mode_illustration.pdf")
    args = ap.parse_args()

    # Any 2-constraint problem provides a valid Problem object for the operator's constructor;
    # compute_order only uses the thresholds and the values we pass, not the problem's landscape.
    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = next(p for p in cfg["problems"] if len(p["constraint_symbols"]) == 2)
    obj_sym = pcfg["objective_symbol"]
    c_syms = list(pcfg["constraint_symbols"])[:2]
    niching = str(cfg.get("ranking_niching", "kth"))
    niching_k = int(cfg.get("ranking_niching_k", 3))
    problem = PROBLEM_BUILDERS[pcfg["name"]]()
    modes = args.modes
    t_op = THRESH_LINE - STRICT_LINE  # operator threshold

    def make_selector(mode: str, pop: int) -> SingleObjectiveConstrainedRankingSelector:
        return SingleObjectiveConstrainedRankingSelector(
            problem=problem,
            verbosity=0,
            publisher=Publisher(),
            population_size=pop,
            target_objective_symbol=obj_sym,
            mode=mode,
            constraints={c: t_op for c in c_syms},
            niching=niching,
            niching_k=niching_k,
        )

    # Solutions per region, along each region's diagonal so they differ in objective (exposing
    # intra-region ranking). Beyond-threshold regions get fewer solutions since the in-box
    # trade-off is what the modes actually differ on.
    edges = [0.0, STRICT_LINE, THRESH_LINE, 1.0]

    def region_category(bi: int, bj: int) -> str:
        if bi == 0 and bj == 0:
            return "strict"
        if bi <= 1 and bj <= 1:
            return "threshold"
        return "infeasible"

    def subs_for(count: int) -> np.ndarray:
        return np.array([0.5]) if count == 1 else np.linspace(0.30, 0.70, count)

    sc1, sc2 = [], []
    for bi in range(3):
        for bj in range(3):
            count = args.n_infeasible if region_category(bi, bj) == "infeasible" else args.n_feasible
            w1, w2 = edges[bi + 1] - edges[bi], edges[bj + 1] - edges[bj]
            for u in subs_for(count):
                sc1.append(edges[bi] + u * w1)
                sc2.append(edges[bj] + u * w2)
    sc1, sc2 = np.asarray(sc1), np.asarray(sc2)
    n = len(sc1)
    cats = [feas_category(sc1[i], sc2[i]) for i in range(n)]

    ranks_by_mode = {m: rank_with_mode(make_selector(m, n), sc1, sc2) for m in modes}

    # Objective backdrop.
    gg = np.linspace(0.0, 1.0, 300)
    GX, GY = np.meshgrid(gg, gg, indexing="ij")
    Z = objective(GX, GY)
    obj_levels = np.linspace(0.0, 1.0, 11)

    # Ranking field: rank a probe placed at every grid cell (all probes ranked together).
    fc = np.linspace(0.0, 1.0, args.rank_nbins)
    FX, FY = np.meshgrid(fc, fc, indexing="ij")
    pf1, pf2 = FX.ravel(), FY.ravel()
    fields = {}
    for m in modes:
        rk = rank_with_mode(make_selector(m, len(pf1)), pf1, pf2)
        fields[m] = ((rk - 1) / max(1, len(pf1) - 1)).reshape(args.rank_nbins, args.rank_nbins)

    strict_opt = (STRICT_LINE, STRICT_LINE)  # best objective within the strict-feasible region
    thresh_opt = (THRESH_LINE, THRESH_LINE)  # best objective within the threshold-feasible box

    # Thick lines with a white halo so that they stay visible over the filled contours in print.
    halo = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]

    def ref_lines(ax: plt.Axes) -> None:
        for draw in (ax.axvline, ax.axhline):
            draw(STRICT_LINE, color="black", ls="--", lw=2.0, path_effects=halo, zorder=4)
            draw(THRESH_LINE, color="red", ls=":", lw=2.4, path_effects=halo, zorder=4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("$c_1$")

    feas_handles = [
        plt.Line2D([], [], marker="o", ls="", ms=10, mfc=s["color"], mec="black", label=s["label"])
        for s in _FEAS_STYLE.values()
    ]
    star_handles = [
        plt.Line2D([], [], marker="*", ls="", ms=16, mfc="gold", mec="black", label="strict optimum"),
        plt.Line2D([], [], marker="*", ls="", ms=16, mfc="magenta", mec="black", label="threshold optimum"),
    ]
    line_handles = [
        plt.Line2D([], [], color="black", ls="--", lw=2.0, label="strict-feasibility boundary"),
        plt.Line2D([], [], color="red", ls=":", lw=2.4, label="relaxation threshold"),
    ]

    n_modes = len(modes)
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(4.0 * n_modes, 10.2))
    top_sf, bot_sf = fig.subfigures(2, 1, hspace=0.03)

    # Objective field row.
    ax_top = np.atleast_1d(top_sf.subplots(1, n_modes, sharex=True, sharey=True))
    cf_obj = None
    for i, m in enumerate(modes):
        ax = ax_top[i]
        cf_obj = ax.contourf(GX, GY, Z, levels=obj_levels, cmap="viridis_r", extend="neither")
        for cat, style in _FEAS_STYLE.items():
            idx = [k for k in range(n) if cats[k] == cat]
            ax.scatter(sc1[idx], sc2[idx], c=style["color"], edgecolor="black", linewidths=0.8, s=150, zorder=6)
        rank = ranks_by_mode[m]
        for k in range(n):
            ax.annotate(
                str(rank[k]),
                (sc1[k], sc2[k]),
                ha="left",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )
        ax.scatter(*strict_opt, marker="*", c="gold", edgecolor="black", s=280, zorder=8)
        ax.scatter(*thresh_opt, marker="*", c="magenta", edgecolor="black", s=280, zorder=8)
        ref_lines(ax)
        ax.set_title(f"{m.capitalize()} mode")
    ax_top[0].set_ylabel("$c_2$")
    cb1 = top_sf.colorbar(cf_obj, ax=list(ax_top), pad=0.02)
    cb1.set_label("objective $f$ (0 = best)")
    cb1.set_ticks(np.linspace(0.0, 1.0, 6))
    top_sf.suptitle("Objective field", fontweight="bold")

    # Ranking field row.
    ax_bot = np.atleast_1d(bot_sf.subplots(1, n_modes, sharex=True, sharey=True))
    rank_levels = np.linspace(0.0, 1.0, 11)
    cf_rank = None
    for i, m in enumerate(modes):
        field = fields[m]
        # Only the greedy hypervolume field is speckled; the lexicographic modes stay crisp.
        if m == "hypervolume" and args.field_smooth > 0:
            field = gaussian_filter(field, sigma=args.field_smooth)
        cf_rank = ax_bot[i].contourf(FX, FY, field, levels=rank_levels, cmap="plasma_r")
        ref_lines(ax_bot[i])
        ax_bot[i].set_title(f"{m.capitalize()} mode")
    ax_bot[0].set_ylabel("$c_2$")
    cb2 = bot_sf.colorbar(cf_rank, ax=list(ax_bot), pad=0.02)
    cb2.set_label("ranking field (normalized rank, 0 = best)")
    cb2.set_ticks(np.linspace(0.0, 1.0, 6))
    bot_sf.suptitle("Ranking field", fontweight="bold")
    fig.legend(
        handles=feas_handles + star_handles + line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=14,
        framealpha=0.92,
        handletextpad=0.4,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
