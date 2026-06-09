"""Contour of best achievable objective in (c_1, c_2) space...

...with the threshold-feasible non-dominated front overlaid per mode.

For a two-constraint problem, builds a backdrop where each (c_1, c_2)
bin shows the minimum objective observed at that constraint level across
all modes / runs / generations of an experiment. Then, per mode, scatters
the threshold-feasible non-dominated front (non-dominated in
(f, c_1, c_2) jointly, minimizing all three) on top.

Dashed lines mark strict feasibility (c=0) and the relaxation thresholds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import moocore
import numpy as np
import polars as pl
import yaml
from scipy.ndimage import gaussian_filter


def bin_min(c1: np.ndarray, c2: np.ndarray, f: np.ndarray, xlim: tuple, ylim: tuple, nbins: int) -> np.ndarray:
    """Return a (nbins, nbins) grid of min(f) per (c1, c2) bin; NaN where empty."""
    ix = np.clip(np.floor((c1 - xlim[0]) / (xlim[1] - xlim[0]) * nbins).astype(int), 0, nbins - 1)
    iy = np.clip(np.floor((c2 - ylim[0]) / (ylim[1] - ylim[0]) * nbins).astype(int), 0, nbins - 1)
    flat = ix * nbins + iy
    grid = np.full(nbins * nbins, np.nan)
    order = np.argsort(flat, kind="stable")
    flat_s, f_s = flat[order], f[order]
    starts = np.searchsorted(flat_s, np.arange(nbins * nbins))
    ends = np.append(starts[1:], len(flat_s))
    for k, (a, b) in enumerate(zip(starts, ends, strict=False)):
        if b > a:
            grid[k] = f_s[a:b].min()
    return grid.reshape(nbins, nbins)


def smooth_with_nan(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth treating NaNs as missing (Stocker's trick)."""
    mask = np.isfinite(grid).astype(float)
    g = np.where(mask > 0, grid, 0.0)
    num = gaussian_filter(g, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask, sigma=sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


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
    ap.add_argument("--xrange", nargs=2, type=float, default=None)
    ap.add_argument("--yrange", nargs=2, type=float, default=None)
    ap.add_argument(
        "--clip-pct",
        nargs=2,
        type=float,
        default=[1.0, 99.0],
        help="Percentile clipping for the color scale (default 1, 99)",
    )
    ap.add_argument(
        "--all-gens",
        action="store_true",
        help="use points from all generations for the ND front (default: final gen only)",
    )
    ap.add_argument(
        "--run", type=int, default=None, help="restrict the ND front to a single run id (default: all runs)"
    )
    ap.add_argument(
        "--density-contours", action="store_true", help="overlay per-mode density contour lines of population points"
    )
    ap.add_argument("--density-bins", type=int, default=50)
    ap.add_argument("--density-smooth", type=float, default=1.5)
    ap.add_argument("--density-levels", type=int, default=5)
    ap.add_argument(
        "--marginals", action="store_true", help="attach 1D marginal histograms (c_1 on top, c_2 on right) per panel"
    )
    ap.add_argument("--marginal-bins", type=int, default=50)
    ap.add_argument("--marker-size", type=float, default=8.0)
    ap.add_argument("--marker-alpha", type=float, default=0.55)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    problem_cfg = next(p for p in cfg["problems"] if p["name"] == args.problem)
    c_cols = problem_cfg["constraint_symbols"]
    if len(c_cols) != 2:
        raise SystemExit(f"{args.problem} has {len(c_cols)} constraints; this script requires exactly 2.")
    f_col = f"{problem_cfg['objective_symbol']}_min"

    thr_doc = yaml.safe_load((Path(args.results_dir) / "thresholds" / f"{args.problem}.yaml").read_text())
    thresholds = {c: float(thr_doc["levels"][args.ct].get(c, 0.0)) for c in c_cols}
    tau1, tau2 = thresholds[c_cols[0]], thresholds[c_cols[1]]

    modes = cfg["modes"]
    data_dir = Path(args.results_dir) / "data"
    mode_dfs_all = {}  # all runs, all gens (for contour backdrop)
    mode_dfs_nd = {}  # the subset used for the ND overlay
    for mode in modes:
        path = data_dir / f"{args.problem}_{mode}_gen{args.n_gen}_runs{args.runs}_ct{args.ct}_psize{args.psize}.parquet"
        if not path.exists():
            raise SystemExit(f"missing data file: {path}")
        full = pl.read_parquet(path)
        mode_dfs_all[mode] = full.select([f_col, *c_cols])

        sub = full
        if args.run is not None:
            sub = sub.filter(pl.col("run") == args.run)
            if sub.height == 0:
                raise SystemExit(f"run id {args.run} not present in {path.name}")
        if not args.all_gens:
            sub = sub.filter(pl.col("generation") == int(sub["generation"].max()))
        mode_dfs_nd[mode] = sub.select([f_col, *c_cols])

    # Reference-front optima for the strict and threshold best.
    fronts_dir = Path(args.results_dir) / "fronts"
    front_psize = int(cfg["population_size_front"])
    front_cands = sorted(fronts_dir.glob(f"{args.problem}_gen*_psize{front_psize}.parquet"))
    ref_strict_pt = None
    ref_relax_pt = None
    if front_cands:
        df_front = pl.read_parquet(front_cands[-1]).select([f_col, *c_cols])
        strict_front = df_front.filter(pl.all_horizontal([pl.col(c) <= 0.0 for c in c_cols]))
        if strict_front.height > 0:
            row = strict_front.sort(f_col).row(0, named=True)
            ref_strict_pt = (row[c_cols[0]], row[c_cols[1]], row[f_col])
        relax_front = df_front.filter(pl.all_horizontal([pl.col(c) <= thresholds[c] for c in c_cols]))
        if relax_front.height > 0:
            row = relax_front.sort(f_col).row(0, named=True)
            ref_relax_pt = (row[c_cols[0]], row[c_cols[1]], row[f_col])

    combined = pl.concat(list(mode_dfs_all.values()))
    c1_all = combined[c_cols[0]].to_numpy()
    c2_all = combined[c_cols[1]].to_numpy()
    f_all = combined[f_col].to_numpy()

    # Window: centered around the origin, just past the thresholds in the +/+ direction
    if args.xrange:
        xlim = tuple(args.xrange)
    else:
        xlim = (-tau1 * 1.2 if tau1 > 0 else -1.0, tau1 * 1.4 if tau1 > 0 else 1.0)
    if args.yrange:
        ylim = tuple(args.yrange)
    else:
        ylim = (-tau2 * 1.2 if tau2 > 0 else -1.0, tau2 * 1.4 if tau2 > 0 else 1.0)

    # Backdrop: bin-min objective over combined cloud (so all 3 panels share the same surface)
    in_window = (c1_all >= xlim[0]) & (c1_all <= xlim[1]) & (c2_all >= ylim[0]) & (c2_all <= ylim[1])
    grid = bin_min(c1_all[in_window], c2_all[in_window], f_all[in_window], xlim, ylim, args.nbins)
    if args.smooth > 0:
        grid = smooth_with_nan(grid, sigma=args.smooth)

    # Color scale: clip to percentiles of in-window f values
    f_in = f_all[in_window]
    vmin, vmax = np.percentile(f_in, args.clip_pct)

    xedges = np.linspace(xlim[0], xlim[1], args.nbins + 1)
    yedges = np.linspace(ylim[0], ylim[1], args.nbins + 1)
    Xc, Yc = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:]), indexing="ij")

    # Plot
    import matplotlib.gridspec as gridspec  # local import to keep top tidy

    n_modes = len(modes)
    if args.marginals:
        fig = plt.figure(figsize=(5.6 * n_modes, 5.4))
        outer = gridspec.GridSpec(
            1,
            n_modes + 1,
            width_ratios=[5.0] * n_modes + [0.4],
            wspace=0.18,
        )
        axes = []
        top_axes = []
        right_axes = []
        for i in range(n_modes):
            inner = gridspec.GridSpecFromSubplotSpec(
                2,
                2,
                subplot_spec=outer[0, i],
                width_ratios=[4.0, 1.0],
                height_ratios=[1.0, 4.0],
                wspace=0.04,
                hspace=0.04,
            )
            ax_main = fig.add_subplot(
                inner[1, 0], sharex=(axes[0] if axes else None), sharey=(axes[0] if axes else None)
            )
            ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main, sharey=(top_axes[0] if top_axes else None))
            ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main, sharex=(right_axes[0] if right_axes else None))
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            axes.append(ax_main)
            top_axes.append(ax_top)
            right_axes.append(ax_right)
        cax = fig.add_subplot(outer[0, n_modes])
    else:
        fig, axes = plt.subplots(1, n_modes, figsize=(5.0 * n_modes, 4.6), sharex=True, sharey=True)
        if n_modes == 1:
            axes = [axes]
        else:
            axes = list(axes)
        top_axes = [None] * n_modes
        right_axes = [None] * n_modes
        cax = None

    contour_levels = np.linspace(vmin, vmax, 12)
    for i, (ax, mode) in enumerate(zip(axes, modes)):
        cf = ax.contourf(Xc, Yc, grid, levels=contour_levels, cmap="viridis_r", extend="both")

        # Optional: per-mode density contour lines of population points within the window
        if args.density_contours:
            pts_arr = mode_dfs_nd[mode].select(c_cols).to_numpy()
            in_win = (
                (pts_arr[:, 0] >= xlim[0])
                & (pts_arr[:, 0] <= xlim[1])
                & (pts_arr[:, 1] >= ylim[0])
                & (pts_arr[:, 1] <= ylim[1])
            )
            if in_win.any():
                counts, xe, ye = np.histogram2d(
                    pts_arr[in_win, 0],
                    pts_arr[in_win, 1],
                    bins=args.density_bins,
                    range=[xlim, ylim],
                )
                density = gaussian_filter(counts.T, sigma=args.density_smooth)
                xc_d = 0.5 * (xe[1:] + xe[:-1])
                yc_d = 0.5 * (ye[1:] + ye[:-1])
                # Levels at log-spaced percentiles of nonzero density for robustness
                nz = density[density > 0]
                if nz.size > 0:
                    qs = np.linspace(0.55, 0.98, args.density_levels)
                    levels = np.unique(np.quantile(nz, qs))
                    if levels.size >= 1:
                        ax.contour(
                            xc_d,
                            yc_d,
                            density,
                            levels=levels,
                            colors="white",
                            linewidths=0.7,
                            alpha=0.85,
                        )

        # ND front per mode, on threshold-feasible points only
        df = mode_dfs_nd[mode]
        tf = df.filter((pl.col(c_cols[0]) <= tau1) & (pl.col(c_cols[1]) <= tau2))
        arr = tf.select([f_col, *c_cols]).to_numpy()
        n_strict = n_relax_only = 0
        if arr.shape[0] > 0:
            nd_mask = moocore.is_nondominated(arr, maximise=False)
            nd = arr[nd_mask]
            strict_mask = (nd[:, 1] <= 0.0) & (nd[:, 2] <= 0.0)
            n_strict = int(strict_mask.sum())
            n_relax_only = nd.shape[0] - n_strict
            # Strict-feasible ND: white-edged circles. Relax-only ND: red crosses.
            ax.scatter(
                nd[strict_mask, 1],
                nd[strict_mask, 2],
                marker="o",
                s=args.marker_size,
                facecolor="white",
                edgecolor="black",
                linewidths=0.5,
                alpha=args.marker_alpha,
                label=f"strict ND ({n_strict})",
            )
            ax.scatter(
                nd[~strict_mask, 1],
                nd[~strict_mask, 2],
                marker="x",
                s=args.marker_size * 1.4,
                c="red",
                linewidths=1.1,
                alpha=args.marker_alpha,
                label=f"relax-only ND ({n_relax_only})",
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.axvline(tau1, color="red", linestyle=":", linewidth=1.1)
        ax.axhline(tau2, color="red", linestyle=":", linewidth=1.1)

        # Reference-front optima (same on every panel)
        if ref_strict_pt is not None:
            ax.scatter(
                [ref_strict_pt[0]],
                [ref_strict_pt[1]],
                marker="*",
                s=180,
                facecolor="gold",
                edgecolor="black",
                linewidths=1.0,
                zorder=6,
                label=f"strict optimum (f={ref_strict_pt[2]:.4g})",
            )
        if ref_relax_pt is not None:
            ax.scatter(
                [ref_relax_pt[0]],
                [ref_relax_pt[1]],
                marker="*",
                s=180,
                facecolor="magenta",
                edgecolor="black",
                linewidths=1.0,
                zorder=6,
                label=f"threshold optimum (f={ref_relax_pt[2]:.4g})",
            )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left", framealpha=0.9, fontsize=7)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(c_cols[0])
        if not args.marginals:
            ax.set_title(mode.capitalize())
        ax.set_aspect("auto")

        # Marginal histograms
        if args.marginals:
            pts_arr = mode_dfs_nd[mode].select(c_cols).to_numpy()
            in_x = (pts_arr[:, 0] >= xlim[0]) & (pts_arr[:, 0] <= xlim[1])
            in_y = (pts_arr[:, 1] >= ylim[0]) & (pts_arr[:, 1] <= ylim[1])
            ax_top = top_axes[i]
            ax_right = right_axes[i]
            if in_x.any():
                ax_top.hist(
                    pts_arr[in_x, 0],
                    bins=args.marginal_bins,
                    range=xlim,
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.3,
                )
            ax_top.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax_top.axvline(tau1, color="red", linestyle=":", linewidth=1.0)
            ax_top.set_xlim(xlim)
            ax_top.tick_params(axis="y", labelsize=7)
            ax_top.set_ylabel("count", fontsize=8)
            ax_top.set_title(mode.capitalize())

            if in_y.any():
                ax_right.hist(
                    pts_arr[in_y, 1],
                    bins=args.marginal_bins,
                    range=ylim,
                    orientation="horizontal",
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.3,
                )
            ax_right.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax_right.axhline(tau2, color="red", linestyle=":", linewidth=1.0)
            ax_right.set_ylim(ylim)
            ax_right.tick_params(axis="x", labelsize=7)
            ax_right.set_xlabel("count", fontsize=8)

    axes[0].set_ylabel(c_cols[1])

    if cax is not None:
        cbar = fig.colorbar(cf, cax=cax)
    else:
        cbar = fig.colorbar(cf, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label(f"min objective ({f_col})")
    if args.run is None:
        scope = "all runs, all gens" if args.all_gens else "all runs, final gen"
    else:
        scope = f"run {args.run}, all gens" if args.all_gens else f"run {args.run}, final gen"
    fig.suptitle(
        f"{args.problem}  ct={args.ct}  "
        f"thresholds: {c_cols[0]}≤{tau1:.4g}, {c_cols[1]}≤{tau2:.4g}  ND overlay: {scope}",
        y=1.02,
    )

    out_path = args.output or f"results/figures/{args.problem}_ct{args.ct}_constraint_contour.pdf"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
