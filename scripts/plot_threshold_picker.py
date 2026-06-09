"""Interactive Plotly constraint-image plots for picking thresholds.

For each problem:

- 1 constraint: (c_1, f) scatter (objective on y-axis, no color needed).
- 2 constraints: filled contour of bin-min(f) over (c_1, c_2).
- >=3 constraints: subplot grid of bin-min(f) contours over each (c_i, c_j) pair,
  sharing a common color scale.

Overlays the strict-feasibility line (c = 0) and the current low/med/high
threshold lines per constraint so the gap to the actual image is visible.

Output: one self-contained HTML file per problem in
results/figures/threshold_picker/ (plotly.js loaded from CDN).
"""

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

THRESHOLD_COLORS = {"low": "#33aa33", "med": "#ff8800", "high": "#cc3333"}


def bin_min(
    c1: np.ndarray,
    c2: np.ndarray,
    f: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nbins: int,
) -> np.ndarray:
    """(nbins, nbins) grid of min(f) per (c1, c2) bin, indexed [ix, iy]; NaN where empty."""
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


def compute_range(
    values: np.ndarray,
    thresholds: list[float | None],
    neg_factor: float = 1.2,
    pos_factor: float = 1.4,
) -> tuple[float, float]:
    """Window centered around the current thresholds: [-neg_factor*tau_max, pos_factor*tau_max].

    Falls back to data range with a 10% margin when no positive threshold exists.
    """
    valid_pos = [float(t) for t in thresholds if t is not None and float(t) > 0.0]
    if valid_pos:
        tau_max = max(valid_pos)
        return -neg_factor * tau_max, pos_factor * tau_max
    return full_range(values)


def full_range(values: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """Range covering all data with `margin` extension on both sides."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    width = hi - lo if hi > lo else 1.0
    return lo - margin * width, hi + margin * width


def load_reference_front(results_dir: Path, problem: str, f_col: str, c_cols: list[str]) -> pl.DataFrame:
    fronts_dir = results_dir / "fronts"
    cands = sorted(fronts_dir.glob(f"{problem}_gen*_psize*.parquet"))
    if not cands:
        raise SystemExit(f"no reference front found for {problem} in {fronts_dir}")
    return pl.read_parquet(cands[-1]).select([f_col, *c_cols])


def add_threshold_lines_1d(
    fig: go.Figure,
    c_col: str,
    thresholds_by_level: dict[str, dict],
    row: int | None = None,
    col: int | None = None,
) -> None:
    fig.add_vline(x=0, line={"color": "black", "dash": "dash", "width": 1}, row=row, col=col)
    for lvl, levels in thresholds_by_level.items():
        val = levels.get(c_col)
        if val is None:
            continue
        color = THRESHOLD_COLORS.get(lvl, "#888")
        fig.add_vline(
            x=float(val),
            line={"color": color, "dash": "dot", "width": 1},
            annotation={"text": lvl, "font": {"color": color}},
            annotation_position="top",
            row=row,
            col=col,
        )


def add_threshold_lines_2d(
    fig: go.Figure,
    row: int,
    col: int,
    c_x: str,
    c_y: str,
    thresholds_by_level: dict[str, dict],
) -> None:
    fig.add_vline(x=0, line={"color": "black", "dash": "dash", "width": 1}, row=row, col=col)
    fig.add_hline(y=0, line={"color": "black", "dash": "dash", "width": 1}, row=row, col=col)
    for lvl, levels in thresholds_by_level.items():
        color = THRESHOLD_COLORS.get(lvl, "#888")
        vx, vy = levels.get(c_x), levels.get(c_y)
        if vx is not None:
            fig.add_vline(x=float(vx), line={"color": color, "dash": "dot", "width": 1}, row=row, col=col)
        if vy is not None:
            fig.add_hline(y=float(vy), line={"color": color, "dash": "dot", "width": 1}, row=row, col=col)


def make_contour_trace(
    c_x: np.ndarray,
    c_y: np.ndarray,
    f: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nbins: int,
    sigma: float,
    f_col: str,
) -> go.Contour:
    grid = bin_min(c_x, c_y, f, xlim, ylim, nbins)
    if sigma > 0:
        grid = smooth_with_nan(grid, sigma=sigma)
    xedges = np.linspace(xlim[0], xlim[1], nbins + 1)
    yedges = np.linspace(ylim[0], ylim[1], nbins + 1)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    # bin_min returns grid[ix, iy]; plotly Contour expects z[iy, ix]
    return go.Contour(
        x=x_centers,
        y=y_centers,
        z=grid.T,
        coloraxis="coloraxis",
        contours={"coloring": "fill", "showlines": False},
        connectgaps=False,
        hovertemplate=f"x=%{{x:.4g}}<br>y=%{{y:.4g}}<br>min({f_col})=%{{z:.4g}}<extra></extra>",
        showscale=False,
    )


def plot_problem(
    problem_cfg: dict,
    results_dir: Path,
    output_dir: Path,
    nbins: int,
    sigma: float,
    clip_pct: tuple[float, float],
    scatter_max_points: int | None,
) -> Path:
    name = problem_cfg["name"]
    f_col = f"{problem_cfg['objective_symbol']}_min"
    c_cols = list(problem_cfg["constraint_symbols"])
    df = load_reference_front(results_dir, name, f_col, c_cols)

    # Thresholds: prefer the hand-picked values in experiment_config.yaml. Fall
    # back to results/thresholds/<name>.yaml when 'manual_thresholds' is absent
    # (e.g. running this script against a config that pre-dates the switch).
    thresholds_by_level: dict[str, dict] = {}
    if problem_cfg.get("manual_thresholds"):
        thresholds_by_level = {lvl: dict(vals) for lvl, vals in problem_cfg["manual_thresholds"].items()}
    else:
        thr_path = results_dir / "thresholds" / f"{name}.yaml"
        if thr_path.exists():
            thresholds_by_level = yaml.safe_load(thr_path.read_text()).get("levels", {}) or {}

    f_vals = df[f_col].to_numpy()

    if len(c_cols) == 1:
        c = c_cols[0]
        scatter_df = df
        if scatter_max_points and scatter_df.height > scatter_max_points:
            step = max(1, scatter_df.height // scatter_max_points)
            scatter_df = scatter_df.gather_every(step)
        x_vals = scatter_df[c].to_numpy()
        f_scatter = scatter_df[f_col].to_numpy()
        thr_x = [lvls.get(c) for lvls in thresholds_by_level.values()]
        xlim_zoom = compute_range(x_vals, thr_x)
        xlim_full = full_range(x_vals)

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Zoomed (threshold vicinity)", "Full landscape"],
            vertical_spacing=0.10,
        )
        for row, xlim in [(1, xlim_zoom), (2, xlim_full)]:
            fig.add_trace(
                go.Scattergl(
                    x=x_vals,
                    y=f_scatter,
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": f_scatter,
                        "colorscale": "Viridis",
                        "showscale": (row == 1),
                        "coloraxis": "coloraxis",
                        "opacity": 0.7,
                    },
                    hovertemplate=f"{c}=%{{x:.6g}}<br>{f_col}=%{{y:.6g}}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            add_threshold_lines_1d(fig, c, thresholds_by_level, row=row, col=1)
            fig.update_xaxes(title_text=c, range=list(xlim), row=row, col=1)
            fig.update_yaxes(title_text=f_col, row=row, col=1)

        f_in = f_scatter[np.isfinite(f_scatter)]
        vmin, vmax = np.percentile(f_in, clip_pct)
        fig.update_layout(
            title=f"{name} — ({c}, {f_col}). n={scatter_df.height} (of {df.height})",
            hovermode="closest",
            height=1100,
            template="plotly_white",
            coloraxis={
                "colorscale": "Viridis",
                "cmin": float(vmin),
                "cmax": float(vmax),
                "colorbar": {"title": f_col},
            },
        )
    else:
        pairs = list(combinations(c_cols, 2))
        n_pairs = len(pairs)
        n_cols = int(math.ceil(math.sqrt(n_pairs)))
        n_rows_per_sec = int(math.ceil(n_pairs / n_cols))
        total_rows = 2 * n_rows_per_sec

        f_in = f_vals[np.isfinite(f_vals)]
        vmin, vmax = np.percentile(f_in, clip_pct)

        subplot_titles = [""] * (total_rows * n_cols)
        for k, (a, b) in enumerate(pairs):
            r, c = divmod(k, n_cols)
            subplot_titles[r * n_cols + c] = f"zoomed: {a} vs {b}"
            subplot_titles[(n_rows_per_sec + r) * n_cols + c] = f"full: {a} vs {b}"

        fig = make_subplots(
            rows=total_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.10,
            vertical_spacing=0.10,
        )

        for k, (a, b) in enumerate(pairs):
            r_block, c_grid = divmod(k, n_cols)
            r_block, c_grid = r_block + 1, c_grid + 1
            x_all = df[a].to_numpy()
            y_all = df[b].to_numpy()
            thr_x = [lvls.get(a) for lvls in thresholds_by_level.values()]
            thr_y = [lvls.get(b) for lvls in thresholds_by_level.values()]
            xlim_zoom = compute_range(x_all, thr_x)
            ylim_zoom = compute_range(y_all, thr_y)
            xlim_full = full_range(x_all)
            ylim_full = full_range(y_all)

            for r_offset, xlim, ylim in [
                (0, xlim_zoom, ylim_zoom),
                (n_rows_per_sec, xlim_full, ylim_full),
            ]:
                row = r_block + r_offset
                in_window = (x_all >= xlim[0]) & (x_all <= xlim[1]) & (y_all >= ylim[0]) & (y_all <= ylim[1])
                fig.add_trace(
                    make_contour_trace(
                        x_all[in_window],
                        y_all[in_window],
                        f_vals[in_window],
                        xlim,
                        ylim,
                        nbins,
                        sigma,
                        f_col,
                    ),
                    row=row,
                    col=c_grid,
                )
                add_threshold_lines_2d(fig, row, c_grid, a, b, thresholds_by_level)
                fig.update_xaxes(title_text=a, row=row, col=c_grid, range=list(xlim))
                fig.update_yaxes(title_text=b, row=row, col=c_grid, range=list(ylim))

        fig.update_layout(
            title=f"{name} — bin-min({f_col}). Top half: zoomed (threshold vicinity), bottom half: full landscape. n={df.height}",
            height=450 * total_rows,
            coloraxis={
                "colorscale": "Viridis",
                "cmin": float(vmin),
                "cmax": float(vmax),
                "colorbar": {"title": f_col},
            },
            template="plotly_white",
        )

    out = output_dir / f"{name}_constraint_image.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiment_config.yaml")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="results/figures/threshold_picker")
    ap.add_argument("--problems", nargs="*", default=None, help="defaults to summary_problems from config")
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--smooth", type=float, default=1.2, help="Gaussian sigma in bins; 0 disables")
    ap.add_argument(
        "--clip-pct",
        nargs=2,
        type=float,
        default=[1.0, 99.0],
        help="Percentile clipping for the shared color scale (default 1, 99)",
    )
    ap.add_argument(
        "--scatter-max-points",
        type=int,
        default=25000,
        help="Subsample 1-constraint scatter plots above this size (0 disables)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = args.problems or cfg["summary_problems"]
    for pname in problems:
        problem_cfg = next((p for p in cfg["problems"] if p["name"] == pname), None)
        if problem_cfg is None:
            print(f"  skip: {pname} not in config")
            continue
        out = plot_problem(
            problem_cfg,
            Path(args.results_dir),
            output_dir,
            args.nbins,
            args.smooth,
            tuple(args.clip_pct),
            args.scatter_max_points if args.scatter_max_points > 0 else None,
        )
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
