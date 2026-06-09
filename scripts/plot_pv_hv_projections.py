"""Interactive plotly visualization of HV archives for pressure_vessel.

For each of the three modes, builds the cumulative non-dominated archive
across N_RUNS (50 by default) and plots pairwise projections in original
(unnormalized) coordinates. Toggle modes via the legend; hover for exact
values; switch panels via the dropdown menu.
"""

import sys
from pathlib import Path

import moocore
import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from desdeo.tools.non_dominated_sorting import non_dominated_merge

PROBLEM = "pressure_vessel"
CT_LEVEL = "high"
PSIZE = 36
N_RUNS_TOTAL = 500  # how many runs the data file contains
N_RUNS = 50  # how many to actually use for plotting (smaller = faster, smaller HTML)
N_GEN = 200
EPS_PERCENT = 0.01

OBJECTIVE_SYMBOL = "f_1"
CONSTRAINT_SYMBOLS = ["c_1", "c_2", "c_3", "c_4"]
F_COL = f"{OBJECTIVE_SYMBOL}_min"
DIM_COLS = [F_COL] + CONSTRAINT_SYMBOLS

OUT_PATH = Path(f"results/figures/pv_hv_archives_{CT_LEVEL}_psize{PSIZE}.html")


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

    f_opt = thr_doc["objective_optimum"]
    shadow_opt = thr_doc.get("objective_shadow_optima", {}).get(CT_LEVEL)
    thresholds = {c: float(thr_doc["levels"][CT_LEVEL].get(c, 0.0)) for c in CONSTRAINT_SYMBOLS}

    active_mask = hv_ctx["active_mask"]
    active_dim_names = [DIM_COLS[i] for i in range(len(DIM_COLS)) if active_mask[i]]
    ideal = hv_ctx["ideal"]
    nadir = hv_ctx["nadir"]
    range_full = nadir - ideal

    modes = ["baseline", "relaxed", "ranking"]
    palette = {"baseline": "#000000", "relaxed": "#0072B2", "ranking": "#E69F00"}

    # Collect per-mode archive points (in original coordinates) and HV stats
    mode_data: dict[str, dict] = {}
    for mode in modes:
        path = f"results/data/{PROBLEM}_{mode}_gen{N_GEN}_runs{N_RUNS_TOTAL}_ct{CT_LEVEL}_psize{PSIZE}.parquet"
        df = pl.read_parquet(path)
        run_ids = sorted(df["run"].unique().to_list())[:N_RUNS]

        all_pts_norm = []
        hvs = []
        for r in run_ids:
            sub = df.filter(pl.col("run") == r)
            arch, hv = cumulative_archive(sub, hv_ctx)
            if arch.shape[0] > 0:
                all_pts_norm.append(arch)
            hvs.append(hv)

        if all_pts_norm:
            all_pts_norm = np.vstack(all_pts_norm)
            all_pts_orig = all_pts_norm * range_full[active_mask] + ideal[active_mask]
        else:
            all_pts_orig = np.empty((0, len(active_dim_names)))

        mode_data[mode] = {
            "points": all_pts_orig,
            "mean_hv": np.mean(hvs),
            "mean_archive_size": np.mean([a.shape[0] for a in all_pts_norm])
            if isinstance(all_pts_norm, list)
            else all_pts_orig.shape[0] / N_RUNS,
        }

    # Build 2x3 grid: 3 columns (modes), 2 rows
    # Row 1: f_1 vs c_1, c_2, c_3 (selectable via dropdown if we want, but easier to just show 3 projections)
    # Actually: 3 rows (one per active constraint), 3 columns (modes)

    constraint_dims = [d for d in active_dim_names if d != F_COL]
    n_proj = len(constraint_dims)

    fig = make_subplots(
        rows=n_proj,
        cols=len(modes),
        subplot_titles=[
            f"{mode.capitalize()}<br><sub>HV={mode_data[mode]['mean_hv']:.4f}, archive≈{mode_data[mode]['mean_archive_size']:.0f}</sub>"
            if i == 0
            else ""
            for i in range(n_proj)
            for mode in modes
        ],
        shared_xaxes=True,
        shared_yaxes="rows",
        horizontal_spacing=0.05,
        vertical_spacing=0.07,
        x_title=F_COL,
    )

    for col_idx, mode in enumerate(modes, start=1):
        pts = mode_data[mode]["points"]
        x = pts[:, 0]  # f_1_min
        for row_idx, c_dim in enumerate(constraint_dims, start=1):
            c_idx = active_dim_names.index(c_dim)
            y = pts[:, c_idx]
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=3, color=palette[mode], opacity=0.25),
                    name=mode if row_idx == 1 else None,
                    legendgroup=mode,
                    showlegend=(row_idx == 1),
                    hovertemplate=f"{F_COL}: %{{x:.4f}}<br>{c_dim}: %{{y:.4f}}<extra>{mode}</extra>",
                ),
                row=row_idx,
                col=col_idx,
            )

            # Reference lines: c=0, c=threshold, f_opt, shadow_opt
            fig.add_hline(y=0, line=dict(color="red", width=1), row=row_idx, col=col_idx)
            thr = thresholds.get(c_dim, 0.0)
            if thr > 0:
                fig.add_hline(y=thr, line=dict(color="red", width=1, dash="dash"), row=row_idx, col=col_idx)
            fig.add_vline(x=f_opt, line=dict(color="green", width=1, dash="dot"), row=row_idx, col=col_idx)
            if shadow_opt is not None:
                fig.add_vline(x=shadow_opt, line=dict(color="purple", width=1, dash="dot"), row=row_idx, col=col_idx)

            if col_idx == 1:
                fig.update_yaxes(title_text=c_dim, row=row_idx, col=1)

    fig.update_layout(
        title=(
            f"Pressure vessel: cumulative non-dominated archives "
            f"({N_RUNS} runs, ct={CT_LEVEL}, psize={PSIZE})<br>"
            f"<sub>Red solid = strict feasibility (c=0); red dash = threshold; "
            f"green = f*; purple = shadow f*. Click legend to toggle modes.</sub>"
        ),
        height=300 * n_proj + 100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_PATH), include_plotlyjs="cdn")
    print(f"Written {OUT_PATH}")


if __name__ == "__main__":
    main()
