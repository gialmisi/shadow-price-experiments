"""Schematic of the relaxation-gain hypervolume used in the paper (2D, one constraint).

This is a hand-built illustrative figure (no experiment data) whose only purpose is to
communicate, for the single-constraint case, *what* hypervolume we compute and *why*:

  * The horizontal axis is a single constraint value ``c`` (lower is better; ``c <= 0`` is
    strictly feasible, ``c <= t`` is threshold-feasible).
  * The vertical axis is the objective ``f`` (lower is better).
  * As we relax the constraint from its original boundary ``c = 0`` up to the threshold
    ``c = t``, the best attainable objective improves from ``f*_strict`` to ``f*_relaxed``.
    The non-dominated trade-off between "how much we relax" and "how much objective we
    gain" is the *shadow front*.
  * The hypervolume reference box spans ``[f*_relaxed, f*_strict]`` on the objective axis
    and ``[0, t]`` on the constraint axis. The reference point is the nadir corner
    ``(t, f*_strict)`` expanded outward by a small ``epsilon`` on each axis, so that
    solutions lying exactly on the nadir boundary (the strict and threshold optima) still
    contribute volume. The shaded area is the hypervolume dominated by the shadow front
    w.r.t. that reference point: it measures the quality of the shadow front, and
    therefore how well an algorithm has mapped out the relaxation trade-off.

The vertical extent ``f*_strict - f*_relaxed`` is exactly the (best attainable) shadow
price: the objective gain from relaxing the constraint up to the threshold.

The epsilon is exaggerated here for visibility; in the experiments it is a small fraction
of each axis range.

Example:
    uv run python scripts/plot_hv_schematic.py
"""

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="results/figures/hv_schematic.pdf")
    args = ap.parse_args()

    # Illustrative values.
    t = 1.0  # relaxation threshold on the constraint
    f_strict = 10.0  # best objective while strictly feasible (c <= 0)
    f_relaxed = 4.0  # best objective when relaxing up to the threshold (c <= t)

    # Epsilon expansion of the nadir (exaggerated for visibility).
    eps_c = 0.10 * t
    eps_f = 0.10 * (f_strict - f_relaxed)
    ref = (t + eps_c, f_strict + eps_f)  # reference point = nadir + epsilon

    # A convex, decreasing shadow front: relaxing more (larger c) buys a better (smaller) f.
    front_c = np.array([0.0, 0.16, 0.36, 0.60, 0.84, 1.00])
    front_f = np.array([f_strict, 8.1, 6.7, 5.5, 4.6, f_relaxed])

    # Dominated region (the hypervolume): for each c up to the reference, the objective spans
    # from the front's step value up to the reference objective.
    xs = np.linspace(0.0, ref[0], 600)
    idx = np.clip(np.searchsorted(front_c, xs, side="right") - 1, 0, len(front_c) - 1)
    f_step = front_f[idx]

    fig, ax = plt.subplots(figsize=(7.6, 6.2))

    # Hypervolume area.
    ax.fill_between(xs, f_step, ref[1], color="#4C72B0", alpha=0.25, zorder=1, label="hypervolume (dominated region)")

    # Reference box outline (nadir box, before epsilon expansion).
    ax.plot(
        [0, t, t, 0, 0], [f_relaxed, f_relaxed, f_strict, f_strict, f_relaxed], color="grey", lw=1.0, ls="-", zorder=2
    )

    # Shadow front: staircase + markers.
    step_c = np.repeat(front_c, 2)[1:]
    step_f = np.repeat(front_f, 2)[:-1]
    ax.plot(step_c, step_f, color="#4C72B0", lw=1.6, zorder=3)
    ax.plot(
        front_c, front_f, "o", color="#4C72B0", markeredgecolor="black", markersize=8, zorder=4, label="shadow front"
    )

    # Nadir corner and the epsilon-expanded reference point.
    ax.plot(t, f_strict, marker="o", markerfacecolor="white", markeredgecolor="grey", markersize=7, zorder=5)
    ax.annotate("nadir", (t, f_strict), xytext=(6, -12), textcoords="offset points", fontsize=8, color="grey")
    ax.plot(*ref, marker="s", color="#C44E52", markeredgecolor="black", markersize=11, zorder=6)
    ax.annotate(
        "reference point\n= nadir $+\\,\\varepsilon$",
        ref,
        xytext=(10, 2),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        color="#C44E52",
    )

    # Epsilon gaps (nadir -> reference) on each axis.
    ax.annotate(
        "", xy=(ref[0], f_strict), xytext=(t, f_strict), arrowprops={"arrowstyle": "<->", "color": "#C44E52", "lw": 1.1}
    )
    ax.text(t + eps_c / 2, f_strict - 0.28, "$\\varepsilon$", color="#C44E52", ha="center", va="top", fontsize=10)
    ax.annotate(
        "", xy=(t, ref[1]), xytext=(t, f_strict), arrowprops={"arrowstyle": "<->", "color": "#C44E52", "lw": 1.1}
    )
    ax.text(t + 0.02, f_strict + eps_f / 2, "$\\varepsilon$", color="#C44E52", ha="left", va="center", fontsize=10)

    # Strict and threshold optima (the two ends of the front).
    ax.annotate(
        "strict optimum",
        (0.0, f_strict),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=9,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
    )
    ax.annotate(
        "threshold optimum",
        (t, f_relaxed),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
    )

    # Guide lines.
    ax.axvline(0.0, color="black", ls="--", lw=1.0)
    ax.axvline(t, color="red", ls=":", lw=1.2)
    ax.axhline(f_strict, color="grey", ls=":", lw=0.8)
    ax.axhline(f_relaxed, color="grey", ls=":", lw=0.8)

    # Shadow-price bracket on the objective axis (gain from relaxation).
    x_br = -0.26
    ax.annotate(
        "",
        xy=(x_br, f_relaxed),
        xytext=(x_br, f_strict),
        arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.4},
        annotation_clip=False,
    )
    ax.text(
        x_br - 0.07,
        0.5 * (f_strict + f_relaxed),
        "shadow price $=f^{*}_{\\mathrm{strict}}-f^{*}_{\\mathrm{relaxed}}$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )

    # "hypervolume" label inside the shaded area.
    ax.text(
        0.66 * t,
        0.5 * (f_strict + front_f[3]),
        "hypervolume",
        fontsize=12,
        color="#1f3b6e",
        ha="center",
        va="center",
        style="italic",
    )

    # Axis cosmetics.
    ax.set_xlim(-0.5, ref[0] + 0.42)
    ax.set_ylim(f_relaxed - 1.5, ref[1] + 0.8)
    ax.set_xlabel("constraint value $c$  (lower is better)")
    ax.set_ylabel("objective $f$  (lower is better)")
    ax.set_xticks([0.0, t])
    ax.set_xticklabels(["$0$\n(strict feasibility)", "$t$\n(threshold)"])
    ax.set_yticks([f_relaxed, f_strict])
    ax.set_yticklabels(["$f^{*}_{\\mathrm{relaxed}}$", "$f^{*}_{\\mathrm{strict}}$"])
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title("Relaxation-gain hypervolume (single-constraint illustration)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
