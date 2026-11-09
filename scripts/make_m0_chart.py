"""M0-only chart for the feasibility video / proposal, from results/experiments.csv.

Run scripts/run_experiments.py first (it writes results/experiments.csv).
Writes results/m0_chart.png and prints the numbers behind it.
M0 = undirected road graph, no demand weighting, so the bus measure is the share of routes affected.
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from run_experiments import BLUE, GREY, INK, MUTED, SURFACE, band, style  # noqa: E402

SRC = Path("results/experiments.csv")
OUT = Path("results/m0_chart.png")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run: PYTHONPATH=src python scripts/run_experiments.py --seeds 100")
    res = pd.read_csv(SRC)
    m0 = res[res["model"] == "M0"]
    rand, targ = m0[m0["strategy"] == "random"], m0[m0["strategy"] == "targeted"].set_index("n_removed")
    n_seeds = rand["seed"].nunique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160, facecolor=SURFACE)
    panels = ((axes[0], "road_retained", "Road connectivity kept", "Road connectivity", 1),
              (axes[1], "routes_affected_share", "Bus routes affected", "Bus service: routes affected (no rerouting)", 0))
    for ax, col, ylabel, title, decimals in panels:
        m, lo, hi = band(rand, col)
        ax.fill_between(m.index, lo, hi, color=GREY, alpha=0.2, linewidth=0)
        ax.plot(m.index, m, color=GREY, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"Random ({n_seeds} seeds, band = 5th-95th pct)")
        ax.plot(targ.index, targ[col], color=BLUE, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="Targeted (segments on most bus routes)")
        style(ax, ylabel, title, decimals=decimals)
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.suptitle("M0 baseline: undirected road graph, no demand weighting", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)

    table = pd.DataFrame({
        "random_connectivity": rand.groupby("n_removed")["road_retained"].mean(),
        "targeted_connectivity": targ["road_retained"],
        "random_routes_affected": rand.groupby("n_removed")["routes_affected_share"].mean(),
        "targeted_routes_affected": targ["routes_affected_share"],
    })
    print(table.round(4).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
