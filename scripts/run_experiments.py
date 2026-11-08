"""Many seeds x several removal sizes, for all four models. Summarises and draws the main chart.

Reads  data/processed/road_segments.gpkg, route_segments.csv, route_demand.csv
Writes results/experiments.csv (every run), results/experiments_summary.csv, results/experiments.png
Failures are drawn from public roads only (private roads stay in the graph) unless
--include-private-failures is given. Random scenarios remove the same segments for every model.

Example: python scripts/run_experiments.py --seeds 100
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullLocator  # noqa: E402

from meteoric.robustness import build_network, run_all_models  # noqa: E402
from run_scenarios import DEMAND, ROUTE_SEG, SEGMENTS, load_demand  # noqa: E402

OUT_DIR = Path("results")
SIZES = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

# Validated palette (light): blue = undirected, orange = directed, aqua / violet = targeted rankings.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8985"


def band(df, col):
    g = df.groupby("n_removed")[col]
    return g.mean(), g.quantile(0.05), g.quantile(0.95)


def style(ax, ylabel, title):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Road segments removed", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def draw(rand, targ, path, n_seeds):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160, facecolor=SURFACE)
    # Panel 1: road connectivity retained, undirected (M0/M2) vs directed (M1/M3), random removal
    for model, color, label in (("M0", BLUE, "Undirected (M0, M2)"), ("M1", ORANGE, "Directed (M1, M3)")):
        m, lo, hi = band(rand[rand["model"] == model], "road_retained")
        a1.fill_between(m.index, lo, hi, color=color, alpha=0.18, linewidth=0)
        a1.plot(m.index, m, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    style(a1, "Largest connected piece kept", f"Road connectivity: random removal ({n_seeds} seeds, band = 5th-95th pct)")
    a1.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    # Panel 2: demand affected, random vs the two targeted rankings
    m, lo, hi = band(rand[rand["model"] == "M0"], "demand_affected_share")
    a2.fill_between(m.index, lo, hi, color=GREY, alpha=0.2, linewidth=0)
    a2.plot(m.index, m, color=GREY, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="Random")
    for model, color, label, dy in (("M0", AQUA, "Targeted: most routes (M0, M1)", 8),
                                    ("M2", VIOLET, "Targeted: most demand (M2, M3)", -14)):
        t = targ[targ["model"] == model].set_index("n_removed")["demand_affected_share"]
        a2.plot(t.index, t, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    style(a2, "Passenger demand on affected routes", "Bus demand affected (no rerouting)")
    a2.set_ylim(0, 1.02)
    a2.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--include-private-failures", action="store_true")
    args = parser.parse_args()

    segs = gpd.read_file(SEGMENTS, layer="segments", ignore_geometry=True)
    rs = pd.read_csv(ROUTE_SEG, dtype={"route": str})
    net = build_network(segs, rs, load_demand())
    cand = None if args.include_private_failures else ~segs["is_private"].to_numpy(dtype=bool)
    sizes = [n for n in SIZES if n <= (len(segs) if cand is None else int(cand.sum()))]
    print(f"{len(net.segment_ids):,} segments; failures drawn from "
          f"{len(segs) if cand is None else int(cand.sum()):,}; sizes {sizes}; {args.seeds} seeds")

    rows = []
    for n in sizes:
        for seed in range(args.seeds):
            rows += run_all_models(net, "random", n, seed, cand)
        rows += run_all_models(net, "targeted", n, 0, cand)
        print(f"  n={n:,} done", flush=True)
    res = pd.DataFrame(rows).drop(columns="removed_segment_ids")
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / "experiments.csv", index=False)

    rand, targ = res[res["strategy"] == "random"], res[res["strategy"] == "targeted"]

    m0 = rand[rand["model"] == "M0"].set_index(["n_removed", "seed"])["road_retained"]
    m1 = rand[rand["model"] == "M1"].set_index(["n_removed", "seed"])["road_retained"]
    cmp = pd.DataFrame({
        "M0_retained": m0.groupby("n_removed").mean(),
        "M1_retained": m1.groupby("n_removed").mean(),
        "loss_ratio_M1_over_M0": (1 - m1).groupby("n_removed").mean() / (1 - m0).groupby("n_removed").mean(),
        "share_seeds_M1_lower": (m1 < m0).groupby("n_removed").mean(),
    })
    print("\nRandom removal - road connectivity, undirected (M0) vs directed (M1):")
    print(cmp.round(4).to_string())

    bus = rand[rand["model"] == "M0"].groupby("n_removed").agg(
        routes_mean=("routes_affected_share", "mean"), demand_mean=("demand_affected_share", "mean"),
        demand_p5=("demand_affected_share", lambda s: s.quantile(0.05)),
        demand_p95=("demand_affected_share", lambda s: s.quantile(0.95)))
    print("\nRandom removal - bus measures (identical for all models):")
    print(bus.round(4).to_string())

    print("\nTargeted removal (deterministic):")
    print(targ.pivot(index="n_removed", columns="model",
                     values=["road_retained", "routes_affected_share", "demand_affected_share"]).round(4).to_string())

    summary = pd.concat({"connectivity_compare": cmp, "bus_random": bus}, axis=1)
    summary.to_csv(OUT_DIR / "experiments_summary.csv")
    draw(rand, targ, OUT_DIR / "experiments.png", args.seeds)
    print(f"\nSaved results/experiments.csv, experiments_summary.csv and experiments.png")


if __name__ == "__main__":
    main()