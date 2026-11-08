"""Run the disruption engine on the real Dubai network for all four models.

Reads  data/processed/road_segments.gpkg, route_segments.csv, route_demand.csv, route_coverage.csv
Writes results/scenarios.csv (git-ignored)

Example: python scripts/run_scenarios.py --n 1000 --seed 42
Random scenarios remove the SAME segments for all four models; targeted ones rank by route count
(M0/M1) or passenger demand (M2/M3).
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from meteoric.robustness import MODELS, build_network, run_scenario

SEGMENTS = Path("data/processed/road_segments.gpkg")
ROUTE_SEG = Path("data/processed/route_segments.csv")
DEMAND = Path("data/processed/route_demand.csv")
OUT = Path("results/scenarios.csv")


def load_demand() -> pd.Series:
    dem = pd.read_csv(DEMAND, dtype={"route_short_name": str, "route": str})
    rcol = next((c for c in ("route_short_name", "route") if c in dem.columns), None)
    if rcol is None or "mean_all_months" not in dem.columns:
        raise SystemExit(f"Cannot find route / mean_all_months columns in {DEMAND}: {list(dem.columns)}")
    return dem.set_index(rcol)["mean_all_months"].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="number of segments to remove")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    segs = gpd.read_file(SEGMENTS, layer="segments", ignore_geometry=True)
    rs = pd.read_csv(ROUTE_SEG, dtype={"route": str})
    net = build_network(segs, rs, load_demand())
    print(f"{len(net.segment_ids):,} segments, {net.n_nodes:,} junctions, {len(net.route_names)} routes "
          f"({net.routes_without_demand} without a demand figure, counted as 0 demand)")

    print("\nSanity check: direction labels by road type (rows = segment type, cols = travel_dirs)")
    check = rs.merge(segs[["segment_id", "oneway", "is_private"]], on="segment_id")
    check["road"] = np.where(check["oneway"], "one-way", "two-way")
    print(pd.crosstab(check["road"], check["travel_dirs"]).to_string())
    used = check.drop_duplicates("segment_id")
    print(f"\nPrivate segments among those used by buses: {int(used['is_private'].sum()):,} of {len(used):,} "
          f"({used['is_private'].mean():.1%}); across the whole network: {segs['is_private'].mean():.1%}")

    rows = []
    for strategy in ("random", "targeted"):
        for model in MODELS:
            out = run_scenario(net, model, strategy, args.n, args.seed)
            out.pop("removed_segment_ids")
            rows.append(out)
    res = pd.DataFrame(rows)
    show = res[["strategy", "model", "road_share_before", "road_share_after", "road_retained",
                "routes_affected", "routes_affected_share", "demand_affected_share"]]
    print(f"\nRemoving {args.n:,} segments (seed {args.seed}):")
    print(show.round(4).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
