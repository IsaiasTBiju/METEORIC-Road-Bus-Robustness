"""Read-only: print a few real examples of unbridged breaks, with road names, for each cause group.

Reads  data/interim/bus_breaks.csv, data/processed/bus_route_segments.csv, data/processed/road_segments.gpkg
Prints 3 random examples per group (fixed seed, so the same ones each run). Changes nothing.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

BREAKS = Path("data/interim/bus_breaks.csv")
SEQ = Path("data/processed/bus_route_segments.csv")
SEGMENTS = Path("data/processed/road_segments.gpkg")
PER_GROUP = 3
BEFORE, AFTER = 2, 3
COLS = ["seq_order", "segment_id", "name", "highway", "oneway", "travel", "length_m", "from_node", "to_node", "note"]


def main() -> None:
    br = pd.read_csv(BREAKS)
    br = br[br["gap_m"].isna()].copy()
    seq = pd.read_csv(SEQ)
    segs = gpd.read_file(SEGMENTS, layer="segments", ignore_geometry=True).set_index("segment_id")

    groups = {
        "A: legal route exists the OTHER way round": br[br["reverse_route_exists"].astype(str) == "True"],
        "B: no route either way, ends within 250 m": br[(br["reverse_route_exists"].astype(str) != "True") & (br["straight_m"] <= 250)],
        "C: no route either way, ends over 250 m apart": br[(br["reverse_route_exists"].astype(str) != "True") & (br["straight_m"] > 250)],
    }
    for title, g in groups.items():
        print(f"\n{'=' * 90}\nGROUP {title}  ({len(g):,} breaks)\n{'=' * 90}")
        for row in g.sample(min(PER_GROUP, len(g)), random_state=1).itertuples(index=False):
            print(f"\nRoute {row.route}, line {row.shape_id}, break after step {row.seq_order}, "
                  f"ends {row.straight_m:.0f} m apart")
            w = seq[(seq["shape_id"] == row.shape_id)
                    & seq["seq_order"].between(row.seq_order - BEFORE, row.seq_order + AFTER)].copy()
            w = w.join(segs[["name", "highway", "oneway", "length_m", "from_node", "to_node"]], on="segment_id")
            w["length_m"] = w["length_m"].round(0)
            w["note"] = ["<-- break after this one" if s == row.seq_order else "" for s in w["seq_order"]]
            print(w[COLS].to_string(index=False))


if __name__ == "__main__":
    main()
