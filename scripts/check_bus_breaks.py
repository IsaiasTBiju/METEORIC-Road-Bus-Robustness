"""Read-only diagnostic: for each 'break' in a bus line's matched segment list, how far is the gap?

A break = two consecutive matched segments where the bus cannot go straight from the end of one
to the start of the next. For each break we look for the shortest legal (direction-respecting)
route along the road network between them, up to CUTOFF_M metres.

Reads  data/processed/bus_route_segments.csv, data/processed/road_segments.gpkg
Writes data/interim/bus_breaks.csv. Changes nothing else.
"""
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

from meteoric.graph_build import build_directed

SEQ = Path("data/processed/bus_route_segments.csv")
SEGMENTS = Path("data/processed/road_segments.gpkg")
OUT = Path("data/interim/bus_breaks.csv")
CUTOFF_M = 500
BANDS = [-0.001, 0.001, 50, 100, 250, 500]
LABELS = ["0 m (joined)", "up to 50 m", "50-100 m", "100-250 m", "250-500 m"]


def main() -> None:
    segs_geo = gpd.read_file(SEGMENTS, layer="segments").to_crs(32640)
    segs = pd.DataFrame(segs_geo.drop(columns="geometry"))
    ends = {sid: (g.coords[0], g.coords[-1]) for sid, g in zip(segs_geo["segment_id"], segs_geo.geometry)}
    seq = pd.read_csv(SEQ)
    info = segs.set_index("segment_id")[["from_node", "to_node", "length_m"]]
    graph = build_directed(segs[["segment_id", "from_node", "to_node", "oneway", "length_m"]])

    seq = seq.sort_values(["shape_id", "seq_order"]).reset_index(drop=True)
    seq["next_segment"] = seq.groupby("shape_id")["segment_id"].shift(-1)
    seq["next_travel"] = seq.groupby("shape_id")["travel"].shift(-1)
    seq["after_next"] = seq.groupby("shape_id")["segment_id"].shift(-2)
    pairs = seq.dropna(subset=["next_segment"]).copy()
    a, b = info.loc[pairs["segment_id"]], info.loc[pairs["next_segment"]]
    pairs["exit_node"] = [t if tr == "fwd" else f for f, t, tr in zip(a["from_node"], a["to_node"], pairs["travel"])]
    pairs["entry_node"] = [f if tr == "fwd" else t for f, t, tr in zip(b["from_node"], b["to_node"], pairs["next_travel"])]
    print(f"{len(seq):,} matched segment steps, {len(pairs):,} consecutive pairs")

    dist_cache = {}
    gaps = []
    for exit_node, entry_node in zip(pairs["exit_node"], pairs["entry_node"]):
        if exit_node not in dist_cache:
            dist_cache[exit_node] = nx.single_source_dijkstra_path_length(
                graph, exit_node, cutoff=CUTOFF_M, weight="length_m")
        gaps.append(dist_cache[exit_node].get(entry_node))
    pairs["gap_m"] = gaps

    breaks = pairs[pairs["gap_m"].isna() | (pairs["gap_m"] > 0.001)].copy()
    print(f"Pairs joined directly (gap 0): {len(pairs) - len(breaks):,}")
    print(f"Pairs with a break: {len(breaks):,} ({len(breaks) / len(pairs) * 100:.1f}%)")

    found = breaks.dropna(subset=["gap_m"])
    print(f"\nBreaks bridged by a legal route of up to {CUTOFF_M} m: {len(found):,}")
    print(pd.cut(found["gap_m"], BANDS, labels=LABELS).value_counts().reindex(LABELS).to_string())
    print(f"Breaks with NO legal route within {CUTOFF_M} m: {breaks['gap_m'].isna().sum():,}")

    unbridged = breaks[breaks["gap_m"].isna()].copy()
    print(f"\n--- Why are {len(unbridged):,} breaks unbridged? ---")

    def point(segment_id, travel, leaving):
        first, last = ends[segment_id]
        return last if (travel == "fwd") == leaving else first

    exit_xy = [point(sg, tr, True) for sg, tr in zip(unbridged["segment_id"], unbridged["travel"])]
    entry_xy = [point(sg, tr, False) for sg, tr in zip(unbridged["next_segment"], unbridged["next_travel"])]
    unbridged["straight_m"] = [((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
                               for (x0, y0), (x1, y1) in zip(exit_xy, entry_xy)]
    bands = [-1, 50, 250, 1000, 1e9]
    names = ["ends within 50 m", "50-250 m apart", "250-1000 m apart", "over 1000 m apart"]
    print("Straight-line distance between the two ends:")
    print(pd.cut(unbridged["straight_m"], bands, labels=names).value_counts().reindex(names).to_string())

    rev_cache = {}
    reverse_ok = []
    for exit_node, entry_node in zip(unbridged["exit_node"], unbridged["entry_node"]):
        if entry_node not in rev_cache:
            rev_cache[entry_node] = nx.single_source_dijkstra_path_length(
                graph, entry_node, cutoff=CUTOFF_M, weight="length_m")
        reverse_ok.append(exit_node in rev_cache[entry_node])
    unbridged["reverse_route_exists"] = reverse_ok
    print(f"\nA legal route exists the OTHER way round (order/direction problem?): {sum(reverse_ok):,}")
    back = (unbridged["after_next"] == unbridged["segment_id"]).sum()
    print(f"Line returns to the segment it just left (back-and-forth jitter?): {back:,}")

    per_line = breaks.groupby("shape_id").size()
    print(f"\nLines with no breaks at all: {seq['shape_id'].nunique() - len(per_line):,} of {seq['shape_id'].nunique():,}")
    print("Lines with the most unbridged breaks:")
    unb = breaks[breaks["gap_m"].isna()].groupby(["shape_id", "route"]).size().sort_values(ascending=False)
    print(unb.head(10).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    breaks = breaks.merge(unbridged[["shape_id", "seq_order", "straight_m", "reverse_route_exists"]],
                          on=["shape_id", "seq_order"], how="left")
    breaks[["shape_id", "route", "seq_order", "segment_id", "next_segment", "gap_m",
            "straight_m", "reverse_route_exists"]].to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()