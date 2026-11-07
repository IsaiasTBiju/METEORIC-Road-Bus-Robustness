"""Turn the OSM road graph into junction-to-junction segments (one failure unit each).

Reads  data/interim/osm_roads.gpkg  (layer 'edges')
Writes data/processed/road_segments.gpkg  (layer 'segments', one row per segment)
       data/processed/road_edges.gpkg     (layer 'edges', every directed edge + its segment_id)
Private roads are kept and flagged (is_private); nothing is dropped here.
"""
from pathlib import Path

import geopandas as gpd

from meteoric.road_segments import assign_segments, segment_checks, segments_table

SRC = Path("data/interim/osm_roads.gpkg")
OUT_SEG = Path("data/processed/road_segments.gpkg")
OUT_EDGES = Path("data/processed/road_edges.gpkg")


def main() -> None:
    edges = gpd.read_file(SRC, layer="edges")
    print(f"Read {len(edges):,} directed edges")
    edges = assign_segments(edges)

    print("\nChecks")
    for name, value in segment_checks(edges).items():
        print(f"  {name}: {value:,}")

    segments = segments_table(edges)
    print(f"\nSegments: {len(segments):,}")
    print(f"  one-way: {int(segments['oneway'].sum()):,}   two-way: {int((~segments['oneway']).sum()):,}")
    print(f"  private: {int(segments['is_private'].sum()):,}   loops: {int(segments['is_loop'].sum()):,}")
    print(f"  segments < 1 m: {int((segments['length_m'] < 1).sum()):,}")

    OUT_SEG.parent.mkdir(parents=True, exist_ok=True)
    segments.to_file(OUT_SEG, layer="segments", driver="GPKG")
    edges.to_file(OUT_EDGES, layer="edges", driver="GPKG")
    print(f"\nSaved {OUT_SEG} and {OUT_EDGES}")


if __name__ == "__main__":
    main()
