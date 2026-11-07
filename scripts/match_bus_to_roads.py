"""Match every bus route line onto road segments (direction-aware). Nothing is dropped.

Reads  data/processed/dubai_layers.gpkg (layer 'route_shapes')
       data/processed/road_segments.gpkg (layer 'segments')
Writes data/processed/bus_route_segments.csv  (ordered segments each bus line uses)
       data/processed/bus_match_quality.csv   (one row per bus line: shares ok / mismatch / unmatched, breaks)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from meteoric.bus_matching import build_sequences, match_points, sample_shape_points

SHAPES = Path("data/processed/dubai_layers.gpkg")
SEGMENTS = Path("data/processed/road_segments.gpkg")
OUT_SEQ = Path("data/processed/bus_route_segments.csv")
OUT_QUAL = Path("data/processed/bus_match_quality.csv")
UTM = 32640


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["viterbi", "nearest"], default="viterbi")
    method = parser.parse_args().method
    print(f"Matching method: {method}")
    shapes = gpd.read_file(SHAPES, layer="route_shapes").to_crs(UTM)
    segs = gpd.read_file(SEGMENTS, layer="segments").to_crs(UTM)

    points = sample_shape_points(shapes)
    print(f"{len(shapes):,} bus lines -> {len(points):,} sample points")

    matched = match_points(points, segs, method=method)
    print("\nPoint status:")
    print((matched["status"].value_counts(normalize=True) * 100).round(2).to_string())

    seq, qual = build_sequences(matched, pd.DataFrame(segs.drop(columns="geometry")))
    print(f"\nBus lines: {len(qual):,}   with 0 breaks: {(qual['breaks'] == 0).sum():,}")
    print(f"Lines with >=95% points ok: {(qual['share_ok'] >= 0.95).sum():,}")
    print(f"Lines with >5% direction_mismatch: {(qual['share_direction_mismatch'] > 0.05).sum():,}")
    print(f"Lines with >5% unmatched: {(qual['share_unmatched'] > 0.05).sum():,}")
    print("\nWorst 10 lines by share ok:")
    print(qual.sort_values("share_ok").head(10).round(3).to_string(index=False))

    OUT_SEQ.parent.mkdir(parents=True, exist_ok=True)
    seq.to_csv(OUT_SEQ, index=False)
    qual.to_csv(OUT_QUAL, index=False)
    print(f"\nSaved {OUT_SEQ} and {OUT_QUAL}")


if __name__ == "__main__":
    main()