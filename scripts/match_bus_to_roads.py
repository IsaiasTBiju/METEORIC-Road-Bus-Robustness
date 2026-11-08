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

from meteoric.bus_matching import (MIN_SHARE_OK, build_sequences, fill_gaps, match_points,
                                   sample_shape_points)

SHAPES = Path("data/processed/dubai_layers.gpkg")
SEGMENTS = Path("data/processed/road_segments.gpkg")
OUT_SEQ = Path("data/processed/bus_route_segments.csv")
OUT_QUAL = Path("data/processed/bus_match_quality.csv")
OUT_GAPS = Path("data/processed/bus_gaps.csv")
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

    segs_df = pd.DataFrame(segs.drop(columns="geometry"))
    seq, qual = build_sequences(matched, segs_df)
    qual = qual.rename(columns={"breaks": "breaks_before_fill"})
    filled, gaps = fill_gaps(seq, segs_df)

    for status in ("filled", "unbridged"):
        counts = gaps[gaps["status"] == status].groupby("shape_id").size()
        qual[f"n_{status}"] = qual["shape_id"].map(counts).fillna(0).astype(int)
    qual["complete"] = (qual["n_unbridged"] == 0) & (qual["share_ok"] >= MIN_SHARE_OK)

    print(f"\nGaps filled with a short legal route: {(gaps['status'] == 'filled').sum():,} "
          f"(inserted {int(gaps['n_inserted'].sum()):,} segments)")
    print(f"Gaps left unbridged: {(gaps['status'] == 'unbridged').sum():,}")
    print(f"\nBus lines: {len(qual):,}   complete: {int(qual['complete'].sum()):,}   "
          f"incomplete: {int((~qual['complete']).sum()):,}")
    print(f"  incomplete because of unbridged gaps: {int((qual['n_unbridged'] > 0).sum()):,}")
    print(f"  incomplete because under {MIN_SHARE_OK:.0%} of points matched: "
          f"{int((qual['share_ok'] < MIN_SHARE_OK).sum()):,}")

    per_route = qual.groupby("route")["complete"].agg(["sum", "size"])
    print(f"\nRoutes: {len(per_route):,}")
    print(f"  all lines complete: {int((per_route['sum'] == per_route['size']).sum()):,}")
    print(f"  some lines complete: {int(((per_route['sum'] > 0) & (per_route['sum'] < per_route['size'])).sum()):,}")
    print(f"  no line complete: {int((per_route['sum'] == 0).sum()):,}")
    print("\nWorst 10 lines by unbridged gaps:")
    print(qual.sort_values("n_unbridged", ascending=False).head(10)[
        ["shape_id", "route", "share_ok", "n_filled", "n_unbridged"]].round(3).to_string(index=False))

    seq = filled
    OUT_SEQ.parent.mkdir(parents=True, exist_ok=True)
    seq.to_csv(OUT_SEQ, index=False)
    qual.to_csv(OUT_QUAL, index=False)
    gaps.to_csv(OUT_GAPS, index=False)
    print(f"\nSaved {OUT_SEQ}, {OUT_QUAL} and {OUT_GAPS}")


if __name__ == "__main__":
    main()