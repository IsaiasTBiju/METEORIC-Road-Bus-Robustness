"""Read-only check: how far are bus route lines from the nearest road segment?

Reads  data/processed/dubai_layers.gpkg (layer 'route_shapes')
       data/processed/road_segments.gpkg (layer 'segments')
Writes data/interim/bus_road_distance_by_route.csv
Changes no input file. Nothing is matched or dropped here.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

SHAPES = Path("data/processed/dubai_layers.gpkg")
SEGMENTS = Path("data/processed/road_segments.gpkg")
OUT = Path("data/interim/bus_road_distance_by_route.csv")
UTM = 32640
STEP_M = 50
BANDS = [0, 5, 10, 25, 50, 100, 250, np.inf]


def sample_points(shapes: gpd.GeoDataFrame, route_col: str) -> gpd.GeoDataFrame:
    rows = []
    for idx, row in shapes.iterrows():
        line = row.geometry
        if line is None or line.is_empty:
            continue
        parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
        for part in parts:
            n = max(int(part.length // STEP_M), 1)
            for d in np.linspace(0, part.length, n + 1):
                rows.append((idx, row[route_col], part.interpolate(d)))
    return gpd.GeoDataFrame(
        {"shape_row": [r[0] for r in rows], "route": [r[1] for r in rows]},
        geometry=[r[2] for r in rows], crs=shapes.crs,
    )


def main() -> None:
    shapes = gpd.read_file(SHAPES, layer="route_shapes").to_crs(UTM)
    print("route_shapes columns:", list(shapes.columns))
    route_col = next((c for c in ("route", "route_short_name") if c in shapes.columns), None)
    if route_col is None:
        raise SystemExit("No route column found (looked for 'route', 'route_short_name'). "
                         "Paste the columns line above to me.")
    segs = gpd.read_file(SEGMENTS, layer="segments").to_crs(UTM)[["segment_id", "is_private", "geometry"]]

    pts = sample_points(shapes, route_col)
    print(f"{len(shapes):,} route lines -> {len(pts):,} sample points (every {STEP_M} m)")

    near = gpd.sjoin_nearest(pts, segs, how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]

    d = near["dist_m"]
    band = pd.cut(d, BANDS, right=False)
    table = band.value_counts().sort_index().to_frame("points")
    table["share_%"] = (table["points"] / len(near) * 100).round(1)
    print("\nDistance from bus line to nearest road segment (m):")
    print(table.to_string())
    print(f"\nmedian {d.median():.1f} m   95th percentile {d.quantile(0.95):.1f} m   max {d.max():.0f} m")

    near["far"] = near["dist_m"] > 25
    near["on_private"] = near["is_private"].fillna(False).astype(bool)
    per_route = near.groupby("route").agg(
        points=("dist_m", "size"),
        median_m=("dist_m", "median"),
        share_far_over_25m=("far", "mean"),
        share_nearest_is_private=("on_private", "mean"),
    ).round(3).sort_values("share_far_over_25m", ascending=False)
    print("\nRoutes with the largest share of points more than 25 m from any road:")
    print(per_route.head(10).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_route.to_csv(OUT)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
