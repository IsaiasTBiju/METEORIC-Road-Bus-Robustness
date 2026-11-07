"""Build Dubai-only bus stop and route-shape layers from the public RTA GTFS.

Inputs : data/interim/gtfs/feed (unpacked GTFS), data/raw/boundary/dubai_boundary.geojson
Outputs: data/processed/dubai_layers.gpkg (layers: stops, route_shapes)
         data/processed/route_summary.csv
Nothing is dropped silently: every exclusion is counted and printed.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[1]
METRIC_CRS = 32640  # UTM zone 40N, metres; used only for lengths
BUS = "3"           # GTFS route_type for bus


def read(feed: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(feed / name, dtype=str, encoding="utf-8-sig", **kw)


def main(feed: Path, boundary_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    boundary_geom = boundary.geometry.union_all()

    routes = read(feed, "routes.txt")
    trips = read(feed, "trips.txt")
    stops = read(feed, "stops.txt")
    shapes = read(feed, "shapes.txt")
    stop_times = read(feed, "stop_times.txt", usecols=["trip_id", "stop_id"])

    bus_routes = routes[routes["route_type"] == BUS]
    bus_trips = trips[trips["route_id"].isin(bus_routes["route_id"])]
    print(f"bus routes: {len(bus_routes)} | bus trips: {len(bus_trips):,}")

    # ---- stops: only those served by at least one bus trip ----
    bus_stop_ids = set(stop_times.loc[stop_times["trip_id"].isin(bus_trips["trip_id"]), "stop_id"])
    st = stops[stops["stop_id"].isin(bus_stop_ids)].copy()
    print(f"stops in feed: {len(stops):,} | served by a bus trip: {len(st):,} "
          f"| excluded (metro/tram/ferry only): {len(stops) - len(st):,}")
    st["stop_lat"] = st["stop_lat"].astype(float)
    st["stop_lon"] = st["stop_lon"].astype(float)
    st_gdf = gpd.GeoDataFrame(
        st, geometry=gpd.points_from_xy(st["stop_lon"], st["stop_lat"]), crs=4326)
    st_gdf["in_dubai"] = st_gdf.within(boundary_geom)
    print(f"bus stops inside Dubai boundary: {int(st_gdf['in_dubai'].sum()):,} "
          f"| outside: {int((~st_gdf['in_dubai']).sum()):,}")

    # sanity check by name: stops named for other emirates should be outside
    for place in ["Sharjah", "Ajman", "Abu Dhabi", "Umm Al Quwain", "Fujairah", "Hatta"]:
        m = st_gdf[st_gdf["stop_name"].str.contains(place, case=False, na=False)]
        if len(m):
            print(f"  named '{place}': {len(m)} bus stops, {int(m['in_dubai'].sum())} inside boundary")

    # ---- route shapes: one line per shape used by a bus trip ----
    shapes["lat"] = shapes["shape_pt_lat"].astype(float)
    shapes["lon"] = shapes["shape_pt_lon"].astype(float)
    shapes["seq"] = shapes["shape_pt_sequence"].astype(int)
    used = (bus_trips.merge(bus_routes[["route_id", "route_short_name"]], on="route_id")
            .groupby(["shape_id", "route_id", "route_short_name"])
            .agg(n_trips=("trip_id", "size"), direction_ids=("direction_id", lambda s: ",".join(sorted(set(s)))))
            .reset_index())
    lines = {}
    for sid, g in shapes[shapes["shape_id"].isin(used["shape_id"])].sort_values("seq").groupby("shape_id"):
        if len(g) >= 2:
            lines[sid] = LineString(zip(g["lon"], g["lat"]))
    used = used[used["shape_id"].isin(lines)].copy()
    print(f"bus shapes with a usable line: {len(used)} (of {used['shape_id'].nunique()} used by bus trips)")
    sh = gpd.GeoDataFrame(used, geometry=[lines[s] for s in used["shape_id"]], crs=4326)

    sh["length_full_m"] = sh.to_crs(METRIC_CRS).length
    clipped = sh.geometry.intersection(boundary_geom)
    sh["geometry"] = clipped
    sh["length_dubai_m"] = sh.to_crs(METRIC_CRS).length
    sh["share_in_dubai"] = (sh["length_dubai_m"] / sh["length_full_m"]).round(4)
    broken = sh["length_dubai_m"].isna() | sh.geometry.isna()
    if broken.any():
        raise SystemExit(f"Clipping produced {int(broken.sum())} broken shapes; fix before continuing")
    outside_only = sh["length_dubai_m"] <= 0
    print(f"shapes entirely outside Dubai (excluded): {int(outside_only.sum())}")
    partial = (~outside_only) & (sh["share_in_dubai"] < 0.95)
    print(f"shapes partly outside (kept, clipped; 'intercity-like'): {int(partial.sum())}")
    sh = sh[~outside_only].copy()

    # ---- route summary (one row per bus route, for the audit trail) ----
    rs = (sh.groupby(["route_id", "route_short_name"])
          .agg(n_shapes=("shape_id", "size"), n_trips=("n_trips", "sum"),
               length_full_km=("length_full_m", lambda s: round(s.sum() / 1000, 1)),
               length_dubai_km=("length_dubai_m", lambda s: round(s.sum() / 1000, 1)))
          .reset_index())
    rs["share_in_dubai"] = (rs["length_dubai_km"] / rs["length_full_km"]).round(3)
    dropped = sorted(set(bus_routes["route_short_name"]) - set(rs["route_short_name"]))
    print(f"bus routes kept: {len(rs)} of {len(bus_routes)} | dropped (no length in Dubai): {len(dropped)}")
    print("dropped routes:", dropped)

    gpkg = out_dir / "dubai_layers.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    st_gdf[st_gdf["in_dubai"]].drop(columns=["in_dubai"]).to_file(gpkg, layer="stops", driver="GPKG")
    sh.to_file(gpkg, layer="route_shapes", driver="GPKG")
    rs.to_csv(out_dir / "route_summary.csv", index=False)
    print("\nwrote:", gpkg, "and route_summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feed", type=Path, default=ROOT / "data/interim/gtfs/feed")
    p.add_argument("--boundary", type=Path, default=ROOT / "data/raw/boundary/dubai_boundary.geojson")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed")
    a = p.parse_args()
    main(a.feed, a.boundary, a.out)
