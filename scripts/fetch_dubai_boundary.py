"""Download the Dubai emirate outline from OpenStreetMap (via OSMnx) and report checks.

Network use: one geocoding request. Output: data/raw/boundary/dubai_boundary.geojson
"""
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "boundary" / "dubai_boundary.geojson"

gdf = ox.geocode_to_gdf("Dubai, United Arab Emirates")
print("matched name:", gdf.loc[0, "display_name"])
print("OSM class/type:", gdf.loc[0, "class"], "/", gdf.loc[0, "type"])

geom = gdf.geometry.iloc[0]
print("geometry type:", geom.geom_type, "| parts:", len(geom.geoms) if hasattr(geom, "geoms") else 1)
print("bounds (lon_min, lat_min, lon_max, lat_max):", [round(v, 3) for v in geom.bounds])

# Area in a metric projection (UTM zone 40N), not in degrees
area_km2 = gdf.to_crs(32640).geometry.iloc[0].area / 1e6
print(f"area: {area_km2:,.0f} km2")

gdf[["display_name", "geometry"]].to_file(OUT, driver="GeoJSON")
print("saved:", OUT)
