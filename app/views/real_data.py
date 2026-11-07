"""Real-data page: Dubai bus stops and routes from the cleaned layers.

Reads only the prepared files in data/processed (public GTFS-derived layers and the
route-level demand table). No passenger transaction data is used here.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
GPKG = PROCESSED / "dubai_layers.gpkg"
DEMAND = PROCESSED / "route_demand.csv"

st.title("METEORIC: Dubai bus network")
st.caption(
    "Bus stops and route shapes from the RTA GTFS feed, clipped to the Dubai emirate outline. "
    "Road network and disruption models are not on this page yet."
)

if not (GPKG.exists() and DEMAND.exists()):
    st.warning("Prepared data not found in data/processed/. Run the scripts in scripts/ first "
               "(build_dubai_layers.py, build_route_demand.py).")
    st.stop()


@st.cache_data
def load():
    stops = gpd.read_file(GPKG, layer="stops")
    shapes = gpd.read_file(GPKG, layer="route_shapes")
    shapes["geometry"] = shapes.geometry.simplify(0.0002)  # about 20 m; for drawing only
    demand = pd.read_csv(DEMAND, dtype={"route_id": str})
    return stops, shapes, demand


stops, shapes, demand = load()
route_names = sorted(shapes["route_short_name"].unique(), key=lambda s: (len(s), s))

with st.sidebar:
    st.header("Map controls")
    selected = st.multiselect("Highlight routes", route_names)
    show_stops = st.checkbox("Show bus stops", value=True)

c1, c2, c3 = st.columns(3)
c1.metric("Bus routes", demand["route_short_name"].nunique())
c2.metric("Bus stops inside Dubai", f"{len(stops):,}")
c3.metric("Routes with demand data", int(demand["has_demand_row"].sum()))


def lines_geojson(route_set, inside):
    """Features for routes that are (inside=True) or are not (inside=False) in route_set."""
    features = []
    for row in shapes.itertuples():
        if (row.route_short_name in route_set) != inside:
            continue
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {"route": row.route_short_name},
        })
    return {"type": "FeatureCollection", "features": features}


# Width is fixed in pixels by clamping (min/max), so it cannot blow up at any zoom.
layers = [
    pdk.Layer(
        "GeoJsonLayer", lines_geojson(set(selected), inside=False), pickable=True,
        stroked=True, filled=False, line_joint_rounded=True,
        get_line_color=[31, 119, 180, 110], get_line_width=1,
        line_width_min_pixels=1, line_width_max_pixels=2,
    )
]
if selected:
    layers.append(pdk.Layer(
        "GeoJsonLayer", lines_geojson(set(selected), inside=True), pickable=True,
        stroked=True, filled=False, line_joint_rounded=True,
        get_line_color=[214, 39, 40, 240], get_line_width=1,
        line_width_min_pixels=4, line_width_max_pixels=5,
    ))
pts = pd.DataFrame({"lon": stops.geometry.x, "lat": stops.geometry.y})
if show_stops:
    layers.append(pdk.Layer(
        "ScatterplotLayer", pts, get_position="[lon, lat]", get_radius=30,
        radius_min_pixels=1.5, radius_max_pixels=3, get_fill_color=[60, 60, 60, 160], pickable=False,
    ))

# Start view fitted to the stop locations (instead of a hand-picked zoom)
view = pdk.data_utils.compute_view(pts[["lon", "lat"]].values.tolist(), view_proportion=0.95)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="light",
        tooltip={"text": "Route {properties.route}"},
    ),
    height=600,
)

st.subheader("Route demand")
st.caption(
    "Demand is the portal field TRIPS (number of passengers), averaged over Feb to Jul 2026 per route. "
    "It is route-level only, not passengers on individual road segments. "
    "'Share in Dubai' is the fraction of the route's length inside the Dubai outline."
)
table = demand[demand["route_short_name"].isin(selected)] if selected else demand.nlargest(15, "mean_all_months")
view = table[["route_short_name", "mean_all_months", "months_active", "share_in_dubai"]].rename(columns={
    "route_short_name": "Route", "mean_all_months": "Avg monthly passengers",
    "months_active": "Months with service (of 6)", "share_in_dubai": "Share in Dubai",
})
view["Avg monthly passengers"] = view["Avg monthly passengers"].round(0)
st.dataframe(view, hide_index=True)
if not selected:
    st.caption("Showing the 15 routes with the most passengers. Pick routes in the sidebar to see others.")
