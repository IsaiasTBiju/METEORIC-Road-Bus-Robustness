"""M0 baseline page: undirected road graph, no demand weighting, real Dubai data.

Pick a disruption (random or targeted), a size and a seed; see the road network and bus services
before and after, on a map, in tables, and export the results.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from meteoric.robustness import build_network, run_scenario  # noqa: E402

SEGMENTS = ROOT / "data/processed/road_segments.gpkg"
ROUTE_SEG = ROOT / "data/processed/route_segments.csv"
COVERAGE = ROOT / "data/processed/route_coverage.csv"
MODEL = "M0"


@st.cache_resource(show_spinner="Loading the road network (first time only)...")
def load_data():
    gdf = gpd.read_file(SEGMENTS, layer="segments")
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-5, mode="pointwise")
    for col in ("name", "highway"):
        gdf[col] = gdf[col].fillna("").astype(str) if col in gdf else ""
    rs = pd.read_csv(ROUTE_SEG, dtype={"route": str})
    net = build_network(pd.DataFrame(gdf.drop(columns="geometry")), rs, None)
    geo = gdf.set_index("segment_id")
    bus_ids = rs["segment_id"].unique()
    bus_geojson = json.loads(geo.loc[bus_ids, ["geometry"]].to_json())
    cov = pd.read_csv(COVERAGE, dtype={"route": str}) if COVERAGE.exists() else None
    return net, geo, rs, cov, bus_geojson


def run(net, geo, rs, cov, strategy, n, seed, allow_private):
    candidates = None if allow_private else ~geo["is_private"].to_numpy(dtype=bool)
    out = run_scenario(net, MODEL, strategy, n, seed, candidates)
    removed_ids = out.pop("removed_segment_ids")

    removed = geo.loc[removed_ids, ["name", "highway", "length_m", "is_private", "geometry"]].copy()
    uses = rs.groupby("segment_id").size()
    removed["routes_using"] = uses.reindex(removed.index).fillna(0).astype(int)
    removed = removed.reset_index().sort_values(["routes_using", "length_m"], ascending=False)

    hit = rs[rs["segment_id"].isin(set(removed_ids))]
    tbl = pd.concat([hit.groupby("route").size().rename("segments_lost"),
                     rs.groupby("route").size().rename("segments_on_route")], axis=1, join="inner")
    tbl["share_of_path_lost_pct"] = tbl["segments_lost"] / tbl["segments_on_route"] * 100
    if cov is not None:
        tbl = tbl.join(cov.set_index("route")[["coverage", "high_coverage"]])
    tbl = tbl.sort_values("segments_lost", ascending=False).reset_index().rename(columns={"index": "route"})
    affected_segments = rs[rs["route"].isin(set(tbl["route"]))]["segment_id"].unique()
    return {
        "metrics": out, "removed": removed, "routes": tbl, "affected_segments": affected_segments,
        "settings": {"strategy": strategy, "n": int(n), "seed": int(seed), "allow_private": bool(allow_private),
                     "pool": len(geo) if allow_private else int((~geo["is_private"]).sum())},
    }


def make_map(geo, bus_geojson, result):
    removed = result["removed"]
    affected = json.loads(geo.loc[result["affected_segments"], ["geometry"]].to_json())
    gone = json.loads(gpd.GeoDataFrame(
        removed[["segment_id", "name", "highway", "routes_using", "geometry"]], crs=4326).to_json())

    def layer(data, color, width, max_px, pickable=False):
        return pdk.Layer("GeoJsonLayer", data, get_line_color=color, get_line_width=width,
                         line_width_units="meters", line_width_min_pixels=1, line_width_max_pixels=max_px,
                         line_joint_rounded=True, line_cap_rounded=True, pickable=pickable, stroked=True, filled=False)

    layers = [layer(bus_geojson, [150, 150, 150, 110], 10, 2),
              layer(affected, [235, 104, 52, 200], 14, 3),
              layer(gone, [200, 20, 20, 255], 40, 6, pickable=True)]

    pts = []
    for g in removed.geometry:
        if g is not None and not g.is_empty:
            x, y = g.coords[0]
            pts.append([x, y])
    if len(pts) < 2:
        minx, miny, maxx, maxy = geo.loc[result["affected_segments"]].total_bounds if len(
            result["affected_segments"]) else geo.total_bounds
        pts = [[minx, miny], [maxx, maxy]]
    view = pdk.data_utils.compute_view(pts, view_proportion=0.95)
    return pdk.Deck(layers=layers, initial_view_state=view, map_style="light",
                    tooltip={"text": "Segment {properties.segment_id}\n{properties.name}\n{properties.highway}\n"
                                     "Bus routes using it: {properties.routes_using}"})


net, geo, rs, cov, bus_geojson = load_data()

st.title("M0 baseline: undirected, no demand weighting")
st.caption("Dubai road network and bus routes. M0 treats every road as two-way and every bus route as equally "
           "important. Directed (M1) and demand-aware (M2, M3) models come next.")

with st.sidebar:
    st.header("Scenario")
    st.selectbox("Model", ["M0 (baseline)"], help="M1 to M3 will appear here once they have a page.")
    strategy_label = st.radio("Disruption strategy", ["Random", "Targeted"],
                              help="Random: segments chosen at random (seed makes it repeatable). Targeted: the "
                                   "segments used by the most bus routes, removed first.")
    strategy = strategy_label.lower()
    allow_private = st.checkbox("Allow private roads to fail", value=False,
                                help="Off by default: private roads stay in the network but are never chosen to fail.")
    pool = len(geo) if allow_private else int((~geo["is_private"]).sum())
    amount_mode = st.radio("Amount", ["Number of segments", "Proportion of segments"], horizontal=True)
    if amount_mode == "Number of segments":
        n = st.number_input("Segments to remove", min_value=1, max_value=min(5000, pool), value=100, step=1)
    else:
        pct = st.slider("Proportion removed (%)", 0.01, 5.0, 0.1, 0.01)
        n = max(1, round(pct / 100 * pool))
        st.caption(f"= {n:,} of {pool:,} segments that may fail")
    seed = st.number_input("Seed (random only)", 0, 1_000_000, 42, disabled=(strategy == "targeted"))
    clicked = st.button("Run disruption", type="primary", width="stretch")

if clicked or "m0_result" not in st.session_state:
    with st.spinner("Running..."):
        st.session_state["m0_result"] = run(net, geo, rs, cov, strategy, n, seed, allow_private)
result = st.session_state["m0_result"]
m, s = result["metrics"], result["settings"]

st.subheader(f"{s['strategy'].capitalize()} removal of {s['n']:,} segment{'s' if s['n'] != 1 else ''}"
             + (f" (seed {s['seed']})" if s["strategy"] == "random" else ""))
c1, c2, c3 = st.columns(3)
c1.metric("Road connectivity", f"{m['road_share_after']:.2%}",
          f"{(m['road_share_after'] - m['road_share_before']) * 100:+.2f} points vs {m['road_share_before']:.2%} before",
          help="Share of road junctions still in the largest connected piece.")
c2.metric("Bus routes affected", f"{m['routes_affected']} of {m['routes_total']}", f"{m['routes_affected_share']:.1%} of routes",
          delta_color="off", delta_arrow="off")
c3.metric("Segments removed", f"{s['n']:,}", f"from {s['pool']:,} that may fail", delta_color="off",
          delta_arrow="off")

tab_map, tab_routes, tab_seg, tab_about = st.tabs(["Map", "Affected routes", "Removed segments", "About M0"])

with tab_map:
    st.pydeck_chart(make_map(geo, bus_geojson, result), width="stretch")
    st.caption("Red: removed segments. Orange: bus routes that lose at least one segment. Grey: all other road "
               "segments used by buses. Hover over a red segment for details.")

with tab_routes:
    st.write(f"{len(result['routes'])} routes lose at least one segment (no rerouting assumed).")
    st.dataframe(result["routes"], width="stretch", hide_index=True, column_config={
        "route": "Route", "segments_lost": "Segments lost", "segments_on_route": "Segments on route",
        "share_of_path_lost_pct": st.column_config.ProgressColumn("Share of path lost", format="%.1f%%", min_value=0, max_value=100),
        "coverage": st.column_config.NumberColumn("Match coverage", format="%.2f"),
        "high_coverage": "High coverage"})
    st.download_button("Download affected routes (CSV)", result["routes"].to_csv(index=False),
                       "m0_affected_routes.csv", "text/csv")

with tab_seg:
    show = result["removed"].drop(columns="geometry")
    st.dataframe(show, width="stretch", hide_index=True)
    c_a, c_b, c_c = st.columns(3)
    c_a.download_button("Removed segments (CSV)", show.to_csv(index=False), "m0_removed_segments.csv", "text/csv")
    c_b.download_button("Removed segments (GeoJSON)",
                        gpd.GeoDataFrame(result["removed"], crs=4326).to_json(), "m0_removed_segments.geojson",
                        "application/geo+json")
    summary = pd.DataFrame([{**s, **{k: v for k, v in m.items() if k not in ("model", "strategy", "seed", "n_removed")},
                             "model": MODEL}])
    c_c.download_button("Scenario summary (CSV)", summary.to_csv(index=False), "m0_scenario_summary.csv", "text/csv")

with tab_about:
    st.markdown("""
**What M0 assumes**
- The road network is **undirected**: every road can be used both ways, so one-way streets are ignored.
- The failure unit is a **segment**, one stretch of road between two junctions. Closing it closes it for good.
- **No rerouting**: a bus route that loses any segment of its path is counted as affected.
- **No demand weighting**: every route counts equally, whatever its passenger numbers.

**Measures**
- *Road connectivity*: the share of road junctions still in the largest connected piece, compared with before.
- *Bus routes affected*: the share of the 188 routes that lose at least one segment.

**Strategies**
- *Random*: segments drawn at random from those that may fail. The same seed gives the same segments.
- *Targeted*: segments used by the most bus routes are removed first. Ties are broken by segment order.

**Data notes**
- Roads come from OpenStreetMap; bus routes from the Dubai GTFS feed, matched onto road segments.
  Routes are matched with a coverage score (share of the route path matched); low-coverage routes use the segments
  we could match and may understate impact.
- Private roads stay in the network but are not chosen to fail unless you allow it.
""")