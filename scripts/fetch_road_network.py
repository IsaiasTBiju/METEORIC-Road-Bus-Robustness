"""Download the drivable road network inside the Dubai outline (OSMnx) and save it.

Network use: Overpass requests through OSMnx (cached in data/raw/osm_cache, which Git ignores).
Output     : data/interim/osm_roads.gpkg  (layers: nodes, edges) + a printed summary.

Choices (all explained in the README/EDA notes):
- network_type="drive": roads cars can use (bus-only lanes are not captured; a known limit).
- simplify=True: graph nodes only at junctions and dead ends, not at every bend.
- retain_all=True: keep every connected piece, so remote parts such as Hatta are not
  silently discarded. The summary reports how many pieces there are.
- The outline is lightly simplified (about 50 m) to keep the Overpass queries manageable.
"""
import argparse
import re
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "data/raw/boundary/dubai_boundary.geojson"
CACHE = ROOT / "data/raw/osm_cache"
OUT = ROOT / "data/interim/osm_roads.gpkg"


def first_value(v):
    """OSM tags can be lists after simplification; use the first entry for counting."""
    return v[0] if isinstance(v, (list, tuple)) else v


def gpkg_ready(gdf):
    """GeoPackage cannot store list values: join them into 'a;b' text. Counts are reported."""
    gdf = gdf.copy()
    for col in gdf.columns:
        if col == "geometry":
            continue
        is_list = gdf[col].map(lambda v: isinstance(v, (list, tuple, set)))
        if is_list.any():
            print(f"  column '{col}': {int(is_list.sum())} list values joined as 'a;b' text")
            gdf[col] = gdf[col].map(lambda v: ";".join(map(str, v)) if isinstance(v, (list, tuple, set)) else v)
    return gdf


def report(G):
    nodes, edges = ox.graph_to_gdfs(G)
    print(f"\nnodes: {len(nodes):,} | directed edges: {len(edges):,} | CRS: {edges.crs}")
    oneway = edges["oneway"].map(lambda v: v is True) if "oneway" in edges else None
    if oneway is not None:
        print(f"edges flagged one-way: {int(oneway.sum()):,} ({oneway.mean():.1%} of directed edges)")
    print(f"total directed edge length: {edges['length'].sum() / 1000:,.0f} km "
          "(a two-way road counts once per direction)")
    pieces = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    print(f"weakly connected pieces: {len(pieces)} | largest 5 (nodes): {pieces[:5]} "
          f"| largest holds {pieces[0] / len(nodes):.1%} of nodes")
    print("strongly connected pieces:", nx.number_strongly_connected_components(G))
    print("edge road classes (first tag):", edges["highway"].map(first_value).value_counts().head(8).to_dict())
    return nodes, edges


def save(G, out: Path):
    nodes, edges = report(G)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    print("\npreparing layers for GeoPackage:")
    gpkg_ready(nodes.reset_index()).to_file(out, layer="nodes", driver="GPKG")
    gpkg_ready(edges.reset_index()).to_file(out, layer="edges", driver="GPKG")
    print("saved:", out)


TRANSIENT = re.compile(r"502|503|504|429|timed out|timeout|Gateway|capacity|Connection", re.I)


def with_retry(fn, tries=6, base_wait=60, sleep=time.sleep):
    """Call fn(); on a transient server error wait (doubling each time) and retry.

    Finished Overpass chunks are cached, so a retry only repeats the missing ones.
    Errors that do not look transient are raised immediately.
    """
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as err:
            text = str(err)
            first = text.splitlines()[0] if text else type(err).__name__
            if attempt == tries or not TRANSIENT.search(text):
                raise
            wait = min(base_wait * 2 ** (attempt - 1), 600)
            print(f"attempt {attempt} of {tries} failed ({first}); waiting {wait}s, then retrying...")
            sleep(wait)


def main(xml: Path | None, endpoint: str | None = None) -> None:
    print(f"osmnx {ox.__version__} | run date {date.today()}")
    if xml:
        print("building from local OSM XML (no download):", xml)
        G = ox.graph_from_xml(xml, simplify=True, retain_all=True)
    else:
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(CACHE)
        ox.settings.requests_timeout = 300
        ox.settings.log_console = True
        if endpoint:
            ox.settings.overpass_url = endpoint
            print("Overpass endpoint:", endpoint)
        poly = gpd.read_file(BOUNDARY).to_crs(4326).geometry.union_all()
        poly = poly.simplify(0.0005, preserve_topology=True)
        print("downloading drivable network inside the outline (this can take several minutes)...")
        G = with_retry(lambda: ox.graph_from_polygon(
            poly, network_type="drive", simplify=True, retain_all=True))
    save(G, OUT)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=None, help="build from a local .osm file instead (testing)")
    p.add_argument("--endpoint", default=None,
                   help="alternative Overpass server, e.g. https://overpass.kumi.systems/api")
    a = p.parse_args()
    main(a.xml, a.endpoint)
