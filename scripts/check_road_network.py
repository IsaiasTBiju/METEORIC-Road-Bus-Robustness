"""Read-only checks of data/interim/osm_roads.gpkg: connectivity, access tags, classes, edge quirks.

Prints aggregate figures only. Writes nothing.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(gpkg: Path) -> None:
    nodes = gpd.read_file(gpkg, layer="nodes")
    edges = gpd.read_file(gpkg, layer="edges")
    print(f"nodes {len(nodes):,} | directed edges {len(edges):,}")

    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes["osmid"])
    G.add_edges_from(zip(edges["u"], edges["v"], edges["key"]))
    xy = nodes.set_index("osmid")[["x", "y"]]
    n = G.number_of_nodes()

    weak = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    print(f"\nweak pieces: {len(weak)} | largest: {[len(c) for c in weak[:5]]}")
    for i, comp in enumerate(weak[1:3], start=2):
        c = xy.loc[list(comp)].mean()
        print(f"  piece {i}: {len(comp)} nodes, centre lat {c['y']:.3f}, lon {c['x']:.3f}")

    strong = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    giant_weak = weak[0]
    giant_strong = strong[0]
    print(f"\nstrong pieces: {len(strong)} | largest 5: {[len(c) for c in strong[:5]]}")
    print(f"largest strong piece holds {len(giant_strong) / n:.1%} of all nodes "
          f"and {len(giant_strong) / len(giant_weak):.1%} of the largest weak piece")
    print(f"single-node strong pieces: {sum(1 for c in strong if len(c) == 1):,}")
    outside = len(giant_weak) - len(giant_weak & giant_strong)
    print(f"nodes in the big weak piece but outside the big strong piece: {outside:,}")

    first = edges["highway"].astype(str).str.split(";").str[0]
    ow = edges["oneway"].astype(str).eq("True")
    cls = pd.DataFrame({"class": first, "oneway": ow}).groupby("class").agg(
        edges=("oneway", "size"), oneway_share=("oneway", "mean")).sort_values("edges", ascending=False)
    cls["oneway_share"] = cls["oneway_share"].round(3)
    print("\nroad classes (first tag):")
    print(cls.to_string())

    if "access" in edges:
        print("\naccess tag values:", edges["access"].fillna("(none)").value_counts().head(8).to_dict())
    for col in ["motor_vehicle", "motorcar"]:
        if col in edges:
            print(f"{col} tag values:", edges[col].fillna("(none)").value_counts().head(5).to_dict())

    ln = edges["length"]
    print(f"\nedge length (m): min {ln.min():.1f} | median {ln.median():.1f} | max {ln.max():,.0f}")
    print(f"edges shorter than 1 m: {int((ln < 1).sum())} | self-loops: {int((edges['u'] == edges['v']).sum())}")
    pairs = edges.groupby(["u", "v"]).size()
    print(f"node pairs joined by more than one edge: {int((pairs > 1).sum()):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpkg", type=Path, default=ROOT / "data/interim/osm_roads.gpkg")
    main(p.parse_args().gpkg)
