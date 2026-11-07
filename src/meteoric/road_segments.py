"""Group directed road edges (from OSMnx) into junction-to-junction segments.

A segment is one stretch of road between two junctions. A two-way road appears in OSMnx as two
directed edges with the same geometry (one reversed); both belong to ONE segment. A one-way road
is a segment with a single directed edge. Roads joining the same two junctions along different
paths (different OSM way or different shape) are different segments.
"""
import geopandas as gpd
import pandas as pd


def _shape(geom, ndigits: int = 6) -> tuple:
    return tuple((round(x, ndigits), round(y, ndigits)) for x, y in geom.coords)


def assign_segments(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add segment_id, direction ('fwd'/'rev' relative to a canonical shape) and n_dirs.

    Needs columns: u, v, osmid, geometry. Ids are deterministic (sorted keys): R000001, R000002...
    """
    e = edges.copy()
    shapes = [_shape(g) for g in e.geometry]
    canon = [min(s, s[::-1]) for s in shapes]
    e["direction"] = ["fwd" if s == c else "rev" for s, c in zip(shapes, canon)]
    keys = pd.Series(
        list(zip(e[["u", "v"]].min(axis=1), e[["u", "v"]].max(axis=1),
                 e["osmid"].astype(str), [str(c) for c in canon])),
        index=e.index,
    )
    codes, _ = pd.factorize(keys, sort=True)
    e["segment_id"] = ["R" + str(c + 1).zfill(6) for c in codes]
    e["n_dirs"] = e.groupby("segment_id")["segment_id"].transform("size")
    return e


def segment_checks(e: gpd.GeoDataFrame) -> dict:
    """Consistency counts between OSM's one-way tag and what the grouping found."""
    tag_oneway = e["oneway"].astype(str).eq("True")
    per_seg = e.drop_duplicates("segment_id")
    return {
        "directed_edges": len(e),
        "segments": len(per_seg),
        "segments_one_direction": int((per_seg["n_dirs"] == 1).sum()),
        "segments_two_directions": int((per_seg["n_dirs"] == 2).sum()),
        "segments_with_more_than_2_edges": int((per_seg["n_dirs"] > 2).sum()),
        "tagged_oneway_but_paired": int((tag_oneway & (e["n_dirs"] == 2)).sum()),
        "tagged_twoway_but_unpaired": int((~tag_oneway & (e["n_dirs"] == 1)).sum()),
    }


def segments_table(e: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per segment, in the same schema as the synthetic table (+ extra attributes)."""
    first = e.sort_values(["segment_id", "direction"]).drop_duplicates("segment_id", keep="first")
    access = first["access"].fillna("").astype(str) if "access" in first else pd.Series("", index=first.index)
    out = gpd.GeoDataFrame(
        {
            "segment_id": first["segment_id"].values,
            "from_node": first["u"].values,
            "to_node": first["v"].values,
            "oneway": (first["n_dirs"] == 1).values,
            "length_m": first["length"].values,
            "highway": first["highway"].astype(str).str.split(";").str[0].values,
            "name": first["name"].values if "name" in first else None,
            "osmid": first["osmid"].astype(str).values,
            "access": access.values,
            "is_private": access.str.contains("private").values,
            "is_loop": (first["u"] == first["v"]).values,
            "n_edges": first["n_dirs"].values,
        },
        geometry=first.geometry.values,
        crs=e.crs,
    )
    return out.reset_index(drop=True)
