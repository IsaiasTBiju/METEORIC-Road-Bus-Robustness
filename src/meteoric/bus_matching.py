"""Direction-aware matching of bus route lines onto road segments.

All geometries must be in a metric CRS (we use UTM 40N, EPSG:32640).

Idea: walk along each bus line, and at every sample point ask "which road segment is the bus on?"
A segment is eligible if it is within TOL_M metres AND points the same way as the bus
(one-way segments: same direction only; two-way segments: either direction, but parallel).
The nearest eligible segment wins. If only wrong-direction segments are near, the point is kept
but flagged 'direction_mismatch'. If nothing is near, it is 'unmatched'. Nothing is dropped.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

TOL_M = 25       # how far a bus point may be from a segment
STEP_M = 25      # spacing of sample points along the bus line
ANGLE_TOL = 60   # degrees; bus and segment count as "same way" within this
HALF_M = 5       # half-length used to measure the local direction


def _bearing(x0, y0, x1, y1):
    return np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360


def _angle_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def sample_shape_points(shapes: gpd.GeoDataFrame, route_col: str = "route_short_name",
                        step_m: float = STEP_M) -> gpd.GeoDataFrame:
    """Points every step_m along each bus line, each with the bus's local bearing."""
    rows = []
    for shape_id, route, geom in zip(shapes["shape_id"], shapes[route_col], shapes.geometry):
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        order = 0
        for part in parts:
            length = part.length
            if length == 0:
                continue
            d = np.linspace(0, length, max(int(np.ceil(length / step_m)), 1) + 1)
            pts = shapely.line_interpolate_point(part, d)
            a = shapely.line_interpolate_point(part, np.clip(d - HALF_M, 0, length))
            b = shapely.line_interpolate_point(part, np.clip(d + HALF_M, 0, length))
            brg = _bearing(shapely.get_x(a), shapely.get_y(a), shapely.get_x(b), shapely.get_y(b))
            for p, bb in zip(pts, brg):
                rows.append((shape_id, route, order, bb, p))
                order += 1
    return gpd.GeoDataFrame(
        pd.DataFrame(rows, columns=["shape_id", "route", "order", "bearing", "geometry"]),
        geometry="geometry", crs=shapes.crs,
    )


def match_points(points: gpd.GeoDataFrame, segs: gpd.GeoDataFrame, tol_m: float = TOL_M) -> pd.DataFrame:
    """One row per point: status ('ok' / 'direction_mismatch' / 'unmatched'), segment_id, dist_m, travel."""
    tree = STRtree(segs.geometry.values)
    pi, si = tree.query(points.geometry.values, predicate="dwithin", distance=tol_m)
    pg, sg = points.geometry.values[pi], segs.geometry.values[si]
    dist = shapely.distance(pg, sg)
    length = shapely.length(sg)
    s = shapely.line_locate_point(sg, pg)
    a = shapely.line_interpolate_point(sg, np.clip(s - HALF_M, 0, length))
    b = shapely.line_interpolate_point(sg, np.clip(s + HALF_M, 0, length))
    ax, ay, bx, by = shapely.get_x(a), shapely.get_y(a), shapely.get_x(b), shapely.get_y(b)
    degenerate = (ax == bx) & (ay == by)          # tiny segment: no usable direction
    diff = _angle_diff(points["bearing"].values[pi], _bearing(ax, ay, bx, by))
    aligned = degenerate | (diff <= ANGLE_TOL)
    opposite = ~degenerate & (diff >= 180 - ANGLE_TOL)
    oneway = segs["oneway"].values[si].astype(bool)
    cand = pd.DataFrame({
        "point": pi, "seg": si, "dist_m": dist,
        "eligible": np.where(oneway, aligned, aligned | opposite),
        "travel": np.where(aligned, "fwd", "rev"),
    })
    cand["segment_id"] = segs["segment_id"].values[cand["seg"].values]

    best = cand[cand["eligible"]].sort_values(["point", "dist_m"]).drop_duplicates("point")
    best = best.assign(status="ok")
    left = cand[~cand["point"].isin(best["point"])].sort_values(["point", "dist_m"]).drop_duplicates("point")
    left = left.assign(status="direction_mismatch")
    found = pd.concat([best, left]).set_index("point")[["status", "segment_id", "dist_m", "travel"]]

    out = pd.DataFrame(index=np.arange(len(points)))
    out = out.join(found)
    out["status"] = out["status"].fillna("unmatched")
    out["shape_id"] = points["shape_id"].values
    out["route"] = points["route"].values
    out["order"] = points["order"].values
    return out


def build_sequences(matched: pd.DataFrame, segs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per bus line: ordered list of segments it uses (only 'ok' points), and a quality table.

    'breaks' counts steps between consecutive segments that do not share a junction.
    """
    nodes = {r.segment_id: {r.from_node, r.to_node} for r in segs.itertuples(index=False)}
    seq_rows, qual_rows = [], []
    for shape_id, g in matched.sort_values(["shape_id", "order"]).groupby("shape_id", sort=False):
        ok = g[g["status"] == "ok"]
        keep = ok["segment_id"] != ok["segment_id"].shift()
        seq = ok[keep]
        breaks = sum(
            1 for a, b in zip(seq["segment_id"].values[:-1], seq["segment_id"].values[1:])
            if not (nodes[a] & nodes[b])
        )
        for i, r in enumerate(seq.itertuples(index=False)):
            seq_rows.append((shape_id, r.route, i, r.segment_id, r.travel))
        n = len(g)
        qual_rows.append({
            "shape_id": shape_id, "route": g["route"].iloc[0], "points": n,
            "share_ok": (g["status"] == "ok").sum() / n,
            "share_direction_mismatch": (g["status"] == "direction_mismatch").sum() / n,
            "share_unmatched": (g["status"] == "unmatched").sum() / n,
            "segments": len(seq), "breaks": breaks,
        })
    return (pd.DataFrame(seq_rows, columns=["shape_id", "route", "seq_order", "segment_id", "travel"]),
            pd.DataFrame(qual_rows))
