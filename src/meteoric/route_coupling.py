"""Turn matched bus lines into the road-bus coupling tables used by all four models.

route_segment_table: one row per (route, road segment): which directions the route uses it in.
route_coverage:      one row per route: how much of its path we could match onto the road network.
"""
import pandas as pd

HIGH_COVERAGE = 0.95   # sensitivity runs keep only routes with at least this coverage


def route_segment_table(seq: pd.DataFrame) -> pd.DataFrame:
    """seq: shape_id, route, segment_id, travel, source (from fill_gaps).

    travel_dirs is 'fwd', 'rev' or 'both' (relative to the stored segment geometry).
    origin is 'matched' if any bus line matched the segment directly, else 'filled'.
    """
    g = seq.groupby(["route", "segment_id"])
    out = pd.DataFrame({
        "travel_dirs": g["travel"].agg(lambda s: "both" if s.nunique() > 1 else s.iloc[0]),
        "n_lines": g["shape_id"].nunique(),
        "origin": g["source"].agg(lambda s: "matched" if (s == "matched").any() else "filled"),
    })
    return out.reset_index()


def route_coverage(qual: pd.DataFrame, seq: pd.DataFrame, high: float = HIGH_COVERAGE) -> pd.DataFrame:
    """qual: per bus line (route, points, share_ok, n_unbridged, n_filled). seq as above.

    coverage = share of the route's sample points that matched a segment (all its lines together).
    """
    q = qual.assign(ok_points=qual["points"] * qual["share_ok"])
    g = q.groupby("route")
    out = pd.DataFrame({
        "n_lines": g.size(),
        "points": g["points"].sum(),
        "coverage": g["ok_points"].sum() / g["points"].sum(),
        "worst_line_coverage": g["share_ok"].min(),
        "unbridged_gaps": g["n_unbridged"].sum(),
        "filled_gaps": g["n_filled"].sum(),
    })
    out["n_segments"] = seq.groupby("route")["segment_id"].nunique()
    out["high_coverage"] = out["coverage"] >= high
    return out.reset_index()
