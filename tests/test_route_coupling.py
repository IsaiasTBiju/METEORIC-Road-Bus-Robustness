"""Tests for the road-bus coupling tables (tiny hand-made inputs)."""
import pandas as pd

from meteoric.route_coupling import route_coverage, route_segment_table


def test_segment_table_merges_lines_and_directions():
    seq = pd.DataFrame({
        "shape_id": ["a", "a", "b", "b", "c"],
        "route":    ["R1", "R1", "R1", "R1", "R2"],
        "segment_id": ["S1", "S2", "S1", "S3", "S1"],
        "travel":   ["fwd", "fwd", "rev", "fwd", "fwd"],
        "source":   ["matched", "filled", "matched", "matched", "filled"],
    })
    t = route_segment_table(seq).set_index(["route", "segment_id"])
    assert len(t) == 4                                            # R1: S1,S2,S3 ; R2: S1
    assert t.loc[("R1", "S1"), "travel_dirs"] == "both" and t.loc[("R1", "S1"), "n_lines"] == 2
    assert t.loc[("R1", "S2"), "origin"] == "filled"
    assert t.loc[("R2", "S1"), "travel_dirs"] == "fwd"


def test_coverage_is_point_weighted_and_flags_high_routes():
    qual = pd.DataFrame({
        "route": ["R1", "R1", "R2"], "shape_id": ["a", "b", "c"],
        "points": [100, 300, 200], "share_ok": [1.0, 0.8, 0.96],
        "n_unbridged": [0, 4, 1], "n_filled": [2, 3, 0],
    })
    seq = pd.DataFrame({"route": ["R1", "R1", "R2"], "segment_id": ["S1", "S2", "S1"]})
    c = route_coverage(qual, seq).set_index("route")
    assert abs(c.loc["R1", "coverage"] - (100 * 1.0 + 300 * 0.8) / 400) < 1e-9     # 0.85
    assert c.loc["R1", "worst_line_coverage"] == 0.8 and c.loc["R1", "unbridged_gaps"] == 4
    assert c.loc["R1", "n_segments"] == 2
    assert not c.loc["R1", "high_coverage"] and c.loc["R2", "high_coverage"]
