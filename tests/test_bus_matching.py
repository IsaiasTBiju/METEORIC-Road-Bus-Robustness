"""Tests for direction-aware bus matching on a tiny hand-made road (metres, no real data)."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from meteoric.bus_matching import build_sequences, match_points, sample_shape_points

CRS = 32640


def segs_table():
    rows = [
        # id, from, to, oneway, coords
        ("EAST", 1, 2, True,  [(0, 0), (200, 0)]),       # one-way eastbound carriageway at y=0
        ("WEST", 3, 4, True,  [(200, 10), (0, 10)]),     # one-way westbound carriageway at y=10
        ("TWO",  5, 6, False, [(0, 100), (200, 100)]),   # two-way road at y=100
        ("NEXT", 2, 7, True,  [(200, 0), (400, 0)]),     # continues EAST (shares node 2)
        ("FAR",  8, 9, True,  [(1000, 0), (1200, 0)]),   # not connected to EAST
    ]
    return gpd.GeoDataFrame(
        {"segment_id": [r[0] for r in rows], "from_node": [r[1] for r in rows],
         "to_node": [r[2] for r in rows], "oneway": [r[3] for r in rows]},
        geometry=[LineString(r[4]) for r in rows], crs=CRS,
    )


def bus(shape_id, coords):
    return gpd.GeoDataFrame({"shape_id": [shape_id], "route_short_name": ["T"]},
                            geometry=[LineString(coords)], crs=CRS)


def run(coords):
    pts = sample_shape_points(bus("s", coords))
    return match_points(pts, segs_table())


def test_wrong_direction_nearest_road_is_not_chosen():
    # Bus drives WEST at y=3: EAST (3 m away) is nearer but points the wrong way; WEST is 7 m away.
    m = run([(180, 3), (20, 3)])
    assert (m["status"] == "ok").all()
    assert set(m["segment_id"]) == {"WEST"}


def test_bus_going_east_gets_the_eastbound_carriageway():
    m = run([(20, 3), (180, 3)])
    assert set(m["segment_id"]) == {"EAST"}


def test_two_way_road_accepts_both_directions_and_reports_travel():
    east = run([(20, 98), (180, 98)])
    west = run([(180, 98), (20, 98)])
    assert set(east["segment_id"]) == set(west["segment_id"]) == {"TWO"}
    assert set(east["travel"]) != set(west["travel"])


def test_far_points_are_unmatched_not_dropped():
    m = run([(20, 500), (180, 500)])
    assert (m["status"] == "unmatched").all() and len(m) > 0


def test_only_wrong_direction_road_nearby_is_flagged():
    # Bus drives west at y=-20: WEST (30 m away) is out of range: only EAST is near.
    m = run([(180, -20), (20, -20)])
    assert (m["status"] == "direction_mismatch").all()


def test_sequence_and_breaks():
    segs = pd.DataFrame(segs_table().drop(columns="geometry"))
    seq, qual = build_sequences(run([(20, 3), (380, 3)]), segs)
    assert list(seq["segment_id"]) == ["EAST", "NEXT"]
    assert qual.loc[0, "breaks"] == 0

    # a bus line that jumps from EAST to FAR: two segments that do not touch -> 1 break
    jumpy = pd.concat([run([(20, 3), (180, 3)]), run([(1020, 3), (1180, 3)]).assign(shape_id="s")],
                      ignore_index=True)
    jumpy["order"] = range(len(jumpy))
    _, qual2 = build_sequences(jumpy, segs)
    assert qual2.loc[0, "breaks"] == 1


def test_viterbi_ignores_a_merging_link_that_is_briefly_nearer():
    # Main road M1 (node 1->2) then M2 (2->3) along y=0. A link L (9->2) runs beside M1 at y=4
    # and merges into node 2. A bus on the main road at y=5 is nearer to L for a while.
    rows = [
        ("M1", 1, 2, True, [(0, 0), (200, 0)]),
        ("M2", 2, 3, True, [(200, 0), (400, 0)]),
        ("L",  9, 2, True, [(100, 4), (200, 4)]),
    ]
    segs = gpd.GeoDataFrame(
        {"segment_id": [r[0] for r in rows], "from_node": [r[1] for r in rows],
         "to_node": [r[2] for r in rows], "oneway": [r[3] for r in rows]},
        geometry=[LineString(r[4]) for r in rows], crs=CRS,
    )
    pts = sample_shape_points(bus("s", [(10, 5), (390, 5)]))
    nearest = match_points(pts, segs, method="nearest")
    smart = match_points(pts, segs, method="viterbi")
    assert "L" in set(nearest["segment_id"])                # the problem we saw on real data
    assert set(smart["segment_id"]) == {"M1", "M2"}         # fixed
    seq, qual = build_sequences(smart, pd.DataFrame(segs.drop(columns="geometry")))
    assert list(seq["segment_id"]) == ["M1", "M2"] and qual.loc[0, "breaks"] == 0


def test_viterbi_keeps_unmatched_points_and_wrong_way_flags():
    pts = sample_shape_points(bus("s", [(20, 500), (180, 500)]))
    assert (match_points(pts, segs_table(), method="viterbi")["status"] == "unmatched").all()
    pts = sample_shape_points(bus("s", [(180, -20), (20, -20)]))
    assert (match_points(pts, segs_table(), method="viterbi")["status"] == "direction_mismatch").all()


# ---- gap filling -------------------------------------------------------------------------------
from meteoric.bus_matching import fill_gaps  # noqa: E402


def gap_segs():
    rows = [
        # id, from, to, oneway, length
        ("A", 1, 2, True, 100.0),
        ("B", 2, 3, True, 50.0),     # short connector A -> C
        ("C", 3, 4, True, 100.0),
        ("T", 3, 2, False, 40.0),    # two-way connector stored 3->2 (so 2->3 is 'rev')
        ("L", 4, 5, True, 400.0),    # long connector
        ("M", 5, 6, True, 100.0),
        ("W", 8, 7, True, 30.0),     # one-way the WRONG way (7 <- 8)
        ("X", 6, 7, True, 100.0),
    ]
    return pd.DataFrame({"segment_id": [r[0] for r in rows], "from_node": [r[1] for r in rows],
                         "to_node": [r[2] for r in rows], "oneway": [r[3] for r in rows],
                         "length_m": [r[4] for r in rows]})


def seq_of(*items):
    return pd.DataFrame({"shape_id": "s", "route": "R", "seq_order": range(len(items)),
                         "segment_id": [i[0] for i in items], "travel": [i[1] for i in items]})


def test_short_gap_is_filled_with_the_connecting_segment():
    filled, gaps = fill_gaps(seq_of(("A", "fwd"), ("C", "fwd")), gap_segs().query("segment_id != 'T'"))
    assert list(filled["segment_id"]) == ["A", "B", "C"]
    assert list(filled["source"]) == ["matched", "filled", "matched"]
    assert list(filled["seq_order"]) == [0, 1, 2]
    assert gaps.loc[0, "status"] == "filled" and gaps.loc[0, "n_inserted"] == 1


def test_two_way_connector_is_used_in_the_right_direction():
    # remove B so the only way from node 2 to node 3 is the two-way T (stored 3->2)
    filled, _ = fill_gaps(seq_of(("A", "fwd"), ("C", "fwd")), gap_segs().query("segment_id != 'B'"))
    assert list(filled["segment_id"]) == ["A", "T", "C"]
    assert filled.loc[1, "travel"] == "rev"


def test_gap_longer_than_limit_is_reported_not_filled():
    filled, gaps = fill_gaps(seq_of(("C", "fwd"), ("M", "fwd")), gap_segs())   # needs 400 m link L
    assert list(filled["segment_id"]) == ["C", "M"]
    assert gaps.loc[0, "status"] == "unbridged"


def test_one_way_connector_in_wrong_direction_is_not_used():
    _, gaps = fill_gaps(seq_of(("X", "fwd"), ("W", "fwd")), gap_segs())
    assert gaps.loc[0, "status"] == "unbridged"


def test_already_joined_pairs_are_left_alone():
    filled, gaps = fill_gaps(seq_of(("A", "fwd"), ("B", "fwd"), ("C", "fwd")), gap_segs())
    assert list(filled["segment_id"]) == ["A", "B", "C"] and set(filled["source"]) == {"matched"}
    assert len(gaps) == 0