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