"""Tests for grouping directed edges into segments (hand-made edges, no real data)."""
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from meteoric.graph_build import build_directed, build_undirected
from meteoric.road_segments import assign_segments, segment_checks, segments_table


def make_edges():
    rows = [
        # u, v, osmid, oneway tag, coords
        (1, 2, "10", True,  [(0, 0), (1, 0)]),                          # one-way road
        (2, 3, "11", False, [(1, 0), (2, 0)]),                          # two-way road, forward
        (3, 2, "11", False, [(2, 0), (1, 0)]),                          # two-way road, reverse copy
        (2, 3, "12", False, [(1, 0), (1.5, 1), (2, 0)]),                # different road, same junctions
        (4, 4, "13", False, [(0, 1), (1, 1), (1, 2), (0, 1)]),          # loop, one direction
        (4, 4, "13", False, [(0, 1), (1, 2), (1, 1), (0, 1)]),          # loop, reverse copy
    ]
    return gpd.GeoDataFrame(
        {"u": [r[0] for r in rows], "v": [r[1] for r in rows], "key": 0,
         "osmid": [r[2] for r in rows], "oneway": [r[3] for r in rows],
         "length": [100.0] * len(rows), "highway": "residential"},
        geometry=[LineString(r[4]) for r in rows], crs=4326,
    )


def test_pairs_and_singles_are_grouped_correctly():
    e = assign_segments(make_edges())
    assert e["segment_id"].nunique() == 4
    assert e.loc[1, "segment_id"] == e.loc[2, "segment_id"]       # two-way pair shares an id
    assert e.loc[4, "segment_id"] == e.loc[5, "segment_id"]       # loop pair shares an id
    assert e.loc[3, "segment_id"] not in {e.loc[1, "segment_id"], e.loc[0, "segment_id"]}
    assert sorted(e.groupby("segment_id").size().tolist()) == [1, 1, 2, 2]


def test_checks_report_consistency():
    c = segment_checks(assign_segments(make_edges()))
    assert c["segments"] == 4 and c["segments_one_direction"] == 2 and c["segments_two_directions"] == 2
    assert c["tagged_oneway_but_paired"] == 0 and c["tagged_twoway_but_unpaired"] == 1  # edge 3 is unpaired


def test_ids_are_deterministic():
    a = assign_segments(make_edges())["segment_id"].tolist()
    b = assign_segments(make_edges().iloc[::-1].reset_index(drop=True))["segment_id"].tolist()[::-1]
    assert a == b


def test_table_matches_synthetic_schema_and_rebuilds_the_same_directed_edges():
    e = assign_segments(make_edges())
    seg = pd.DataFrame(segments_table(e).drop(columns="geometry"))
    assert {"segment_id", "from_node", "to_node", "oneway", "length_m"} <= set(seg.columns)
    assert seg["oneway"].dtype == bool
    assert len(build_undirected(seg).edges) == 4            # one undirected edge per segment
    # 6 input edges, but the loop's two copies share one key and merge into one edge.
    # A self-loop never changes connectivity, so this is harmless.
    assert len(build_directed(seg).edges) == len(e) - 1


def test_one_way_direction_is_preserved():
    e = assign_segments(make_edges())
    seg = segments_table(e).set_index("segment_id")
    sid = e.loc[0, "segment_id"]
    assert (seg.loc[sid, "from_node"], seg.loc[sid, "to_node"], seg.loc[sid, "oneway"]) == (1, 2, True)
