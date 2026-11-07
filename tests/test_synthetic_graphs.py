"""Tests on the SYNTHETIC toy network. Expected values were worked out by hand
from the network table, not copied from the code's output. Not Dubai/RTA data."""
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from meteoric.disruption import affected_routes, connectivity_summary, remove_segments
from meteoric.graph_build import build_directed, build_undirected, load_segments

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.fixture(scope="module")
def segments():
    return load_segments(DATA / "SYNTHETIC_segments.csv")


@pytest.fixture(scope="module")
def route_segments():
    return pd.read_csv(DATA / "SYNTHETIC_route_segments.csv")


@pytest.fixture()
def undirected(segments):
    return build_undirected(segments)


@pytest.fixture()
def directed(segments):
    return build_directed(segments)


def test_graph_sizes(undirected, directed):
    assert (undirected.number_of_nodes(), undirected.number_of_edges()) == (6, 7)
    # 4 two-way segments count twice, 3 one-way segments count once
    assert (directed.number_of_nodes(), directed.number_of_edges()) == (6, 11)


def test_one_way_direction_is_respected(directed):
    # S5 is one-way B -> E: it leaves B but must not leave E
    assert ("B", "E", "S5") in directed.edges(keys=True)
    assert ("E", "B", "S5") not in directed.edges(keys=True)


def test_two_way_segment_has_both_directions(directed):
    assert ("A", "B", "S1") in directed.edges(keys=True)
    assert ("B", "A", "S1") in directed.edges(keys=True)


def test_closing_s2_splits_directed_but_not_undirected(undirected, directed):
    u = connectivity_summary(remove_segments(undirected, ["S2"]))
    d = connectivity_summary(remove_segments(directed, ["S2"]))
    assert u["n_pieces"] == 1
    assert d["n_pieces"] == 2
    assert d["largest_share"] == pytest.approx(4 / 6)


def test_closing_s1_isolates_a_in_both(undirected, directed):
    for graph in (undirected, directed):
        result = connectivity_summary(remove_segments(graph, ["S1"]))
        assert result["n_pieces"] == 2
        assert result["largest_share"] == pytest.approx(5 / 6)


def test_closing_s3_keeps_roads_connected_but_cuts_two_routes(
    undirected, directed, route_segments
):
    for graph in (undirected, directed):
        assert connectivity_summary(remove_segments(graph, ["S3"]))["n_pieces"] == 1
    assert affected_routes(route_segments, ["S3"]) == ["R1", "R3"]


def test_closing_s5_affects_only_r2(route_segments):
    assert affected_routes(route_segments, ["S5"]) == ["R2"]


def test_unknown_segment_is_rejected(undirected):
    with pytest.raises(ValueError):
        remove_segments(undirected, ["S99"])


def test_remove_segments_does_not_change_the_original(directed):
    edges_before = directed.number_of_edges()
    remove_segments(directed, ["S2"])
    assert directed.number_of_edges() == edges_before


def test_oneway_column_must_be_clean_bool(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "segment_id,from_node,to_node,oneway,length_m\nX1,A,B,maybe,100\n"
    )
    with pytest.raises(ValueError):
        load_segments(bad)
