"""Build undirected (M0/M2) and directed (M1/M3) road graphs from a segment table.

Segment table columns: segment_id, from_node, to_node, oneway (bool), length_m.
"""
from pathlib import Path

import networkx as nx
import pandas as pd


def load_segments(path: Path) -> pd.DataFrame:
    """Read a segment table and fail loudly if 'oneway' is not a clean bool."""
    segments = pd.read_csv(path)
    if segments["oneway"].dtype != bool:
        raise ValueError(
            f"'oneway' must be True/False only, found dtype {segments['oneway'].dtype}"
        )
    if segments["segment_id"].duplicated().any():
        raise ValueError("segment_id values must be unique")
    return segments


def build_undirected(segments: pd.DataFrame) -> nx.MultiGraph:
    """One edge per segment, usable in both directions (M0 and M2)."""
    graph = nx.MultiGraph()
    for row in segments.itertuples(index=False):
        graph.add_edge(
            row.from_node, row.to_node, key=row.segment_id,
            segment_id=row.segment_id, length_m=row.length_m,
        )
    return graph


def build_directed(segments: pd.DataFrame) -> nx.MultiDiGraph:
    """One-way segments get one edge; two-way segments get one edge per direction."""
    graph = nx.MultiDiGraph()
    for row in segments.itertuples(index=False):
        attrs = {"segment_id": row.segment_id, "length_m": row.length_m}
        graph.add_edge(row.from_node, row.to_node, key=row.segment_id, **attrs)
        if not row.oneway:
            # The reverse edge shares the segment_id so both directions
            # are removed together when the segment is closed.
            graph.add_edge(row.to_node, row.from_node, key=row.segment_id, **attrs)
    return graph
