"""Close road segments and summarise the effect (no rerouting assumed)."""
import networkx as nx
import pandas as pd


def remove_segments(graph, segment_ids):
    """Return a copy of the graph with every edge of the given segments removed.

    Nodes are kept, so a junction left with no roads becomes its own piece.
    Raises ValueError for unknown segment ids instead of ignoring them.
    """
    ids = set(segment_ids)
    known = {data["segment_id"] for _, _, data in graph.edges(data=True)}
    unknown = ids - known
    if unknown:
        raise ValueError(f"Unknown segment ids: {sorted(unknown)}")

    damaged = graph.copy()
    for u, v, key, data in list(damaged.edges(keys=True, data=True)):
        if data["segment_id"] in ids:
            damaged.remove_edge(u, v, key=key)
    return damaged


def connectivity_summary(graph):
    """Count pieces and the largest piece's share of all nodes.

    Undirected graph: connected components.
    Directed graph: strongly connected components.
    """
    if graph.is_directed():
        pieces = list(nx.strongly_connected_components(graph))
    else:
        pieces = list(nx.connected_components(graph))
    total_nodes = graph.number_of_nodes()
    return {
        "n_pieces": len(pieces),
        "largest_share": max(len(p) for p in pieces) / total_nodes,
    }


def affected_routes(route_segments: pd.DataFrame, closed_segment_ids):
    """Routes that use at least one closed segment (no rerouting assumed)."""
    hit = route_segments["segment_id"].isin(set(closed_segment_ids))
    return sorted(route_segments.loc[hit, "route_id"].unique())
