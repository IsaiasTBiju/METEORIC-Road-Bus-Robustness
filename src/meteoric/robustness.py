"""Disruption engine for the coupled road-bus network (models M0-M3).

Failure unit = one road segment (junction-to-junction link); closing it closes both directions.
No rerouting: a bus route that loses any segment of its path is 'affected'.

Models (2x2 design):
    M0  undirected,  no demand        M1  directed,  no demand
    M2  undirected,  demand-aware     M3  directed,  demand-aware
Directed models measure road connectivity with strongly connected components (one-way streets
respected); undirected models use weakly connected components. Demand-aware models rank
'targeted' removals by passenger demand exposure; the others by number of routes using a segment.

Measures reported for every run:
    road_share            largest connected piece, as a share of all junctions
    road_retained         road_share after disruption / road_share before
    routes_affected_share share of bus routes that lose at least one segment
    demand_affected_share passenger demand on affected routes / total demand
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MODELS = {
    "M0": {"directed": False, "demand": False},
    "M1": {"directed": True, "demand": False},
    "M2": {"directed": False, "demand": True},
    "M3": {"directed": True, "demand": True},
}
STRATEGIES = ("random", "targeted")


@dataclass
class Network:
    segment_ids: np.ndarray
    u: np.ndarray                 # junction index at the start of each segment
    v: np.ndarray                 # junction index at the end
    oneway: np.ndarray            # bool
    n_nodes: int
    route_names: np.ndarray
    route_seg: np.ndarray         # all routes' segment indices, concatenated
    route_offsets: np.ndarray     # where each route starts in route_seg
    route_demand: np.ndarray      # per route (0 if unknown)
    routes_without_demand: int
    n_routes_per_segment: np.ndarray
    demand_per_segment: np.ndarray
    _base: dict = field(default_factory=dict)


def build_network(segments: pd.DataFrame, route_segments: pd.DataFrame,
                  demand: pd.Series | None = None) -> Network:
    """segments: segment_id, from_node, to_node, oneway. route_segments: route, segment_id.
    demand: passengers per route (index = route name); routes missing from it count as 0."""
    seg_ids = segments["segment_id"].to_numpy()
    if len(set(seg_ids)) != len(seg_ids):
        raise ValueError("segment_id must be unique")
    n_seg = len(seg_ids)
    nodes, inv = np.unique(np.concatenate([segments["from_node"], segments["to_node"]]), return_inverse=True)

    lookup = pd.Series(np.arange(n_seg), index=seg_ids)
    rs = route_segments[["route", "segment_id"]].drop_duplicates()
    unknown = ~rs["segment_id"].isin(lookup.index)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} route-segment links use unknown segment ids, e.g. "
                         f"{rs.loc[unknown, 'segment_id'].iloc[0]}")
    rs = rs.assign(seg=lookup[rs["segment_id"]].to_numpy()).sort_values(["route", "seg"])
    counts = rs.groupby("route", sort=True).size()
    names = counts.index.to_numpy()
    offsets = np.concatenate([[0], np.cumsum(counts.to_numpy())[:-1]])

    if demand is None:
        dem = np.zeros(len(names))
        missing = len(names)
    else:
        aligned = demand.reindex(names)
        missing = int(aligned.isna().sum())
        dem = aligned.fillna(0).to_numpy(dtype=float)
    seg = rs["seg"].to_numpy()
    return Network(
        segment_ids=seg_ids, u=inv[:n_seg], v=inv[n_seg:], oneway=segments["oneway"].to_numpy(dtype=bool),
        n_nodes=len(nodes), route_names=names, route_seg=seg, route_offsets=offsets,
        route_demand=dem, routes_without_demand=missing,
        n_routes_per_segment=np.bincount(seg, minlength=n_seg),
        demand_per_segment=np.bincount(seg, weights=np.repeat(dem, counts.to_numpy()), minlength=n_seg),
    )


def _largest_piece(net: Network, kept: np.ndarray, directed: bool) -> int:
    u, v = net.u[kept], net.v[kept]
    if directed:
        two_way = ~net.oneway[kept]
        rows, cols = np.concatenate([u, v[two_way]]), np.concatenate([v, u[two_way]])
    else:
        rows, cols = u, v
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(net.n_nodes, net.n_nodes))
    _, labels = connected_components(graph, directed=True, connection="strong" if directed else "weak")
    return int(np.bincount(labels).max())


def road_share(net: Network, removed: np.ndarray, directed: bool) -> float:
    """Largest connected piece / all junctions, with the given segment indices removed."""
    kept = np.ones(len(net.segment_ids), dtype=bool)
    kept[removed] = False
    return _largest_piece(net, kept, directed) / net.n_nodes


def affected_routes(net: Network, removed: np.ndarray) -> np.ndarray:
    """Boolean per route: True if the route uses at least one removed segment."""
    gone = np.zeros(len(net.segment_ids), dtype=bool)
    gone[removed] = True
    return np.add.reduceat(gone[net.route_seg].astype(np.int64), net.route_offsets) > 0


def choose_segments(net: Network, model: str, strategy: str, n: int, seed: int = 0,
                    candidates: np.ndarray | None = None) -> np.ndarray:
    """Indices of the n segments to remove.

    random:   uniform without replacement (same seed -> same set for every model).
    targeted: top-n by score (demand exposure for M2/M3, number of routes for M0/M1);
              ties are broken by segment order so results are repeatable.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    n_seg = len(net.segment_ids)
    pool = np.arange(n_seg) if candidates is None else np.flatnonzero(candidates)
    if not 0 <= n <= len(pool):
        raise ValueError(f"n must be between 0 and {len(pool)}")
    if strategy == "random":
        return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))
    score = net.demand_per_segment if MODELS[model]["demand"] else net.n_routes_per_segment.astype(float)
    order = pool[np.lexsort((pool, -score[pool]))]
    return np.sort(order[:n])


def run_scenario(net: Network, model: str, strategy: str, n: int, seed: int = 0,
                 candidates: np.ndarray | None = None) -> dict:
    removed = choose_segments(net, model, strategy, n, seed, candidates)
    directed = MODELS[model]["directed"]
    base = road_share(net, np.array([], dtype=int), directed)
    after = road_share(net, removed, directed)
    hit = affected_routes(net, removed)
    total_demand = net.route_demand.sum()
    return {
        "model": model, "strategy": strategy, "n_removed": int(n), "seed": seed,
        "road_share_before": base, "road_share_after": after,
        "road_retained": after / base if base else np.nan,
        "routes_affected": int(hit.sum()), "routes_total": len(hit),
        "routes_affected_share": hit.mean(),
        "demand_affected_share": net.route_demand[hit].sum() / total_demand if total_demand else np.nan,
        "removed_segment_ids": net.segment_ids[removed],
    }
