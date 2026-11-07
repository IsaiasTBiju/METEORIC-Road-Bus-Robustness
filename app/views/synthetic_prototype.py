"""SYNTHETIC prototype of the METEORIC app layout. Invented data only.
Not Dubai or RTA results."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # temporary shortcut; real packaging comes later

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from meteoric.disruption import affected_routes, connectivity_summary, remove_segments
from meteoric.graph_build import build_directed, build_undirected, load_segments

DATA = ROOT / "data" / "synthetic"

# model -> (respects one-way direction, demand-weighted route measure)
MODELS = {
    "M0: Baseline": (False, False),
    "M1: Direction-aware": (True, False),
    "M2: Demand-aware": (False, True),
    "M3: Direction- and demand-aware": (True, True),
}


@st.cache_data
def load_data():
    segments = load_segments(DATA / "SYNTHETIC_segments.csv")
    nodes = pd.read_csv(DATA / "SYNTHETIC_nodes.csv")
    route_segments = pd.read_csv(DATA / "SYNTHETIC_route_segments.csv")
    demand = pd.read_csv(DATA / "SYNTHETIC_route_demand.csv")
    return segments, nodes, route_segments, demand


def draw_map(segments, nodes, closed):
    from matplotlib.lines import Line2D

    pos = {r.node_id: (r.x, r.y) for r in nodes.itertuples()}
    fig, ax = plt.subplots(figsize=(7, 4.4))

    for row in segments.itertuples():
        (x1, y1), (x2, y2) = pos[row.from_node], pos[row.to_node]
        is_closed = row.segment_id in closed
        colour = "#d62728" if is_closed else "#1f77b4"
        style = "--" if is_closed else "-"

        if row.oneway:
            # shrinkA/shrinkB stop the arrow short of the junction circles
            # so the arrowhead is not hidden behind them.
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.5,
                                linestyle=style, shrinkA=18, shrinkB=18,
                                mutation_scale=22),
            )
            label_at = 0.35  # nearer the tail, so crossing segments do not share a label spot
        else:
            ax.plot([x1, x2], [y1, y2], color=colour, lw=2.5, ls=style, zorder=2)
            label_at = 0.5

        ax.text(
            x1 + (x2 - x1) * label_at, y1 + (y2 - y1) * label_at, row.segment_id,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=colour, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
        )

    for node_id, (x, y) in pos.items():
        ax.scatter(x, y, s=600, color="white", edgecolor="black", linewidth=1.5, zorder=3)
        ax.text(x, y, node_id, ha="center", va="center", fontweight="bold", zorder=4)

    legend = [
        Line2D([0], [0], color="#1f77b4", lw=2.5, label="Two-way road"),
        Line2D([0, 1], [0, 0], color="#1f77b4", lw=2.5, marker=">", markersize=8,
               markevery=[1], label="One-way road"),
        Line2D([0], [0], color="#d62728", lw=2.5, ls="--", label="Closed segment"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False)
    ax.set_title("SYNTHETIC network", pad=14)
    ax.margins(0.12)
    ax.axis("off")
    return fig


st.title("METEORIC: synthetic prototype")
st.warning("SYNTHETIC DATA ONLY. These results are not Dubai or RTA findings.")

segments, nodes, route_segments, demand = load_data()

with st.sidebar:
    st.header("Controls")
    model = st.radio("Model", list(MODELS))
    closed = st.multiselect("Segments to close", list(segments["segment_id"]))

directional, demand_weighted = MODELS[model]
graph = build_directed(segments) if directional else build_undirected(segments)
before = connectivity_summary(graph)
after = connectivity_summary(remove_segments(graph, closed))
hit = affected_routes(route_segments, closed)

st.subheader(f"{model}")
st.caption(
    f"Road graph: {'directed (one-way respected)' if directional else 'undirected (all roads two-way)'}. "
    f"Route measure: {'demand-weighted' if demand_weighted else 'routes counted equally'}. "
    "No rerouting is assumed: a route is affected if it uses a closed segment."
)

col_map, col_tables = st.columns([3, 2])
with col_map:
    st.pyplot(draw_map(segments, nodes, set(closed)))

with col_tables:
    piece_label = "strongly connected components" if directional else "connected components"
    st.markdown("**Road-graph connectivity**")
    st.table(pd.DataFrame({
        "Before": [before["n_pieces"], f"{before['largest_share']:.3f}"],
        "After": [after["n_pieces"], f"{after['largest_share']:.3f}"],
    }, index=[f"Pieces ({piece_label})", "Largest piece (share of junctions)"]))

    st.markdown("**Bus-route effect (provisional measure)**")
    if demand_weighted:
        total = demand["monthly_total"].sum()
        hit_total = demand.loc[demand["route_id"].isin(hit), "monthly_total"].sum()
        st.metric("Share of route demand affected", f"{hit_total / total:.3f}",
                  help="Summed monthly totals of affected routes / summed totals of all routes. "
                       "Route-level exposure, not passengers on individual segments.")
    else:
        st.metric("Share of routes affected", f"{len(hit) / demand['route_id'].nunique():.3f}",
                  help="Affected routes / all routes. Each route counts equally.")

routes_view = demand.assign(affected=demand["route_id"].isin(hit))
st.markdown("**Routes**")
st.dataframe(routes_view.drop(columns="data_status"), hide_index=True)
st.caption("The share measures above are provisional placeholders for the layout test. "
           "The final definitions are a research decision for later stages.")
