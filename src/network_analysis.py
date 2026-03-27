"""
network_analysis.py
===================
Utility functions for bibliometric network analysis.
Supports: keyword co-occurrence, co-authorship, and other bipartite projections.
"""

from __future__ import annotations

import re
from itertools import combinations
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------------------------

# Default harmonization map: maps variant → canonical form.
# Users should extend this dict based on exploratory frequency analysis.
DEFAULT_KEYWORD_MAP = {
    # ABM variants → single canonical term
    "agent-based model": "agent-based modeling",
    "agent-based models": "agent-based modeling",
    "agent-based modelling": "agent-based modeling",
    "agent based model": "agent-based modeling",
    "agent based models": "agent-based modeling",
    "agent based modeling": "agent-based modeling",
    "agent based modelling": "agent-based modeling",
    "agent-based model (abm)": "agent-based modeling",
    "abm": "agent-based modeling",
    "abms": "agent-based modeling",
    # Decision-making variants
    "decision making": "decision-making",
    # Multi-agent variants
    "multi-agent system": "multi-agent systems",
    "multi agent systems": "multi-agent systems",
    "multi agent system": "multi-agent systems",
    "mas": "multi-agent systems",
    # Sensitivity analysis
    "sensitivity analyses": "sensitivity analysis",
    # Land use
    "land use change": "land-use change",
    "land use": "land-use change",
    # Agent-based simulation
    "agent-based simulations": "agent-based simulation",
}


def harmonize_keywords(
    kw_list: list[str],
    keyword_map: dict[str, str] | None = None,
) -> list[str]:
    """
    Map keyword variants to canonical forms using a lookup dictionary.
    """
    if keyword_map is None:
        keyword_map = DEFAULT_KEYWORD_MAP
    return [keyword_map.get(kw, kw) for kw in kw_list]


def parse_keywords(
    series: pd.Series,
    sep: str = ";",
    keyword_map: dict[str, str] | None = None,
) -> pd.Series:
    """
    Split a Series of delimited keyword strings into lists of cleaned,
    harmonized keywords.
    Returns a Series of lists (rows without keywords become empty lists).
    """
    if keyword_map is None:
        keyword_map = DEFAULT_KEYWORD_MAP

    def _clean(raw: str) -> list[str]:
        if pd.isna(raw) or str(raw).strip() == "":
            return []
        tokens = [t.strip().lower() for t in str(raw).split(sep)]
        tokens = [t for t in tokens if t]
        # Harmonize
        tokens = harmonize_keywords(tokens, keyword_map)
        # Remove duplicates that arise after harmonization (preserve order)
        seen = set()
        unique = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    return series.apply(_clean)


def build_cooccurrence_edges(
    keyword_lists: pd.Series,
    min_edge_weight: int = 1,
) -> pd.DataFrame:
    """
    From a Series of keyword lists, count pairwise co-occurrences.
    Returns DataFrame with columns [source, target, weight].
    """
    pair_counter: Counter = Counter()
    for kw_list in keyword_lists:
        unique = sorted(set(kw_list))
        for a, b in combinations(unique, 2):
            pair_counter[(a, b)] += 1

    rows = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in pair_counter.items()
        if w >= min_edge_weight
    ]
    return pd.DataFrame(rows)


def build_network(edge_df: pd.DataFrame) -> nx.Graph:
    """Build an undirected weighted graph from an edge DataFrame."""
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G


# ---------------------------------------------------------------------------
# 2. NETWORK METRICS
# ---------------------------------------------------------------------------

def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute standard centrality metrics for every node.
    Returns a DataFrame indexed by node label.
    """
    metrics = pd.DataFrame(index=list(G.nodes()))
    metrics.index.name = "keyword"

    metrics["degree"] = pd.Series(dict(G.degree()))
    metrics["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")))
    metrics["degree_centrality"] = pd.Series(nx.degree_centrality(G))
    metrics["betweenness_centrality"] = pd.Series(
        nx.betweenness_centrality(G, weight="weight")
    )
    metrics["closeness_centrality"] = pd.Series(
        nx.closeness_centrality(G)
    )
    try:
        metrics["eigenvector_centrality"] = pd.Series(
            nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
        )
    except nx.PowerIterationFailedConvergence:
        metrics["eigenvector_centrality"] = np.nan

    metrics["clustering_coefficient"] = pd.Series(nx.clustering(G, weight="weight"))

    return metrics.sort_values("weighted_degree", ascending=False)


def compute_global_metrics(G: nx.Graph) -> dict:
    """Compute graph-level summary statistics."""
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "avg_weighted_degree": np.mean([d for _, d in G.degree(weight="weight")]),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "connected_components": nx.number_connected_components(G),
    }

    if nx.is_connected(G):
        stats["diameter"] = nx.diameter(G)
        stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
        stats["diameter_largest_cc"] = nx.diameter(H)
        stats["avg_shortest_path_largest_cc"] = nx.average_shortest_path_length(H)
        stats["largest_cc_nodes"] = H.number_of_nodes()
        stats["largest_cc_pct"] = H.number_of_nodes() / G.number_of_nodes()

    return stats


def detect_communities(G: nx.Graph, resolution: float = 1.0) -> dict:
    """
    Louvain community detection.
    Returns dict {node: community_id}.
    """
    return community_louvain.best_partition(G, weight="weight", resolution=resolution)


# ---------------------------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------------------------

def _get_community_colormap(partition: dict) -> dict:
    """Map each community id to a distinct color (paper-quality palette)."""
    communities = sorted(set(partition.values()))
    n = len(communities)
    PAPER_COLORS = [
        "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
        "#264653", "#A8DADC", "#6A0572", "#1D3557", "#B5838D",
        "#8338EC", "#FF006E", "#3A86A7", "#06D6A0", "#118AB2",
    ]
    if n <= len(PAPER_COLORS):
        return {c: PAPER_COLORS[i] for i, c in enumerate(communities)}
    palette = plt.cm.turbo(np.linspace(0.1, 0.9, n))
    return {c: palette[i] for i, c in enumerate(communities)}


def plot_network(
    G: nx.Graph,
    partition: dict,
    node_metrics: pd.DataFrame,
    title: str = "Keyword Co-occurrence Network",
    figsize: tuple = (14, 11),
    node_scale: float = 40,
    edge_alpha: float = 0.20,
    font_size: int = 7,
    top_n_labels: int = 35,
    min_edge_weight_display: int = 1,
    layout_k: Optional[float] = None,
    layout_seed: int = 42,
    layout_iterations: int = 150,
    save_path: Optional[str] = None,
):
    """
    Draw the co-occurrence network with communities, sized by weighted degree.
    Compact layout optimized for publication.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")

    # --- Compact layout: low k = tighter packing ---
    k = layout_k or (0.7 / np.sqrt(G.number_of_nodes()))
    pos = nx.spring_layout(
        G, k=k, seed=layout_seed, weight="weight",
        iterations=layout_iterations, scale=1.0,
    )

    # Colors
    cmap = _get_community_colormap(partition)
    node_colors = [cmap[partition[n]] for n in G.nodes()]

    # Sizes – non-linear scaling
    w_deg = node_metrics.loc[list(G.nodes()), "weighted_degree"]
    node_sizes = (w_deg / w_deg.max()) ** 0.65 * (node_scale ** 1.5) + 12

    # Edges
    edges_to_draw = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] >= min_edge_weight_display
    ]
    edge_weights = [G[u][v]["weight"] for u, v in edges_to_draw]
    max_ew = max(edge_weights) if edge_weights else 1
    edge_widths = [0.2 + 2.2 * (w / max_ew) for w in edge_weights]
    edge_alphas = [0.06 + edge_alpha * (w / max_ew) for w in edge_weights]

    # Draw edges with varying alpha for depth
    for (u, v), ew, ea in zip(edges_to_draw, edge_widths, edge_alphas):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#888888", linewidth=ew, alpha=ea, zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.4,
        edgecolors="white",
        ax=ax,
    )

    # Labels – top N by weighted degree
    top_nodes = node_metrics.head(top_n_labels).index.tolist()
    labels = {n: n for n in top_nodes if n in G.nodes()}

    try:
        from adjustText import adjust_text
        texts = []
        for node, label in labels.items():
            x, y = pos[node]
            txt = ax.text(
                x, y, label, fontsize=font_size, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          alpha=0.75, edgecolor="none", linewidth=0),
            )
            texts.append(txt)
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
            expand=(1.2, 1.4),
            force_text=(0.6, 0.8),
        )
    except ImportError:
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=font_size, font_weight="bold", ax=ax,
        )

    # Legend
    for comm_id, color in sorted(cmap.items()):
        members = [n for n, c in partition.items() if c == comm_id]
        ax.scatter([], [], c=[color], s=70, label=f"C{comm_id} ({len(members)} kw)",
                   edgecolors="white", linewidths=0.5)
    ax.legend(
        loc="upper left", fontsize=7, framealpha=0.95,
        title="Communities (Louvain)", title_fontsize=8,
        ncol=2 if len(cmap) > 6 else 1,
        borderpad=0.8, handletextpad=0.3,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    ax.margins(0.03)
    plt.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig, ax


def plot_degree_distribution(G: nx.Graph, save_path: Optional[str] = None):
    """Plot degree distribution histogram + log-log."""
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")

    axes[0].hist(degrees, bins=range(1, max(degrees) + 2), color="#457B9D",
                 edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Degree (k)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Degree Distribution", fontsize=13, fontweight="bold")

    deg_count = Counter(degrees)
    x = sorted(deg_count.keys())
    y = [deg_count[d] for d in x]
    axes[1].scatter(x, y, color="#E63946", s=40, zorder=3)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Degree k (log)", fontsize=11)
    axes[1].set_ylabel("P(k) (log)", fontsize=11)
    axes[1].set_title("Degree Distribution (log-log)", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_centrality_comparison(
    node_metrics: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """Horizontal bar charts comparing top keywords across centrality measures."""
    measures = [
        ("weighted_degree", "Weighted Degree"),
        ("betweenness_centrality", "Betweenness Centrality"),
        ("closeness_centrality", "Closeness Centrality"),
        ("eigenvector_centrality", "Eigenvector Centrality"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), facecolor="white")
    colors = ["#457B9D", "#E9C46A", "#2A9D8F", "#E63946"]

    for idx, (col, label) in enumerate(measures):
        ax = axes[idx // 2][idx % 2]
        top = node_metrics.nlargest(top_n, col)[[col]].sort_values(col)
        ax.barh(top.index, top[col], color=colors[idx], alpha=0.85, edgecolor="white")
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(f"Top {top_n} – {label}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_community_heatmap(
    G: nx.Graph,
    partition: dict,
    save_path: Optional[str] = None,
):
    """Heatmap of inter-community edge density."""
    communities = sorted(set(partition.values()))
    n_comm = len(communities)
    matrix = np.zeros((n_comm, n_comm))

    for u, v, d in G.edges(data=True):
        ci = partition[u]
        cj = partition[v]
        w = d.get("weight", 1)
        matrix[communities.index(ci)][communities.index(cj)] += w
        if ci != cj:
            matrix[communities.index(cj)][communities.index(ci)] += w

    labels = [f"C{c}" for c in communities]
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Inter-Community Co-occurrence Strength", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def summarize_communities(
    partition: dict,
    node_metrics: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each community, list size and top keywords by weighted degree.
    Returns a summary DataFrame.
    """
    rows = []
    for comm_id in sorted(set(partition.values())):
        members = [n for n, c in partition.items() if c == comm_id]
        sub = node_metrics.loc[node_metrics.index.isin(members)]
        top_kws = sub.nlargest(top_n, "weighted_degree").index.tolist()
        rows.append({
            "community": comm_id,
            "size": len(members),
            "top_keywords": "; ".join(top_kws),
            "avg_degree": sub["degree"].mean(),
            "avg_betweenness": sub["betweenness_centrality"].mean(),
        })
    return pd.DataFrame(rows)
