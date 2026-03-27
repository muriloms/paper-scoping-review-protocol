"""
methodological_network.py
=========================
Utility functions for building and analyzing networks from
methodological classification columns (Data Source, Model Objective,
Sensitivity Analysis).

Supports:
- Within-column co-occurrence networks
- Cross-column co-occurrence (unified methodological network)
- Bipartite heatmaps between classification dimensions
- Alluvial / Sankey-style flow analysis
"""

from __future__ import annotations

from itertools import combinations, product
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. COLUMN DEFINITIONS & PARSING
# ---------------------------------------------------------------------------

# Canonical categories for each dimension
CATEGORY_DEFINITIONS = {
    "Data Source (Parameterization)": [
        "Theoretical (literature)",
        "Empirical (micro)",
        "Empirical (macro)",
        "Expert-based (qualitative, interview, participatory)",
        "Hybrid",
        "Synthetic (machine learning, AI, other)",
    ],
    "Model Objective (optional)": [
        "Exploratory",
        "Explanatory",
        "Predictive",
        "Descriptive / Phenomena-based",
        "Decision Support / Policy Evaluation",
        "Theoretical / Illustrative",
        "Communication / Education",
    ],
    "Sensitivity Analysis": [
        "Local (OAT/OFAT)",
        "Global (Variance-based)",
        "Global (Screening)",
        "Metamodeling / Surrogate",
        "Exploratory / Scenario-based",
        "Regression / Statistical Inference",
        "NA",
    ],
}

# Short labels for visualization
SHORT_LABELS = {
    # Data Source
    "Theoretical (literature)": "Theoretical",
    "Empirical (micro)": "Emp. Micro",
    "Empirical (macro)": "Emp. Macro",
    "Expert-based (qualitative, interview, participatory)": "Expert-based",
    "Hybrid": "Hybrid",
    "Synthetic (machine learning, AI, other)": "Synthetic/ML",
    # Model Objective
    "Exploratory": "Exploratory",
    "Explanatory": "Explanatory",
    "Predictive": "Predictive",
    "Descriptive / Phenomena-based": "Descriptive",
    "Decision Support / Policy Evaluation": "Decision Support",
    "Theoretical / Illustrative": "Theoretical/Illust.",
    "Communication / Education": "Communication",
    # Sensitivity Analysis
    "Local (OAT/OFAT)": "Local (OAT)",
    "Global (Variance-based)": "Global (Variance)",
    "Global (Screening)": "Global (Screening)",
    "Metamodeling / Surrogate": "Metamodeling",
    "Exploratory / Scenario-based": "Scenario-based",
    "Regression / Statistical Inference": "Regression/Stat.",
    "NA": "Not Reported",
}

# Dimension short names
DIM_SHORT = {
    "Data Source (Parameterization)": "DS",
    "Model Objective (optional)": "MO",
    "Sensitivity Analysis": "SA",
}

# Color palettes per dimension (paper-quality)
DIM_COLORS = {
    "Data Source (Parameterization)": "#E63946",
    "Model Objective (optional)": "#457B9D",
    "Sensitivity Analysis": "#2A9D8F",
}

DIM_PALETTES = {
    "Data Source (Parameterization)": [
        "#E63946", "#F28482", "#FFB4A2", "#B5183D", "#FF6B6B", "#C9184A",
    ],
    "Model Objective (optional)": [
        "#457B9D", "#1D3557", "#A8DADC", "#2B6A8E", "#6BAED6", "#48CAE4", "#023E8A",
    ],
    "Sensitivity Analysis": [
        "#2A9D8F", "#264653", "#E9C46A", "#F4A261", "#06D6A0", "#118AB2", "#B7B7A4",
    ],
}


def parse_classification_column(
    series: pd.Series,
    sep: str = ",",
) -> pd.Series:
    """
    Parse a multi-value classification column into lists of cleaned categories.
    """
    def _clean(raw):
        if pd.isna(raw) or str(raw).strip() == "":
            return []
        tokens = [t.strip() for t in str(raw).split(sep)]
        return [t for t in tokens if t]
    return series.apply(_clean)


def get_frequency(parsed_series: pd.Series) -> pd.Series:
    """Count frequency of each category in a parsed series."""
    all_cats = [c for clist in parsed_series for c in clist]
    return pd.Series(all_cats).value_counts()


# ---------------------------------------------------------------------------
# 2. WITHIN-COLUMN CO-OCCURRENCE NETWORK
# ---------------------------------------------------------------------------

def build_within_cooccurrence(
    parsed_series: pd.Series,
    min_weight: int = 1,
) -> pd.DataFrame:
    """
    Build co-occurrence edges from a single parsed classification column.
    Nodes are categories; edge weight = number of articles with both categories.
    """
    pair_counter: Counter = Counter()
    for cat_list in parsed_series:
        unique = sorted(set(cat_list))
        for a, b in combinations(unique, 2):
            pair_counter[(a, b)] += 1

    rows = [
        {"source": a, "target": b, "weight": w}
        for (a, b), w in pair_counter.items()
        if w >= min_weight
    ]
    return pd.DataFrame(rows)


def build_network_from_edges(edge_df: pd.DataFrame) -> nx.Graph:
    """Build an undirected weighted graph from an edge DataFrame."""
    G = nx.Graph()
    if edge_df.empty:
        return G
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight"])
    return G


# ---------------------------------------------------------------------------
# 3. CROSS-COLUMN CO-OCCURRENCE NETWORK
# ---------------------------------------------------------------------------

def build_cross_column_edges(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    parsed_a: pd.Series,
    parsed_b: pd.Series,
    min_weight: int = 1,
) -> pd.DataFrame:
    """
    Build bipartite-style co-occurrence edges between two classification dimensions.
    An edge connects category_a to category_b if they co-occur in the same article.
    """
    pair_counter: Counter = Counter()
    for cats_a, cats_b in zip(parsed_a, parsed_b):
        for a in set(cats_a):
            for b in set(cats_b):
                pair_counter[(a, b)] += 1

    rows = [
        {"source": a, "target": b, "weight": w, "dim_source": col_a, "dim_target": col_b}
        for (a, b), w in pair_counter.items()
        if w >= min_weight
    ]
    return pd.DataFrame(rows)


def build_unified_methodological_network(
    parsed_columns: dict[str, pd.Series],
    min_weight_within: int = 1,
    min_weight_cross: int = 1,
) -> nx.Graph:
    """
    Build a unified network combining within-column AND cross-column co-occurrences.
    Each node carries a 'dimension' attribute for coloring.
    """
    G = nx.Graph()

    col_names = list(parsed_columns.keys())

    # Add within-column edges
    for col_name, parsed in parsed_columns.items():
        edges = build_within_cooccurrence(parsed, min_weight=min_weight_within)
        for _, row in edges.iterrows():
            if G.has_edge(row["source"], row["target"]):
                G[row["source"]][row["target"]]["weight"] += row["weight"]
            else:
                G.add_edge(row["source"], row["target"],
                           weight=row["weight"], edge_type="within")
            G.nodes[row["source"]]["dimension"] = col_name
            G.nodes[row["target"]]["dimension"] = col_name

    # Add cross-column edges
    for i, col_a in enumerate(col_names):
        for col_b in col_names[i + 1:]:
            edges = build_cross_column_edges(
                None, col_a, col_b,
                parsed_columns[col_a], parsed_columns[col_b],
                min_weight=min_weight_cross,
            )
            for _, row in edges.iterrows():
                if G.has_edge(row["source"], row["target"]):
                    G[row["source"]][row["target"]]["weight"] += row["weight"]
                else:
                    G.add_edge(row["source"], row["target"],
                               weight=row["weight"], edge_type="cross")
                G.nodes[row["source"]]["dimension"] = col_a
                G.nodes[row["target"]]["dimension"] = col_b

    # Ensure all nodes have dimension
    for n in G.nodes():
        if "dimension" not in G.nodes[n]:
            G.nodes[n]["dimension"] = "unknown"

    return G


# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------

def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute centrality metrics for network nodes."""
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    metrics = pd.DataFrame(index=list(G.nodes()))
    metrics.index.name = "category"

    metrics["degree"] = pd.Series(dict(G.degree()))
    metrics["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")))
    metrics["degree_centrality"] = pd.Series(nx.degree_centrality(G))

    if G.number_of_nodes() > 2:
        metrics["betweenness_centrality"] = pd.Series(
            nx.betweenness_centrality(G, weight="weight")
        )
        metrics["closeness_centrality"] = pd.Series(nx.closeness_centrality(G))
        try:
            metrics["eigenvector_centrality"] = pd.Series(
                nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
            )
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            metrics["eigenvector_centrality"] = np.nan
    else:
        metrics["betweenness_centrality"] = 0.0
        metrics["closeness_centrality"] = 0.0
        metrics["eigenvector_centrality"] = np.nan

    # Add dimension info if available
    dims = nx.get_node_attributes(G, "dimension")
    if dims:
        metrics["dimension"] = pd.Series(dims)

    return metrics.sort_values("weighted_degree", ascending=False)


def compute_global_metrics(G: nx.Graph) -> dict:
    """Compute graph-level summary statistics."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "avg_weighted_degree": np.mean([d for _, d in G.degree(weight="weight")]),
        "connected_components": nx.number_connected_components(G),
    }

    if G.number_of_nodes() > 2:
        stats["avg_clustering"] = nx.average_clustering(G, weight="weight")

    if nx.is_connected(G) and G.number_of_nodes() > 1:
        stats["diameter"] = nx.diameter(G)
        stats["avg_shortest_path"] = nx.average_shortest_path_length(G)
    elif G.number_of_nodes() > 1:
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > 1:
            H = G.subgraph(largest_cc).copy()
            stats["diameter_largest_cc"] = nx.diameter(H)
            stats["avg_shortest_path_largest_cc"] = nx.average_shortest_path_length(H)
            stats["largest_cc_nodes"] = H.number_of_nodes()

    return stats


# ---------------------------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------------------------

def plot_category_frequencies(
    freq_dict: dict[str, pd.Series],
    figsize: tuple = (16, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar charts showing frequency distribution for each classification dimension."""
    n_cols = len(freq_dict)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, facecolor="white")
    if n_cols == 1:
        axes = [axes]

    for ax, (col_name, freq) in zip(axes, freq_dict.items()):
        short = freq.index.map(lambda x: SHORT_LABELS.get(x, x))
        color = DIM_COLORS.get(col_name, "#457B9D")
        ax.barh(short[::-1], freq.values[::-1], color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_title(DIM_SHORT.get(col_name, col_name), fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        # Add count labels
        for i, v in enumerate(freq.values[::-1]):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_cross_heatmap(
    parsed_a: pd.Series,
    parsed_b: pd.Series,
    col_a: str,
    col_b: str,
    figsize: tuple = (10, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap showing co-occurrence counts between two classification dimensions.
    Rows = col_a categories, Columns = col_b categories.
    """
    pair_counter: Counter = Counter()
    for cats_a, cats_b in zip(parsed_a, parsed_b):
        for a in set(cats_a):
            for b in set(cats_b):
                pair_counter[(a, b)] += 1

    cats_a_all = sorted(set(a for (a, _) in pair_counter.keys()))
    cats_b_all = sorted(set(b for (_, b) in pair_counter.keys()))

    matrix = pd.DataFrame(0, index=cats_a_all, columns=cats_b_all)
    for (a, b), w in pair_counter.items():
        matrix.loc[a, b] = w

    # Use short labels
    matrix.index = [SHORT_LABELS.get(x, x) for x in matrix.index]
    matrix.columns = [SHORT_LABELS.get(x, x) for x in matrix.columns]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Co-occurrence count"},
    )
    dim_a = DIM_SHORT.get(col_a, col_a)
    dim_b = DIM_SHORT.get(col_b, col_b)
    ax.set_title(f"Cross-tabulation: {dim_a} × {dim_b}", fontsize=13, fontweight="bold")
    ax.set_ylabel(dim_a, fontsize=11)
    ax.set_xlabel(dim_b, fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_within_network(
    G: nx.Graph,
    col_name: str,
    node_freq: pd.Series,
    figsize: tuple = (9, 7),
    layout_seed: int = 42,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a within-column co-occurrence network.
    Node size ∝ frequency; edge width ∝ co-occurrence weight.
    """
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.text(0.5, 0.5, "No co-occurrences found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    palette = DIM_PALETTES.get(col_name, ["#457B9D"] * 10)
    pos = nx.spring_layout(G, k=2.0, seed=layout_seed, weight="weight", iterations=100)

    # Node sizes from frequency
    sizes = []
    for n in G.nodes():
        freq = node_freq.get(n, 1)
        sizes.append(300 + freq * 60)

    # Node colors
    node_list = list(G.nodes())
    colors = [palette[i % len(palette)] for i in range(len(node_list))]

    # Edge widths
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew = max(edge_weights) if edge_weights else 1
    widths = [0.5 + 4.0 * (w / max_ew) for w in edge_weights]

    # Draw
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35, edge_color="#888", ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_size=sizes,
        alpha=0.9, edgecolors="white", linewidths=1, ax=ax,
    )

    # Labels
    labels = {n: SHORT_LABELS.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    # Edge labels
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, font_color="#555", ax=ax)

    dim = DIM_SHORT.get(col_name, col_name)
    ax.set_title(f"Within-Dimension Co-occurrence: {dim}", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_unified_network(
    G: nx.Graph,
    node_metrics: pd.DataFrame,
    title: str = "Unified Methodological Network",
    figsize: tuple = (14, 11),
    node_scale: float = 35,
    edge_alpha: float = 0.25,
    font_size: int = 8,
    layout_seed: int = 42,
    save_path: Optional[str] = None,
) -> tuple:
    """
    Visualize the unified cross-dimensional methodological network.
    Node color encodes dimension; node size encodes weighted degree.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty network", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    # Layout
    k = 1.2 / np.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, k=k, seed=layout_seed, weight="weight", iterations=150, scale=1.0)

    # Node colors by dimension
    dims = nx.get_node_attributes(G, "dimension")
    node_colors = [DIM_COLORS.get(dims.get(n, ""), "#999999") for n in G.nodes()]

    # Node sizes
    w_deg = node_metrics.loc[list(G.nodes()), "weighted_degree"]
    max_wd = w_deg.max() if w_deg.max() > 0 else 1
    node_sizes = (w_deg / max_wd) ** 0.6 * (node_scale ** 1.5) + 80

    # Edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 3.0 * (w / max_ew) for w in edge_weights]
    edge_types = nx.get_edge_attributes(G, "edge_type")

    # Draw edges with different styles for within vs cross
    for (u, v), ew in zip(G.edges(), edge_widths):
        w = G[u][v]["weight"]
        etype = edge_types.get((u, v), edge_types.get((v, u), "cross"))
        ea = 0.08 + edge_alpha * (w / max_ew)
        style = "-" if etype == "within" else "--"
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#888", linewidth=ew, alpha=ea, linestyle=style, zorder=1)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.9, linewidths=0.5, edgecolors="white", ax=ax,
    )

    # Labels
    labels = {n: SHORT_LABELS.get(n, n) for n in G.nodes()}
    try:
        from adjustText import adjust_text
        texts = []
        for node, label in labels.items():
            x, y = pos[node]
            txt = ax.text(
                x, y, label, fontsize=font_size, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"),
            )
            texts.append(txt)
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4),
                    expand=(1.3, 1.5))
    except ImportError:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size,
                                font_weight="bold", ax=ax)

    # Legend for dimensions
    for dim_name, color in DIM_COLORS.items():
        dim_short = DIM_SHORT.get(dim_name, dim_name)
        ax.scatter([], [], c=[color], s=100, label=dim_short,
                   edgecolors="white", linewidths=0.5)
    # Edge type legend
    ax.plot([], [], "-", color="#888", linewidth=1.5, alpha=0.5, label="Within-dim edge")
    ax.plot([], [], "--", color="#888", linewidth=1.5, alpha=0.5, label="Cross-dim edge")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95,
              title="Dimensions", title_fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    ax.margins(0.05)
    plt.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig, ax


def plot_stacked_combinations(
    df: pd.DataFrame,
    parsed_columns: dict[str, pd.Series],
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked bar chart showing co-occurrence patterns per article.
    Each bar = one article, stacked segments = categories present.
    Useful to see methodological "profiles".
    """
    # Build a binary presence matrix
    all_cats = []
    cat_dim = {}
    for col_name, parsed in parsed_columns.items():
        freq = get_frequency(parsed)
        for cat in freq.index:
            all_cats.append(cat)
            cat_dim[cat] = col_name

    presence = pd.DataFrame(0, index=df.index, columns=all_cats)
    for col_name, parsed in parsed_columns.items():
        for idx, cats in parsed.items():
            for c in cats:
                if c in presence.columns:
                    presence.loc[idx, c] = 1

    # Count combination patterns
    combo_counts = presence.apply(lambda row: " + ".join(
        [SHORT_LABELS.get(c, c) for c in presence.columns if row[c] == 1]
    ), axis=1).value_counts().head(20)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    combo_counts[::-1].plot.barh(ax=ax, color="#457B9D", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Articles", fontsize=11)
    ax.set_title("Top 20 Methodological Profiles (Category Combinations)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
