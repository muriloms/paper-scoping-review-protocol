"""
null_model.py
=============
Statistical significance testing for modularity scores using
null models (configuration model / degree-preserving randomization).

Implements the methodology recommended by:
- Guimerà, R., Sales-Pardo, M., & Amaral, L.A.N. (2004).
  Modularity and community structure in complex networks.
  Physical Review E, 70(2), 025101.
- Lancichinetti, A., & Fortunato, S. (2012).
  Consensus clustering in complex networks.
  Scientific Reports, 2, 336.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def generate_configuration_model(G: nx.Graph, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a random graph using the configuration model,
    preserving the degree sequence of G.
    Removes self-loops and multi-edges to produce a simple graph.
    """
    degree_sequence = [d for _, d in G.degree()]
    G_random = nx.configuration_model(degree_sequence, seed=seed)
    G_random = nx.Graph(G_random)  # Remove multi-edges
    G_random.remove_edges_from(nx.selfloop_edges(G_random))  # Remove self-loops
    return G_random


def generate_weighted_null_model(
    G: nx.Graph,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a weighted null model by:
    1. Creating a configuration model preserving degree sequence
    2. Redistributing original edge weights randomly across new edges
    """
    rng = np.random.default_rng(seed)

    # Generate topology
    G_random = generate_configuration_model(G, seed=seed)

    # Redistribute weights
    original_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    rng.shuffle(original_weights)

    new_edges = list(G_random.edges())
    n_new = len(new_edges)
    n_orig = len(original_weights)

    # If different number of edges, resample weights
    if n_new != n_orig:
        weights = rng.choice(original_weights, size=n_new, replace=True)
    else:
        weights = original_weights

    for (u, v), w in zip(new_edges, weights):
        G_random[u][v]["weight"] = w

    return G_random


def modularity_significance_test(
    G: nx.Graph,
    partition: dict,
    n_random: int = 1000,
    resolution: float = 1.0,
    weighted: bool = True,
    seed: int = 42,
) -> dict:
    """
    Test the statistical significance of a modularity score against
    a null ensemble of configuration-model random networks.

    Parameters
    ----------
    G : nx.Graph
        The observed network.
    partition : dict
        The observed community partition {node: community_id}.
    n_random : int
        Number of random networks to generate (default: 1000).
    resolution : float
        Resolution parameter for Louvain (same as used on observed network).
    weighted : bool
        If True, use weighted null model; otherwise, unweighted.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - Q_observed: observed modularity
        - Q_random_mean: mean modularity of null ensemble
        - Q_random_std: std of null ensemble
        - Q_random_values: array of all null Q values
        - z_score: (Q_obs - Q_mean) / Q_std
        - p_value: proportion of null Q values >= Q_observed
        - n_random: number of random networks generated
        - significant: bool (p < 0.05)
    """
    # Compute observed modularity
    communities_sets = [
        {n for n, c in partition.items() if c == cid}
        for cid in sorted(set(partition.values()))
    ]
    Q_observed = nx.community.modularity(G, communities_sets)

    # Generate null ensemble
    Q_random_values = np.zeros(n_random)
    rng_seeds = np.random.SeedSequence(seed).spawn(n_random)

    for i in range(n_random):
        s = rng_seeds[i].generate_state(1)[0]

        if weighted:
            G_rand = generate_weighted_null_model(G, seed=int(s % (2**31)))
        else:
            G_rand = generate_configuration_model(G, seed=int(s % (2**31)))

        # Detect communities on random network
        try:
            part_rand = community_louvain.best_partition(
                G_rand, weight="weight" if weighted else None,
                resolution=resolution, random_state=int(s % (2**31)),
            )
            comm_sets_rand = [
                {n for n, c in part_rand.items() if c == cid}
                for cid in sorted(set(part_rand.values()))
            ]
            Q_random_values[i] = nx.community.modularity(G_rand, comm_sets_rand)
        except Exception:
            Q_random_values[i] = 0.0

    # Statistics
    Q_mean = np.mean(Q_random_values)
    Q_std = np.std(Q_random_values)
    z_score = (Q_observed - Q_mean) / Q_std if Q_std > 0 else np.inf
    p_value = np.mean(Q_random_values >= Q_observed)

    return {
        "Q_observed": Q_observed,
        "Q_random_mean": Q_mean,
        "Q_random_std": Q_std,
        "Q_random_values": Q_random_values,
        "z_score": z_score,
        "p_value": p_value,
        "n_random": n_random,
        "significant": p_value < 0.05,
    }


def plot_modularity_significance(
    results: dict,
    title: str = "Modularity Significance Test",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the null distribution of modularity scores with the observed Q.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    Q_obs = results["Q_observed"]
    Q_rand = results["Q_random_values"]
    Q_mean = results["Q_random_mean"]
    Q_std = results["Q_random_std"]
    z = results["z_score"]
    p = results["p_value"]
    n = results["n_random"]

    # Histogram of null distribution
    ax.hist(
        Q_rand, bins=40, color="#A8DADC", edgecolor="white",
        alpha=0.85, density=True, label=f"Null model (n={n})",
    )

    # Null mean
    ax.axvline(
        Q_mean, color="#457B9D", linestyle="--", linewidth=1.5,
        label=f"Null mean: {Q_mean:.4f} ± {Q_std:.4f}",
    )

    # Observed Q
    ax.axvline(
        Q_obs, color="#E63946", linestyle="-", linewidth=2.5,
        label=f"Observed Q: {Q_obs:.4f}",
    )

    # Shade p-value region
    ax.axvspan(
        Q_obs, ax.get_xlim()[1] if Q_obs < max(Q_rand) * 1.2 else Q_obs * 1.3,
        alpha=0.1, color="#E63946",
    )

    # Annotation
    sig_label = "significant" if results["significant"] else "not significant"
    ax.text(
        0.97, 0.95,
        f"z = {z:.2f}\np = {p:.4f}\n({sig_label} at α = 0.05)",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#888", alpha=0.9),
    )

    ax.set_xlabel("Modularity (Q)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    return fig


def format_significance_report(results: dict) -> str:
    """
    Generate a formatted text report of the significance test,
    suitable for inclusion in a manuscript.
    """
    Q_obs = results["Q_observed"]
    Q_mean = results["Q_random_mean"]
    Q_std = results["Q_random_std"]
    z = results["z_score"]
    p = results["p_value"]
    n = results["n_random"]

    if p < 0.001:
        p_str = "p < 0.001"
    elif p < 0.01:
        p_str = f"p = {p:.3f}"
    elif p < 0.05:
        p_str = f"p = {p:.3f}"
    else:
        p_str = f"p = {p:.3f}"

    report = (
        f"Modularity Significance Test (Configuration Model)\n"
        f"{'='*55}\n"
        f"  Observed modularity (Q):     {Q_obs:.4f}\n"
        f"  Null model mean (μ):         {Q_mean:.4f}\n"
        f"  Null model std (σ):          {Q_std:.4f}\n"
        f"  Z-score:                     {z:.2f}\n"
        f"  P-value:                     {p_str}\n"
        f"  Random networks generated:   {n}\n"
        f"  Significant (α=0.05):        {results['significant']}\n"
        f"\n"
        f"  Interpretation: The observed modularity is {z:.1f} standard\n"
        f"  deviations above the null expectation, indicating that the\n"
        f"  community structure is {'statistically significant' if results['significant'] else 'not statistically significant'}\n"
        f"  despite {'a low' if Q_obs < 0.3 else 'a moderate'} absolute Q value.\n"
    )
    return report
