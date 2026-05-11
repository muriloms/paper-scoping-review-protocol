"""
REPLACEMENT CELLS FOR NOTEBOOK 01 — SECTION 5
==============================================
Replace the existing Section 5 cells with these.
Copy each block between the #### markers into a separate cell.
"""

# ========================================================================
# MARKDOWN CELL — Section 5 header (replace existing)
# ========================================================================
"""
## 5. Community Detection (Louvain Algorithm)

### Theoretical Foundation

**Community detection** identifies groups of densely interconnected nodes that are sparsely
connected to the rest of the network. In bibliometric networks, communities correspond to
**thematic clusters** — groups of keywords that tend to appear together in the same articles.

The **Louvain algorithm** (Blondel et al., 2008) is a greedy modularity optimization method
that iteratively assigns nodes to communities to maximize the **modularity** $Q$:

$$Q = \\frac{1}{2m} \\sum_{ij} \\left[ A_{ij} - \\frac{k_i k_j}{2m} \\right] \\delta(c_i, c_j)$$

where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total
number of edges, and $\\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ belong to the same community.

**Important:** An absolute $Q$ value alone is insufficient to assess the significance of
community structure. Following best practices in network science (Guimerà et al., 2004),
we compare the observed $Q$ against a **null ensemble** of random networks generated via the
**configuration model**, which preserves the degree sequence but randomizes edge placement.
The z-score $z = (Q_{obs} - \\mu_{null}) / \\sigma_{null}$ and an empirical p-value
quantify whether the detected structure is statistically significant.
"""

# ========================================================================
# CODE CELL — 5.1 Detect communities
# ========================================================================

# --- 5.1 Detect communities ---
partition = detect_communities(G, resolution=1.0)
n_communities = len(set(partition.values()))

# Compute modularity
communities_sets = [{n for n, c in partition.items() if c == cid}
                    for cid in sorted(set(partition.values()))]
modularity = nx.community.modularity(G, communities_sets)

print(f"Number of communities detected: {n_communities}")
print(f"Modularity (Q): {modularity:.4f}")


# ========================================================================
# MARKDOWN CELL — 5.2 Null model test header
# ========================================================================
"""
### 5.2 Modularity Significance Test (Configuration Model)

To assess whether the observed modularity reflects genuine community structure rather than
an artifact of the degree distribution, we compare $Q_{obs}$ against an ensemble of **1,000
random networks** generated using the **configuration model** (Newman, 2003). Each random
network preserves the original degree sequence but randomizes edge connections, thereby
serving as a null hypothesis of "no community structure beyond what is expected from the
degree distribution alone."

For each random network, the Louvain algorithm is applied with the same resolution parameter,
and the resulting modularity $Q_{null}$ is recorded. The statistical significance is evaluated
through:
- **Z-score:** $z = (Q_{obs} - \\mu_{null}) / \\sigma_{null}$
- **Empirical p-value:** proportion of null networks with $Q_{null} \\geq Q_{obs}$
"""

# ========================================================================
# CODE CELL — 5.2 Run null model test
# ========================================================================

from src.null_model import (
    modularity_significance_test,
    plot_modularity_significance,
    format_significance_report,
)

# Run significance test (1000 configuration-model random networks)
print("Running modularity significance test (1,000 random networks)...")
print("This may take a few minutes...\n")

sig_results = modularity_significance_test(
    G, partition,
    n_random=1000,
    resolution=1.0,
    weighted=True,
    seed=42,
)

# Print report
print(format_significance_report(sig_results))

# ========================================================================
# CODE CELL — 5.3 Plot null distribution
# ========================================================================

fig_sig = plot_modularity_significance(
    sig_results,
    title="Modularity Significance Test — KCN (Configuration Model Null)",
    figsize=(10, 5),
    save_path=os.path.join(FIG_DIR, "modularity_significance_test.png"),
)
plt.show()

# ========================================================================
# MARKDOWN CELL — 5.3 Interpretation
# ========================================================================
"""
### 5.3 Interpretation

The significance test compares the observed modularity against the null expectation
from degree-preserving random networks. Key considerations:

- A **z-score > 2** (approximately p < 0.05) indicates that the community structure is
  statistically significant — the observed clustering is unlikely to arise from the
  degree distribution alone.
- In **small, dense co-occurrence networks**, absolute modularity values tend to be low
  because many keywords co-occur broadly. The null model comparison contextualizes this:
  even a "low" Q can be highly significant if it substantially exceeds the null expectation.
- Conversely, if Q is not significantly above the null, the detected communities may
  reflect degree heterogeneity rather than genuine thematic structure.
"""

# ========================================================================
# CODE CELL — 5.4 Community summary (same as before)
# ========================================================================

comm_summary = summarize_communities(partition, node_metrics, top_n=7)
comm_summary

# ========================================================================
# CODE CELL — 5.5 Community sizes (same as before)
# ========================================================================

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
comm_summary_sorted = comm_summary.sort_values("size", ascending=False)
ax.bar(
    comm_summary_sorted["community"].astype(str),
    comm_summary_sorted["size"],
    color="#457B9D", edgecolor="white"
)
ax.set_xlabel("Community", fontsize=12)
ax.set_ylabel("Number of Keywords", fontsize=12)
ax.set_title("Community Sizes (Louvain)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "community_sizes.png"), dpi=300, bbox_inches="tight")
plt.show()

# ========================================================================
# MARKDOWN CELL — 5.5 heatmap header
# ========================================================================
"""
### 5.5 Inter-Community Heatmap

The **inter-community heatmap** shows the total co-occurrence weight between every pair
of communities. Diagonal cells represent **intra-community** co-occurrence (internal cohesion),
while off-diagonal cells represent **inter-community** co-occurrence (thematic bridges).
"""

# ========================================================================
# CODE CELL — 5.6 Heatmap (same as before)
# ========================================================================

fig_hm = plot_community_heatmap(
    G, partition,
    save_path=os.path.join(FIG_DIR, "community_heatmap.png")
)
plt.show()

# ========================================================================
# CODE CELL — 5.7 Export significance results
# ========================================================================

# Export significance test results
sig_export = {
    "Q_observed": sig_results["Q_observed"],
    "Q_null_mean": sig_results["Q_random_mean"],
    "Q_null_std": sig_results["Q_random_std"],
    "z_score": sig_results["z_score"],
    "p_value": sig_results["p_value"],
    "n_random": sig_results["n_random"],
    "significant_alpha_005": sig_results["significant"],
}
pd.DataFrame([sig_export]).T.rename(columns={0: "value"}).to_csv(
    os.path.join(FIG_DIR, "modularity_significance.csv")
)
print("Significance test results exported to: modularity_significance.csv")

# ========================================================================
# Update SUMMARY TABLE (Section 8) — add these rows:
# ========================================================================
"""
Add to summary_table dict:

    "Modularity (Q)": f"{modularity:.4f}",
    "Null model Q (mean ± std)": f"{sig_results['Q_random_mean']:.4f} ± {sig_results['Q_random_std']:.4f}",
    "Z-score": f"{sig_results['z_score']:.2f}",
    "P-value": f"{sig_results['p_value']:.4f}" if sig_results['p_value'] >= 0.001 else "< 0.001",
    "Significant (α=0.05)": str(sig_results['significant']),
"""
