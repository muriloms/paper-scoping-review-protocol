# Scoping Review Protocol: Guidelines for the Selection of Parameters and Probability Distributions in Agent-Based Models in the Social Sciences

## Bibliometric Network Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/package%20manager-uv-blueviolet)](https://docs.astral.sh/uv/)

This repository contains the computational analysis pipeline for a **scoping review** following the JBI methodology and PRISMA-ScR guidelines. The review synthesizes the state of the art regarding methodological choices in social science simulations — specifically how researchers select and justify parameters and probability distributions in Agent-Based Models (ABMs) and Multi-Agent Systems (MAS).

**Databases:** SCOPUS, Web of Science, IEEE Xplore | **Period:** 2015–2025 | **Language:** English

---

## Project Structure

```
project/
├── data/
│   └── [ScReview] Articles.xlsx      # Exported dataset from Rayyan
├── notebooks/
│   ├── 01_keyword_cooccurrence_network.ipynb
│   ├── 02_methodological_classification_network.ipynb
│   ├── figures/                       # Outputs from Notebook 01
│   └── figures_meth/                  # Outputs from Notebook 02
├── src/
│   ├── __init__.py
│   ├── network_analysis.py            # Module for keyword co-occurrence analysis
│   └── methodological_network.py      # Module for methodological classification analysis
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Install `uv` (Python package manager)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust. It replaces `pip`, `venv`, and `pip-tools` with a single, faster tool.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Homebrew (macOS):**

```bash
brew install uv
```

After installation, verify:

```bash
uv --version
```

### 2. Clone or download this repository

```bash
cd /path/to/your/workspace
# If using git:
git clone <repository-url>
cd project
```

### 3. Create virtual environment and install dependencies

```bash
# Create a virtual environment with Python 3.11+
uv venv --python 3.11

# Activate it
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt
```

### 4. Place your data file

Copy your exported dataset into the `data/` directory:

```
data/[ScReview] Articles.xlsx
```

The Excel file should contain, at minimum, the following columns:

| Column | Description |
|--------|-------------|
| `Key` | Unique article identifier |
| `Author` | Author list |
| `Title` | Article title |
| `Publication Year` | Year of publication |
| `Publication Title` | Journal name |
| `Manual Tags` | Author keywords (semicolon-separated) |
| `Data Source (Parameterization)` | Expert classification: data source type (comma-separated) |
| `Model Objective (optional)` | Expert classification: model objective (comma-separated) |
| `Sensitivity Analysis` | Expert classification: SA method used (comma-separated) |

### 5. Launch Jupyter

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory.

---

## Notebooks

### Notebook 01 — Keyword Co-occurrence Network (KCN)

**File:** `notebooks/01_keyword_cooccurrence_network.ipynb`
**Module:** `src/network_analysis.py`

**Purpose:** Map the conceptual landscape and identify thematic clusters within the reviewed literature through a bibliometric network analysis based on author keywords.

#### Sections

| # | Section | Description |
|---|---------|-------------|
| 0 | Setup & Imports | Load libraries and configure paths |
| 1 | Data Loading | Import the Excel dataset |
| 2 | Exploratory Data Analysis | Publication trend over time, top journals, keyword frequency analysis. Includes **keyword harmonization** — a controlled vocabulary mapping that consolidates morphological variants (e.g., *agent-based model*, *agent-based modelling*, *ABM* → *agent-based modeling*) into canonical terms to avoid inflated node counts |
| 3 | Network Construction | Build the Keyword Co-occurrence Network (KCN) as an undirected weighted graph where nodes are keywords and edge weights represent the number of articles in which two keywords co-occur. A configurable minimum co-occurrence threshold filters noise |
| 4 | Network Metrics | **Global metrics:** density, average degree, average clustering coefficient, connected components, diameter. **Node-level centrality:** degree, weighted degree, betweenness, closeness, eigenvector, clustering coefficient. Includes centrality comparison bar charts and degree distribution (histogram + log-log) |
| 5 | Community Detection | Louvain algorithm for modularity-based community detection. Reports modularity score with interpretation thresholds (Q > 0.3 significant, > 0.5 strong, > 0.7 very strong). Community characterization table with top keywords per cluster. Inter-community co-occurrence heatmap |
| 6 | Network Visualization | Force-directed (Fruchterman-Reingold) layout with compact parameterization optimized for publication. Node size ∝ weighted degree, node color = community, edge thickness ∝ co-occurrence weight, edge transparency ∝ weight |
| 7 | Export | CSV files (node metrics, edge list, community summary, global metrics) and GEXF file for interactive exploration in Gephi |
| 8 | Summary Table | Consolidated statistics table ready for the manuscript results section |

#### Key Outputs

| File | Description |
|------|-------------|
| `figures/kcn_network.png` | Main network visualization (300 DPI) |
| `figures/kcn_network.gexf` | Graph file for Gephi |
| `figures/node_metrics.csv` | Per-keyword centrality metrics + community ID |
| `figures/edge_list.csv` | Weighted co-occurrence edge list |
| `figures/community_summary.csv` | Community characterization (size, top keywords) |
| `figures/global_metrics.csv` | Graph-level summary statistics |
| `figures/centrality_comparison.png` | Top-20 keywords across 4 centrality measures |
| `figures/degree_distribution.png` | Degree distribution (linear + log-log) |
| `figures/community_heatmap.png` | Inter-community co-occurrence strength |
| `figures/publication_trend.png` | Articles per year |
| `figures/top_journals.png` | Top 15 publication venues |
| `figures/top30_keywords.png` | 30 most frequent keywords (post-harmonization) |

---

### Notebook 02 — Methodological Classification Network

**File:** `notebooks/02_methodological_classification_network.ipynb`
**Module:** `src/methodological_network.py`

**Purpose:** Analyze co-occurrence patterns among the methodological choices made by researchers when designing ABMs, based on expert classifications across three dimensions: data source for parameterization, model objective, and sensitivity analysis method.

#### Classification Dimensions

**Data Source (DS)** — How model parameters were grounded:

| Category | Description |
|----------|-------------|
| Theoretical (literature) | Parameters derived from theoretical frameworks or prior literature |
| Empirical (micro) | Parameters from micro-level empirical data (surveys, experiments, individual records) |
| Empirical (macro) | Parameters from macro-level empirical data (census, aggregate statistics) |
| Expert-based | Parameters from qualitative methods (interviews, participatory modeling, Delphi) |
| Hybrid | Combination of multiple data source types |
| Synthetic (ML/AI) | Parameters generated via machine learning, AI, or other computational methods |

**Model Objective (MO)** — Intended purpose of the simulation:

| Category | Description |
|----------|-------------|
| Exploratory | Exploring emergent phenomena and "what-if" scenarios |
| Explanatory | Identifying causal mechanisms underlying observed patterns |
| Predictive | Forecasting future states or quantitative outcomes |
| Descriptive / Phenomena-based | Reproducing observed phenomena without causal claims |
| Decision Support / Policy Evaluation | Informing policy decisions or evaluating interventions |
| Theoretical / Illustrative | Demonstrating theoretical concepts or proofs of concept |
| Communication / Education | Teaching or communicating complex systems to non-specialists |

**Sensitivity Analysis (SA)** — Method for assessing parameter uncertainty:

| Category | Description |
|----------|-------------|
| Local (OAT/OFAT) | One-at-a-time / one-factor-at-a-time perturbation |
| Global (Variance-based) | Sobol indices, ANOVA decomposition, variance-based methods |
| Global (Screening) | Morris method, elementary effects, factor prioritization |
| Metamodeling / Surrogate | Surrogate models (Gaussian processes, polynomial chaos, etc.) |
| Exploratory / Scenario-based | Systematic scenario exploration without formal SA framework |
| Regression / Statistical Inference | Regression analysis or statistical tests on simulation outputs |
| NA | No sensitivity analysis reported |

#### Sections

| # | Section | Description |
|---|---------|-------------|
| 0 | Setup & Imports | Load libraries and configure paths |
| 1 | Data Loading | Import dataset and identify the three classification columns |
| 2 | Exploratory Data Analysis | Parse multi-value cells, category frequency distributions with visual bars, multi-label statistics (how many articles combine multiple categories per dimension) |
| 3 | Within-Dimension Co-occurrence Networks | Three separate networks showing which categories are combined *within the same dimension* (e.g., Theoretical + Empirical macro as data sources in the same article). Reveals complementary vs. substitutive methodological choices |
| 4 | Cross-Dimension Heatmaps | Three cross-tabulation heatmaps (DS×MO, DS×SA, MO×SA) showing how choices in one dimension relate to choices in another. Reveals dominant methodological archetypes and unexplored combinations |
| 5 | Unified Methodological Network | A single graph integrating all three dimensions with within-dimension edges (solid) and cross-dimension edges (dashed). Node color encodes dimension, node size encodes weighted degree. Centrality analysis identifies the most structurally important methodological choices |
| 6 | Network Visualization | Force-directed layout of the unified network with full visual encoding (color, size, line style, transparency) |
| 7 | Methodological Profiles | Top 20 most frequent complete profiles (full combination of categories across all three dimensions). Reveals dominant archetypes of ABM design |
| 8 | Temporal Trends | Stacked area charts showing the evolution of each dimension's category proportions over the 2015–2025 period |
| 9 | Export | CSV files, GEXF for Gephi |
| 10 | Summary Table | Consolidated statistics for the manuscript |

#### Key Outputs

| File | Description |
|------|-------------|
| `figures_meth/unified_network.png` | Main unified network visualization (300 DPI) |
| `figures_meth/unified_methodological_network.gexf` | GEXF file for Gephi |
| `figures_meth/unified_node_metrics.csv` | Centrality metrics for all categories |
| `figures_meth/unified_centrality.png` | Top-12 categories across 3 centrality measures |
| `figures_meth/category_frequencies.png` | Frequency distribution per dimension |
| `figures_meth/heatmap_DS_x_MO.png` | Data Source × Model Objective cross-tabulation |
| `figures_meth/heatmap_DS_x_SA.png` | Data Source × Sensitivity Analysis cross-tabulation |
| `figures_meth/heatmap_MO_x_SA.png` | Model Objective × Sensitivity Analysis cross-tabulation |
| `figures_meth/within_network_DS.png` | Within-dimension network: Data Source |
| `figures_meth/within_network_MO.png` | Within-dimension network: Model Objective |
| `figures_meth/within_network_SA.png` | Within-dimension network: Sensitivity Analysis |
| `figures_meth/methodological_profiles.png` | Top 20 methodological profile combinations |
| `figures_meth/temporal_trends.png` | Temporal evolution of category proportions |

---

## Source Modules

### `src/network_analysis.py`

Reusable functions for keyword co-occurrence analysis:

- `parse_keywords()` — Tokenize, clean, and harmonize keywords using a configurable thesaurus (`DEFAULT_KEYWORD_MAP`)
- `build_cooccurrence_edges()` — Count pairwise co-occurrences from keyword lists
- `build_network()` — Construct a NetworkX weighted graph
- `compute_node_metrics()` — Degree, weighted degree, betweenness, closeness, eigenvector, clustering coefficient
- `compute_global_metrics()` — Density, average degree, clustering, components, diameter
- `detect_communities()` — Louvain modularity-based community detection
- `plot_network()` — Publication-ready network visualization with compact layout
- `plot_degree_distribution()` — Histogram + log-log degree distribution
- `plot_centrality_comparison()` — Multi-panel centrality bar charts
- `plot_community_heatmap()` — Inter-community co-occurrence heatmap
- `summarize_communities()` — Community characterization table

### `src/methodological_network.py`

Reusable functions for methodological classification analysis:

- `parse_classification_column()` — Parse multi-value classification cells
- `build_within_cooccurrence()` — Within-dimension co-occurrence edges
- `build_cross_column_edges()` — Cross-dimension co-occurrence edges
- `build_unified_methodological_network()` — Unified graph combining all dimensions
- `plot_category_frequencies()` — Multi-panel frequency bar charts
- `plot_cross_heatmap()` — Cross-tabulation heatmap between two dimensions
- `plot_within_network()` — Within-dimension network visualization
- `plot_unified_network()` — Unified network with dimensional encoding
- `plot_stacked_combinations()` — Top methodological profiles bar chart

---

## Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 2.0 | Data manipulation |
| openpyxl | ≥ 3.1 | Excel file reading |
| networkx | ≥ 3.1 | Graph construction and analysis |
| python-louvain | ≥ 0.16 | Louvain community detection |
| matplotlib | ≥ 3.7 | Visualization |
| seaborn | ≥ 0.12 | Statistical visualization |
| numpy | ≥ 1.24 | Numerical computation |
| jupyter | ≥ 1.0 | Notebook environment |
| ipykernel | ≥ 6.25 | Jupyter kernel |
| adjustText | ≥ 1.1 | Non-overlapping label placement |

---

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Setup environment
uv venv --python 3.11
source .venv/bin/activate          # macOS/Linux
uv pip install -r requirements.txt

# 3. Place your data
cp /path/to/your/[ScReview]\ Articles.xlsx data/

# 4. Run notebooks
jupyter notebook notebooks/
```

---

## Customization

### Keyword Harmonization

To add or modify keyword mappings, edit `DEFAULT_KEYWORD_MAP` in `src/network_analysis.py`:

```python
DEFAULT_KEYWORD_MAP = {
    "agent-based model": "agent-based modeling",
    "your-variant": "canonical-form",
    # ...
}
```

### Co-occurrence Threshold

Adjust `MIN_COOCCURRENCE` in Notebook 01 (cell 3.1) to control network filtering:
- `MIN_COOCCURRENCE = 1` → full network (noisy, all connections)
- `MIN_COOCCURRENCE = 2` → recommended (filters rare, uninformative pairs)
- `MIN_COOCCURRENCE = 3` → conservative (only well-established connections)

### Classification Categories

To modify the expected categories for the methodological dimensions, edit `CATEGORY_DEFINITIONS` in `src/methodological_network.py`.

---

## Citation

If you use this analysis pipeline in your research, please cite:

```
@article{scoping_review_abm_2025,
  title={Scoping Review Protocol: Guidelines for the Selection of Parameters 
         and Probability Distributions in Agent-Based Models in the Social Sciences},
  year={2025}
}
```

---

## License

This project is provided for academic research purposes.