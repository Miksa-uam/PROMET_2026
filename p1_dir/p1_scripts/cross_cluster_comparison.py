r"""
================================================================================
CROSS-CLUSTER STABILITY & PHENOTYPE MAPPING ENGINE
================================================================================

Research Context & Scientific Task:
-----------------------------------
In translational clinical phenotyping, identifying optimal patient clusters requires
rigorous post hoc evaluation across different methodological choices (algorithms,
distance weighting schemes) and resolution levels (cluster count k).

This module provides two distinct visual and analytical evaluation frameworks:

1. Within-k Concordance Evaluation (Methodological Sensitivity):
   - Computes pairwise Adjusted Rand Index (ARI) matrices across all algorithms
     (e.g., PAM, Complete Linkage) and distance block weightings (e.g., 0.50/0.50,
     0.60/0.40, 0.67/0.33, 0.70/0.30) at a fixed k.
   - Outputs a publication-ready correlation-style heatmap for each k explored.
   - Assesses whether the discovered clinical phenotypes are robust to algorithmic
     heuristics and weighting parameters.

2. Cross-k Stability & Phenotype Splitting Evaluation:
   - Macro View: An all-by-all cross-solution agreement grid across a
     pre-selected set of candidate methods and k resolutions, highlighting overall
     partition preservation.
   - Micro View: A patient flow alluvial diagram tracing cluster
     heritage and patient transitions across increasing k values, revealing which
     specific phenotypes remain intact as "core" clusters versus which phenotypes
     cleanly divide into sub-phenotypes.

Mathematical Consideration (Permutation Invariance):
----------------------------------------------------
The Adjusted Rand Index (ARI) evaluates patient-pair concordance rather than cluster ID
numbers. Thus, all analyses are strictly permutation-invariant and do not require
arbitrary label rematching across solutions.

Inputs & Outputs:
-----------------
- Input: SQLite database containing patient-level cluster assignments
         ('jaccard_gower_cluster_labels') and performance metrics.
- Outputs:
  * High-resolution annotated heatmaps (PNG, 300 DPI) for within-k and cross-k macro views.
  * Publication-grade patient flow alluvial diagrams (PNG, 300 DPI) for micro views.
  * Structured pandas DataFrames containing the pairwise ARI concordance matrices.

Pseudocode:
-----------
1. Load patient cluster label assignments from SQLite database.
2. Functionality 1: Within-k Concordance Heatmaps
     For each k in specified k_values:
       a. Identify all valid cluster label columns matching k.
       b. Compute symmetric pairwise Adjusted Rand Index (ARI) matrix.
       c. Render annotated heatmap with standardized human-readable method labels.
       d. Save figure (overwriting safely) and return matrix.
3. Functionality 2: Cross-k Macro & Micro Evaluation
     Macro View:
       a. Identify selected candidate solution columns across methods and k.
       b. Compute all-by-all pairwise ARI matrix.
       c. Render annotated macro grid with visual block delimiters and save figure.
     Micro View:
       a. Extract sequential cluster solutions (e.g. k=4 -> 5 -> 6 -> 7) for candidate method.
       b. Calculate patient transition contingency matrices between consecutive steps.
       c. Compute vertically aligned cluster blocks and smooth cubic Bézier flow ribbons.
       d. Render alluvial diagram with patient counts and percentages, and save figure.

Author: Gemini 3.8 Flash High in Antigravity
Date: October 2026
Target Environment: Python 3.10+ / SQLite3 / NumPy / pandas / matplotlib / seaborn / scikit-learn
================================================================================
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import adjusted_rand_score

# Ensure standard UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# DATA LOADING & STRING PARSING HELPERS
# =============================================================================

def load_cluster_labels_from_db(
    db_path: str,
    table_name: str = "jaccard_gower_cluster_labels",) -> pd.DataFrame:
    """
    Loads the unified cluster labels table from SQLite and validates patient identifiers.

    Parameters:
    -----------
    db_path : str
        Path to SQLite database containing the cluster labels table.
    table_name : str
        Name of the cluster assignments table (default: 'jaccard_gower_cluster_labels').

    Returns:
    --------
    pd.DataFrame:
        DataFrame containing patient identifiers and cluster assignment columns.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at path: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    print(f"[Data Loading] Loaded {len(df)} patients and {len(df.columns)} columns from '{table_name}'.")
    return df


def parse_solution_column_name(col_name: str) -> Dict[str, Union[str, int]]:
    """
    Deconstructs a standardized cluster label column name into its constituent metadata.
    Example: 'hybrid_wgc067_eb033_pam_k5' ->
             {'distance_table': 'hybrid_wgc067_eb033', 'algorithm': 'pam', 'k': 5,
              'weight_str': '0.67/0.33', 'algo_title': 'PAM'}
    """
    parts = col_name.split("_")
    # Extract k from the last component
    k_val = -1
    for p in reversed(parts):
        if p.startswith("k") and p[1:].isdigit():
            k_val = int(p[1:])
            break

    # Extract algorithm
    algo = "unknown"
    if "pam" in parts:
        algo = "pam"
    elif "hclust_complete" in col_name:
        algo = "hclust_complete"
    elif "hclust_average" in col_name:
        algo = "hclust_average"

    # Extract distance weighting
    weight_str = "Custom"
    if "wgc05_eb05" in col_name:
        weight_str = "0.50/0.50"
    elif "wgc06_eb04" in col_name:
        weight_str = "0.60/0.40"
    elif "wgc067_eb033" in col_name:
        weight_str = "0.67/0.33"
    elif "wgc07_eb03" in col_name:
        weight_str = "0.70/0.30"

    algo_titles = {
        "pam": "PAM",
        "hclust_complete": "Complete Linkage",
        "hclust_average": "Average Linkage",
    }
    algo_title = algo_titles.get(algo, algo.title())

    return {
        "column": col_name,
        "algorithm": algo,
        "algo_title": algo_title,
        "weight_str": weight_str,
        "k": k_val,
    }


def format_solution_label(col_name: str, include_k: bool = True) -> str:
    """
    Formats a raw database column name into a clean, publication-ready human-readable string.
    Example:
      'hybrid_wgc067_eb033_pam_k5' (include_k=False) -> 'PAM (0.67/0.33)'
      'hybrid_wgc067_eb033_pam_k5' (include_k=True)  -> 'PAM (0.67/0.33) k=5'
    """
    meta = parse_solution_column_name(col_name)
    base_label = f"{meta['algo_title']} ({meta['weight_str']})"
    if include_k and meta["k"] > 0:
        return f"{base_label} k={meta['k']}"
    return base_label


def safe_save_figure(fig: plt.Figure, save_path: str, dpi: int = 300) -> None:
    """
    Safely saves a Matplotlib figure, explicitly overriding file locks and permissions on Windows.
    """
    out_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(save_path):
        try:
            os.chmod(save_path, 0o777)
            os.remove(save_path)
        except Exception:
            pass
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[Plotting] Figure saved (overwriting existing) to: {save_path}")


# =============================================================================
# FUNCTIONALITY 1: WITHIN-k CROSS-METHOD CONCORDANCE HEATMAPS
# =============================================================================

def compute_within_k_ari_matrix(
    labels_df: pd.DataFrame,
    k: int,
    algorithms: Optional[List[str]] = None,
    distance_tables: Optional[List[str]] = None,) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extracts all method columns for a specific k and computes the pairwise ARI matrix.

    Parameters:
    -----------
    labels_df : pd.DataFrame
        DataFrame of patient cluster assignments.
    k : int
        The cluster count to evaluate.
    algorithms : Optional[List[str]]
        Filter for algorithms (e.g. ['pam', 'hclust_complete']).
    distance_tables : Optional[List[str]]
        Filter for distance tables (e.g. ['hybrid_wgc05_eb05', ...]).

    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]:
        - Symmetric pairwise ARI matrix indexed and columned with clean method names.
        - List of the original raw column names.
    """
    candidate_cols = [
        c for c in labels_df.columns
        if not "_medoid_" in c
        and c.endswith(f"_k{k}")
        and c not in ["patient_id", "medical_record_id"]
    ]

    # Filter by algorithm if requested
    if algorithms:
        candidate_cols = [
            c for c in candidate_cols
            if any(algo in c for algo in algorithms)
        ]

    # Filter by distance table if requested
    if distance_tables:
        candidate_cols = [
            c for c in candidate_cols
            if any(tbl in c for tbl in distance_tables)
        ]

    if len(candidate_cols) == 0:
        raise ValueError(f"No label columns found matching k={k} with the specified filters.")

    # Sort columns systematically: by algorithm then by weight
    candidate_cols.sort(key=lambda c: (parse_solution_column_name(c)["algorithm"], c))

    n_cols = len(candidate_cols)
    clean_labels = [format_solution_label(c, include_k=False) for c in candidate_cols]

    ari_matrix = np.zeros((n_cols, n_cols), dtype=np.float64)

    for i in range(n_cols):
        ari_matrix[i, i] = 1.0
        for j in range(i + 1, n_cols):
            score = adjusted_rand_score(labels_df[candidate_cols[i]], labels_df[candidate_cols[j]])
            ari_matrix[i, j] = score
            ari_matrix[j, i] = score

    ari_df = pd.DataFrame(ari_matrix, index=clean_labels, columns=clean_labels)
    return ari_df, candidate_cols


def plot_within_k_heatmaps(
    labels_df: pd.DataFrame,
    k_values: List[int],
    algorithms: Optional[List[str]] = None,
    distance_tables: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    fig_prefix: str = "within_k_ari_heatmap",
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (8.5, 7.5),
    mask_upper: bool = True,
    include_diagonal: bool = True,) -> Dict[int, pd.DataFrame]:
    """
    Renders and saves a correlation-style ARI heatmap for each requested k value.

    Parameters:
    -----------
    labels_df : pd.DataFrame
        DataFrame of patient cluster assignments.
    k_values : List[int]
        List of cluster counts to evaluate (e.g., [4, 5, 6, 7]).
    algorithms : Optional[List[str]]
        Algorithms to include.
    distance_tables : Optional[List[str]]
        Distance matrix weighting schemes to include.
    output_dir : Optional[str]
        Directory to save heatmap image files.
    fig_prefix : str
        Prefix for saved heatmap image files (default: 'within_k_ari_heatmap').
    cmap : str
        Color palette for heatmap (default: 'YlGnBu').
    figsize : Tuple[float, float]
        Figure size in inches.
    mask_upper : bool
        Whether to mask the upper triangle to remove duplicate mirror values (default: True).
    include_diagonal : bool
        Whether to include the diagonal 1.00 values when mask_upper is True (default: False).

    Returns:
    --------
    Dict[int, pd.DataFrame]:
        Dictionary mapping each k to its pairwise ARI DataFrame.
    """
    results_by_k = {}

    for k in k_values:
        print(f"\n[Within-k Heatmap] Generating cross-method concordance matrix for k={k}...")
        ari_df, _ = compute_within_k_ari_matrix(
            labels_df=labels_df,
            k=k,
            algorithms=algorithms,
            distance_tables=distance_tables,
        )
        results_by_k[k] = ari_df

        # Generate upper triangle mask to show only the lower triangle under the diagonal
        if mask_upper:
            k_offset = 1 if include_diagonal else 0
            mask = np.triu(np.ones_like(ari_df, dtype=bool), k=k_offset)
        else:
            mask = None

        # Compute average pairwise agreement across unique off-diagonal comparisons
        n_dim = ari_df.shape[0]
        if n_dim > 1:
            tril_vals = ari_df.values[np.tril_indices(n_dim, k=-1)]
            mean_ari = float(np.mean(tril_vals))
            std_ari = float(np.std(tril_vals))
            ari_subtitle = f"Mean Pairwise ARI = {mean_ari:.3f} (SD = {std_ari:.3f})"
        else:
            mean_ari = 1.0
            std_ari = 0.0
            ari_subtitle = "Mean Pairwise ARI = 1.000"

        print(f"  -> k={k}: {ari_subtitle} across {len(tril_vals) if n_dim > 1 else 1} comparisons")

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            ari_df,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Adjusted Rand Index (ARI)", "shrink": 0.85},
            linewidths=0.6,
            linecolor="#e0e0e0",
            square=True,
            ax=ax,
        )

        ax.set_title(
            f"Within-k Methodological Agreement (ARI)\n"
            f"Fixed Cluster Resolution: k = {k}\n"
            f"{ari_subtitle}",
            fontsize=11.5,
            fontweight="bold",
            pad=14,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{fig_prefix}_k{k}.png")
            safe_save_figure(fig, save_path)

        # Display in interactive Jupyter environments, then close cleanly
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
        except Exception:
            pass
        finally:
            plt.close(fig)

    return results_by_k


def run_within_k_comparisons(
    db_path: str,
    k_values: List[int],
    algorithms: Optional[List[str]] = None,
    distance_tables: Optional[List[str]] = None,
    labels_table: str = "jaccard_gower_cluster_labels",
    output_dir: Optional[str] = None,
    fig_prefix: str = "within_k_ari_heatmap",
    mask_upper: bool = True,
    include_diagonal: bool = True,) -> Dict[int, pd.DataFrame]:
    """
    Primary user-facing runner for Functionality 1 (Within-k Cross-Method Concordance).
    Loads labels from database and executes the heatmap generation pipeline.
    """
    print("\n" + "=" * 80)
    print("STARTING WITHIN-k METHODOLOGICAL CONCORDANCE EVALUATION")
    print("=" * 80)
    print(f"Database Path:   {db_path}")
    print(f"Evaluated k:     {k_values}")
    print(f"Algorithms:      {algorithms if algorithms else 'All available'}")
    print(f"Distance Tables: {distance_tables if distance_tables else 'All available'}")
    print("=" * 80)

    labels_df = load_cluster_labels_from_db(db_path, table_name=labels_table)

    results = plot_within_k_heatmaps(
        labels_df=labels_df,
        k_values=k_values,
        algorithms=algorithms,
        distance_tables=distance_tables,
        output_dir=output_dir,
        fig_prefix=fig_prefix,
        mask_upper=mask_upper,
        include_diagonal=include_diagonal,
    )

    print("\n[Complete] Within-k concordance evaluation finished successfully.")
    return results


# =============================================================================
# FUNCTIONALITY 2A: CROSS-k MACRO VIEW (ALL-BY-ALL CROSS-SOLUTION GRID)
# =============================================================================

def plot_cross_k_macro_grid(
    labels_df: pd.DataFrame,
    selected_solutions: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    distance_tables: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    fig_name: str = "cross_k_macro_ari_grid.png",
    cmap: str = "YlGnBu",
    figsize: Tuple[float, float] = (13, 11),
    mask_upper: bool = True,
    include_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Generates a comprehensive macro grid comparing all selected solutions across both
    different methods and different k values.

    Parameters:
    -----------
    labels_df : pd.DataFrame
        DataFrame of patient cluster assignments.
    selected_solutions : Optional[List[str]]
        Explicit list of database column names to compare. If None, resolved via
        algorithms, distance_tables, and k_values.
    algorithms : Optional[List[str]]
        Filter algorithms (if selected_solutions is None).
    distance_tables : Optional[List[str]]
        Filter distance tables (if selected_solutions is None).
    k_values : Optional[List[int]]
        Filter k values (if selected_solutions is None).
    output_dir : Optional[str]
        Directory to save the resulting figure.
    fig_name : str
        File name for the saved figure (default: 'cross_k_macro_ari_grid.png').
    cmap : str
        Colormap name.
    figsize : Tuple[float, float]
        Figure size in inches.
    mask_upper : bool
        Whether to mask the upper triangle to remove duplicate mirror values (default: True).
    include_diagonal : bool
        Whether to include the diagonal 1.00 values when mask_upper is True (default: False).

    Returns:
    --------
    pd.DataFrame:
        All-by-all symmetric pairwise ARI matrix.
    """
    # Resolve columns if not explicitly provided
    if selected_solutions is None:
        candidates = [
            c for c in labels_df.columns
            if not "_medoid_" in c and c not in ["patient_id", "medical_record_id"]
        ]
        if algorithms:
            candidates = [c for c in candidates if any(a in c for a in algorithms)]
        if distance_tables:
            candidates = [c for c in candidates if any(t in c for t in distance_tables)]
        if k_values:
            candidates = [c for c in candidates if any(c.endswith(f"_k{k}") for k in k_values)]
        selected_solutions = candidates

    if len(selected_solutions) == 0:
        raise ValueError("No solution columns selected for cross-k macro evaluation.")

    # Sort solutions systematically by algorithm, weight, then k
    def _sort_key(c: str):
        meta = parse_solution_column_name(c)
        return (meta["algorithm"], meta["weight_str"], meta["k"])

    selected_solutions = sorted(selected_solutions, key=_sort_key)

    n_sol = len(selected_solutions)
    clean_labels = [format_solution_label(c, include_k=True) for c in selected_solutions]

    print(f"\n[Cross-k Macro Grid] Computing all-by-all ARI matrix across {n_sol} selected solutions...")

    ari_matrix = np.zeros((n_sol, n_sol), dtype=np.float64)
    for i in range(n_sol):
        ari_matrix[i, i] = 1.0
        for j in range(i + 1, n_sol):
            score = adjusted_rand_score(labels_df[selected_solutions[i]], labels_df[selected_solutions[j]])
            ari_matrix[i, j] = score
            ari_matrix[j, i] = score

    ari_df = pd.DataFrame(ari_matrix, index=clean_labels, columns=clean_labels)

    # Compute average pairwise agreement across unique off-diagonal comparisons
    if n_sol > 1:
        tril_vals = ari_matrix[np.tril_indices(n_sol, k=-1)]
        mean_ari = float(np.mean(tril_vals))
        std_ari = float(np.std(tril_vals))
        ari_subtitle = f"Mean Pairwise ARI = {mean_ari:.3f} (SD = {std_ari:.3f})"
    else:
        mean_ari = 1.0
        std_ari = 0.0
        ari_subtitle = "Mean Pairwise ARI = 1.000"

    print(f"  -> Cross-k Macro Grid: {ari_subtitle} across {len(tril_vals) if n_sol > 1 else 1} comparisons")

    # Generate upper triangle mask to show only the lower triangle under the diagonal
    if mask_upper:
        k_offset = 1 if include_diagonal else 0
        mask = np.triu(np.ones_like(ari_df, dtype=bool), k=k_offset)
    else:
        mask = None

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        ari_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Adjusted Rand Index (ARI)", "shrink": 0.8},
        linewidths=0.5,
        linecolor="#dddddd",
        square=True,
        ax=ax,
    )

    ax.set_title(
        f"Cross-k Macro Agreement Grid\n"
        f"Pairwise Adjusted Rand Index (ARI) Across Candidate Solutions\n"
        f"{ari_subtitle}",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, fig_name)
        safe_save_figure(fig, save_path)

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)

    return ari_df


# =============================================================================
# FUNCTIONALITY 2B: CROSS-k MICRO VIEW (PATIENT FLOW ALLUVIAL DIAGRAM)
# =============================================================================

def plot_patient_flow_alluvial(
    labels_df: pd.DataFrame,
    solution_columns: List[str],
    step_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    fig_name: str = "patient_flow_alluvial.png",
    color_mode: str = "white_intermediates",
    figsize: Tuple[float, float] = (13, 7.5),
    palette: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Renders a publication-grade patient flow alluvial diagram tracking cluster
    splitting and patient conservation across sequential solutions (e.g. k=4 -> 5 -> 6 -> 7).

    Parameters:
    -----------
    labels_df : pd.DataFrame
        DataFrame of patient cluster assignments.
    solution_columns : List[str]
        Ordered list of column names representing the sequential clustering solutions.
    step_labels : Optional[List[str]]
        Display titles for each step column. If None, auto-formatted from column names.
    title : Optional[str]
        Figure title.
    output_dir : Optional[str]
        Directory to save the resulting figure.
    fig_name : str
        File name for the saved figure (default: 'patient_flow_alluvial.png').
    color_mode : str
        Coloring scheme for alluvial ribbons and cluster blocks:
        - 'lineage': Downstream cluster boxes and outflow ribbons carry over the color
          of their dominant Step 0 (root resolution) ancestor phenotype.
        - 'white_intermediates': Step 0 boxes are solid colored; downstream intermediate
          boxes are white with colored borders, while ribbons carry the ancestor colors.
        - 'cluster_id': Clusters are colored by their cluster index modulo palette length.
    figsize : Tuple[float, float]
        Figure size in inches.
    palette : Optional[List[str]]
        List of hex color codes for clusters.

    Returns:
    --------
    pd.DataFrame:
        Transition audit DataFrame summarizing patient counts and flow percentages between steps.
    """
    if len(solution_columns) < 2:
        raise ValueError("At least 2 sequential solution columns are required for an alluvial flow diagram.")

    valid_color_modes = ["lineage", "white_intermediates", "cluster_id"]
    if color_mode not in valid_color_modes:
        raise ValueError(f"Unknown color_mode '{color_mode}'. Must be one of {valid_color_modes}.")

    for col in solution_columns:
        if col not in labels_df.columns:
            raise KeyError(f"Solution column '{col}' not found in labels DataFrame.")

    if step_labels is None:
        step_labels = [format_solution_label(c, include_k=True) for c in solution_columns]

    if palette is None:
        # Publication-grade categorical palette
        palette = [
            "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
            "#66a61e", "#e6ab02", "#a6761d", "#386cb0",
            "#f0027f", "#bf5b17", "#666666"
        ]

    n_steps = len(solution_columns)
    total_patients = len(labels_df)
    col_xs = np.linspace(0, n_steps - 1, n_steps)
    block_width = 0.20
    gap = total_patients * 0.025  # Vertical gap between cluster blocks

    # Lineage tracking setup: map each unique cluster from the initial resolution (Step 0)
    origin_col = solution_columns[0]
    unique_origins = sorted(labels_df[origin_col].unique())
    origin_colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_origins)}

    # Determine dominant Step 0 ancestor cluster for every cluster at every step
    dominant_origins = {}
    for s_idx, col in enumerate(solution_columns):
        dominant_origins[s_idx] = {}
        if s_idx == 0:
            for c_id in labels_df[col].unique():
                dominant_origins[s_idx][c_id] = c_id
        else:
            ct_origin = pd.crosstab(labels_df[col], labels_df[origin_col])
            for c_id in labels_df[col].unique():
                dominant_origins[s_idx][c_id] = ct_origin.loc[c_id].idxmax()

    # Store block positions: step_idx -> cluster_id -> dict(y0, y1, count, curr_out_y, curr_in_y)
    block_pos = {}
    for s_idx, col in enumerate(solution_columns):
        counts = labels_df[col].value_counts().sort_index()
        k = len(counts)
        total_gap = (k - 1) * gap
        # Center stack vertically
        curr_y = total_gap / 2.0
        block_pos[s_idx] = {}
        for c_id, count in counts.items():
            y0 = curr_y
            y1 = curr_y + count
            block_pos[s_idx][c_id] = {
                "y0": y0,
                "y1": y1,
                "count": count,
                "curr_out_y": y0,
                "curr_in_y": y0,
            }
            curr_y = y1 + gap

    fig, ax = plt.subplots(figsize=figsize)

    # 1. Draw smooth Bézier transition ribbons between consecutive steps
    transition_records = []

    for s_idx in range(n_steps - 1):
        col_a = solution_columns[s_idx]
        col_b = solution_columns[s_idx + 1]
        x_a = col_xs[s_idx] + block_width / 2.0
        x_b = col_xs[s_idx + 1] - block_width / 2.0
        dx = x_b - x_a

        # Cross-tabulation of patient transitions
        ct = pd.crosstab(labels_df[col_a], labels_df[col_b])

        # Systematic sorting: iterate over source clusters, sort target clusters ascending
        for c_a in ct.index:
            for c_b in ct.columns:
                flow = ct.loc[c_a, c_b]
                if flow == 0:
                    continue

                y_a0 = block_pos[s_idx][c_a]["curr_out_y"]
                y_a1 = y_a0 + flow
                block_pos[s_idx][c_a]["curr_out_y"] = y_a1

                y_b0 = block_pos[s_idx + 1][c_b]["curr_in_y"]
                y_b1 = y_b0 + flow
                block_pos[s_idx + 1][c_b]["curr_in_y"] = y_b1

                # Construct cubic Bézier curve patch
                verts = [
                    (x_a, y_a0),
                    (x_a + dx * 0.5, y_a0),
                    (x_b - dx * 0.5, y_b0),
                    (x_b, y_b0),
                    (x_b, y_b1),
                    (x_b - dx * 0.5, y_b1),
                    (x_a + dx * 0.5, y_a1),
                    (x_a, y_a1),
                    (x_a, y_a0),
                ]
                codes = [
                    Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                    Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY
                ]

                # Ribbon color: carry over Step 0 origin color if in lineage / white_intermediates mode
                if color_mode in ["lineage", "white_intermediates"]:
                    sub_pts = labels_df[(labels_df[col_a] == c_a) & (labels_df[col_b] == c_b)]
                    dom_trans_orig = sub_pts[origin_col].value_counts().idxmax()
                    flow_color = origin_colors[dom_trans_orig]
                else:
                    flow_color = palette[int(c_a) % len(palette)]

                patch = patches.PathPatch(
                    Path(verts, codes),
                    facecolor=flow_color,
                    alpha=0.45,
                    edgecolor="none",
                    zorder=2,
                )
                ax.add_patch(patch)

                transition_records.append({
                    "step_from": step_labels[s_idx],
                    "cluster_from": int(c_a),
                    "step_to": step_labels[s_idx + 1],
                    "cluster_to": int(c_b),
                    "patient_count": flow,
                    "pct_of_source": float(flow / block_pos[s_idx][c_a]["count"]) * 100.0,
                })

    # 2. Draw cluster blocks and patient count labels on top
    for s_idx, col in enumerate(solution_columns):
        x_center = col_xs[s_idx]
        x_left = x_center - block_width / 2.0

        for c_id, pos in block_pos[s_idx].items():
            y0 = pos["y0"]
            h = pos["count"]
            dom_orig = dominant_origins[s_idx][c_id]

            if color_mode == "lineage":
                block_color = origin_colors[dom_orig]
                edge_color = "#222222"
                line_w = 1.2
                text_color = "white"
            elif color_mode == "white_intermediates":
                if s_idx == 0:
                    block_color = origin_colors[dom_orig]
                    edge_color = "#222222"
                    line_w = 1.2
                    text_color = "white"
                else:
                    block_color = "#ffffff"
                    edge_color = origin_colors[dom_orig]
                    line_w = 2.0
                    text_color = "#111111"
            else:  # 'cluster_id'
                block_color = palette[int(c_id) % len(palette)]
                edge_color = "#222222"
                line_w = 1.2
                text_color = "white"

            rect = patches.Rectangle(
                (x_left, y0),
                block_width,
                h,
                facecolor=block_color,
                edgecolor=edge_color,
                linewidth=line_w,
                zorder=4,
            )
            ax.add_patch(rect)

            pct = (h / total_patients) * 100.0
            # Label inside block if sufficiently tall; otherwise compact text
            lbl_text = f"C{int(c_id) + 1}\nn={h}\n({pct:.0f}%)" if h >= (total_patients * 0.05) else f"C{int(c_id) + 1}"
            ax.text(
                x_center,
                y0 + h / 2.0,
                lbl_text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=text_color,
                zorder=5,
            )

    # 3. Formatting
    ax.set_xticks(col_xs)
    ax.set_xticklabels(step_labels, fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.35, n_steps - 1 + 0.35)
    max_height = max([pos["y1"] for s in block_pos.values() for pos in s.values()])
    ax.set_ylim(-15, max_height + 25)

    # Hide all spine borders except clean bottom axis
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.set_visible(False)

    if title is None:
        title = "Patient Flow Alluvial Diagram\nTracking Phenotype Heritage & Cluster Splitting Across Resolutions"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, fig_name)
        safe_save_figure(fig, save_path)

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)

    transition_df = pd.DataFrame(transition_records)
    return transition_df


# =============================================================================
# FUNCTIONALITY 2: MASTER CROSS-k EVALUATION RUNNER
# =============================================================================

def run_cross_k_evaluation(
    db_path: str,
    macro_algorithms: Optional[List[str]] = None,
    macro_distance_tables: Optional[List[str]] = None,
    macro_k_values: Optional[List[int]] = None,
    alluvial_distance_table: Optional[str] = None,
    alluvial_algorithm: str = "pam",
    alluvial_k_values: Optional[List[int]] = None,
    labels_table: str = "jaccard_gower_cluster_labels",
    output_dir: Optional[str] = None,
    macro_fig_name: str = "cross_k_macro_ari_grid.png",
    alluvial_fig_name: str = "patient_flow_alluvial.png",
    alluvial_color_mode: str = "lineage",
    mask_upper: bool = True,
    include_diagonal: bool = True,) -> Dict[str, pd.DataFrame]:
    """
    Primary user-facing runner for Functionality 2 (Cross-k Stability Evaluation).
    Executes both the Macro View (all-by-all agreement grid) and Micro View (patient flow alluvial).

    Parameters:
    -----------
    db_path : str
        Path to SQLite database containing cluster assignments.
    macro_algorithms : Optional[List[str]]
        Algorithms to include in the macro agreement grid.
    macro_distance_tables : Optional[List[str]]
        Distance matrix weighting schemes to include in the macro agreement grid.
    macro_k_values : Optional[List[int]]
        k values to include in the macro grid (e.g. [4, 5, 6, 7]).
    alluvial_distance_table : Optional[str]
        Distance table for alluvial tracking (defaults to first table in macro_distance_tables).
    alluvial_algorithm : str
        Algorithm to track across k in the alluvial diagram (default: 'pam').
    alluvial_k_values : Optional[List[int]]
        Sequential k values to trace in the alluvial diagram (defaults to macro_k_values).
    labels_table : str
        Name of cluster labels table.
    output_dir : Optional[str]
        Directory for saving plots.
    macro_fig_name : str
        Custom file name for the macro ARI agreement grid figure (default: 'cross_k_macro_ari_grid.png').
    alluvial_fig_name : str
        Custom file name for the alluvial patient flow figure (default: 'patient_flow_alluvial.png').
    alluvial_color_mode : str
        Coloring scheme for alluvial: 'lineage' (default), 'white_intermediates', or 'cluster_id'.
    mask_upper : bool
        Whether to mask the upper triangle in the macro grid (default: True).
    include_diagonal : bool
        Whether to include the diagonal in the macro grid (default: False).

    Returns:
    --------
    Dict[str, pd.DataFrame]:
        Dictionary containing 'macro_ari_df' and 'alluvial_transitions_df'.
    """
    # Resolve default alluvial distance table and k values if not explicitly provided
    if alluvial_distance_table is None and macro_distance_tables:
        alluvial_distance_table = macro_distance_tables[0]
    elif alluvial_distance_table is None:
        alluvial_distance_table = "hybrid_wgc067_eb033"

    if alluvial_k_values is None and macro_k_values:
        alluvial_k_values = sorted(macro_k_values)
    elif alluvial_k_values is None:
        alluvial_k_values = [4, 5, 6, 7]

    print("\n" + "=" * 80)
    print("STARTING CROSS-k EVALUATION (MACRO GRID & MICRO ALLUVIAL FLOW)")
    print("=" * 80)
    print(f"Database Path:           {db_path}")
    print(f"Macro Algorithms:        {macro_algorithms if macro_algorithms else 'All'}")
    print(f"Macro Distance Tables:   {macro_distance_tables if macro_distance_tables else 'All'}")
    print(f"Macro k Range:           {macro_k_values if macro_k_values else 'All'}")
    print(f"Macro Figure Name:       {macro_fig_name}")
    print(f"Alluvial Target Method:  {alluvial_algorithm} on {alluvial_distance_table}")
    print(f"Alluvial k Range:        {alluvial_k_values}")
    print(f"Alluvial Color Mode:     {alluvial_color_mode}")
    print(f"Alluvial Figure Name:    {alluvial_fig_name}")
    print("=" * 80)

    labels_df = load_cluster_labels_from_db(db_path, table_name=labels_table)

    # 1. Macro View: All-by-all Cross-Solution Grid
    macro_ari_df = plot_cross_k_macro_grid(
        labels_df=labels_df,
        algorithms=macro_algorithms,
        distance_tables=macro_distance_tables,
        k_values=macro_k_values,
        output_dir=output_dir,
        fig_name=macro_fig_name,
        mask_upper=mask_upper,
        include_diagonal=include_diagonal,
    )

    # 2. Micro View: Patient Flow Alluvial Diagram
    alluvial_cols = [
        f"{alluvial_distance_table}_{alluvial_algorithm}_k{k}"
        for k in alluvial_k_values
    ]

    alluvial_step_labels = [f"k = {k}" for k in alluvial_k_values]
    target_meta = parse_solution_column_name(alluvial_cols[0])
    alluvial_title = (
        f"Patient Flow Alluvial Diagram\n"
        f"Phenotype Splitting & Heritage: {target_meta['algo_title']} [{target_meta['weight_str']}]"
    )

    alluvial_transitions_df = plot_patient_flow_alluvial(
        labels_df=labels_df,
        solution_columns=alluvial_cols,
        step_labels=alluvial_step_labels,
        title=alluvial_title,
        output_dir=output_dir,
        fig_name=alluvial_fig_name,
        color_mode=alluvial_color_mode,
    )

    print("\n[Complete] Cross-k macro and micro evaluation finished successfully.")
    return {
        "macro_ari_df": macro_ari_df,
        "alluvial_transitions_df": alluvial_transitions_df,
    }


# =============================================================================
# CLI EXECUTION ENTRY POINT (STANDALONE TESTING)
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_file = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in_2025.sqlite")
    if not os.path.exists(db_file):
        db_file = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in.sqlite")
    out_dir = os.path.join(base_dir, "p1_outputs", "cross_cluster_comparison")

    print("[CLI Execution] Running module smoke test...")

    # Test Functionality 1
    within_res = run_within_k_comparisons(
        db_path=db_file,
        k_values=[4, 5],
        algorithms=["pam", "hclust_complete"],
        distance_tables=["hybrid_wgc067_eb033", "hybrid_wgc07_eb03"],
        output_dir=out_dir,
    )

    # Test Functionality 2
    cross_res = run_cross_k_evaluation(
        db_path=db_file,
        macro_algorithms=["pam", "hclust_complete"],
        macro_distance_tables=["hybrid_wgc067_eb033", "hybrid_wgc07_eb03"],
        macro_k_values=[4, 5, 6],
        alluvial_distance_table="hybrid_wgc067_eb033",
        alluvial_algorithm="pam",
        alluvial_k_values=[4, 5, 6],
        output_dir=out_dir,
    )
    print("[CLI Execution] All smoke tests passed.")

