"""
================================================================================
CLUSTER_PROJECTIONS.PY
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module provides 2D and 3D geometric ordination and dimensionality reduction
visualizations for behavioral obesity phenotype clusters (Paper #1).

Because patients were clustered using a non-Euclidean hybrid dissimilarity
matrix (Jaccard distance for 12 asymmetric binary Weight Gain Causes + Gower
distance for 6 mixed Eating Behaviors), standard Euclidean PCA on raw variables
distorts the space and violates the asymmetric binary premise.

This module operates directly on the precomputed hybrid pairwise distance matrix:
1. Metric Multidimensional Scaling / PCoA (Principal Coordinate Analysis):
   - Computes analytical Gower-centered spectral decomposition of the distance matrix.
   - Zero hyperparameters, completely deterministic, preserving global pairwise distances.
   - Computes and plots both 2D and 3D coordinate embeddings.
   - Reports exact variance explained by Dimension 1, Dimension 2, and Dimension 3.
   - Saves figures as '2d_pcoa_{prefix}.png' / '2d_pca_{prefix}.png' and
     '3d_pcoa_{prefix}.png' / '3d_pca_{prefix}.png'.
2. Non-Linear Manifold Learning (UMAP / t-SNE):
   - Constructs neighborhood graphs directly from precomputed hybrid distances.
   - Reveals non-linear cluster boundaries and local manifold topology in 2D and 3D.
   - Uses UMAP with metric='precomputed' (with automatic fallback to precomputed t-SNE
     if umap-learn is not installed).
   - Saves figures as '2d_umap_{prefix}.png' and '3d_umap_{prefix}.png'.

Pseudocode: 
-----------
PIPELINE run_cluster_projection_pipeline(source_db, labels_db, distance_config, algorithm, k, output_dir, ...):
    1. LOAD DATA:
       - Query distance matrix from SQLite: "hybrid_{distance_config}" (308 x 308 symmetric array)
       - Query cluster labels from "jaccard_gower_cluster_labels" aligned by medical_record_id
       - Extract exemplar medoid IDs if algorithm == 'pam'

    2. METRIC PCoA / MDS ORDINATION (2D & 3D):
       - Double-centering of squared distance matrix: B = -0.5 * H * (D^2) * H
       - Eigen-decomposition of B -> extract eigenvalues and eigenvectors
       - Calculate percentage of total positive variance explained by Axes 1, 2, and 3
       - Scale coordinates: X_i = V_i * sqrt(Lambda_i)
       - Render 2D scatter plot:
         * Color points by cluster (using cluster_config_path if provided)
         * Mark cluster centroids with 'X'
         * Highlight exemplar medoids with star symbols (★) if present
         * Title: Explained Variance (% on Axis 1, % on Axis 2)
       - Save 2D & 3D publication figures (2d_pcoa_{prefix}.png and 3d_pcoa_{prefix}.png)

    3. NON-LINEAR MANIFOLD EMBEDDING (UMAP / t-SNE in 2D & 3D):
       - If umap-learn is available: run UMAP(metric='precomputed', n_neighbors=15, min_dist=0.2)
       - If umap-learn is absent: fall back gracefully to TSNE(metric='precomputed', perplexity=30)
       - Render 2D & 3D scatter plots colored by cluster with medoids highlighted
       - Save publication figures (2d_umap_{prefix}.png and 3d_umap_{prefix}.png)

    4. SAVE PROJECTION COORDINATES:
       - If save_to_db=True: persist [medical_record_id, cluster_id, pcoa_dim1..3, umap_dim1..2, umap_3d_dim1..3] to SQLite
       - Return (projection_df, figures_dict)

I/O and Frontend Architecture:
------------------------------
Adheres strictly to the decoupled parameter interface of cluster_comparisons.py,
allowing seamless execution from notebook cells with direct export to project output folders.

Author: Gemini 3.8 Flash High in Antigravity
Revised: 28Oct26
Target Environment: Python 3.10+ / SQLite3 / Pandas / NumPy / Scikit-learn / Matplotlib / Seaborn
================================================================================
"""

import os
import sys
import re
import json
import sqlite3
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# Optional UMAP import with graceful fallback
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# DEFAULT COLOR PALETTE & CONFIGURATION HELPERS
# =============================================================================

def _resolve_path(path: str) -> str:
    """Resolve relative file paths to absolute paths."""
    if not path or os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate1 = os.path.join(script_dir, path)
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.abspath(path)
    if os.path.exists(candidate2):
        return candidate2
    return candidate1


def _load_default_cluster_palette() -> List[str]:
    """Loads default 10-color publication palette from cluster_color_dict.json if available."""
    fallback_palette = [
        "#1f77b4",  # Cluster 1: Blue
        "#ff7f0e",  # Cluster 2: Orange
        "#2ca02c",  # Cluster 3: Green
        "#d62728",  # Cluster 4: Red
        "#9467bd",  # Cluster 5: Purple
        "#8c564b",  # Cluster 6: Brown
        "#e377c2",  # Cluster 7: Pink
        "#7f7f7f",  # Cluster 8: Gray
        "#bcbd22",  # Cluster 9: Olive / Yellow-green
        "#17becf",  # Cluster 10: Teal
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "cluster_color_dict.json"),
        os.path.join(os.getcwd(), "cluster_color_dict.json"),
        os.path.join(os.getcwd(), "p1_dir", "p1_scripts", "cluster_color_dict.json"),
    ]
    for c_path in candidates:
        if os.path.exists(c_path):
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                c_dict = data.get("cluster_colors", data)
                palette = []
                for i in range(1, 11):
                    col = c_dict.get(str(i), c_dict.get(f"Cluster {i}", fallback_palette[i - 1]))
                    palette.append(col)
                return palette
            except Exception:
                pass
    return fallback_palette


DEFAULT_CLUSTER_COLORS = _load_default_cluster_palette()


def load_cluster_config(config_source: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
    """Loads custom cluster labels and colors from a dictionary or JSON file.
    Automatically merges fixed colors from cluster_color_dict.json by default.
    """
    config: Dict[str, Any] = {"cluster_labels": {}, "cluster_colors": {}}

    # Load base colors from cluster_color_dict.json
    default_json = _resolve_path("cluster_color_dict.json")
    if os.path.exists(default_json):
        try:
            with open(default_json, "r", encoding="utf-8") as f:
                d_colors = json.load(f)
            config["cluster_colors"].update(d_colors.get("cluster_colors", d_colors))
        except Exception:
            pass

    if config_source is None:
        return config

    if isinstance(config_source, dict):
        if "cluster_colors" in config_source:
            config["cluster_colors"].update(config_source["cluster_colors"])
        if "cluster_labels" in config_source:
            config["cluster_labels"].update(config_source["cluster_labels"])
        return config

    if isinstance(config_source, str):
        target = config_source
        if not target.endswith(".json") and os.path.exists(_resolve_path(f"{target}.json")):
            target = f"{target}.json"
        json_path = _resolve_path(target)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config["cluster_labels"].update(data.get("cluster_labels", {}))
                config["cluster_colors"].update(data.get("cluster_colors", data))
                print(f"[Config] Loaded cluster configuration from: '{json_path}'")
            except Exception as e:
                print(f"Warning: Failed to parse cluster config JSON '{json_path}': {e}")
    return config


def get_cluster_label(cluster_id: int, cluster_config: Optional[Dict[str, Any]] = None) -> str:
    """Returns human-readable cluster label, falling back to 1-indexed 'Cluster {id + 1}'."""
    if cluster_config:
        labels = cluster_config.get("cluster_labels", {})
        if str(cluster_id) in labels:
            return labels[str(cluster_id)]
        if cluster_id in labels:
            return labels[cluster_id]
    try:
        cid_int = int(cluster_id)
        if cid_int == -1:
            return "Noise / Outliers"
        return f"Cluster {cid_int + 1}"
    except (ValueError, TypeError):
        return f"Cluster {cluster_id}"


def get_cluster_color(cluster_id: Union[int, str], cluster_config: Optional[Dict[str, Any]] = None) -> str:
    """Returns cluster-specific hex color code from fixed cluster_color_dict or default palette."""
    if cluster_config is None:
        cluster_config = load_cluster_config(None)

    colors = cluster_config.get("cluster_colors", {}) if cluster_config else {}

    # 1. Match via human-readable label (e.g. 0-indexed DB ID 0 -> 'Cluster 1')
    c_lbl = get_cluster_label(cluster_id, cluster_config)
    if c_lbl in colors:
        return colors[c_lbl]

    # 2. Match via direct string representation
    str_id = str(cluster_id)
    if str_id in colors:
        return colors[str_id]
    if cluster_id in colors:
        return colors[cluster_id]

    # 3. Numeric conversion & fallback
    try:
        cid_int = int(cluster_id)
        if cid_int == -1:
            return "#7f7f7f"
        return DEFAULT_CLUSTER_COLORS[cid_int % len(DEFAULT_CLUSTER_COLORS)]
    except (ValueError, TypeError):
        match = re.search(r"(\d+)", str_id)
        if match:
            c_num = int(match.group(1))
            if c_num >= 1:
                return DEFAULT_CLUSTER_COLORS[(c_num - 1) % len(DEFAULT_CLUSTER_COLORS)]
        return "#7f7f7f"


# =============================================================================
# DATA LOADING (DISTANCE MATRIX & CLUSTER LABELS)
# =============================================================================

def resolve_cluster_column_name(
    weight: str = "wgc07_eb03",
    algorithm: str = "pam",
    k: int = 7,
    available_columns: Optional[List[str]] = None,) -> str:
    """Constructs or resolves the canonical cluster column name in labels table."""
    clean_weight = weight.replace("hybrid_", "")
    algo_key = "pam" if "pam" in algorithm.lower() else "hclust_complete"
    target_col = f"hybrid_{clean_weight}_{algo_key}_k{k}"

    if available_columns and target_col not in available_columns:
        candidates = [c for c in available_columns if f"k{k}" in c and algo_key in c and clean_weight in c]
        if candidates:
            return candidates[0]
        candidates = [c for c in available_columns if f"k{k}" in c and algo_key in c]
        if candidates:
            return candidates[0]
    return target_col


def load_cluster_and_distance_data(
    source_db_path: str,
    labels_db_path: Optional[str] = None,
    distance_config: str = "wgc07_eb03",
    algorithm: str = "pam",
    k: int = 7,
    cluster_table: str = "jaccard_gower_cluster_labels",
    distance_table: Optional[str] = None,
    medical_record_id_col: str = "medical_record_id",) -> Tuple[np.ndarray, pd.Series, List[str], Optional[List[str]]]:
    """
    Loads pairwise distance matrix and aligned cluster labels from SQLite.

    Parameters:
    -----------
    source_db_path : str
        Path to database containing distance matrix.
    labels_db_path : Optional[str]
        Path to database containing cluster labels (defaults to source_db_path).
    distance_config : str
        Dissimilarity scheme name (e.g. 'wgc07_eb03').
    algorithm : str
        Clustering algorithm ('pam' or 'hclust_complete').
    k : int
        Number of clusters (2..10).
    cluster_table : str
        Cluster labels table name.
    distance_table : Optional[str]
        Explicit distance matrix table name (defaults to 'hybrid_{distance_config}').
    medical_record_id_col : str
        Medical record identifier column.

    Returns:
    --------
    Tuple[np.ndarray, pd.Series, List[str], Optional[List[str]]]:
        - dist_matrix: N x N symmetric numpy array of pairwise distances.
        - cluster_labels: pandas Series of cluster assignments aligned with matrix rows.
        - record_ids: List of medical record IDs in matrix order.
        - medoid_ids: List of exemplar medoid medical record IDs (if available).
    """
    src_db = _resolve_path(source_db_path)
    lbl_db = _resolve_path(labels_db_path or source_db_path)

    clean_dist = distance_config.replace("hybrid_", "")
    dist_tbl = distance_table or f"hybrid_{clean_dist}"

    print("\n" + "=" * 80)
    print("LOADING DISTANCE MATRIX & CLUSTER LABELS FOR 2D PROJECTIONS")
    print("=" * 80)
    print(f"Distance Matrix DB:     {src_db} :: Table: '{dist_tbl}'")
    print(f"Cluster Labels DB:      {lbl_db} :: Table: '{cluster_table}'")
    print(f"Configuration:          Algorithm={algorithm}, Weight={clean_dist}, k={k}")

    # 1. Load Distance Matrix
    with sqlite3.connect(src_db) as conn:
        df_dist = pd.read_sql_query(f"SELECT * FROM `{dist_tbl}`", conn)

    record_ids = df_dist[medical_record_id_col].tolist()
    dist_cols = [c for c in df_dist.columns if c not in [medical_record_id_col, "avg_dist_to_all"]]
    dist_matrix = df_dist[dist_cols].values.astype(float)

    # Ensure symmetry
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)
    np.fill_diagonal(dist_matrix, 0.0)
    print(f"  [OK] Loaded Symmetric Distance Matrix: {dist_matrix.shape[0]} x {dist_matrix.shape[1]}")

    # 2. Load Cluster Labels
    with sqlite3.connect(lbl_db) as conn:
        lbl_cols = [r[1] for r in conn.execute(f"PRAGMA table_info(`{cluster_table}`)").fetchall()]
        cluster_col = resolve_cluster_column_name(
            weight=clean_dist, algorithm=algorithm, k=k, available_columns=lbl_cols
        )
        medoid_col = cluster_col.replace(f"_{k}", f"_medoid_{k}")
        has_medoids = medoid_col in lbl_cols

        select_cols = [f"`{medical_record_id_col}`", f"`{cluster_col}` AS cluster_id"]
        if has_medoids:
            select_cols.append(f"`{medoid_col}` AS medoid_id")

        df_labels = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM `{cluster_table}`", conn
        )

    # Align labels with distance matrix ordering
    df_labels = df_labels.set_index(medical_record_id_col).reindex(record_ids).reset_index()
    cluster_labels = df_labels["cluster_id"].astype(int)

    medoid_ids = None
    if has_medoids and "medoid_id" in df_labels.columns:
        valid_medoids = df_labels["medoid_id"].dropna().unique().tolist()
        medoid_ids = [m for m in valid_medoids if m in record_ids]
        print(f"  [OK] Identified {len(medoid_ids)} Exemplar Medoid Patients.")

    print(f"  [OK] Aligned {len(cluster_labels)} Patient Labels across {cluster_labels.nunique()} Clusters.")
    print("=" * 80 + "\n")

    return dist_matrix, cluster_labels, record_ids, medoid_ids


# =============================================================================
# PART 1: CLASSICAL PCOA (PRINCIPAL COORDINATE ANALYSIS / METRIC MDS)
# =============================================================================

def compute_pcoa_coordinates(
    dist_matrix: np.ndarray,
    n_components: int = 3,) -> Tuple[np.ndarray, List[float]]:
    """
    Computes exact classical Principal Coordinate Analysis (PCoA / Metric MDS)
    via Gower's double-centering matrix decomposition for up to n_components dimensions.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        N x N symmetric pairwise distance matrix.
    n_components : int
        Number of principal coordinates to extract (default: 3).

    Returns:
    --------
    Tuple[np.ndarray, List[float]]:
        - coords: N x n_components array of principal coordinates.
        - var_pcts: List of variance percentages explained by each principal coordinate axis.
    """
    n = dist_matrix.shape[0]

    # Double-centering matrix: H = I - (1/n) * 1 * 1^T
    H = np.eye(n) - np.ones((n, n)) / n

    # Squared distance matrix
    D_sq = dist_matrix ** 2

    # Centered inner-product matrix B = -0.5 * H * D^2 * H
    B = -0.5 * H.dot(D_sq).dot(H)

    # Eigen-decomposition (Hermitian/symmetric)
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance explained by positive eigenvalues
    pos_evals = eigenvalues[eigenvalues > 0]
    total_pos_variance = np.sum(pos_evals) if len(pos_evals) > 0 else 1.0

    var_pcts = [
        float((max(0.0, eigenvalues[i]) / total_pos_variance) * 100.0)
        for i in range(n_components)
    ]

    # Project into principal coordinates: X_i = V_i * sqrt(Lambda_i)
    coords_list = []
    for i in range(n_components):
        scale_i = np.sqrt(max(0.0, eigenvalues[i]))
        coords_list.append(eigenvectors[:, i] * scale_i)

    coords = np.column_stack(coords_list)
    return coords, var_pcts


def plot_cluster_pcoa(
    coords: np.ndarray,
    cluster_labels: pd.Series,
    var_pct_1: float,
    var_pct_2: float,
    var_pct_3: Optional[float] = None,
    medoid_mask: Optional[np.ndarray] = None,
    output_filepath: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (10, 8),) -> plt.Figure:
    """
    Renders publication-ready 2D or 3D PCoA (Metric MDS) scatter plot colored by cluster.
    Automatically switches between 2D and 3D visualization based on the shape of coords.

    Parameters:
    -----------
    coords : np.ndarray
        N x 2 or N x 3 array of PCoA coordinates.
    cluster_labels : pd.Series
        Cluster assignments per patient.
    var_pct_1 : float
        Variance explained by Axis 1.
    var_pct_2 : float
        Variance explained by Axis 2.
    var_pct_3 : Optional[float]
        Variance explained by Axis 3 (required if coords is 3D).
    medoid_mask : Optional[np.ndarray]
        Boolean array of length N indicating exemplar medoids (if PAM).
    output_filepath : Optional[str]
        File path to save the 300 DPI PNG figure.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    title_suffix : str
        Additional suffix for plot title.
    figsize : Tuple[int, int]
        Figure dimensions.

    Returns:
    --------
    plt.Figure: Rendered matplotlib figure.
    """
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(cluster_labels.unique())
    is_3d = coords.shape[1] >= 3

    if is_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        v3 = var_pct_3 if var_pct_3 is not None else 0.0

        for cid in clusters:
            c_lbl = get_cluster_label(cid, cluster_config)
            c_col = get_cluster_color(cid, cluster_config)
            mask = (cluster_labels == cid).values

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=c_col,
                label=f"{c_lbl} (n={mask.sum()})",
                alpha=0.75,
                s=45,
                edgecolors="white",
                linewidth=0.5,
            )

            centroid = coords[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                c=c_col,
                s=120,
                marker="X",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )

        if medoid_mask is not None and np.any(medoid_mask):
            ax.scatter(
                coords[medoid_mask, 0],
                coords[medoid_mask, 1],
                coords[medoid_mask, 2],
                c="yellow",
                s=220,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                label="Exemplar Medoids",
                zorder=6,
            )

        total_var = var_pct_1 + var_pct_2 + v3
        ax.set_title(
            f"3D PCoA Ordination of Hybrid Distance Matrix {title_suffix}\n"
            f"3D Explained Variance: {total_var:.1f}% (Axis 1: {var_pct_1:.1f}%, Axis 2: {var_pct_2:.1f}%, Axis 3: {v3:.1f}%)",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(f"PCoA Axis 1 ({var_pct_1:.1f}%)", fontsize=10, labelpad=8)
        ax.set_ylabel(f"PCoA Axis 2 ({var_pct_2:.1f}%)", fontsize=10, labelpad=8)
        ax.set_zlabel(f"PCoA Axis 3 ({v3:.1f}%)", fontsize=10, labelpad=12)
        ax.view_init(elev=25, azim=45)
        leg = ax.legend(title="Behavioral Cluster", loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=9, framealpha=0.9)
        fig.subplots_adjust(left=0.08, right=0.85, bottom=0.08, top=0.90)

    else:
        fig, ax = plt.subplots(figsize=figsize)
        leg = None

        for cid in clusters:
            c_lbl = get_cluster_label(cid, cluster_config)
            c_col = get_cluster_color(cid, cluster_config)
            mask = (cluster_labels == cid).values

            # Scatter points for cluster members
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=c_col,
                label=f"{c_lbl} (n={mask.sum()})",
                alpha=0.75,
                s=45,
                edgecolors="white",
                linewidth=0.5,
            )

            # Plot cluster centroid marker
            centroid = coords[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                c=c_col,
                s=120,
                marker="X",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )

        # Highlight exemplar medoids if provided
        if medoid_mask is not None and np.any(medoid_mask):
            ax.scatter(
                coords[medoid_mask, 0],
                coords[medoid_mask, 1],
                c="yellow",
                s=220,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                label="Exemplar Medoids",
                zorder=6,
            )

        total_var = var_pct_1 + var_pct_2
        ax.set_title(
            f"2D PCoA Ordination of Hybrid Distance Matrix {title_suffix}\n"
            f"2D Explained Variance: {total_var:.1f}% (Axis 1: {var_pct_1:.1f}%, Axis 2: {var_pct_2:.1f}%)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel(f"PCoA Axis 1 ({var_pct_1:.1f}% Variance)", fontsize=11)
        ax.set_ylabel(f"PCoA Axis 2 ({var_pct_2:.1f}% Variance)", fontsize=11)
        ax.legend(title="Behavioral Cluster", loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

    if output_filepath is not None:
        dest_path = _resolve_path(output_filepath)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        extra_artists = [ax.zaxis.label, leg] if (is_3d and leg is not None) else None
        fig.savefig(dest_path, dpi=300, bbox_inches="tight", bbox_extra_artists=extra_artists)
        print(f"  [OK] Saved {'3D' if is_3d else '2D'} PCoA Ordination Figure: '{dest_path}'")

    return fig


# =============================================================================
# PART 2: NON-LINEAR MANIFOLD LEARNING (UMAP / PRECOMPUTED T-SNE)
# =============================================================================

def compute_manifold_coordinates(
    dist_matrix: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,) -> Tuple[np.ndarray, str]:
    """
    Computes 2D or 3D non-linear manifold coordinates directly from precomputed distance matrix.
    Uses UMAP if installed; gracefully falls back to precomputed t-SNE if umap-learn is absent.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        N x N symmetric pairwise distance matrix.
    n_components : int
        Number of embedding dimensions (default: 2, or 3 for 3D).
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[np.ndarray, str]:
        - coords: N x n_components array of manifold coordinates.
        - method_name: 'UMAP' or 't-SNE'.
    """
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            n_neighbors=25, # default is 15 which works well for large datasets - for n=308, set it to a higher number (20-40) 
                # to not focus too much on local structures with very few samples
            min_dist=0.3, # default is 0.2, set it also a bit higher (up to 0.4) to not pack clusters too tightly
            random_state=random_state,
        )
        coords = reducer.fit_transform(dist_matrix)
        method_name = "UMAP"
    else:
        # Graceful fallback to precomputed t-SNE
        perplexity = min(30, max(5, int(dist_matrix.shape[0] / 10)))
        tsne = TSNE(
            n_components=n_components,
            metric="precomputed",
            perplexity=perplexity,
            random_state=random_state,
            init="random",
            max_iter=1000,
        )
        coords = tsne.fit_transform(dist_matrix)
        method_name = "t-SNE"

    return coords, method_name


def plot_cluster_manifold(
    coords: np.ndarray,
    cluster_labels: pd.Series,
    method_name: str = "UMAP",
    medoid_mask: Optional[np.ndarray] = None,
    output_filepath: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (10, 8),) -> plt.Figure:
    """
    Renders publication-ready 2D or 3D non-linear manifold (UMAP / t-SNE) scatter plot.
    Automatically switches between 2D and 3D visualization based on the shape of coords.

    Parameters:
    -----------
    coords : np.ndarray 
        N x 2 or N x 3 manifold coordinates.
    cluster_labels : pd.Series
        Cluster assignments per patient.
    method_name : str
        'UMAP' or 't-SNE'.
    medoid_mask : Optional[np.ndarray]
        Boolean array for exemplar medoids.
    output_filepath : Optional[str]
        File path to save PNG figure.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    title_suffix : str
        Additional title suffix.
    figsize : Tuple[int, int]
        Figure size.

    Returns:
    --------
    plt.Figure: Rendered matplotlib figure.
    """
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(cluster_labels.unique())
    is_3d = coords.shape[1] >= 3

    if is_3d:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        for cid in clusters:
            c_lbl = get_cluster_label(cid, cluster_config)
            c_col = get_cluster_color(cid, cluster_config)
            mask = (cluster_labels == cid).values

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                c=c_col,
                label=f"{c_lbl} (n={mask.sum()})",
                alpha=0.75,
                s=45,
                edgecolors="white",
                linewidth=0.5,
            )

            centroid = coords[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                c=c_col,
                s=120,
                marker="X",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )

        if medoid_mask is not None and np.any(medoid_mask):
            ax.scatter(
                coords[medoid_mask, 0],
                coords[medoid_mask, 1],
                coords[medoid_mask, 2],
                c="yellow",
                s=220,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                label="Exemplar Medoids",
                zorder=6,
            )

        ax.set_title(
            f"3D {method_name} manifold embedding of hybrid distance matrix {title_suffix}".strip(),
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=10, labelpad=8)
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=10, labelpad=8)
        ax.set_zlabel(f"{method_name} Dimension 3", fontsize=10, labelpad=12)
        ax.view_init(elev=25, azim=45)
        leg = ax.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=9, framealpha=0.9)
        fig.subplots_adjust(left=0.08, right=0.85, bottom=0.08, top=0.90)

    else:
        fig, ax = plt.subplots(figsize=figsize)
        leg = None

        for cid in clusters:
            c_lbl = get_cluster_label(cid, cluster_config)
            c_col = get_cluster_color(cid, cluster_config)
            mask = (cluster_labels == cid).values

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=c_col,
                label=f"{c_lbl} (n={mask.sum()})",
                alpha=0.75,
                s=45,
                edgecolors="white",
                linewidth=0.5,
            )

            centroid = coords[mask].mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                c=c_col,
                s=120,
                marker="X",
                edgecolors="black",
                linewidth=1.2,
                zorder=5,
            )

        if medoid_mask is not None and np.any(medoid_mask):
            ax.scatter(
                coords[medoid_mask, 0],
                coords[medoid_mask, 1],
                c="yellow",
                s=220,
                marker="*",
                edgecolors="black",
                linewidth=1.5,
                label="Exemplar Medoids",
                zorder=6,
            )

        ax.set_title(
            f"2D {method_name} Manifold Embedding of Hybrid Distance Matrix {title_suffix}".strip(),
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
        ax.legend(title="Behavioral Cluster", loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

    if output_filepath is not None:
        dest_path = _resolve_path(output_filepath)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        extra_artists = [ax.zaxis.label, leg] if (is_3d and leg is not None) else None
        fig.savefig(dest_path, dpi=300, bbox_inches="tight", bbox_extra_artists=extra_artists)
        print(f"  [OK] Saved {'3D' if is_3d else '2D'} {method_name} Manifold Figure: '{dest_path}'")

    return fig


# =============================================================================
# PART 3: END-TO-END PIPELINE WRAPPER (NOTEBOOK CALLABLE)
# =============================================================================

def run_cluster_projection_pipeline(
    source_db_path: str = "../p1_dbs/pnk_db2_p1_in_2025.sqlite",
    labels_db_path: Optional[str] = None,
    distance_config: str = "wgc07_eb03",
    algorithm: str = "hclust_complete",
    k: int = 5,
    cluster_table: str = "jaccard_gower_cluster_labels",
    distance_table: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_to_db: bool = True,
    output_db_path: Optional[str] = "../p1_dbs/pnk_db2_p1_out_2025.sqlite",
    db_table_name: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    generate_pcoa: bool = True,
    generate_umap: bool = True,) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    End-to-end execution pipeline conducting 2D & 3D geometric ordination and manifold projections:
    1. Loads hybrid distance matrix and aligned cluster assignments.
    2. Computes and plots classical PCoA (Metric MDS) in both 2D and 3D with explained variance.
    3. Computes and plots non-linear manifold embeddings (UMAP / t-SNE) in both 2D and 3D.
    4. Persists 2D & 3D coordinates to SQLite and saves 300 DPI publication figures.

    Parameters:
    -----------
    source_db_path : str
        Path to database containing pairwise distance matrix.
    labels_db_path : Optional[str]
        Path to database containing cluster labels (defaults to source_db_path).
    distance_config : str
        Dissimilarity weighting scheme (default: 'wgc07_eb03').
    algorithm : str
        Clustering algorithm ('hclust_complete' or 'pam').
    k : int
        Number of clusters (2..10).
    cluster_table : str
        Labels table name (default: 'jaccard_gower_cluster_labels').
    distance_table : Optional[str]
        Distance table name (defaults to 'hybrid_{distance_config}').
    output_dir : Optional[str]
        Directory where figures will be saved.
    save_to_db : bool
        Whether to save projection coordinates to SQLite.
    output_db_path : Optional[str]
        Output SQLite database path.
    db_table_name : Optional[str]
        Table name for saving coordinates in SQLite.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    generate_pcoa : bool
        Whether to generate 2D and 3D PCoA plots (default: True).
    generate_umap : bool
        Whether to generate 2D and 3D UMAP/t-SNE plots (default: True).

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
        - projection_df: DataFrame with patient medical_record_id, cluster_id, PCoA (2D/3D) and UMAP (2D/3D) coordinates.
        - figures: Dictionary containing {'pcoa_2d': fig, 'pcoa_3d': fig, 'umap_2d': fig, 'umap_3d': fig}.
    """
    clean_dist = distance_config.replace("hybrid_", "")
    prefix = f"{algorithm}_{clean_dist}_k{k}"

    if db_table_name is None:
        db_table_name = f"{prefix}_projection_coords"

    out_dir = _resolve_path(output_dir) if output_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    dist_mat, labels, record_ids, medoid_ids = load_cluster_and_distance_data(
        source_db_path=source_db_path,
        labels_db_path=labels_db_path,
        distance_config=distance_config,
        algorithm=algorithm,
        k=k,
        cluster_table=cluster_table,
        distance_table=distance_table,
    )

    medoid_mask = np.isin(record_ids, medoid_ids) if medoid_ids else None

    figures: Dict[str, plt.Figure] = {}
    projection_data: Dict[str, Any] = {
        "medical_record_id": record_ids,
        "cluster_id": labels.values,
    }

    # 2. PCoA Ordination (2D and 3D)
    if generate_pcoa:
        coords_pcoa_3d, var_pcts = compute_pcoa_coordinates(dist_mat, n_components=3)
        coords_pcoa_2d = coords_pcoa_3d[:, :2]
        var_pct_1, var_pct_2, var_pct_3 = var_pcts[0], var_pcts[1], var_pcts[2]

        projection_data["pcoa_dim1"] = coords_pcoa_3d[:, 0]
        projection_data["pcoa_dim2"] = coords_pcoa_3d[:, 1]
        projection_data["pcoa_dim3"] = coords_pcoa_3d[:, 2]

        # 2D PCoA
        fig_pcoa_2d_path = os.path.join(out_dir, f"2d_pcoa_{prefix}.png") if out_dir else None
        fig_pcoa_2d = plot_cluster_pcoa(
            coords=coords_pcoa_2d,
            cluster_labels=labels,
            var_pct_1=var_pct_1,
            var_pct_2=var_pct_2,
            medoid_mask=medoid_mask,
            output_filepath=fig_pcoa_2d_path,
            cluster_config_path=cluster_config_path,
            title_suffix=f"({prefix.upper()})",
        )
        figures["pcoa_2d"] = fig_pcoa_2d
        figures["pcoa"] = fig_pcoa_2d  # backward compatibility alias

        # 3D PCoA
        fig_pcoa_3d_path = os.path.join(out_dir, f"3d_pcoa_{prefix}.png") if out_dir else None
        fig_pcoa_3d = plot_cluster_pcoa(
            coords=coords_pcoa_3d,
            cluster_labels=labels,
            var_pct_1=var_pct_1,
            var_pct_2=var_pct_2,
            var_pct_3=var_pct_3,
            medoid_mask=medoid_mask,
            output_filepath=fig_pcoa_3d_path,
            cluster_config_path=cluster_config_path,
            title_suffix=f"({prefix.upper()})",
        )
        figures["pcoa_3d"] = fig_pcoa_3d

        # Save duplicate alias with '2d_pca' / '3d_pca' naming to satisfy both conventions
        if out_dir:
            fig_pcoa_2d.savefig(os.path.join(out_dir, f"2d_pca_{prefix}.png"), dpi=300, bbox_inches="tight")
            fig_pcoa_3d_extra = [fig_pcoa_3d.axes[0].zaxis.label]
            if fig_pcoa_3d.axes[0].get_legend():
                fig_pcoa_3d_extra.append(fig_pcoa_3d.axes[0].get_legend())
            fig_pcoa_3d.savefig(
                os.path.join(out_dir, f"3d_pca_{prefix}.png"),
                dpi=300,
                bbox_inches="tight",
                bbox_extra_artists=fig_pcoa_3d_extra,
            )

    # 3. Manifold Learning (UMAP / t-SNE in 2D and 3D)
    if generate_umap:
        # 2D Manifold
        coords_manifold_2d, method_name = compute_manifold_coordinates(dist_mat, n_components=2)
        m_name = method_name.lower().replace("-", "_")

        projection_data[f"{m_name}_dim1"] = coords_manifold_2d[:, 0]
        projection_data[f"{m_name}_dim2"] = coords_manifold_2d[:, 1]
        projection_data[f"{m_name}_2d_dim1"] = coords_manifold_2d[:, 0]
        projection_data[f"{m_name}_2d_dim2"] = coords_manifold_2d[:, 1]

        fig_manifold_2d_path = (
            os.path.join(out_dir, f"2d_{m_name}_{prefix}.png") if out_dir else None
        )
        fig_manifold_2d = plot_cluster_manifold(
            coords=coords_manifold_2d,
            cluster_labels=labels,
            method_name=method_name,
            medoid_mask=medoid_mask,
            output_filepath=fig_manifold_2d_path,
            cluster_config_path=cluster_config_path,
            title_suffix=f"({prefix.upper()})",
        )
        figures["manifold_2d"] = fig_manifold_2d
        figures["umap_2d"] = fig_manifold_2d
        figures["manifold"] = fig_manifold_2d  # backward compatibility alias
        figures["umap"] = fig_manifold_2d

        # 3D Manifold
        coords_manifold_3d, _ = compute_manifold_coordinates(dist_mat, n_components=3)
        projection_data[f"{m_name}_3d_dim1"] = coords_manifold_3d[:, 0]
        projection_data[f"{m_name}_3d_dim2"] = coords_manifold_3d[:, 1]
        projection_data[f"{m_name}_3d_dim3"] = coords_manifold_3d[:, 2]

        fig_manifold_3d_path = (
            os.path.join(out_dir, f"3d_{m_name}_{prefix}.png") if out_dir else None
        )
        fig_manifold_3d = plot_cluster_manifold(
            coords=coords_manifold_3d,
            cluster_labels=labels,
            method_name=method_name,
            medoid_mask=medoid_mask,
            output_filepath=fig_manifold_3d_path,
            cluster_config_path=cluster_config_path,
            title_suffix=f"({prefix.upper()})",
        )
        figures["manifold_3d"] = fig_manifold_3d
        figures["umap_3d"] = fig_manifold_3d

    projection_df = pd.DataFrame(projection_data)

    # 4. Save Coordinate Table to SQLite
    if save_to_db and output_db_path and db_table_name:
        out_db = _resolve_path(output_db_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
        with sqlite3.connect(out_db) as conn:
            projection_df.to_sql(db_table_name, conn, if_exists="replace", index=False)
        print(f"  [OK] Saved 2D & 3D Projection Coordinates: '{db_table_name}' in '{out_db}'")

    print(f"\n[Projection Complete] Successfully generated {len(figures)} 2D & 3D ordination plots in '{out_dir or 'memory'}'.\n")
    return projection_df, figures


# =============================================================================
# CLI TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in_2025.sqlite")
    out_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_out_2025.sqlite")
    out_dir = os.path.join(base_dir, "p1_outputs", "cluster_projections_test")

    df_proj, figs = run_cluster_projection_pipeline(
        source_db_path=in_db,
        labels_db_path=in_db,
        output_db_path=out_db,
        output_dir=out_dir,
        distance_config="wgc07_eb03",
        algorithm="hclust_complete",
        k=5,
        save_to_db=True,
    )
