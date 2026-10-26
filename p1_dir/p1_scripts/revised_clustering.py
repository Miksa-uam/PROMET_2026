r"""
================================================================================
REVISED CLUSTERING & ROBUST EVALUATION ENGINE FOR CLINICAL PHENOTYPING
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module executes and rigorously compares multiple clustering paradigms across
precomputed clinical distance matrices (such as the two-block Jaccard-Gower hybrid
matrices). It conducts a multi-dimensional mathematical evaluation of cluster validity
and stability, designations of exemplar patients (medoids), and hierarchical tree
structure visualizations.

Supported Algorithms:
---------------------
1. 'pam' (Partitioning Around Medoids / k-medoids):
   - Algorithm: Exact PAM swapping heuristic (`sklearn_extra.cluster.KMedoids`).
   - Seeding: Multi-start `k-medoids++` probabilistic initialization across N random seeds,
     selecting the solution that minimizes total inertia.
   - Paradigm: Global flat partitioning into k Voronoi-like cells around observed patient exemplars.

2. 'hclust_complete' (Hierarchical Agglomerative Clustering — Complete Linkage):
   - Linkage: Farthest neighbor / maximum pairwise dissimilarity (`scipy.cluster.hierarchy` / `sklearn`).
   - Paradigm: Bottom-up agglomerative tree merging based on compact, spherical cluster bounds.
   - Mathematical Property: Operates strictly on pairwise distances without assuming Euclidean
     geometry or coordinate centroids, avoiding the Euclidean embeddability concerns of Ward's method.
   - Preserves sparse binary signatures (e.g. distinct weight gain causes) as isolated branches.

3. 'hclust_average' (Hierarchical Agglomerative Clustering — Average Linkage / UPGMA):
   - Linkage: Unweighted Pair Group Method with Arithmetic Mean (mean pairwise distance).
   - Paradigm: Agglomerates clusters based on average mutual dissimilarity between all member pairs.
   - Mathematical Property: Operates strictly on pairwise distances without centroid coordinates.

4. Extensible Architecture:
   - Designed with an algorithm registry (`ALGORITHM_REGISTRY`) allowing seamless addition
     of future clustering algorithms (e.g., HDBSCAN, Spectral, or other linkages).

Mathematical & Methodological Evaluation Framework:
---------------------------------------------------
1. Native Precomputed Silhouette Score:
   Directly evaluates the balance between within-cluster cohesion and nearest-cluster separation
   on the exact precomputed metric space without dimensionality projection.
2. Inertia (Within-Cluster Dissimilarity to Medoid):
   Computed uniformly across both PAM and Hierarchical algorithms by identifying the true empirical
   medoid for each cluster (the patient minimizing within-cluster distance sum).
3. Cluster Stability via Subsampling Without Replacement:
   Evaluated through Monte Carlo Subsample Perturbation (Monti et al., 2003; Hennig, 2007)
   using 80% subsampling without replacement across 100 iterations. Measures Adjusted Rand Index (ARI)
   between subsampled and full-cohort partitionings. Parallelized across all CPU cores.
4. Hierarchical Dendrogram Visualizations:
   Generates cophenetic distance tree plots with color-coded cluster branches and cutoff thresholds.

Flexible Input Matrix Selection:
--------------------------------
The input distance matrix table(s) can be passed directly from notebook call scripts
(as a single table name string, a list of table names, or a dictionary), completely
decoupling execution from hardcoded database backends.

Pseudocode:
-----------
1. Ingest specified distance matrix table(s) from SQLite.
2. For each distance matrix:
     a. If hierarchical algorithms are enabled, compute and plot cophenetic dendrograms.
     b. Compute 10D Metric MDS coordinates once for auxiliary metric calculation.
     c. For each clustering algorithm (PAM, Complete Linkage, Average Linkage):
          For each k in k_range:
             i. Fit algorithm on distance matrix.
            ii. Extract cluster labels and identify cluster medoid patients.
           iii. Compute native Silhouette, Inertia, and auxiliary MDS indices.
            iv. Run 100-iteration 80% subsampling stability (parallelized).
             v. Log performance metrics and cluster assignments.
3. Save unified performance table and unified patient labels table to SQLite.
4. Generate multi-panel evaluation profile figures.

Author: Clinical Data Science Pair Programmer
Date: October 2026
Target Environment: Python 3.10+ / SQLite3 / NumPy / SciPy / scikit-learn / joblib / matplotlib
================================================================================
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from joblib import Parallel, delayed

# Ensure standard UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# =============================================================================
# CONSTANTS & DEFAULT CONFIGURATIONS
# =============================================================================

# Default cluster range (k = 2 to 8)
DEFAULT_K_RANGE: List[int] = list(range(2, 9))

# Default multi-start random initializations for PAM
DEFAULT_PAM_RUNS: int = 10

# Default Monte Carlo subsampling stability parameters (80% without replacement, 100 iterations)
DEFAULT_N_SUBSAMPLES: int = 100
DEFAULT_SUBSAMPLE_RATIO: float = 0.80

# Default hybrid distance tables in SQLite (generated by revised_hybrid_matrix.py)
DEFAULT_DISTANCE_TABLES: List[str] = [
    "hybrid_wgc05_eb05",
    "hybrid_wgc06_eb04",
    "hybrid_wgc067_eb033",
    "hybrid_wgc07_eb03",
]

# Default table names for database persistence
DEFAULT_PERFORMANCE_TABLE: str = "clustering_performance_scores"
DEFAULT_LABELS_TABLE: str = "clustering_labels"


# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================

def load_distance_matrix_from_db(
    db_path: str,
    table_name: str,
    fallback_db_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Loads a saved square distance matrix from SQLite, extracting patient identifiers and average distances.
    Conventionally looks in the input database (db_path), falling back to fallback_db_path if necessary.

    Parameters:
    -----------
    db_path : str
        Primary SQLite database path (conventionally the input database).
    table_name : str
        Name of the matrix table to load.
    fallback_db_path : Optional[str]
        Secondary SQLite database path to search if table is not found in db_path.

    Returns:
    --------
    Tuple[np.ndarray, List[str], np.ndarray]:
        - dist_matrix: Symmetric N x N float64 NumPy array.
        - medical_record_ids: List of N patient medical record identifiers.
        - avg_distances: Array of length N containing mean distances to all other patients.
    """
    target_db = db_path
    if not os.path.exists(target_db):
        if fallback_db_path and os.path.exists(fallback_db_path):
            target_db = fallback_db_path
        else:
            raise FileNotFoundError(f"Database not found at path: {db_path}")

    # Check if table exists in target_db
    with sqlite3.connect(target_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = cursor.fetchone()[0] > 0

    if not exists:
        if fallback_db_path and os.path.exists(fallback_db_path):
            with sqlite3.connect(fallback_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if cursor.fetchone()[0] > 0:
                    print(f"[Data Loading] Table '{table_name}' found in fallback database: '{fallback_db_path}'.")
                    target_db = fallback_db_path
                    exists = True

    if not exists:
        raise ValueError(f"Table Error: Distance matrix table '{table_name}' not found in '{db_path}' (or fallback).")

    print(f"\n[Data Loading] Loading distance matrix from table '{table_name}' in '{target_db}'...")
    with sqlite3.connect(target_db) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    if "medical_record_id" not in df.columns:
        raise ValueError(f"Schema Error: Table '{table_name}' missing required column 'medical_record_id'.")

    medical_record_ids = df["medical_record_id"].astype(str).tolist()

    if "avg_dist_to_all" in df.columns:
        avg_distances = df["avg_dist_to_all"].to_numpy(dtype=np.float64)
        dist_df = df.drop(columns=["medical_record_id", "avg_dist_to_all"])
    else:
        dist_df = df.drop(columns=["medical_record_id"])
        avg_distances = np.zeros(len(medical_record_ids))

    dist_matrix = dist_df.to_numpy(dtype=np.float64)
    n_patients = len(medical_record_ids)

    if dist_matrix.shape != (n_patients, n_patients):
        raise ValueError(
            f"Shape Error: Matrix shape {dist_matrix.shape} does not match patient count ({n_patients}, {n_patients})."
        )

    # Check symmetry and zero diagonal
    if not np.allclose(dist_matrix, dist_matrix.T, atol=1e-7):
        print("  [Warning] Distance matrix is slightly asymmetric; symmetrizing via (D + D.T) / 2.")
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    print(f"[Data Loading] Successfully loaded symmetric matrix of shape {dist_matrix.shape} for {n_patients} patients.")
    return dist_matrix, medical_record_ids, avg_distances


# =============================================================================
# CLUSTERING ALGORITHMS ENGINE & MEDOID COMPUTATION
# =============================================================================

def compute_cluster_medoids_and_inertia(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Identifies the empirical medoid for each cluster and calculates total inertia.

    Mathematical Logic:
    -------------------
    For each cluster c in {0, ..., k-1}:
    The medoid is the patient m_c within cluster c who minimizes the sum of distances
    to all other patients in cluster c:
        m_c = argmin_{i in C_c} sum_{j in C_c} D(i, j)

    Total Inertia (within-cluster sum of distances to medoids):
        Inertia = sum_{c=0}^{k-1} sum_{i in C_c} D(i, m_c)

    This provides a unified, directly comparable definition of cluster compactness
    across both PAM and hierarchical clustering algorithms.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        (N x N) distance matrix.
    labels : np.ndarray
        (N,) cluster assignment labels in {0, ..., k-1}.

    Returns:
    --------
    Tuple[np.ndarray, float]:
        - medoid_indices: Array of patient indices (length k) representing cluster medoids.
        - total_inertia: Float sum of within-cluster distances to cluster medoids.
    """
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)
    medoid_indices = np.zeros(k, dtype=int)
    total_inertia = 0.0

    for idx, c in enumerate(unique_clusters):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        if len(members) == 1:
            medoid_indices[idx] = members[0]
            continue

        # Sub-distance matrix for members of cluster c
        sub_dist = dist_matrix[np.ix_(members, members)]
        within_sums = sub_dist.sum(axis=1)
        best_local_idx = np.argmin(within_sums)
        best_patient_idx = members[best_local_idx]
        medoid_indices[idx] = best_patient_idx

        # Accumulate distance from all members to this medoid
        total_inertia += float(within_sums[best_local_idx])

    return medoid_indices, total_inertia


# --- 1. PAM Algorithm ---

def fit_single_pam(
    distance_matrix: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Fits PAM for a single seed using k-medoids++ initialization."""
    kmedoids = KMedoids(
        n_clusters=k,
        metric="precomputed",
        method="pam",
        init="k-medoids++",
        random_state=seed,
        max_iter=max_iter,
    )
    kmedoids.fit(distance_matrix)
    return kmedoids.labels_, kmedoids.medoid_indices_, float(kmedoids.inertia_), seed


def fit_pam(
    distance_matrix: np.ndarray,
    k: int,
    n_runs: int = DEFAULT_PAM_RUNS,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fits PAM across n_runs random seeds with k-medoids++ and returns the minimum inertia solution.
    """
    if seed is not None:
        return fit_single_pam(distance_matrix, k, seed=seed)

    best_inertia = np.inf
    best_labels = None
    best_medoids = None
    best_seed = -1

    for s in range(n_runs):
        labels, medoids, inertia, current_seed = fit_single_pam(distance_matrix, k, seed=s)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_medoids = medoids
            best_seed = current_seed

    return best_labels, best_medoids, best_inertia, best_seed


# --- 2. Hierarchical Complete Linkage ---

def fit_hclust_complete(
    distance_matrix: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fits Hierarchical Agglomerative Clustering with Complete Linkage (Farthest Neighbor).
    Identifies empirical cluster medoids and calculates total inertia for consistency with PAM.
    """
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="complete",
    )
    labels = model.fit_predict(distance_matrix)
    medoids, inertia = compute_cluster_medoids_and_inertia(distance_matrix, labels)
    return labels, medoids, inertia, 0


# --- 3. Hierarchical Average Linkage ---

def fit_hclust_average(
    distance_matrix: np.ndarray,
    k: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fits Hierarchical Agglomerative Clustering with Average Linkage (UPGMA).
    Identifies empirical cluster medoids and calculates total inertia for consistency with PAM.
    """
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distance_matrix)
    medoids, inertia = compute_cluster_medoids_and_inertia(distance_matrix, labels)
    return labels, medoids, inertia, 0


# --- Algorithm Registry for Extensibility ---

ALGORITHM_REGISTRY: Dict[str, Dict[str, any]] = {
    "pam": {
        "name": "Partitioning Around Medoids (k-medoids++)",
        "fit_fn": fit_pam,
        "is_probabilistic": True,
        "color": "#1b9e77",  # Teal
    },
    "hclust_complete": {
        "name": "Hierarchical Complete Linkage (Farthest Neighbor)",
        "fit_fn": fit_hclust_complete,
        "is_probabilistic": False,
        "color": "#d95f02",  # Orange
    },
    "hclust_average": {
        "name": "Hierarchical Average Linkage (UPGMA)",
        "fit_fn": fit_hclust_average,
        "is_probabilistic": False,
        "color": "#7570b3",  # Purple
    },
}


# =============================================================================
# EVALUATION METRICS: SILHOUETTE, INERTIA & MDS INDICES
# =============================================================================

def compute_cluster_metrics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    mds_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes mathematical cluster validity metrics.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        (N x N) precomputed distance matrix.
    labels : np.ndarray
        Cluster assignment labels.
    mds_coords : Optional[np.ndarray]
        Precomputed 10D MDS coordinates for auxiliary DBI/CHI computation.

    Returns:
    --------
    Dict[str, float]:
        Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin_mds": np.nan, "calinski_harabasz_mds": np.nan}

    # 1. Native Silhouette Score on precomputed distance matrix
    sil_score = float(silhouette_score(distance_matrix, labels, metric="precomputed"))

    # 2. Auxiliary MDS-based metrics
    if mds_coords is not None:
        try:
            db_score = float(davies_bouldin_score(mds_coords, labels))
            ch_score = float(calinski_harabasz_score(mds_coords, labels))
        except Exception:
            db_score, ch_score = np.nan, np.nan
    else:
        db_score, ch_score = np.nan, np.nan

    return {
        "silhouette": sil_score,
        "davies_bouldin_mds": db_score,
        "calinski_harabasz_mds": ch_score,
    }


# =============================================================================
# SUBSAMPLING STABILITY ASSESSMENT (PARALLELIZED)
# =============================================================================

def _single_subsample_worker(
    algo_key: str,
    distance_matrix: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    subsample_size: int,
    subsample_seed: int,
) -> float:
    """Helper worker evaluating Adjusted Rand Index on an 80% subsample without replacement."""
    rng = np.random.RandomState(subsample_seed)
    n_samples = distance_matrix.shape[0]

    # Sample without replacement
    sub_indices = rng.choice(n_samples, size=subsample_size, replace=False)
    sub_indices.sort()
    sub_matrix = distance_matrix[np.ix_(sub_indices, sub_indices)]

    # Fit algorithm on sub-matrix
    fit_fn = ALGORITHM_REGISTRY[algo_key]["fit_fn"]
    if ALGORITHM_REGISTRY[algo_key]["is_probabilistic"]:
        sub_labels, _, _, _ = fit_fn(sub_matrix, k, seed=subsample_seed)
    else:
        sub_labels, _, _, _ = fit_fn(sub_matrix, k)

    # Compare to ground-truth full-cohort labels on the shared patients
    true_sub_labels = full_labels[sub_indices]
    ari = float(adjusted_rand_score(true_sub_labels, sub_labels))
    return ari


def evaluate_cluster_stability_subsampling(
    algo_key: str,
    distance_matrix: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO,
    n_jobs: int = -1,
) -> Tuple[float, float, List[float]]:
    """
    Assesses cluster stability via Monte Carlo subsampling without replacement (Monti et al., 2003).
    Parallelized across all CPU cores.
    """
    n_samples = distance_matrix.shape[0]
    subsample_size = int(np.round(n_samples * subsample_ratio))

    ari_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_subsample_worker)(
            algo_key=algo_key,
            distance_matrix=distance_matrix,
            full_labels=full_labels,
            k=k,
            subsample_size=subsample_size,
            subsample_seed=iter_idx * 100 + k,
        )
        for iter_idx in range(n_subsamples)
    )

    mean_ari = float(np.mean(ari_scores))
    std_ari = float(np.std(ari_scores))
    return mean_ari, std_ari, ari_scores


# =============================================================================
# HIERARCHICAL DENDROGRAM VISUALIZATION
# =============================================================================

def plot_hierarchical_dendrograms(
    dist_matrix: np.ndarray,
    table_name: str,
    output_dir: Optional[str] = None,
    k_cutoff: int = 6,
) -> None:
    """
    Generates and saves side-by-side hierarchical dendrograms for Complete and Average Linkage.

    Parameters:
    -----------
    dist_matrix : np.ndarray
        (N x N) distance matrix.
    table_name : str
        Name of the distance matrix table (used for titles and filenames).
    output_dir : Optional[str]
        Directory to save the resulting image file.
    k_cutoff : int
        Number of clusters to highlight with horizontal threshold line (default: 6).
    """
    condensed = squareform(dist_matrix)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    methods = [
        ("complete", "Complete Linkage (Farthest Neighbor)", axes[0]),
        ("average", "Average Linkage (UPGMA)", axes[1]),
    ]

    for method, title, ax in methods:
        Z = linkage(condensed, method=method)
        # Determine height threshold corresponding to k_cutoff
        # In linkage matrix Z of shape (N-1, 4), the merge producing k clusters is step (N - k)
        cut_step = len(Z) - k_cutoff
        if 0 <= cut_step < len(Z):
            cut_height = Z[cut_step, 2]
        else:
            cut_height = Z[-1, 2] * 0.7

        dendrogram(
            Z,
            ax=ax,
            color_threshold=cut_height,
            above_threshold_color="#555555",
            no_labels=True,
        )
        ax.axhline(cut_height, color="#d95f02", linestyle="--", linewidth=1.5, label=f"Threshold (k≈{k_cutoff}, h={cut_height:.3f})")
        ax.set_title(f"{title}\nDistance Matrix: '{table_name}'", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Cohort Patients (N={dist_matrix.shape[0]})", fontsize=10)
        ax.set_ylabel("Cophenetic Distance", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=9, loc="upper right")

    plt.suptitle(f"Hierarchical Clustering Dendrograms across Patient Cohort ({table_name})", fontsize=13, y=1.02)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"dendrogram_{table_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plotting] Dendrogram saved to: {save_path}")

    # Display in Jupyter notebook if interactive
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()


# =============================================================================
# MULTI-PANEL EVALUATION PROFILES PLOTTER
# =============================================================================

def plot_clustering_evaluation_profiles(
    performance_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Generates multi-panel comparative evaluation plots comparing algorithms and distance configurations
    across Silhouette, Inertia, and Subsampling Stability ARI.
    """
    algorithms = performance_df["algorithm"].unique()
    tables = performance_df["distance_table"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    algo_styles = {
        "pam": {"marker": "o", "linestyle": "-", "name": "PAM (k-medoids++)"},
        "hclust_complete": {"marker": "s", "linestyle": "--", "name": "Complete Linkage"},
        "hclust_average": {"marker": "^", "linestyle": "-.", "name": "Average Linkage"},
    }

    # Distinct colors per distance table
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    table_colors = {t: palette[i % len(palette)] for i, t in enumerate(tables)}

    # Panel 1: Silhouette Score (Higher is Better)
    ax1 = axes[0]
    for algo in algorithms:
        style = algo_styles.get(algo, {"marker": "o", "linestyle": "-", "name": algo})
        for tbl in tables:
            sub = performance_df[(performance_df["algorithm"] == algo) & (performance_df["distance_table"] == tbl)].sort_values("k")
            if len(sub) == 0:
                continue
            color = table_colors[tbl]
            lbl = f"{style['name']} [{tbl.replace('hybrid_', '')}]"
            ax1.plot(sub["k"], sub["silhouette"], marker=style["marker"], linestyle=style["linestyle"], color=color, label=lbl, alpha=0.85)

    ax1.set_title("A. Native Precomputed Silhouette Score\n(Higher is Better)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Mean Silhouette Coefficient", fontsize=10)
    ax1.set_xticks(performance_df["k"].unique())
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: Total Inertia (Elbow Method — Lower is Better)
    ax2 = axes[1]
    for algo in algorithms:
        style = algo_styles.get(algo, {"marker": "o", "linestyle": "-", "name": algo})
        for tbl in tables:
            sub = performance_df[(performance_df["algorithm"] == algo) & (performance_df["distance_table"] == tbl)].sort_values("k")
            if len(sub) == 0:
                continue
            color = table_colors[tbl]
            lbl = f"{style['name']} [{tbl.replace('hybrid_', '')}]"
            ax2.plot(sub["k"], sub["inertia"], marker=style["marker"], linestyle=style["linestyle"], color=color, label=lbl, alpha=0.85)

    ax2.set_title("B. Total Cluster Inertia (Elbow Method)\n(Sum of Distances to Medoids — Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Inertia", fontsize=10)
    ax2.set_xticks(performance_df["k"].unique())
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="upper right")

    # Panel 3: Stability ARI (Higher is Better)
    ax3 = axes[2]
    for algo in algorithms:
        style = algo_styles.get(algo, {"marker": "o", "linestyle": "-", "name": algo})
        for tbl in tables:
            sub = performance_df[(performance_df["algorithm"] == algo) & (performance_df["distance_table"] == tbl)].sort_values("k")
            if len(sub) == 0:
                continue
            color = table_colors[tbl]
            lbl = f"{style['name']} [{tbl.replace('hybrid_', '')}]"
            ax3.errorbar(
                sub["k"],
                sub["stability_ari_mean"],
                yerr=sub["stability_ari_std"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=color,
                capsize=3,
                label=lbl,
                alpha=0.85,
            )

    ax3.axhline(0.75, color="red", linestyle=":", linewidth=1.5, label="High Stability Threshold (0.75)")
    ax3.set_title("C. Subsampling Stability (Adjusted Rand Index)\n(80% Subsampling Without Replacement — 100 Iterations)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax3.set_ylabel("Mean Adjusted Rand Index (ARI)", fontsize=10)
    ax3.set_xticks(performance_df["k"].unique())
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=8, loc="lower left")

    plt.suptitle("Multi-Algorithm Cluster Validity & Stability Comparison", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plotting] Comparative evaluation figure saved to: {save_path}")

    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()


# =============================================================================
# DATABASE PERSISTENCE FOR PERFORMANCE AND LABELS
# =============================================================================

def save_clustering_results_to_sqlite(
    all_runs: List[Dict],
    input_db_path: str,
    output_db_path: str,
    performance_table: str = DEFAULT_PERFORMANCE_TABLE,
    labels_table: str = DEFAULT_LABELS_TABLE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Saves unified performance scores and patient cluster labels into SQLite.
    """
    print("\n" + "=" * 80)
    print("SAVING UNIFIED CLUSTERING RESULTS TO SQLITE")
    print("=" * 80)

    # 1. Performance Scores Table
    all_scores: List[Dict] = []
    for run in all_runs:
        all_scores.extend(run["scores"])
    performance_df = pd.DataFrame(all_scores)

    # 2. Patient Labels Table
    with sqlite3.connect(input_db_path) as conn:
        base_df = pd.read_sql_query(
            "SELECT patient_id, medical_record_id FROM timetoevent_eb_wgc_compl ORDER BY medical_record_id",
            conn,
        )
    labels_df = base_df.copy()

    for run in all_runs:
        tbl_name = run["distance_table"]
        algo = run["algorithm"]
        med_ids = run["medical_record_ids"]
        labels_by_k = run["labels_by_k"]
        medoids_by_k = run["medoids_by_k"]

        id_to_idx = {rid: i for i, rid in enumerate(med_ids)}

        for k, lbls in labels_by_k.items():
            # Standard column naming: {table}_{algo}_k{k} and {table}_{algo}_medoid_k{k}
            prefix = f"{tbl_name}_{algo}"
            col_label = f"{prefix}_k{k}"
            col_medoid = f"{prefix}_medoid_k{k}"

            labels_df[col_label] = labels_df["medical_record_id"].map(
                lambda rid: int(lbls[id_to_idx[rid]]) if rid in id_to_idx else -1
            )

            medoid_indices = medoids_by_k[k]
            medoid_ids = set([med_ids[idx] for idx in medoid_indices if idx < len(med_ids)])
            labels_df[col_medoid] = labels_df["medical_record_id"].map(
                lambda rid: 1 if rid in medoid_ids else 0
            )

    with sqlite3.connect(output_db_path) as conn:
        performance_df.to_sql(performance_table, conn, if_exists="replace", index=False)
        labels_df.to_sql(labels_table, conn, if_exists="replace", index=False)

    print(f"[Persistence] Saved '{performance_table}' ({len(performance_df)} rows) into '{output_db_path}'.")
    print(f"[Persistence] Saved '{labels_table}' ({len(labels_df)} patients, {len(labels_df.columns)} columns) into '{output_db_path}'.")

    return performance_df, labels_df


# =============================================================================
# MASTER PIPELINE DRIVER
# =============================================================================

def run_clustering_pipeline(
    input_db_path: str,
    output_db_path: Optional[str] = None,
    distance_tables: Optional[Union[str, List[str], Dict[str, str]]] = None,
    algorithms: Optional[List[str]] = None,
    k_range: List[int] = DEFAULT_K_RANGE,
    n_pam_runs: int = DEFAULT_PAM_RUNS,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    plot_dendrograms: bool = True,
    output_dir: Optional[str] = None,
    performance_table_name: str = DEFAULT_PERFORMANCE_TABLE,
    labels_table_name: str = DEFAULT_LABELS_TABLE,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master driver function to execute clustering and evaluation across specified distance matrices.

    Parameters:
    -----------
    input_db_path : str
        Path to SQLite database containing cohort table 'timetoevent_eb_wgc_compl'.
    output_db_path : Optional[str]
        Path to database containing distance matrices and receiving results (defaults to input_db_path).
    distance_tables : Optional[Union[str, List[str], Dict[str, str]]]
        Distance matrix table(s) to process. Can be:
        - A single table name (str), e.g. "hybrid_wgc067_eb033"
        - A list of table names, e.g. ["hybrid_wgc05_eb05", "hybrid_wgc067_eb033"]
        - A dictionary mapping keys to table names
        - None (defaults to all 4 hybrid tables).
    algorithms : Optional[List[str]]
        List of algorithms to run. Supported: 'pam', 'hclust_complete', 'hclust_average'.
        Defaults to all three.
    k_range : List[int]
        Range of clusters to evaluate (default: 2 to 8).
    n_pam_runs : int
        Number of random initializations for PAM multi-start (default: 10).
    n_subsamples : int
        Number of subsampling stability iterations (default: 100).
    plot_dendrograms : bool
        Whether to generate and save hierarchical dendrograms (default: True).
    output_dir : Optional[str]
        Directory for saving plots (defaults to ../p1_outputs).
    performance_table_name : str
        SQLite table name for performance scores.
    labels_table_name : str
        SQLite table name for patient cluster labels.
    n_jobs : int
        Number of parallel CPU workers (-1 for all cores).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - performance_df: Benchmarking validity and stability scores across all runs.
        - labels_df: Patient-level cluster assignments and medoid designations.
    """
    if output_db_path is None:
        output_db_path = input_db_path

    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(base_dir, "..", "p1_outputs"))

    # Resolve distance tables list
    if distance_tables is None:
        resolved_tables = DEFAULT_DISTANCE_TABLES
    elif isinstance(distance_tables, str):
        resolved_tables = [distance_tables]
    elif isinstance(distance_tables, list):
        resolved_tables = distance_tables
    elif isinstance(distance_tables, dict):
        resolved_tables = list(distance_tables.values())
    else:
        raise TypeError(f"Invalid type for distance_tables: {type(distance_tables)}")

    # Resolve algorithms list
    if algorithms is None:
        algorithms = ["pam", "hclust_complete", "hclust_average"]
    for algo in algorithms:
        if algo not in ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm '{algo}'. Registered algorithms: {list(ALGORITHM_REGISTRY.keys())}")

    print("\n" + "#" * 80)
    print("STARTING MULTI-ALGORITHM CLUSTERING & ROBUST EVALUATION ENGINE")
    print("#" * 80)
    print(f"Input DB:             {input_db_path}")
    print(f"Output DB:            {output_db_path}")
    print(f"Distance Matrices:    {resolved_tables}")
    print(f"Algorithms:           {algorithms}")
    print(f"Cluster Range:        k in {k_range}")
    print(f"PAM Multi-Starts:     {n_pam_runs} initializations per k (k-medoids++)")
    print(f"Stability Assessment: {n_subsamples} iterations (80% subsampling without replacement)")
    print(f"Parallel Workers:     n_jobs = {n_jobs} (All available CPU cores)")
    print("#" * 80)

    all_runs: List[Dict] = []

    for tbl_name in resolved_tables:
        print("\n" + "=" * 80)
        print(f"PROCESSING DISTANCE MATRIX TABLE: '{tbl_name}'")
        print("=" * 80)

        # 1. Load distance matrix (primary: input_db_path, fallback: output_db_path)
        dist_matrix, medical_record_ids, avg_dists = load_distance_matrix_from_db(
            db_path=input_db_path,
            table_name=tbl_name,
            fallback_db_path=output_db_path,
        )
        n_patients = len(medical_record_ids)

        # 2. Plot Dendrograms if hierarchical algorithms are requested
        has_hierarchical = any("hclust" in a for a in algorithms)
        if plot_dendrograms and has_hierarchical:
            print(f"[Dendrogram] Plotting hierarchical cophenetic tree for '{tbl_name}'...")
            plot_hierarchical_dendrograms(dist_matrix, tbl_name, output_dir=output_dir, k_cutoff=6)

        # 3. Precompute 10D MDS coordinates once per distance matrix for auxiliary metrics
        print(f"[MDS Coordinates] Fitting 10D Metric MDS for auxiliary metrics...")
        try:
            mds = MDS(n_components=min(10, n_patients - 1), dissimilarity="precomputed", random_state=42, n_init=4)
            mds_coords = mds.fit_transform(dist_matrix)
        except Exception as e:
            print(f"[Warning] MDS embedding failed ({e}). Proceeding without auxiliary coordinate metrics.")
            mds_coords = None

        # 4. Iterate over each requested algorithm
        for algo_key in algorithms:
            algo_info = ALGORITHM_REGISTRY[algo_key]
            algo_name = algo_info["name"]
            fit_fn = algo_info["fit_fn"]

            print(f"\n>>> Running Algorithm: '{algo_name}' on '{tbl_name}' <<<")

            algo_scores: List[Dict] = []
            algo_labels_by_k: Dict[int, np.ndarray] = {}
            algo_medoids_by_k: Dict[int, np.ndarray] = {}

            for k in k_range:
                print(f"  Fitting k={k}...")
                if algo_key == "pam":
                    labels, medoids, inertia, best_seed = fit_fn(dist_matrix, k, n_runs=n_pam_runs)
                else:
                    labels, medoids, inertia, best_seed = fit_fn(dist_matrix, k)

                # Internal validity metrics
                metrics = compute_cluster_metrics(dist_matrix, labels, mds_coords)

                # Subsampling Stability (80% without replacement, parallelized)
                mean_ari, std_ari, _ = evaluate_cluster_stability_subsampling(
                    algo_key=algo_key,
                    distance_matrix=dist_matrix,
                    full_labels=labels,
                    k=k,
                    n_subsamples=n_subsamples,
                    subsample_ratio=DEFAULT_SUBSAMPLE_RATIO,
                    n_jobs=n_jobs,
                )

                cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()
                medoid_rids = [medical_record_ids[idx] for idx in medoids if idx < len(medical_record_ids)]

                print(f"    [k={k}] Silhouette: {metrics['silhouette']:.4f} | Inertia: {inertia:.2f} | Stability ARI: {mean_ari:.4f} (±{std_ari:.4f})")
                print(f"           Cluster sizes: {cluster_sizes}")

                algo_scores.append({
                    "algorithm": algo_key,
                    "distance_table": tbl_name,
                    "k": k,
                    "silhouette": metrics["silhouette"],
                    "inertia": inertia,
                    "stability_ari_mean": mean_ari,
                    "stability_ari_std": std_ari,
                    "davies_bouldin_mds": metrics["davies_bouldin_mds"],
                    "calinski_harabasz_mds": metrics["calinski_harabasz_mds"],
                    "best_seed": best_seed,
                    "cluster_sizes": str(cluster_sizes),
                })

                algo_labels_by_k[k] = labels
                algo_medoids_by_k[k] = medoids

            all_runs.append({
                "distance_table": tbl_name,
                "algorithm": algo_key,
                "medical_record_ids": medical_record_ids,
                "scores": algo_scores,
                "labels_by_k": algo_labels_by_k,
                "medoids_by_k": algo_medoids_by_k,
            })

    # Save to SQLite
    perf_df, lbls_df = save_clustering_results_to_sqlite(
        all_runs=all_runs,
        input_db_path=input_db_path,
        output_db_path=output_db_path,
        performance_table=performance_table_name,
        labels_table=labels_table_name,
    )

    # Generate comparative evaluation curves
    eval_fig_path = os.path.join(output_dir, "clustering_evaluation_profiles.png")
    plot_clustering_evaluation_profiles(perf_df, save_path=eval_fig_path)

    # Print summary performance comparison table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARKING TABLE: MATHEMATICAL VALIDITY & STABILITY")
    print("=" * 80)
    disp_cols = ["algorithm", "distance_table", "k", "silhouette", "inertia", "stability_ari_mean", "cluster_sizes"]
    print(perf_df[disp_cols].to_string(index=False))
    print("=" * 80 + "\n")

    return perf_df, lbls_df


# =============================================================================
# CLI EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in.sqlite")
    out_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_out.sqlite")
    out_dir = os.path.join(base_dir, "p1_outputs")

    # Example standalone run on the proportional 0.67/0.33 hybrid matrix across all 3 algorithms
    perf_df, lbls_df = run_clustering_pipeline(
        input_db_path=in_db,
        output_db_path=out_db,
        distance_tables=["hybrid_wgc067_eb033"],
        algorithms=["pam", "hclust_complete", "hclust_average"],
        k_range=list(range(2, 9)),
        n_pam_runs=10,
        n_subsamples=100,
        output_dir=out_dir,
        n_jobs=-1,
    )
