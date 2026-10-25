"""
================================================================================
REVISED PAM CLUSTERING & ROBUST EVALUATION ENGINE
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module executes Partitioning Around Medoids (PAM / k-medoids) clustering across
precomputed mixed-metric Gower distance matrices and conducts a multi-dimensional
evaluation of cluster validity and stability.

Methodological & Theoretical Justification (Addressing Reviewer Comments 3 & 4):
---------------------------------------------------------------------------------
1. Dissimilarity-Based Partitioning vs. Euclidean Embeddability (Comment 3d & 3e):
   Ward's minimum-variance hierarchical linkage theoretically assumes Euclidean-embeddable
   distances. Gower dissimilarity containing asymmetric binary components is a general
   metric/semi-metric space that is not guaranteed to satisfy isometric Euclidean embeddability.
   PAM operates natively on *any* valid symmetric dissimilarity matrix without coordinate
   projections, minimizing the sum of absolute dissimilarities to exemplar medoids.

2. Clinical Interpretability via Real Exemplar Patients (Medoids):
   Unlike k-means or Ward's virtual cluster centroids, PAM designates actual patients
   from the cohort as cluster centers (medoids). Clinicians can review the narrative
   etiologies, symptom severity scores, and clinical trajectories of these exemplar individuals.

3. Optimization Rigor (`k-medoids++` with Multi-Start):
   To avoid sub-optimal local minima, initialization uses the `k-medoids++` probabilistic
   seeding strategy across 10 random seeds per k, selecting the run that minimizes total inertia.

4. Internal Validity Metrics: Native vs. Projected (Comment 4a):
   - Silhouette Score: Computed natively and directly on the precomputed Gower matrix.
   - Inertia: Total within-cluster dissimilarity to medoids; serves as the native elbow metric.
   - Davies-Bouldin & Calinski-Harabasz: Computed on classical Metric MDS coordinates (10 components)
     solely for methodological completeness and comparison with historical literature, while
     highlighting that native Silhouette and Inertia remain the primary unprojected criteria.

5. Cluster Stability via Subsampling Without Replacement (Comment 4b):
   Resampling *with replacement* directly on a distance matrix produces identical "clone"
   patients (D=0.0), which artificially creates dense clusters and inflates stability scores.
   Instead, we implement Monte Carlo Subsample Perturbation (Monti et al., 2003; Hennig, 2007)
   using 80% subsampling *without replacement* over 100 iterations. This tests genuine
   phenotypic robustness against cohort variation without introducing duplicate records.
================================================================================
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from joblib import Parallel, delayed

# Ensure stdout encodes UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import MDS


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Range of clusters to evaluate
DEFAULT_K_RANGE: List[int] = list(range(2, 9))  # k = 2, 3, 4, 5, 6, 7, 8

# Number of random initializations per k
DEFAULT_N_RUNS: int = 10

# Subsampling stability parameters (80% subsampling without replacement)
DEFAULT_N_SUBSAMPLES: int = 100
DEFAULT_SUBSAMPLE_RATIO: float = 0.80

# Predefined distance matrix tables in SQLite (matching revised_gower_distance.py)
DISTANCE_MATRIX_CONFIGS: Dict[str, Dict[str, str]] = {
    "wgc05_eb05": {
        "table_name": "revised_gower_wgc05_eb05",
        "description": "Equal block weight (0.50 WGC / 0.50 EB)",
    },
    "wgc07_eb03": {
        "table_name": "revised_gower_wgc07_eb03",
        "description": "Proportional block weight (0.70 WGC / 0.30 EB)",
    },
    "wgc06_eb04": {
        "table_name": "revised_gower_wgc06_eb04",
        "description": "Intermediate block weight (0.60 WGC / 0.40 EB)",
    },
}

# Unified output table names
PERFORMANCE_TABLE_NAME: str = "pam_performance_scores"
LABELS_TABLE_NAME: str = "pam_labels"


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_distance_matrix_from_db(
    db_path: str,
    table_name: str,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Loads a saved square distance matrix from SQLite, extracting patient identifiers and average distances.

    Parameters:
    -----------
    db_path : str
        Path to SQLite database containing the matrix table.
    table_name : str
        Name of the matrix table.

    Returns:
    --------
    Tuple[np.ndarray, List[str], np.ndarray]:
        (dist_matrix, medical_record_ids, avg_distances)
    """
    print(f"[Data Loading] Loading distance matrix from table '{table_name}' in '{db_path}'...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at path: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    medical_record_ids = df["medical_record_id"].astype(str).tolist()
    avg_distances = df["avg_dist_to_all"].to_numpy(dtype=np.float64)

    # Drop non-distance columns to extract purely numeric square matrix
    dist_df = df.drop(columns=["medical_record_id", "avg_dist_to_all"])
    dist_matrix = dist_df.to_numpy(dtype=np.float64)

    n_patients = len(medical_record_ids)
    if dist_matrix.shape != (n_patients, n_patients):
        raise ValueError(
            f"Shape Error: Matrix shape {dist_matrix.shape} does not match patient count ({n_patients}, {n_patients})."
        )

    print(f"[Data Loading] Successfully loaded symmetric matrix of shape {dist_matrix.shape} for {n_patients} patients.")
    return dist_matrix, medical_record_ids, avg_distances


# =============================================================================
# CORE PAM CLUSTERING & MULTI-START OPTIMIZATION
# =============================================================================

def fit_single_pam(
    distance_matrix: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fits PAM for a single random seed using k-medoids++ initialization.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        (N x N) precomputed symmetric distance matrix.
    k : int
        Number of clusters.
    seed : int
        Random state seed.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, float, int]:
        (labels, medoid_indices, inertia, seed)
    """
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


def perform_multi_start_pam(
    distance_matrix: np.ndarray,
    k: int,
    n_runs: int = DEFAULT_N_RUNS,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Runs PAM across n_runs random seeds and returns the solution with minimum inertia.

    Programmatic & Mathematical Rationale:
    --------------------------------------
    The objective function of PAM is total inertia:
        Inertia = sum_{c=1}^k sum_{i in Cluster_c} D(i, medoid_c)
    Because medoid-swapping heuristics can converge to local minima depending on initial
    medoid candidates, iterating across multiple distinct k-medoids++ random seeds
    guarantees finding the lowest-inertia clustering solution.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        (N x N) precomputed distance matrix.
    k : int
        Number of clusters.
    n_runs : int
        Number of random initializations.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, float, int]:
        (best_labels, best_medoids, best_inertia, best_seed)
    """
    best_inertia = np.inf
    best_labels = None
    best_medoids = None
    best_seed = -1

    for seed in range(n_runs):
        labels, medoids, inertia, s = fit_single_pam(distance_matrix, k, seed)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_medoids = medoids
            best_seed = s

    return best_labels, best_medoids, best_inertia, best_seed


# =============================================================================
# EVALUATION METRICS: NATIVE & AUXILIARY
# =============================================================================

def compute_cluster_metrics(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    mds_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes mathematical cluster validity metrics.

    Metrics:
    --------
    1. Silhouette Score: Native precomputed distance calculation.
    2. Davies-Bouldin Index (DBI): Computed on 10D MDS coordinates (lower is better).
    3. Calinski-Harabasz Index (CHI): Computed on 10D MDS coordinates (higher is better).

    Parameters:
    -----------
    distance_matrix : np.ndarray
        (N x N) precomputed distance matrix.
    labels : np.ndarray
        Cluster assignment labels.
    mds_coords : Optional[np.ndarray]
        Precomputed MDS coordinates for fast auxiliary evaluation.

    Returns:
    --------
    Dict[str, float]:
        Dictionary of computed validity indices.
    """
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1 or n_clusters >= len(labels):
        return {"silhouette": np.nan, "davies_bouldin_mds": np.nan, "calinski_harabasz_mds": np.nan}

    # 1. Native Silhouette Score
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
# ROBUST CLUSTER STABILITY EVALUATION (SUBSAMPLING WITHOUT REPLACEMENT)
# =============================================================================

def _single_subsample_iteration(
    distance_matrix: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    subsample_size: int,
    subsample_seed: int,
) -> float:
    """
    Helper worker function for a single subsampling stability iteration.
    Draws a random subsample without replacement, clusters the sub-matrix with k-medoids++,
    and calculates Adjusted Rand Index (ARI) against full-cohort labels on shared patients.
    """
    rng = np.random.RandomState(subsample_seed)
    n_samples = distance_matrix.shape[0]

    # Draw random patient indices strictly WITHOUT replacement (no clone patients!)
    sub_indices = rng.choice(n_samples, size=subsample_size, replace=False)
    sub_indices.sort()

    # Slice sub-matrix
    sub_matrix = distance_matrix[np.ix_(sub_indices, sub_indices)]

    # Single robust fit with k-medoids++ using the iteration's unique seed
    sub_labels, _, _, _ = fit_single_pam(sub_matrix, k, seed=subsample_seed)

    # Ground-truth labels of these specific patients from the full-cohort clustering
    true_sub_labels = full_labels[sub_indices]

    # Calculate Adjusted Rand Index between full-cohort partitioning and subsample partitioning
    ari = float(adjusted_rand_score(true_sub_labels, sub_labels))
    return ari


def evaluate_cluster_stability_subsampling(
    distance_matrix: np.ndarray,
    full_labels: np.ndarray,
    k: int,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    subsample_ratio: float = DEFAULT_SUBSAMPLE_RATIO,
    n_jobs: int = -1,
) -> Tuple[float, float, List[float]]:
    """
    Assesses cluster stability via Monte Carlo subsampling without replacement.

    Scientific & Methodological Rationale:
    --------------------------------------
    Unlike bootstrap resampling with replacement—which duplicates distance matrix rows/columns
    and creates artificial clone clusters—subsampling without replacement (Monti et al., 2003)
    evaluates whether clusters reproduce reliably when 15-20% of the patient population is absent.
    For small clusters of 15-30 patients, an 80% subsample preserves 12-24 patients per iteration,
    providing ample statistical power to test if the phenotypic boundary is genuine or brittle.

    Parameters:
    -----------
    distance_matrix : np.ndarray
        (N x N) distance matrix.
    full_labels : np.ndarray
        Cluster labels from the full-cohort run.
    k : int
        Number of clusters.
    n_subsamples : int
        Number of random subsample iterations (default: 100).
    subsample_ratio : float
        Fraction of cohort to retain per subsample (default: 0.80).
    n_jobs : int
        CPU cores to utilize for parallel execution (default: -1 for all cores).

    Returns:
    --------
    Tuple[float, float, List[float]]:
        (mean_ari, std_ari, all_ari_scores)
    """
    n_samples = distance_matrix.shape[0]
    subsample_size = int(np.round(n_samples * subsample_ratio))

    # Parallelize across all CPU cores
    ari_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_single_subsample_iteration)(
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
# SINGLE-CONFIGURATION WORKER PIPELINE
# =============================================================================

def process_single_configuration(
    config_key: str,
    table_name: str,
    db_path: str,
    k_range: List[int] = DEFAULT_K_RANGE,
    n_runs: int = DEFAULT_N_RUNS,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    n_jobs: int = -1,
) -> Dict:
    """
    Executes PAM clustering and stability evaluation across all k for one distance matrix.
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING DISTANCE CONFIGURATION: '{config_key}' (Table: {table_name})")
    print("=" * 80)

    # 1. Load distance matrix
    dist_matrix, medical_record_ids, avg_dists = load_distance_matrix_from_db(db_path, table_name)
    n_patients = len(medical_record_ids)

    # 2. Precompute 10D MDS coordinates once per distance matrix for auxiliary DBI/CHI metrics
    print(f"[MDS Coordinates] Fitting 10D Metric MDS for auxiliary metrics...")
    try:
        mds = MDS(n_components=min(10, n_patients - 1), dissimilarity="precomputed", random_state=42, n_init=4)
        mds_coords = mds.fit_transform(dist_matrix)
    except Exception as e:
        print(f"[Warning] MDS embedding failed ({e}). Proceeding without auxiliary coordinate metrics.")
        mds_coords = None

    config_scores: List[Dict] = []
    config_labels: Dict[int, np.ndarray] = {}
    config_medoids: Dict[int, np.ndarray] = {}

    for k in k_range:
        print(f"\n--- Fitting PAM for k={k} ({n_runs} initializations) ---")
        best_labels, best_medoids, best_inertia, best_seed = perform_multi_start_pam(
            distance_matrix=dist_matrix,
            k=k,
            n_runs=n_runs,
        )

        # Compute internal metrics
        metrics = compute_cluster_metrics(dist_matrix, best_labels, mds_coords)

        # Compute Subsampling Stability (80% subsampling without replacement)
        print(f"  Calculating Subsampling Stability ARI ({n_subsamples} iterations, 80% subsample, all cores)...")
        mean_ari, std_ari, _ = evaluate_cluster_stability_subsampling(
            distance_matrix=dist_matrix,
            full_labels=best_labels,
            k=k,
            n_subsamples=n_subsamples,
            subsample_ratio=DEFAULT_SUBSAMPLE_RATIO,
            n_jobs=n_jobs,
        )

        # Medoid patient mapping
        medoid_patient_ids = [medical_record_ids[idx] for idx in best_medoids]
        cluster_sizes = pd.Series(best_labels).value_counts().sort_index().to_dict()

        print(f"  [Results k={k}] Silhouette: {metrics['silhouette']:.4f} | Inertia: {best_inertia:.2f} | Stability ARI: {mean_ari:.4f} (±{std_ari:.4f})")
        print(f"  Cluster sizes: {cluster_sizes}")
        print(f"  Exemplar Medoids (Record IDs): {medoid_patient_ids}")

        # Store score row
        config_scores.append({
            "distance_config": config_key,
            "k": k,
            "silhouette": metrics["silhouette"],
            "inertia": best_inertia,
            "stability_ari_mean": mean_ari,
            "stability_ari_std": std_ari,
            "davies_bouldin_mds": metrics["davies_bouldin_mds"],
            "calinski_harabasz_mds": metrics["calinski_harabasz_mds"],
            "best_seed": best_seed,
            "cluster_sizes": str(cluster_sizes),
        })

        config_labels[k] = best_labels
        config_medoids[k] = best_medoids

    return {
        "config_key": config_key,
        "table_name": table_name,
        "medical_record_ids": medical_record_ids,
        "scores": config_scores,
        "labels_by_k": config_labels,
        "medoids_by_k": config_medoids,
    }


# =============================================================================
# PERSISTENCE & DATABASE INTEGRATION
# =============================================================================

def save_clustering_results_to_sqlite(
    all_config_results: List[Dict],
    input_db_path: str,
    output_db_path: str,
    performance_table: str = PERFORMANCE_TABLE_NAME,
    labels_table: str = LABELS_TABLE_NAME,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Saves unified performance scores and unified patient cluster labels into SQLite.

    Schema:
    -------
    1. pam_performance_scores:
       Rows: 3 distance configs * 7 k values = 21 benchmarking rows.
       Columns: distance_config, k, silhouette, inertia, stability_ari_mean, stability_ari_std,
                davies_bouldin_mds, calinski_harabasz_mds, best_seed, cluster_sizes.

    2. pam_labels:
       Rows: 312 patients.
       Columns: medical_record_id, patient_id,
                wgc05_eb05_k2..k8, wgc05_eb05_medoid_k2..k8,
                wgc07_eb03_k2..k8, wgc07_eb03_medoid_k2..k8,
                wgc06_eb04_k2..k8, wgc06_eb04_medoid_k2..k8.
    """
    print("\n" + "=" * 80)
    print("SAVING UNIFIED CLUSTERING RESULTS TO SQLITE")
    print("=" * 80)

    # 1. Assemble Unified Performance Scores DataFrame
    all_scores: List[Dict] = []
    for cfg_res in all_config_results:
        all_scores.extend(cfg_res["scores"])
    performance_df = pd.DataFrame(all_scores)

    # 2. Assemble Unified Labels DataFrame
    # Retrieve base identifiers from input database
    with sqlite3.connect(input_db_path) as conn:
        base_df = pd.read_sql_query(
            "SELECT patient_id, medical_record_id FROM timetoevent_eb_wgc_compl ORDER BY medical_record_id",
            conn,
        )

    labels_df = base_df.copy()

    for cfg_res in all_config_results:
        cfg_key = cfg_res["config_key"]
        med_ids = cfg_res["medical_record_ids"]
        labels_by_k = cfg_res["labels_by_k"]
        medoids_by_k = cfg_res["medoids_by_k"]

        id_to_idx = {rid: i for i, rid in enumerate(med_ids)}

        for k, lbls in labels_by_k.items():
            col_label = f"{cfg_key}_k{k}"
            col_medoid = f"{cfg_key}_medoid_k{k}"

            # Map cluster labels to DataFrame rows
            labels_df[col_label] = labels_df["medical_record_id"].map(
                lambda rid: int(lbls[id_to_idx[rid]]) if rid in id_to_idx else -1
            )

            # Map medoid flags (1 if patient is medoid for this k, 0 otherwise)
            medoid_indices = medoids_by_k[k]
            medoid_ids = set([med_ids[idx] for idx in medoid_indices])
            labels_df[col_medoid] = labels_df["medical_record_id"].map(
                lambda rid: 1 if rid in medoid_ids else 0
            )

    # Save both tables to output database
    with sqlite3.connect(output_db_path) as conn:
        performance_df.to_sql(performance_table, conn, if_exists="replace", index=False)
        labels_df.to_sql(labels_table, conn, if_exists="replace", index=False)

    print(f"[Persistence] Saved '{performance_table}' ({len(performance_df)} rows) into '{output_db_path}'.")
    print(f"[Persistence] Saved '{labels_table}' ({len(labels_df)} patients, {len(labels_df.columns)} columns) into '{output_db_path}'.")

    return performance_df, labels_df


# =============================================================================
# DIAGNOSTIC PLOTTING: RAW METRIC PROFILES (COMMENT 4A DEFENSE)
# =============================================================================

def plot_pam_evaluation_profiles(
    performance_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Generates multi-panel comparative evaluation plots displaying raw (non-normalized)
    Silhouette, Inertia, and Subsampling Stability ARI curves.

    Parameters:
    -----------
    performance_df : pd.DataFrame
        DataFrame containing performance scores.
    save_path : Optional[str]
        Optional file path to save the generated figure.
    """
    configs = performance_df["distance_config"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    config_colors = {
        "wgc05_eb05": "#1b9e77",  # Teal (Equal)
        "wgc07_eb03": "#d95f02",  # Orange (Proportional)
        "wgc06_eb04": "#7570b3",  # Purple (Intermediate)
    }
    config_labels_map = {
        "wgc05_eb05": "Equal (0.50 WGC / 0.50 EB)",
        "wgc07_eb03": "Proportional (0.70 WGC / 0.30 EB)",
        "wgc06_eb04": "Intermediate (0.60 WGC / 0.40 EB)",
    }

    # Panel 1: Native Precomputed Silhouette Score (Higher is Better)
    ax1 = axes[0]
    for cfg in configs:
        sub = performance_df[performance_df["distance_config"] == cfg].sort_values("k")
        color = config_colors.get(cfg, "black")
        lbl = config_labels_map.get(cfg, cfg)
        ax1.plot(sub["k"], sub["silhouette"], marker="o", linewidth=2.0, color=color, label=lbl)

    ax1.set_title("A. Native Gower Silhouette Score\n(Precomputed Dissimilarity — Higher is Better)", fontsize=11)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Mean Silhouette Coefficient", fontsize=10)
    ax1.set_xticks(performance_df["k"].unique())
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: Total Inertia / Within-Cluster Dissimilarity (Elbow Method — Lower is Better)
    ax2 = axes[1]
    for cfg in configs:
        sub = performance_df[performance_df["distance_config"] == cfg].sort_values("k")
        color = config_colors.get(cfg, "black")
        lbl = config_labels_map.get(cfg, cfg)
        ax2.plot(sub["k"], sub["inertia"], marker="s", linewidth=2.0, linestyle="--", color=color, label=lbl)

    ax2.set_title("B. PAM Objective Function (Inertia)\n(Sum of Gower Dissimilarities to Medoids — Elbow)", fontsize=11)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Total Inertia", fontsize=10)
    ax2.set_xticks(performance_df["k"].unique())
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")

    # Panel 3: Subsampling Stability (Adjusted Rand Index — Higher is Better)
    ax3 = axes[2]
    for cfg in configs:
        sub = performance_df[performance_df["distance_config"] == cfg].sort_values("k")
        color = config_colors.get(cfg, "black")
        lbl = config_labels_map.get(cfg, cfg)
        ax3.errorbar(
            sub["k"],
            sub["stability_ari_mean"],
            yerr=sub["stability_ari_std"],
            marker="^",
            linewidth=2.0,
            capsize=4,
            color=color,
            label=lbl,
        )

    ax3.axhline(0.75, color="red", linestyle=":", linewidth=1.5, label="High Stability Threshold (0.75)")
    ax3.set_title("C. Cluster Stability (Adjusted Rand Index)\n(80% Subsampling Without Replacement — 100 Iterations)", fontsize=11)
    ax3.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax3.set_ylabel("Mean Adjusted Rand Index (ARI)", fontsize=10)
    ax3.set_xticks(performance_df["k"].unique())
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="lower left")

    plt.suptitle("Mathematical Cluster Validity & Stability Profiles across Gower Weighting Configurations", fontsize=13, y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plotting] Evaluation figure saved to: {save_path}")

    # Display plot in interactive environments (e.g. Jupyter notebook)
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            plt.show()
        else:
            plt.close()
    except Exception:
        plt.close()


# =============================================================================
# MASTER PIPELINE DRIVER
# =============================================================================

def run_pam_pipeline(
    input_db_path: str,
    output_db_path: Optional[str] = None,
    distance_configs: Optional[Dict[str, Dict[str, str]]] = None,
    k_range: List[int] = DEFAULT_K_RANGE,
    n_runs: int = DEFAULT_N_RUNS,
    n_subsamples: int = DEFAULT_N_SUBSAMPLES,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the comprehensive PAM clustering pipeline across all 3 distance matrices.

    Workflow:
    ---------
    1. Sequentially processes each distance configuration (wgc05_eb05, wgc07_eb03, wgc06_eb04).
    2. For each k in k_range:
       - Multi-start PAM (10 runs with k-medoids++).
       - Native precomputed Silhouette score.
       - 100-iteration Subsampling Stability (80% without replacement, parallelized across all cores).
       - Auxiliary 10D MDS coordinate metrics (Davies-Bouldin, Calinski-Harabasz).
    3. Persists unified 'pam_performance_scores' and unified 'pam_labels' into SQLite.
    4. Generates publication-ready comparative validation plots.

    Parameters:
    -----------
    input_db_path : str
        Path to source database containing timetoevent_eb_wgc_compl.
    output_db_path : Optional[str]
        Path to database containing distance matrices and receiving clustering results.
    distance_configs : Optional[Dict]
        Distance matrix configurations. Defaults to DISTANCE_MATRIX_CONFIGS.
    k_range : List[int]
        Range of k values (default: 2 to 8).
    n_runs : int
        Random initializations per k (default: 10).
    n_subsamples : int
        Subsampling stability iterations (default: 100).
    n_jobs : int
        CPU cores to use for parallelization (-1 for all cores).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (performance_df, labels_df)
    """
    if output_db_path is None:
        output_db_path = input_db_path

    if distance_configs is None:
        distance_configs = DISTANCE_MATRIX_CONFIGS

    print("\n" + "#" * 80)
    print("STARTING REVISED PAM CLUSTERING & ROBUST STABILITY PIPELINE")
    print("#" * 80)
    print(f"Input Database:       {input_db_path}")
    print(f"Output Database:      {output_db_path}")
    print(f"Distance Matrices:    {list(distance_configs.keys())}")
    print(f"Cluster Range:        k in {k_range}")
    print(f"PAM Multi-Starts:     {n_runs} runs per k (k-medoids++)")
    print(f"Stability Assessment: {n_subsamples} iterations (80% subsample without replacement)")
    print(f"Parallel Workers:     n_jobs = {n_jobs} (All available CPU cores)")
    print("#" * 80)

    all_config_results: List[Dict] = []

    for cfg_key, cfg_info in distance_configs.items():
        res = process_single_configuration(
            config_key=cfg_key,
            table_name=cfg_info["table_name"],
            db_path=output_db_path,
            k_range=k_range,
            n_runs=n_runs,
            n_subsamples=n_subsamples,
            n_jobs=n_jobs,
        )
        all_config_results.append(res)

    # Save unified tables to SQLite
    performance_df, labels_df = save_clustering_results_to_sqlite(
        all_config_results=all_config_results,
        input_db_path=input_db_path,
        output_db_path=output_db_path,
        performance_table=PERFORMANCE_TABLE_NAME,
        labels_table=LABELS_TABLE_NAME,
    )

    # Print summary performance table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARKING TABLE: MATHEMATICAL VALIDITY & STABILITY")
    print("=" * 80)
    display_cols = ["distance_config", "k", "silhouette", "inertia", "stability_ari_mean", "stability_ari_std", "cluster_sizes"]
    print(performance_df[display_cols].to_string(index=False))
    print("=" * 80 + "\n")

    return performance_df, labels_df


# =============================================================================
# STANDALONE EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

    in_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_in.sqlite")
    out_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_out.sqlite")

    perf_df, lbls_df = run_pam_pipeline(
        input_db_path=in_db,
        output_db_path=out_db,
        k_range=list(range(2, 9)),
        n_runs=10,
        n_subsamples=100,
        n_jobs=-1,
    )

    eval_fig_path = os.path.join(project_root, "p1_dir", "p1_outputs", "pam_evaluation_profiles.png")
    plot_pam_evaluation_profiles(perf_df, save_path=eval_fig_path)
