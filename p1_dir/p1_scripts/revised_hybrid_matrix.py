r"""
================================================================================
REVISED HYBRID DISTANCE MATRIX MODULE FOR CLINICAL PHENOTYPING
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module computes a two-block hybrid distance matrix for patient phenotyping
in obesity and lifestyle medicine research. Patients are characterized across
two clinically and methodologically distinct domains:

1. Weight Gain Causes (WGC):
   12 categorical triggers derived from spontaneous, unstructured patient narratives
   via Natural Language Processing (NLP). Each patient typically reports 1 to 3 causes
   salient to their personal history, while omitting the rest.

2. Eating Behaviors (EB):
   6 symptoms and self-regulation parameters captured via structured, closed-ended
   questionnaires answered by all patients (3 binary yes/no symptoms and 3 ordinal
   1-10 Likert scales).

Methodological & Theoretical Justification:
-------------------------------------------
1. The Denominator Shrinkage Trap of Unified Gower:
   In a single unified Gower distance where WGC variables are treated as asymmetric
   binary, variable pairs where neither patient reported a cause (0-0) are omitted
   from both numerator and denominator. Because patients report few causes on average,
   10 to 11 of the 12 WGC variables are 0-0 for any pair. In a unified denominator,
   the total weight of the WGC block shrinks dramatically (from a nominal 70% down
   to ~15-28%), while the 6 fully observed EB variables retain 100% of their weight.
   This inadvertently causes eating behavior to overpower etiology by up to 4:1,
   washing out distinct clinical cause phenotypes.

2. The Two-Block Hybrid Solution:
   To ensure that the intended weighting between etiology and eating behavior is
   unconditionally preserved for EVERY patient pair, distance is computed in two
   separate, self-contained metric blocks and combined linearly:
   
       D_hybrid(i, j) = W_WGC * D_WGC_Jaccard(i, j) + W_EB * D_EB_Gower(i, j)

   - Block 1 (WGC): Pure Jaccard distance on the 12 binary causes:
     D_Jaccard(i, j) = 1 - (|A \cap B| / |A \cup B|).
     Treats unshared non-reports as uninformative, gives distance 0.0 to patients with
     identical reported causes, and 1.0 to patients with no overlapping causes.
     Its contribution is strictly bounded in [0, 1] and weighted by W_WGC.

   - Block 2 (EB): Range-normalized Gower distance on the 6 eating behaviors:
     Corrects the Euclidean squaring distortion noted by peer reviewers.
     - 3 binary symptoms (hunger, satiety, emotional eating): symmetric binary
       where both joint presence (1-1) and joint absence (0-0) count as clinical agreement.
     - 3 Likert scales: absolute difference normalized by the fixed theoretical scale
       range (R = 10 - 1 = 9.0), equating an extreme Likert disparity (1 vs 10) to a
       binary mismatch (1.0).
     - Each of the 6 EB variables receives an equal 1/6 share of D_EB_Gower, bounded in [0, 1].

3. Sensitivity Configurations:
   Four configurations are computed and saved to enable sensitivity analysis:
   - 0.50 / 0.50 (Equal block weighting)
   - 0.60 / 0.40 (Intermediate weighting)
   - 0.67 / 0.33 (Proportional to variable count: 12 WGC vs 6 EB variables, 2/3 vs 1/3)
   - 0.70 / 0.30 (High etiological prioritization)

Author: Clinical Data Science Pair Programmer
Date: October 2026
Target Environment: Python 3.10+ / SQLite3 / NumPy / SciPy / Pandas
================================================================================
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform

# Ensure standard UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# CONSTANTS & VARIABLE CATALOGUE
# =============================================================================

# Primary patient identification keys
IDENTIFIER_COLUMNS: List[str] = ["patient_id", "medical_record_id"]

# 12 Weight Gain Cause (WGC) variables — Jaccard Distance Block
WGC_COLUMNS: List[str] = [
    "womens_health_and_pregnancy",
    "mental_health",
    "family_issues",
    "medication_disease_injury",
    "physical_inactivity",
    "eating_habits",
    "schedule",
    "smoking_cessation",
    "treatment_discontinuation_or_relapse",
    "pandemic",
    "lifestyle_circumstances",
    "none_of_above",  # Positive assertion of idiopathic weight gain
]

# 3 Structured Eating Behavior Binary variables — Symmetric Binary matching
EB_BINARY_COLUMNS: List[str] = [
    "hunger_yn",
    "satiety_yn",
    "emotional_eating_yn",
]

# 3 Eating Behavior Likert variables (1-10 scale) — Ordinal, normalized by R=9.0
EB_ORDINAL_COLUMNS: List[str] = [
    "emotional_eating_value_likert",
    "quantity_control_likert",
    "impulse_control_likert",
]

# Fixed theoretical range of the 1-10 Likert scale instrument: Max(10) - Min(1) = 9.0
LIKERT_THEORETICAL_RANGE: float = 9.0

# 4 Hybrid block weight sensitivity configurations
HYBRID_CONFIGURATIONS: Dict[str, Dict[str, any]] = {
    "equal_0.50_0.50": {
        "wgc_weight": 0.50,
        "eb_weight": 0.50,
        "table_name": "hybrid_wgc05_eb05",
        "description": "Equal block weight: 50% etiology (WGC) + 50% eating behavior (EB)",
    },
    "intermediate_0.60_0.40": {
        "wgc_weight": 0.60,
        "eb_weight": 0.40,
        "table_name": "hybrid_wgc06_eb04",
        "description": "Intermediate balance: 60% etiology (WGC) + 40% eating behavior (EB)",
    },
    "proportional_0.67_0.33": {
        "wgc_weight": 0.67,
        "eb_weight": 0.33,
        "table_name": "hybrid_wgc067_eb033",
        "description": "Proportional block weight (2/3 vs 1/3) matching 12 WGC vs 6 EB variables",
    },
    "high_wgc_0.70_0.30": {
        "wgc_weight": 0.70,
        "eb_weight": 0.30,
        "table_name": "hybrid_wgc07_eb03",
        "description": "Etiology prioritization: 70% etiology (WGC) + 30% eating behavior (EB)",
    },
}


# =============================================================================
# DATA INGESTION & COHORT VALIDATION
# =============================================================================

def load_cohort_clustering_data(
    db_path: str,
    table_name: str = "timetoevent_eb_wgc_compl",
) -> pd.DataFrame:
    """
    Loads clustering cohort data from SQLite and rigorously verifies schema integrity.

    Mathematical & Clinical Logic:
    ------------------------------
    Before computing pairwise distances, we ensure that every patient possesses a unique
    medical record identifier, that all 18 clinical variables are present, and that
    no invalid missing values or out-of-bounds scale scores compromise the metric space.

    Parameters:
    -----------
    db_path : str
        Absolute or relative path to the SQLite source database.
    table_name : str
        Name of the clinical analysis table (default: 'timetoevent_eb_wgc_compl').

    Returns:
    --------
    pd.DataFrame
        Validated DataFrame sorted reproducibly by medical_record_id.
    """
    print(f"\n[Data Ingestion] Connecting to database: {db_path}")
    print(f"[Data Ingestion] Reading source table: '{table_name}'...")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at path: {db_path}")

    # Build exact query to extract only the required clustering variables
    all_required_cols = IDENTIFIER_COLUMNS + WGC_COLUMNS + EB_BINARY_COLUMNS + EB_ORDINAL_COLUMNS
    col_str = ", ".join(all_required_cols)
    query = f"SELECT {col_str} FROM {table_name} ORDER BY {IDENTIFIER_COLUMNS[1]}"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    n_patients, n_cols = df.shape
    print(f"[Data Ingestion] Successfully loaded {n_patients} patients across {n_cols} columns.")

    # 1. Verify Uniqueness of Patient Identifiers
    n_unique_mr = df["medical_record_id"].nunique()
    if n_unique_mr != n_patients:
        raise ValueError(
            f"Integrity Error: Found {n_patients} rows but only {n_unique_mr} unique medical_record_ids."
        )
    print(f"[Diagnostics] Patient identifier integrity verified: {n_unique_mr} unique records.")

    # 2. Check for Missing Values across all 18 Clustering Variables
    clustering_vars = WGC_COLUMNS + EB_BINARY_COLUMNS + EB_ORDINAL_COLUMNS
    missing_counts = df[clustering_vars].isna().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        bad_cols = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing Data Error: Unexpected NaNs found in clustering variables: {bad_cols}")
    print("[Diagnostics] Complete-case integrity verified: 0 missing values across all 18 variables.")

    # 3. Check Binary Encodings {0, 1}
    binary_vars = WGC_COLUMNS + EB_BINARY_COLUMNS
    for col in binary_vars:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Encoding Error: Column '{col}' contains invalid non-binary values: {unique_vals}")
    print(f"[Diagnostics] Binary integrity verified across {len(binary_vars)} categorical variables.")

    # 4. Check Likert Scales [1, 10]
    for col in EB_ORDINAL_COLUMNS:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val < 1.0 or max_val > 10.0:
            raise ValueError(
                f"Range Error: Column '{col}' out of theoretical bounds [1, 10]: min={min_val}, max={max_val}"
            )
    print(f"[Diagnostics] Likert instrument integrity verified: all scores within theoretical range [1, 10].")

    # 5. Verify Cause Reporting Density
    cause_sums = df[WGC_COLUMNS].sum(axis=1)
    n_zero_causes = (cause_sums == 0).sum()
    print(f"[Diagnostics] WGC reporting density: mean = {cause_sums.mean():.2f} causes/patient (range: {cause_sums.min():.0f}-{cause_sums.max():.0f}).")
    if n_zero_causes > 0:
        print(f"[Diagnostics] Note: {n_zero_causes} patients reported 0 of 12 causes.")
    else:
        print("[Diagnostics] All patients report at least 1 cause trigger (including 'none_of_above').")

    return df


# =============================================================================
# BLOCK 1: WEIGHT GAIN CAUSES (WGC) JACCARD DISTANCE
# =============================================================================

def compute_wgc_jaccard_distance(
    df: pd.DataFrame,
    wgc_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Computes pairwise Jaccard distance matrix for the 12 Weight Gain Cause variables.

    Pseudocode:
    -----------
    1. Extract binary WGC matrix X (N x 12) as boolean array.
    2. Compute pairwise Jaccard distance:
       For patient pair (i, j):
           a = sum(X[i] == 1 and X[j] == 1)   # Shared causes
           b = sum(X[i] == 1 and X[j] == 0)   # Causes unique to i
           c = sum(X[i] == 0 and X[j] == 1)   # Causes unique to j
           if (a + b + c) > 0:
               D_Jaccard(i, j) = (b + c) / (a + b + c) = 1 - (a / (a + b + c))
           else (both vectors all zeros):
               D_Jaccard(i, j) = 0.0 (if i == j) else 1.0
    3. Ensure diagonal is strictly 0.0 and matrix is symmetric.

    Clinical Logic:
    ---------------
    In spontaneous patient narratives, non-reporting of a trigger is uninformative
    (absence of evidence is not evidence of absence). Jaccard distance strictly evaluates
    the intersection over union of reported causes. Two patients who both report
    'womens_health_and_pregnancy' share a distance of 0.0 (if neither reports other causes),
    whereas two patients reporting entirely disjoint causes receive distance 1.0.

    Parameters:
    -----------
    df : pd.DataFrame
        Cohort DataFrame containing WGC binary columns.
    wgc_cols : Optional[List[str]]
        List of WGC column names (defaults to WGC_COLUMNS).

    Returns:
    --------
    np.ndarray
        Symmetric N x N matrix of Jaccard distances in [0.0, 1.0].
    """
    if wgc_cols is None:
        wgc_cols = WGC_COLUMNS

    print(f"\n[Metric Block 1] Computing WGC Jaccard distance matrix across {len(wgc_cols)} causes...")
    cause_matrix = df[wgc_cols].to_numpy(dtype=bool)
    n_patients = len(df)

    # Compute pairwise Jaccard distance vector
    # scipy's pdist computes (c_TF + c_FT) / (c_TT + c_FT + c_TF)
    dist_vec = pdist(cause_matrix, metric="jaccard")

    # If any pair had 0 causes in union, pdist returns NaN; replace with 0.0 if identical or 1.0
    nan_count = np.isnan(dist_vec).sum()
    if nan_count > 0:
        print(f"[Metric Block 1] Warning: {nan_count} pairwise comparisons yielded NaN (empty union). Imputing 1.0.")
        dist_vec = np.nan_to_num(dist_vec, nan=1.0)

    dist_matrix = squareform(dist_vec)
    np.fill_diagonal(dist_matrix, 0.0)

    # Diagnostics
    triu_idx = np.triu_indices(n_patients, k=1)
    off_diag = dist_matrix[triu_idx]
    print(f"[Metric Block 1] WGC Jaccard matrix computed: shape = {dist_matrix.shape}")
    print(f"                 Min: {off_diag.min():.4f} | Max: {off_diag.max():.4f} | Mean: {off_diag.mean():.4f} | Std: {off_diag.std():.4f}")
    zero_dist_pairs = (off_diag == 0.0).sum()
    max_dist_pairs = (off_diag == 1.0).sum()
    total_pairs = len(off_diag)
    print(f"                 Identical cause profiles (d=0.0): {zero_dist_pairs:,} / {total_pairs:,} ({zero_dist_pairs / total_pairs * 100:.2f}%)")
    print(f"                 Completely disjoint causes (d=1.0): {max_dist_pairs:,} / {total_pairs:,} ({max_dist_pairs / total_pairs * 100:.2f}%)")

    return dist_matrix


# =============================================================================
# BLOCK 2: EATING BEHAVIOR (EB) GOWER DISTANCE
# =============================================================================

def compute_eb_gower_distance(
    df: pd.DataFrame,
    binary_cols: Optional[List[str]] = None,
    ordinal_cols: Optional[List[str]] = None,
    likert_range: float = LIKERT_THEORETICAL_RANGE,
) -> np.ndarray:
    """
    Computes pairwise Gower distance matrix on the 6 Eating Behavior variables.

    Pseudocode:
    -----------
    1. Extract binary EB matrix B (N x 3) and Likert EB matrix L (N x 3).
    2. For binary symptoms (hunger, satiety, emotional eating):
       - Treated as symmetric binary:
         d_bin(i, j, k) = |B[i, k] - B[j, k]|   (0 if both 0 or both 1; 1 if discordant)
    3. For ordinal Likert scales (emotional eating value, quantity control, impulse control):
       - Normalized by theoretical scale range R = 9.0:
         d_likert(i, j, k) = |L[i, k] - L[j, k]| / 9.0
    4. Compute equal-weight average across all 6 EB variables:
       D_EB_Gower(i, j) = ( sum_k d_bin(i, j, k) + sum_k d_likert(i, j, k) ) / 6.0
    5. Ensure diagonal is strictly 0.0 and matrix is symmetric.

    Clinical Logic:
    ---------------
    Unlike narrative causes, structured questionnaires are answered by every patient.
    Answering 'No' (0) to hunger or satiety is clinically informative (indicating normal
    appetite regulation), so symmetric binary matching is essential. Furthermore,
    normalizing Likert discrepancies by the 9.0 theoretical range ensures that a 9-point
    gap (extreme divergence from 1 to 10) carries the exact same mathematical weight (1.0)
    as a complete binary symptom mismatch, eliminating the quadratic scaling distortion
    inherent in Euclidean distances.

    Parameters:
    -----------
    df : pd.DataFrame
        Cohort DataFrame containing EB columns.
    binary_cols : Optional[List[str]]
        List of 3 binary EB column names.
    ordinal_cols : Optional[List[str]]
        List of 3 ordinal Likert EB column names.
    likert_range : float
        Theoretical scale range for Likert normalization (default: 9.0).

    Returns:
    --------
    np.ndarray
        Symmetric N x N matrix of EB Gower distances in [0.0, 1.0].
    """
    if binary_cols is None:
        binary_cols = EB_BINARY_COLUMNS
    if ordinal_cols is None:
        ordinal_cols = EB_ORDINAL_COLUMNS

    n_vars = len(binary_cols) + len(ordinal_cols)
    print(f"\n[Metric Block 2] Computing EB Gower distance matrix across {n_vars} variables (3 symmetric binary + 3 Likert R={likert_range:.1f})...")

    n_patients = len(df)
    bin_matrix = df[binary_cols].to_numpy(dtype=np.float64)
    likert_matrix = df[ordinal_cols].to_numpy(dtype=np.float64)

    # Pairwise absolute differences (broadcasted across N x N x K)
    diff_bin = np.abs(bin_matrix[:, None, :] - bin_matrix[None, :, :])
    diff_likert = np.abs(likert_matrix[:, None, :] - likert_matrix[None, :, :]) / likert_range

    # Equal weighting across all 6 EB variables
    sum_diffs = np.sum(diff_bin, axis=2) + np.sum(diff_likert, axis=2)
    dist_matrix = sum_diffs / float(n_vars)
    np.fill_diagonal(dist_matrix, 0.0)

    # Diagnostics
    triu_idx = np.triu_indices(n_patients, k=1)
    off_diag = dist_matrix[triu_idx]
    print(f"[Metric Block 2] EB Gower matrix computed: shape = {dist_matrix.shape}")
    print(f"                 Min: {off_diag.min():.4f} | Max: {off_diag.max():.4f} | Mean: {off_diag.mean():.4f} | Std: {off_diag.std():.4f}")

    return dist_matrix


# =============================================================================
# HYBRID COMBINATION & SENSITIVITY GENERATOR
# =============================================================================

def compute_hybrid_distance(
    wgc_dist: np.ndarray,
    eb_dist: np.ndarray,
    wgc_weight: float,
    eb_weight: float,
) -> np.ndarray:
    """
    Combines WGC Jaccard and EB Gower distance matrices according to specified block weights.

    Mathematical Formulation:
    -------------------------
        D_hybrid(i, j) = w_wgc * D_wgc(i, j) + w_eb * D_eb(i, j)

    Properties:
    -----------
    - Convex combination: w_wgc + w_eb = 1.0.
    - Symmetric: D_hybrid(i, j) = D_hybrid(j, i).
    - Identity: D_hybrid(i, i) = 0.0.
    - Bounded: 0.0 <= D_hybrid(i, j) <= 1.0.
    - Zero denominator collapse: WGC weight is strictly preserved for every patient pair.

    Parameters:
    -----------
    wgc_dist : np.ndarray
        N x N Jaccard distance matrix.
    eb_dist : np.ndarray
        N x N EB Gower distance matrix.
    wgc_weight : float
        Weight assigned to etiology block (e.g., 0.70).
    eb_weight : float
        Weight assigned to eating behavior block (e.g., 0.30).

    Returns:
    --------
    np.ndarray
        Combined hybrid distance matrix.
    """
    if not np.isclose(wgc_weight + eb_weight, 1.0):
        raise ValueError(f"Block weights must sum to 1.0; received WGC={wgc_weight}, EB={eb_weight} (sum={wgc_weight + eb_weight})")

    hybrid_dist = (wgc_weight * wgc_dist) + (eb_weight * eb_dist)
    np.fill_diagonal(hybrid_dist, 0.0)
    return hybrid_dist


# =============================================================================
# PERSISTENCE & DATABASE FORMATTING
# =============================================================================

def save_distance_matrix_to_sqlite(
    dist_matrix: np.ndarray,
    medical_record_ids: List[str],
    db_path: str,
    table_name: str,
) -> pd.DataFrame:
    """
    Persists an N x N distance matrix to SQLite with standard patient indexing.

    Output Schema:
    --------------
    - Column 0: 'medical_record_id' (TEXT)
    - Column 1: 'avg_dist_to_all' (REAL) — Outlier screening metric: mean distance to all other patients
    - Columns 2..N+1: '<medical_record_id>' (REAL) — Pairwise distances to patient j

    Parameters:
    -----------
    dist_matrix : np.ndarray
        Symmetric N x N distance matrix.
    medical_record_ids : List[str]
        List of patient medical record IDs.
    db_path : str
        SQLite destination database path.
    table_name : str
        Name of table to write or overwrite.

    Returns:
    --------
    pd.DataFrame
        Formatted DataFrame as written to SQLite.
    """
    n_patients = len(medical_record_ids)
    print(f"[Persistence] Formatting matrix for SQLite table '{table_name}' ({n_patients} x {n_patients})...")

    # Compute mean distance to all other patients (outlier metric)
    if n_patients > 1:
        avg_distances = np.sum(dist_matrix, axis=1) / float(n_patients - 1)
    else:
        avg_distances = np.zeros(n_patients)

    matrix_df = pd.DataFrame(
        dist_matrix,
        index=medical_record_ids,
        columns=medical_record_ids,
    ).reset_index()

    matrix_df.rename(columns={"index": "medical_record_id"}, inplace=True)
    matrix_df.insert(1, "avg_dist_to_all", avg_distances)

    with sqlite3.connect(db_path) as conn:
        matrix_df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"[Persistence] Successfully saved table '{table_name}' into '{db_path}'.")
    return matrix_df


def load_hybrid_matrix_from_sqlite(
    db_path: str,
    table_name: str,
) -> Tuple[np.ndarray, List[str], pd.Series]:
    """
    Loads a saved distance matrix from SQLite.

    Parameters:
    -----------
    db_path : str
        Path to SQLite database.
    table_name : str
        Name of table to load.

    Returns:
    --------
    Tuple[np.ndarray, List[str], pd.Series]:
        (dist_matrix, medical_record_ids, avg_distances_series)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at path: {db_path}")

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    medical_record_ids = df["medical_record_id"].astype(str).tolist()
    avg_distances = df["avg_dist_to_all"].copy()

    # Drop non-matrix identifier and summary columns
    matrix_values_df = df.drop(columns=["medical_record_id", "avg_dist_to_all"])
    dist_matrix = matrix_values_df.to_numpy(dtype=np.float64)

    return dist_matrix, medical_record_ids, avg_distances


# =============================================================================
# MASTER PIPELINE DRIVER
# =============================================================================

def run_hybrid_matrix_pipeline(
    input_db_path: str,
    output_db_path: Optional[str] = None,
    source_table_name: str = "timetoevent_eb_wgc_compl",
    configs: Optional[Dict[str, Dict[str, any]]] = None,
    save_component_matrices: bool = True,
    matrix_destination_db: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Executes the revised hybrid distance preparation pipeline across all 4 sensitivity configurations.
    Conventionally persists distance matrices directly into the input database (input_db_path)
    as distance matrices serve as the input for clustering algorithms.

    Parameters:
    -----------
    input_db_path : str
        Path to source SQLite database containing timetoevent_eb_wgc_compl.
    output_db_path : Optional[str]
        Optional path for outputs; retained for backwards compatibility.
    source_table_name : str
        Source table name (default: 'timetoevent_eb_wgc_compl').
    configs : Optional[Dict]
        Sensitivity configurations (defaults to HYBRID_CONFIGURATIONS).
    save_component_matrices : bool
        Whether to save wgc_jaccard_distance and eb_gower_distance tables (default: True).
    matrix_destination_db : Optional[str]
        Explicit destination SQLite database for distance matrix tables.
        Defaults to input_db_path (matching project convention).

    Returns:
    --------
    Dict[str, Dict]:
        Structured dictionary of results for each configuration.
    """
    # Convention: Distance matrices belong in the input database as inputs to clustering
    if matrix_destination_db is None:
        matrix_destination_db = input_db_path

    if configs is None:
        configs = HYBRID_CONFIGURATIONS

    print("\n" + "=" * 80)
    print("STARTING REVISED JACCARD-GOWER HYBRID DISTANCE PIPELINE")
    print(f"Destination DB for Matrices: {matrix_destination_db}")
    print("=" * 80)

    # 1. Ingest Cohort
    df = load_cohort_clustering_data(input_db_path, table_name=source_table_name)
    medical_record_ids = df["medical_record_id"].astype(str).tolist()
    n_patients = len(medical_record_ids)

    # 2. Compute Metric Block 1: WGC Jaccard
    wgc_dist = compute_wgc_jaccard_distance(df, wgc_cols=WGC_COLUMNS)

    # 3. Compute Metric Block 2: EB Gower
    eb_dist = compute_eb_gower_distance(
        df,
        binary_cols=EB_BINARY_COLUMNS,
        ordinal_cols=EB_ORDINAL_COLUMNS,
        likert_range=LIKERT_THEORETICAL_RANGE,
    )

    # 4. Save Component Matrices for Auditing
    if save_component_matrices:
        print("\n[Persistence] Saving standalone component matrices for auditability...")
        save_distance_matrix_to_sqlite(wgc_dist, medical_record_ids, matrix_destination_db, "hybrid_component_wgc_jaccard")
        save_distance_matrix_to_sqlite(eb_dist, medical_record_ids, matrix_destination_db, "hybrid_component_eb_gower")

    # 5. Generate and Persist All Hybrid Configurations
    results = {
        "medical_record_ids": medical_record_ids,
        "n_patients": n_patients,
        "wgc_dist": wgc_dist,
        "eb_dist": eb_dist,
        "configurations": {},
    }

    triu_idx = np.triu_indices(n_patients, k=1)

    print("\n" + "-" * 80)
    print("GENERATING 4 SENSITIVITY HYBRID CONFIGURATIONS")
    print("-" * 80)

    for cfg_key, cfg_info in configs.items():
        w_wgc = cfg_info["wgc_weight"]
        w_eb = cfg_info["eb_weight"]
        table_name = cfg_info["table_name"]
        desc = cfg_info["description"]

        print(f"\n>> Configuration: '{cfg_key}'")
        print(f"   Description: {desc}")
        print(f"   Weights: WGC={w_wgc:.2f} ({w_wgc * 100:.0f}%), EB={w_eb:.2f} ({w_eb * 100:.0f}%)")

        hybrid_matrix = compute_hybrid_distance(
            wgc_dist=wgc_dist,
            eb_dist=eb_dist,
            wgc_weight=w_wgc,
            eb_weight=w_eb,
        )

        off_diag = hybrid_matrix[triu_idx]
        print(f"   Summary: Min={off_diag.min():.4f} | Max={off_diag.max():.4f} | Mean={off_diag.mean():.4f} | Std={off_diag.std():.4f}")

        # Persist to SQLite (convention: input database)
        save_distance_matrix_to_sqlite(
            dist_matrix=hybrid_matrix,
            medical_record_ids=medical_record_ids,
            db_path=matrix_destination_db,
            table_name=table_name,
        )

        results["configurations"][cfg_key] = {
            "table_name": table_name,
            "wgc_weight": w_wgc,
            "eb_weight": w_eb,
            "matrix": hybrid_matrix,
            "mean_dist": float(off_diag.mean()),
            "std_dist": float(off_diag.std()),
        }

    print("\n" + "=" * 80)
    print("REVISED HYBRID DISTANCE PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total Configurations Saved: {len(configs)} tables in '{matrix_destination_db}'")
    print("=" * 80 + "\n")

    return results


# =============================================================================
# CLI EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Default project paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_in_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in.sqlite")

    print(f"Project Base Directory: {base_dir}")
    print(f"Distance Matrix Target (Input DB): {default_in_db}")

    run_hybrid_matrix_pipeline(
        input_db_path=default_in_db,
        matrix_destination_db=default_in_db,
    )
