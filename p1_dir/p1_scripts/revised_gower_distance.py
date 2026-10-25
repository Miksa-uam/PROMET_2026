"""
================================================================================
REVISED GOWER DISTANCE MODULE FOR MIXED CLINICAL PHENOTYPING
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module implements a rigorous, mixed-metric Gower distance framework tailored
for patient phenotyping in obesity and lifestyle medicine research.

Patients are characterized across two distinct clinical domains:
1. Weight Gain Causes (WGC): 12 categorical triggers extracted from unstructured,
   spontaneous patient narratives using Natural Language Processing (NLP).
2. Eating Behaviors (EB): 6 symptoms and self-regulation metrics captured through
   structured clinical questionnaires (3 binary indicators and 3 ordinal Likert scales).

Methodological & Theoretical Justification (Addressing Reviewer Comment 3):
---------------------------------------------------------------------------
Traditional hybrid distances that linearly add Jaccard distances and Euclidean
distances (e.g., 0.67 * D_Jaccard + 0.33 * D_Euclidean) suffer from three severe
methodological vulnerabilities:
1. Mixed Scaling Distortions: Within a Euclidean block, binary discrepancies (0 vs 1)
   contribute squared differences (1.0), whereas continuous/ordinal discrepancies
   contribute fractional squares (e.g., 0.33^2 = 0.11), allowing binary variables
   to inadvertently dominate the distance space.
2. Loss of Metric Structure: Adding disparate metric spaces voids Euclidean embeddability
   and distorts the geometric interpretation of minimum-variance hierarchical linkage.
3. Violation of Asymmetric Binary Properties: Standard off-the-shelf Gower libraries
   treat all binary variables symmetrically (simple matching coefficient), falsely
   penalizing or rewarding patients who both omit a free-text narrative category.

This module resolves these issues through a unified Gower (1971) dissimilarity
formulation with domain-specific matching rules and structured block-level weighting:

================================================================================
MATHEMATICAL FORMULATION vs. CLINICAL INTERPRETATION
================================================================================

1. Weight Gain Causes (WGC) -> Asymmetric Binary Treatment:
   --------------------------------------------------------
   * Programmatic / Mathematical Formulation:
     Let x_{ik}, x_{jk} in {0, 1} represent the presence/absence of cause k.
     The comparison validity indicator delta_k(i, j) and partial distance d_k(i, j) are:
         delta_k(i, j) = 0  if x_{ik} == 0 and x_{jk} == 0
         delta_k(i, j) = 1  otherwise (if x_{ik} == 1 or x_{jk} == 1)
         d_k(i, j)     = 0  if x_{ik} == x_{jk} == 1
         d_k(i, j)     = 1  if x_{ik} != x_{jk}
     When delta_k(i, j) = 0, the variable contributes 0 to BOTH numerator and
     denominator of Gower's formula, completely excluding the unshared attribute.

   * Clinical & Translational Meaning:
     In free-text medical history narratives, patients recall and express whatever
     etiological factors are salient to them; they are NOT presented with an exhaustive
     checklist. Consequently, if two patients neither mention 'smoking cessation',
     this joint absence is clinically uninformative—it does not imply they share an
     underlying biological etiology. Conversely, if both explicitly report 'emotional
     eating' or both report 'none_of_above' (positive assertion of idiopathic weight gain),
     they share a meaningful clinical phenotype. Hence, 0-0 matches must be excluded.

2. Eating Behavior Binary Symptoms -> Symmetric Binary Treatment:
   --------------------------------------------------------------
   * Programmatic / Mathematical Formulation:
     Let x_{ik}, x_{jk} in {0, 1} represent binary questionnaire answers.
         delta_k(i, j) = 1  always (for complete records)
         d_k(i, j)     = 0  if x_{ik} == x_{jk}  (both 1 or both 0)
         d_k(i, j)     = 1  if x_{ik} != x_{jk}

   * Clinical & Translational Meaning:
     Variables like hunger_yn, satiety_yn, and emotional_eating_yn originate from
     structured closed-ended screening questions ('Do you experience constant hunger?').
     Here, an answer of 'No' (0) is an active, informative clinical finding (e.g., normal
     satiety responsiveness). Two patients who both deny hyperphagia are genuinely
     concordant in that clinical symptom. Therefore, both 1-1 and 0-0 count as agreement.

3. Eating Behavior Likert Scales -> Ordinal / Continuous Treatment:
   ----------------------------------------------------------------
   * Programmatic / Mathematical Formulation:
     Let x_{ik}, x_{jk} in [1, 10] denote scores on a 1-to-10 Likert scale.
     Dissimilarity is normalized by the fixed theoretical scale range:
         R_k = Scale_max - Scale_min = 10.0 - 1.0 = 9.0
         delta_k(i, j) = 1  always
         d_k(i, j)     = |x_{ik} - x_{jk}| / R_k = |x_{ik} - x_{jk}| / 9.0

   * Clinical & Translational Meaning:
     Subjective self-regulation capacities (e.g., impulse control or quantity control)
     exist along a continuous spectrum. A one-point shift between adjacent ratings
     (e.g., 7 vs 8) represents minor behavioral nuance (d = 1/9 = 0.111), whereas
     divergence between opposite extremes (1 vs 10) represents profound behavioral
     heterogeneity (d = 9/9 = 1.0). Normalizing by 9.0 puts an extreme Likert disparity
     on the exact same numerical footing as a complete mismatch on a binary symptom,
     eliminating artificial scale dominance.

4. Aggregate Gower Distance with Block Weighting:
   ----------------------------------------------
   * Mathematical Formulation:
     For any pair of patients (i, j):
                       sum_{k=1}^{K} [ w_k * delta_k(i, j) * d_k(i, j) ]
         D_Gower(i, j) = -------------------------------------------------
                            sum_{k=1}^{K} [ w_k * delta_k(i, j) ]

     Where block weights W_WGC and W_EB satisfy W_WGC + W_EB = 1.0, and individual
     variable weights are allocated uniformly within each block:
         w_k = W_WGC / 12  for each of the 12 WGC variables
         w_k = W_EB  / 6   for each of the 6 EB variables

   * Clinical & Translational Meaning:
     Because the WGC block contains 12 features while the EB block contains 6,
     unweighted algorithms would give etiology twice the influence of active eating
     behavior. Block weighting allows researchers to explicitly control whether
     etiological attribution (patient-perceived cause) and behavioral presentation
     (psychological symptoms) contribute equally (0.50/0.50), proportionally to variable
     counts (0.70/0.30), or according to an intermediate sensitivity stance (0.60/0.40).
================================================================================
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CONSTANTS & VARIABLE CATALOGUE
# =============================================================================

# Primary patient identification keys
IDENTIFIER_COLUMNS: List[str] = ["patient_id", "medical_record_id"]

# 12 Weight Gain Cause (WGC) variables — Treated as Asymmetric Binary (0-0 excluded)
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

# 3 Structured Eating Behavior Binary variables — Treated as Symmetric Binary (0-0 agreement)
EB_BINARY_COLUMNS: List[str] = [
    "hunger_yn",
    "satiety_yn",
    "emotional_eating_yn",
]

# 3 Eating Behavior Likert variables (1-10 scale) — Treated as Ordinal (Normalized by Range = 9.0)
EB_ORDINAL_COLUMNS: List[str] = [
    "emotional_eating_value_likert",
    "quantity_control_likert",
    "impulse_control_likert",
]

# Fixed theoretical range of the 1-10 Likert scale instrument: Max(10) - Min(1) = 9.0
LIKERT_THEORETICAL_RANGE: float = 9.0

# Predefined block weight sensitivity configurations
SENSITIVITY_CONFIGURATIONS: Dict[str, Dict[str, float]] = {
    "equal_0.50_0.50": {
        "wgc_weight": 0.50,
        "eb_weight": 0.50,
        "table_name": "revised_gower_wgc05_eb05",
        "description": "Equal block weight: 50% etiology (WGC) + 50% eating behavior (EB)",
    },
    "proportional_0.70_0.30": {
        "wgc_weight": 0.70,
        "eb_weight": 0.30,
        "table_name": "revised_gower_wgc07_eb03",
        "description": "Proportional block weight: ~67/33% matching 12 WGC vs 6 EB variable counts",
    },
    "intermediate_0.60_0.40": {
        "wgc_weight": 0.60,
        "eb_weight": 0.40,
        "table_name": "revised_gower_wgc06_eb04",
        "description": "Intermediate balance: 60% etiology (WGC) + 40% eating behavior (EB)",
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
    for b_col in binary_vars:
        unique_vals = set(df[b_col].dropna().unique())
        if not unique_vals.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Encoding Error: Variable '{b_col}' contains non-binary values: {unique_vals}")
    print(f"[Diagnostics] Binary integrity verified for {len(binary_vars)} categorical variables.")

    # 4. Check Likert Scales [1, 10]
    for likert_col in EB_ORDINAL_COLUMNS:
        min_val = float(df[likert_col].min())
        max_val = float(df[likert_col].max())
        print(f"  - {likert_col}: observed range = [{min_val:.1f}, {max_val:.1f}]")
        if min_val < 1.0 or max_val > 10.0:
            raise ValueError(
                f"Range Error: Likert variable '{likert_col}' has values outside [1.0, 10.0]: [{min_val}, {max_val}]"
            )

    return df


# =============================================================================
# CUSTOM VECTORIZED GOWER DISTANCE COMPUTATION
# =============================================================================

def compute_custom_gower_matrix(
    df: pd.DataFrame,
    wgc_cols: List[str] = WGC_COLUMNS,
    eb_bin_cols: List[str] = EB_BINARY_COLUMNS,
    eb_ord_cols: List[str] = EB_ORDINAL_COLUMNS,
    wgc_weight: float = 0.50,
    eb_weight: float = 0.50,
    likert_range: float = LIKERT_THEORETICAL_RANGE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes the exact, mixed-metric Gower distance matrix using vectorized NumPy tensor broadcasting.

    Methodological & Mathematical Implementation:
    ---------------------------------------------
    Given N patients and K variables:
    1. We construct two (N x N) matrices:
       - Numerator Matrix:   sum_k [ w_k * delta_k(i, j) * d_k(i, j) ]
       - Denominator Matrix: sum_k [ w_k * delta_k(i, j) ]

    2. Asymmetric Binary (WGC):
       For each variable vector v of shape (N,):
       Pairwise difference: diff_mask = (v[:, None] != v[None, :])
       Pairwise validity:   valid_mask = (v[:, None] == 1) | (v[None, :] == 1)
       Numerator += w_k * (diff_mask & valid_mask)
       Denominator += w_k * valid_mask
       *Note: 0-0 pairs evaluate to valid_mask=False, adding 0 to both numerator and denominator.*

    3. Symmetric Binary (EB):
       Numerator += w_k * (v[:, None] != v[None, :])
       Denominator += w_k * 1.0

    4. Ordinal Likert (EB):
       Numerator += w_k * (|v[:, None] - v[None, :]| / likert_range)
       Denominator += w_k * 1.0

    5. Normalization:
       D = Numerator / Denominator
       D[i, i] = 0.0
       Enforce exact floating-point symmetry: D = (D + D.T) / 2.0

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing cohort patients and clustering variables.
    wgc_cols : List[str]
        List of 12 asymmetric binary weight gain cause column names.
    eb_bin_cols : List[str]
        List of 3 symmetric binary eating behavior column names.
    eb_ord_cols : List[str]
        List of 3 ordinal Likert eating behavior column names.
    wgc_weight : float
        Total weight allocated to the WGC block (default: 0.50).
    eb_weight : float
        Total weight allocated to the EB block (default: 0.50).
    likert_range : float
        Theoretical denominator for Likert normalization (default: 9.0).

    Returns:
    --------
    Tuple[np.ndarray, List[str]]:
        (dist_matrix, medical_record_ids)
        dist_matrix: (N x N) symmetric NumPy array where D[i, j] in [0, 1].
        medical_record_ids: list of patient identifiers corresponding to matrix rows/columns.
    """
    n_patients = len(df)
    medical_record_ids = df["medical_record_id"].astype(str).tolist()

    # Normalize block weights so they strictly sum to 1.0
    total_block_weight = wgc_weight + eb_weight
    if not np.isclose(total_block_weight, 1.0, atol=1e-6):
        print(f"[Warning] Block weights did not sum to 1.0 ({total_block_weight:.4f}). Normalizing...")
        wgc_weight /= total_block_weight
        eb_weight /= total_block_weight

    n_wgc = len(wgc_cols)
    n_eb_total = len(eb_bin_cols) + len(eb_ord_cols)

    # Per-variable weight derivation
    w_wgc_per_var = wgc_weight / n_wgc
    w_eb_per_var = eb_weight / n_eb_total

    print("\n" + "=" * 70)
    print("CUSTOM GOWER DISTANCE CALCULATION — CONFIGURATION")
    print("=" * 70)
    print(f"Cohort sample size (N):          {n_patients} patients")
    print(f"Total clustering variables (K):  {n_wgc + n_eb_total} (12 WGC + 6 EB)")
    print(f"WGC Block Weight:                {wgc_weight:.2f} ({n_wgc} variables -> {w_wgc_per_var:.5f} each)")
    print(f"EB Block Weight:                 {eb_weight:.2f} ({n_eb_total} variables -> {w_eb_per_var:.5f} each)")
    print(f"Likert theoretical range:        {likert_range:.1f} (1-10 scale)")
    print("-" * 70)

    # Initialize accumulation matrices in float64 for numerical precision
    numerator = np.zeros((n_patients, n_patients), dtype=np.float64)
    denominator = np.zeros((n_patients, n_patients), dtype=np.float64)

    # -------------------------------------------------------------------------
    # STEP 1: Accumulate Asymmetric Binary Variables (WGC Block)
    # -------------------------------------------------------------------------
    # Clinical rationale: Joint absences (0-0) are uninformative narrative omissions
    # and must not artificially increase patient similarity.
    print(f"[Computation] Processing {n_wgc} Asymmetric Binary WGC variables...")
    for col in wgc_cols:
        v = df[col].to_numpy(dtype=np.float64)  # Shape: (N,)
        
        # Outer boolean operations: v[:, None] is column vector, v[None, :] is row vector
        diff_mask = (v[:, None] != v[None, :])                   # True if (1,0) or (0,1)
        valid_mask = (v[:, None] == 1.0) | (v[None, :] == 1.0)  # True if at least one patient is 1

        # Mismatch contributes d=1 when valid; agreement contributes d=0 when valid
        numerator += w_wgc_per_var * (diff_mask & valid_mask).astype(np.float64)
        denominator += w_wgc_per_var * valid_mask.astype(np.float64)

    # -------------------------------------------------------------------------
    # STEP 2: Accumulate Symmetric Binary Variables (EB Binary Block)
    # -------------------------------------------------------------------------
    # Clinical rationale: Both symptom presence (1-1) and absence (0-0) on
    # structured screening questionnaires represent true clinical concordances.
    print(f"[Computation] Processing {len(eb_bin_cols)} Symmetric Binary EB variables...")
    for col in eb_bin_cols:
        v = df[col].to_numpy(dtype=np.float64)
        diff_mask = (v[:, None] != v[None, :]).astype(np.float64)

        numerator += w_eb_per_var * diff_mask
        denominator += w_eb_per_var * 1.0  # Fully valid for all pairs

    # -------------------------------------------------------------------------
    # STEP 3: Accumulate Ordinal Likert Variables (EB Ordinal Block)
    # -------------------------------------------------------------------------
    # Clinical rationale: Graded psychological self-regulation differences divided
    # by theoretical span of 9.0 points, matching binary disparity at extremes.
    print(f"[Computation] Processing {len(eb_ord_cols)} Ordinal Likert EB variables...")
    for col in eb_ord_cols:
        v = df[col].to_numpy(dtype=np.float64)
        abs_diff = np.abs(v[:, None] - v[None, :]) / likert_range

        numerator += w_eb_per_var * abs_diff
        denominator += w_eb_per_var * 1.0  # Fully valid for all pairs

    # -------------------------------------------------------------------------
    # STEP 4: Compute Final Gower Dissimilarity Matrix & Verify Properties
    # -------------------------------------------------------------------------
    print("[Computation] Normalizing weighted dissimilarities by valid denominators...")
    # Safe division: if denominator == 0 (impossible here as EB is complete), result is 0.0
    dist_matrix = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=(denominator > 0.0),
    )

    # Distance from a patient to themselves is mathematically 0.0
    np.fill_diagonal(dist_matrix, 0.0)

    # Enforce exact floating-point symmetry: D = (D + D.T) / 2.0
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)

    # -------------------------------------------------------------------------
    # STEP 5: Rigorous Diagnostics & Health Checks
    # -------------------------------------------------------------------------
    nan_count = np.isnan(dist_matrix).sum()
    if nan_count > 0:
        raise ValueError(f"Numerical Error: Distance matrix contains {nan_count} NaN values.")

    symmetry_deviation = np.max(np.abs(dist_matrix - dist_matrix.T))
    if symmetry_deviation > 1e-12:
        raise ValueError(f"Symmetry Error: Maximum asymmetry deviation ({symmetry_deviation}) exceeds tolerance.")

    # Extract upper triangle (condensed distance values excluding diagonal) for distribution diagnostics
    upper_tri_indices = np.triu_indices(n_patients, k=1)
    condensed_distances = dist_matrix[upper_tri_indices]

    mean_dist = float(np.mean(condensed_distances))
    std_dist = float(np.std(condensed_distances))
    min_dist = float(np.min(condensed_distances))
    max_dist = float(np.max(condensed_distances))
    q25, median_dist, q75 = np.percentile(condensed_distances, [25, 50, 75])

    print("\n[Diagnostics] Pairwise Distance Distribution Summary:")
    print(f"  - Total pairwise comparisons: {len(condensed_distances):,} pairs")
    print(f"  - Range:   [{min_dist:.4f}, {max_dist:.4f}]")
    print(f"  - Mean:    {mean_dist:.4f} (SD = {std_dist:.4f})")
    print(f"  - Median:  {median_dist:.4f} (IQR = [{q25:.4f}, {q75:.4f}])")
    print(f"  - Diagonal zero check:  np.diag(D).max() = {np.max(np.diag(dist_matrix)):.1e}")
    print(f"  - Max symmetry residual: {symmetry_deviation:.1e}")
    print("=" * 70 + "\n")

    return dist_matrix, medical_record_ids


# =============================================================================
# PERSISTENCE & SQLITE STORAGE
# =============================================================================

def save_distance_matrix_to_sqlite(
    dist_matrix: np.ndarray,
    medical_record_ids: List[str],
    db_path: str,
    table_name: str,
) -> pd.DataFrame:
    """
    Saves the symmetric distance matrix to SQLite matching the established project schema.

    Schema Design & Downstream Compatibility:
    -----------------------------------------
    The table is structured with:
    - Column 0: 'medical_record_id' (string index)
    - Column 1: 'avg_dist_to_all' (mean distance from patient i to all other patients, for outlier screening)
    - Columns 2..N+1: Patient medical record IDs representing the square distance matrix values.

    This guarantees 100% backward compatibility with existing clustering and visualization
    downstream cells (e.g., load_hybrid_matrix_with_medical_record_ids).

    Parameters:
    -----------
    dist_matrix : np.ndarray
        (N x N) symmetric distance matrix.
    medical_record_ids : List[str]
        List of medical record identifiers corresponding to rows and columns.
    db_path : str
        Target SQLite database path.
    table_name : str
        Name of the SQL table to replace/create.

    Returns:
    --------
    pd.DataFrame
        The formatted DataFrame as persisted to SQLite.
    """
    print(f"[Persistence] Formatting distance matrix for SQLite storage in table '{table_name}'...")
    n_patients = len(medical_record_ids)

    # Compute outlier screening metric: average distance from patient i to all other patients
    # Sum of row i divided by (N - 1)
    if n_patients > 1:
        avg_distances = np.sum(dist_matrix, axis=1) / (n_patients - 1)
    else:
        avg_distances = np.zeros(n_patients)

    # Assemble square DataFrame
    matrix_df = pd.DataFrame(
        dist_matrix,
        index=medical_record_ids,
        columns=medical_record_ids,
    ).reset_index()

    matrix_df.rename(columns={"index": "medical_record_id"}, inplace=True)
    matrix_df.insert(1, "avg_dist_to_all", avg_distances)

    # Save to SQLite
    with sqlite3.connect(db_path) as conn:
        matrix_df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"[Persistence] Successfully saved table '{table_name}' ({matrix_df.shape[0]} rows x {matrix_df.shape[1]} cols) into '{db_path}'.")
    return matrix_df


def load_gower_matrix_from_sqlite(
    db_path: str,
    table_name: str,
) -> Tuple[np.ndarray, List[str], pd.Series]:
    """
    Loads a saved Gower distance matrix from SQLite.

    Parameters:
    -----------
    db_path : str
        Path to SQLite database containing the matrix table.
    table_name : str
        Name of the matrix table.

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

def run_gower_distance_pipeline(
    input_db_path: str,
    output_db_path: Optional[str] = None,
    source_table_name: str = "timetoevent_eb_wgc_compl",
    configs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict]:
    """
    Executes the comprehensive Gower distance preparation pipeline across all sensitivity configs.

    Workflow:
    ---------
    1. Ingests and validates clinical cohort data from input_db_path.
    2. Iterates across 3 block-weight sensitivity configurations:
       - Equal weighting (0.50 WGC / 0.50 EB)
       - Proportional weighting (0.70 WGC / 0.30 EB)
       - Intermediate weighting (0.60 WGC / 0.40 EB)
    3. Computes the custom Gower distance matrix for each configuration.
    4. Persists each resulting matrix into SQLite with standard outlier detection columns.
    5. Returns a structured dictionary of matrices, patient IDs, and summary statistics.

    Parameters:
    -----------
    input_db_path : str
        Path to source SQLite database containing timetoevent_eb_wgc_compl.
    output_db_path : Optional[str]
        Path to output SQLite database. If None, saves directly into input_db_path.
    source_table_name : str
        Source table name (default: 'timetoevent_eb_wgc_compl').
    configs : Optional[Dict]
        Custom sensitivity configurations. Defaults to SENSITIVITY_CONFIGURATIONS.

    Returns:
    --------
    Dict[str, Dict]:
        Dictionary keyed by configuration name, containing:
        - 'matrix': (N x N) distance ndarray
        - 'medical_record_ids': list of patient IDs
        - 'table_name': SQLite table name
        - 'wgc_weight': float
        - 'eb_weight': float
        - 'stats': dict of distribution statistics
    """
    if output_db_path is None:
        output_db_path = input_db_path

    if configs is None:
        configs = SENSITIVITY_CONFIGURATIONS

    print("\n" + "#" * 80)
    print("STARTING REVISED GOWER DISTANCE PIPELINE")
    print("#" * 80)
    print(f"Source Database: {input_db_path}")
    print(f"Output Database: {output_db_path}")
    print(f"Source Table:    {source_table_name}")
    print(f"Configurations:  {list(configs.keys())}")

    # 1. Load and validate cohort
    df = load_cohort_clustering_data(input_db_path, table_name=source_table_name)
    n_patients = len(df)

    results: Dict[str, Dict] = {}

    # 2. Iterate across sensitivity configurations
    for cfg_key, cfg_info in configs.items():
        wgc_w = cfg_info["wgc_weight"]
        eb_w = cfg_info["eb_weight"]
        table_name = cfg_info["table_name"]
        description = cfg_info.get("description", "")

        print(f"\n>>> Running Configuration: '{cfg_key}'")
        print(f"    Description: {description}")

        dist_matrix, medical_record_ids = compute_custom_gower_matrix(
            df=df,
            wgc_cols=WGC_COLUMNS,
            eb_bin_cols=EB_BINARY_COLUMNS,
            eb_ord_cols=EB_ORDINAL_COLUMNS,
            wgc_weight=wgc_w,
            eb_weight=eb_w,
            likert_range=LIKERT_THEORETICAL_RANGE,
        )

        # Save to SQLite
        save_distance_matrix_to_sqlite(
            dist_matrix=dist_matrix,
            medical_record_ids=medical_record_ids,
            db_path=output_db_path,
            table_name=table_name,
        )

        # Compute summary distribution stats for comparison
        upper_tri = dist_matrix[np.triu_indices(n_patients, k=1)]
        stats = {
            "mean": float(np.mean(upper_tri)),
            "std": float(np.std(upper_tri)),
            "min": float(np.min(upper_tri)),
            "max": float(np.max(upper_tri)),
            "median": float(np.median(upper_tri)),
            "q25": float(np.percentile(upper_tri, 25)),
            "q75": float(np.percentile(upper_tri, 75)),
        }

        results[cfg_key] = {
            "matrix": dist_matrix,
            "medical_record_ids": medical_record_ids,
            "table_name": table_name,
            "wgc_weight": wgc_w,
            "eb_weight": eb_w,
            "description": description,
            "stats": stats,
        }

    # 3. Print Comparative Summary across all configurations
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY: DISTANCE DISTRIBUTIONS ACROSS WEIGHT CONFIGURATIONS")
    print("=" * 80)
    summary_rows = []
    for cfg_key, res in results.items():
        st = res["stats"]
        summary_rows.append({
            "Configuration": cfg_key,
            "W_WGC / W_EB": f"{res['wgc_weight']:.2f} / {res['eb_weight']:.2f}",
            "Mean (SD)": f"{st['mean']:.4f} ({st['std']:.4f})",
            "Median [IQR]": f"{st['median']:.4f} [{st['q25']:.4f}, {st['q75']:.4f}]",
            "Range [Min, Max]": f"[{st['min']:.4f}, {st['max']:.4f}]",
            "SQLite Table": res["table_name"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("\n[Pipeline Complete] All 3 distance matrices successfully generated and saved.\n")

    return results


# =============================================================================
# STANDALONE EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # When executed directly, locate database relative to project directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

    input_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_in.sqlite")
    output_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_out.sqlite")

    print(f"Executing revised_gower_distance.py as standalone script.")
    print(f"Targeting input DB:  {input_db}")
    print(f"Targeting output DB: {output_db}")

    run_gower_distance_pipeline(
        input_db_path=input_db,
        output_db_path=output_db,
    )
