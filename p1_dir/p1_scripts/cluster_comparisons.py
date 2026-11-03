"""
================================================================================
CLUSTER_COMPARISONS.PY
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module provides a comprehensive, self-contained clinical phenotyping and
cluster comparison engine for research on obesity behavioral subtypes (Paper #1).

Patients are clustered based on two complementary behavioral domains:
1. 12 Weight Gain Cause (WGC) categories derived from free-text patient narratives
   via natural language processing (NLP), treated as asymmetric binary attributes
   (measured via Jaccard distance, where joint reporting constitutes similarity
   while joint absence is uninformative).
2. 6 Eating Behavior (EB) variables captured through questionnaires:
   - 3 symmetric binary symptoms (hunger, satiety responsiveness, emotional eating),
     where both joint presence and absence reflect regulatory phenotype;
   - 3 ordinal Likert ratings (emotional eating severity, quantity control ability,
     impulse control ability; range 1-10), theoretical range-normalized (R = 9.0).
   The EB block is measured via Gower's general dissimilarity coefficient.

The two distance matrices are hybridized across pre-specified weighting schemes
(e.g., wgc05_eb05, wgc06_eb04, wgc067_eb033, wgc07_eb03) and clustered using
PAM (Partitioning Around Medoids) or HAC (Complete Linkage / Average Linkage)
for k = 2 to 10.

Cluster labels and exemplar medoids are persisted in SQLite (e.g. table
'jaccard_gower_cluster_labels' in 'pnk_db2_p1_in_2025.sqlite').

Descriptive & Comparative Analytical Methodology:
-------------------------------------------------
This module implements a descriptive framework for comparing clusters:
1. Cluster vs. Entire Clustered Cohort Benchmark:
   - Evaluates whether each cluster deviates significantly from the overall cohort.
   - Continuous variables: Non-parametric Mann-Whitney U test.
   - Categorical / Binary variables: Pearson Chi-squared test with exact Fisher's
     test fallback whenever contingency tables contain sparse cells (< 5).
   - False Discovery Rate (FDR): Multiplicity adjustment using Benjamini-Hochberg (BH)
     procedure, reporting both raw and FDR-corrected p-values.
2. Comprehensive Visualization Suite:
   - Violin plots (continuous variables) with medians, IQR, and significance asterisks.
   - Stacked bar charts (categorical variables, 0-100%) with cohort average lines.
   - Lollipop plots (unified two-panel Standardized Mean Differences [Cohen's d] for continuous
     variables and Risk Differences [RD] for categorical variables vs. Rest-of-Cohort, sharing a 0 parity midline).
   - Heatmaps with cluster-specific color scales and population reference column.
   - Forest plots (Risk Ratios and Risk Differences with 95% CIs for binary clinical outcomes).
   - Multi-panel clinical summary grid (6-panel cohort overview).
3. Complete Decoupling & Notebook Configurability:
   - All database paths, table names, cluster selection parameters (weight, algorithm, k),
     variable subsets, and output paths are strictly passed from notebook calls.

Pseudocode:
-----------
1. RESOLUTION & INGESTION:
   a. Ingest user configuration from notebook:
      - Option A: Explicit weight ('wgc07_eb03'), algorithm ('pam'), and k (7).
      - Option B: Direct cluster_column ('hybrid_wgc07_eb03_pam_k7').
   b. Match active cluster column and medoid column in SQLite table.
   c. Merge cluster assignments with cohort table on 'medical_record_id'.
   d. Standardize active cluster column to 'cluster_id' (0 to k-1).
   e. Identify and report exemplar medoid patients.

2. STATISTICAL PROFILING (CLUSTER VS. ENTIRE COHORT BENCHMARK):
   a. Compute population benchmark statistics:
      - Continuous: Mean (+- SD) and Median [IQR].
      - Categorical/Binary: Count N and Percentage (%).
   b. For each cluster (0 to k-1):
      - Continuous: Compute cluster Mean (+- SD); test vs cohort via Mann-Whitney U test.
      - Categorical: Compute cluster count N (%); test vs cohort via Pearson Chi-square
        with exact Fisher's test fallback if any contingency cell < 5.
   c. Apply Benjamini-Hochberg False Discovery Rate (FDR) adjustment across comparisons.
   d. Format two persistent SQLite tables:
      - Detailed table: Contains raw p-values and FDR-corrected p-values.
      - Publication table: Formats significant divergences using asterisks
        (* p < 0.05 raw, ** p_FDR < 0.05).
   e. Save both tables to output database (e.g. pnk_db2_p1_out_2025.sqlite).

3. MULTI-VARIABLE & CLINICAL VISUALIZATIONS:
   a. Continuous variables: Multi-cluster horizontal/split violin plots with IQR,
      medians, and asterisks (* / **).
   b. Categorical variables: 100% stacked bar charts showing presence/absence proportions,
      population benchmark line, N (%) annotations, and asterisks.
   c. Deviations from rest-of-cohort: Lollipop plot displaying Standardized Mean Differences (Cohen's d)
      for continuous variables and Risk Differences (RD in percentage points) for categorical variables
      sharing a common 0 parity midline.
   d. Multi-attribute profiles: Heatmap with cluster-specific color scales and population reference.
   e. Prospective outcomes: Forest plots of Risk Ratios (RR) and Risk Differences (RD) with 95% CIs.
   f. Summary Grid: 6-panel overview of cohort demographics, BMI, follow-up, and weight loss.

Author: Gemini 3.8 Flash High in Antigravity
Revised: 28Oct26
Target Environment: Python 3.10+ / SQLite3 / Pandas / NumPy / SciPy / Seaborn / Matplotlib
================================================================================
"""

import os
import sys
import json
import sqlite3
import re
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, kruskal, norm
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower, GofChisquarePower
import statsmodels.stats.proportion as smprop
from matplotlib.colors import LinearSegmentedColormap

# Suppress minor visualization warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# DEFAULT COLOR PALETTES & VISUAL CONSTANTS
# =============================================================================

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


DEFAULT_PALETTE = _load_default_cluster_palette()
POPULATION_COLOR = "#4D4D4D"         # Neutral dark gray for cohort benchmark
CLUSTER_COLOR = "aquamarine"         # Accent cluster color for split violins
ACHIEVED_COLOR = "#4361EE"          # Vibrant blue for outcome achieved / category 1
NOT_ACHIEVED_COLOR = "#EC5B57"      # Coral/red for not achieved / category 0


# =============================================================================
# DEFAULT VARIABLE CATALOGUES & HUMAN-READABLE NAME MAPPINGS
# =============================================================================

# Comprehensive clinical catalog across the 4 key evaluation domains
DEFAULT_VARIABLE_DOMAINS: Dict[str, List[str]] = {
    "Weight gain causes": [
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
        "none_of_above",
    ],
    "Eating behaviors": [
        "hunger_yn",
        "satiety_yn",
        "emotional_eating_yn",
        "emotional_eating_value_likert",
        "quantity_control_likert",
        "impulse_control_likert",
    ],
    "Baseline characteristics": [
        "sex_f",
        "age",
        # "height_m",
        "baseline_weight_kg",
        "baseline_bmi",
        "baseline_fat_%",
        "baseline_muscle_%",
    ],
    "Clinical outcomes": [
        "total_followup_days",
        "total_wl_%",
        "bmi_reduction",
        "max_wl_%",
        "days_to_nadir",
        "weight_regain_%", 
        "60d_dropout",
        "60d_wl_%",
        "60d_bmi_reduction",
        "10%_wl_achieved",
        "days_to_10%_wl",
        "5%_wl_achieved",
        "days_to_5%_wl",
    ],
}

# Standard default list of comparison variables
COMPARISON_VARIABLES: List[str] = (
    DEFAULT_VARIABLE_DOMAINS["Weight gain causes"]
    + DEFAULT_VARIABLE_DOMAINS["Eating behaviors"]
    + DEFAULT_VARIABLE_DOMAINS["Baseline characteristics"]
    + DEFAULT_VARIABLE_DOMAINS["Clinical outcomes"]
)

# Comprehensive dictionary mapping raw database column names to publication-ready labels
DEFAULT_NAME_MAP: Dict[str, str] = {
    # Sample size
    "N": "N (sample size)",
    "n": "N (sample size)",
    "N (sample size)": "N (sample size)",

    # Baseline Demographics & Anthropometry
    "sex_f": "Sex (female)",
    "age": "Age (years)",
    "height_m": "Height (m)",
    "baseline_weight_kg": "Baseline weight (kg)",
    "baseline_bmi": "Baseline BMI (kg/m²)",
    "baseline_fat_%": "Baseline fat mass (%)",
    "baseline_muscle_%": "Baseline muscle mass (%)",

    # Eating Behaviors (3 Binary + 3 Likert)
    "hunger_yn": "Hunger (y/n)",
    "satiety_yn": "Satiety (y/n)",
    "emotional_eating_yn": "Emotional eating (y/n)",
    "emotional_eating_value_likert": "Emotional eating score (1-10)",
    "quantity_control_likert": "Quantity control score (1-10)",
    "impulse_control_likert": "Impulse control score (1-10)",

    # 12 Weight Gain Causes (NLP Attributions)
    "womens_health_and_pregnancy": "Women's health & pregnancy",
    "mental_health": "Mental health",
    "family_issues": "Family issues",
    "medication_disease_injury": "Medication, disease or injury",
    "physical_inactivity": "Physical inactivity",
    "eating_habits": "Eating habits",
    "schedule": "Schedule",
    "smoking_cessation": "Smoking cessation",
    "treatment_discontinuation_or_relapse": "Treatment discontinuation or relapse",
    "pandemic": "Pandemic",
    "lifestyle_circumstances": "Lifestyle, circumstances",
    "none_of_above": "None of the above",

    # Weight Loss Outcomes & Trajectories
    "total_followup_days": "Follow-up length (days)",
    "dietitian_visits": "Dietitian visits (n)",
    "nr_total_measurements": "Total measurements (n)",
    "avg_days_between_measurements": "Measurement interval (days)",
    # "last_aval_weight_kg": "Last measured weight (kg)",
    "total_wl_kg": "Total weight loss (kg)",
    "total_wl_%": "Total weight loss (%)",
    # "final_bmi": "Final BMI (kg/m²)",
    "bmi_reduction": "BMI reduction (kg/m²)",
    "max_wl_%": "Maximum weight loss (%)",
    "days_to_nadir": "Days to nadir (days)",
    "weight_regain_%": "Weight regain (%)", 
    # "last_aval_fat_%": "Last measured fat mass (%)",
    "total_fat_loss_%": "Total fat mass loss (%)",
    # "last_aval_muscle_%": "Last measured muscle mass (%)",
    "total_muscle_change_%": "Total muscle mass change (%)",
    "instant_dropout": "Instant dropout (y/n)",

    # Dynamic time windows (40d, 60d, 80d)
    "40d_dropout": "Dropout at 40 days (y/n)",
    "40d_wl_%": "Weight loss at 40 days (%)",
    "40d_bmi_reduction": "BMI reduction at 40 days (kg/m²)",
    "40d_fat_loss_%": "Fat mass loss at 40 days (%)",
    "40d_muscle_change_%": "Muscle mass change at 40 days (%)",
    "60d_dropout": "Dropout at 60 days (y/n)",
    "60d_wl_%": "Weight loss at 60 days (%)",
    "60d_bmi_reduction": "BMI reduction at 60 days (kg/m²)",
    "60d_fat_loss_%": "Fat mass loss at 60 days (%)",
    "60d_muscle_change_%": "Muscle mass change at 60 days (%)",
    "80d_dropout": "Dropout at 80 days (y/n)",
    "80d_wl_%": "Weight loss at 80 days (%)",
    "80d_bmi_reduction": "BMI reduction at 80 days (kg/m²)",
    "80d_fat_loss_%": "Fat mass loss at 80 days (%)",
    "80d_muscle_change_%": "Muscle mass change at 80 days (%)",

    # Milestones
    "5%_wl_achieved": "Achieved 5% weight loss (y/n)",
    "days_to_5%_wl": "Days to 5% weight loss",
    "10%_wl_achieved": "Achieved 10% weight loss (y/n)",
    "days_to_10%_wl": "Days to 10% weight loss",
    "15%_wl_achieved": "Achieved 15% weight loss (y/n)",
    "days_to_15%_wl": "Days to 15% weight loss",
}


# =============================================================================
# HELPER FUNCTIONS - CONFIGURATION & LABEL MANAGEMENT
# =============================================================================

def _resolve_path(path: str) -> str:
    """Resolve relative paths to absolute paths from script location or working directory."""
    if not path or os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved = os.path.join(script_dir, path)
    if os.path.exists(resolved):
        return resolved
    if os.path.exists(path):
        return os.path.abspath(path)
    return path


def load_name_map(source: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, str]:
    """
    Loads variable name mappings from a dictionary or JSON file, falling back to DEFAULT_NAME_MAP.

    Parameters:
    -----------
    source : Optional[Union[str, Dict[str, str]]]
        Path to JSON file or dictionary containing variable mappings.

    Returns:
    --------
    Dict[str, str]: Merged variable name dictionary.
    """
    merged_map = dict(DEFAULT_NAME_MAP)
    if source is None:
        return merged_map

    if isinstance(source, dict):
        merged_map.update(source)
        return merged_map

    if isinstance(source, str):
        json_path = _resolve_path(source)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    custom_map = json.load(f)
                    merged_map.update(custom_map)
                print(f"[Config] Loaded custom variable names from: '{json_path}'")
            except Exception as e:
                print(f"Warning: Failed to load name map JSON '{json_path}': {e}. Using default.")
        else:
            # Only print warning if a non-default custom path was provided
            if source not in ["human_readable_variable_names.json"]:
                print(f"Warning: Name map file not found at '{json_path}'. Using defaults.")
    return merged_map


def load_cluster_config(source: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Loads cluster labels and color configurations from a dictionary or JSON file, if available.

    Parameters:
    -----------
    source : Optional[Union[str, Dict[str, Any]]]
        Path to JSON file or dictionary containing cluster configurations.

    Returns:
    --------
    Dict[str, Any]: Configuration dictionary with 'cluster_labels' and 'cluster_colors'.
    """
    default_config: Dict[str, Any] = {"cluster_labels": {}, "cluster_colors": {}}

    # Load base colors from cluster_color_dict.json
    default_json = _resolve_path("cluster_color_dict.json")
    if os.path.exists(default_json):
        try:
            with open(default_json, "r", encoding="utf-8") as f:
                d_colors = json.load(f)
            default_config["cluster_colors"].update(d_colors.get("cluster_colors", d_colors))
        except Exception:
            pass

    if source is None:
        return default_config

    if isinstance(source, dict):
        if "cluster_colors" in source:
            default_config["cluster_colors"].update(source["cluster_colors"])
        if "cluster_labels" in source:
            default_config["cluster_labels"].update(source["cluster_labels"])
        return default_config

    if isinstance(source, str):
        target = source
        if not target.endswith(".json") and os.path.exists(_resolve_path(f"{target}.json")):
            target = f"{target}.json"
        json_path = _resolve_path(target)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                default_config["cluster_labels"].update(cfg.get("cluster_labels", {}))
                default_config["cluster_colors"].update(cfg.get("cluster_colors", cfg))
                print(f"[Config] Loaded cluster configuration from: '{json_path}'")
                return default_config
            except Exception as e:
                print(f"Warning: Failed to load cluster config '{json_path}': {e}. Using defaults.")
        else:
            if source not in ["cluster_config.json"]:
                print(f"Warning: Cluster config file not found at '{json_path}'. Using defaults.")
    return default_config


def get_nice_name(variable: str, name_map: Optional[Dict[str, str]] = None) -> str:
    """Returns human-readable name for variable, falling back to cleaned title."""
    if name_map is None:
        name_map = DEFAULT_NAME_MAP
    return name_map.get(variable, variable.replace("_", " ").title())


def get_cluster_label(cluster_id: Union[int, str], cluster_config: Optional[Dict[str, Any]] = None) -> str:
    """Returns human-readable label for a given cluster ID, falling back to 1-indexed 'Cluster {id + 1}'."""
    if cluster_config is None:
        cluster_config = {}
    labels = cluster_config.get("cluster_labels", {})
    str_id = str(cluster_id)
    if str_id in labels:
        return labels[str_id]
    if cluster_id in labels:
        return labels[cluster_id]
    try:
        cid_int = int(cluster_id)
        if cid_int == -1:
            return "Noise / Outliers"
        return f"Cluster {cid_int + 1}"
    except (ValueError, TypeError):
        return f"Cluster {cluster_id}"


def get_cluster_color(
    cluster_id: Union[int, str],
    cluster_config: Optional[Dict[str, Any]] = None,
    default_palette: Optional[List[Any]] = None,) -> Any:
    """Returns aesthetic hex color or RGB tuple for a given cluster ID."""
    if default_palette is None:
        default_palette = DEFAULT_PALETTE
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
        return default_palette[cid_int % len(default_palette)]
    except (ValueError, TypeError):
        match = re.search(r"(\d+)", str_id)
        if match:
            c_num = int(match.group(1))
            if c_num >= 1:
                return default_palette[(c_num - 1) % len(default_palette)]
        return "#7f7f7f"


def _extract_percentage_from_table_cell(cell_value: Any) -> float:
    """Extracts percentage float from table cell strings formatted as 'N (X.X%)'."""
    try:
        match = re.search(r"\((\d+\.?\d*)%\)", str(cell_value))
        if match:
            return float(match.group(1))
        return np.nan
    except Exception:
        return np.nan


def _extract_mean_sd_from_table_cell(cell_value: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extracts mean and SD floats from table cell strings formatted as 'Mean (±SD)' or 'Mean (±SD)*'."""
    try:
        s = str(cell_value).strip()
        match = re.search(r"(\d+\.?\d*)\s*\([^\d]*(\d+\.?\d*)\)", s)
        if match:
            return float(match.group(1)), float(match.group(2))
        match_single = re.search(r"^(\d+\.?\d*)", s)
        if match_single:
            return float(match_single.group(1)), None
        return None, None
    except Exception:
        return None, None


def _parse_cluster_column_header(header: str, cluster_config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Parses cluster integer ID from table header strings."""
    if cluster_config:
        cluster_labels = cluster_config.get("cluster_labels", {})
        for c_id_str, label in cluster_labels.items():
            if header.startswith(f"{label}:"):
                try:
                    return int(c_id_str)
                except ValueError:
                    continue
    match = re.match(r"^Cluster\s+(\d+):", header)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# CLUSTER COLUMN RESOLUTION HELPER
# =============================================================================

def resolve_cluster_column_name(
    weight: Optional[str] = None,
    algorithm: Optional[str] = None,
    k: Optional[int] = None,
    cluster_column: Optional[str] = None,
    available_columns: Optional[List[str]] = None,) -> Tuple[str, Optional[str]]:
    """
    Flexibly identifies the exact cluster assignment column and medoid column in SQLite.

    Supports both:
    1. Direct specification via `cluster_column` (e.g. 'hybrid_wgc07_eb03_pam_k7').
    2. Parametric specification via `weight`, `algorithm`, and `k`
       (e.g. weight='wgc07_eb03', algorithm='pam', k=7).

    Parameters:
    -----------
    weight : Optional[str]
        Hybrid weighting key (e.g. 'wgc05_eb05', 'wgc06_eb04', 'wgc067_eb033', 'wgc07_eb03').
    algorithm : Optional[str]
        Clustering algorithm (e.g. 'pam', 'hclust_complete', 'hclust_average').
    k : Optional[int]
        Cluster count (e.g. 2 to 10).
    cluster_column : Optional[str]
        Explicit column name if already known.
    available_columns : Optional[List[str]]
        List of columns present in the labels table for validation.

    Returns:
    --------
    Tuple[str, Optional[str]]:
        (resolved_cluster_column, resolved_medoid_column)
    """
    if cluster_column is not None:
        target_col = cluster_column
    else:
        if weight is None or algorithm is None or k is None:
            raise ValueError(
                "Either 'cluster_column' OR all three of ('weight', 'algorithm', 'k') must be provided."
            )
        # Strip any leading 'hybrid_' if already passed in weight
        clean_weight = weight.replace("hybrid_", "")
        target_col = f"hybrid_{clean_weight}_{algorithm.lower()}_k{k}"

    if available_columns is not None:
        if target_col not in available_columns:
            # Fallback attempts for alternative naming conventions
            clean_weight = (weight or "").replace("hybrid_", "")
            algo = (algorithm or "").lower()
            candidates = [
                target_col,
                f"{clean_weight}_{algo}_k{k}" if weight and algorithm and k else "",
                f"{algo}_k{k}" if algorithm and k else "",
                f"hybrid_{clean_weight}_k{k}" if weight and k else "",
            ]
            matched = None
            for cand in candidates:
                if cand and cand in available_columns:
                    matched = cand
                    break
            if matched is None:
                clustering_cols = [c for c in available_columns if any(a in c for a in ["pam", "hclust", "kmeans"])]
                raise ValueError(
                    f"Cluster column '{target_col}' not found. Available clustering columns: {clustering_cols}"
                )
            target_col = matched

    # Check for corresponding medoid column
    medoid_col = None
    if available_columns is not None:
        # Standard medoid column pattern: replace '_k{k}' with '_medoid_k{k}'
        candidate_medoid = re.sub(r"_k(\d+)$", r"_medoid_k\1", target_col)
        if candidate_medoid in available_columns:
            medoid_col = candidate_medoid

    return target_col, medoid_col


# =============================================================================
# DATA INGESTION & COHORT MERGING
# =============================================================================

def load_and_merge_cluster_data(
    cluster_db_path: str,
    main_db_path: Optional[str] = None,
    cluster_table: str = "jaccard_gower_cluster_labels",
    cluster_column: Optional[str] = None,
    weight: Optional[str] = None,
    algorithm: Optional[str] = None,
    k: Optional[int] = None,
    outcome_table: str = "timetoevent_eb_wgc_compl",
    patient_id_col: str = "medical_record_id",
    include_outliers: bool = True,) -> pd.DataFrame:
    """
    Loads clinical patient data and merges with cluster assignments from SQLite.

    Parameters:
    -----------
    cluster_db_path : str
        Path to SQLite database containing cluster assignments.
    main_db_path : Optional[str]
        Path to database containing patient cohort data (defaults to cluster_db_path if None).
    cluster_table : str
        SQLite table name containing cluster labels (default: 'jaccard_gower_cluster_labels').
    cluster_column : Optional[str]
        Explicit cluster column name (e.g. 'hybrid_wgc07_eb03_pam_k7').
    weight : Optional[str]
        Distance weighting key (e.g. 'wgc07_eb03', 'wgc05_eb05').
    algorithm : Optional[str]
        Clustering algorithm (e.g. 'pam', 'hclust_complete').
    k : Optional[int]
        Number of clusters (e.g. 7).
    outcome_table : str
        SQLite cohort table name (default: 'timetoevent_eb_wgc_compl').
    patient_id_col : str
        Primary key column for cohort merging (default: 'medical_record_id').
    include_outliers : bool
        Whether to keep unclustered patients labeled as -1 (default: True).

    Returns:
    --------
    pd.DataFrame:
        Merged DataFrame with standardized column 'cluster_id' and metadata attribute 'cluster_column'.
    """
    if main_db_path is None:
        main_db_path = cluster_db_path

    cluster_db_path = _resolve_path(cluster_db_path)
    main_db_path = _resolve_path(main_db_path)

    print("\n" + "=" * 70)
    print("LOADING & MERGING CLINICAL COHORT WITH CLUSTER ASSIGNMENTS")
    print("=" * 70)
    print(f"Cluster DB:   {cluster_db_path}")
    print(f"Cohort DB:    {main_db_path}")
    print(f"Labels Table: {cluster_table}")
    print(f"Cohort Table: {outcome_table}")

    if not os.path.exists(cluster_db_path):
        raise FileNotFoundError(f"Cluster database not found at: '{cluster_db_path}'")
    if not os.path.exists(main_db_path):
        raise FileNotFoundError(f"Main cohort database not found at: '{main_db_path}'")

    # 1. Inspect and load cluster labels table
    with sqlite3.connect(cluster_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        if cluster_table not in tables:
            raise ValueError(f"Table '{cluster_table}' not found in '{cluster_db_path}'. Available: {tables}")

        labels_df = pd.read_sql_query(f"SELECT * FROM {cluster_table}", conn)

    # 2. Resolve active cluster column and medoid column
    active_col, medoid_col = resolve_cluster_column_name(
        weight=weight,
        algorithm=algorithm,
        k=k,
        cluster_column=cluster_column,
        available_columns=labels_df.columns.tolist(),
    )
    print(f"  ✓ Resolved Cluster Column: '{active_col}'")
    if medoid_col:
        print(f"  ✓ Detected Medoid Designation Column: '{medoid_col}'")

    # 3. Load cohort data
    with sqlite3.connect(main_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        if outcome_table not in tables:
            raise ValueError(f"Table '{outcome_table}' not found in '{main_db_path}'. Available: {tables}")

        cohort_df = pd.read_sql_query(f"SELECT * FROM {outcome_table}", conn)

    print(f"  ✓ Loaded {len(cohort_df)} cohort records across {len(cohort_df.columns)} variables.")

    # 4. Merge on patient identifier
    cols_to_pull = [patient_id_col, active_col]
    if medoid_col and medoid_col in labels_df.columns:
        cols_to_pull.append(medoid_col)

    merged_df = cohort_df.merge(labels_df[cols_to_pull], on=patient_id_col, how="inner")

    # Standardize column naming: create 'cluster_id' as integer
    merged_df["cluster_id"] = merged_df[active_col].astype(int)

    # Exclude outliers (-1) if requested
    if not include_outliers:
        n_before = len(merged_df)
        merged_df = merged_df[merged_df["cluster_id"] != -1].copy()
        n_excluded = n_before - len(merged_df)
        if n_excluded > 0:
            print(f"  ✓ Excluded {n_excluded} unclustered outlier records (-1).")

    unique_clusters = sorted([c for c in merged_df["cluster_id"].unique() if c != -1])
    print(f"  ✓ Merged cohort: {len(merged_df)} patients across {len(unique_clusters)} valid clusters: {unique_clusters}")

    # Cluster sizes diagnostic
    cluster_counts = merged_df["cluster_id"].value_counts().sort_index()
    for cid, cnt in cluster_counts.items():
        pct = (cnt / len(merged_df)) * 100
        label_str = f"Cluster {int(cid) + 1}" if cid != -1 else "Outliers"
        print(f"    - {label_str}: {cnt} patients ({pct:.1f}%)")

    # Exemplar medoid patient IDs diagnostic
    if medoid_col and medoid_col in merged_df.columns:
        medoids_found: Dict[int, str] = {}
        for cid in unique_clusters:
            med_rows = merged_df[(merged_df["cluster_id"] == cid) & (merged_df[medoid_col] == 1)]
            if len(med_rows) > 0:
                med_id = str(med_rows.iloc[0][patient_id_col])
                medoids_found[cid] = med_id
        if medoids_found:
            print(f"  ✓ Designated Exemplar Medoids (Representative Patients): {medoids_found}")
            merged_df.attrs["medoids"] = medoids_found

    # Attach metadata to dataframe
    merged_df.attrs["cluster_column"] = active_col
    merged_df.attrs["unique_clusters"] = unique_clusters
    print("=" * 70 + "\n")

    return merged_df


# =============================================================================
# STATISTICAL TESTING FUNCTIONS
# =============================================================================

def mann_whitney_u_test(series1: pd.Series, series2: pd.Series) -> float:
    """Performs two-sided Mann-Whitney U test between two numeric continuous series."""
    s1 = pd.to_numeric(series1, errors="coerce").dropna()
    s2 = pd.to_numeric(series2, errors="coerce").dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p_val = mannwhitneyu(s1, s2, alternative="two-sided")
        return float(p_val)
    except ValueError:
        return 1.0


def chi_squared_test(series1: pd.Series, series2: pd.Series) -> float:
    """
    Performs Pearson's Chi-squared test on categorical distributions,
    falling back to Fisher's exact test for 2x2 tables with expected frequencies < 5.
    """
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return np.nan

    # Create 2xK contingency table comparing sample 1 vs sample 2
    table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2]),
    )
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    try:
        chi2, p_val, dof, expected = chi2_contingency(table)
        # Use Fisher's exact test if expected frequency is sparse (< 5) in 2x2 table
        if (expected < 5).any() and table.shape == (2, 2):
            _, p_val = fisher_exact(table)
        return float(p_val)
    except Exception:
        return np.nan


def _infer_variable_type(
    series: pd.Series,
    threshold: int = 10,
    variable_name: Optional[str] = None,) -> str:
    """
    Infers if a clinical variable is continuous or categorical.
    Incorporates explicit biological domain rules for clinical variables.
    """
    if variable_name:
        var_lower = variable_name.lower()
        # Explicit binary indicators
        if any(ind in var_lower for ind in ["_yn", "_achieved", "_dropout", "sex_f", "instant_dropout"]):
            return "categorical"
        # Explicit continuous/ordinal indicators
        if any(ind in var_lower for ind in ["_likert", "score", "weight", "bmi", "fat", "muscle", "age", "days", "_kg", "_%"]):
            return "continuous"

    clean_series = series.dropna()
    if len(clean_series) == 0:
        return "categorical"

    # Boolean or strictly binary 0/1 data
    if clean_series.dtype == bool or set(clean_series.unique()).issubset({0, 1, 0.0, 1.0, True, False}):
        return "categorical"

    unique_count = clean_series.nunique()
    return "categorical" if unique_count <= threshold else "continuous"


def calculate_cluster_pvalues(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str = "cluster_id",
    is_categorical: bool = False,) -> Dict[int, float]:
    """
    Calculates p-values comparing each individual cluster to the entire clustered cohort.

    Returns:
    --------
    Dict[int, float]: Mapping cluster_id -> raw p-value.
    """
    pvalues: Dict[int, float] = {}
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    population_data = cluster_df[variable]

    for cid in clusters:
        cluster_subset = cluster_df[cluster_df[cluster_col] == cid][variable]
        if is_categorical:
            p_val = chi_squared_test(cluster_subset, population_data)
        else:
            p_val = mann_whitney_u_test(cluster_subset, population_data)
        pvalues[cid] = p_val

    return pvalues


def apply_fdr_correction(pvalues: Dict[int, float]) -> Dict[int, float]:
    """Applies Benjamini-Hochberg (BH) False Discovery Rate correction to p-values."""
    if not pvalues:
        return {}
    valid_items = [(k, v) for k, v in pvalues.items() if pd.notna(v)]
    if not valid_items:
        return pvalues
    keys, vals = zip(*valid_items)
    try:
        _, corrected, _, _ = multipletests(vals, method="fdr_bh")
        return dict(zip(keys, [float(c) for c in corrected]))
    except Exception:
        return pvalues


def calculate_cohens_d_vs_rest(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str = "cluster_id",
    use_absolute_magnitude: Optional[bool] = None,) -> Dict[int, Dict[str, float]]:
    """
    Calculates Standardized Mean Differences (Cohen's d) for a continuous variable
    comparing each cluster against the remaining cohort (Rest-of-Cohort).

    Statistical Methodology:
    ------------------------
    Let cluster C have sample size n_C, mean m_C, and sample variance s_C^2.
    Let the remaining cohort R (all patients where cluster != C and cluster != -1)
    have sample size n_R, mean m_R, and sample variance s_R^2.

    Pooled standard deviation:
        s_pooled = sqrt( ((n_C - 1) * s_C^2 + (n_R - 1) * s_R^2) / (n_C + n_R - 2) )

    Cohen's d:
        d = (m_C - m_R) / s_pooled

    Absolute Magnitude & Directionality:
    ------------------------------------
    In prospective clinical databases, weight loss, fat mass reduction, and BMI reductions
    are frequently stored as negative numbers (e.g. -10% delta from baseline).
    If evaluated directly on raw signed negative numbers, a cluster that lost LESS weight
    (e.g. -5%) has a higher arithmetic mean than a cluster that lost MORE weight (e.g. -12%),
    which artificially inverts Cohen's d (making lesser weight loss appear as a positive effect).
    
    When `use_absolute_magnitude=True` (or auto-detected when the cohort mean < 0 or variable
    name indicates a loss/reduction), the absolute magnitude of reduction (|x|) is evaluated:
      - Positive Cohen's d (d > 0): Cluster achieved GREATER weight loss / reduction than Rest-of-Cohort.
      - Negative Cohen's d (d < 0): Cluster achieved LESS weight loss / reduction than Rest-of-Cohort.
      - Parity (d = 0): Exact equality in weight loss between cluster and Rest-of-Cohort.

    Statistical Significance:
        - Two-sample non-parametric Mann-Whitney U test between cluster C and rest-of-cohort R.
        - Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjustment across all clusters.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with standardized cluster assignments.
    variable : str
        Continuous variable column name.
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    use_absolute_magnitude : Optional[bool]
        Whether to analyze absolute values (|x|). If None, automatically set to True
        if the cohort mean is negative (< 0) or column name indicates a loss/reduction.

    Returns:
    --------
    Dict[int, Dict[str, float]]:
        Mapping from cluster_id to dictionary containing:
        - 'd': Standardized Mean Difference (Cohen's d)
        - 'p_raw': Two-sided Mann-Whitney U p-value vs Rest-of-Cohort
        - 'p_fdr': Benjamini-Hochberg FDR-adjusted q-value
        - 'mean_cluster': Mean in cluster C
        - 'mean_rest': Mean in rest-of-cohort R
        - 'sd_pooled': Pooled standard deviation
        - 'is_absolute': Whether absolute magnitude transformation was applied
    """
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    results: Dict[int, Dict[str, float]] = {}
    raw_p_map: Dict[int, float] = {}

    # Auto-detect whether to focus on absolute magnitude (for loss / reduction variables)
    if use_absolute_magnitude is None:
        clean_pop = cluster_df[variable].dropna()
        var_lower = variable.lower()
        is_reduction_name = any(k in var_lower for k in ["_wl_", "loss", "reduction", "_wl_kg", "_wl_%"])
        pop_mean = clean_pop.mean() if len(clean_pop) > 0 else 0.0
        use_absolute_magnitude = bool(is_reduction_name or pop_mean < 0)

    for cid in clusters:
        c_vals = cluster_df[cluster_df[cluster_col] == cid][variable].dropna()
        r_vals = cluster_df[(cluster_df[cluster_col] != cid) & (cluster_df[cluster_col] != -1)][variable].dropna()

        nc, nr = len(c_vals), len(r_vals)
        if nc < 2 or nr < 2:
            results[cid] = {
                "d": 0.0,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_raw": 1.0,
                "p_fdr": 1.0,
                "mean_cluster": float(c_vals.mean()) if nc > 0 else np.nan,
                "mean_rest": float(r_vals.mean()) if nr > 0 else np.nan,
                "sd_pooled": np.nan,
                "is_absolute": use_absolute_magnitude,
            }
            raw_p_map[cid] = 1.0
            continue

        if use_absolute_magnitude:
            c_vals_calc = c_vals.abs()
            r_vals_calc = r_vals.abs()
        else:
            c_vals_calc = c_vals
            r_vals_calc = r_vals

        mc, mr = float(c_vals_calc.mean()), float(r_vals_calc.mean())
        vc, vr = float(c_vals_calc.var(ddof=1)), float(r_vals_calc.var(ddof=1))

        # Pooled sample standard deviation
        s_pooled = float(np.sqrt(((nc - 1) * vc + (nr - 1) * vr) / (nc + nr - 2)))
        d_val = float((mc - mr) / s_pooled) if s_pooled > 0 else 0.0

        # Standard error & 95% Confidence Interval for Cohen's d (Hedges & Olkin 1985 / Borenstein 2009)
        # SE(d) = sqrt((nc + nr)/(nc * nr) + d^2 / (2 * (nc + nr - 2)))
        se_d = float(np.sqrt((nc + nr) / (nc * nr) + (d_val ** 2) / (2.0 * (nc + nr - 2))))
        ci_lower = float(d_val - 1.96 * se_d)
        ci_upper = float(d_val + 1.96 * se_d)

        p_raw = mann_whitney_u_test(c_vals_calc, r_vals_calc)
        raw_p_map[cid] = p_raw

        results[cid] = {
            "d": d_val,
            "se": se_d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_raw": p_raw,
            "mean_cluster": mc,
            "mean_rest": mr,
            "sd_pooled": s_pooled,
            "is_absolute": use_absolute_magnitude,
        }

    fdr_map = apply_fdr_correction(raw_p_map)
    for cid in clusters:
        results[cid]["p_fdr"] = fdr_map.get(cid, results[cid]["p_raw"])

    return results


def calculate_risk_difference_vs_rest(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str = "cluster_id",
    as_percent: bool = True,) -> Dict[int, Dict[str, float]]:
    """
    Calculates Risk Differences (RD) for a binary variable comparing each cluster
    against the remaining cohort (Rest-of-Cohort).

    Statistical Methodology:
    ------------------------
    Let cluster C have event prevalence p_C = events_C / n_C.
    Let the remaining cohort R (where cluster != C and cluster != -1) have
    event prevalence p_R = events_R / n_R.

    Risk Difference:
        RD = p_C - p_R  (or expressed in percentage points: (p_C - p_R) * 100)

    Interpretation:
        - RD = 0 represents exact parity between cluster and the rest-of-cohort.
        - Positive values (RD > 0) indicate higher event rate in the cluster.
        - Negative values (RD < 0) indicate lower event rate in the cluster.

    Statistical Significance:
        - Two-sided Chi-squared test with Fisher's exact test fallback for small cell counts.
        - Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjustment across all clusters.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with standardized cluster assignments.
    variable : str
        Binary/categorical variable column name (coded as 0/1, boolean, or integers).
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    as_percent : bool
        Whether to express Risk Difference in percentage points (default: True, e.g. +15.0%).

    Returns:
    --------
    Dict[int, Dict[str, float]]:
        Mapping from cluster_id to dictionary containing:
        - 'rd': Risk Difference (in percentage points if as_percent=True)
        - 'p_raw': Two-sided Chi-squared / Fisher's exact p-value vs Rest-of-Cohort
        - 'p_fdr': Benjamini-Hochberg FDR-adjusted q-value
        - 'prop_cluster': Proportion in cluster C
        - 'prop_rest': Proportion in rest-of-cohort R
    """
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    results: Dict[int, Dict[str, float]] = {}
    raw_p_map: Dict[int, float] = {}

    for cid in clusters:
        c_vals = cluster_df[cluster_df[cluster_col] == cid][variable].dropna()
        r_vals = cluster_df[(cluster_df[cluster_col] != cid) & (cluster_df[cluster_col] != -1)][variable].dropna()

        nc, nr = len(c_vals), len(r_vals)
        if nc == 0 or nr == 0:
            results[cid] = {
                "rd": 0.0,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_raw": 1.0,
                "p_fdr": 1.0,
                "prop_cluster": np.nan,
                "prop_rest": np.nan,
            }
            raw_p_map[cid] = 1.0
            continue

        pos_c = float((c_vals == 1).sum() if c_vals.dtype in [bool, int, np.int64] else c_vals.sum())
        pos_r = float((r_vals == 1).sum() if r_vals.dtype in [bool, int, np.int64] else r_vals.sum())

        pc = pos_c / nc
        pr = pos_r / nr

        rd_raw = pc - pr
        # Wald standard error for Risk Difference:
        # SE(RD) = sqrt(pc * (1 - pc) / nc + pr * (1 - pr) / nr)
        se_rd = float(np.sqrt((pc * (1.0 - pc) / nc) + (pr * (1.0 - pr) / nr)))
        mult = 100.0 if as_percent else 1.0
        rd_val = float(rd_raw * mult)
        ci_lower = float((rd_raw - 1.96 * se_rd) * mult)
        ci_upper = float((rd_raw + 1.96 * se_rd) * mult)

        p_raw = chi_squared_test(c_vals, r_vals)
        raw_p_map[cid] = p_raw

        results[cid] = {
            "rd": rd_val,
            "se": float(se_rd * mult),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_raw": p_raw,
            "prop_cluster": float(pc),
            "prop_rest": float(pr),
        }

    fdr_map = apply_fdr_correction(raw_p_map)
    for cid in clusters:
        results[cid]["p_fdr"] = fdr_map.get(cid, results[cid]["p_raw"])

    return results


def extract_pvalues_for_lollipop(
    results_df: pd.DataFrame,
    variables: List[str],
    cluster_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    cluster_config_path: Optional[Union[str, Dict]] = None,
    name_map_path: Optional[Union[str, Dict]] = None,) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Extracts raw and FDR-corrected p-values from results_df for lollipop plotting.
    """
    cluster_config = load_cluster_config(cluster_config_path)
    name_map = load_name_map(name_map_path)
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])

    pvalues_raw: Dict[str, Dict[int, float]] = {}
    pvalues_fdr: Dict[str, Dict[int, float]] = {}

    for var in variables:
        pvalues_raw[var] = {}
        pvalues_fdr[var] = {}
        nice_name = get_nice_name(var, name_map)
        var_row = results_df[results_df["Variable"] == nice_name]

        if var_row.empty:
            continue

        for cid in clusters:
            c_label = get_cluster_label(cid, cluster_config)
            p_raw_col = f"{c_label}: p-value"
            p_fdr_col = f"{c_label}: p-value (FDR-corrected)"

            if p_raw_col not in var_row.columns:
                p_raw_col = f"Cluster {int(cid) + 1}: p-value" if cid != -1 else "Cluster -1: p-value"
                if p_raw_col not in var_row.columns:
                    p_raw_col = f"Cluster {cid}: p-value"
            if p_fdr_col not in var_row.columns:
                p_fdr_col = f"Cluster {int(cid) + 1}: p-value (FDR-corrected)" if cid != -1 else "Cluster -1: p-value (FDR-corrected)"
                if p_fdr_col not in var_row.columns:
                    p_fdr_col = f"Cluster {cid}: p-value (FDR-corrected)"

            if p_raw_col in var_row.columns:
                val = var_row[p_raw_col].values[0]
                try:
                    pvalues_raw[var][cid] = float(val) if pd.notna(val) and val != "N/A" else np.nan
                except ValueError:
                    pvalues_raw[var][cid] = np.nan

            if p_fdr_col in var_row.columns:
                val_fdr = var_row[p_fdr_col].values[0]
                try:
                    pvalues_fdr[var][cid] = float(val_fdr) if pd.notna(val_fdr) and val_fdr != "N/A" else np.nan
                except ValueError:
                    pvalues_fdr[var][cid] = np.nan

    return pvalues_raw, pvalues_fdr


def calculate_risk_metrics(
    cluster_df: pd.DataFrame,
    outcome_variable: str,
    cluster_col: str = "cluster_id",) -> pd.DataFrame:
    """
    Calculates Risk Ratios (RR) and Risk Differences (RD) comparing each cluster to the entire cohort.

    Statistical Formulae:
    ---------------------
    - Risk Ratio: RR = Risk_cluster / Risk_population
      95% CI computed via log-transformation: exp(ln(RR) ± 1.96 * SE(ln(RR)))
    - Risk Difference: RD = Risk_cluster - Risk_population
      95% CI computed via Wald interval: RD ± 1.96 * SE(RD)
    """
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    results = []

    pop_events = float(cluster_df[outcome_variable].sum())
    pop_n = float(len(cluster_df))
    pop_risk = pop_events / pop_n if pop_n > 0 else 0.0

    for cid in clusters:
        subset = cluster_df[cluster_df[cluster_col] == cid]
        cluster_events = float(subset[outcome_variable].sum())
        cluster_n = float(len(subset))
        cluster_risk = cluster_events / cluster_n if cluster_n > 0 else 0.0

        # Risk Ratio (RR)
        if pop_risk > 0 and cluster_risk > 0:
            rr = cluster_risk / pop_risk
            se_log_rr = np.sqrt((1.0 / cluster_events - 1.0 / cluster_n) + (1.0 / pop_events - 1.0 / pop_n))
            rr_ci_lower = float(np.exp(np.log(rr) - 1.96 * se_log_rr))
            rr_ci_upper = float(np.exp(np.log(rr) + 1.96 * se_log_rr))
        else:
            rr, rr_ci_lower, rr_ci_upper = np.nan, np.nan, np.nan

        # Risk Difference (RD)
        rd = cluster_risk - pop_risk
        if cluster_n > 0 and pop_n > 0:
            se_rd = np.sqrt((cluster_risk * (1.0 - cluster_risk) / cluster_n) + (pop_risk * (1.0 - pop_risk) / pop_n))
            rd_ci_lower = float(rd - 1.96 * se_rd)
            rd_ci_upper = float(rd + 1.96 * se_rd)
        else:
            rd, rd_ci_lower, rd_ci_upper = np.nan, np.nan, np.nan

        results.append({
            "cluster_id": cid,
            "risk_cluster": cluster_risk,
            "risk_population": pop_risk,
            "risk_ratio": rr,
            "rr_ci_lower": rr_ci_lower,
            "rr_ci_upper": rr_ci_upper,
            "risk_difference": rd,
            "rd_ci_lower": rd_ci_lower,
            "rd_ci_upper": rd_ci_upper,
        })

    return pd.DataFrame(results)


def calculate_rest_of_cohort_effects(
    cluster_df: pd.DataFrame,
    continuous_variables: Optional[List[str]] = None,
    categorical_variables: Optional[List[str]] = None,
    cluster_col: str = "cluster_id",
    alpha: float = 0.05,
    rd_as_percent: bool = True,
    use_absolute_magnitude: Optional[bool] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes pairwise standardized effect sizes and 95% confidence intervals comparing
    each cluster against the remaining cohort (Rest-of-Cohort).

    Separates statistical estimation from visualization to maintain clean separation of concerns.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with cluster assignments.
    continuous_variables : Optional[List[str]]
        List of continuous column names to test (Cohen's d).
    categorical_variables : Optional[List[str]]
        List of binary/categorical column names to test (Risk Difference).
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    alpha : float
        Nominal significance threshold for significance asterisks (default: 0.05).
    rd_as_percent : bool
        Whether Risk Differences are scaled as percentage points (default: True).
    use_absolute_magnitude : Optional[bool]
        Whether continuous variables with negative conventions use absolute values (default: True).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - df_cont: DataFrame with columns ['variable', 'cluster_id', 'value', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'sig_str']
        - df_cat: DataFrame with columns ['variable', 'cluster_id', 'value', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'sig_str']
    """
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])

    cont_rows = []
    if continuous_variables:
        for var in continuous_variables:
            if var not in cluster_df.columns:
                continue
            smd_metrics = calculate_cohens_d_vs_rest(
                cluster_df,
                variable=var,
                cluster_col=cluster_col,
                use_absolute_magnitude=use_absolute_magnitude,
            )
            for cid in clusters:
                m = smd_metrics.get(
                    cid,
                    {"d": 0.0, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_raw": 1.0, "p_fdr": 1.0},
                )
                sig_str = ""
                if m["p_fdr"] < alpha:
                    sig_str = "**"
                elif m["p_raw"] < alpha:
                    sig_str = "*"
                cont_rows.append({
                    "variable": var,
                    "cluster_id": cid,
                    "value": m["d"],
                    "se": m.get("se", np.nan),
                    "ci_lower": m.get("ci_lower", np.nan),
                    "ci_upper": m.get("ci_upper", np.nan),
                    "p_raw": m.get("p_raw", np.nan),
                    "p_fdr": m.get("p_fdr", np.nan),
                    "sig_str": sig_str,
                })

    cat_rows = []
    if categorical_variables:
        for var in categorical_variables:
            if var not in cluster_df.columns:
                continue
            rd_metrics = calculate_risk_difference_vs_rest(
                cluster_df,
                variable=var,
                cluster_col=cluster_col,
                as_percent=rd_as_percent,
            )
            for cid in clusters:
                m = rd_metrics.get(
                    cid,
                    {"rd": 0.0, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "p_raw": 1.0, "p_fdr": 1.0},
                )
                sig_str = ""
                if m["p_fdr"] < alpha:
                    sig_str = "**"
                elif m["p_raw"] < alpha:
                    sig_str = "*"
                cat_rows.append({
                    "variable": var,
                    "cluster_id": cid,
                    "value": m["rd"],
                    "se": m.get("se", np.nan),
                    "ci_lower": m.get("ci_lower", np.nan),
                    "ci_upper": m.get("ci_upper", np.nan),
                    "p_raw": m.get("p_raw", np.nan),
                    "p_fdr": m.get("p_fdr", np.nan),
                    "sig_str": sig_str,
                })

    df_cont = pd.DataFrame(cont_rows) if cont_rows else pd.DataFrame()
    df_cat = pd.DataFrame(cat_rows) if cat_rows else pd.DataFrame()
    return df_cont, df_cat


# =============================================================================
# STATISTICAL POWER & MINIMUM DETECTABLE EFFECT SIZE (MDES) ANALYSIS
# =============================================================================

def calculate_cluster_power_analysis(
    cluster_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    alpha: float = 0.05,
    power: float = 0.80,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    print_summary: bool = True,) -> Dict[str, Any]:
    """
    Performs statistical power and Minimum Detectable Effect Size (MDES) analysis
    for Rest-of-Cohort comparisons based on the smallest cluster sample size in the cohort.

    Methodological Logic:
    --------------------
    1. Smallest Cluster Identification:
       - Sample size n_smallest is determined dynamically from the active cluster assignments
         (excluding unassigned outlier patients with cluster_id == -1).
       - Rest-of-Cohort sample size: n_rest = N_total - n_smallest.
       - Imbalance ratio: ratio = n_rest / n_smallest.

    2. Continuous Variables (Cohen's d):
       - Uses two-sample independent t-test power formula via statsmodels `TTestIndPower`.
       - Solves for minimum detectable Cohen's d at nominal alpha (0.05) and after
         multiplicity / FDR correction across the K clusters (alpha / K).

    3. Categorical Variables (Risk Difference):
       - Uses independent two-proportion z-test power formula via statsmodels `power_proportions_2indep`.
       - Evaluated across two standard clinical baselines:
         * 50% event rate: Maximum variance / conservative benchmark.
         * 20% event rate: Typical clinical event / dropout rate benchmark.
       - Solves for minimum detectable absolute Risk Difference (in percentage points, %)
         at nominal alpha (0.05) and multiplicity / FDR-adjusted alpha (alpha / K).

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with standardized cluster assignments.
    cluster_col : str
        Column name containing cluster identifiers (default: 'cluster_id').
    alpha : float
        Two-sided nominal significance threshold (default: 0.05).
    power : float
        Target statistical power (default: 0.80, i.e. 80% power).
    cluster_config_path : Optional[Union[str, Dict]]
        Path or dictionary for cluster configuration and labels.
    print_summary : bool
        Whether to print formatted diagnostic table to console (default: True).

    Returns:
    --------
    Dict[str, Any]:
        Dictionary with sample size metrics and minimum detectable effect sizes.
    """
    clean_clusters = cluster_df[cluster_df[cluster_col] != -1][cluster_col]
    counts = clean_clusters.value_counts().sort_index()
    if len(counts) < 2:
        if print_summary:
            print("  Warning: Power analysis requires at least 2 distinct clusters.")
        return {}

    k_clusters = len(counts)
    min_cid = counts.idxmin()
    n_smallest = int(counts.min())
    n_total = int(counts.sum())
    n_rest = n_total - n_smallest
    ratio = float(n_rest / n_smallest)

    alpha_nom = float(alpha)
    # Multiplicity adjustment across K cluster comparisons (Bonferroni / conservative FDR bound)
    alpha_adj = float(alpha / k_clusters)

    # 1. Continuous Outcomes (Cohen's d)
    tt_power = TTestIndPower()
    min_d_nom = float(tt_power.solve_power(
        effect_size=None, nobs1=n_smallest, ratio=ratio, alpha=alpha_nom, power=power
    ))
    min_d_adj = float(tt_power.solve_power(
        effect_size=None, nobs1=n_smallest, ratio=ratio, alpha=alpha_adj, power=power
    ))

    def _interpret_d(d_val: float) -> str:
        if d_val < 0.2:
            return "negligible effect"
        elif d_val < 0.5:
            return "small effect"
        elif d_val < 0.8:
            return "medium effect"
        else:
            return "large effect"

    # 2. Categorical Outcomes (Risk Difference, percentage points)
    def _solve_min_rd(prop2: float, alpha_val: float) -> float:
        def obj(diff: float) -> float:
            return float(smprop.power_proportions_2indep(
                diff, prop2=prop2, nobs1=n_smallest, ratio=ratio, alpha=alpha_val, return_results=False
            )) - power
        upper_limit = min(0.999 - prop2, 0.999)
        try:
            return float(brentq(obj, 1e-4, upper_limit))
        except Exception:
            z_alpha = float(norm.ppf(1.0 - alpha_val / 2.0))
            z_beta = float(norm.ppf(power))
            se = float(np.sqrt(prop2 * (1.0 - prop2) * (1.0 / n_smallest + 1.0 / n_rest)))
            return float((z_alpha + z_beta) * se)

    rd_50_nom = _solve_min_rd(0.50, alpha_nom) * 100.0
    rd_50_adj = _solve_min_rd(0.50, alpha_adj) * 100.0
    rd_20_nom = _solve_min_rd(0.20, alpha_nom) * 100.0
    rd_20_adj = _solve_min_rd(0.20, alpha_adj) * 100.0

    # 3. Omnibus Continuous Outcomes (Kruskal-Wallis / One-Way ANOVA, Cohen's f & Eta-squared)
    anova_power = FTestAnovaPower()
    min_f_nom = float(anova_power.solve_power(
        effect_size=None, nobs=n_total, k_groups=k_clusters, alpha=alpha_nom, power=power
    ))
    min_f_adj = float(anova_power.solve_power(
        effect_size=None, nobs=n_total, k_groups=k_clusters, alpha=alpha_adj, power=power
    ))
    eta2_nom = float(min_f_nom ** 2 / (1.0 + min_f_nom ** 2))
    eta2_adj = float(min_f_adj ** 2 / (1.0 + min_f_adj ** 2))

    def _interpret_f(f_val: float) -> str:
        if f_val < 0.10:
            return "negligible effect"
        elif f_val < 0.25:
            return "small effect"
        elif f_val < 0.40:
            return "medium effect"
        else:
            return "large effect"

    # 4. Omnibus Categorical / Binary Outcomes (Pearson Chi-Square / Fisher, Cohen's w & Cramer's V)
    # Degrees of freedom for 2 x K table: df = (2 - 1) * (K - 1) = K - 1. GofChisquarePower df = n_bins - 1
    chisq_power = GofChisquarePower()
    min_w_nom = float(chisq_power.solve_power(
        effect_size=None, nobs=n_total, alpha=alpha_nom, power=power, n_bins=k_clusters
    ))
    min_w_adj = float(chisq_power.solve_power(
        effect_size=None, nobs=n_total, alpha=alpha_adj, power=power, n_bins=k_clusters
    ))
    # In a 2 x K contingency table, min(rows-1, cols-1) = 1, so Cramer's V is mathematically identical to Cohen's w
    min_v_nom = min_w_nom
    min_v_adj = min_w_adj

    def _interpret_w(w_val: float) -> str:
        if w_val < 0.10:
            return "negligible effect"
        elif w_val < 0.30:
            return "small effect"
        elif w_val < 0.50:
            return "medium effect"
        else:
            return "large effect"

    cluster_config = load_cluster_config(cluster_config_path)
    smallest_label = get_cluster_label(min_cid, cluster_config)
    pct_smallest = (n_smallest / n_total) * 100.0
    pct_rest = (n_rest / n_total) * 100.0

    power_results = {
        "n_smallest": n_smallest,
        "smallest_cluster_id": min_cid,
        "smallest_cluster_label": smallest_label,
        "n_rest": n_rest,
        "n_total": n_total,
        "k_clusters": k_clusters,
        "ratio": ratio,
        "target_power": power,
        "alpha_nominal": alpha_nom,
        "alpha_adjusted": alpha_adj,
        "min_cohens_d_nominal": min_d_nom,
        "min_cohens_d_adjusted": min_d_adj,
        "min_risk_diff_50_nominal_pct": rd_50_nom,
        "min_risk_diff_50_adjusted_pct": rd_50_adj,
        "min_risk_diff_20_nominal_pct": rd_20_nom,
        "min_risk_diff_20_adjusted_pct": rd_20_adj,
        "omnibus_df": k_clusters - 1,
        "omnibus_continuous_cohens_f_nominal": min_f_nom,
        "omnibus_continuous_cohens_f_adjusted": min_f_adj,
        "omnibus_continuous_eta2_nominal": eta2_nom,
        "omnibus_continuous_eta2_adjusted": eta2_adj,
        "omnibus_categorical_cohens_w_nominal": min_w_nom,
        "omnibus_categorical_cohens_w_adjusted": min_w_adj,
        "omnibus_categorical_cramers_v_nominal": min_v_nom,
        "omnibus_categorical_cramers_v_adjusted": min_v_adj,
    }

    if print_summary:
        print("\n" + "=" * 80)
        print("STATISTICAL POWER & MINIMUM DETECTABLE EFFECT SIZE (MDES) ANALYSIS")
        print("=" * 80)
        print(f"Cohort Clustered N:        {n_total} patients across {k_clusters} clusters")
        print(f"Smallest Cluster Analyzed: {smallest_label} (n = {n_smallest}, {pct_smallest:.1f}% of cohort)")
        print(f"Rest-of-Cohort Benchmark:  n = {n_rest} patients ({pct_rest:.1f}% of cohort)")
        print(f"Sample Size Ratio (n2/n1): {ratio:.2f}")
        print(f"Target Statistical Power:  {power * 100:.1f}% (beta = {1.0 - power:.2f}, two-sided test)")
        print("-" * 80)
        print("PART 1: PAIRWISE COMPARISONS (Smallest Cluster vs Rest-of-Cohort)")
        print("-" * 80)
        print("A. CONTINUOUS OUTCOMES (Standardized Mean Difference, Cohen's d):")
        print(f"   - Nominal Significance (p < {alpha_nom:.4f}):")
        print(f"       Minimum Detectable Cohen's d:  d >= {min_d_nom:.2f} ({_interpret_d(min_d_nom)})")
        print(f"   - Multiplicity / FDR Adjusted (p < {alpha_adj:.4f}, alpha / {k_clusters}):")
        print(f"       Minimum Detectable Cohen's d:  d >= {min_d_adj:.2f} ({_interpret_d(min_d_adj)})")
        print("-" * 80)
        print("B. CATEGORICAL / BINARY OUTCOMES (Risk Difference, percentage points):")
        print("   - Baseline Event Rate = 50% (Conservative / Maximum Variance Benchmark):")
        print(f"       * Nominal (p < {alpha_nom:.4f}):       Min Risk Difference >= +/-{rd_50_nom:.1f} percentage points")
        print(f"       * Adjusted (p < {alpha_adj:.4f}):      Min Risk Difference >= +/-{rd_50_adj:.1f} percentage points")
        print("   - Baseline Event Rate = 20% (Typical Clinical Event / Dropout Rate Benchmark):")
        print(f"       * Nominal (p < {alpha_nom:.4f}):       Min Risk Difference >= +/-{rd_20_nom:.1f} percentage points")
        print(f"       * Adjusted (p < {alpha_adj:.4f}):      Min Risk Difference >= +/-{rd_20_adj:.1f} percentage points")
        print("-" * 80)
        print(f"PART 2: OMNIBUS MULTI-GROUP COMPARISONS (Across All {k_clusters} Clusters)")
        print("-" * 80)
        print(f"C. CONTINUOUS OUTCOMES (Kruskal-Wallis / One-Way ANOVA, Cohen's f & Eta-Squared):")
        print(f"   - Nominal Significance (p < {alpha_nom:.4f}):")
        print(f"       Minimum Detectable Cohen's f:  f >= {min_f_nom:.2f} ({_interpret_f(min_f_nom)}; eta^2 = {eta2_nom:.3f})")
        print(f"   - Multiplicity / FDR Adjusted (p < {alpha_adj:.4f}, alpha / {k_clusters}):")
        print(f"       Minimum Detectable Cohen's f:  f >= {min_f_adj:.2f} ({_interpret_f(min_f_adj)}; eta^2 = {eta2_adj:.3f})")
        print(f"   * Methodological Note: Parametric ANOVA F-power serves as the standardized")
        print(f"     power benchmark for Kruskal-Wallis. Pitman Asymptotic Relative Efficiency (ARE)")
        print(f"     is at least 0.864 (uniform) and 0.955 (normal), and >= 1.0 for skewed distributions.")
        print("-" * 80)
        print(f"D. CATEGORICAL / BINARY OUTCOMES (Pearson Chi-Square / Fisher, Cohen's w & Cramer's V):")
        print(f"   - Nominal Significance (p < {alpha_nom:.4f}, df = {k_clusters - 1}):")
        print(f"       Minimum Detectable Cohen's w:  w >= {min_w_nom:.2f} (Cramer's V >= {min_v_nom:.2f}; {_interpret_w(min_w_nom)})")
        print(f"   - Multiplicity / FDR Adjusted (p < {alpha_adj:.4f}, alpha / {k_clusters}):")
        print(f"       Minimum Detectable Cohen's w:  w >= {min_w_adj:.2f} (Cramer's V >= {min_v_adj:.2f}; {_interpret_w(min_w_adj)})")
        print(f"   * Methodological Note: For 2 x K contingency tables, Cramer's V = Cohen's w")
        print(f"     because min(rows - 1, cols - 1) = min(1, {k_clusters - 1}) = 1.")
        print("=" * 80 + "\n")

    return power_results


# =============================================================================
# STATISTICAL ANALYSIS & TABLE CONSTRUCTION (CLUSTER VS. POPULATION MEAN)
# =============================================================================

def analyze_cluster_vs_population(
    cluster_df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    output_db_path: Optional[str] = None,
    output_table_name: Optional[str] = None,
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    variable_types: Optional[Dict[str, str]] = None,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    save_to_db: bool = True,) -> pd.DataFrame:
    """
    Performs comprehensive statistical comparisons of each cluster against the entire cohort benchmark,
    as well as non-parametric omnibus multi-group comparisons across all clusters.

    Generates three tables:
    1. Publication-Ready Table ({output_table_name}): Formats significant cluster divergences
       with asterisks (* p < 0.05 raw, ** p < 0.05 FDR-corrected) and suppresses raw p-value columns.
    2. Detailed Statistical Table ({output_table_name}_detailed): Contains population Mean/N,
       cluster-specific Mean/N, raw p-values, and FDR-corrected p-values comparing each cluster to population.
    3. Omnibus Comparison Table ({output_table_name}_omnibus): Performs non-parametric omnibus tests
       across all clusters (Kruskal-Wallis for continuous, Chi-squared with Fisher's exact fallback
       for categorical), reporting omnibus p-values, FDR-adjusted p-values, and test names.

    All three tables are persisted to SQLite when `save_to_db=True`.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with merged cluster assignments.
    variables : Optional[List[str]]
        List of clinical variables to profile (defaults to COMPARISON_VARIABLES).
    output_db_path : Optional[str]
        Path to output SQLite database (e.g. '../p1_dbs/pnk_db2_p1_out_2025.sqlite').
    output_table_name : Optional[str]
        Base name for output tables (e.g. 'pam_comparison_wgc07_eb03_k7').
    cluster_col : str
        Column containing cluster identifiers (default: 'cluster_id').
    name_map_path : Optional[Union[str, Dict]]
        Path or dictionary mapping variable names to human-readable labels.
    cluster_config_path : Optional[Union[str, Dict]]
        Path or dictionary mapping cluster IDs to custom titles and colors.
    variable_types : Optional[Dict[str, str]]
        Explicit mapping of variable -> 'continuous' or 'categorical' (inferred if None).
    fdr_correction : bool
        Whether to calculate Benjamini-Hochberg FDR p-values (default: True).
    alpha : float
        Significance threshold for asterisk marking (default: 0.05).
    save_to_db : bool
        Whether to save results to SQLite database (default: True).

    Returns:
    --------
    pd.DataFrame:
        Detailed results DataFrame.
    """
    if variables is None:
        variables = [v for v in COMPARISON_VARIABLES if v in cluster_df.columns]

    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])

    print("\n" + "=" * 80)
    print("STATISTICAL PROFILING: CLUSTERS VS. ENTIRE COHORT BENCHMARK")
    print("=" * 80)
    print(f"Cohort Size (Population N): {len(cluster_df)}")
    print(f"Clusters Analyzed:           {clusters}")
    print(f"Variables Profiled:          {len(variables)}")
    print(f"FDR Correction:              {fdr_correction} (Benjamini-Hochberg)")
    print("=" * 80)

    # Infer variable types if not supplied
    if variable_types is None:
        variable_types = {}
        for var in variables:
            if var in cluster_df.columns:
                variable_types[var] = _infer_variable_type(cluster_df[var], variable_name=var)

    results = []

    # 1. Sample Size (N) Row
    n_row = {
        "Variable": get_nice_name("N", name_map),
        "Whole population: Mean (±SD) / N (%)": f"{len(cluster_df)}",
    }
    for cid in clusters:
        c_label = get_cluster_label(cid, cluster_config)
        c_len = len(cluster_df[cluster_df[cluster_col] == cid])
        pct = (c_len / len(cluster_df)) * 100
        n_row[f"{c_label}: Mean (±SD) / N (%)"] = f"{c_len} ({pct:.1f}%)"
        n_row[f"{c_label}: p-value"] = "N/A"
    results.append(n_row)

    # 2. Iterate across all requested variables
    for var in variables:
        if var not in cluster_df.columns:
            print(f"  Warning: Variable '{var}' not found in cohort dataframe. Skipping.")
            continue

        var_type = variable_types.get(var, "categorical")
        row: Dict[str, Any] = {"Variable": get_nice_name(var, name_map)}

        # Population benchmark formatting
        clean_pop = cluster_df[var].dropna()
        if var_type == "continuous":
            pop_mean = clean_pop.mean()
            pop_sd = clean_pop.std()
            row["Whole population: Mean (±SD) / N (%)"] = f"{pop_mean:.2f} (±{pop_sd:.2f})"
        else:
            pop_pos = clean_pop.sum() if clean_pop.dtype in [bool, int, np.int64] else (clean_pop == 1).sum()
            pop_pct = (pop_pos / len(clean_pop) * 100) if len(clean_pop) > 0 else 0.0
            row["Whole population: Mean (±SD) / N (%)"] = f"{int(pop_pos)} ({pop_pct:.1f}%)"

        # Cluster-specific metrics & statistical testing
        for cid in clusters:
            c_label = get_cluster_label(cid, cluster_config)
            c_subset = cluster_df[cluster_df[cluster_col] == cid][var].dropna()

            if var_type == "continuous":
                c_mean = c_subset.mean() if len(c_subset) > 0 else np.nan
                c_sd = c_subset.std() if len(c_subset) > 0 else np.nan
                row[f"{c_label}: Mean (±SD) / N (%)"] = f"{c_mean:.2f} (±{c_sd:.2f})"
                p_val = mann_whitney_u_test(c_subset, clean_pop)
            else:
                c_pos = c_subset.sum() if c_subset.dtype in [bool, int, np.int64] else (c_subset == 1).sum()
                c_pct = (c_pos / len(c_subset) * 100) if len(c_subset) > 0 else 0.0
                row[f"{c_label}: Mean (±SD) / N (%)"] = f"{int(c_pos)} ({c_pct:.1f}%)"
                p_val = chi_squared_test(c_subset, clean_pop)

            row[f"{c_label}: p-value"] = p_val

        results.append(row)

    results_df = pd.DataFrame(results)

    # 3. Apply Benjamini-Hochberg FDR correction and order columns dynamically
    if fdr_correction:
        p_cols = [c for c in results_df.columns if "p-value" in c and "FDR" not in c]
        new_columns = []
        for col in results_df.columns:
            new_columns.append(col)
            if col in p_cols:
                pvals = pd.to_numeric(results_df[col], errors="coerce")
                valid_mask = pvals.notna() & (results_df[col] != "N/A")
                fdr_col = col.replace("p-value", "p-value (FDR-corrected)")
                results_df[fdr_col] = np.nan
                if valid_mask.sum() > 0:
                    _, corrected, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
                    results_df.loc[valid_mask, fdr_col] = corrected
                new_columns.append(fdr_col)
        results_df = results_df[new_columns]

    # 4. Create Publication Table with Asterisks (* raw p < alpha, ** FDR p < alpha)
    pub_df = results_df.copy()
    data_cols = [c for c in pub_df.columns if ": Mean (±SD) / N (%)" in c and "Whole population" not in c]
    all_p_cols = [c for c in pub_df.columns if "p-value" in c]

    for d_col in data_cols:
        c_prefix = d_col.split(": Mean")[0]
        raw_p_col = f"{c_prefix}: p-value"
        fdr_p_col = f"{c_prefix}: p-value (FDR-corrected)"

        if raw_p_col in pub_df.columns:
            raw_p = pd.to_numeric(pub_df[raw_p_col], errors="coerce")
            fdr_p = (
                pd.to_numeric(pub_df[fdr_p_col], errors="coerce")
                if fdr_p_col in pub_df.columns
                else pd.Series([np.nan] * len(pub_df))
            )

            for idx in pub_df.index:
                val_str = str(pub_df.at[idx, d_col])
                if pd.notna(fdr_p.iloc[idx]) and fdr_p.iloc[idx] < alpha:
                    pub_df.at[idx, d_col] = f"{val_str}**"
                elif pd.notna(raw_p.iloc[idx]) and raw_p.iloc[idx] < alpha:
                    pub_df.at[idx, d_col] = f"{val_str}*"

    pub_df = pub_df.drop(columns=all_p_cols, errors="ignore")

    # 5. Create Omnibus Multi-Group Comparison Table across clusters
    omnibus_df = compare_clusters_multigroup(
        cluster_df=cluster_df,
        variables=variables,
        cluster_col=cluster_col,
        name_map_path=name_map_path,
        cluster_config_path=cluster_config_path,
        variable_types=variable_types,
        fdr_correction=fdr_correction,
        save_to_db=False,
    )

    # 6. Database Persistence if requested (saves 3 table versions: publication, detailed, omnibus)
    if save_to_db and output_db_path and output_table_name:
        out_db = _resolve_path(output_db_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
        detailed_tbl = f"{output_table_name}_detailed"
        omnibus_tbl = f"{output_table_name}_omnibus"
        with sqlite3.connect(out_db) as conn:
            pub_df.to_sql(output_table_name, conn, if_exists="replace", index=False)
            results_df.to_sql(detailed_tbl, conn, if_exists="replace", index=False)
            omnibus_df.to_sql(omnibus_tbl, conn, if_exists="replace", index=False)
        print(f"  ✓ Saved Publication Results Table: '{output_table_name}' in '{out_db}'")
        print(f"  ✓ Saved Detailed Results Table:    '{detailed_tbl}' in '{out_db}'")
        print(f"  ✓ Saved Omnibus Results Table:     '{omnibus_tbl}' in '{out_db}'")

    # Store related tables in DataFrame attributes for in-memory access
    results_df.attrs["publication_df"] = pub_df
    results_df.attrs["omnibus_df"] = omnibus_df

    # Print summary of results
    print("\n" + "=" * 80)
    print("RESULTS PREVIEW (Publication-Ready with Asterisks: * p<0.05, ** p_FDR<0.05)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(pub_df.head(15).to_string(index=False))
    print("=" * 80 + "\n")

    print("=" * 80)
    print("OMNIBUS MULTI-GROUP COMPARISONS PREVIEW (Cohen's f & Cramér's V)")
    print("=" * 80)
    omnibus_preview_cols = [
        c for c in [
            "Variable",
            "Whole population: Mean (±SD) / N (%)",
            "p-value",
            "p-value (FDR-corrected)",
            "Test_Used",
            "Effect_Size",
        ]
        if c in omnibus_df.columns
    ]
    print(omnibus_df[omnibus_preview_cols].head(15).to_string(index=False))
    print("=" * 80 + "\n")

    return results_df


# =============================================================================
# MULTI-GROUP OMNIBUS STATISTICAL COMPARISONS (KRUSKAL-WALLIS & FISHER'S EXACT)
# =============================================================================

def compare_clusters_multigroup(
    cluster_df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    output_db_path: Optional[str] = None,
    output_table_name: Optional[str] = None,
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    variable_types: Optional[Dict[str, str]] = None,
    fdr_correction: bool = True,
    save_to_db: bool = True,) -> pd.DataFrame:
    """
    Performs multi-group omnibus statistical comparisons across all clusters.

    Uses non-parametric methods:
    - Kruskal-Wallis non-parametric ANOVA across all clusters for continuous variables.
    - Pearson Chi-squared contingency test with Fisher's exact test fallback for categorical variables.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with cluster assignments.
    variables : Optional[List[str]]
        List of clinical variables to test (defaults to COMPARISON_VARIABLES).
    output_db_path : Optional[str]
        Path to SQLite database to save results.
    output_table_name : Optional[str]
        Table name for the omnibus comparison table (e.g. '{output_table_name}_omnibus').
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    name_map_path : Optional[Union[str, Dict]]
        Variable name map path or dictionary.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster display configuration path or dictionary.
    variable_types : Optional[Dict[str, str]]
        Mapping of variable -> 'continuous' or 'categorical'.
    fdr_correction : bool
        Whether to calculate Benjamini-Hochberg FDR-corrected p-values (default: True).
    save_to_db : bool
        Whether to save to SQLite database (default: True).

    Returns:
    --------
    pd.DataFrame:
        Omnibus comparison table containing population and cluster descriptives,
        omnibus p-values, FDR-corrected p-values, and non-parametric test names.
    """
    if variables is None:
        variables = [v for v in COMPARISON_VARIABLES if v in cluster_df.columns]

    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    cluster_cols = [f"{get_cluster_label(c, cluster_config)}: Mean (±SD) / N (%)" for c in clusters]
    pop_col = "Whole population: Mean (±SD) / N (%)"

    if variable_types is None:
        variable_types = {}
        for var in variables:
            if var in cluster_df.columns:
                variable_types[var] = _infer_variable_type(cluster_df[var], variable_name=var)

    results = []

    # 1. Sample size row
    size_row = {
        "Variable": get_nice_name("N", name_map),
        pop_col: f"{len(cluster_df)}",
    }
    for cid in clusters:
        c_label = get_cluster_label(cid, cluster_config)
        c_len = len(cluster_df[cluster_df[cluster_col] == cid])
        pct = (c_len / len(cluster_df)) * 100
        size_row[f"{c_label}: Mean (±SD) / N (%)"] = f"{c_len} ({pct:.1f}%)"
    size_row["p-value"] = "N/A"
    if fdr_correction:
        size_row["p-value (FDR-corrected)"] = "N/A"
    size_row["Test_Used"] = "N/A"
    size_row["Effect_Size"] = "N/A"
    results.append(size_row)

    # 2. Iterate across variables
    for var in variables:
        if var not in cluster_df.columns:
            continue

        var_type = variable_types.get(var, "categorical")
        row: Dict[str, Any] = {"Variable": get_nice_name(var, name_map)}

        # Whole population benchmark
        clean_pop = cluster_df[var].dropna()
        if var_type == "continuous":
            clean_pop_num = pd.to_numeric(clean_pop, errors="coerce").dropna()
            pop_mean = clean_pop_num.mean() if len(clean_pop_num) > 0 else np.nan
            pop_sd = clean_pop_num.std() if len(clean_pop_num) > 0 else np.nan
            row[pop_col] = f"{pop_mean:.2f} (±{pop_sd:.2f})"
        else:
            pop_pos = clean_pop.sum() if clean_pop.dtype in [bool, int, np.int64] else (clean_pop == 1).sum()
            pop_pct = (pop_pos / len(clean_pop) * 100) if len(clean_pop) > 0 else 0.0
            row[pop_col] = f"{int(pop_pos)} ({pop_pct:.1f}%)"

        # Descriptive values per cluster
        for cid in clusters:
            c_label = get_cluster_label(cid, cluster_config)
            c_series = cluster_df[cluster_df[cluster_col] == cid][var].dropna()
            if var_type == "continuous":
                c_num = pd.to_numeric(c_series, errors="coerce").dropna()
                m = c_num.mean() if len(c_num) > 0 else np.nan
                s = c_num.std() if len(c_num) > 0 else np.nan
                row[f"{c_label}: Mean (±SD) / N (%)"] = f"{m:.2f} (±{s:.2f})"
            else:
                pos = c_series.sum() if c_series.dtype in [bool, int, np.int64] else (c_series == 1).sum()
                pct = (pos / len(c_series) * 100) if len(c_series) > 0 else 0.0
                row[f"{c_label}: Mean (±SD) / N (%)"] = f"{int(pos)} ({pct:.1f}%)"

        # Omnibus test across all groups
        if var_type == "continuous":
            valid_groups = [
                pd.to_numeric(cluster_df[cluster_df[cluster_col] == cid][var], errors="coerce").dropna().values
                for cid in clusters
            ]
            valid_groups = [g for g in valid_groups if len(g) > 0]
        else:
            valid_groups = [cluster_df[cluster_df[cluster_col] == cid][var].dropna() for cid in clusters]
            valid_groups = [g for g in valid_groups if len(g) > 0]

        if len(valid_groups) < 2:
            row["p-value"] = np.nan
            row["Test_Used"] = "insufficient_data"
            row["Effect_Size"] = "N/A"
            results.append(row)
            continue

        if var_type == "continuous":
            try:
                all_vals = np.concatenate(valid_groups)
                if np.all(all_vals == all_vals[0]):
                    p_val, test_used, effect_size_str = 1.0, "no_variation", "f = 0.00"
                else:
                    _, p_val = kruskal(*valid_groups)
                    test_used = "kruskal_wallis"
                    # Cohen's f calculation from one-way variance dispersion:
                    # f = sqrt(eta^2 / (1 - eta^2)) where eta^2 = SS_between / SS_total
                    gm = np.mean(all_vals)
                    ss_total = np.sum((all_vals - gm) ** 2)
                    ss_between = np.sum([len(g) * (np.mean(g) - gm) ** 2 for g in valid_groups])
                    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
                    cohen_f = float(np.sqrt(eta2 / (1.0 - eta2))) if (eta2 < 1.0 and ss_total > 0) else 0.0
                    effect_size_str = f"f = {cohen_f:.2f}"
            except Exception:
                p_val, test_used, effect_size_str = np.nan, "error", "N/A"
        else:
            try:
                contingency = []
                for g_series in valid_groups:
                    pos = int(g_series.sum() if g_series.dtype in [bool, int, np.int64] else (g_series == 1).sum())
                    neg = len(g_series) - pos
                    contingency.append([pos, neg])
                cont_mat = np.array(contingency)
                if np.any(cont_mat.sum(axis=0) == 0):
                    p_val, test_used, effect_size_str = 1.0, "no_variation", "V = 0.00"
                elif cont_mat.shape == (2, 2):
                    try:
                        chi2, p_val, dof, exp = chi2_contingency(cont_mat)
                        N_mat = cont_mat.sum()
                        min_dim = min(cont_mat.shape[0] - 1, cont_mat.shape[1] - 1)
                        cramers_v = float(np.sqrt(chi2 / (N_mat * min_dim))) if N_mat > 0 and min_dim > 0 else 0.0
                        effect_size_str = f"V = {cramers_v:.2f}"
                        if exp.min() < 5:
                            _, p_val = fisher_exact(cont_mat)
                            test_used = "fisher_exact"
                        else:
                            test_used = "chi_squared"
                    except Exception:
                        _, p_val = fisher_exact(cont_mat)
                        test_used = "fisher_exact"
                        effect_size_str = "N/A"
                else:
                    chi2, p_val, dof, exp = chi2_contingency(cont_mat)
                    test_used = "chi_squared"
                    N_mat = cont_mat.sum()
                    min_dim = min(cont_mat.shape[0] - 1, cont_mat.shape[1] - 1)
                    cramers_v = float(np.sqrt(chi2 / (N_mat * min_dim))) if N_mat > 0 and min_dim > 0 else 0.0
                    effect_size_str = f"V = {cramers_v:.2f}"
            except Exception:
                p_val, test_used, effect_size_str = np.nan, "error", "N/A"

        row["p-value"] = p_val
        row["Test_Used"] = test_used
        row["Effect_Size"] = effect_size_str
        results.append(row)

    omnibus_df = pd.DataFrame(results)

    # 3. Apply Benjamini-Hochberg FDR correction across tested variables
    if fdr_correction and "p-value" in omnibus_df.columns:
        pvals = pd.to_numeric(omnibus_df["p-value"], errors="coerce")
        valid_mask = pvals.notna()
        omnibus_df["p-value (FDR-corrected)"] = "N/A"
        if valid_mask.sum() > 0:
            _, corrected, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
            omnibus_df.loc[valid_mask, "p-value (FDR-corrected)"] = corrected

    end_cols = ["p-value"]
    if fdr_correction and "p-value (FDR-corrected)" in omnibus_df.columns:
        end_cols.append("p-value (FDR-corrected)")
    end_cols.append("Test_Used")
    end_cols.append("Effect_Size")

    col_order = ["Variable", pop_col] + cluster_cols + end_cols
    omnibus_df = omnibus_df.reindex(columns=col_order)

    # 4. Database Persistence if requested
    if save_to_db and output_db_path and output_table_name:
        out_db = _resolve_path(output_db_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
        with sqlite3.connect(out_db) as conn:
            omnibus_df.to_sql(output_table_name, conn, if_exists="replace", index=False)
        print(f"  ✓ Saved Omnibus Multigroup Comparison Table: '{output_table_name}' in '{out_db}'")

    return omnibus_df


# =============================================================================
# VISUALIZATION 1: CONTINUOUS DISTRIBUTIONS (VIOLIN PLOTS)
# =============================================================================

def cluster_continuous_distributions(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    calculate_significance: bool = True,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    variable_configs: Optional[Dict[str, Dict[str, Any]]] = None,) -> None:
    """
    Generates horizontal/split violin plots for continuous variables comparing each cluster to the entire cohort.
    """
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)

    print(f"\n[Violin Plots] Generating violin distributions for {len(variables)} variables in: '{output_dir}'")

    for var in variables:
        if var not in cluster_df.columns or cluster_df[var].dropna().empty:
            continue

        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_cluster_pvalues(cluster_df, var, cluster_col, is_categorical=False)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)

        var_cfg = (variable_configs or {}).get(var, {})
        v_title = var_cfg.get("title", title)
        v_ylabel = var_cfg.get("ylabel", ylabel)
        v_xlabel = var_cfg.get("xlabel", xlabel)

        _plot_single_violin(
            cluster_df=cluster_df,
            variable=var,
            cluster_col=cluster_col,
            name_map=name_map,
            cluster_config=cluster_config,
            output_dir=output_dir,
            sig_raw=sig_raw,
            sig_fdr=sig_fdr,
            alpha=alpha,
            title=v_title,
            ylabel=v_ylabel,
            xlabel=v_xlabel,
        )


def _plot_single_violin(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str,
    name_map: Dict[str, str],
    cluster_config: Dict[str, Any],
    output_dir: str,
    sig_raw: Optional[Dict[int, float]],
    sig_fdr: Optional[Dict[int, float]],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,) -> None:
    """Internal renderer for a single multi-cluster split violin figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    n_clusters = len(clusters)

    pop_data = cluster_df[variable].dropna()
    pop_median = pop_data.median()
    pop_n = len(pop_data)

    plot_rows = []
    stats_rows = []

    for cid in clusters:
        c_label = get_cluster_label(cid, cluster_config)
        c_color = get_cluster_color(cid, cluster_config, DEFAULT_PALETTE)
        c_data = cluster_df[cluster_df[cluster_col] == cid][variable].dropna()

        for val in pop_data:
            plot_rows.append({"value": val, "cluster": c_label, "type": "Population"})
        for val in c_data:
            plot_rows.append({"value": val, "cluster": c_label, "type": "Cluster"})

        stats_rows.append({
            "cluster_id": cid,
            "cluster_label": c_label,
            "cluster_color": c_color,
            "cluster_median": c_data.median() if len(c_data) > 0 else np.nan,
            "cluster_n": len(c_data),
            "p_raw": sig_raw.get(cid) if sig_raw else None,
            "p_fdr": sig_fdr.get(cid) if sig_fdr else None,
        })

    plot_df = pd.DataFrame(plot_rows)
    stats_df = pd.DataFrame(stats_rows)

    fig = plt.figure(figsize=(max(14, n_clusters * 2.2), 10))
    ax = fig.add_axes([0.1, 0.22, 0.72, 0.65])

    cluster_labels_ordered = [get_cluster_label(cid, cluster_config) for cid in clusters]

    sns.violinplot(
        data=plot_df,
        x="cluster",
        y="value",
        hue="type",
        split=True,
        inner="quart",
        order=cluster_labels_ordered,
        palette={"Population": POPULATION_COLOR, "Cluster": CLUSTER_COLOR},
        ax=ax,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=True,
        fontsize=13,
        edgecolor="gray",
    )

    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f"Distribution of {nice_name} by Cluster"
    plot_ylabel = ylabel.format(variable=nice_name) if ylabel else nice_name

    ax.set_ylabel(plot_ylabel, fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(labelsize=12)
    ax.set_xticklabels([])

    fig_w, _ = fig.get_size_inches()

    # Annotate medians and significance below/above violins
    for i, row in stats_df.iterrows():
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig_w

        # Top box: Cluster median
        fig.text(
            x_fig,
            0.88,
            f"Median: {row['cluster_median']:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        # Significance marker
        sig_str = ""
        if row["p_fdr"] is not None and pd.notna(row["p_fdr"]) and row["p_fdr"] < alpha:
            sig_str = "**"
        elif row["p_raw"] is not None and pd.notna(row["p_raw"]) and row["p_raw"] < alpha:
            sig_str = "*"

        if sig_str:
            fig.text(x_fig, 0.18, sig_str, ha="center", va="center", fontsize=20, weight="bold")

        # P-values
        if row["p_raw"] is not None and pd.notna(row["p_raw"]):
            p_txt = f"p={row['p_raw']:.3f}"
            if row["p_fdr"] is not None and pd.notna(row["p_fdr"]):
                p_txt += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.14, p_txt, ha="center", va="top", fontsize=11, style="italic")

        # Cluster labels at bottom
        fig.text(
            x_fig,
            0.05,
            f"{row['cluster_label']}\n(n={row['cluster_n']})",
            ha="right",
            va="top",
            fontsize=14,
            rotation=45,
        )

    # Reference legend box for population
    pop_box = f"Cohort Benchmark (n={pop_n}):\nMedian = {pop_median:.1f}"
    fig.text(
        0.86,
        0.75,
        pop_box,
        ha="left",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    sig_legend = "Significance:\n*  p < 0.05 (raw)\n** p < 0.05 (FDR)"
    fig.text(
        0.86,
        0.62,
        sig_legend,
        ha="left",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.suptitle(plot_title, fontsize=19, weight="bold", y=0.96)
    out_file = os.path.join(output_dir, f"{variable}_violin.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: '{out_file}'")


# =============================================================================
# VISUALIZATION 2: CATEGORICAL DISTRIBUTIONS (STACKED BAR CHARTS)
# =============================================================================

def cluster_categorical_distributions(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    calculate_significance: bool = True,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_labels: Optional[Dict[str, str]] = None,
    variable_configs: Optional[Dict[str, Dict[str, Any]]] = None,) -> None:
    """
    Generates 100% stacked bar charts for binary/categorical variables comparing clusters to the cohort.
    """
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)

    print(f"\n[Stacked Bars] Generating categorical distributions for {len(variables)} variables in: '{output_dir}'")

    for var in variables:
        if var not in cluster_df.columns or cluster_df[var].dropna().empty:
            continue

        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_cluster_pvalues(cluster_df, var, cluster_col, is_categorical=True)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)

        var_cfg = (variable_configs or {}).get(var, {})
        v_title = var_cfg.get("title", title)
        v_ylabel = var_cfg.get("ylabel", ylabel)
        v_xlabel = var_cfg.get("xlabel", xlabel)
        v_legend = var_cfg.get("legend_labels", legend_labels)

        _plot_single_stacked_bar(
            cluster_df=cluster_df,
            variable=var,
            cluster_col=cluster_col,
            name_map=name_map,
            cluster_config=cluster_config,
            output_dir=output_dir,
            sig_raw=sig_raw,
            sig_fdr=sig_fdr,
            alpha=alpha,
            title=v_title,
            ylabel=v_ylabel,
            xlabel=v_xlabel,
            legend_labels=v_legend,
        )


def _plot_single_stacked_bar(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str,
    name_map: Dict[str, str],
    cluster_config: Dict[str, Any],
    output_dir: str,
    sig_raw: Optional[Dict[int, float]],
    sig_fdr: Optional[Dict[int, float]],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_labels: Optional[Dict[str, str]] = None,) -> None:
    """Internal renderer for a single stacked bar figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])

    # Population benchmark row
    clean_pop = cluster_df[variable].dropna()
    pop_pos = clean_pop.sum() if clean_pop.dtype in [bool, int, np.int64] else (clean_pop == 1).sum()
    pop_pct_1 = (pop_pos / len(clean_pop) * 100) if len(clean_pop) > 0 else 0.0
    pop_pct_0 = 100.0 - pop_pct_1

    rows = [{
        "group": "Whole population",
        "prop_0": pop_pct_0,
        "prop_1": pop_pct_1,
        "n_0": int(len(clean_pop) - pop_pos),
        "n_1": int(pop_pos),
        "n_total": len(clean_pop),
        "group_id": None,
        "p_raw": None,
        "p_fdr": None,
    }]

    for cid in clusters:
        c_sub = cluster_df[cluster_df[cluster_col] == cid][variable].dropna()
        c_pos = c_sub.sum() if c_sub.dtype in [bool, int, np.int64] else (c_sub == 1).sum()
        c_pct_1 = (c_pos / len(c_sub) * 100) if len(c_sub) > 0 else 0.0
        c_pct_0 = 100.0 - c_pct_1

        rows.append({
            "group": get_cluster_label(cid, cluster_config),
            "prop_0": c_pct_0,
            "prop_1": c_pct_1,
            "n_0": int(len(c_sub) - c_pos),
            "n_1": int(c_pos),
            "n_total": len(c_sub),
            "group_id": cid,
            "p_raw": sig_raw.get(cid) if sig_raw else None,
            "p_fdr": sig_fdr.get(cid) if sig_fdr else None,
        })

    plot_df = pd.DataFrame(rows)
    fig = plt.figure(figsize=(max(14, len(plot_df) * 2.0), 10))
    ax = fig.add_axes([0.1, 0.25, 0.72, 0.60])

    x_pos = np.arange(len(plot_df))
    lbl_1 = (legend_labels or {}).get("achieved", "Present / Achieved")
    lbl_0 = (legend_labels or {}).get("not_achieved", "Absent / Not Achieved")

    ax.bar(x_pos, plot_df["prop_1"], label=lbl_1, color=ACHIEVED_COLOR, alpha=0.85)
    ax.bar(x_pos, plot_df["prop_0"], bottom=plot_df["prop_1"], label=lbl_0, color=NOT_ACHIEVED_COLOR, alpha=0.85)

    # Population dashed line across plot
    ax.axhline(y=pop_pct_1, color="black", linestyle="--", linewidth=2.0, alpha=0.8)

    # Asterisks inside plot area above the dashed line
    for i, row in plot_df.iterrows():
        if row["group_id"] is not None:
            sig_str = ""
            if row["p_fdr"] is not None and pd.notna(row["p_fdr"]) and row["p_fdr"] < alpha:
                sig_str = "**"
            elif row["p_raw"] is not None and pd.notna(row["p_raw"]) and row["p_raw"] < alpha:
                sig_str = "*"
            if sig_str:
                ax.text(i, pop_pct_1 + 1.5, sig_str, ha="center", va="bottom", fontsize=20, weight="bold")

    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])

    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f"Prevalence of {nice_name} by Cluster"
    plot_ylabel = ylabel if ylabel else "Percentage (%)"
    ax.set_ylabel(plot_ylabel, fontsize=16)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=True,
        fontsize=13,
        edgecolor="gray",
    )

    fig_w, _ = fig.get_size_inches()

    # Annotation boxes above and below the bars
    for i, row in plot_df.iterrows():
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig_w

        # Top box: absent (class 0)
        fig.text(
            x_fig,
            0.88,
            f"{row['n_0']} ({row['prop_0']:.1f}%)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=NOT_ACHIEVED_COLOR, alpha=0.7, edgecolor="gray"),
        )
        # Bottom box: present (class 1)
        fig.text(
            x_fig,
            0.22,
            f"{row['n_1']} ({row['prop_1']:.1f}%)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=ACHIEVED_COLOR, alpha=0.7, edgecolor="gray"),
        )

        # P-value display
        if row["group_id"] is not None and row["p_raw"] is not None and pd.notna(row["p_raw"]):
            p_txt = f"p={row['p_raw']:.3f}"
            if row["p_fdr"] is not None and pd.notna(row["p_fdr"]):
                p_txt += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.17, p_txt, ha="center", va="top", fontsize=11, style="italic")

        # Cluster label
        fig.text(
            x_fig,
            0.08,
            f"{row['group']}\n(n={row['n_total']})",
            ha="right",
            va="top",
            fontsize=14,
            rotation=45,
        )

    # Reference box
    fig.text(
        0.86,
        0.72,
        f"- - - - Cohort Average\n        ({pop_pct_1:.1f}%)",
        ha="left",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    fig.text(
        0.86,
        0.60,
        "Significance:\n*  p < 0.05 (raw)\n** p < 0.05 (FDR)",
        ha="left",
        va="center",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.suptitle(plot_title, fontsize=19, weight="bold", y=0.96)
    out_file = os.path.join(output_dir, f"{variable}_stacked_bar.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: '{out_file}'")


# =============================================================================
# VISUALIZATION 3: MULTI-VARIABLE LOLLIPOP PLOT (SMD & RISK DIFFERENCE VS REST)
# =============================================================================

def plot_cluster_lollipop(
    cluster_df: pd.DataFrame,
    continuous_variables: Optional[List[str]] = None,
    categorical_variables: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    output_filename: str = "lollipop_plot.png",
    output_dir: str = ".",
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    rd_as_percent: bool = True,
    alpha: float = 0.05,
    show_ci: bool = True,
    show_stick: bool = False,
    title: Optional[str] = None,
    xlabel_cont: Optional[str] = None,
    xlabel_cat: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    pvalues_raw: Optional[Dict[str, Dict[int, float]]] = None,
    pvalues_fdr: Optional[Dict[str, Dict[int, float]]] = None,
    results_df: Optional[pd.DataFrame] = None,
    use_absolute_magnitude: Optional[bool] = None,
    effects_cont: Optional[pd.DataFrame] = None,
    effects_cat: Optional[pd.DataFrame] = None,) -> None:
    """
    Generates a unified publication-ready multi-variable lollipop/forest plot comparing each cluster
    to the remaining cohort (Rest-of-Cohort).

    Visual & Methodological Architecture:
    -------------------------------------
    1. Top Section (Continuous Variables):
       - Displays Standardized Mean Differences (Cohen's d) vs. Rest-of-Cohort.
       - Formula: d = (mean_cluster - mean_rest) / s_pooled.
       - Whiskers display 95% Confidence Intervals for Cohen's d (Hedges & Olkin 1985).
       - A vertical solid black line at x=0 represents parity.
       - Positive values indicate scores above the rest of the cohort; negative below.
       - Statistical significance marked via asterisks (* raw p < alpha, ** FDR q < alpha)
         computed from two-sample Mann-Whitney U tests vs. Rest-of-Cohort.

    2. Bottom Section (Binary/Categorical Characteristics):
       - Displays Risk Differences (RD = prop_cluster - prop_rest) vs. Rest-of-Cohort.
       - Plotted in percentage points (%) when rd_as_percent=True.
       - Whiskers display 95% Confidence Intervals for Risk Difference (Wald interval).
       - A vertical solid black line at x=0 represents parity.
       - Statistical significance marked via asterisks (* raw p < alpha, ** FDR q < alpha)
         computed from Chi-squared / Fisher's exact tests vs. Rest-of-Cohort.

    3. Clean Separation of Concerns:
       - Pure visualization module. Statistical estimations vs Rest-of-Cohort are computed
         via `calculate_rest_of_cohort_effects` (or accepted via effects_cont / effects_cat).
       - Statistical power and MDES analyses are strictly separated into `calculate_cluster_power_analysis`.

    Parameters:
    -----------
    cluster_df : pd.DataFrame
        Cohort dataframe with standardized cluster assignments.
    continuous_variables : Optional[List[str]]
        Continuous variables to plot as SMDs (defaults to age, baseline weight, max and total weight loss).
    categorical_variables : Optional[List[str]]
        Binary variables to plot as Risk Differences (defaults to sex, 10% achievement, 60d dropout).
    variables : Optional[List[str]]
        Legacy list of variables (partitioned into continuous and categorical if provided).
    output_filename : str
        Name of output image file (default: 'lollipop_plot.png').
    output_dir : str
        Directory to save figure.
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    name_map_path : Optional[Union[str, Dict]]
        Path or dictionary mapping raw column names to display labels.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster display configuration.
    rd_as_percent : bool
        Whether to report Risk Difference in percentage points (default: True).
    alpha : float
        Significance threshold for asterisks (default: 0.05).
    show_ci : bool
        Whether to render 95% Confidence Interval whiskers (default: True).
    show_stick : bool
        Whether to draw horizontal stems from 0 to point estimate (default: False for clean forest plot).
    title : Optional[str]
        Overall figure title.
    xlabel_cont : Optional[str]
        X-axis label for continuous SMD panel.
    xlabel_cat : Optional[str]
        X-axis label for categorical RD panel.
    ylabel : Optional[str]
        Optional custom Y-axis label (for legacy compatibility).
    xlabel : Optional[str]
        Optional general X-axis label (for legacy single-panel compatibility).
    pvalues_raw : Optional[Dict]
        Optional precomputed raw p-values.
    pvalues_fdr : Optional[Dict]
        Optional precomputed FDR-adjusted p-values.
    results_df : Optional[pd.DataFrame]
        Optional results dataframe for backwards compatibility.
    use_absolute_magnitude : Optional[bool]
        Whether weight loss variables use absolute magnitude (default: True).
    effects_cont : Optional[pd.DataFrame]
        Precomputed continuous effect sizes from `calculate_rest_of_cohort_effects`.
    effects_cat : Optional[pd.DataFrame]
        Precomputed categorical effect sizes from `calculate_rest_of_cohort_effects`.
    """
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)

    # 1. Resolve variables
    if continuous_variables is None and categorical_variables is None:
        if variables is not None:
            # Legacy call: partition supplied variables
            continuous_variables = [
                    "age", 
                    "baseline_weight_kg", 
                    "baseline_bmi", 
                    "total_followup_days", 
                    # "days_to_nadir",
                    "max_wl_%",
                    "weight_regain_%", 
                    "total_wl_%",
                    # "60d_wl_%"
            ]
            categorical_variables = [
                    "sex_f", 
                    "10%_wl_achieved", 
                    "60d_dropout"
            ]
            for v in variables:
                if v in cluster_df.columns:
                    vtype = _infer_variable_type(cluster_df[v], variable_name=v)
                    if vtype == "categorical":
                        categorical_variables.append(v)
                    else:
                        continuous_variables.append(v)
        else:
            # Default clinical reporting set requested by researcher
            continuous_variables = [
                v for v in [
                    "age", 
                    "baseline_weight_kg", 
                    "baseline_bmi", 
                    "total_followup_days", 
                    # "days_to_nadir",
                    "max_wl_%",
                    "weight_regain_%", 
                    "total_wl_%",
                    # "60d_wl_%"
                    ]
                if v in cluster_df.columns
            ]
            categorical_variables = [
                v for v in [
                    "sex_f", 
                    "10%_wl_achieved", 
                    "60d_dropout"
                    ]
                if v in cluster_df.columns
            ]

    else:
        if continuous_variables is not None:
            continuous_variables = [v for v in continuous_variables if v in cluster_df.columns]
        else:
            continuous_variables = []
        if categorical_variables is not None:
            categorical_variables = [v for v in categorical_variables if v in cluster_df.columns]
        else:
            categorical_variables = []

    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    if not clusters:
        print("  Warning: No valid clusters found in cohort dataframe.")
        return

    has_cont = len(continuous_variables) > 0
    has_cat = len(categorical_variables) > 0

    if not has_cont and not has_cat:
        print("  Warning: No continuous or categorical variables available for lollipop plot.")
        return

    print(f"\n[Forest/Lollipop Plot] Generating SMD & Risk Difference plot vs Rest-of-Cohort...")
    print(f"  - Continuous Variables ({len(continuous_variables)}): {continuous_variables}")
    print(f"  - Categorical Variables ({len(categorical_variables)}): {categorical_variables}")

    # 2. Extract or compute effect sizes and 95% CIs vs. Rest-of-Cohort
    if effects_cont is not None or effects_cat is not None:
        df_cont = effects_cont if effects_cont is not None else pd.DataFrame()
        df_cat = effects_cat if effects_cat is not None else pd.DataFrame()
    else:
        df_cont, df_cat = calculate_rest_of_cohort_effects(
            cluster_df=cluster_df,
            continuous_variables=continuous_variables if has_cont else None,
            categorical_variables=categorical_variables if has_cat else None,
            cluster_col=cluster_col,
            alpha=alpha,
            rd_as_percent=rd_as_percent,
            use_absolute_magnitude=use_absolute_magnitude,
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    # Helper function to render a single lollipop panel
    def _render_panel(
        ax: plt.Axes,
        panel_df: pd.DataFrame,
        vars_list: List[str],
        panel_title: str,
        panel_xlabel: str,
        is_percent: bool = False,) -> float:
        all_vals = [panel_df["value"].dropna().values]
        if show_ci and "ci_lower" in panel_df.columns:
            all_vals.append(panel_df["ci_lower"].dropna().values)
        if show_ci and "ci_upper" in panel_df.columns:
            all_vals.append(panel_df["ci_upper"].dropna().values)
        flat_vals = np.concatenate(all_vals) if all_vals else np.array([1.0])
        max_abs = float(np.nanmax(np.abs(flat_vals))) if len(flat_vals) > 0 and not np.all(np.isnan(flat_vals)) else 1.0

        if is_percent:
            xlim_val = float(np.ceil(max(max_abs, 10.0) * 1.25 / 5.0) * 5.0)
        else:
            xlim_val = float(np.ceil(max(max_abs, 0.5) * 1.25 * 10.0) / 10.0)

        ax.set_xlim(-xlim_val, xlim_val)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, zorder=3)

        y_pos = 0.0
        y_ticks, y_labels = [], []

        for var in vars_list:
            v_sub = panel_df[panel_df["variable"] == var]
            y_range = y_pos + np.arange(len(v_sub), dtype=float)

            for idx, (_, r) in enumerate(v_sub.iterrows()):
                cid = r["cluster_id"]
                val = r["value"]
                color = get_cluster_color(cid, cluster_config, DEFAULT_PALETTE)
                y_coord = y_range[idx]

                # Optional stem representing point estimate deviation from parity (omitted by default for clean forest display)
                if show_stick:
                    ax.hlines(y=y_coord, xmin=0, xmax=val, colors=color, linewidth=2.2, alpha=0.70, zorder=3)

                # 95% Confidence Interval error bar
                if show_ci and pd.notna(r.get("ci_lower")) and pd.notna(r.get("ci_upper")):
                    ci_low = r["ci_lower"]
                    ci_high = r["ci_upper"]
                    err_low = max(0.0, val - ci_low)
                    err_high = max(0.0, ci_high - val)
                    ax.errorbar(
                        x=val,
                        y=y_coord,
                        xerr=[[err_low], [err_high]],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.8,
                        capsize=3.8,
                        capthick=1.4,
                        alpha=0.90,
                        zorder=4,
                    )

                # Point estimate marker head
                ax.scatter(val, y_coord, color=color, s=80, edgecolor="white", linewidth=0.7, zorder=5)

                sig_str = r.get("sig_str", "")
                if sig_str:
                    offset = 0.02 * (2 * xlim_val)
                    if show_ci and pd.notna(r.get("ci_upper")) and pd.notna(r.get("ci_lower")):
                        x_txt = max(val, r["ci_upper"]) + offset if val >= 0 else min(val, r["ci_lower"]) - offset
                    else:
                        x_txt = val + offset if val >= 0 else val - offset
                    align = "left" if val >= 0 else "right"
                    ax.text(x_txt, y_coord, sig_str, ha=align, va="center", fontsize=14, weight="bold", zorder=6)

            y_ticks.append(float(np.mean(y_range)))
            # Clean display label: remove redundant '(y/n)'
            raw_label = get_nice_name(var, name_map)
            cleaned_label = raw_label.replace(" (y/n)", "").replace(" (1/0)", "")
            y_labels.append(cleaned_label)

            y_pos += len(v_sub) + 0.8
            if var != vars_list[-1]:
                ax.axhline(y=y_pos - 0.4, color="#E0E0E0", linestyle="--", linewidth=0.8)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=13)
        ax.invert_yaxis()
        ax.set_title(panel_title, fontsize=13, weight="bold", loc="left", pad=10)
        ax.set_xlabel(panel_xlabel, fontsize=12, labelpad=8)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        return xlim_val

    # 4. Render Layout
    pct_label = "percentage points" if rd_as_percent else "proportion"
    def_xlabel_cont = (
        "Standardized mean difference (Cohen's d with 95% CI, Parity = 0)"
        if show_ci
        else "Standardized mean difference (Cohen's d, Parity = 0)"
    )
    def_xlabel_cat = (
        f"Risk difference ({pct_label} with 95% CI, Parity = 0)"
        if show_ci
        else f"Risk difference ({pct_label}, Parity = 0)"
    )

    if has_cont and has_cat:
        h_top = max(4.0, len(continuous_variables) * len(clusters) * 0.28 + 1.2)
        h_bottom = max(3.5, len(categorical_variables) * len(clusters) * 0.28 + 1.2)
        fig, (ax_top, ax_bottom) = plt.subplots(
            nrows=2, ncols=1, figsize=(14, h_top + h_bottom),
            gridspec_kw={"height_ratios": [h_top, h_bottom], "hspace": 0.22}
        )

        _render_panel(
            ax=ax_top,
            panel_df=df_cont,
            vars_list=continuous_variables,
            panel_title="Continuous variables: cluster vs. rest-of-cohort standardized mean differences",
            panel_xlabel=xlabel_cont if xlabel_cont else def_xlabel_cont,
            is_percent=False,
        )

        _render_panel(
            ax=ax_bottom,
            panel_df=df_cat,
            vars_list=categorical_variables,
            panel_title="Binary variables: cluster vs. rest-of-cohort risk differences",
            panel_xlabel=xlabel_cat if xlabel_cat else def_xlabel_cat,
            is_percent=rd_as_percent,
        )

        legend_ax = ax_top
        fig_title = title if title else "Cluster-wise deviations from the rest of the cohort"
        fig.suptitle(fig_title, fontsize=16, weight="bold", y=0.995)

    elif has_cont:
        h_top = max(4.5, len(continuous_variables) * len(clusters) * 0.32 + 1.5)
        fig, ax = plt.subplots(figsize=(14, h_top))
        _render_panel(
            ax=ax,
            panel_df=df_cont,
            vars_list=continuous_variables,
            panel_title="Continuous variables: cluster vs. rest-of-cohort standardized mean differences",
            panel_xlabel=xlabel_cont or xlabel or def_xlabel_cont,
            is_percent=False,
        )
        legend_ax = ax
        fig_title = title if title else "Cluster-wise standardized mean differences from the rest of the cohort",
        fig.suptitle(fig_title, fontsize=16, weight="bold", y=0.995)

    else:
        h_bottom = max(4.5, len(categorical_variables) * len(clusters) * 0.32 + 1.5)
        fig, ax = plt.subplots(figsize=(14, h_bottom))
        _render_panel(
            ax=ax,
            panel_df=df_cat,
            vars_list=categorical_variables,
            panel_title="Binary variables: cluster vs. rest-of-cohort risk differences",
            panel_xlabel=xlabel_cat or xlabel or def_xlabel_cat,
            is_percent=rd_as_percent,
        )
        legend_ax = ax
        fig_title = title if title else "Cluster-wise risk differences from the rest of the cohort"
        fig.suptitle(fig_title, fontsize=16, weight="bold", y=0.995)

    # 5. Unified Cluster Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=get_cluster_color(cid, cluster_config, DEFAULT_PALETTE),
                   label=get_cluster_label(cid, cluster_config), markersize=10)
        for cid in clusters
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Clusters",
        fontsize=12,
        title_fontsize=13,
        frameon=True,
        fancybox=True,
        edgecolor="gray",
    )

    info_note = (
        "\n*  p < 0.05 (raw)\n** p < 0.05 (FDR)"
    )
    legend_ax.text(
        1.02,
        0.35 if (has_cont and has_cat) else 0.25,
        info_note,
        transform=legend_ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", alpha=0.9, edgecolor="#CCCCCC"),
    )

    out_path = os.path.normpath(os.path.join(output_dir, output_filename))
    try:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    except OSError:
        # Fallback for Windows file handles: save via temporary file and atomic replace
        import tempfile
        import time
        temp_dir = os.path.dirname(out_path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plt.savefig(tmp_path, dpi=300, bbox_inches="tight")
        for attempt in range(5):
            try:
                os.replace(tmp_path, out_path)
                break
            except OSError:
                time.sleep(0.5)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    plt.close()
    print(f"  ✓ Saved: '{out_path}'")


# Modern semantic alias: plot_cluster_forest_vs_rest
plot_cluster_forest_vs_rest = plot_cluster_lollipop


# =============================================================================
# VISUALIZATION 4: PREVALENCE HEATMAP WITH CLUSTER COLOR SCALES
# =============================================================================

def plot_cluster_heatmap(
    results_df: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    variables: List[str],
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,) -> None:
    """
    Generates a multi-column publication heatmap of attribute prevalence and 1-10 Likert scores across clusters,
    featuring a population reference column and cluster-specific gradient color scales.
    
    Color intensity / heat:
    - Percentage variables (0-100%): Saturation proportional to reported percentage (vmin=0, vmax=100).
    - 1-10 Likert scores: Scaled to 0-100% heat intensity (score * 10), where 10 has maximum heat (100%)
      and 1 has lowest heat (~10%, weakly colored tint).
    """
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)

    print(f"\n[Heatmap] Generating prevalence & Likert heatmap for {len(variables)} variables...")

    prevalence_data = []

    cluster_cols = [
        c for c in results_df.columns
        if (": Mean" in c or ": Mean/N" in c) and "Whole population" not in c
    ]
    # Map column headers to canonical cluster label (e.g. 'Cluster 1', 'Cluster 2', ...)
    cluster_label_map: Dict[str, str] = {}
    for col in cluster_cols:
        c_label = col.split(":")[0].strip()
        cluster_label_map[col] = c_label

    for _, row in results_df.iterrows():
        v_display = row["Variable"]
        if v_display in ["N", "N (sample size)"]:
            continue

        raw_var = None
        for cand in variables:
            if v_display == get_nice_name(cand, name_map):
                raw_var = cand
                break
        if raw_var is None:
            continue

        # Population benchmark column
        pop_str = str(row.get("Whole population: Mean (±SD) / N (%)", "")).strip()
        if "%" in pop_str:
            pop_pct = _extract_percentage_from_table_cell(pop_str)
            try:
                pop_n = int(float(pop_str.split("(")[0].strip()))
            except Exception:
                pop_n = 0
            pop_heat = pop_pct if pd.notna(pop_pct) else 0.0
            pop_txt = f"{pop_n} ({pop_pct:.1f}%)" if pd.notna(pop_pct) else ""
        else:
            pop_mean, pop_sd = _extract_mean_sd_from_table_cell(pop_str)
            # Map 1-10 Likert score to 0-100 heat intensity: 10 has highest (100%), 1 has lowest (10%)
            pop_heat = np.clip(pop_mean * 10.0, 0.0, 100.0) if pop_mean is not None else 0.0
            if pop_sd is not None:
                pop_txt = f"{pop_mean:.1f} (±{pop_sd:.1f})"
            elif pop_mean is not None:
                pop_txt = f"{pop_mean:.1f}"
            else:
                pop_txt = ""

        prevalence_data.append({
            "variable": raw_var,
            "cluster_label": "Whole population",
            "heat": pop_heat,
            "text": pop_txt,
        })

        # Cluster columns
        for col, c_label in cluster_label_map.items():
            c_str = str(row.get(col, "")).strip()
            if "%" in c_str:
                pct = _extract_percentage_from_table_cell(c_str)
                try:
                    cnt = int(float(c_str.split("(")[0].strip()))
                except Exception:
                    cnt = 0
                c_heat = pct if pd.notna(pct) else 0.0
                c_base_txt = f"{cnt} ({pct:.1f}%)" if pd.notna(pct) else ""
            else:
                c_mean, c_sd = _extract_mean_sd_from_table_cell(c_str)
                # Map 1-10 Likert score to 0-100 heat intensity: 10 has highest (100%), 1 has lowest (10%)
                c_heat = np.clip(c_mean * 10.0, 0.0, 100.0) if c_mean is not None else 0.0
                if c_sd is not None:
                    c_base_txt = f"{c_mean:.1f} (±{c_sd:.1f})"
                elif c_mean is not None:
                    c_base_txt = f"{c_mean:.1f}"
                else:
                    c_base_txt = ""

            # Check significance p-values
            p_r_col = f"{c_label}: p-value"
            p_f_col = f"{c_label}: p-value (FDR-corrected)"
            p_f = None
            p_r = None
            if p_f_col in row and pd.notna(row[p_f_col]):
                try:
                    p_f = float(row[p_f_col])
                except ValueError:
                    pass
            if p_r_col in row and pd.notna(row[p_r_col]):
                try:
                    p_r = float(row[p_r_col])
                except ValueError:
                    pass

            sig_str = ""
            if p_f is not None and p_f < alpha:
                sig_str = "**"
            elif p_r is not None and p_r < alpha:
                sig_str = "*"

            prevalence_data.append({
                "variable": raw_var,
                "cluster_label": c_label,
                "heat": c_heat,
                "text": f"{c_base_txt}{sig_str}" if c_base_txt else "",
            })

    if not prevalence_data:
        print("  Warning: No prevalence or score data available for heatmap.")
        return

    prev_df = pd.DataFrame(prevalence_data)
    heat_matrix = prev_df.pivot(index="variable", columns="cluster_label", values="heat")
    text_matrix = prev_df.pivot(index="variable", columns="cluster_label", values="text")

    ordered_cols = ["Whole population"] + [c for c in cluster_label_map.values() if c in heat_matrix.columns]
    heat_matrix = heat_matrix[ordered_cols].reindex(variables)
    text_matrix = text_matrix[ordered_cols].reindex(variables)

    row_labels = [get_nice_name(v, name_map) for v in variables]
    n_cols = len(ordered_cols)
    n_rows = len(variables)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(12, n_cols * 1.8), max(8, n_rows * 0.65)),
        gridspec_kw={"wspace": 0.02},
    )
    if n_cols == 1:
        axes = [axes]

    for j, (c_label, ax) in enumerate(zip(ordered_cols, axes)):
        col_heat = heat_matrix.iloc[:, j:j + 1].values

        if c_label == "Whole population":
            cmap = LinearSegmentedColormap.from_list("pop", ["white", "#4D4D4D"])
        else:
            color = get_cluster_color(c_label, cluster_config, DEFAULT_PALETTE)
            cmap = LinearSegmentedColormap.from_list(f"cluster_{j}", ["white", color])

        ax.imshow(col_heat, cmap=cmap, vmin=0, vmax=100, aspect="auto")

        for i in range(n_rows):
            cell_txt = text_matrix.iloc[i, j]
            ax.text(0, i, cell_txt if pd.notna(cell_txt) else "", ha="center", va="center", fontsize=11, weight="normal")

        ax.set_xlabel(c_label, fontsize=14, rotation=45, ha="right")
        ax.set_xticks([])
        ax.set_yticks(range(n_rows))
        ax.tick_params(axis="both", length=0)

        if j == 0:
            ax.set_yticklabels(row_labels, fontsize=13)
        else:
            ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
        ax.grid(False)

    plot_title = title if title else "Weight gain cause and eating behavior profiles across clusters"
    fig.suptitle(plot_title, fontsize=18, weight="bold", y=0.98)

    out_path = os.path.normpath(os.path.join(output_dir, output_filename))
    try:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    except OSError:
        # Fallback for Windows file handles: save via temporary file and atomic replace
        import tempfile
        import time
        temp_dir = os.path.dirname(out_path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plt.savefig(tmp_path, dpi=300, bbox_inches="tight")
        for attempt in range(5):
            try:
                os.replace(tmp_path, out_path)
                break
            except OSError:
                time.sleep(0.5)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    plt.close()
    print(f"  ✓ Saved: '{out_path}'")


# =============================================================================
# VISUALIZATION 5: FOREST PLOTS (RISK RATIO & RISK DIFFERENCE)
# =============================================================================

def plot_cluster_forest(
    cluster_df: pd.DataFrame,
    outcome_variable: str,
    output_dir: str,
    cluster_col: str = "cluster_id",
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    effect_type: str = "both",
    title_rr: Optional[str] = None,
    title_rd: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_rr: Optional[str] = None,
    xlabel_rd: Optional[str] = None,) -> None:
    """
    Generates forest plots for Risk Ratios (RR) and/or Risk Differences (RD)
    comparing each cluster against the entire cohort for a binary outcome.
    """
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)

    if outcome_variable not in cluster_df.columns or cluster_df[outcome_variable].dropna().empty:
        print(f"  Warning: Outcome variable '{outcome_variable}' not found in cohort data.")
        return

    print(f"\n[Forest Plots] Calculating risk metrics for '{outcome_variable}'...")
    risk_df = calculate_risk_metrics(cluster_df, outcome_variable, cluster_col)

    if effect_type in ["RR", "both"]:
        _plot_single_forest(
            risk_df=risk_df,
            outcome_variable=outcome_variable,
            plot_type="RR",
            output_dir=output_dir,
            name_map=name_map,
            cluster_config=cluster_config,
            title=title_rr,
            xlabel=xlabel_rr,
        )

    if effect_type in ["RD", "both"]:
        _plot_single_forest(
            risk_df=risk_df,
            outcome_variable=outcome_variable,
            plot_type="RD",
            output_dir=output_dir,
            name_map=name_map,
            cluster_config=cluster_config,
            title=title_rd,
            xlabel=xlabel_rd,
        )


def _plot_single_forest(
    risk_df: pd.DataFrame,
    outcome_variable: str,
    plot_type: str,
    output_dir: str,
    name_map: Dict[str, str],
    cluster_config: Dict[str, Any],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,) -> None:
    """Internal renderer for a single forest plot."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sort_col = "risk_ratio" if plot_type == "RR" else "risk_difference"
    plot_data = risk_df.sort_values(sort_col, ascending=True).reset_index(drop=True)

    fig, (ax, ax2) = plt.subplots(
        1, 2,
        figsize=(14, max(6, len(plot_data) * 0.7)),
        gridspec_kw={"width_ratios": [3.2, 0.9], "wspace": 0.05},
    )

    y_pos = np.arange(len(plot_data))

    if plot_type == "RR":
        effect = plot_data["risk_ratio"]
        ci_lower = plot_data["rr_ci_lower"]
        ci_upper = plot_data["rr_ci_upper"]
        ref_val = 1.0
        default_xlabel = "Risk Ratio (vs. Entire Cohort)"
        default_title = f"Risk Ratios: {get_nice_name(outcome_variable, name_map)}"
    else:
        effect = plot_data["risk_difference"] * 100.0
        ci_lower = plot_data["rd_ci_lower"] * 100.0
        ci_upper = plot_data["rd_ci_upper"] * 100.0
        ref_val = 0.0
        default_xlabel = "Risk Difference (%) (vs. Entire Cohort)"
        default_title = f"Risk Differences: {get_nice_name(outcome_variable, name_map)}"

    colors = [get_cluster_color(cid, cluster_config, DEFAULT_PALETTE) for cid in plot_data["cluster_id"]]

    for i in range(len(plot_data)):
        if pd.notna(effect.iloc[i]) and pd.notna(ci_lower.iloc[i]) and pd.notna(ci_upper.iloc[i]):
            ax.errorbar(
                x=effect.iloc[i],
                y=y_pos[i],
                xerr=[[effect.iloc[i] - ci_lower.iloc[i]], [ci_upper.iloc[i] - effect.iloc[i]]],
                fmt="o",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=2.5,
                capsize=5,
                markersize=9,
            )

    ax.axvline(x=ref_val, color="red", linestyle="--", linewidth=2.0, alpha=0.85)

    labels = [get_cluster_label(cid, cluster_config) for cid in plot_data["cluster_id"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_title(title if title else default_title, fontsize=17, weight="bold", pad=12)
    ax.set_xlabel(xlabel if xlabel else default_xlabel, fontsize=14)

    # Right axis with numeric labels
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.grid(False)

    ci_texts = []
    for i in range(len(plot_data)):
        if plot_type == "RR":
            ci_texts.append(f"RR: {plot_data['risk_ratio'].iloc[i]:.2f} [{plot_data['rr_ci_lower'].iloc[i]:.2f}, {plot_data['rr_ci_upper'].iloc[i]:.2f}]")
        else:
            ci_texts.append(f"RD: {plot_data['risk_difference'].iloc[i]*100:.1f}% [{plot_data['rd_ci_lower'].iloc[i]*100:.1f}%, {plot_data['rd_ci_upper'].iloc[i]*100:.1f}%]")

    ax2.set_yticklabels(ci_texts, fontsize=12, ha="left")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)

    suffix = "rr" if plot_type == "RR" else "rd"
    out_file = os.path.join(output_dir, f"{outcome_variable}_forest_{suffix}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: '{out_file}'")


# =============================================================================
# VISUALIZATION 6: 6-PANEL CLINICAL SUMMARY OVERVIEW FIGURE
# =============================================================================

def plot_cluster_summary_grid(
    cluster_df: pd.DataFrame,
    output_filepath: str,
    cluster_col: str = "cluster_id",
    cluster_config_path: Optional[Union[str, Dict]] = None,) -> None:
    """
    Creates a publication-grade 3x2 clinical summary grid comparing core clinical domains:
    Panel A: Cluster size distribution (Bar chart)
    Panel B: Female sex distribution (Stacked bar chart)
    Panel C: Age distribution (Boxplot)
    Panel D: Baseline body mass index (BMI) (Boxplot)
    Panel E: Total program follow-up length in days (Boxplot)
    Panel F: Total body weight loss (%) (Boxplot)
    """
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted([c for c in cluster_df[cluster_col].unique() if c != -1])
    ordered_labels = [get_cluster_label(c, cluster_config) for c in clusters]

    viz_df = cluster_df[cluster_df[cluster_col] != -1].copy()
    viz_df["cluster_label"] = viz_df[cluster_col].apply(lambda c: get_cluster_label(c, cluster_config))

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    # 1. Cluster Sizes
    ax = axes[0]
    counts = viz_df["cluster_label"].value_counts().reindex(ordered_labels)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, order=ordered_labels, palette=DEFAULT_PALETTE)
    ax.set_title("A. Cluster Patient Distribution", fontsize=15, weight="bold")
    ax.set_ylabel("Patient Count (n)", fontsize=13)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for i, cnt in enumerate(counts):
        if pd.notna(cnt):
            ax.text(i, cnt + 1, str(int(cnt)), ha="center", va="bottom", fontsize=12, weight="bold")

    # 2. Female Sex Distribution
    ax = axes[1]
    if "sex_f" in viz_df.columns:
        crosstab = pd.crosstab(viz_df["cluster_label"], viz_df["sex_f"]).reindex(ordered_labels)
        crosstab.plot(kind="bar", stacked=True, ax=ax, color=["#4361EE", "#EC5B57"], alpha=0.85)
        ax.set_title("B. Sex Distribution by Cluster", fontsize=15, weight="bold")
        ax.set_ylabel("Patient Count (n)", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(["Male", "Female"], loc="upper right")

    # 3. Age Boxplot
    ax = axes[2]
    if "age" in viz_df.columns:
        sns.boxplot(data=viz_df, x="cluster_label", y="age", order=ordered_labels, ax=ax, palette=DEFAULT_PALETTE)
        ax.set_title("C. Patient Age (years) by Cluster", fontsize=15, weight="bold")
        ax.set_ylabel("Age (years)", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    # 4. Baseline BMI Boxplot
    ax = axes[3]
    if "baseline_bmi" in viz_df.columns:
        sns.boxplot(data=viz_df, x="cluster_label", y="baseline_bmi", order=ordered_labels, ax=ax, palette=DEFAULT_PALETTE)
        ax.set_title("D. Baseline BMI (kg/m²) by Cluster", fontsize=15, weight="bold")
        ax.set_ylabel("Baseline BMI (kg/m²)", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    # 5. Follow-up Days Boxplot
    ax = axes[4]
    if "total_followup_days" in viz_df.columns:
        sns.boxplot(data=viz_df, x="cluster_label", y="total_followup_days", order=ordered_labels, ax=ax, palette=DEFAULT_PALETTE)
        ax.set_title("E. Total Follow-up Duration (days) by Cluster", fontsize=15, weight="bold")
        ax.set_ylabel("Follow-up Length (days)", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    # 6. Total Weight Loss % Boxplot
    ax = axes[5]
    if "total_wl_%" in viz_df.columns:
        sns.boxplot(data=viz_df, x="cluster_label", y="total_wl_%", order=ordered_labels, ax=ax, palette=DEFAULT_PALETTE)
        ax.set_title("F. Total Weight Loss (%) by Cluster", fontsize=15, weight="bold")
        ax.set_ylabel("Total Weight Loss (%)", fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Clinical Phenotype Summary Across Clusters", fontsize=19, weight="bold", y=0.99)
    plt.tight_layout()
    out_file = _resolve_path(output_filepath)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Clinical Summary Figure: '{out_file}'")


# =============================================================================
# HIGH-LEVEL MASTER PIPELINE FUNCTION
# =============================================================================

def run_cluster_comparison_pipeline(
    source_db_path: str,
    labels_db_path: Optional[str] = None,
    distance_config: Optional[str] = "wgc07_eb03", # these are default configs that are overwritten from notebook! 
    algorithm: str = "pam",
    k: int = 7,
    cluster_column: Optional[str] = None,
    cluster_table: str = "jaccard_gower_cluster_labels",
    source_table: str = "timetoevent_eb_wgc_compl",
    variables: Optional[List[str]] = None,
    save_to_db: bool = True,
    output_db_path: Optional[str] = None,
    db_table_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_excel_path: Optional[str] = None,
    output_fig_path: Optional[str] = None,
    name_map_path: Optional[Union[str, Dict]] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    alpha: float = 0.05,
    generate_plots: bool = True,
    generate_full_figure_suite: bool = True,
    lollipop_continuous_vars: Optional[List[str]] = None,
    lollipop_categorical_vars: Optional[List[str]] = None,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master driver function to execute end-to-end cluster comparisons from notebook calls.

    Parameters:
    -----------
    source_db_path : str
        Path to SQLite database containing clinical cohort data.
    labels_db_path : Optional[str]
        Path to database containing cluster labels (defaults to source_db_path).
    distance_config : Optional[str]
        Hybrid distance weighting configuration (e.g. 'wgc07_eb03', 'wgc05_eb05').
    algorithm : str
        Clustering algorithm (default: 'pam').
    k : int
        Number of clusters (default: 7).
    cluster_column : Optional[str]
        Explicit cluster column if provided directly.
    cluster_table : str
        Labels table name (default: 'jaccard_gower_cluster_labels').
    source_table : str
        Clinical cohort table name (default: 'timetoevent_eb_wgc_compl').
    variables : Optional[List[str]]
        List of clinical variables to compare across clusters.
    save_to_db : bool
        Whether to save detailed and publication tables into SQLite (default: True).
    output_db_path : Optional[str]
        Database path to receive comparison tables.
    db_table_name : Optional[str]
        SQLite table name for saved results.
    output_dir : Optional[str]
        Directory for saving figure outputs.
    output_excel_path : Optional[str]
        File path to save results as Excel workbook.
    output_fig_path : Optional[str]
        File path to save 6-panel summary figure.
    name_map_path : Optional[Union[str, Dict]]
        Variable name map dictionary or JSON path.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors dictionary or JSON path.
    alpha : float
        Significance threshold for asterisks (default: 0.05).
    generate_plots : bool
        Whether to generate standard overview figures (default: True).
    generate_full_figure_suite : bool
        Whether to also automatically generate all individual granular figures
        (violins, stacked bars, lollipop, heatmap, forest) in output_dir (default: True).
    lollipop_continuous_vars : Optional[List[str]]
        Explicit continuous clinical variables to display on the lollipop plot (SMD vs Rest-of-Cohort).
        Defaults to ['age', 'baseline_weight_kg', 'max_wl_%', 'total_wl_%'].
    lollipop_categorical_vars : Optional[List[str]]
        Explicit binary/categorical clinical variables to display on the lollipop plot (Risk Difference vs Rest-of-Cohort).
        Defaults to ['sex_f', '10%_wl_achieved', '60d_dropout', '5%_wl_achieved'].

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (comparison_table, merged_cohort_df)
    """
    if labels_db_path is None:
        labels_db_path = source_db_path

    # Construct default table name if not provided
    if db_table_name is None:
        clean_dist = (distance_config or "cluster").replace("hybrid_", "")
        db_table_name = f"{algorithm}_comparison_{clean_dist}_k{k}"

    # Step 1: Load and merge data
    merged_df = load_and_merge_cluster_data(
        cluster_db_path=labels_db_path,
        main_db_path=source_db_path,
        cluster_table=cluster_table,
        cluster_column=cluster_column,
        weight=distance_config,
        algorithm=algorithm,
        k=k,
        outcome_table=source_table,
    )

    # Step 1b: Statistical power & minimum detectable effect size (MDES) analysis
    power_results = calculate_cluster_power_analysis(
        cluster_df=merged_df,
        cluster_col="cluster_id",
        alpha=alpha,
        power=0.80,
        cluster_config_path=cluster_config_path,
        print_summary=True,
    )

    # Step 2: Perform statistical profiling (cluster vs cohort mean)
    results_df = analyze_cluster_vs_population(
        cluster_df=merged_df,
        variables=variables,
        output_db_path=output_db_path,
        output_table_name=db_table_name,
        cluster_col="cluster_id",
        name_map_path=name_map_path,
        cluster_config_path=cluster_config_path,
        alpha=alpha,
        save_to_db=save_to_db,
    )

    # Attach power analysis results to dataframe attributes for seamless programmatic retrieval
    if power_results:
        results_df.attrs["power_analysis"] = power_results
        merged_df.attrs["power_analysis"] = power_results

    # Step 3: Export to Excel if requested (includes publication, detailed, and omnibus sheets)
    if output_excel_path:
        out_excel = _resolve_path(output_excel_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_excel)), exist_ok=True)
        try:
            pub_df = results_df.attrs.get("publication_df")
            omnibus_df = results_df.attrs.get("omnibus_df")
            with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
                if pub_df is not None:
                    pub_df.to_excel(writer, sheet_name="publication", index=False)
                results_df.to_excel(writer, sheet_name="detailed", index=False)
                if omnibus_df is not None:
                    omnibus_df.to_excel(writer, sheet_name="omnibus", index=False)
            print(f"  ✓ Exported Results Tables to Excel: '{out_excel}' (sheets: publication, detailed, omnibus)")
        except Exception as e:
            results_df.to_excel(out_excel, index=False)
            print(f"  ✓ Exported Results Table to Excel: '{out_excel}'")

    # Step 4: Generate figures
    if generate_plots:
        # A. 6-Panel Clinical Summary Overview Figure
        fig_dest = output_fig_path
        if fig_dest is None and output_dir is not None:
            fig_dest = os.path.join(output_dir, f"clinical_comparison_{db_table_name}.png")
        if fig_dest is not None:
            plot_cluster_summary_grid(
                cluster_df=merged_df,
                output_filepath=fig_dest,
                cluster_col="cluster_id",
                cluster_config_path=cluster_config_path,
            )

        # B. Optional Full Figure Suite (Granular publication figures)
        if generate_full_figure_suite and output_dir is not None:
            out_d = _resolve_path(output_dir)
            os.makedirs(out_d, exist_ok=True)
            print(f"\n[Figure Suite] Generating full publication figure set in: '{out_d}'...")

            # 1. Multi-attribute Prevalence & Likert Heatmap (WGCs + 3 bin EBs + 3 Likert EBs)
            wgc_vars = [v for v in DEFAULT_VARIABLE_DOMAINS["Weight gain causes"] if v in merged_df.columns]
            eb_bin_vars = [v for v in ["hunger_yn", "satiety_yn", "emotional_eating_yn"] if v in merged_df.columns]
            eb_likert_vars = [
                v for v in [
                    "emotional_eating_value_likert",
                    "quantity_control_likert",
                    "impulse_control_likert",
                ] if v in merged_df.columns
            ]
            heatmap_vars = wgc_vars + eb_bin_vars + eb_likert_vars
            if heatmap_vars:
                plot_cluster_heatmap(
                    results_df=results_df,
                    output_filename=f"heatmap_{db_table_name}.png",
                    output_dir=out_d,
                    variables=heatmap_vars,
                    name_map_path=name_map_path,
                    cluster_config_path=cluster_config_path,
                    alpha=alpha,
                )

            # 2. Multi-variable Lollipop Plot (SMD & Risk Difference vs. Rest-of-Cohort)
            if lollipop_continuous_vars is None:
                lp_cont = [
                    v for v in [
                        "age", 
                        "baseline_weight_kg", 
                        "baseline_bmi", 
                        "total_followup_days", 
                        # "days_to_nadir",
                        "max_wl_%",
                        "weight_regain_%", 
                        "total_wl_%",
                        # "60d_wl_%"
                    ]
                    if v in merged_df.columns
                ]
            else:
                lp_cont = [v for v in lollipop_continuous_vars if v in merged_df.columns]

            if lollipop_categorical_vars is None:
                lp_cat = [
                    v for v in [
                        "sex_f", 
                        "10%_wl_achieved", 
                        "60d_dropout", 
                        ]
                    if v in merged_df.columns
                ]
            else:
                lp_cat = [v for v in lollipop_categorical_vars if v in merged_df.columns]

            if lp_cont or lp_cat:
                plot_cluster_lollipop(
                    cluster_df=merged_df,
                    continuous_variables=lp_cont,
                    categorical_variables=lp_cat,
                    output_filename=f"lollipop_{db_table_name}.png",
                    output_dir=out_d,
                    cluster_col="cluster_id",
                    name_map_path=name_map_path,
                    cluster_config_path=cluster_config_path,
                    alpha=alpha,
                )

            # 3. Categorical Stacked Bar Charts for Core Outcomes
            cat_vars = [v for v in ["sex_f", "10%_wl_achieved", "60d_dropout", "5%_wl_achieved"] if v in merged_df.columns]
            if cat_vars:
                cluster_categorical_distributions(
                    cluster_df=merged_df,
                    variables=cat_vars,
                    output_dir=out_d,
                    cluster_col="cluster_id",
                    name_map_path=name_map_path,
                    cluster_config_path=cluster_config_path,
                    fdr_correction=True,
                    alpha=alpha,
                )

            # 4. Continuous Violin Plots for Core Anthropometrics & Appetite Scores
            violin_vars = [
                v for v in ["age", "baseline_bmi", "emotional_eating_value_likert", "quantity_control_likert", "total_wl_%"]
                if v in merged_df.columns
            ]
            if violin_vars:
                cluster_continuous_distributions(
                    cluster_df=merged_df,
                    variables=violin_vars,
                    output_dir=out_d,
                    cluster_col="cluster_id",
                    name_map_path=name_map_path,
                    cluster_config_path=cluster_config_path,
                    fdr_correction=True,
                    alpha=alpha,
                )

            # 5. Forest Plots for Key Prospective Outcomes
            for o_var in ["10%_wl_achieved", "60d_dropout"]:
                if o_var in merged_df.columns:
                    plot_cluster_forest(
                        cluster_df=merged_df,
                        outcome_variable=o_var,
                        output_dir=out_d,
                        cluster_col="cluster_id",
                        name_map_path=name_map_path,
                        cluster_config_path=cluster_config_path,
                        effect_type="both",
                    )

    return results_df, merged_df


# =============================================================================
# CLI / STANDALONE TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in_2025.sqlite")
    out_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_out_2025.sqlite")
    out_dir = os.path.join(base_dir, "p1_outputs", "cluster_comparisons_test")

    print("\n" + "#" * 80)
    print("RUNNING CLUSTER_COMPARISONS TEST RUN (PAM, wgc07_eb03, k=7)")
    print("#" * 80)

    res_df, cohort_df = run_cluster_comparison_pipeline(
        source_db_path=in_db,
        labels_db_path=in_db,
        distance_config="wgc07_eb03",
        algorithm="pam",
        k=7,
        save_to_db=True,
        output_db_path=out_db,
        db_table_name="pam_comparison_wgc07_eb03_k7",
        output_dir=out_dir,
        generate_plots=True,
    )

    print("\n" + "=" * 80)
    print("TEST EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Results shape: {res_df.shape}")
    print(f"Cohort shape:  {cohort_df.shape}")
    print("=" * 80 + "\n")
