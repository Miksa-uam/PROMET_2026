"""
================================================================================
CLUSTER_BASED_MODELING.PY
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module implements longitudinal trajectory and prospective clinical outcome
modeling for behavioral obesity phenotypes (Paper #1).

Building on cluster assignments generated from hybrid Jaccard-Gower dissimilarity
matrices (12 Weight Gain Causes + 6 Eating Behavior variables), this module
evaluates whether behavioral clusters translate into distinct longitudinal weight loss
trajectories and prospective clinical outcomes:

1. Descriptive Trajectory Analysis (Spaghetti Plots):
   - Longitudinal weight curves per patient with LOWESS non-parametric smoothed means.
   - Multi-panel comparison layout: Whole population benchmark, individual cluster panels,
     and an overlay panel.
   - Length-adjusted visualization: Solid trajectory up to the cohort/cluster mean follow-up,
     transitioning to dashed line beyond mean follow-up, accompanied by 95% confidence bands.

2. Model-Based Trajectory Prediction (Linear Mixed-Effects Models - LMM):
   - Repeated-measures model: weight_kg ~ cluster * time_spline (+ baseline covariates).
   - Random intercepts and slopes per patient/medical record account for intra-individual
     correlation of repeated measurements.
   - Natural B-spline time basis functions capture non-linear weight curves (rapid early
     loss followed by plateaus).
   - Generates model-predicted trajectory curves per cluster across a standardized follow-up grid.

3. Regression-Based Outcome Prediction:
   - Multivariable linear regression (OLS) for continuous clinical endpoints:
     * Total weight loss percentage (total_wl_%)
     * 60-day weight loss percentage (60d_wl_%)
     * Total follow-up duration in days (total_followup_days)
   - Multivariable binary logistic regression (Logit) for critical adherence endpoints:
     * 60-day dropout (60d_dropout)
   - Reports effect sizes (Beta coefficients or Odds Ratios), 95% CIs, and p-values.

4. Time-to-Event Survival Analysis (Kaplan-Meier):
   - Evaluates prospective time to achieving clinically meaningful weight loss (10% weight loss).
   - Event: 10%_wl_achieved (1 = achieved, 0 = censored).
   - Duration: days_to_10%_wl (if achieved) or total_followup_days (if censored).
   - Computes Kaplan-Meier survival curves, median time-to-event, and multivariate log-rank tests.

Author: Clinical Bioinformatician & Data Science Pair Programmer
Date: October 2026
Target Environment: Python 3.10+ / SQLite3 / Pandas / NumPy / SciPy / Statsmodels / Lifelines / Matplotlib
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
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# Suppress minor statistical and visualization convergence warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# DEFAULT CONSTANTS & COLOR PALETTES
# =============================================================================

DEFAULT_CLUSTER_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-green
    "#17becf",  # Teal
]

POPULATION_COLOR = "#D62728"  # High-contrast red for whole population benchmark
NEUTRAL_GRAY = "#7F7F7F"      # Gray for background individual trajectories


# =============================================================================
# PATH & CONFIGURATION RESOLUTION HELPERS
# =============================================================================

def _resolve_path(path: str) -> str:
    """
    Resolves relative file paths to absolute paths, checking script directory
    and current working directory.
    """
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


def load_cluster_config(config_source: Optional[Union[str, Dict]] = None) -> Dict[str, Any]:
    """
    Loads custom cluster labels and colors from a dictionary or JSON file.

    Parameters:
    -----------
    config_source : Optional[Union[str, Dict]]
        Path to cluster config JSON or dictionary.

    Returns:
    --------
    Dict[str, Any]: Dictionary containing 'cluster_labels' and 'cluster_colors'.
    """
    config = {"cluster_labels": {}, "cluster_colors": {}}
    if config_source is None:
        return config

    if isinstance(config_source, dict):
        config["cluster_labels"] = config_source.get("cluster_labels", config_source)
        config["cluster_colors"] = config_source.get("cluster_colors", {})
        return config

    if isinstance(config_source, str):
        json_path = _resolve_path(config_source)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                config["cluster_labels"] = data.get("cluster_labels", {})
                config["cluster_colors"] = data.get("cluster_colors", {})
                print(f"[Config] Loaded cluster configuration from: '{json_path}'")
            except Exception as e:
                print(f"Warning: Failed to parse cluster config JSON '{json_path}': {e}")
    return config


def get_cluster_label(cluster_id: int, cluster_config: Optional[Dict[str, Any]] = None) -> str:
    """Returns human-readable cluster label, falling back to 'Cluster {id}'."""
    if cluster_config:
        labels = cluster_config.get("cluster_labels", {})
        if str(cluster_id) in labels:
            return labels[str(cluster_id)]
        if cluster_id in labels:
            return labels[cluster_id]
    return f"Cluster {cluster_id}"


def get_cluster_color(cluster_id: int, cluster_config: Optional[Dict[str, Any]] = None) -> str:
    """Returns cluster-specific hex color code, falling back to default palette."""
    if cluster_config:
        colors = cluster_config.get("cluster_colors", {})
        if str(cluster_id) in colors:
            return colors[str(cluster_id)]
        if cluster_id in colors:
            return colors[cluster_id]
    return DEFAULT_CLUSTER_COLORS[int(cluster_id) % len(DEFAULT_CLUSTER_COLORS)]


# =============================================================================
# DATA LOADING & PREPARATION FUNCTIONS
# =============================================================================

def resolve_cluster_column_name(
    weight: str = "wgc07_eb03",
    algorithm: str = "pam",
    k: int = 7,
    available_columns: Optional[List[str]] = None,
) -> str:
    """
    Constructs the canonical cluster column name stored in the SQLite labels table.
    """
    clean_weight = weight.replace("hybrid_", "")
    algo_key = "pam" if "pam" in algorithm.lower() else "hclust_complete"
    target_col = f"hybrid_{clean_weight}_{algo_key}_k{k}"

    if available_columns and target_col not in available_columns:
        # Fallback search for partial matches
        candidates = [c for c in available_columns if f"k{k}" in c and algo_key in c and clean_weight in c]
        if candidates:
            return candidates[0]
        # More flexible fallback
        candidates = [c for c in available_columns if f"k{k}" in c and algo_key in c]
        if candidates:
            return candidates[0]
    return target_col


def load_and_prepare_trajectory_data(
    cluster_db_path: str,
    measurements_db_path: Optional[str] = None,
    cluster_table: str = "jaccard_gower_cluster_labels",
    measurements_table: str = "measurements_p1",
    cohort_table: str = "timetoevent_eb_wgc_compl",
    cluster_column: Optional[str] = None,
    weight: str = "wgc07_eb03",
    algorithm: str = "pam",
    k: int = 7,
    cutoff_days: Optional[int] = 365,
    patient_id_col: str = "patient_id",
    medical_record_id_col: str = "medical_record_id",
    measurement_date_col: str = "measurement_date",
    body_weight_col: str = "weight_kg",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads, merges, and prepares longitudinal measurement trajectories and cross-sectional
    cohort outcome data with cluster labels.

    Parameters:
    -----------
    cluster_db_path : str
        Path to SQLite database containing cluster assignments.
    measurements_db_path : Optional[str]
        Path to SQLite database containing measurements and outcomes (defaults to cluster_db_path).
    cluster_table : str
        Table name containing cluster labels (default: 'jaccard_gower_cluster_labels').
    measurements_table : str
        Table name containing repeated weight measurements (default: 'measurements_p1').
    cohort_table : str
        Table name containing prospective clinical outcomes (default: 'timetoevent_eb_wgc_compl').
    cluster_column : Optional[str]
        Explicit cluster column name (inferred if None).
    weight : str
        Distance weighting scheme (default: 'wgc07_eb03').
    algorithm : str
        Clustering algorithm ('pam' or 'hclust_complete').
    k : int
        Number of clusters (default: 7).
    cutoff_days : Optional[int]
        Maximum follow-up days to retain (e.g. 365, or None for full timespan).
    patient_id_col : str
        Patient ID column name (default: 'patient_id').
    medical_record_id_col : str
        Medical record ID column name (default: 'medical_record_id').
    measurement_date_col : str
        Date of measurement column name (default: 'measurement_date').
    body_weight_col : str
        Weight in kg column name (default: 'weight_kg').

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - trajectory_df: Longitudinal repeated measurements with 'days_from_baseline' and 'cluster_id'.
        - cohort_df: Cross-sectional patient cohort with outcomes and 'cluster_id'.
    """
    cluster_db = _resolve_path(cluster_db_path)
    meas_db = _resolve_path(measurements_db_path or cluster_db_path)

    print("\n" + "=" * 80)
    print("LOADING & PREPARING LONGITUDINAL TRAJECTORY & OUTCOME COHORT")
    print("=" * 80)
    print(f"Cluster Database:      {cluster_db}")
    print(f"Measurements Database: {meas_db}")
    print(f"Clustering Scheme:     Algorithm={algorithm}, Weight={weight}, k={k}")

    # 1. Load Cluster Labels
    with sqlite3.connect(cluster_db) as conn:
        label_cols = [r[1] for r in conn.execute(f"PRAGMA table_info(`{cluster_table}`)").fetchall()]
        if cluster_column is None:
            cluster_column = resolve_cluster_column_name(
                weight=weight, algorithm=algorithm, k=k, available_columns=label_cols
            )
        print(f"  ✓ Resolved Cluster Column: '{cluster_column}'")

        query_labels = (
            f"SELECT `{patient_id_col}`, `{medical_record_id_col}`, `{cluster_column}` AS cluster_id "
            f"FROM `{cluster_table}` WHERE `{cluster_column}` IS NOT NULL AND `{cluster_column}` != -1"
        )
        labels_df = pd.read_sql_query(query_labels, conn)
    labels_df["cluster_id"] = labels_df["cluster_id"].astype(int)
    print(f"  ✓ Loaded {len(labels_df)} valid patient cluster assignments across {labels_df['cluster_id'].nunique()} clusters.")

    # 2. Load Patient Cross-Sectional Outcome Cohort
    with sqlite3.connect(meas_db) as conn:
        cohort_raw = pd.read_sql_query(f"SELECT * FROM `{cohort_table}`", conn)
    cohort_df = pd.merge(
        cohort_raw,
        labels_df[[patient_id_col, medical_record_id_col, "cluster_id"]],
        on=[patient_id_col, medical_record_id_col],
        how="inner",
    )
    print(f"  ✓ Merged Cross-Sectional Cohort: {len(cohort_df)} patients.")

    # 3. Load Longitudinal Repeated Measurements
    with sqlite3.connect(meas_db) as conn:
        labels_df.to_sql("temp_cluster_patients", conn, if_exists="replace", index=False)
        query_meas = f"""
            SELECT m.`{patient_id_col}`, m.`{medical_record_id_col}`,
                   m.`{measurement_date_col}`, m.`{body_weight_col}`
            FROM `{measurements_table}` m
            JOIN temp_cluster_patients tcp
              ON m.`{patient_id_col}` = tcp.`{patient_id_col}`
             AND m.`{medical_record_id_col}` = tcp.`{medical_record_id_col}`
            WHERE m.`{body_weight_col}` IS NOT NULL AND m.`{body_weight_col}` > 20
            ORDER BY m.`{patient_id_col}`, m.`{medical_record_id_col}`, m.`{measurement_date_col}`
        """
        meas_df = pd.read_sql_query(query_meas, conn)

    # 4. Calculate Days from Baseline
    meas_df[measurement_date_col] = pd.to_datetime(meas_df[measurement_date_col])
    meas_df["days_from_baseline"] = (
        meas_df.groupby([patient_id_col, medical_record_id_col])[measurement_date_col]
        .transform(lambda x: (x - x.min()).dt.days)
    )

    # 5. Merge Measurements with Cluster Labels and Baseline Covariates
    covariate_cols = [c for c in ["age", "sex_f", "baseline_bmi", "baseline_weight_kg"] if c in cohort_df.columns]
    covariates_df = cohort_df[[patient_id_col, medical_record_id_col, "cluster_id"] + covariate_cols].drop_duplicates()

    trajectory_df = pd.merge(
        meas_df,
        covariates_df,
        on=[patient_id_col, medical_record_id_col],
        how="inner",
    )

    # 5b. Compute Patient Baseline Weight & Percentage Weight Loss per Measurement
    first_weight = (
        trajectory_df.sort_values(["days_from_baseline", measurement_date_col])
        .groupby([patient_id_col, medical_record_id_col])[body_weight_col]
        .transform("first")
    )
    trajectory_df["baseline_weight_measured"] = first_weight
    trajectory_df["weight_loss_pct"] = (
        (trajectory_df[body_weight_col] - first_weight) / first_weight * 100.0
    )

    # 6. Optional Time Cutoff Filtering
    if cutoff_days is not None:
        raw_count = len(trajectory_df)
        trajectory_df = trajectory_df[trajectory_df["days_from_baseline"] <= cutoff_days].copy()
        print(f"  ✓ Applied {cutoff_days}-day cutoff: {len(trajectory_df)} rows retained (from {raw_count}).")

    print(f"  ✓ Prepared Trajectory Dataset: {len(trajectory_df)} measurements for {trajectory_df[patient_id_col].nunique()} patients.")
    print("=" * 80 + "\n")

    return trajectory_df, cohort_df


# =============================================================================
# PART 1: RAW TRAJECTORY SPAGHETTI PLOTS (LENGTH-ADJUSTED & LOWESS SMOOTHED)
# =============================================================================

def _plot_trajectory_panel(
    ax: plt.Axes,
    sub_df: pd.DataFrame,
    title: str,
    color: str,
    smoothing_frac: float = 0.3,
    patient_id_col: str = "patient_id",
    medical_record_id_col: str = "medical_record_id",
    body_weight_col: str = "weight_kg",
    use_pct: bool = True,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Internal helper to render individual faint lines and a length-adjusted LOWESS smoothed mean.
    Solid line is plotted up to mean follow-up duration; dashed line beyond mean follow-up.
    """
    n_trajectories = sub_df[[patient_id_col, medical_record_id_col]].drop_duplicates().shape[0]

    y_col = "weight_loss_pct" if (use_pct and "weight_loss_pct" in sub_df.columns) else body_weight_col
    unit_str = "%" if y_col == "weight_loss_pct" else " kg"
    ylabel = "Weight Loss (%)" if y_col == "weight_loss_pct" else "Body Weight (kg)"

    # 1. Plot individual patient trajectories
    for _, grp in sub_df.groupby([patient_id_col, medical_record_id_col]):
        ax.plot(
            grp["days_from_baseline"],
            grp[y_col],
            alpha=0.35,
            linewidth=1.0,
            color=color if color != "red" else NEUTRAL_GRAY,
        )

    # 2. Compute follow-up duration and total weight delta
    followup_times = sub_df.groupby([patient_id_col, medical_record_id_col])["days_from_baseline"].max()
    mean_followup = float(followup_times.mean()) if len(followup_times) > 0 else 0.0

    if y_col == "weight_loss_pct":
        # In percentage mode, baseline is 0%, so delta is final observed percentage
        deltas = sub_df.groupby([patient_id_col, medical_record_id_col])[y_col].last()
    else:
        deltas = sub_df.groupby([patient_id_col, medical_record_id_col])[y_col].apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0.0
        )
    avg_delta = float(deltas.mean()) if len(deltas) > 0 else 0.0

    # 3. Compute Daily Aggregations and LOWESS Smoothed Mean
    daily_stats = (
        sub_df.groupby("days_from_baseline")[y_col]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    daily_stats["sem"] = daily_stats["std"] / np.sqrt(daily_stats["count"])

    smoothed, sem_smooth_full = None, None
    if len(daily_stats) > 3:
        smoothed = sm.nonparametric.lowess(
            daily_stats["mean"],
            daily_stats["days_from_baseline"],
            frac=smoothing_frac,
            return_sorted=True,
        )
        sem_smooth_full = sm.nonparametric.lowess(
            daily_stats["sem"].fillna(0),
            daily_stats["days_from_baseline"],
            frac=smoothing_frac,
            return_sorted=False,
        )

        # Split into solid (<= mean follow-up) and dashed (> mean follow-up) portions
        mask_solid = smoothed[:, 0] <= mean_followup
        mask_dashed = smoothed[:, 0] > mean_followup

        x_solid, y_solid = smoothed[:, 0][mask_solid], smoothed[:, 1][mask_solid]
        sem_solid = sem_smooth_full[mask_solid]

        x_dashed, y_dashed = smoothed[:, 0][mask_dashed], smoothed[:, 1][mask_dashed]
        sem_dashed = sem_smooth_full[mask_dashed]

        # Draw solid line with 95% CI using cluster-specific color
        line_color = color
        ax.plot(x_solid, y_solid, color=line_color, linewidth=2.8, label="Smoothed Mean")
        ax.fill_between(
            x_solid,
            y_solid - 1.96 * sem_solid,
            y_solid + 1.96 * sem_solid,
            color=line_color,
            alpha=0.25,
            label="95% CI",
        )

        # Draw dashed line beyond mean follow-up
        if len(x_dashed) > 0:
            ax.plot(x_dashed, y_dashed, color=line_color, linewidth=2.8, linestyle="--")
            ax.fill_between(
                x_dashed,
                y_dashed - 1.96 * sem_dashed,
                y_dashed + 1.96 * sem_dashed,
                color=line_color,
                alpha=0.15,
            )

    # Reference clinical thresholds for percentage weight loss
    if y_col == "weight_loss_pct":
        ax.axhline(0, color="#666666", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.axhline(-5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(-10, color="#888888", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_title(
        f"{title} (n={n_trajectories})\nAvg Δ: {avg_delta:+.1f}{unit_str} | Mean Follow-up: {mean_followup:.0f} d",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Days from Baseline", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

    return mean_followup, smoothed, sem_smooth_full


def plot_cluster_spaghetti(
    trajectory_df: pd.DataFrame,
    output_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    smoothing_frac: float = 0.3,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (20, 12),
    use_pct: bool = True,
    patient_id_col: str = "patient_id",
    medical_record_id_col: str = "medical_record_id",
    body_weight_col: str = "weight_kg",
) -> plt.Figure:
    """
    Renders multi-panel length-adjusted spaghetti plots for the whole population,
    individual clusters, and a consolidated overlay comparison panel.

    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        Longitudinal trajectory DataFrame.
    output_filepath : Optional[str]
        Exact path to save the generated figure PNG.
    output_dir : Optional[str]
        Directory to save figure if output_filepath is not given.
    cluster_config_path : Optional[Union[str, Dict]]
        Config mapping cluster IDs to custom titles and colors.
    smoothing_frac : float
        LOWESS smoothing fraction parameter (default: 0.3).
    title_suffix : str
        Additional suffix for the figure super-title.
    figsize : Tuple[int, int]
        Overall figure size (width, height).
    use_pct : bool
        If True, plots percentage weight loss (%) relative to baseline (default: True).
        If False, plots absolute body weight in kg.
    patient_id_col : str
        Patient ID column name (default: 'patient_id').
    medical_record_id_col : str
        Medical record ID column name (default: 'medical_record_id').
    body_weight_col : str
        Weight column in kg (default: 'weight_kg').

    Returns:
    --------
    plt.Figure: The rendered matplotlib figure object.
    """
    # Ensure weight_loss_pct column exists if use_pct is True
    if use_pct and "weight_loss_pct" not in trajectory_df.columns:
        first_w = (
            trajectory_df.sort_values("days_from_baseline")
            .groupby([patient_id_col, medical_record_id_col])[body_weight_col]
            .transform("first")
        )
        trajectory_df["weight_loss_pct"] = (
            (trajectory_df[body_weight_col] - first_w) / first_w * 100.0
        )

    y_col = "weight_loss_pct" if (use_pct and "weight_loss_pct" in trajectory_df.columns) else body_weight_col
    ylabel = "Weight Loss (%)" if y_col == "weight_loss_pct" else "Body Weight (kg)"

    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(trajectory_df["cluster_id"].unique())

    # Calculate grid dimensions: Panel 0 (Population) + K Clusters + Overlay Panel
    n_panels = len(clusters) + 2
    n_cols = min(3, len(clusters) + 1)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], (figsize[1] / 3) * n_rows),
        sharey=True,
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    print(f"\n[Spaghetti Plot] Rendering {n_panels}-panel trajectory overview ({len(clusters)} clusters + population + overlay, {ylabel})...")

    # Panel 0: Whole Population
    pop_followup, pop_smoothed, pop_sem = _plot_trajectory_panel(
        ax=axes_flat[0],
        sub_df=trajectory_df,
        title="Whole Population",
        color="red",
        smoothing_frac=smoothing_frac,
        patient_id_col=patient_id_col,
        medical_record_id_col=medical_record_id_col,
        body_weight_col=body_weight_col,
        use_pct=use_pct,
    )

    # Panels 1..K: Individual Clusters
    cluster_followups: Dict[int, float] = {}
    for idx, cid in enumerate(clusters):
        c_label = get_cluster_label(cid, cluster_config)
        c_color = get_cluster_color(cid, cluster_config)
        c_data = trajectory_df[trajectory_df["cluster_id"] == cid]

        followup, _, _ = _plot_trajectory_panel(
            ax=axes_flat[idx + 1],
            sub_df=c_data,
            title=c_label,
            color=c_color,
            smoothing_frac=smoothing_frac,
            patient_id_col=patient_id_col,
            medical_record_id_col=medical_record_id_col,
            body_weight_col=body_weight_col,
            use_pct=use_pct,
        )
        cluster_followups[cid] = followup

    # Final Panel: Overlay of Smoothed Mean Curves
    overlay_ax = axes_flat[len(clusters) + 1]
    overlay_ax.set_title("Overlay of Smoothed Mean Trajectories", fontsize=13, fontweight="bold")

    # Reference lines on overlay
    if y_col == "weight_loss_pct":
        overlay_ax.axhline(0, color="#666666", linestyle="-", linewidth=0.8, alpha=0.6)
        overlay_ax.axhline(-5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5)
        overlay_ax.axhline(-10, color="#888888", linestyle=":", linewidth=0.7, alpha=0.5)

    # Draw population line on overlay
    if pop_smoothed is not None:
        mask_solid_pop = pop_smoothed[:, 0] <= pop_followup
        mask_dashed_pop = pop_smoothed[:, 0] > pop_followup
        overlay_ax.plot(
            pop_smoothed[:, 0][mask_solid_pop],
            pop_smoothed[:, 1][mask_solid_pop],
            color="red",
            linewidth=3.0,
            linestyle="--",
            label="Population Benchmark",
        )
        if np.any(mask_dashed_pop):
            overlay_ax.plot(
                pop_smoothed[:, 0][mask_dashed_pop],
                pop_smoothed[:, 1][mask_dashed_pop],
                color="red",
                linewidth=1.8,
                linestyle=":",
            )

    # Reference clinical thresholds for overlay panel when in percentage mode
    if y_col == "weight_loss_pct":
        overlay_ax.axhline(0, color="#666666", linestyle="-", linewidth=0.8, alpha=0.6)
        overlay_ax.axhline(-5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5)
        overlay_ax.axhline(-10, color="#888888", linestyle=":", linewidth=0.7, alpha=0.5)

    # Draw each cluster trajectory on overlay
    for cid in clusters:
        c_label = get_cluster_label(cid, cluster_config)
        c_color = get_cluster_color(cid, cluster_config)
        c_data = trajectory_df[trajectory_df["cluster_id"] == cid]

        daily = c_data.groupby("days_from_baseline")[y_col].agg(["mean"]).reset_index()
        if len(daily) > 3:
            c_smoothed = sm.nonparametric.lowess(
                daily["mean"],
                daily["days_from_baseline"],
                frac=smoothing_frac,
                return_sorted=True,
            )
            mask_solid = c_smoothed[:, 0] <= cluster_followups[cid]
            mask_dashed = c_smoothed[:, 0] > cluster_followups[cid]

            overlay_ax.plot(
                c_smoothed[:, 0][mask_solid],
                c_smoothed[:, 1][mask_solid],
                color=c_color,
                linewidth=2.5,
                label=c_label,
            )
            if np.any(mask_dashed):
                overlay_ax.plot(
                    c_smoothed[:, 0][mask_dashed],
                    c_smoothed[:, 1][mask_dashed],
                    color=c_color,
                    linewidth=2.5,
                    linestyle="--",
                )

    overlay_ax.set_xlabel("Days from Baseline", fontsize=10)
    overlay_ax.set_ylabel(ylabel, fontsize=10)
    overlay_ax.legend(loc="upper right", fontsize=8)
    overlay_ax.grid(True, linestyle="--", alpha=0.5)

    # Hide any unused subplots
    for i in range(len(clusters) + 2, len(axes_flat)):
        axes_flat[i].set_visible(False)

    metric_name = "Percentage Weight Loss" if y_col == "weight_loss_pct" else "Weight Loss"
    fig.suptitle(
        f"{metric_name} Trajectories by Behavioral Phenotype Cluster {title_suffix}".strip(),
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure if path specified
    dest_path = output_filepath
    if dest_path is None and output_dir is not None:
        dest_path = os.path.join(_resolve_path(output_dir), f"spaghetti_trajectories_{title_suffix.strip(' ()')}.png")

    if dest_path is not None:
        dest_path = _resolve_path(dest_path)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        fig.savefig(dest_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved Spaghetti Trajectory Figure: '{dest_path}'")

    return fig


# =============================================================================
# PART 1B: CLUSTER-BASED NADIR & WEIGHT REGAIN ANALYSIS
# =============================================================================

def compute_nadir_weight_loss_metrics(
    trajectory_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    patient_id_col: str = "patient_id",
    medical_record_id_col: str = "medical_record_id",
    body_weight_col: str = "weight_kg",
    days_col: str = "days_from_baseline",
    cluster_config_path: Optional[Union[str, Dict]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes patient-level nadir and weight regain metrics and performs a statistical
    comparison across behavioral phenotype clusters.

    Scientific Task & Clinical Rationale:
    -------------------------------------
    Weight loss trajectories in clinical cohorts generally exhibit three dynamic phases:
      1. Maximum Weight Loss (%): Initial baseline to lowest achieved body weight (nadir).
         Formula: ((W_nadir - W_baseline) / W_baseline) * 100
      2. Weight Regain (%): Percentage weight recidivism from nadir to final follow-up.
         Formula: ((W_final - W_nadir) / W_nadir) * 100
      3. Total Weight Loss (%): Overall net change from baseline to final follow-up.
         Formula: ((W_final - W_baseline) / W_baseline) * 100

    This function tests between-cluster differences using:
      - One-way ANOVA (parametric F-test for differences in cluster means)
      - Kruskal-Wallis test (non-parametric rank-sum test for differences in cluster medians)

    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        Longitudinal measurements containing patient identifiers, cluster assignments,
        days from baseline, and body weight.
    cluster_col : str
        Column containing cluster labels (default: 'cluster_id').
    patient_id_col : str
        Patient identifier column.
    medical_record_id_col : str
        Medical record identifier column.
    body_weight_col : str
        Body weight in kg column.
    days_col : str
        Days from baseline column.
    cluster_config_path : Optional[Union[str, Dict]]
        Path or dictionary mapping cluster IDs to names/colors.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        - summary_df: Consolidated comparison table across clusters with p-values.
        - patient_df: Patient-level dataset containing computed nadir and regain metrics.
    """
    cluster_config = load_cluster_config(cluster_config_path)

    # 1. Sort observations chronologically per patient
    sorted_df = trajectory_df.sort_values(
        [patient_id_col, medical_record_id_col, days_col]
    )

    # 2. Extract baseline, nadir, and final weights per patient
    patient_records = []
    for (pid, mid), grp in sorted_df.groupby([patient_id_col, medical_record_id_col]):
        cid = grp[cluster_col].iloc[0]
        w_bl = grp[body_weight_col].iloc[0]
        w_nadir = grp[body_weight_col].min()
        w_final = grp[body_weight_col].iloc[-1]

        max_wl = ((w_nadir - w_bl) / w_bl) * 100.0 if w_bl > 0 else np.nan
        regain = ((w_final - w_nadir) / w_nadir) * 100.0 if w_nadir > 0 else np.nan
        total_wl = ((w_final - w_bl) / w_bl) * 100.0 if w_bl > 0 else np.nan

        patient_records.append({
            patient_id_col: pid,
            medical_record_id_col: mid,
            cluster_col: cid,
            "max_wl_%": max_wl,
            "regain_%": regain,
            "total_wl_%": total_wl,
        })

    patient_df = pd.DataFrame(patient_records)
    clusters = sorted(patient_df[cluster_col].unique())

    metrics_meta = [
        {"id": "max_wl_%", "name": "Max Weight Loss (%) [Baseline to Nadir]"},
        {"id": "regain_%", "name": "Weight Regain (%) [Nadir to Final]"},
        {"id": "total_wl_%", "name": "Total Weight Loss (%) [Baseline to Final]"},
    ]

    summary_rows = []
    for m in metrics_meta:
        m_id = m["id"]
        row: Dict[str, Any] = {"Metric": m["name"]}

        # Whole population summary
        pop_vals = patient_df[m_id].dropna()
        row["Whole Population"] = f"{pop_vals.mean():+.2f}% (+/-{pop_vals.std():.2f}%)"

        # Per-cluster summary
        cluster_groups = []
        for cid in clusters:
            c_label = get_cluster_label(cid, cluster_config)
            c_vals = patient_df.loc[patient_df[cluster_col] == cid, m_id].dropna()
            cluster_groups.append(c_vals.values)
            row[f"{c_label} (n={len(c_vals)})"] = f"{c_vals.mean():+.2f}% (+/-{c_vals.std():.2f}%)"

        # Statistical tests across clusters
        valid_groups = [g for g in cluster_groups if len(g) > 0]
        if len(valid_groups) > 1:
            try:
                f_stat, f_p = stats.f_oneway(*valid_groups)
                kw_stat, kw_p = stats.kruskal(*valid_groups)
                row["ANOVA p-val"] = f"{f_p:.4f}"
                row["Kruskal p-val"] = f"{kw_p:.4f}"
            except Exception:
                row["ANOVA p-val"] = "N/A"
                row["Kruskal p-val"] = "N/A"
        else:
            row["ANOVA p-val"] = "N/A"
            row["Kruskal p-val"] = "N/A"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 3. Print clean diagnostics summary table
    print("\n" + "=" * 80)
    print("CLUSTER-BASED NADIR & WEIGHT REGAIN STATISTICAL COMPARISON")
    print("=" * 80)
    print(f"Cohort Size:        {len(patient_df)} patients across {len(clusters)} clusters")
    print("Metrics Evaluated:  Max Weight Loss (%), Weight Regain (%), Total Weight Loss (%)")
    print("Statistical Tests:  One-way ANOVA (F-test) & Kruskal-Wallis (H-test)")
    print("=" * 80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    return summary_df, patient_df


# =============================================================================
# PART 2: LINEAR MIXED-EFFECTS MODEL (LMM) FOR TRAJECTORY ESTIMATION
# =============================================================================

def fit_trajectory_mixed_model(
    trajectory_df: pd.DataFrame,
    use_deviation_coding: bool = True,
    reference_cluster: int = 0,
    adjust_for_covariates: bool = False,
    time_degree: int = 2,
    patient_id_col: str = "patient_id",
    medical_record_id_col: str = "medical_record_id",
    body_weight_col: str = "weight_kg",
) -> Tuple[Any, pd.DataFrame]:
    """
    Fits a Linear Mixed-Effects Model (LMM) with non-linear spline time curves and random effects.

    Model Formula:
      weight_kg ~ cluster_term * bs(days_from_baseline, df=3, degree=2) [+ age + C(sex_f) + baseline_bmi]
      Random effects: ~days_from_baseline grouped by medical_record_id (patient level).

    Parameters:
    -----------
    trajectory_df : pd.DataFrame
        Longitudinal trajectory DataFrame.
    use_deviation_coding : bool
        If True, contrasts clusters against the grand mean (Sum coding).
        If False, uses reference coding against reference_cluster.
    reference_cluster : int
        Reference cluster index when use_deviation_coding=False.
    adjust_for_covariates : bool
        Whether to adjust for age, sex_f, and baseline_bmi.
    time_degree : int
        Polynomial degree for the time spline (default: 2, quadratic B-spline).
        Degree=2 with df=3 prevents runaway cubic tail extrapolations in sparse dropout clusters.

    Returns:
    --------
    Tuple[MixedLMResults, pd.DataFrame]:
        Fitted statsmodels MixedLMResults object and model-ready DataFrame.
    """
    df_model = trajectory_df.copy()
    df_model["cluster_id"] = pd.Categorical(df_model["cluster_id"])

    # Spline time curves:
    # In clinical cohorts with substantial attrition, cubic splines (degree=3) can extrapolate
    # violently in sparse clusters beyond the knot boundary. Quadratic B-splines (degree=2, df=3)
    # capture rapid early weight loss, nadir, and stabilization while maintaining numerical stability.
    if time_degree == 2:
        time_spline = f"patsy.bs(days_from_baseline, df=3, degree={time_degree})"
    else:
        unique_days = df_model["days_from_baseline"].unique()
        knots = np.quantile(unique_days, [0.25, 0.5, 0.75])
        time_spline = f"patsy.bs(days_from_baseline, knots={tuple(knots)}, degree={time_degree})"

    if use_deviation_coding:
        coding_str = "Deviation Coding (contrasts vs. Grand Mean)"
        cluster_term = "C(cluster_id, Sum)"
    else:
        coding_str = f"Reference Coding (vs. Cluster {reference_cluster})"
        cats = sorted(df_model["cluster_id"].cat.categories)
        if reference_cluster in cats:
            cats.remove(reference_cluster)
            new_order = [reference_cluster] + cats
            df_model["cluster_id"] = df_model["cluster_id"].cat.reorder_categories(new_order, ordered=True)
        cluster_term = "cluster_id"

    formula = f"{body_weight_col} ~ {cluster_term} * {time_spline}"

    covariates_included = False
    if adjust_for_covariates:
        available_covs = [c for c in ["age", "sex_f", "baseline_bmi"] if c in df_model.columns]
        if available_covs:
            cov_terms = []
            for c in available_covs:
                cov_terms.append(f"C({c})" if c == "sex_f" else c)
            formula += " + " + " + ".join(cov_terms)
            covariates_included = True

    print("\n" + "-" * 80)
    print("FITTING LINEAR MIXED-EFFECTS MODEL (LMM)")
    print("-" * 80)
    print(f"Strategy:     {coding_str}")
    print(f"Formula:      {formula}")
    print(f"Grouping:     Random intercept & slope per '{medical_record_id_col}'")
    print(f"Covariates:   {'Adjusted (' + ', '.join(available_covs) + ')' if covariates_included else 'Unadjusted'}")
    print("Fitting model via L-BFGS optimizer (this may take a few moments)...")

    try:
        # First attempt: Random intercept and random slope
        model = smf.mixedlm(
            formula,
            data=df_model,
            groups=df_model[medical_record_id_col],
            re_formula="~days_from_baseline",
        )
        result = model.fit(method=["lbfgs"], maxiter=200)

        # Check whether random slope covariance is well-conditioned & positive definite
        is_cov_valid = False
        try:
            cov = result.cov_params()
            if np.all(np.linalg.eigvals(cov) > 0) and not np.any(np.isnan(result.bse)):
                is_cov_valid = True
        except Exception:
            is_cov_valid = False

        if not result.converged or not is_cov_valid:
            print("  ⚠️ Random slope covariance is singular/non-positive-definite. Falling back to robust random intercept model...")
            model = smf.mixedlm(
                formula,
                data=df_model,
                groups=df_model[medical_record_id_col],
                re_formula="~1",
            )
            result = model.fit(method=["lbfgs"], maxiter=200)
    except Exception as e:
        print(f"  Note: Random slope optimization encountered: {e}. Fitting random intercept model...")
        model = smf.mixedlm(
            formula,
            data=df_model,
            groups=df_model[medical_record_id_col],
            re_formula="~1",
        )
        result = model.fit(method=["lbfgs"], maxiter=200)

    print(f"  ✓ LMM Fitting Complete! Converged: {result.converged}, Log-Likelihood: {result.llf:.2f}")
    return result, df_model


def analyze_trajectory_mixed_model(
    lmm_result: Any,
    df_model: pd.DataFrame,
    use_deviation_coding: bool = True,
    reference_cluster: int = 0,
    patient_id_col: str = "medical_record_id",
    cluster_config_path: Optional[Union[str, Dict]] = None,
    body_weight_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs rigorous hypothesis testing on Linear Mixed-Effects Model (LMM) trajectories.

    Statistical Testing Strategy:
    -----------------------------
    1. Omnibus Trajectory Difference Test (Cluster x Time Spline Interaction):
       Joint Wald test across all interaction coefficients:
         H0: beta_{cluster x time} = 0
       Tests whether non-linear weight trajectories differ significantly across clusters.

    2. Omnibus Baseline Difference Test (Cluster Main Effects):
       Joint Wald test across cluster main effect coefficients:
         H0: beta_{cluster} = 0
       Tests whether baseline body weight differs significantly across clusters.

    3. Per-Cluster Trajectory Contrasts:
       - Deviation Coding (use_deviation_coding=True):
         Contrasts each cluster's trajectory shape against the grand mean population trajectory.
       - Reference Coding (use_deviation_coding=False):
         Contrasts each non-reference cluster against reference_cluster (e.g. Cluster 0).

    Parameters:
    -----------
    lmm_result : MixedLMResults
        Fitted MixedLM results object.
    df_model : pd.DataFrame
        DataFrame used to fit the model.
    use_deviation_coding : bool
        Whether deviation coding (Sum) or reference coding was used.
    reference_cluster : int
        Reference cluster index if reference coding was used.
    patient_id_col : str
        Column identifying individual patients (default: 'medical_record_id').
    cluster_config_path : Optional[Union[str, Dict]]
        Path to cluster config JSON or dictionary for human-readable labels.
    body_weight_col : Optional[str]
        Dependent variable column name ('weight_kg' or 'weight_loss_pct').
        If None, automatically inferred from model.

    Returns:
    --------
    Dict[str, Any]:
        Dictionary containing:
        - 'omnibus_trajectory': dict(chi2, df, p_value)
        - 'omnibus_baseline': dict(chi2, df, p_value)
        - 'cluster_contrasts_df': pd.DataFrame with per-cluster metrics & tests
    """
    if body_weight_col is None:
        endog = getattr(lmm_result.model, "endog_names", "")
        body_weight_col = endog if endog in df_model.columns else "weight_kg"

    is_pct = (body_weight_col == "weight_loss_pct")
    delta_col_name = "observed_mean_delta_%" if is_pct else "observed_mean_delta_kg"

    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(df_model["cluster_id"].unique())
    k = len(clusters)
    param_names = lmm_result.params.index.tolist()

    # 1. Omnibus Trajectory Test (Cluster x Time Spline interaction terms)
    inter_terms = [p for p in param_names if ":" in p and "patsy.bs" in p]
    if len(inter_terms) > 0:
        w_traj = lmm_result.wald_test(inter_terms)
        chi2_traj = float(np.squeeze(w_traj.statistic))
        df_traj = len(inter_terms)
        p_traj = float(w_traj.pvalue)
    else:
        chi2_traj, df_traj, p_traj = np.nan, 0, np.nan

    # 2. Omnibus Baseline Test (Cluster main terms)
    cluster_main = [p for p in param_names if "cluster_id" in p and ":" not in p]
    if len(cluster_main) > 0:
        w_base = lmm_result.wald_test(cluster_main)
        chi2_base = float(np.squeeze(w_base.statistic))
        df_base = len(cluster_main)
        p_base = float(w_base.pvalue)
    else:
        chi2_base, df_base, p_base = np.nan, 0, np.nan

    # Identify spline degrees from parameter names
    spline_indices = sorted(list(set([
        re.findall(r"patsy\.bs.*?\[(\d+)\]", t)[0]
        for t in inter_terms if re.search(r"patsy\.bs.*?\[(\d+)\]", t)
    ])))
    n_spline = len(spline_indices) if len(spline_indices) > 0 else 3

    # 3. Per-Cluster Trajectory Contrasts
    rows = []
    for cid in clusters:
        sub_c = df_model[df_model["cluster_id"].astype(str) == str(cid)]
        pt_fus = sub_c.groupby(patient_id_col)["days_from_baseline"].max()
        n_pts = len(pt_fus)
        mean_fu = float(pt_fus.mean()) if n_pts > 0 else np.nan
        median_fu = float(pt_fus.median()) if n_pts > 0 else np.nan

        # Observed weight change
        if is_pct and "weight_loss_pct" in sub_c.columns:
            delta_val = float(sub_c.sort_values("days_from_baseline").groupby(patient_id_col)["weight_loss_pct"].last().mean())
        else:
            bw_col = body_weight_col if body_weight_col in sub_c.columns else "weight_kg"
            w_first = sub_c.sort_values("days_from_baseline").groupby(patient_id_col)[bw_col].first().mean()
            w_last = sub_c.sort_values("days_from_baseline").groupby(patient_id_col)[bw_col].last().mean()
            delta_val = float(w_last - w_first) if (pd.notna(w_last) and pd.notna(w_first)) else np.nan

        c_label = get_cluster_label(cid, cluster_config)

        if use_deviation_coding:
            contrast_type = "vs. Grand Mean"
            if cid < k - 1:
                # Explicit level in Sum coding
                terms = [p for p in inter_terms if f"[S.{cid}]" in p]
                if terms:
                    w_c = lmm_result.wald_test(terms)
                    stat_c = float(np.squeeze(w_c.statistic))
                    pval_c = float(w_c.pvalue)
                else:
                    stat_c, pval_c = np.nan, np.nan
            else:
                # Omitted level in Sum coding: beta_{k-1} = -sum(beta_0 ... beta_{k-2})
                R = np.zeros((n_spline, len(param_names)))
                for i_s, s_idx in enumerate(spline_indices):
                    for c_other in clusters[:-1]:
                        p_matches = [p for p in inter_terms if f"[S.{c_other}]" in p and f"[{s_idx}]" in p]
                        for p_m in p_matches:
                            R[i_s, param_names.index(p_m)] = 1.0
                try:
                    w_c = lmm_result.wald_test(R)
                    stat_c = float(np.squeeze(w_c.statistic))
                    pval_c = float(w_c.pvalue)
                except Exception:
                    stat_c, pval_c = np.nan, np.nan
        else:
            contrast_type = f"vs. Cluster {reference_cluster}"
            if cid == reference_cluster:
                stat_c, pval_c = np.nan, np.nan
            else:
                terms = [p for p in inter_terms if f"[{cid}]" in p or f"[T.{cid}]" in p]
                if terms:
                    w_c = lmm_result.wald_test(terms)
                    stat_c = float(np.squeeze(w_c.statistic))
                    pval_c = float(w_c.pvalue)
                else:
                    stat_c, pval_c = np.nan, np.nan

        # Formatting significance flag
        if pd.isna(pval_c):
            stars = "Ref" if not use_deviation_coding and cid == reference_cluster else "N/A"
        elif pval_c < 0.001:
            stars = "***"
        elif pval_c < 0.01:
            stars = "**"
        elif pval_c < 0.05:
            stars = "*"
        elif pval_c < 0.1:
            stars = "."
        else:
            stars = "ns"

        rows.append({
            "cluster_id": cid,
            "cluster_label": c_label,
            "contrast_type": contrast_type,
            "n_patients": n_pts,
            "mean_followup_days": round(mean_fu, 1) if not np.isnan(mean_fu) else np.nan,
            "median_followup_days": round(median_fu, 1) if not np.isnan(median_fu) else np.nan,
            delta_col_name: round(delta_val, 2) if not np.isnan(delta_val) else np.nan,
            "wald_chi2": round(stat_c, 2) if not np.isnan(stat_c) else np.nan,
            "df": n_spline,
            "p_value": pval_c,
            "significance": stars,
        })

    contrast_df = pd.DataFrame(rows)

    # Print publication-grade diagnostics
    print("\n" + "=" * 80)
    print("LONGITUDINAL TRAJECTORY STATISTICAL ANALYSIS (LMM WALD TESTS)")
    print("=" * 80)
    p_traj_str = "< 0.0001" if p_traj < 0.0001 else f"= {p_traj:.4f}"
    p_base_str = "< 0.0001" if p_base < 0.0001 else f"= {p_base:.4f}"
    print(f"1. Trajectory Shape Differences (Cluster x Time Interaction):")
    print(f"   Omnibus Wald Chi2 = {chi2_traj:.2f} (df={df_traj}), p-value {p_traj_str}")
    if p_traj < 0.05:
        print("   -> Conclusion: Longitudinal weight loss trajectories differ significantly between clusters.")
    else:
        print("   -> Conclusion: Trajectories do not differ significantly between clusters (parallel profiles).")

    print(f"\n2. Baseline Differences (Cluster Main Effect):")
    print(f"   Omnibus Wald Chi2 = {chi2_base:.2f} (df={df_base}), p-value {p_base_str}")

    print(f"\n3. Per-Cluster Trajectory Contrasts ({contrast_type}):")
    disp_cols = ["cluster_id", "cluster_label", "n_patients", "mean_followup_days", delta_col_name, "wald_chi2", "p_value", "significance"]
    print(contrast_df[disp_cols].to_string(index=False))
    print("=" * 80 + "\n")

    return {
        "omnibus_trajectory": {"chi2": chi2_traj, "df": df_traj, "p_value": p_traj},
        "omnibus_baseline": {"chi2": chi2_base, "df": df_base, "p_value": p_base},
        "cluster_contrasts_df": contrast_df,
    }


def plot_lmm_predicted_trajectories(
    lmm_result: Any,
    df_model: pd.DataFrame,
    output_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (12, 8),
    stats_summary: Optional[Dict[str, Any]] = None,
    dashing_cutoff: str = "cluster_mean",
    annotate_statistics: bool = True,
    show_ci: bool = True,
    use_pct: Optional[bool] = None,
    patient_id_col: str = "medical_record_id",
) -> plt.Figure:
    """
    Renders model-predicted mean weight loss trajectories per cluster over a standardized time grid.

    Follow-Up Length Adjustment:
    ----------------------------
    Solid curves are plotted across the high-density observation window (<= cluster mean follow-up).
    Dashed curves indicate sparse cohort extrapolation (> cluster mean follow-up).

    Confidence Bands:
    -----------------
    Shaded 95% confidence bands (+/- 1.96 * SE) illustrate pointwise estimation precision when show_ci=True.

    Statistical Significance Annotation:
    -----------------------------------
    If annotate_statistics is True, the omnibus trajectory difference Wald test is annotated
    under the figure title, and per-cluster contrast p-values are annotated in the legend.

    Parameters:
    -----------
    lmm_result : MixedLMResults
        Fitted MixedLM model results from fit_trajectory_mixed_model.
    df_model : pd.DataFrame
        DataFrame used to fit the model.
    output_filepath : Optional[str]
        Exact path to save the generated figure.
    output_dir : Optional[str]
        Directory to save figure if output_filepath is not given.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    title_suffix : str
        Additional title suffix.
    figsize : Tuple[int, int]
        Figure size dimensions.
    stats_summary : Optional[Dict[str, Any]]
        Pre-computed trajectory statistical analysis (from analyze_trajectory_mixed_model).
        If None and annotate_statistics is True, computed automatically.
    dashing_cutoff : str
        Threshold where solid trajectory transitions to dashed line:
        - 'cluster_mean': Each cluster's mean follow-up duration (default).
        - 'cohort_mean': Whole cohort's mean follow-up duration.
        - 'max': Last observed measurement in each cluster.
        - numeric value: Specific fixed day cutoff.
    annotate_statistics : bool
        Whether to annotate Wald test statistics in title and legend.
    show_ci : bool
        Whether to render shaded 95% confidence intervals around trajectory curves (default: True).
    use_pct : Optional[bool]
        If True, plots percentage weight loss (%) relative to baseline.
        If False, plots absolute body weight in kg.
        If None, automatically inferred from model dependent variable.
    patient_id_col : str
        Patient ID column name (default: 'medical_record_id').

    Returns:
    --------
    plt.Figure: The rendered matplotlib figure object.
    """
    if use_pct is None:
        endog = getattr(lmm_result.model, "endog_names", "")
        use_pct = (endog == "weight_loss_pct") or ("weight_loss_pct" in str(getattr(lmm_result.model.data, "ynames", "")))

    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(df_model["cluster_id"].unique())
    time_grid = np.linspace(
        df_model["days_from_baseline"].min(),
        df_model["days_from_baseline"].max(),
        200,
    )

    # Compute trajectory statistical tests if not already provided
    if annotate_statistics and stats_summary is None:
        try:
            stats_summary = analyze_trajectory_mixed_model(
                lmm_result=lmm_result,
                df_model=df_model,
                patient_id_col=patient_id_col,
                cluster_config_path=cluster_config_path,
            )
        except Exception as e:
            print(f"Warning: Could not compute trajectory statistics for annotation: {e}")
            stats_summary = None

    # Calculate overall cohort mean follow-up
    all_pt_fus = df_model.groupby(patient_id_col)["days_from_baseline"].max()
    cohort_mean_fu = float(all_pt_fus.mean()) if len(all_pt_fus) > 0 else 0.0

    # Extract design info and covariance for analytical standard errors
    orig_di = getattr(lmm_result.model.data, "design_info", None)
    cov_fe = None
    fe_names = []
    if show_ci and orig_di is not None:
        try:
            fe_names = lmm_result.fe_params.index.tolist()
            cov_fe = lmm_result.cov_params().loc[fe_names, fe_names].values
        except Exception:
            cov_fe = None

    fig, ax = plt.subplots(figsize=figsize)

    for cid in clusters:
        pred_df = pd.DataFrame({
            "days_from_baseline": time_grid,
            "cluster_id": pd.Series([cid] * len(time_grid), dtype="category").cat.set_categories(
                df_model["cluster_id"].cat.categories
            ),
        })

        # Set covariates at median / reference values if present in model
        if "age" in lmm_result.model.exog_names:
            pred_df["age"] = df_model["age"].median()
        if "C(sex_f)[T.1]" in lmm_result.model.exog_names or "C(sex_f)[T.1.0]" in lmm_result.model.exog_names:
            pred_df["sex_f"] = 1
        if "baseline_bmi" in lmm_result.model.exog_names:
            pred_df["baseline_bmi"] = df_model["baseline_bmi"].median()

        try:
            preds = lmm_result.predict(pred_df)
        except Exception:
            # Fallback direct design matrix multiplication
            exog = patsy.dmatrix(lmm_result.model.data.orig_fe_formula, pred_df, return_type="dataframe")
            preds = np.dot(exog, lmm_result.fe_params)

        # Analytical pointwise standard errors for mean curve
        se_pred = None
        if show_ci and cov_fe is not None and orig_di is not None:
            try:
                exog_mat = patsy.build_design_matrices([orig_di], pred_df)[0]
                X = np.asarray(exog_mat)
                var_pred = np.sum((X @ cov_fe) * X, axis=1)
                if np.all(var_pred >= 0) and not np.any(np.isnan(var_pred)):
                    se_pred = np.sqrt(var_pred)
            except Exception:
                se_pred = None

        c_label = get_cluster_label(cid, cluster_config)
        c_color = get_cluster_color(cid, cluster_config)

        # Statistical annotation in legend
        legend_lbl = c_label
        if annotate_statistics and stats_summary is not None:
            c_df = stats_summary["cluster_contrasts_df"]
            c_row = c_df[c_df["cluster_id"].astype(str) == str(cid)]
            if not c_row.empty:
                pval_c = c_row.iloc[0]["p_value"]
                stars_c = c_row.iloc[0]["significance"]
                if pd.isna(pval_c) or stars_c == "Ref":
                    legend_lbl = f"{c_label} (Ref)"
                elif pval_c < 0.001:
                    legend_lbl = f"{c_label} (p < 0.001***)"
                else:
                    star_suffix = f"{stars_c}" if stars_c != "ns" else ""
                    legend_lbl = f"{c_label} (p = {pval_c:.3f}{star_suffix})"

        # Determine cutoff day for dashing
        sub_c = df_model[df_model["cluster_id"].astype(str) == str(cid)]
        pt_fus = sub_c.groupby(patient_id_col)["days_from_baseline"].max()
        c_mean_fu = float(pt_fus.mean()) if len(pt_fus) > 0 else cohort_mean_fu

        if dashing_cutoff == "cluster_mean":
            cutoff_day = c_mean_fu
        elif dashing_cutoff == "cohort_mean":
            cutoff_day = cohort_mean_fu
        elif dashing_cutoff == "max":
            cutoff_day = float(sub_c["days_from_baseline"].max())
        elif isinstance(dashing_cutoff, (int, float)):
            cutoff_day = float(dashing_cutoff)
        else:
            cutoff_day = c_mean_fu

        pred_vals = np.asarray(preds)

        # 1. High-density observed follow-up window (solid line & confidence band)
        obs_mask = time_grid <= cutoff_day
        ax.plot(
            time_grid[obs_mask],
            pred_vals[obs_mask],
            color=c_color,
            linewidth=2.8,
            label=legend_lbl,
            zorder=3,
        )
        if show_ci and se_pred is not None:
            ci_low = pred_vals - 1.96 * se_pred
            ci_high = pred_vals + 1.96 * se_pred
            ax.fill_between(
                time_grid[obs_mask],
                ci_low[obs_mask],
                ci_high[obs_mask],
                color=c_color,
                alpha=0.20 if use_pct else 0.18,
                zorder=2,
            )

        # 2. Sparse cohort extrapolation window (dashed line & light band)
        ext_mask = time_grid >= cutoff_day
        if np.sum(ext_mask) > 1:
            ax.plot(
                time_grid[ext_mask],
                pred_vals[ext_mask],
                color=c_color,
                linewidth=1.8,
                linestyle="--",
                alpha=0.75,
                zorder=3,
            )
            if show_ci and se_pred is not None:
                ax.fill_between(
                    time_grid[ext_mask],
                    ci_low[ext_mask],
                    ci_high[ext_mask],
                    color=c_color,
                    alpha=0.08 if use_pct else 0.07,
                    zorder=1,
                )

    # Reference clinical thresholds for percentage weight loss mode
    if use_pct:
        ax.axhline(0, color="#666666", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.axhline(-5, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.axhline(-10, color="#888888", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.axhline(-15, color="#aaaaaa", linestyle="-.", linewidth=0.7, alpha=0.5)

    metric_name = "Percentage Weight Loss" if use_pct else "Weight"
    ylabel = "Predicted Weight Loss (%)" if use_pct else "Predicted Body Weight (kg)"
    legend_loc = "lower left" if use_pct else "upper right"

    # Title & Subtitle with statistical test results
    if annotate_statistics and stats_summary is not None:
        omni = stats_summary["omnibus_trajectory"]
        p_val = omni["p_value"]
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
        stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        sub_title = (
            f"Trajectory Differences (Cluster × Time Spline): Omnibus Wald χ² = {omni['chi2']:.2f} "
            f"(df={omni['df']}), {p_str} [{stars}]"
        )
        full_title = f"LMM-Predicted {metric_name} Trajectories by Cluster {title_suffix}\n{sub_title}".strip()
    else:
        full_title = f"LMM-Predicted {metric_name} Trajectories by Cluster {title_suffix}".strip()

    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Follow-up Length (Days)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title="Behavioral Cluster", loc=legend_loc, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Informative footnote explaining confidence bands and the solid vs dashed transitions
    ci_note = "Shaded bands indicate 95% confidence intervals (±1.96 × SE). " if show_ci and cov_fe is not None else ""
    ax.text(
        0.015,
        0.025,
        f"Note: {ci_note}Solid lines represent trajectories up to cluster mean follow-up; dashed lines indicate sparse cohort extrapolation.",
        transform=ax.transAxes,
        fontsize=8.5,
        fontstyle="italic",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.90),
    )

    plt.tight_layout()

    dest_path = output_filepath
    if dest_path is None and output_dir is not None:
        dest_path = os.path.join(_resolve_path(output_dir), f"lmm_predicted_trajectories_{title_suffix.strip(' ()')}.png")

    if dest_path is not None:
        dest_path = _resolve_path(dest_path)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        fig.savefig(dest_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved LMM Predicted Trajectory Figure: '{dest_path}'")

    return fig


# =============================================================================
# PART 3: REGRESSION-BASED PROSPECTIVE OUTCOME PREDICTIONS
# =============================================================================

def fit_outcome_regressions(
    cohort_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    reference_cluster: int = 0,
    adjust_for_covariates: bool = False,
    output_db_path: Optional[str] = None,
    output_table_name: Optional[str] = "cluster_outcome_regressions",
    cluster_config_path: Optional[Union[str, Dict]] = None,
    save_to_db: bool = True,
) -> pd.DataFrame:
    """
    Fits multivariable regression models predicting key prospective clinical outcomes:
    1. total_wl_% (OLS Linear Regression)
    2. 60d_wl_% (OLS Linear Regression)
    3. total_followup_days (OLS Linear Regression)
    4. 60d_dropout (Binary Logistic Regression)

    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Patient cohort with clinical outcomes and cluster assignments.
    cluster_col : str
        Cluster assignment column name (default: 'cluster_id').
    reference_cluster : int
        Cluster to use as reference benchmark.
    adjust_for_covariates : bool
        Whether to adjust for age, sex_f, and baseline_bmi.
    output_db_path : Optional[str]
        Path to SQLite database to persist summary table.
    output_table_name : Optional[str]
        Database table name for saved results.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster title/color configuration.
    save_to_db : bool
        Whether to save results to SQLite.

    Returns:
    --------
    pd.DataFrame:
        Consolidated summary table reporting effect sizes, 95% CIs, and p-values for each outcome.
    """
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(cohort_df[cluster_col].unique())
    clusters_non_ref = [c for c in clusters if c != reference_cluster]

    ref_label = get_cluster_label(reference_cluster, cluster_config)
    cov_str = " + age + C(sex_f) + baseline_bmi" if adjust_for_covariates else ""

    print("\n" + "=" * 80)
    print("REGRESSION OUTCOME PREDICTION PER CLUSTER")
    print("=" * 80)
    print(f"Reference Cluster:  {ref_label} (ID: {reference_cluster})")
    print(f"Comparison Groups:  {[get_cluster_label(c, cluster_config) for c in clusters_non_ref]}")
    print(f"Covariate Control:  {'Adjusted (age, sex, baseline BMI)' if adjust_for_covariates else 'Unadjusted'}")
    print("=" * 80)

    # Reorder cluster categories so reference_cluster is baseline
    df_reg = cohort_df.copy()
    df_reg["cluster_cat"] = pd.Categorical(
        df_reg[cluster_col],
        categories=[reference_cluster] + clusters_non_ref,
        ordered=True,
    )

    models_spec = [
        {"var": "total_wl_%", "type": "ols", "name": "Total Weight Loss (%)"},
        {"var": "60d_wl_%", "type": "ols", "name": "60-Day Weight Loss (%)"},
        {"var": "total_followup_days", "type": "ols", "name": "Follow-Up Duration (Days)"},
        {"var": "60d_dropout", "type": "logit", "name": "60-Day Early Dropout (Binary)"},
    ]

    results_rows = []

    for spec in models_spec:
        v = spec["var"]
        if v not in df_reg.columns:
            continue

        clean_sub = df_reg.dropna(subset=[v, "cluster_cat"])
        if adjust_for_covariates:
            clean_sub = clean_sub.dropna(subset=["age", "sex_f", "baseline_bmi"])

        # Patsy formula with Q() syntax to handle variables with %, digits, or special characters
        formula = f"Q('{v}') ~ C(cluster_cat){cov_str}"
        row: Dict[str, Any] = {
            "Outcome": spec["name"],
            "Outcome_Variable": v,
            "Model_Type": "OLS Linear" if spec["type"] == "ols" else "Logistic (Logit)",
            "Sample_N": len(clean_sub),
        }

        try:
            if spec["type"] == "ols":
                fit = smf.ols(formula, data=clean_sub).fit()
                row["Omnibus_p_value"] = float(fit.f_pvalue) if hasattr(fit, "f_pvalue") and fit.f_pvalue else np.nan
                row["Fit_Metric (R2)"] = f"{fit.rsquared:.3f}"

                for cid in clusters_non_ref:
                    c_term = f"C(cluster_cat)[T.{cid}]"
                    c_lbl = get_cluster_label(cid, cluster_config)
                    if c_term in fit.params:
                        b = fit.params[c_term]
                        ci_l, ci_u = fit.conf_int().loc[c_term]
                        p = fit.pvalues[c_term]
                        row[f"{c_lbl}: Beta [95% CI]"] = f"{b:+.2f} [{ci_l:+.2f}, {ci_u:+.2f}]"
                        row[f"{c_lbl}: p-value"] = p
                    else:
                        row[f"{c_lbl}: Beta [95% CI]"] = "N/A"
                        row[f"{c_lbl}: p-value"] = np.nan
            else:
                fit = smf.logit(formula, data=clean_sub).fit(disp=False)
                row["Omnibus_p_value"] = float(fit.llr_pvalue) if hasattr(fit, "llr_pvalue") else np.nan
                row["Fit_Metric (R2)"] = f"{fit.prsquared:.3f}"

                for cid in clusters_non_ref:
                    c_term = f"C(cluster_cat)[T.{cid}]"
                    c_lbl = get_cluster_label(cid, cluster_config)
                    if c_term in fit.params:
                        b = fit.params[c_term]
                        or_val = np.exp(b)
                        ci_l = np.exp(fit.conf_int().loc[c_term][0])
                        ci_u = np.exp(fit.conf_int().loc[c_term][1])
                        p = fit.pvalues[c_term]
                        row[f"{c_lbl}: Beta [95% CI]"] = f"OR {or_val:.2f} [{ci_l:.2f}, {ci_u:.2f}]"
                        row[f"{c_lbl}: p-value"] = p
                    else:
                        row[f"{c_lbl}: Beta [95% CI]"] = "N/A"
                        row[f"{c_lbl}: p-value"] = np.nan

        except Exception as e:
            row["Omnibus_p_value"] = np.nan
            row["Fit_Metric (R2)"] = f"Error: {e}"

        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)

    # Print clean diagnostics table
    print("\nREGRESSION OUTCOME SUMMARY:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(results_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Persist to SQLite
    if save_to_db and output_db_path and output_table_name:
        out_db = _resolve_path(output_db_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
        with sqlite3.connect(out_db) as conn:
            results_df.to_sql(output_table_name, conn, if_exists="replace", index=False)
        print(f"  ✓ Saved Outcome Regressions Table: '{output_table_name}' in '{out_db}'")

    return results_df


# =============================================================================
# PART 4: SURVIVAL ANALYSIS FOR 10% WEIGHT LOSS (KAPLAN-MEIER & LOG-RANK)
# =============================================================================

def plot_cluster_kaplan_meier(
    cohort_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    output_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    max_days: Optional[int] = 365,
    cumulative_event: bool = True,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Renders publication-ready Kaplan-Meier curves assessing time to achieving
    10% weight loss across behavioral phenotype clusters.

    Parameters:
    -----------
    cohort_df : pd.DataFrame
        Cohort dataframe with '10%_wl_achieved', 'days_to_10%_wl', and 'total_followup_days'.
    cluster_col : str
        Cluster assignment column (default: 'cluster_id').
    output_filepath : Optional[str]
        Exact path to save figure PNG.
    output_dir : Optional[str]
        Directory to save figure if output_filepath is not given.
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    max_days : Optional[int]
        Censoring time horizon (default: 365 days).
    cumulative_event : bool
        If True, plots Cumulative Incidence (1 - S(t)), showing % of patients achieving 10% WL.
        If False, plots survival curve S(t) (proportion remaining without 10% WL).
    figsize : Tuple[int, int]
        Figure size dimensions.

    Returns:
    --------
    Tuple[plt.Figure, pd.DataFrame]:
        - fig: Rendered matplotlib figure.
        - summary_df: Summary metrics per cluster (events, median time to 10% WL, 60d/90d rates).
    """
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(cohort_df[cluster_col].unique())

    # 1. Prepare Survival Duration and Event Status
    df_surv = cohort_df.copy()
    event_raw = df_surv["10%_wl_achieved"].fillna(0).astype(int)
    duration_raw = np.where(event_raw == 1, df_surv["days_to_10%_wl"], df_surv["total_followup_days"])
    duration_raw = pd.Series(duration_raw, index=df_surv.index).fillna(df_surv["total_followup_days"]).fillna(1.0)

    # Apply right-censoring at max_days horizon
    if max_days is not None:
        event = np.where((duration_raw > max_days) | (event_raw == 0), 0, 1)
        duration = np.minimum(duration_raw, max_days)
    else:
        event = event_raw.values
        duration = duration_raw.values

    df_surv["duration"] = duration
    df_surv["event"] = event

    # 2. Multivariate Log-Rank Test across all clusters
    valid_mask = df_surv["duration"].notna() & df_surv["event"].notna()
    logrank_res = multivariate_logrank_test(
        df_surv.loc[valid_mask, "duration"],
        df_surv.loc[valid_mask, cluster_col],
        df_surv.loc[valid_mask, "event"],
    )
    logrank_p = logrank_res.p_value
    logrank_stat = logrank_res.test_statistic

    print("\n" + "=" * 80)
    print("SURVIVAL ANALYSIS: TIME TO 10% WEIGHT LOSS (KAPLAN-MEIER)")
    print("=" * 80)
    print(f"Cohort Size:             {len(df_surv)} patients")
    print(f"Total 10% WL Events:     {int(df_surv['event'].sum())} ({df_surv['event'].mean() * 100:.1f}%)")
    print(f"Multivariate Log-Rank:   Chi2 = {logrank_stat:.2f}, p = {logrank_p:.4f} (Omnibus across {len(clusters)} clusters)")
    print("=" * 80)

    # 3. Fit KM Curves per cluster and compile summary statistics
    fig, ax = plt.subplots(figsize=figsize)
    summary_rows = []

    for cid in clusters:
        c_lbl = get_cluster_label(cid, cluster_config)
        c_col = get_cluster_color(cid, cluster_config)
        c_mask = df_surv[cluster_col] == cid
        c_dur = df_surv.loc[c_mask, "duration"]
        c_evt = df_surv.loc[c_mask, "event"]

        kmf = KaplanMeierFitter()
        kmf.fit(c_dur, event_observed=c_evt, label=c_lbl)

        # Plot curve
        if cumulative_event:
            # 1 - S(t) = Cumulative incidence of achieving 10% weight loss
            cum_inc = (1.0 - kmf.survival_function_) * 100
            ax.step(
                cum_inc.index,
                cum_inc.iloc[:, 0],
                where="post",
                color=c_col,
                linewidth=2.5,
                label=f"{c_lbl} (Events: {c_evt.sum()}/{len(c_evt)})",
            )
        else:
            kmf.plot_survival_function(ax=ax, color=c_col, linewidth=2.5)

        # Extract survival statistics
        median_time = kmf.median_survival_time_
        median_str = f"{median_time:.0f} d" if pd.notna(median_time) and median_time < np.inf else "Not Reached"

        # 60-day and 90-day rates
        s60 = kmf.predict(60) if 60 <= c_dur.max() else np.nan
        s90 = kmf.predict(90) if 90 <= c_dur.max() else np.nan
        rate_60 = (1.0 - s60) * 100 if pd.notna(s60) else np.nan
        rate_90 = (1.0 - s90) * 100 if pd.notna(s90) else np.nan

        summary_rows.append({
            "Cluster": c_lbl,
            "Cluster_ID": cid,
            "Total_N": len(c_evt),
            "Events_Achieved": int(c_evt.sum()),
            "Event_Rate_%": f"{(c_evt.sum() / len(c_evt)) * 100:.1f}%",
            "Median_Days_to_10%_WL": median_str,
            "60d_Achieved_%": f"{rate_60:.1f}%" if pd.notna(rate_60) else "N/A",
            "90d_Achieved_%": f"{rate_90:.1f}%" if pd.notna(rate_90) else "N/A",
        })

    summary_df = pd.DataFrame(summary_rows)

    # 4. Styling & Annotations
    p_format = f"{logrank_p:.2E}" if logrank_p < 0.001 else f"{logrank_p:.3f}"
    ax.set_title(
        f"Time to Achieving 10% Weight Loss by Behavioral Phenotype Cluster\n"
        f"Log-Rank Omnibus Test: Chi2 = {logrank_stat:.2f}, p = {p_format}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Days from Baseline", fontsize=12)
    y_label = "Cumulative Probability of Achieving 10% Weight Loss (%)" if cumulative_event else "Probability of Remaining Without 10% Weight Loss"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_ylim(0, 105 if cumulative_event else 1.05)
    ax.legend(title="Behavioral Cluster", loc="lower right" if cumulative_event else "lower left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    print("\nKAPLAN-MEIER 10% WEIGHT LOSS SUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    dest_path = output_filepath
    if dest_path is None and output_dir is not None:
        dest_path = os.path.join(_resolve_path(output_dir), "kaplan_meier_10pct_wl.png")

    if dest_path is not None:
        dest_path = _resolve_path(dest_path)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        fig.savefig(dest_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved Kaplan-Meier Figure: '{dest_path}'")

    return fig, summary_df


# =============================================================================
# PART 5: END-TO-END PIPELINE WRAPPER (NOTEBOOK CALLABLE)
# =============================================================================

def run_cluster_modeling_pipeline(
    source_db_path: str = "../p1_dbs/pnk_db2_p1_in_2025.sqlite",
    labels_db_path: Optional[str] = None,
    output_db_path: Optional[str] = "../p1_dbs/pnk_db2_p1_out_2025.sqlite",
    output_dir: Optional[str] = "../p1_outputs/cluster_modeling",
    cluster_table: str = "jaccard_gower_cluster_labels",
    measurements_table: str = "measurements_p1",
    cohort_table: str = "timetoevent_eb_wgc_compl",
    distance_config: str = "wgc07_eb03",
    algorithm: str = "pam",
    k: int = 7,
    cutoff_days: int = 365,
    reference_cluster: int = 0,
    adjust_for_covariates: bool = False,
    time_degree: int = 2,
    cluster_config_path: Optional[Union[str, Dict]] = None,
    save_to_db: bool = True,
    ci_bands: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end execution pipeline conducting:
    1. Trajectory spaghetti plots (both absolute body weight kg and percentage weight loss %).
    2. Linear mixed-effects trajectory prediction (both absolute kg and zero-centered percentage %).
    3. Configurable toggle for 95% confidence bands (ci_bands).
    4. Multivariable regression outcome models (total WL, 60d WL, follow-up length, 60d dropout).
    5. Kaplan-Meier survival curves for time to 10% weight loss.

    Can be executed directly from a single notebook cell.

    Parameters:
    -----------
    source_db_path : str
        Path to database containing measurements and cohort outcomes.
    labels_db_path : Optional[str]
        Path to database containing cluster labels (defaults to source_db_path).
    output_db_path : Optional[str]
        Path to output SQLite database for persisting regression and KM summary tables.
    output_dir : Optional[str]
        Directory where figures will be saved (300 DPI PNGs).
    cluster_table : str
        Cluster labels table name.
    measurements_table : str
        Longitudinal measurements table name.
    cohort_table : str
        Cross-sectional outcomes table name.
    distance_config : str
        Dissimilarity weighting scheme (default: 'wgc07_eb03').
    algorithm : str
        Clustering algorithm ('pam' or 'hclust_complete').
    k : int
        Number of clusters (default: 7).
    cutoff_days : int
        Time cutoff horizon for trajectory analysis (default: 365 days).
    reference_cluster : int
        Reference cluster index for regressions.
    adjust_for_covariates : bool
        Whether to adjust models for age, sex_f, and baseline_bmi.
    time_degree : int
        Polynomial degree for spline time curves (default: 2, quadratic).
    cluster_config_path : Optional[Union[str, Dict]]
        Cluster titles/colors configuration.
    save_to_db : bool
        Whether to save output tables to SQLite.
    ci_bands : bool
        Whether to render 95% confidence interval bands on LMM trajectory curves (default: True).

    Returns:
    --------
    Dict[str, Any]:
        Dictionary containing trajectory dataframes, models, figures, and summary tables.
    """
    if labels_db_path is None:
        labels_db_path = source_db_path

    out_dir = _resolve_path(output_dir) if output_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    clean_dist = distance_config.replace("hybrid_", "")
    prefix = f"{algorithm}_{clean_dist}_k{k}"

    print("\n" + "#" * 80)
    print(f"STARTING CLUSTER-BASED TRAJECTORY & OUTCOME MODELING PIPELINE: {prefix.upper()}")
    print("#" * 80)

    # 1. Load Data
    trajectory_df, cohort_df = load_and_prepare_trajectory_data(
        cluster_db_path=labels_db_path,
        measurements_db_path=source_db_path,
        cluster_table=cluster_table,
        measurements_table=measurements_table,
        cohort_table=cohort_table,
        weight=distance_config,
        algorithm=algorithm,
        k=k,
        cutoff_days=cutoff_days,
    )

    # 2. Descriptive Spaghetti Plots (Absolute kg & Percentage %)
    fig_spaghetti_kg = plot_cluster_spaghetti(
        trajectory_df=trajectory_df,
        output_filepath=os.path.join(out_dir, f"spaghetti_kg_{prefix}_{cutoff_days}d.png") if out_dir else None,
        cluster_config_path=cluster_config_path,
        title_suffix=f"({cutoff_days}-Day Cutoff, Absolute kg)",
        use_pct=False,
    )
    fig_spaghetti_pct = plot_cluster_spaghetti(
        trajectory_df=trajectory_df,
        output_filepath=os.path.join(out_dir, f"spaghetti_pct_{prefix}_{cutoff_days}d.png") if out_dir else None,
        cluster_config_path=cluster_config_path,
        title_suffix=f"({cutoff_days}-Day Cutoff, % Weight Loss)",
        use_pct=True,
    )
    # Save standard path for backward compatibility
    if out_dir:
        plot_cluster_spaghetti(
            trajectory_df=trajectory_df,
            output_filepath=os.path.join(out_dir, f"spaghetti_{prefix}_{cutoff_days}d.png"),
            cluster_config_path=cluster_config_path,
            title_suffix=f"({cutoff_days}-Day Cutoff)",
            use_pct=True,
        )

    # 2b. Cluster-Based Nadir & Weight Regain Metrics Comparison
    nadir_summary_df, patient_nadir_df = compute_nadir_weight_loss_metrics(
        trajectory_df=trajectory_df,
        cluster_col="cluster_id",
        cluster_config_path=cluster_config_path,
    )

    # 3. Linear Mixed-Effects Models (LMM)
    # 3a. Absolute Body Weight (kg)
    lmm_result_kg, df_model_kg = fit_trajectory_mixed_model(
        trajectory_df=trajectory_df,
        body_weight_col="weight_kg",
        use_deviation_coding=True,
        adjust_for_covariates=adjust_for_covariates,
        time_degree=time_degree,
    )
    trajectory_analysis_kg = analyze_trajectory_mixed_model(
        lmm_result=lmm_result_kg,
        df_model=df_model_kg,
        use_deviation_coding=True,
        patient_id_col="medical_record_id",
        cluster_config_path=cluster_config_path,
        body_weight_col="weight_kg",
    )
    fig_lmm_kg = plot_lmm_predicted_trajectories(
        lmm_result=lmm_result_kg,
        df_model=df_model_kg,
        output_filepath=os.path.join(out_dir, f"lmm_predicted_kg_{prefix}.png") if out_dir else None,
        cluster_config_path=cluster_config_path,
        title_suffix=f"({prefix}, {'Adjusted' if adjust_for_covariates else 'Unadjusted'}, Absolute kg)",
        stats_summary=trajectory_analysis_kg,
        dashing_cutoff="cluster_mean",
        show_ci=ci_bands,
        use_pct=False,
        patient_id_col="medical_record_id",
    )

    # 3b. Percentage Weight Loss (%) - Centered at Zero
    lmm_result_pct, df_model_pct = fit_trajectory_mixed_model(
        trajectory_df=trajectory_df,
        body_weight_col="weight_loss_pct",
        use_deviation_coding=True,
        adjust_for_covariates=adjust_for_covariates,
        time_degree=time_degree,
    )
    trajectory_analysis_pct = analyze_trajectory_mixed_model(
        lmm_result=lmm_result_pct,
        df_model=df_model_pct,
        use_deviation_coding=True,
        patient_id_col="medical_record_id",
        cluster_config_path=cluster_config_path,
        body_weight_col="weight_loss_pct",
    )
    fig_lmm_pct = plot_lmm_predicted_trajectories(
        lmm_result=lmm_result_pct,
        df_model=df_model_pct,
        output_filepath=os.path.join(out_dir, f"lmm_predicted_pct_{prefix}.png") if out_dir else None,
        cluster_config_path=cluster_config_path,
        title_suffix=f"({prefix}, {'Adjusted' if adjust_for_covariates else 'Unadjusted'}, % Weight Loss)",
        stats_summary=trajectory_analysis_pct,
        dashing_cutoff="cluster_mean",
        show_ci=ci_bands,
        use_pct=True,
        patient_id_col="medical_record_id",
    )
    # Save standard path for backward compatibility
    if out_dir:
        plot_lmm_predicted_trajectories(
            lmm_result=lmm_result_pct,
            df_model=df_model_pct,
            output_filepath=os.path.join(out_dir, f"lmm_predicted_{prefix}.png"),
            cluster_config_path=cluster_config_path,
            title_suffix=f"({prefix}, {'Adjusted' if adjust_for_covariates else 'Unadjusted'})",
            stats_summary=trajectory_analysis_pct,
            dashing_cutoff="cluster_mean",
            show_ci=ci_bands,
            use_pct=True,
            patient_id_col="medical_record_id",
        )

    # Save trajectory statistical test results to SQLite
    if save_to_db and output_db_path:
        try:
            with sqlite3.connect(_resolve_path(output_db_path)) as conn:
                trajectory_analysis_kg["cluster_contrasts_df"].to_sql(
                    f"{prefix}_trajectory_lmm_tests_kg", conn, if_exists="replace", index=False
                )
                trajectory_analysis_pct["cluster_contrasts_df"].to_sql(
                    f"{prefix}_trajectory_lmm_tests_pct", conn, if_exists="replace", index=False
                )
                trajectory_analysis_pct["cluster_contrasts_df"].to_sql(
                    f"{prefix}_trajectory_lmm_tests", conn, if_exists="replace", index=False
                )
            print(f"  ✓ Saved LMM Trajectory Tests to DB: '{prefix}_trajectory_lmm_tests_kg' & '{prefix}_trajectory_lmm_tests_pct'")
        except Exception as e:
            print(f"Warning: Failed to save trajectory test table to database: {e}")

    # 4. Regression Outcome Models
    reg_table_name = f"{prefix}_outcome_regressions"
    reg_df = fit_outcome_regressions(
        cohort_df=cohort_df,
        cluster_col="cluster_id",
        reference_cluster=reference_cluster,
        adjust_for_covariates=adjust_for_covariates,
        output_db_path=output_db_path,
        output_table_name=reg_table_name,
        cluster_config_path=cluster_config_path,
        save_to_db=save_to_db,
    )

    # 5. Kaplan-Meier Survival Analysis for 10% Weight Loss
    fig_km, km_summary_df = plot_cluster_kaplan_meier(
        cohort_df=cohort_df,
        cluster_col="cluster_id",
        output_filepath=os.path.join(out_dir, f"kaplan_meier_10pct_wl_{prefix}.png") if out_dir else None,
        cluster_config_path=cluster_config_path,
        max_days=cutoff_days,
        cumulative_event=True,
    )

    # Save KM summary table to SQLite
    if save_to_db and output_db_path:
        km_table_name = f"{prefix}_km_10pct_wl_summary"
        with sqlite3.connect(_resolve_path(output_db_path)) as conn:
            km_summary_df.to_sql(km_table_name, conn, if_exists="replace", index=False)
        print(f"  ✓ Saved Kaplan-Meier Summary Table: '{km_table_name}' in '{output_db_path}'")

    print("\n" + "=" * 80)
    print("CLUSTER MODELING PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    return {
        "trajectory_df": trajectory_df,
        "cohort_df": cohort_df,
        "lmm_result_kg": lmm_result_kg,
        "lmm_result_pct": lmm_result_pct,
        "lmm_result": lmm_result_pct,
        "trajectory_analysis_kg": trajectory_analysis_kg,
        "trajectory_analysis_pct": trajectory_analysis_pct,
        "trajectory_statistical_analysis": trajectory_analysis_pct,
        "trajectory_tests_df": trajectory_analysis_pct["cluster_contrasts_df"],
        "nadir_metrics_df": nadir_summary_df,
        "patient_nadir_df": patient_nadir_df,
        "regression_results_df": reg_df,
        "km_summary_df": km_summary_df,
        "figures": {
            "spaghetti_kg": fig_spaghetti_kg,
            "spaghetti_pct": fig_spaghetti_pct,
            "spaghetti": fig_spaghetti_pct,
            "lmm_predicted_kg": fig_lmm_kg,
            "lmm_predicted_pct": fig_lmm_pct,
            "lmm_predicted": fig_lmm_pct,
            "kaplan_meier": fig_km,
        },
    }


# =============================================================================
# CLI TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_in_2025.sqlite")
    out_db = os.path.join(base_dir, "p1_dbs", "pnk_db2_p1_out_2025.sqlite")
    out_dir = os.path.join(base_dir, "p1_outputs", "cluster_modeling_test")

    results = run_cluster_modeling_pipeline(
        source_db_path=in_db,
        labels_db_path=in_db,
        output_db_path=out_db,
        output_dir=out_dir,
        distance_config="wgc07_eb03",
        algorithm="pam",
        k=7,
        cutoff_days=365,
        save_to_db=True,
    )
