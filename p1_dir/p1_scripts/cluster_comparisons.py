"""
================================================================================
REVISED CLUSTER COMPARISON & CLINICAL PHENOTYPING MODULE
================================================================================

Research Context & Scientific Task:
-----------------------------------
This module performs comprehensive, multi-domain statistical profiling and clinical
visualization of patient clusters derived from PAM / k-medoids clustering.

It is fully configurable across:
1. Any distance matrix weighting configuration (e.g. 'wgc05_eb05', 'wgc07_eb03', 'wgc06_eb04').
2. Any cluster count k (e.g. k=2, 3, 4, 5, 6, 7, 8).

Clinical Evaluation Domains:
----------------------------
1. Baseline Demographics & Anthropometry: Age, sex, baseline weight, BMI, body fat %, muscle %.
2. Etiological Triggers (12 WGC): Free-text NLP weight gain causes (asymmetric binary).
3. Appetite & Eating Behaviors (6 EB): Binary screening symptoms and 1-10 Likert self-regulation scales.
4. Prospective Clinical Outcomes: Total & 60-day weight loss %, BMI reduction, fat loss %,
   milestone achievements (5% and 10% WL), and 60-day program attrition (dropout).

Statistical Rigor:
------------------
- Continuous Variables: Tested for normality using the Shapiro-Wilk test. If data is normally
  distributed across groups, One-Way ANOVA is used; if skewed, the non-parametric Kruskal-Wallis test is used.
- Categorical / Binary Variables: Evaluated via Pearson's Chi-Square contingency test, with automatic
  fallback to Fisher's Exact Test whenever contingency tables contain sparse expected counts (< 5).
- Exemplar Medoids: Pulls actual patient medical record identifiers directly from the database,
  anchoring each cluster profile to a real individual.
================================================================================
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import f_oneway, kruskal, chi2_contingency, fisher_exact, shapiro

# Ensure stdout encodes UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# CLINICAL VARIABLE CATALOGUE
# =============================================================================

CLINICAL_VARIABLE_DOMAINS: Dict[str, List[str]] = {
    "Demographics & Anthropometrics": [
        "age",
        "sex_f",
        "baseline_weight_kg",
        "baseline_bmi",
        "baseline_fat_%",
        "baseline_muscle_%",
    ],
    "Eating Behaviors (Symptoms & Scales)": [
        "hunger_yn",
        "satiety_yn",
        "emotional_eating_yn",
        "emotional_eating_value_likert",
        "quantity_control_likert",
        "impulse_control_likert",
    ],
    "Weight Gain Causes (Etiological Narratives)": [
        "eating_habits",
        "mental_health",
        "physical_inactivity",
        "medication_disease_injury",
        "schedule",
        "family_issues",
        "lifestyle_circumstances",
        "womens_health_and_pregnancy",
        "smoking_cessation",
        "treatment_discontinuation_or_relapse",
        "pandemic",
        "none_of_above",
    ],
    "Prospective Clinical Outcomes": [
        "total_followup_days",
        "total_wl_%",
        "60d_wl_%",
        "60d_bmi_reduction",
        "total_fat_loss_%",
        "60d_fat_loss_%",
        "5%_wl_achieved",
        "10%_wl_achieved",
        "60d_dropout",
    ],
}

# Clean display names for publication tables
VARIABLE_LABELS: Dict[str, str] = {
    "age": "Age (years)",
    "sex_f": "Female sex, n (%)",
    "baseline_weight_kg": "Baseline Weight (kg)",
    "baseline_bmi": "Baseline BMI (kg/m²)",
    "baseline_fat_%": "Baseline Body Fat (%)",
    "baseline_muscle_%": "Baseline Muscle (%)",
    "hunger_yn": "Constant Hunger, n (%)",
    "satiety_yn": "Impaired Satiety, n (%)",
    "emotional_eating_yn": "Emotional Eating Present, n (%)",
    "emotional_eating_value_likert": "Emotional Eating Severity (1-10)",
    "quantity_control_likert": "Quantity Control Ability (1-10)",
    "impulse_control_likert": "Impulse Control Ability (1-10)",
    "eating_habits": "Cause: Eating Habits, n (%)",
    "mental_health": "Cause: Mental Health, n (%)",
    "physical_inactivity": "Cause: Physical Inactivity, n (%)",
    "medication_disease_injury": "Cause: Medical/Injury, n (%)",
    "schedule": "Cause: Schedule/Work, n (%)",
    "family_issues": "Cause: Family Issues, n (%)",
    "lifestyle_circumstances": "Cause: Lifestyle Changes, n (%)",
    "womens_health_and_pregnancy": "Cause: Pregnancy/Women's Health, n (%)",
    "smoking_cessation": "Cause: Smoking Cessation, n (%)",
    "treatment_discontinuation_or_relapse": "Cause: Prior Treatment Relapse, n (%)",
    "pandemic": "Cause: Pandemic, n (%)",
    "none_of_above": "Cause: None of Above (Idiopathic), n (%)",
    "total_followup_days": "Follow-up Duration (days)",
    "total_wl_%": "Total Weight Loss (%)",
    "60d_wl_%": "60-Day Weight Loss (%)",
    "60d_bmi_reduction": "60-Day BMI Reduction (kg/m²)",
    "total_fat_loss_%": "Total Fat Mass Loss (%)",
    "60d_fat_loss_%": "60-Day Fat Mass Loss (%)",
    "5%_wl_achieved": "5% Weight Loss Achieved, n (%)",
    "10%_wl_achieved": "10% Weight Loss Achieved, n (%)",
    "60d_dropout": "60-Day Program Attrition, n (%)",
}


# =============================================================================
# DATA INGESTION & CLUSTER MERGING
# =============================================================================

def load_merged_cluster_data(
    source_db_path: str,
    labels_db_path: str,
    distance_config: str = "wgc05_eb05",
    k: int = 3,
    source_table: str = "timetoevent_eb_wgc_compl",
    labels_table: str = "pam_labels",
) -> Tuple[pd.DataFrame, str, Dict[int, str]]:
    """
    Loads clinical patient data and merges with PAM cluster assignments and exemplar medoids.

    Parameters:
    -----------
    source_db_path : str
        Path to SQLite database containing source patient clinical data.
    labels_db_path : str
        Path to SQLite database containing PAM cluster labels.
    distance_config : str
        Distance configuration key (e.g. 'wgc05_eb05', 'wgc07_eb03', 'wgc06_eb04').
    k : int
        Number of clusters to extract (e.g. 3, 5, 6).
    source_table : str
        Source patient data table name.
    labels_table : str
        PAM labels table name.

    Returns:
    --------
    Tuple[pd.DataFrame, str, Dict[int, str]]:
        (merged_df, cluster_col_name, medoid_map)
        merged_df: Patient DataFrame with attached cluster labels.
        cluster_col_name: Name of the active cluster column (e.g. 'wgc05_eb05_k3').
        medoid_map: Dict mapping cluster ID -> exemplar patient medical_record_id.
    """
    print(f"\n[Data Loading] Loading clinical cohort from '{source_table}' in '{source_db_path}'...")
    with sqlite3.connect(source_db_path) as conn:
        patient_df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

    print(f"[Data Loading] Loading PAM cluster labels from '{labels_table}' in '{labels_db_path}'...")
    with sqlite3.connect(labels_db_path) as conn:
        labels_df = pd.read_sql_query(f"SELECT * FROM {labels_table}", conn)

    cluster_col = f"{distance_config}_k{k}"
    medoid_col = f"{distance_config}_medoid_k{k}"

    if cluster_col not in labels_df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in '{labels_table}'. Available: {labels_df.columns.tolist()}")

    # Merge on medical_record_id
    merged_df = patient_df.merge(
        labels_df[["medical_record_id", cluster_col, medoid_col]],
        on="medical_record_id",
        how="inner",
    )

    # Extract exemplar medoid patient IDs
    medoid_map: Dict[int, str] = {}
    for c_id in sorted(merged_df[cluster_col].unique()):
        c_sub = merged_df[merged_df[cluster_col] == c_id]
        medoid_rows = c_sub[c_sub[medoid_col] == 1]
        if len(medoid_rows) > 0:
            medoid_map[c_id] = medoid_rows["medical_record_id"].iloc[0]
        else:
            medoid_map[c_id] = "Unknown"

    print(f"[Data Loading] Merged {len(merged_df)} patients. Clusters for k={k}: {sorted(merged_df[cluster_col].unique())}")
    print(f"[Diagnostics] Exemplar Medoid Patients by Cluster:")
    for c_id, m_id in medoid_map.items():
        n_c = (merged_df[cluster_col] == c_id).sum()
        pct_c = (n_c / len(merged_df)) * 100
        print(f"  - Cluster {c_id}: n={n_c} ({pct_c:.1f}%) | Medoid Record ID: {m_id}")

    return merged_df, cluster_col, medoid_map


# =============================================================================
# STATISTICAL HYPOTHESIS TESTING & FORMATTING
# =============================================================================

def format_continuous_stat(series: pd.Series, decimals: int = 1) -> str:
    """Formats Mean ± SD for continuous variables."""
    valid = series.dropna()
    if len(valid) == 0:
        return "N/A"
    return f"{valid.mean():.{decimals}f} ± {valid.std():.{decimals}f}"


def format_categorical_stat(series: pd.Series) -> str:
    """Formats n (%) for binary/categorical variables."""
    valid = series.dropna()
    if len(valid) == 0:
        return "N/A"
    pos = int(valid.sum())
    pct = (pos / len(valid)) * 100.0
    return f"{pos} ({pct:.1f}%)"


def test_continuous_variable(groups: List[pd.Series]) -> Tuple[float, str]:
    """
    Performs One-Way ANOVA if data across groups is normally distributed;
    otherwise falls back to non-parametric Kruskal-Wallis test.
    """
    valid_groups = [g.dropna() for g in groups if len(g.dropna()) > 0]
    if len(valid_groups) < 2:
        return np.nan, "Insufficient Data"

    # Test normality with Shapiro-Wilk
    normal_flags = []
    for g in valid_groups:
        if len(g) >= 3:
            try:
                _, p_norm = shapiro(g)
                normal_flags.append(p_norm > 0.05)
            except Exception:
                normal_flags.append(False)

    use_anova = len(normal_flags) > 0 and sum(normal_flags) >= 0.5 * len(normal_flags)

    try:
        if use_anova:
            _, p_val = f_oneway(*valid_groups)
            test_name = "One-Way ANOVA"
        else:
            _, p_val = kruskal(*valid_groups)
            test_name = "Kruskal-Wallis"
        return float(p_val), test_name
    except Exception:
        return np.nan, "Error"


def test_categorical_variable(groups: List[pd.Series]) -> Tuple[float, str]:
    """
    Performs Chi-Square test with automatic fallback to Fisher's Exact test for sparse 2x2 tables.
    """
    # Build contingency table: rows = clusters, cols = [Positive, Negative]
    table = []
    for g in groups:
        valid = g.dropna()
        pos = int(valid.sum())
        neg = len(valid) - pos
        table.append([pos, neg])
    
    table_arr = np.array(table)

    # Check for no variation
    if np.all(table_arr[:, 0] == 0) or np.all(table_arr[:, 1] == 0):
        return 1.0, "No Variation"

    try:
        chi2, p_val, dof, expected = chi2_contingency(table_arr)
        test_name = "Chi-Square"

        # Check if table is 2x2 and sparse -> use Fisher's exact
        if table_arr.shape == (2, 2) and (np.any(expected < 5.0)):
            _, p_val = fisher_exact(table_arr)
            test_name = "Fisher's Exact"

        return float(p_val), test_name
    except Exception:
        return np.nan, "Error"


# =============================================================================
# COMPARATIVE TABLE GENERATOR
# =============================================================================

def generate_cluster_comparison_table(
    df: pd.DataFrame,
    cluster_col: str,
    medoid_map: Optional[Dict[int, str]] = None,
    domains: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Builds a comprehensive, publication-grade clinical comparison table across clusters.

    Parameters:
    -----------
    df : pd.DataFrame
        Merged DataFrame containing clinical features and cluster labels.
    cluster_col : str
        Column name specifying cluster assignments.
    medoid_map : Optional[Dict[int, str]]
        Mapping of cluster IDs to exemplar patient medical_record_ids.
    domains : Optional[Dict[str, List[str]]]
        Dictionary defining clinical domains and variable lists.

    Returns:
    --------
    pd.DataFrame
        Publication-formatted comparison table.
    """
    if domains is None:
        domains = CLINICAL_VARIABLE_DOMAINS

    cluster_ids = sorted(df[cluster_col].unique())
    n_total = len(df)

    # Prepare column header names with sample sizes and exemplar medoids
    cluster_headers = {}
    for c_id in cluster_ids:
        n_c = (df[cluster_col] == c_id).sum()
        pct_c = (n_c / n_total) * 100.0
        medoid_str = f" [Medoid: {medoid_map[c_id]}]" if (medoid_map and c_id in medoid_map) else ""
        cluster_headers[c_id] = f"Cluster {c_id} (n={n_c}, {pct_c:.1f}%){medoid_str}"

    rows: List[Dict] = []

    # Row 0: Sample Size Header Row
    size_row = {
        "Domain": "Sample Size",
        "Clinical Variable": "N (Patients)",
        f"Overall Cohort (N={n_total})": f"{n_total} (100.0%)",
    }
    for c_id in cluster_ids:
        n_c = (df[cluster_col] == c_id).sum()
        pct_c = (n_c / n_total) * 100.0
        size_row[cluster_headers[c_id]] = f"{n_c} ({pct_c:.1f}%)"
    size_row["Statistical Test"] = "N/A"
    size_row["p-value"] = "N/A"
    size_row["p_numeric"] = np.nan
    size_row["Significant (p<0.05)"] = "N/A"
    rows.append(size_row)

    for domain_name, var_list in domains.items():
        for var in var_list:
            if var not in df.columns:
                continue

            var_label = VARIABLE_LABELS.get(var, var)
            series_total = df[var]

            # Determine whether variable is binary or continuous
            unique_vals = set(series_total.dropna().unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0, True, False})

            # Format total cohort statistic
            if is_binary:
                total_stat = format_categorical_stat(series_total)
            else:
                total_stat = format_continuous_stat(series_total)

            # Extract per-cluster groups and statistics
            cluster_stats = {}
            groups_data = []
            for c_id in cluster_ids:
                c_series = df[df[cluster_col] == c_id][var]
                groups_data.append(c_series)
                if is_binary:
                    cluster_stats[cluster_headers[c_id]] = format_categorical_stat(c_series)
                else:
                    cluster_stats[cluster_headers[c_id]] = format_continuous_stat(c_series)

            # Hypothesis Testing
            if is_binary:
                p_val, test_name = test_categorical_variable(groups_data)
            else:
                p_val, test_name = test_continuous_variable(groups_data)

            # Format p-value string
            if np.isnan(p_val):
                p_str = "N/A"
                sig_flag = "No"
            elif p_val < 0.001:
                p_str = "<0.001"
                sig_flag = "Yes (p<0.001)"
            else:
                p_str = f"{p_val:.3f}"
                sig_flag = "Yes" if p_val < 0.05 else "No"

            row_dict = {
                "Domain": domain_name,
                "Clinical Variable": var_label,
                f"Overall Cohort (N={n_total})": total_stat,
            }
            row_dict.update(cluster_stats)
            row_dict["Statistical Test"] = test_name
            row_dict["p-value"] = p_str
            row_dict["p_numeric"] = float(p_val) if not np.isnan(p_val) else np.nan
            row_dict["Significant (p<0.05)"] = sig_flag

            rows.append(row_dict)

    result_df = pd.DataFrame(rows)
    return result_df


# =============================================================================
# MULTI-PANEL CLINICAL VISUALIZATIONS
# =============================================================================

def plot_cluster_comparisons(
    df: pd.DataFrame,
    cluster_col: str,
    medoid_map: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Creates a rich, publication-grade multi-panel figure illustrating cluster differences across:
    1. Cluster size distribution.
    2. Baseline demographics (Age & Baseline BMI).
    3. Eating behavior self-regulation scales (Radar / Profile plot).
    4. Prospective weight loss & attrition (60-day Weight Loss % & Dropout %).

    Parameters:
    -----------
    df : pd.DataFrame
        Merged patient DataFrame.
    cluster_col : str
        Cluster label column name.
    medoid_map : Optional[Dict[int, str]]
        Exemplar patient medoids mapping.
    save_path : Optional[str]
        Optional file path to save the generated PNG.
    """
    cluster_ids = sorted(df[cluster_col].unique())
    k = len(cluster_ids)
    palette = sns.color_palette("Set2", n_colors=k)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # -------------------------------------------------------------------------
    # Panel 1: Cluster Sample Size Distribution
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df[cluster_col].value_counts().sort_index()
    pcts = (counts / len(df)) * 100.0

    bars = ax1.bar(
        [f"C{c}" for c in counts.index],
        counts.values,
        color=palette,
        edgecolor="black",
        alpha=0.85,
    )
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 2,
            f"{int(yval)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_title("A. Cluster Size Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cluster", fontsize=10)
    ax1.set_ylabel("Number of Patients", fontsize=10)
    ax1.set_ylim(0, max(counts.values) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 2: Baseline Age Distribution
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(
        data=df,
        x=cluster_col,
        y="age",
        hue=cluster_col,
        legend=False,
        palette=palette,
        ax=ax2,
        width=0.5,
        boxprops=dict(alpha=0.85),
    )
    ax2.set_title("B. Age by Cluster", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cluster", fontsize=10)
    ax2.set_ylabel("Age (years)", fontsize=10)
    ax2.set_xticks(range(len(cluster_ids)))
    ax2.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 3: Baseline BMI Distribution
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(
        data=df,
        x=cluster_col,
        y="baseline_bmi",
        hue=cluster_col,
        legend=False,
        palette=palette,
        ax=ax3,
        width=0.5,
        boxprops=dict(alpha=0.85),
    )
    ax3.set_title("C. Baseline BMI by Cluster", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Cluster", fontsize=10)
    ax3.set_ylabel("BMI (kg/m²)", fontsize=10)
    ax3.set_xticks(range(len(cluster_ids)))
    ax3.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 4: Eating Behavior Likert Scales (Appetite Phenotype Profile)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0:2])
    eb_scales = [
        ("emotional_eating_value_likert", "Emotional Eating"),
        ("quantity_control_likert", "Quantity Control"),
        ("impulse_control_likert", "Impulse Control"),
    ]

    plot_data = []
    for var, clean_lbl in eb_scales:
        for c_id in cluster_ids:
            c_mean = df[df[cluster_col] == c_id][var].mean()
            c_sd = df[df[cluster_col] == c_id][var].std()
            plot_data.append({
                "Scale": clean_lbl,
                "Cluster": f"Cluster {c_id}",
                "Mean Score": c_mean,
                "SD": c_sd,
            })

    eb_df = pd.DataFrame(plot_data)
    sns.barplot(
        data=eb_df,
        x="Scale",
        y="Mean Score",
        hue="Cluster",
        palette=palette,
        ax=ax4,
        edgecolor="black",
        alpha=0.85,
    )
    ax4.set_title("D. Eating Behavior Self-Regulation Profiles (1-10 Likert)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Behavioral Domain", fontsize=10)
    ax4.set_ylabel("Mean Likert Score", fontsize=10)
    ax4.set_ylim(0, 10.5)
    ax4.legend(title="Cluster", loc="upper right", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel 5: Prospective Outcomes (60-Day Weight Loss %)
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 2])
    sns.boxplot(
        data=df,
        x=cluster_col,
        y="60d_wl_%",
        hue=cluster_col,
        legend=False,
        palette=palette,
        ax=ax5,
        width=0.5,
        boxprops=dict(alpha=0.85),
    )
    ax5.axhline(5.0, color="darkgreen", linestyle=":", label="5% WL Target")
    ax5.axhline(10.0, color="blue", linestyle="--", label="10% WL Target")
    ax5.set_title("E. 60-Day Weight Loss (%)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Cluster", fontsize=10)
    ax5.set_ylabel("Weight Loss (%)", fontsize=10)
    ax5.set_xticks(range(len(cluster_ids)))
    ax5.set_xticklabels([f"C{c}" for c in cluster_ids])
    ax5.legend(loc="lower left", fontsize=8)
    ax5.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(
        f"Multi-Domain Clinical Phenotyping: PAM Clustering ({cluster_col})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plotting] Clinical comparison figure saved to: {save_path}")

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
# MASTER COMPARISON DRIVER
# =============================================================================

def display_clinically_structured_table(comparison_table: pd.DataFrame) -> None:
    """
    Renders the complete, clinically structured comparison table with full column expansion
    and domain separators, supporting both Jupyter interactive HTML display and clean console output.
    """
    # Configure pandas display to prevent truncation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 60)

    print("\n" + "=" * 120)
    print("CLINICALLY STRUCTURED COMPARISON TABLE (Full Cohort vs. PAM Clusters)")
    print("=" * 120)

    # Format output by clinical domain with dividers
    domains = comparison_table["Domain"].unique()
    for d in domains:
        d_sub = comparison_table[comparison_table["Domain"] == d]
        # Drop the Domain column for cleaner printing since it is the section header
        print_cols = [c for c in d_sub.columns if c not in ["Domain", "p_numeric"]]
        print(f"\n--- {d.upper()} ---")
        print(d_sub[print_cols].to_string(index=False))

    print("\n" + "=" * 120)

    # Interactive Jupyter HTML display
    try:
        from IPython.display import display
        display(comparison_table)
    except Exception:
        pass


def run_cluster_comparison_pipeline(
    source_db_path: str,
    labels_db_path: str,
    distance_config: str = "wgc05_eb05",
    k: int = 3,
    save_to_db: bool = True,
    db_table_name: Optional[str] = None,
    output_excel_path: Optional[str] = None,
    output_fig_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main driver function to execute clinical cluster comparisons, save the resulting
    table to SQLite, and render the structured clinical comparison table.

    Parameters:
    -----------
    source_db_path : str
        Path to source database (pnk_db2_p1_in.sqlite).
    labels_db_path : str
        Path to cluster labels database (pnk_db2_p1_out.sqlite).
    distance_config : str
        Distance configuration key (default: 'wgc05_eb05').
    k : int
        Number of clusters to evaluate (default: 3).
    save_to_db : bool
        Whether to save the comparative table into SQLite (default: True).
    db_table_name : Optional[str]
        Custom SQLite table name. Defaults to 'pam_comparison_{distance_config}_k{k}'.
    output_excel_path : Optional[str]
        Optional path to export comparative table as an Excel spreadsheet.
    output_fig_path : Optional[str]
        Optional path to save multi-panel comparison figure.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]:
        (comparison_table_df, merged_patient_df)
    """
    print("\n" + "#" * 80)
    print(f"RUNNING CLINICAL CLUSTER COMPARISON: Config '{distance_config}' | k={k}")
    print("#" * 80)

    # 1. Load and merge patient data with cluster labels
    df_merged, cluster_col, medoid_map = load_merged_cluster_data(
        source_db_path=source_db_path,
        labels_db_path=labels_db_path,
        distance_config=distance_config,
        k=k,
    )

    # 2. Generate comprehensive statistical comparison table
    print("\n[Statistical Analysis] Computing hypothesis tests across all clinical domains...")
    comparison_table = generate_cluster_comparison_table(
        df=df_merged,
        cluster_col=cluster_col,
        medoid_map=medoid_map,
    )

    # 3. Save comparative table to SQLite database
    if save_to_db:
        if db_table_name is None:
            db_table_name = f"pam_comparison_{distance_config}_k{k}"

        with sqlite3.connect(labels_db_path) as conn:
            comparison_table.to_sql(db_table_name, conn, if_exists="replace", index=False)
        print(f"[Persistence] Saved comparative table to database '{labels_db_path}' as table '{db_table_name}' ({len(comparison_table)} rows).")

    # 4. Display the complete, clinically structured table
    display_clinically_structured_table(comparison_table)

    # 5. Create visualizations
    print("\n[Visualization] Generating multi-panel clinical profiling plots...")
    plot_cluster_comparisons(
        df=df_merged,
        cluster_col=cluster_col,
        medoid_map=medoid_map,
        save_path=output_fig_path,
    )

    # 6. Export table if requested
    if output_excel_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)
        comparison_table.to_excel(output_excel_path, index=False)
        print(f"[Export] Saved comparison table to Excel: {output_excel_path}")

    # 7. Print summary of significant differences (p < 0.05)
    sig_vars = comparison_table[
        (comparison_table["Significant (p<0.05)"].str.startswith("Yes")) &
        (comparison_table["Domain"] != "Sample Size")
    ]
    print("\n" + "=" * 80)
    print(f"SUMMARY OF SIGNIFICANT CLINICAL DIFFERENCES (p < 0.05): {len(sig_vars)} / {len(comparison_table) - 1} variables")
    print("=" * 80)
    for _, row in sig_vars.iterrows():
        print(f"  • {row['Domain']} | {row['Clinical Variable']}: p = {row['p-value']} ({row['Statistical Test']})")
    print("=" * 80 + "\n")

    return comparison_table, df_merged


# =============================================================================
# STANDALONE EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "..", ".."))

    in_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_in.sqlite")
    out_db = os.path.join(project_root, "p1_dir", "p1_dbs", "pnk_db2_p1_out.sqlite")
    fig_path = os.path.join(project_root, "p1_dir", "p1_outputs", "cluster_clinical_comparison_k3.png")

    comp_table, merged_data = run_cluster_comparison_pipeline(
        source_db_path=in_db,
        labels_db_path=out_db,
        distance_config="wgc05_eb05",
        k=3,
        save_to_db=True,
        output_fig_path=fig_path,
    )
