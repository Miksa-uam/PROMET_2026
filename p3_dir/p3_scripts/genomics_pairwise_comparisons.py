"""
PAIRWISE COMPARISONS - BOTH TABLE AND VISUAL - AS OF 21MAY26, UNUSED IN THE P3 GENOMICS 'YES/NO' STUDY AND NOT MAINTAINED WELL
"""


# =============================================================================
# TABULAR COMPARISONS
# =============================================================================

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

from genomics_config import descriptive_comparisons_config, master_config
from fdr_correction_utils import collect_pvalues_from_dataframe, apply_fdr_correction, integrate_corrected_pvalues


# =========================
# 1. HELPER FUNCTIONS
# =========================

def format_mean_sd(series):
    """Formats a series into 'mean ± SD' string, handling non-numeric data."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    mean = series.mean()
    sd = series.std()
    return f"{mean:.2f} \u00B1 {sd:.2f}"

def format_n_perc(series):
    """Formats a binary categorical series into 'N (%)' for the positive class (1)."""
    series = series.dropna()
    if series.empty:
        return "0 (0.0%)"
    n_positive = series.sum()
    n_total = len(series)
    perc = (n_positive / n_total) * 100 if n_total > 0 else 0
    return f"{int(n_positive)} ({perc:.1f}%)"

def format_median_iqr(series):
    """Formats a series into 'median [IQR]' string."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.2f} [{q1:.2f}–{q3:.2f}]"

def format_availability(series):
    """Counts non-null values and formats as 'N (%)'."""
    total_count = len(series)
    available_count = series.notna().sum()
    perc = (available_count / total_count) * 100 if total_count > 0 else 0
    return f"{available_count} ({perc:.1f}%)"

def welchs_ttest(series1, series2):
    """Performs Welch's t-test and returns the raw p-value."""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 2 or len(s2) < 2:
        return np.nan
    _, p_val = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
    return p_val

def mann_whitney_u_test(series1, series2):
    """Performs Mann-Whitney U test and returns the raw p-value."""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p_val = mannwhitneyu(s1, s2, alternative='two-sided')
        return p_val
    except ValueError:
        return 1.0

# ! no FISHER!
def categorical_pvalue(series1, series2):
    """Performs Chi-squared test and returns the raw p-value."""
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    contingency_table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return np.nan
    try:
        _, p_val, _, _ = chi2_contingency(contingency_table)
        return p_val
    except ValueError:
        return np.nan

def get_cause_cols(row_order: list) -> list:
    """Identifies weight gain cause columns from the ROW_ORDER config."""
    wgc_cols = []
    in_wgc_section = False
    for var, _ in row_order:
        if var == "delim_wgc":
            in_wgc_section = True
            continue
        if var.startswith("delim_") and in_wgc_section:
            break
        if in_wgc_section:
            wgc_cols.append(var)
    return wgc_cols

def get_variable_types(df, cause_cols):
    """Determines if a variable is continuous, categorical, or availability."""
    var_types = {}
    for col in df.columns:
        if (col in ["sex_f", "instant_dropout"] or col.startswith("country_") or col.endswith("_achieved") or col.endswith("_dropout") or col in cause_cols):
            var_types[col] = "categorical"
        elif col in ["patient_id", "medical_record_id", "medical_record_start_date"]:
            continue
        else:
            var_types[col] = "continuous"
    return var_types

def format_value(df, var, vtype, column_name=None):
    """Formats a single variable based on its type, with conditional logic for continuous."""
    if vtype == "continuous":
        mean_sd = format_mean_sd(df[var])
        if column_name in ["Parent cohort", "Observed cohort"]:
            median_iqr = format_median_iqr(df[var])
            return f"{mean_sd} | {median_iqr}"
        return mean_sd
    elif vtype == "categorical":
        return format_n_perc(df[var])
    elif vtype == "availability":
        return format_availability(df[var])
    else:
        return "N/A"

def add_empty_rows_and_pretty_names(summary_rows, pretty_names):
    """Adds section delimiters and applies pretty names to variables."""
    all_columns = set()
    if summary_rows:
        all_columns.update(summary_rows[0].keys())

    new_rows = []
    name_map = {var: pretty for var, pretty in pretty_names}

    for var, pretty in pretty_names:
        if var.startswith("delim_"):
            row = {col: "" for col in all_columns}
            row["Variable"] = pretty
            new_rows.append(row)
        else:
            found_row = next((r for r in summary_rows if r.get("Variable") == var), None)
            if found_row is not None:
                found_row["Variable"] = pretty
                new_rows.append(found_row)
    return new_rows

# ! HERE is where you determine the type of continous variable comparison test used - currently set to MWU ASSUMING NON-NORMALITY
# IF data is found to be normal, switch to welchs_ttest
def perform_comparison(g0, g1, var, vtype):
    """Compares two groups on a given variable using the appropriate statistical test."""
    if var not in g0.columns or var not in g1.columns:
        return np.nan
    if vtype == "availability":
        s0, s1, effective_vtype = g0[var].notna().astype(int), g1[var].notna().astype(int), "categorical"
    else:
        s0, s1, effective_vtype = g0[var], g1[var], vtype
    if effective_vtype == 'continuous':
        p_value = mann_whitney_u_test(s0, s1)
    elif effective_vtype == 'categorical':
        p_value = categorical_pvalue(s0, s1)
    else:
        p_value = np.nan
    return p_value


def switch_pvalues_to_asterisks(df: pd.DataFrame, data_columns: list) -> pd.DataFrame:
    """Creates a publication-ready table by replacing p-values with significance asterisks."""
    pub_df = df.copy()
    p_value_cols_to_drop = [col for col in pub_df.columns if 'p-value' in col]
    
    # Explicitly convert all p-value columns to numeric at the start
    for p_col in p_value_cols_to_drop:
        pub_df[p_col] = pd.to_numeric(pub_df[p_col], errors='coerce')

    for data_col in data_columns:
        basename = data_col.replace(': Mean/N', '')
        raw_p_col = f"{basename}: p-value"
        fdr_p_col = f"{basename}: p-value (FDR-corrected)"

        if raw_p_col in pub_df.columns and fdr_p_col in pub_df.columns:
            # Now, the data is guaranteed to be numeric
            fdr_p = pub_df[fdr_p_col]
            raw_p = pub_df[raw_p_col]
            
            cond_fdr_sig = fdr_p < 0.05
            cond_raw_only_sig = (raw_p < 0.05) & (fdr_p >= 0.05)
            conditions = [cond_fdr_sig, cond_raw_only_sig]
            choices = [pub_df[data_col].astype(str) + '**', pub_df[data_col].astype(str) + '*']
            
            pub_df[data_col] = np.select(conditions, choices, default=pub_df[data_col].astype(str))
            
    pub_df.drop(columns=p_value_cols_to_drop, inplace=True, errors='ignore')
    return pub_df


# =========================
# 2. STRATIFIED COMPARISON FUNCTIONS
# =========================

def demographic_stratification(df, df_mother, config: descriptive_comparisons_config, conn):
    """Performs demographic stratification and comparison to mother cohort."""
    print("Running Demographic Stratification...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(pd.concat([df, df_mother]), cause_cols)
    median_age = pd.to_numeric(df["age"], errors="coerce").median()
    groups = {
        "Age < Median": df[df["age"] < median_age], "Age \u2265 Median": df[df["age"] >= median_age],
        "Males": df[df["sex_f"] == 0], "Females": df[df["sex_f"] == 1],
        "BMI < 30": df[df["baseline_bmi"] < 30], "BMI \u2265 30": df[df["baseline_bmi"] >= 30],
    }
    summary_rows = []
    n_row = {
        "Variable": "N", "Parent cohort": len(df_mother), "Observed cohort": len(df),
        "Cohort comparison: p-value": "N/A",
        f"Age < Median [{median_age:.2f}]": len(groups["Age < Median"]),
        f"Age \u2265 Median [{median_age:.2f}]": len(groups["Age \u2265 Median"]),
        "Age: p-value": "N/A", "Males": len(groups["Males"]), "Females": len(groups["Females"]),
        "Gender: p-value": "N/A", "BMI < 30": len(groups["BMI < 30"]), "BMI \u2265 30": len(groups["BMI \u2265 30"]),
        "BMI: p-value": "N/A",
    }
    summary_rows.append(n_row)
    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"): continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")
        vtype, row = var_types.get(var, "continuous"), {"Variable": var}
        row["Parent cohort"] = format_value(df_mother, var, vtype, column_name="Parent cohort")
        row["Observed cohort"] = format_value(df, var, vtype, column_name="Observed cohort")
        row["Cohort comparison: p-value"] = perform_comparison(df_mother, df, var, vtype)
        g0, g1 = groups["Age < Median"], groups["Age \u2265 Median"]
        row[f"Age < Median [{median_age:.2f}]"] = format_value(g0, var, vtype)
        row[f"Age \u2265 Median [{median_age:.2f}]"] = format_value(g1, var, vtype)
        row["Age: p-value"] = perform_comparison(g0, g1, var, vtype)
        g0, g1 = groups["Males"], groups["Females"]
        row["Males"] = format_value(g0, var, vtype)
        row["Females"] = format_value(g1, var, vtype)
        row["Gender: p-value"] = perform_comparison(g0, g1, var, vtype)
        g0, g1 = groups["BMI < 30"], groups["BMI \u2265 30"]
        row["BMI < 30"] = format_value(g0, var, vtype)
        row["BMI \u2265 30"] = format_value(g1, var, vtype)
        row["BMI: p-value"] = perform_comparison(g0, g1, var, vtype)
        summary_rows.append(row)
    summary_df = pd.DataFrame(add_empty_rows_and_pretty_names(summary_rows, row_order))
    if config.fdr_correction:
        try:
            print("Applying FDR correction to demographic stratification p-values...")
            pvalue_columns = ["Cohort comparison: p-value", "Age: p-value", "Gender: p-value", "BMI: p-value"]
            pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
            valid_pvals = sum(sum(1 for p in pvals if pd.notna(p) and isinstance(p, (int, float))) for _, pvals in pvalue_dict.items())
            if valid_pvals > 1:
                print(f"FDR correction proceeding with {valid_pvals} valid p-values...")
                corrections = {col: apply_fdr_correction(pvals) for col, pvals in pvalue_dict.items()}
                summary_df = integrate_corrected_pvalues(summary_df, corrections, "(FDR-corrected)")
                print(f"FDR correction applied to {len(pvalue_columns)} demographic p-value columns")
            else:
                print(f"Warning: FDR correction skipped, only {valid_pvals} valid p-value(s) found.")
        except Exception as e:
            print(f"Error: FDR correction failed for demographic stratification. {e}")
    summary_df.to_sql(config.demographic_output_table, conn, if_exists="replace", index=False)
    print(f"Demographic stratification table saved to {config.demographic_output_table}")

# =========================
# 3. MAIN PIPELINE
# =========================

def run_descriptive_comparisons(master_config: master_config):
    """Main execution pipeline for all defined descriptive analyses."""
    try:
        if not master_config.paths or not master_config.descriptive_comparisons:
            raise ValueError("Master config missing 'paths' or 'descriptive_comparisons' configuration.")
        print(f"Starting descriptive comparisons pipeline with {len(master_config.descriptive_comparisons)} analysis configurations.")
        for i, analysis_config in enumerate(master_config.descriptive_comparisons, 1):
            try:
                print(f"\nExecuting analysis {i}/{len(master_config.descriptive_comparisons)}: '{analysis_config.analysis_name}'")
                fdr_enabled = getattr(analysis_config, 'fdr_correction', False)
                print(f"  FDR correction: {'ENABLED' if fdr_enabled else 'DISABLED'}")
                with sqlite3.connect(master_config.paths.paper_in_db) as conn_in:
                    df_input = pd.read_sql_query(f"SELECT * FROM {analysis_config.input_cohort_name}", conn_in)
                    df_mother = pd.read_sql_query(f"SELECT * FROM {analysis_config.mother_cohort_name}", conn_in)
                print(f"  Loaded data: {len(df_input)} input records, {len(df_mother)} mother records.")
                with sqlite3.connect(master_config.paths.paper_out_db) as conn_out:
                    demographic_stratification(df_input, df_mother, analysis_config, conn_out)
                    # WGC analyses removed as per user request
                print(f"  Analysis '{analysis_config.analysis_name}' completed successfully.")
            except Exception as e:
                print(f"  Error: Analysis '{analysis_config.analysis_name}' failed: {e}")
                raise
        print("\n--- All Descriptive Analyses Complete ---")
    except Exception as e:
        print(f"\nError: Descriptive comparisons pipeline failed: {e}")
        raise


# =============================================================================
# VISUAL COMPARISONS
# =============================================================================


"""
COMPARATIVE_VISUALIZATIONS.PY
Version: 1.0

Description:
A visualization module designed to create polished, publication-ready plots comparing two cohorts 
(e.g., General Population vs Genomics Subset). It bridges the visual style of cluster_descriptions.py 
with the simple 2-group logic of descriptive_comparisons.py.

Features:
- Split Violin Plots for continuous variables (Population vs Subset).
- Stacked Bar Charts for categorical variables.
- Auto-detection of variable types based on row_order configuration.
- Integrated statistical annotation (medians, counts, p-values).

Usage:
    from comparative_visualizations import run_comparative_visualizations
    run_comparative_visualizations(df_input, df_mother, config, output_dir)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
from typing import List, Dict, Tuple, Optional, Any
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Visual Style Constants (Inherited from cluster_descriptions_copy.py)
POPULATION_COLOR = '#A0A0A0'   # Neutral Gray for Mother Cohort
SUBSET_COLOR = '#00CC96'       # Distinct Teal/Green for Genomics Subset
ACHIEVED_COLOR = '#4361EE'     # Blue for "Yes/Achieved" in stacked bars
NOT_ACHIEVED_COLOR = '#EC5B57' # Red for "No/Not Achieved" in stacked bars

# Plot Layout Constants
FIG_SIZE_VIOLIN = (10, 6)
FIG_SIZE_BAR = (8, 6)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_ANNOT = 12

# =============================================================================
# HELPER FUNCTIONS - Statistics & Format
# =============================================================================

def calculate_p_value_continuous(s1, s2):
    """Mann-Whitney U test for continuous variables."""
    s1 = pd.to_numeric(s1, errors='coerce').dropna()
    s2 = pd.to_numeric(s2, errors='coerce').dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p = mannwhitneyu(s1, s2, alternative='two-sided')
        return p
    except ValueError:
        return np.nan

def calculate_p_value_categorical(s1, s2):
    """Chi-squared test for categorical variables."""
    s1 = s1.dropna()
    s2 = s2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    # Create contingency table
    contingency = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan
    try:
        _, p, _, _ = chi2_contingency(contingency)
        return p
    except ValueError:
        return np.nan

def format_p_value(p):
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p:.3f}"

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_split_violin(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    variable: str, 
    pretty_name: str, 
    output_path: str
):
    """
    Creates a detailed split violin plot:
    - Left side: Mother Cohort (Population)
    - Right side: Input Cohort (Subset)
    Includes annotation boxes for medians and p-value.
    """
    # Clean Data
    s_sub = pd.to_numeric(df_subset[variable], errors='coerce').dropna()
    s_pop = pd.to_numeric(df_mother[variable], errors='coerce').dropna()
    
    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping {variable}: No valid data.")
        return

    # Prepare Combined DataFrame for Seaborn
    # 'Dummy' y-axis variable needed for simple violin split
    df_plot_sub = pd.DataFrame({'Value': s_sub, 'Group': 'Subset', 'Split': 'Violin'})
    df_plot_pop = pd.DataFrame({'Value': s_pop, 'Group': 'Population', 'Split': 'Violin'})
    df_combined = pd.concat([df_plot_pop, df_plot_sub])

    # Statistics
    med_sub = s_sub.median()
    med_pop = s_pop.median()
    p_val = calculate_p_value_continuous(s_sub, s_pop)
    p_str = format_p_value(p_val)

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_VIOLIN)

    # Draw Violin
    sns.violinplot(
        data=df_combined, 
        x='Split', 
        y='Value', 
        hue='Group', 
        split=True, 
        inner='quart',
        palette={'Population': POPULATION_COLOR, 'Subset': SUBSET_COLOR},
        ax=ax
    )

    # Customization
    ax.set_title(f'Distribution of {pretty_name}', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel(pretty_name, fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('')
    ax.set_xticklabels([]) # Remove the 'Violin' label from x-axis
    
    # Legend
    ax.legend(title='Cohort', loc='upper right', frameon=True, fancybox=True, framealpha=0.9)

    # Annotations (Median Boxes)
    # Get plot dims to position text nicely
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Population Stats Box (Left)
    text_pop = f"Population (n={len(s_pop)})\nMedian: {med_pop:.2f}"
    ax.text(
        -0.25, y_max - (y_range * 0.05), text_pop, 
        ha='center', va='top', fontsize=FONT_SIZE_ANNOT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=POPULATION_COLOR, linewidth=2, alpha=0.9)
    )

    # Subset Stats Box (Right)
    text_sub = f"Subset (n={len(s_sub)})\nMedian: {med_sub:.2f}"
    ax.text(
        0.25, y_max - (y_range * 0.05), text_sub, 
        ha='center', va='top', fontsize=FONT_SIZE_ANNOT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=SUBSET_COLOR, linewidth=2, alpha=0.9)
    )

    # P-value (Center Bottom)
    text_sig = f"Statistical Comparison:\n{p_str}"
    ax.text(
        0, y_min + (y_range * 0.05), text_sig, 
        ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT + 2, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Violin: {os.path.basename(output_path)}")


def plot_stacked_bar(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    variable: str, 
    pretty_name: str, 
    output_path: str
):
    """
    Creates a stacked bar chart:
    - Bar 1: Mother Cohort (Population)
    - Bar 2: Input Cohort (Subset)
    Shows proportion of Yes/No (1/0) or True/False.
    """
    # Clean Data (ensure binary 0/1)
    s_sub = df_subset[variable].dropna().astype(int)
    s_pop = df_mother[variable].dropna().astype(int)
    
    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping {variable}: No valid data.")
        return

    # Calculate Proportions
    prop_sub = s_sub.mean() * 100
    prop_pop = s_pop.mean() * 100
    
    n_sub = len(s_sub)
    n_pop = len(s_pop)

    p_val = calculate_p_value_categorical(s_sub, s_pop)
    p_str = format_p_value(p_val)

    # Prepare Data for Plotting
    # Columns: Group, Prop_Achieved, Prop_NotAchieved
    plot_data = [
        {'Group': 'Population', 'Achieved': prop_pop, 'Not': 100 - prop_pop, 'N': n_pop},
        {'Group': 'Subset',     'Achieved': prop_sub, 'Not': 100 - prop_sub, 'N': n_sub}
    ]
    df_plot = pd.DataFrame(plot_data)

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_BAR)

    # Create Stacked Bars
    indices = np.arange(len(df_plot))
    width = 0.5

    p1 = ax.bar(indices, df_plot['Achieved'], width, color=ACHIEVED_COLOR, label='Yes / Achieved', alpha=0.9)
    p2 = ax.bar(indices, df_plot['Not'], width, bottom=df_plot['Achieved'], color=NOT_ACHIEVED_COLOR, label='No / Not Achieved', alpha=0.9)

    # Customization
    ax.set_title(f'{pretty_name} (%)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Percentage (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_xticks(indices)
    ax.set_xticklabels(df_plot['Group'], fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 115) # Extra space for text

    ax.legend(loc='upper right', frameon=True)

    # Annotations on bars
    for i, row in df_plot.iterrows():
        # Label Achieved % inside blue bar
        if row['Achieved'] > 10: # Only show if bar is large enough
            ax.text(i, row['Achieved']/2, f"{row['Achieved']:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=FONT_SIZE_ANNOT)
        
        # Label N count on top
        ax.text(i, 102, f"n={row['N']}", ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT)

    # P-value annotation joining the bars
    # Draw a line between bars
    x1, x2 = 0, 1
    y, h = 108, 2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*0.5, y+h, p_str, ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Bar Chart: {os.path.basename(output_path)}")


def plot_data_collection_timeline(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    date_col: str, 
    output_path: str
):
    """
    Overlapped histogram of data collection periods (by Month).
    """
    # Convert to datetime if needed
    s_sub = pd.to_datetime(df_subset[date_col], errors='coerce').dropna()
    s_pop = pd.to_datetime(df_mother[date_col], errors='coerce').dropna()

    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping Timeline: No valid dates in {date_col}")
        return

    # Create Month Periods (e.g., 2023-01)
    # converting to period 'M' and then to timestamp for plotting
    sub_months = s_sub.dt.to_period('M').dt.to_timestamp()
    pop_months = s_pop.dt.to_period('M').dt.to_timestamp()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Histograms
    # Using 'stepfilled' or 'bar' with alpha for overlap
    # We define bins based on the full range
    
    all_dates = pd.concat([sub_months, pop_months])
    if all_dates.empty: return

    min_date, max_date = all_dates.min(), all_dates.max()
    
    # Create bins by month
    bins = pd.date_range(start=min_date, end=max_date + pd.offsets.MonthBegin(1), freq='MS')
    
    # Plot Population first (Gray)
    ax.hist(
        pop_months, bins=bins, color=POPULATION_COLOR, alpha=0.5, 
        label=f'Population (n={len(s_pop)})', edgecolor='none'
    )
    
    # Plot Subset (Teal)
    ax.hist(
        sub_months, bins=bins, color=SUBSET_COLOR, alpha=0.7, 
        label=f'Subset (n={len(s_sub)})', edgecolor='none'
    )

    ax.set_title('Data Collection Timeline (Monthly Volumes)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Number of Records', fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Timeline: {os.path.basename(output_path)}")


def plot_kaplan_meier_comparison(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    time_col: str, 
    target_pct: int, 
    output_path: str
):
    """
    Kaplan-Meier Analysis for Time to X% Weight Loss using Lifelines.
    Constructs Time/Event data using 'days_to_X%_wl', 'X%_wl_achieved', and 'total_followup_days'.
    """
    event_col = f"{target_pct}%_wl_achieved"
    censor_col = "total_followup_days" # Fallback if event not achieved

    # Check columns
    required = [time_col, event_col, censor_col]
    
    # Data Preparation Function
    def prep_surv_data(df):
        msg_missing = [c for c in required if c not in df.columns]
        if msg_missing:
            return None, None
            
        T = []
        E = []
        
        # Robust row-wise data construction
        for i, row in df.iterrows():
            achieved = row[event_col]
            time_to_event = row[time_col]
            followup = row[censor_col]
            
            # Logic:
            # If Achieved == 1: Time = time_to_event, Event = 1
            # If Achieved == 0: Time = followup, Event = 0
            if pd.isna(achieved):
                continue
                
            if achieved == 1:
                # If time is missing despite achieved, we have corrupt data -> exclude
                if pd.isna(time_to_event):
                    continue 
                T.append(time_to_event)
                E.append(1)
            else:
                # Censored
                if pd.isna(followup):
                    continue
                T.append(followup)
                E.append(0)
        return np.array(T), np.array(E)

    T_sub, E_sub = prep_surv_data(df_subset)
    T_pop, E_pop = prep_surv_data(df_mother)
    
    if T_sub is None or len(T_sub) < 5 or T_pop is None or len(T_pop) < 5:
        # print(f"  ⚠️ Skipping KM {target_pct}%: Insufficient data.")
        return

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_VIOLIN)

    # Use Lifelines Fitters
    kmf_pop = KaplanMeierFitter()
    kmf_sub = KaplanMeierFitter()

    # Fit and Plot Population (Background/Control)
    kmf_pop.fit(T_pop, event_observed=E_pop, label=f'Population (n={len(T_pop)})')
    kmf_pop.plot_survival_function(ax=ax, color=POPULATION_COLOR, linewidth=2.5, alpha=0.8, ci_alpha=0.1)

    # Fit and Plot Subset (Comparison)
    kmf_sub.fit(T_sub, event_observed=E_sub, label=f'Subset (n={len(T_sub)})')
    kmf_sub.plot_survival_function(ax=ax, color=SUBSET_COLOR, linewidth=2.5, alpha=0.9, ci_alpha=0.2)

    # Log-Rank Test
    results = logrank_test(T_pop, T_sub, event_observed_A=E_pop, event_observed_B=E_sub)
    p_val = results.p_value
    p_str = format_p_value(p_val)

    # Stats Annotation
    text_sig = f"Log-Rank Test:\n{p_str}"
    ax.text(
        0.05, 0.05, text_sig, 
        transform=ax.transAxes,
        ha='left', va='bottom', fontsize=FONT_SIZE_ANNOT + 2, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9)
    )

    ax.set_title(f'Time to {target_pct}% Weight Loss (Kaplan-Meier)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Survival Probability (Not Achieved)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('Days from Baseline', fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 1.05)
    
    # Clean up legend (Lifelines adds one by default, but we customize placement)
    ax.legend(loc='lower left', bbox_to_anchor=(0.3, 0.05), frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved KM Curve: {os.path.basename(output_path)}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_comparative_visualizations(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    config: Any, 
    output_dir: str
):
    """
    Main execution function.
    Iterates through row_order in config, identifies variable types, and generates plots.
    """
    print("\n" + "="*60)
    print("STARTING COMPARATIVE VISUALIZATIONS")
    print("="*60)
    print(f"Output Directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    row_order = config.row_order

    for variable, pretty_name in row_order:
        # Skip UI delimiters/headers
        if variable == 'N' or variable.startswith('delim_'):
            continue
        
        # Verify existence
        if variable not in df_subset.columns or variable not in df_mother.columns:
            # print(f"  ⚠️ Variable {variable} not found in data.")
            continue
            
        # Determine Variable Type
        # Logic adapted from descriptive_comparisons.py heuristics
        is_categorical = False
        
        # Heuristic 1: Explicit known categorical markers
        if (variable == "sex_f" or 
            variable == "instant_dropout" or 
            variable.endswith("_achieved") or 
            variable.endswith("_dropout") or 
            variable.startswith("country_")):
            is_categorical = True
            
        # Heuristic 2: Low cardinality (e.g. < 10 unique values) implies categorical
        # Here, this is NOT optimal, because we have low-cardinality continous vars such as number of medical records - commented out
        # elif df_subset[variable].nunique() <= 10:
        #     is_categorical = True

        # Clean filename
        safe_name = variable.replace('%', 'pct').replace(' ', '_').replace('/', '_')
        
        if is_categorical:
            out_file = os.path.join(output_dir, f"{safe_name}_stack.png")
            plot_stacked_bar(df_subset, df_mother, variable, pretty_name, out_file)
        else:
            out_file = os.path.join(output_dir, f"{safe_name}_violin.png")
            plot_split_violin(df_subset, df_mother, variable, pretty_name, out_file)

    # --- 2. Temporal Analysis (Timeline) ---
    if 'baseline_date' in df_subset.columns and 'baseline_date' in df_mother.columns:
        print("\nPossible Timeline Analysis detected...")
        tl_path = os.path.join(output_dir, "data_collection_timeline.png")
        plot_data_collection_timeline(df_subset, df_mother, 'baseline_date', tl_path)

    # --- 3. Survival Analysis (Kaplan-Meier) ---
    # Targets: 5%, 10%, 15%
    for target in [5, 10, 15]:
        time_col = f"days_to_{target}%_wl"
        # Check if column exists in both
        if time_col in df_subset.columns and time_col in df_mother.columns:
            km_path = os.path.join(output_dir, f"time_to_{target}pct_wl_km.png")
            plot_kaplan_meier_comparison(df_subset, df_mother, time_col, target, km_path)

    print("\n" + "="*60)
    print("VISUALIZATION RUN COMPLETE")
    print("="*60)
