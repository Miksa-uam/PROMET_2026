import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict, Tuple
from scipy.stats import ttest_ind, chi2_contingency
from paper12_config import descriptive_comparisons_config, master_config
from fdr_correction_utils import collect_pvalues_from_dataframe, apply_fdr_correction, integrate_corrected_pvalues

# Configure logging
logger = logging.getLogger(__name__)


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
    from scipy.stats import mannwhitneyu
    
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 2 or len(s2) < 2:
        return np.nan
    
    try:
        _, p_val = mannwhitneyu(s1, s2, alternative='two-sided')
        return p_val
    except ValueError:
        # Handle cases where all values are identical
        return np.nan

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
        return np.nan # Not enough data for a 2x2 table
    try:
        _, p_val, _, _ = chi2_contingency(contingency_table)
        return p_val
    except ValueError: # Catches errors with all 0s in a row/col
        return np.nan

def get_cause_cols(row_order: list) -> list:
    """Identifies weight gain cause columns from the ROW_ORDER config."""
    wgc_cols = [
        # "womens_health_and_pregnancy",
        # "mental_health",
        # "family_issues",
        # "medication_disease_injury",
        # "physical_inactivity",
        # "eating_habits",
        # "schedule",
        # "smoking_cessation",
        # "treatment_discontinuation_or_relapse",
        # "pandemic",
        # "lifestyle_circumstances",
        # "none_of_above"
    ]

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
        if (
            col in ["sex_f", "instant_dropout"]
            or col.endswith("_achieved")
            or col.endswith("_dropout")
            or col in cause_cols
        ):
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

    for var, pretty in pretty_names:
        if var.startswith("delim_"):
            row = {col: "" for col in all_columns}
            row["Variable"] = pretty
            new_rows.append(row)
        else:
            row = next((r for r in summary_rows if r["Variable"] == var), None)
            if row is not None:
                row["Variable"] = pretty
                new_rows.append(row)
    return new_rows

def perform_comparison(g0, g1, var, vtype):
    """
    Compares two groups on a given variable using the appropriate statistical test
    and formats the resulting p-value.
    """
    # Ensure the variable exists in both dataframes
    if var not in g0.columns or var not in g1.columns:
        return np.nan # Return NaN for missing data

    # For 'availability', we compare the presence/absence of data (1/0)
    if vtype == "availability":
        s0 = g0[var].notna().astype(int)
        s1 = g1[var].notna().astype(int)
        effective_vtype = "categorical"
    else:
        s0 = g0[var]
        s1 = g1[var]
        effective_vtype = vtype

    # Select and run the appropriate statistical test
    if effective_vtype == 'continuous':
        p_value = welchs_ttest(s0, s1)
    elif effective_vtype == 'categorical':
        p_value = categorical_pvalue(s0, s1)
    else:
        p_value = np.nan

    # Return the raw p-value
    return p_value

# =========================
# 2. PLOTTING FUNCTION
# =========================

def plot_datacollection_bias(df_cohort, df_mother, cohort_name, mother_cohort_name, output_file):
    """
    Generates and saves a bar plot comparing the distribution of medical record start years.
    Ensures plots are saved to the outputs directory, not the scripts directory.
    """
    import os
    
    # Ensure the output file is in the outputs directory
    if not os.path.isabs(output_file):
        # If it's a relative path, make sure it goes to outputs folder
        outputs_dir = "../outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        output_file = os.path.join(outputs_dir, os.path.basename(output_file))
    
    df_cohort['year'] = pd.to_datetime(df_cohort['baseline_date']).dt.year
    df_mother['year'] = pd.to_datetime(df_mother['baseline_date']).dt.year

    counts_cohort = df_cohort['year'].value_counts(normalize=True).sort_index()
    counts_mother = df_mother['year'].value_counts(normalize=True).sort_index()

    plot_df = pd.DataFrame({
        'Year': counts_mother.index.union(counts_cohort.index)
    }).set_index('Year')
    plot_df[f'Mother Cohort ({mother_cohort_name})'] = counts_mother
    plot_df[f'Cohort ({cohort_name})'] = counts_cohort
    plot_df = plot_df.fillna(0).reset_index().melt(id_vars='Year', var_name='Cohort', value_name='Proportion')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x='Year', y='Proportion', hue='Cohort', ax=ax)

    ax.set_title('Distribution of Medical Record Start Dates', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion of Patients', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Cohort')
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Data collection bias plot saved to {output_file}")

# =========================
# 3. STRATIFIED COMPARISON FUNCTIONS
# =========================

def demographic_stratification(df, df_mother, config: descriptive_comparisons_config, conn):
    """Performs demographic stratification and comparison to mother cohort."""
    print("Running Demographic Stratification...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(pd.concat([df, df_mother]), cause_cols)

    median_age = pd.to_numeric(df["age"], errors="coerce").median()
    groups = {
        "Age < Median": df[df["age"] < median_age],
        "Age ≥ Median": df[df["age"] >= median_age],
        "Males": df[df["sex_f"] == 0],
        "Females": df[df["sex_f"] == 1],
        "BMI < 30": df[df["baseline_bmi"] < 30],
        "BMI ≥ 30": df[df["baseline_bmi"] >= 30],
    }

    summary_rows = []
    n_row = {
        "Variable": "N", "Parent cohort": len(df_mother), "Observed cohort": len(df),
        "Cohort comparison: p-value": "N/A",
        f"Age < Median [{median_age:.2f}]": len(groups["Age < Median"]),
        f"Age ≥ Median [{median_age:.2f}]": len(groups["Age ≥ Median"]),
        "Age: p-value": "N/A",
        "Males": len(groups["Males"]), "Females": len(groups["Females"]),
        "Gender: p-value": "N/A",
        "BMI < 30": len(groups["BMI < 30"]), "BMI ≥ 30": len(groups["BMI ≥ 30"]),
        "BMI: p-value": "N/A",
    }
    summary_rows.append(n_row)

    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"):
            continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")

        vtype = var_types.get(var, "continuous")
        row = {"Variable": var}

        # Parent vs. Observed cohort comparison
        row["Parent cohort"] = format_value(df_mother, var, vtype, column_name="Parent cohort")
        row["Observed cohort"] = format_value(df, var, vtype, column_name="Observed cohort")
        row["Cohort comparison: p-value"] = perform_comparison(df_mother, df, var, vtype)

        # Age group comparison
        g0, g1 = groups["Age < Median"], groups["Age ≥ Median"]
        row[f"Age < Median [{median_age:.2f}]"] = format_value(g0, var, vtype)
        row[f"Age ≥ Median [{median_age:.2f}]"] = format_value(g1, var, vtype)
        row["Age: p-value"] = perform_comparison(g0, g1, var, vtype)

        # Sex group comparison
        g0, g1 = groups["Males"], groups["Females"]
        row["Males"] = format_value(g0, var, vtype)
        row["Females"] = format_value(g1, var, vtype)
        row["Gender: p-value"] = perform_comparison(g0, g1, var, vtype)

        # BMI group comparison
        g0, g1 = groups["BMI < 30"], groups["BMI ≥ 30"]
        row["BMI < 30"] = format_value(g0, var, vtype)
        row["BMI ≥ 30"] = format_value(g1, var, vtype)
        row["BMI: p-value"] = perform_comparison(g0, g1, var, vtype)

        summary_rows.append(row)

    summary_rows = add_empty_rows_and_pretty_names(summary_rows, row_order)
    summary_df = pd.DataFrame(summary_rows)
    
    # Apply FDR correction if enabled
    if config.fdr_correction:
        try:
            print("Applying FDR correction to demographic stratification p-values...")
            logger.info("Starting FDR correction for demographic stratification")
            
            # Define p-value columns for demographic stratification
            pvalue_columns = [
                "Cohort comparison: p-value",
                "Age: p-value", 
                "Gender: p-value",
                "BMI: p-value"
            ]
            
            # Collect p-values from the summary DataFrame
            pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
            
            # Check if we have sufficient valid p-values for correction
            total_pvals = 0
            valid_pvals = 0
            for col, pvals in pvalue_dict.items():
                total_pvals += len(pvals)
                valid_pvals += sum(1 for p in pvals if pd.notna(p) and isinstance(p, (int, float)))
            
            if valid_pvals == 0:
                logger.warning("FDR correction skipped for demographic stratification: no valid p-values found")
                print("Warning: FDR correction skipped - no valid p-values found in demographic comparisons")
            elif valid_pvals < 2:
                logger.warning(f"FDR correction applied with limited data: only {valid_pvals} valid p-value(s) found")
                print(f"Warning: FDR correction applied with only {valid_pvals} valid p-value(s)")
            else:
                logger.info(f"FDR correction proceeding with {valid_pvals} valid p-values out of {total_pvals} total")
            
            # Apply FDR correction to collected p-values
            corrections = {}
            successful_corrections = 0
            for col, pvals in pvalue_dict.items():
                try:
                    corrected_pvals = apply_fdr_correction(pvals)
                    corrections[col] = corrected_pvals
                    successful_corrections += 1
                except Exception as e:
                    logger.error(f"FDR correction failed for column '{col}': {str(e)}. Using original p-values.")
                    corrections[col] = pvals  # Fallback to original p-values
            
            # Integrate corrected p-values back into the DataFrame
            summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
            
            logger.info(f"FDR correction completed for demographic stratification: {successful_corrections}/{len(pvalue_columns)} columns corrected successfully")
            print(f"FDR correction applied to {successful_corrections} demographic p-value columns")
            
        except Exception as e:
            logger.error(f"FDR correction failed for demographic stratification: {str(e)}. Continuing with original p-values.")
            print(f"Error: FDR correction failed for demographic stratification. Continuing with original p-values.")
    
    summary_df.to_sql(config.demographic_output_table, conn, if_exists="replace", index=False)
    print(f"Demographic stratification table saved to {config.demographic_output_table}")

def wgc_stratification(df, config: descriptive_comparisons_config, conn):
    """Performs stratification based on weight gain cause variables."""

    print("Running Weight Gain Cause Stratification...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(df, cause_cols)

    groups = {}
    strata_pairs = []
    for cause in cause_cols:
        pretty_cause = next((p for v, p in row_order if v == cause), cause).replace(" (yes/no)", "")
        yes_label = f"{pretty_cause}: Yes"
        no_label = f"{pretty_cause}: No"
        groups[no_label] = df[df[cause] == 0]
        groups[yes_label] = df[df[cause] == 1]
        strata_pairs.append((no_label, yes_label, pretty_cause))

    summary_rows = []
    n_row = {"Variable": "N"}
    for gA_name, gB_name, label in strata_pairs:
        n_row[gA_name] = len(groups.get(gA_name, []))
        n_row[gB_name] = len(groups.get(gB_name, []))
        n_row[f"{label}: p-value"] = "N/A"
    summary_rows.append(n_row)

    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"):
            continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")

        vtype = var_types.get(var, "continuous")
        row = {"Variable": var}
        for gA_name, gB_name, label in strata_pairs:
            gA, gB = groups.get(gA_name), groups.get(gB_name)
            row[gA_name] = format_value(gA, var, vtype)
            row[gB_name] = format_value(gB, var, vtype)
            row[f"{label}: p-value"] = perform_comparison(gA, gB, var, vtype)
        summary_rows.append(row)

    summary_rows = add_empty_rows_and_pretty_names(summary_rows, row_order)
    summary_df = pd.DataFrame(summary_rows)
    
    # Apply FDR correction if enabled
    if config.fdr_correction:
        try:
            print("Applying FDR correction to weight gain cause stratification p-values...")
            logger.info("Starting FDR correction for weight gain cause stratification")
            
            # Collect p-value column names dynamically based on weight gain causes
            pvalue_columns = []
            for _, _, label in strata_pairs:
                pvalue_columns.append(f"{label}: p-value")
            
            if not pvalue_columns:
                logger.warning("FDR correction skipped for weight gain cause stratification: no weight gain cause columns found")
                print("Warning: FDR correction skipped - no weight gain cause comparisons found")
            else:
                logger.info(f"Found {len(pvalue_columns)} weight gain cause comparison columns for FDR correction")
                
                # Collect p-values from the summary DataFrame
                pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
                
                # Check if we have sufficient valid p-values for correction
                total_pvals = 0
                valid_pvals = 0
                for col, pvals in pvalue_dict.items():
                    total_pvals += len(pvals)
                    valid_pvals += sum(1 for p in pvals if pd.notna(p) and isinstance(p, (int, float)))
                
                if valid_pvals == 0:
                    logger.warning("FDR correction skipped for weight gain cause stratification: no valid p-values found")
                    print("Warning: FDR correction skipped - no valid p-values found in weight gain cause comparisons")
                elif valid_pvals < 2:
                    logger.warning(f"FDR correction applied with limited data: only {valid_pvals} valid p-value(s) found")
                    print(f"Warning: FDR correction applied with only {valid_pvals} valid p-value(s)")
                else:
                    logger.info(f"FDR correction proceeding with {valid_pvals} valid p-values out of {total_pvals} total")
                
                # Apply FDR correction to collected p-values
                corrections = {}
                successful_corrections = 0
                for col, pvals in pvalue_dict.items():
                    try:
                        corrected_pvals = apply_fdr_correction(pvals)
                        corrections[col] = corrected_pvals
                        successful_corrections += 1
                    except Exception as e:
                        logger.error(f"FDR correction failed for column '{col}': {str(e)}. Using original p-values.")
                        corrections[col] = pvals  # Fallback to original p-values
                
                # Integrate corrected p-values back into the DataFrame
                summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
                
                logger.info(f"FDR correction completed for weight gain cause stratification: {successful_corrections}/{len(pvalue_columns)} columns corrected successfully")
                print(f"FDR correction applied to {successful_corrections} weight gain cause p-value columns")
            
        except Exception as e:
            logger.error(f"FDR correction failed for weight gain cause stratification: {str(e)}. Continuing with original p-values.")
            print(f"Error: FDR correction failed for weight gain cause stratification. Continuing with original p-values.")
    
    summary_df.to_sql(config.wgc_output_table, conn, if_exists="replace", index=False)
    print(f"Weight gain cause stratification table saved to {config.wgc_output_table}")

def wgc_vs_population_mean_analysis(df, config: descriptive_comparisons_config, conn):
    """Performs WGC vs population mean analysis comparing each WGC group against population baseline."""
    if not hasattr(config, 'wgc_vs_mean_output_table') or not config.wgc_vs_mean_output_table:
        return  # Skip if not configured
    
    print("Running WGC vs Population Mean Analysis...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(df, cause_cols)

    # Reuse WGC group extraction logic from wgc_stratification
    groups = {}
    wgc_labels = []
    for cause in cause_cols:
        pretty_cause = next((p for v, p in row_order if v == cause), cause).replace(" (yes/no)", "")
        wgc_group = df[df[cause] == 1]
        groups[pretty_cause] = wgc_group
        wgc_labels.append(pretty_cause)

    summary_rows = []
    
    # N row
    n_row = {"Variable": "N", "Population Mean (±SD) or N (%)": len(df)}
    for wgc_label in wgc_labels:
        n_row[f"{wgc_label}: Mean/N"] = len(groups[wgc_label])
        n_row[f"{wgc_label}: p-value"] = "N/A"
        if config.fdr_correction:
            n_row[f"{wgc_label}: p-value (FDR-corrected)"] = "N/A"
    summary_rows.append(n_row)

    # Process each variable
    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"):
            continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")

        vtype = var_types.get(var, "continuous")
        row = {"Variable": var}
        
        # Population statistics
        row["Population Mean (±SD) or N (%)"] = format_value(df, var, vtype)
        
        # Compare each WGC group vs population
        for wgc_label in wgc_labels:
            wgc_group = groups[wgc_label]
            row[f"{wgc_label}: Mean/N"] = format_value(wgc_group, var, vtype)
            row[f"{wgc_label}: p-value"] = perform_comparison(df, wgc_group, var, vtype)  # Population vs WGC group
            if config.fdr_correction:
                row[f"{wgc_label}: p-value (FDR-corrected)"] = row[f"{wgc_label}: p-value"]  # Will be corrected later
        
        summary_rows.append(row)

    summary_rows = add_empty_rows_and_pretty_names(summary_rows, row_order)
    summary_df = pd.DataFrame(summary_rows)
    
    # Apply FDR correction if enabled
    if config.fdr_correction:
        try:
            print("Applying FDR correction to WGC vs population mean p-values...")
            pvalue_columns = [f"{wgc_label}: p-value" for wgc_label in wgc_labels]
            pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
            
            corrections = {}
            for col, pvals in pvalue_dict.items():
                try:
                    corrections[col] = apply_fdr_correction(pvals)
                except Exception as e:
                    corrections[col] = pvals  # Fallback to original p-values
            
            summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
            print(f"FDR correction applied to {len(pvalue_columns)} WGC vs population p-value columns")
            
        except Exception as e:
            print(f"Error: FDR correction failed. Continuing with original p-values.")
    
    summary_df.to_sql(config.wgc_vs_mean_output_table, conn, if_exists="replace", index=False)
    print(f"WGC vs population mean analysis table saved to {config.wgc_vs_mean_output_table}")

# =========================
# 4. TABLE FORMATTING FUNCTIONS
# =========================

def add_fdr_corrected_pvalues_to_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies FDR correction using the fdr_correction_utils module and
    integrates the results back into the DataFrame.
    """
    pvalue_columns = [col for col in summary_df.columns if col.endswith(': p-value')]
    if not pvalue_columns:
        return summary_df

    pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
    
    corrections = {}
    for col, pvals in pvalue_dict.items():
        corrections[col] = apply_fdr_correction(pvals)
        
    df_corrected = integrate_corrected_pvalues(summary_df, corrections)
    
    return df_corrected

def create_publication_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a publication-friendly table with significance markers.
    This function is now robust as it relies on the correctly ordered input DataFrame.
    """
    pub_df = summary_df.copy()
    p_value_cols_to_remove = [col for col in pub_df.columns if ': p-value' in col]

    for data_col_name in [c for c in pub_df.columns if c.startswith('Cluster_') and ': p-value' not in c]:
        raw_p_col = f"{data_col_name}: p-value"
        fdr_p_col = f"{data_col_name}: p-value (FDR-corrected)"

        if raw_p_col in pub_df.columns and fdr_p_col in pub_df.columns:
            def apply_significance(row):
                val, raw_p, fdr_p = row[data_col_name], pd.to_numeric(row[raw_p_col], errors='coerce'), pd.to_numeric(row[fdr_p_col], errors='coerce')
                marker = '**' if pd.notna(fdr_p) and fdr_p < 0.05 else '*' if pd.notna(raw_p) and raw_p < 0.05 else ''
                return f"{val}{marker}"
            pub_df[data_col_name] = pub_df.apply(apply_significance, axis=1)

    pub_df.drop(columns=p_value_cols_to_remove, inplace=True, errors='ignore')
    return pub_df
    
def generate_comparison_tables_with_fdr_and_publication(df, df_mother, config, conn, table_type="demographic"):
    """
    Generate both detailed (with p-values) and publication-friendly comparison tables.
    Args:
        df, df_mother: DataFrames for comparison
        config: Analysis configuration
        conn: Database connection
        table_type: "demographic" or "wgc" to determine which analysis to run
    Returns:
        Tuple of (detailed_df, publication_df)
    """
    if table_type == "demographic":
        # Run demographic stratification to get base table
        demographic_stratification(df, df_mother, config, conn)
        # Load the saved table
        table_name = config.demographic_output_table
    elif table_type == "wgc":
        # Run WGC stratification to get base table
        wgc_stratification(df, config, conn)
        # Load the saved table
        table_name = config.wgc_output_table
    else:
        raise ValueError("table_type must be 'demographic' or 'wgc'")
    
    # Load the base table
    base_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    # Add FDR-corrected p-values
    detailed_df = add_fdr_corrected_pvalues_to_table(base_df)
    
    # Save detailed table (with both raw and FDR p-values)
    detailed_table_name = f"{table_name}_with_fdr"
    detailed_df.to_sql(detailed_table_name, conn, if_exists="replace", index=False)
    print(f"Detailed table saved to: {detailed_table_name}")
    
    # Create publication-friendly table
    pub_df = create_publication_table(detailed_df)
    
    # Save publication table
    pub_table_name = f"{table_name}_publication"
    pub_df.to_sql(pub_table_name, conn, if_exists="replace", index=False)
    print(f"Publication table saved to: {pub_table_name}")
    
    # Display both tables
    print(f"\n=== DETAILED TABLE ({detailed_table_name}) ===")
    print(detailed_df.to_string(index=False))
    
    print(f"\n=== PUBLICATION TABLE ({pub_table_name}) ===")
    print(pub_df.to_string(index=False))
    
    return detailed_df, pub_df

# =========================
# 5. MAIN PIPELINE
# =========================

def run_descriptive_comparisons(master_config: master_config):
    """Main execution pipeline for all defined descriptive analyses."""
    try:
        # Validate master configuration
        if not master_config.paths:
            raise ValueError("Master config missing paths configuration")
        if not master_config.descriptive_comparisons:
            raise ValueError("Master config missing descriptive_comparisons configuration")
        
        input_db_path = master_config.paths.paper_in_db
        output_db_path = master_config.paths.paper_out_db
        
        logger.info(f"Starting descriptive comparisons pipeline with {len(master_config.descriptive_comparisons)} analysis configurations")

        for i, analysis_config in enumerate(master_config.descriptive_comparisons, 1):
            try:
                print(f"\nExecuting analysis {i}/{len(master_config.descriptive_comparisons)}: '{analysis_config.analysis_name}'")
                logger.info(f"Starting analysis: {analysis_config.analysis_name}")
                
                # Validate and log FDR correction configuration
                fdr_enabled = getattr(analysis_config, 'fdr_correction', False)
                if fdr_enabled:
                    print(f"  FDR correction: ENABLED (Benjamini-Hochberg method)")
                    logger.info(f"FDR correction enabled for analysis: {analysis_config.analysis_name}")
                else:
                    print(f"  FDR correction: DISABLED (raw p-values only)")
                    logger.info(f"FDR correction disabled for analysis: {analysis_config.analysis_name}")
                
                # Validate required configuration parameters
                required_params = ['input_cohort_name', 'mother_cohort_name', 'demographic_output_table', 'wgc_output_table']
                for param in required_params:
                    if not hasattr(analysis_config, param) or not getattr(analysis_config, param):
                        raise ValueError(f"Analysis config '{analysis_config.analysis_name}' missing required parameter: {param}")
                
                # Load data
                with sqlite3.connect(input_db_path) as conn_in:
                    df_input = pd.read_sql_query(f"SELECT * FROM {analysis_config.input_cohort_name}", conn_in)
                    df_mother = pd.read_sql_query(f"SELECT * FROM {analysis_config.mother_cohort_name}", conn_in)
                
                logger.info(f"Loaded data: {len(df_input)} input cohort records, {len(df_mother)} mother cohort records")

                # Generate bias plot if configured
                if analysis_config.bias_plot_filename:
                    plot_datacollection_bias(
                        df_input.copy(), df_mother.copy(),
                        analysis_config.input_cohort_name, analysis_config.mother_cohort_name,
                        analysis_config.bias_plot_filename
                    )
                else:
                    print("Warning: 'baseline_date' column not found. Skipping bias plot generation.")

                # Run stratifications with FDR correction configuration properly propagated
                with sqlite3.connect(output_db_path) as conn_out:
                    # Ensure FDR correction parameter is properly propagated
                    logger.info(f"Running demographic stratification with FDR correction: {fdr_enabled}")
                    demographic_stratification(
                        df_input, df_mother, 
                        analysis_config,  # This contains the fdr_correction parameter
                        conn_out
                    )
                    
                    logger.info(f"Running weight gain cause stratification with FDR correction: {fdr_enabled}")
                    wgc_stratification(
                        df_input,
                        analysis_config,  # This contains the fdr_correction parameter
                        conn_out
                    )
                    
                    # Run WGC vs population mean analysis if configured
                    if hasattr(analysis_config, 'wgc_vs_mean_output_table') and analysis_config.wgc_vs_mean_output_table:
                        logger.info(f"Running WGC vs population mean analysis with FDR correction: {fdr_enabled}")
                        wgc_vs_population_mean_analysis(
                            df_input,
                            analysis_config,  # This contains the fdr_correction parameter
                            conn_out
                        )
                
                logger.info(f"Completed analysis: {analysis_config.analysis_name}")
                print(f"  Analysis '{analysis_config.analysis_name}' completed successfully")
                
            except Exception as e:
                logger.error(f"Analysis '{analysis_config.analysis_name}' failed: {str(e)}")
                print(f"  Error: Analysis '{analysis_config.analysis_name}' failed: {str(e)}")
                raise  # Re-raise to stop pipeline on individual analysis failure
        
        logger.info("All descriptive analyses completed successfully")
        print("\n--- All Descriptive Analyses Complete ---")
        
    except Exception as e:
        logger.error(f"Descriptive comparisons pipeline failed: {str(e)}")
        print(f"\nError: Descriptive comparisons pipeline failed: {str(e)}")
        raise

