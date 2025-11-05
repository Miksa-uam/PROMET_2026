import os
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu

from paper12_config import descriptive_comparisons_config, master_config
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
        if (col in ["sex_f", "instant_dropout"] or col.endswith("_achieved") or col.endswith("_dropout") or col in cause_cols):
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
    for data_col in data_columns:
        basename = data_col.replace(': Mean/N', '')
        raw_p_col = f"{basename}: p-value"
        fdr_p_col = f"{basename}: p-value (FDR-corrected)"
        if raw_p_col in pub_df.columns and fdr_p_col in pub_df.columns:
            fdr_p = pd.to_numeric(pub_df[fdr_p_col], errors='coerce')
            raw_p = pd.to_numeric(pub_df[raw_p_col], errors='coerce')
            conditions = [fdr_p < 0.05, raw_p < 0.05]
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
                summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
                print(f"FDR correction applied to {len(pvalue_columns)} demographic p-value columns")
            else:
                print(f"Warning: FDR correction skipped, only {valid_pvals} valid p-value(s) found.")
        except Exception as e:
            print(f"Error: FDR correction failed for demographic stratification. {e}")
    summary_df.to_sql(config.demographic_output_table, conn, if_exists="replace", index=False)
    print(f"Demographic stratification table saved to {config.demographic_output_table}")

def wgc_stratification(df, config: descriptive_comparisons_config, conn):
    """Performs stratification based on weight gain cause variables."""
    print("Running Weight Gain Cause Stratification...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(df, cause_cols)
    groups, strata_pairs = {}, []
    for cause in cause_cols:
        pretty_cause = next((p for v, p in row_order if v == cause), cause).replace(" (yes/no)", "")
        yes_label, no_label = f"{pretty_cause}: Yes", f"{pretty_cause}: No"
        groups[no_label], groups[yes_label] = df[df[cause] == 0], df[df[cause] == 1]
        strata_pairs.append((no_label, yes_label, pretty_cause))
    summary_rows = []
    n_row = {"Variable": "N"}
    for gA_name, gB_name, label in strata_pairs:
        n_row[gA_name], n_row[gB_name], n_row[f"{label}: p-value"] = len(groups.get(gA_name, [])), len(groups.get(gB_name, [])), "N/A"
    summary_rows.append(n_row)
    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"): continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")
        vtype, row = var_types.get(var, "continuous"), {"Variable": var}
        for gA_name, gB_name, label in strata_pairs:
            gA, gB = groups.get(gA_name), groups.get(gB_name)
            row[gA_name], row[gB_name] = format_value(gA, var, vtype), format_value(gB, var, vtype)
            row[f"{label}: p-value"] = perform_comparison(gA, gB, var, vtype)
        summary_rows.append(row)
    summary_df = pd.DataFrame(add_empty_rows_and_pretty_names(summary_rows, row_order))
    if config.fdr_correction:
        try:
            pvalue_columns = [f"{label}: p-value" for _, _, label in strata_pairs]
            if pvalue_columns:
                print("Applying FDR correction to weight gain cause stratification p-values...")
                pvalue_dict = collect_pvalues_from_dataframe(summary_df, pvalue_columns)
                valid_pvals = sum(sum(1 for p in pvals if pd.notna(p) and isinstance(p, (int, float))) for _, pvals in pvalue_dict.items())
                if valid_pvals > 1:
                    print(f"FDR correction proceeding with {valid_pvals} valid p-values...")
                    corrections = {col: apply_fdr_correction(pvals) for col, pvals in pvalue_dict.items()}
                    summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
                    print(f"FDR correction applied to {len(pvalue_columns)} weight gain cause p-value columns")
                else:
                    print(f"Warning: FDR correction skipped, only {valid_pvals} valid p-value(s) found.")
        except Exception as e:
            print(f"Error: FDR correction failed for WGC stratification. {e}")
    summary_df.to_sql(config.wgc_output_table, conn, if_exists="replace", index=False)
    print(f"Weight gain cause stratification table saved to {config.wgc_output_table}")

def wgc_vs_population_mean_analysis(df, config: descriptive_comparisons_config, conn):
    """Performs WGC vs population mean analysis and saves detailed and publication-ready tables."""
    output_table_name = getattr(config, 'wgc_vs_mean_output_table', None)
    if not isinstance(output_table_name, str) or not output_table_name.strip():
        return
    
    print("Running WGC vs Population Mean Analysis...")
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)
    var_types = get_variable_types(df, cause_cols)
    groups, wgc_labels = {}, []
    for cause in cause_cols:
        pretty_cause = next((p for v, p in row_order if v == cause), cause).replace(" (yes/no)", "")
        groups[pretty_cause] = df[df[cause] == 1]
        wgc_labels.append(pretty_cause)
    summary_rows = []
    n_row = {"Variable": "N", "Population Mean (\u00B1SD) or N (%)": len(df)}
    for label in wgc_labels:
        n_row[f"{label}: Mean/N"] = len(groups[label])
        n_row[f"{label}: p-value"] = "N/A"
    summary_rows.append(n_row)
    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"): continue
        print(f"  Processing variable {i}/{len(row_order)}: {var}")
        vtype, row = var_types.get(var, "continuous"), {"Variable": var}
        row["Population Mean (\u00B1SD) or N (%)"] = format_value(df, var, vtype)
        for label in wgc_labels:
            group = groups[label]
            row[f"{label}: Mean/N"] = format_value(group, var, vtype)
            p_val = perform_comparison(df, group, var, vtype)
            row[f"{label}: p-value"] = p_val
            if config.fdr_correction: row[f"{label}: p-value (FDR-corrected)"] = p_val
        summary_rows.append(row)
    summary_df = pd.DataFrame(add_empty_rows_and_pretty_names(summary_rows, row_order))
    if config.fdr_correction:
        try:
            p_cols = [f"{label}: p-value" for label in wgc_labels]
            if p_cols:
                print("Applying FDR correction to WGC vs population mean p-values...")
                p_dict = collect_pvalues_from_dataframe(summary_df, p_cols)
                corrections = {col: apply_fdr_correction(pvals) for col, pvals in p_dict.items()}
                summary_df = integrate_corrected_pvalues(summary_df, corrections, " (FDR-corrected)")
                print(f"FDR correction applied to {len(p_cols)} columns.")
        except Exception as e:
            print(f"Error: FDR correction failed. Continuing with raw p-values. Details: {e}")
    detailed_name = f"{output_table_name}_detailed"
    summary_df.to_sql(detailed_name, conn, if_exists="replace", index=False)
    print(f"Detailed WGC vs population mean analysis table saved to {detailed_name}")
    data_cols = [col for col in summary_df.columns if ': Mean/N' in col]
    pub_df = switch_pvalues_to_asterisks(summary_df, data_cols)
    pub_df.to_sql(output_table_name, conn, if_exists='replace', index=False)
    print(f"Publication-ready WGC vs population mean analysis table saved to {output_table_name}")

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
                    wgc_stratification(df_input, analysis_config, conn_out)
                    if hasattr(analysis_config, 'wgc_vs_mean_output_table'):
                        wgc_vs_population_mean_analysis(df_input, analysis_config, conn_out)
                print(f"  Analysis '{analysis_config.analysis_name}' completed successfully.")
            except Exception as e:
                print(f"  Error: Analysis '{analysis_config.analysis_name}' failed: {e}")
                raise
        print("\n--- All Descriptive Analyses Complete ---")
    except Exception as e:
        print(f"\nError: Descriptive comparisons pipeline failed: {e}")
        raise
