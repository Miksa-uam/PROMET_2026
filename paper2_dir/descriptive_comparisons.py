import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
from paper12_config import descriptive_comparisons_config, master_config


# =========================
# 1. CONFIGURATION
# =========================

# # --- Create a directory for outputs if it doesn't exist ---
# paper2_directory = "."
# plots_directory = os.path.join(paper2_directory, "cohort_comp_data_collection_bias_plots")
# os.makedirs(plots_directory, exist_ok=True)

# INPUT_DB = os.path.join(paper2_directory, "pnk_db2_p2_in.sqlite")
# OUTPUT_DB = os.path.join(paper2_directory, "pnk_db2_p2_out.sqlite")

# # --- Uncomment the desired cohort configuration to run ---

# # CONFIG 1: FULL WGC COHORT (Faster, no bootstrapping)
# INPUT_COHORT_NAME = "timetoevent_wgc_compl"
# MOTHER_COHORT_NAME = "timetoevent_all"
# DEMOGRAPHIC_OUTPUT_TABLE = "wgc_cmpl_dmgrph_strt"
# WGC_OUTPUT_TABLE = "wgc_cmpl_wgc_strt"
# PLOT_OUTPUT_FILE = os.path.join(plots_directory, "wgc_cmpl_datacollection_bias.png")

# # CONFIG 2: FULL WGC + Genomics COHORT (Slower, with bootstrapping)
# # INPUT_COHORT_NAME = "timetoevent_wgc_gen_compl"
# # MOTHER_COHORT_NAME = "timetoevent_wgc_compl"
# # DEMOGRAPHIC_OUTPUT_TABLE = "wgc_gen_cmpl_dmgrph_strt"
# # WGC_OUTPUT_TABLE = "wgc_gen_cmpl_wgc_strt"
# # PLOT_OUTPUT_FILE = os.path.join(plots_directory, "wgc_gen_cmpl_datacollection_bias.png")

# TIME_WINDOWS = [40, 60, 80]
# WL_TARGETS = [5, 10, 15]

# # Row order and pretty names
# ROW_ORDER = [
#     ("N", "N"),
#     # Demographics & baseline anthropometry
#     ("delim_demo", "Demographics and baseline anthropometry"),
#     ("sex_f", "Sex (% of females)"),
#     ("age", "Age (years)"),
#     ("height_m", "Height (m)"),
#     ("baseline_weight_kg", "Baseline weight (kg)"),
#     ("baseline_bmi", "Baseline BMI (kg/m²)"),
#     ("baseline_fat_%", "Baseline fat mass (%)"),
#     ("baseline_muscle_%", "Baseline muscle mass (%)"),
#     # Treatment outcomes
#     ("delim_outcomes", "Treatment outcomes"),
#     ("total_followup_days", "Follow-up length (days)"),
#     ("dietitian_visits", "Number of visits"),
#     ("nr_total_measurements", "Number of total measurements"),
#     ("avg_days_between_measurements", "Average measurement frequency (days)"),
#     ("last_aval_weight_kg", "Last measured weight (kg)"),
#     ("total_wl_kg", "Total weight loss (kg)"),
#     ("total_wl_%", "Total weight loss (%)"),
#     ("final_bmi", "Final BMI (kg/m²)"),
#     ("bmi_reduction", "BMI reduction (kg/m²)"),
#     ("last_aval_fat_%", "Last measured fat mass (%)"),
#     ("total_fat_loss_%", "Total fat mass loss (%)"),
#     ("last_aval_muscle_%", "Last measured muscle mass (%)"),
#     ("total_muscle_change_%", "Total muscle mass change (%)"),
#     ("instant_dropout", "Instant dropouts (n)"),
# ]
# # Add dynamic time window and WL target variables
# for w in TIME_WINDOWS:
#     ROW_ORDER += [
#         (f"{w}d_dropout", f"{w}-day dropouts (n)"),
#         (f"{w}d_wl_%", f"{w}-day weight loss (%)"),
#         (f"{w}d_bmi_reduction", f"{w}-day BMI reduction (kg/m²)"),
#         (f"{w}d_fat_loss_%", f"{w}-day fat mass loss (%)"),
#         (f"{w}d_muscle_change_%", f"{w}-day muscle mass change (%)"),
#     ]
# for t in WL_TARGETS:
#     ROW_ORDER += [
#         (f"{t}%_wl_achieved", f"Achieved {t}% weight loss (n)"),
#         (f"days_to_{t}%_wl", f"Days to {t}% weight loss"),
#     ]
# ROW_ORDER += [
#     # Weight gain causes
#     ("delim_wgc", "Self-reported causes of weight gain"),
#     ("womens_health_and_pregnancy", "Women's health and pregnancy (yes/no)"),
#     ("mental_health", "Mental health (yes/no)"),
#     ("family_issues", "Family issues (yes/no)"),
#     ("medication_disease_injury", "Medication, disease or injury (yes/no)"),
#     ("physical_inactivity", "Physical inactivity (yes/no)"),
#     ("eating_habits", "Eating habits (yes/no)"),
#     ("schedule", "Schedule (yes/no)"),
#     ("smoking_cessation", "Smoking cessation (yes/no)"),
#     ("treatment_discontinuation_or_relapse", "Treatment discontinuation or relapse (yes/no)"),
#     ("pandemic", "COVID-19 pandemic (yes/no)"),
#     ("lifestyle_circumstances", "Lifestyle circumstances (yes/no)"),
#     ("none_of_above", "None of the above (yes/no)"),
# ]

# =========================
# 2. HELPER FUNCTIONS
# =========================

def format_mean_sd(series):
    """Formats a series into 'mean ± SD' string, handling non-numeric data."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    mean = series.mean()
    sd = series.std()
    return f"{mean:.2f} ± {sd:.2f}"

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
    cause_cols_list = get_cause_cols(pretty_names)

    for var, pretty in pretty_names:
        if var.startswith("delim_wgc"):
            break
        if var in cause_cols_list:
            continue

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
# 3. PLOTTING FUNCTION
# =========================

def plot_datacollection_bias(df_cohort, df_mother, cohort_name, mother_cohort_name, output_file):
    """
    Generates and saves a bar plot comparing the distribution of medical record start years.
    """
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
# 4. STRATIFIED COMPARISON FUNCTIONS
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
        if var in cause_cols:
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
    summary_df.to_sql(config.wgc_output_table, conn, if_exists="replace", index=False)
    print(f"Weight gain cause stratification table saved to {config.wgc_output_table}")

# =========================
# 5. MAIN PIPELINE
# =========================

def run_descriptive_comparisons(master_config: master_config):
    """Main execution pipeline for all defined descriptive analyses."""
    input_db_path = master_config.paths.paper_in_db
    output_db_path = master_config.paths.paper_out_db

    for analysis_config in master_config.descriptive_comparisons:
        print(f"\nExecuting analysis: '{analysis_config.analysis_name}'")
        with sqlite3.connect(input_db_path) as conn_in:
            df_input = pd.read_sql_query(f"SELECT * FROM {analysis_config.input_cohort_name}", conn_in)
            df_mother = pd.read_sql_query(f"SELECT * FROM {analysis_config.mother_cohort_name}", conn_in)

        # Generate bias plot if configured
        if analysis_config.bias_plot_filename:
            plot_datacollection_bias(
                df_input.copy(), df_mother.copy(),
                analysis_config.input_cohort_name, analysis_config.mother_cohort_name,
                analysis_config.bias_plot_filename
            )
        else:
            print("Warning: 'baseline_date' column not found. Skipping bias plot generation.")

        # Run stratifications
        with sqlite3.connect(output_db_path) as conn_out:
            demographic_stratification(
                df_input, df_mother, 
                analysis_config,
                conn_out
            )
            wgc_stratification(
                df_input,
                analysis_config,
                conn_out
            )
    
    print("\n--- All Descriptive Analyses Complete ---")

