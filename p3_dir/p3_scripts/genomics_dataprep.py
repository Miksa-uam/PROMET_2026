"""
0. IMPORTS  
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta
from genomics_config import paths_config, timetoevent_config, master_config

# BREAKPOINTS: docstrings missing

"""
1. CREATE A RESEARCH PROJECT-SPECIFIC SQL DATABASE SUBSET
"""

def genomics_database_subset(config: master_config) -> None:
    """Create a Genomics-specific SQL database subset"""
    # Connect to the source database
    source_conn = sqlite3.connect(config.paths.source_db)

    # Identify patient-medical record combinations to include in the analysis, based on a configurable SQL filtering query
    sql_select_ids = config.filtering.filtering_sql_query

    records = pd.read_sql_query(sql_select_ids, source_conn)
    record_ids = tuple(records['medical_record_id'])
    patient_ids = tuple(records['patient_id'])

    # Pull rows from both tables corresponding to the identified records.
    # Create new table names measurements_genomics and medical_records_genomics.
    table_mapping = {
        "medical_records_filtered": ("medical_records_genomics", record_ids),
        "measurements_filtered": ("measurements_genomics", record_ids),
        "alleles_filtered": ("alleles_genomics", record_ids),
    }

    # Create new SQLite database connection to write filtered data.
    gen_in_conn = sqlite3.connect(config.paths.paper_in_db)

    for src_table, (dst_table, mr_ids) in table_mapping.items():
        # Prepare filtering query for tables that include 'medical_record_id'
        if src_table in ("medical_records_filtered", "measurements_filtered", "alleles_filtered"):
            query = f"""
                SELECT *
                FROM {src_table}
                WHERE medical_record_id IN {mr_ids}
            """
        else:
            # For other tables (if needed), one might filter by patient_id.
            query_check_column = f"PRAGMA table_info({src_table});"
            columns = pd.read_sql_query(query_check_column, source_conn)
            if 'patient_id' not in columns['name'].values:
                continue  # Skip if no patient_id column.
            query = f"""
                SELECT *
                FROM {src_table}
                WHERE patient_id IN {patient_ids}
            """
        # Execute the query and write into the output database.
        df_filtered = pd.read_sql_query(query, source_conn)
        df_filtered.to_sql(dst_table, gen_in_conn, index=False, if_exists="replace")
        print(f"Wrote {len(df_filtered)} rows from {src_table} to {dst_table}.")

    # Close both connections
    gen_in_conn.close()
    source_conn.close()

    # Print summary
    # print(f"Filtered data has been saved to {new_db_path}.")
    # print(f"Total first-record combinations: {len(records)} from {len(set(records['patient_id']))} patients.")

    print(f"Data has been saved to {config.paths.paper_in_db}.")
    print(f"Total records: {len(records)} from {len(set(records['patient_id']))} patients.")

"""
2. CREATE A TIME-TO-EVENT TYPE DATA TABLE IN THE PROJECT-SPECIFIC DATABASE SUBSET
"""

'''
2a. helper functions - final column ordering, data loading, identifying baseline measurements to merge with medical records 
'''

def make_column_order(config: timetoevent_config) -> list:
    """
    Generates the final column order based on the provided configuration.
    
    This function acts as the "schema builder" for the final output table. 
    It combines explicitly listed columns (metadata, clinical data) with 
    dynamically generated columns (fixed timepoints, time-to-target flags,
    and the new genomics event/post-personalization outcomes).
    
    If a column is calculated by the script but missing from this list, 
    it will be dropped before saving the database.
    """

    cols = config.metadata_columns.copy()
    cols += config.clinical_data_columns

    # 1. Standard fixed timepoint columns (anchored to baseline)
    for w in config.time_windows:
        prefix = f"{w}d"
        cols += [
            f"{prefix}_weight_kg", 
            f"{prefix}_wl_kg", 
            f"{prefix}_wl_%",
            f"{prefix}_wl_rate_kg_d",   # NEW: standard weight loss rate (kg/day)
            f"{prefix}_wl_rate_%_d",    # NEW: standard weight loss rate (%/day)
            f"{prefix}_bmi", f"{prefix}_bmi_reduction",
            f"{prefix}_fat_%", f"{prefix}_fat_loss_%",
            f"{prefix}_muscle_%", f"{prefix}_muscle_change_%",
            f"{prefix}_vat_%", f"{prefix}_vat_loss_%", # Added vat
            f"{prefix}_date", f"days_to_{prefix}_measurement", f"{prefix}_dropout"
        ]

    # 2. Time-to-target achievement columns
    for t in config.weight_loss_targets:
        prefix = f"{t}%_wl"
        cols += [
            f"{prefix}_achieved", f"{prefix}_%", f"{prefix}_date", f"days_to_{prefix}"
        ]

    # 3. Genomics timed outcomes (Snapshots at event dates)
    for event in ["purchase", "results"]:
        cols += [
            f"measurement_missing_at_genomics_{event}",
            f"measurement_date_at_genomics_{event}",
            f"weight_kg_at_genomics_{event}",
            f"wl_kg_at_genomics_{event}",
            f"wl_%_at_genomics_{event}",
            f"wl_rate_kg_d_at_genomics_{event}",
            f"wl_rate_%_d_at_genomics_{event}",
            f"bmi_at_genomics_{event}",
            f"bmi_reduction_at_genomics_{event}",
            f"fat_%_at_genomics_{event}",
            f"fat_loss_%_at_genomics_{event}",
            f"muscle_%_at_genomics_{event}",
            f"muscle_change_%_at_genomics_{event}",
            f"vat_%_at_genomics_{event}",
            f"vat_loss_%_at_genomics_{event}",
        ]
    # 4. Post-personalization FINAL, overall outcomes
    cols += [
        "post_personalization_final_wl_kg",
        "post_personalization_final_wl_%",
        "post_personalization_final_wl_rate_kg_d",
        "post_personalization_final_wl_rate_%_d",
        "post_personalization_final_bmi_reduction",
        "post_personalization_final_fat_loss_%",
        "post_personalization_final_muscle_change_%",
        "post_personalization_final_vat_loss_%",
        # "post_personalization_final_additional_wl_kg",
        "post_personalization_final_additional_wl_%",
        "post_personalization_final_additional_bmi_reduction",
        "post_personalization_final_additional_fat_loss_%",
        "post_personalization_final_additional_muscle_change_%",
        "post_personalization_final_additional_vat_loss_%",
    ]

    # 5. Post-personalization fixed timepoints (anchored to genomics results)
    for w in config.time_windows:
        prefix = f"post_personalization_{w}d"
        cols += [
            f"{prefix}_measurement_missing",
            f"{prefix}_measurement_date",
            f"days_to_{prefix}_measurement",
            f"{prefix}_weight_kg",

            # Change relative to personalization date:
            f"{prefix}_wl_kg",
            f"{prefix}_wl_%",
            f"{prefix}_wl_rate_kg_d",
            f"{prefix}_wl_rate_%_d",
            f"{prefix}_bmi", f"{prefix}_bmi_reduction",
            f"{prefix}_fat_%", f"{prefix}_fat_loss_%",
            f"{prefix}_muscle_%", f"{prefix}_muscle_change_%",
            f"{prefix}_vat_%", f"{prefix}_vat_loss_%",

            # Total cumulative change at P+X relative to baseline:
            f"{prefix}_total_wl_kg",
            f"{prefix}_total_wl_%",
            f"{prefix}_total_wl_rate_kg_d",
            f"{prefix}_total_wl_rate_%_d",

            # Increment/additional change accumulated AFTER personalization,
            # expressed as the difference between total WL at P+X and total WL at P:
            # f"{prefix}_additional_wl_kg",
            f"{prefix}_additional_wl_%",
            f"{prefix}_additional_wl_rate_kg_d",
            f"{prefix}_additional_wl_rate_%_d",
        ]

    return cols

def load_measurements(conn, config: timetoevent_config) -> pd.DataFrame:
    # Consider selecting only necessary columns if measurements_p1 has many unused ones
    cols_to_fetch = ["patient_id", "medical_record_id", "measurement_date", 
                      "first_in_record", "weight_kg", "bmi", "fat_%", "muscle_%", "vat_%"]
    cols_to_fetch = [f'"{col}"' for col in cols_to_fetch]  # Ensure proper quoting for SQL, as some columns have special characters
    # Check if any other columns from measurements_p1 are implicitly used. If not, this is safer.
    # If you are sure all columns are needed, use "SELECT *"
    # Query measurements from SQL table, getting table name directly from config object
    sql_query = f"SELECT {','.join(cols_to_fetch)} FROM {config.input_measurements}"
    df = pd.read_sql(sql_query, conn, parse_dates=["measurement_date"])
    df = df.sort_values(["patient_id", "medical_record_id", "measurement_date"])
    return df

def load_med_records(conn, config: timetoevent_config) -> pd.DataFrame:
    cols_to_fetch = ', '.join(config.fetch_from_records)
    # We explicitly tell pandas which columns are dates so it doesn't treat them as text
    date_columns = [
        "medical_record_creation_date", "medical_record_closing_date",
        "genomics_prescription_date", "genomics_purchase_date", 
        "genomics_sampling_date", "genomics_results_date"
    ]
    
    df = pd.read_sql(
        f"SELECT {cols_to_fetch} FROM {config.input_records}", 
        conn,
        parse_dates=date_columns
    )
    return df

def load_alleles(conn, config: timetoevent_config) -> pd.DataFrame:
    cols_to_fetch = ', '.join(config.fetch_from_alleles)
    # We explicitly tell pandas which columns are dates so it doesn't treat them as text
    date_columns = [
        "genomics_lab_date", "genomics_purchase_date",
        "genomics_sampling_date", "genomics_results_date", 
        "gdpr10_date",
    ]
    
    df = pd.read_sql(
        f"SELECT {cols_to_fetch} FROM {config.input_alleles}", 
        conn,
        parse_dates=date_columns
    )
    return df

def collapse_cumulative_risk_load(
    alleles_df: pd.DataFrame,
    expected_snp_count: int = 20,
    patient_id_col: str = "patient_id",
    genomics_sample_id_col: str = "genomics_sample_id",
    rs_id_col: str = "rs_id",
    risk_allele_count_col: str = "risk_load",
) -> pd.DataFrame:
    """
    Collapse long-format allele rows into one row per genomics sample, containing
    cumulative genetic risk-load metrics plus basic QC indicators.

    Purpose
    -------
    The alleles table is sample-centric rather than medical-record-centric:
    each patient has a single genomics sample, and allele information is stored
    as multiple rows per sample (typically one row per SNP). For the time-to-event
    table, this long allele structure must be reduced to a single record-level
    summary that can be merged onto the patient / medical-record scaffold.

    What this function does
    -----------------------
    1. Checks that the required columns exist.
    2. Removes rows with missing key identifiers.
    3. Verifies whether duplicate SNP rows exist within the same sample.
    4. Aggregates allele rows to one row per (patient_id, genomics_sample_id).
    5. Computes:
       - cumulative_risk_allele_count: sum of per-SNP risk allele counts
       - observed_snp_count: number of unique SNPs observed in that sample
       - risk_load_complete_20_snps: flag indicating whether the expected panel
         size was fully observed
       - risk_load_has_duplicate_snps: flag indicating duplicate SNP rows within sample

    Parameters
    ----------
    alleles_df : pd.DataFrame
        Long-format allele table loaded from SQL.
    expected_snp_count : int, default 20
        Expected number of SNPs contributing to the cumulative score.
    patient_id_col : str, default "patient_id"
        Patient identifier column.
    genomics_sample_id_col : str, default "genomics_sample_id"
        Genomics sample identifier column used as the sample-level safety key.
    rs_id_col : str, default "rs_id"
        SNP identifier column.
    risk_allele_count_col : str, default "risk_load"
        Numeric per-SNP contribution used in the cumulative score.

    Returns
    -------
    pd.DataFrame
        One row per (patient_id, genomics_sample_id), ready to merge into the
        merged baseline-record scaffold.
    """
    required_columns = [
        patient_id_col,
        genomics_sample_id_col,
        rs_id_col,
        risk_allele_count_col,
    ]
    missing_columns = [col for col in required_columns if col not in alleles_df.columns]
    if missing_columns:
        raise KeyError(
            f"collapse_cumulative_risk_load is missing required columns: {missing_columns}"
        )

    # Work on a copy to avoid mutating upstream objects.
    working_df = alleles_df.copy()

    # Remove rows that cannot be assigned confidently to a patient/sample.
    working_df = working_df.dropna(
        subset=[patient_id_col, genomics_sample_id_col, rs_id_col]
    ).copy()

    # Coerce the per-SNP contribution column to numeric.
    # Non-numeric values become NaN and are ignored in the sum.
    working_df[risk_allele_count_col] = pd.to_numeric(
        working_df[risk_allele_count_col],
        errors="coerce"
    )

    # Mark duplicate SNP entries within the same sample.
    # This is a QC problem because a given SNP should usually appear once per sample.
    working_df["is_duplicate_snp_within_sample"] = working_df.duplicated(
        subset=[patient_id_col, genomics_sample_id_col, rs_id_col],
        keep=False,
    ).astype(int)

    collapsed_df = (
        working_df.groupby([patient_id_col, genomics_sample_id_col], dropna=False)
        .agg(
            cumulative_risk_allele_count=(risk_allele_count_col, "sum"),
            observed_snp_count=(rs_id_col, "nunique"),
            non_missing_risk_allele_rows=(risk_allele_count_col, lambda s: s.notna().sum()),
            risk_load_has_duplicate_snps=("is_duplicate_snp_within_sample", "max"),
        )
        .reset_index()
    )

    # QC flag: was the full intended SNP panel observed?
    collapsed_df["risk_load_complete_20_snps"] = (
        collapsed_df["observed_snp_count"] == expected_snp_count
    ).astype(int)

    # Optional descriptive QC category can be useful during debugging and table checks.
    collapsed_df["risk_load_qc_status"] = np.select(
        [
            collapsed_df["risk_load_has_duplicate_snps"] == 1,
            collapsed_df["observed_snp_count"] < expected_snp_count,
            collapsed_df["observed_snp_count"] > expected_snp_count,
        ],
        [
            "duplicate_snps_within_sample",
            "incomplete_snp_panel",
            "unexpected_extra_snps",
        ],
        default="ok",
    )

    # Sanity check: one row per patient/sample after collapse.
    if collapsed_df.duplicated(subset=[patient_id_col, genomics_sample_id_col]).any():
        raise ValueError(
            "collapse_cumulative_risk_load produced duplicate patient/sample rows."
        )

    print(
        "Collapsed cumulative risk load for "
        f"{len(collapsed_df)} patient-sample combinations."
    )
    print(
        "QC summary:\n"
        f"{collapsed_df['risk_load_qc_status'].value_counts(dropna=False).to_string()}"
    )

    return collapsed_df

def extract_baseline(meas: pd.DataFrame) -> pd.DataFrame:
    base = meas[meas["first_in_record"] == 1].copy()
    base = base.rename(columns={
        "measurement_date": "baseline_measurement_date",
        "weight_kg": "baseline_weight_kg",
        "bmi": "baseline_bmi",
        "fat_%": "baseline_fat_%",
        "muscle_%": "baseline_muscle_%",
        "vat_%": "baseline_vat_%"
    })
    return base[[
        "patient_id", "medical_record_id",
        "baseline_measurement_date", "baseline_weight_kg", "baseline_bmi",
        "baseline_fat_%", "baseline_muscle_%", "baseline_vat_%"
    ]]

def merge_baseline_and_records(baseline, recs):
    df = baseline.merge(recs, on=["patient_id", "medical_record_id"], how="left")
    return df

'''
2b. time-to-event type calculations - overall followup, outcome-at-timestamp, time-to-target data
'''

# DOCSTRING MISSING
# when adding nadir calculations, it could probably come here
def calc_overall_followup(patient_record_measurements: pd.DataFrame, 
                          baseline_row_info: pd.Series
                          ) -> pd.Series:
    baseline_measurement_date = baseline_row_info["baseline_measurement_date"]
    
    followup_measurements = patient_record_measurements[
        patient_record_measurements["measurement_date"] > baseline_measurement_date
    ] # patient_record_measurements is already sorted by date

    last = baseline_row_info.copy() # Start with baseline info (which is a Series)
    last["instant_dropout"] = 0 # Initialize instant_dropout

    if followup_measurements.empty:
        last["final_measurement_date"] = baseline_row_info["baseline_measurement_date"]
        last["total_followup_days"] = 1
        last["instant_dropout"] = 1 # Set to 1 if only baseline measurement exists
        last["final_weight_kg"] = baseline_row_info["baseline_weight_kg"]
        last["total_wl_kg"], last["total_wl_%"] = 0.0, 0.0
        last["final_bmi"] = baseline_row_info["baseline_bmi"]
        last["bmi_reduction"] = 0.0
        last["final_fat_%"] = baseline_row_info["baseline_fat_%"]
        last["total_fat_loss_%"] = 0.0
        last["final_muscle_%"] = baseline_row_info["baseline_muscle_%"]
        last["total_muscle_change_%"] = 0.0
        last["final_vat_%"] = baseline_row_info["baseline_vat_%"]
        last["total_vat_loss_%"] = 0.0
    else:
        last_meas = followup_measurements.iloc[-1]
        dt = (last_meas.measurement_date - baseline_measurement_date).days + 1
        last["final_measurement_date"] = last_meas.measurement_date
        last["total_followup_days"] = dt
        last["final_weight_kg"] = last_meas.weight_kg
        last["total_wl_kg"] = last_meas.weight_kg - baseline_row_info["baseline_weight_kg"]
        last["total_wl_%"] = 100 * last["total_wl_kg"] / baseline_row_info["baseline_weight_kg"] if baseline_row_info["baseline_weight_kg"] else 0
        last["final_bmi"] = last_meas.bmi
        last["bmi_reduction"] = last_meas.bmi - baseline_row_info["baseline_bmi"]
        last["final_fat_%"] = last_meas["fat_%"]
        last["total_fat_loss_%"] = last_meas["fat_%"] - baseline_row_info["baseline_fat_%"]
        last["final_muscle_%"] = last_meas["muscle_%"]
        last["total_muscle_change_%"] = last_meas["muscle_%"] - baseline_row_info["baseline_muscle_%"]
        last["final_vat_%"] = last_meas["vat_%"]
        last["total_vat_loss_%"] = last_meas["vat_%"] - baseline_row_info["baseline_vat_%"]
    
    last["nr_total_measurements"] = len(patient_record_measurements)
    if last["nr_total_measurements"] > 1:
        # Denominator (last["nr_total_measurements"] - 1) cannot be zero here
        last["avg_days_between_measurements"] = \
            (last["total_followup_days"] - 1) / (last["nr_total_measurements"] - 1)
    else:
        last["avg_days_between_measurements"] = np.nan
    return last

def _calc_intermediate_outcomes(take: pd.Series,
                                   reference_date,
                                   reference_weight_kg,
                                   reference_bmi,
                                   reference_fat_pct,
                                   reference_muscle_pct,
                                   reference_vat_pct) -> dict:
    """    
    
    This is a helper function for fixed timepoint and genomics date-anchored intermediate analyses 
    (calc_fixed_timepoints and calc_genomics_timed_outcomes), 
    to calculate intermediate outcome values for a measurement entry, relative to a reference measurement. 
    For example, calculate weight loss at 120 days (entry in question: weight at 120 days) 
    relative to baseline (reference: baseline weight), 
    or calculate weight loss 30 days after personalization (entry: weight 30 days after personalization) 
    relative to personalization date (reference: genomics results date). 

    Importantly, this function does not pick which measurement entry of a patient to use 
    - the entry in question is identified with the next helper, _select_closest_measurement_within_window, 
    and the reference value is set in each use case. 
    This helper only calculates the difference between a predefined entry and its predefined reference. 

    What this helper does:
    1. Stores the selected measurement date.
    2. Calculates elapsed days from the reference date to that measurement.
    3. Calculates weight loss and weight-loss rates relative to the reference weight.
    4. Calculates BMI / body-composition deltas relative to the reference values.

    Sign convention:
    Weight loss is intentionally stored as NEGATIVE in this project.
    Therefore:
        wl_kg = current_weight - reference_weight
    So if current weight is lower than reference weight, wl_kg will be negative.

    Missing-value handling:
    - If weight is missing, weight-loss outputs become NaN.
    - If BMI / fat / muscle / VAT are missing in either the reference or the selected
      measurement, only that specific derived output becomes NaN.
    - This allows weight outcomes to remain available even when BC values are empty.
    """
    out = {}

    # The selected measurement date itself.
    meas_date = take["measurement_date"]
    out["measurement_date"] = meas_date

    # Elapsed time between reference and selected measurement.
    # +1 is used consistently in this project so that same-day measurements count as day 1.
    out["days_to_measurement"] = (
        (meas_date - reference_date).days + 1
        if pd.notna(reference_date) and pd.notna(meas_date)
        else np.nan
    )

    # Raw selected weight.
    weight_kg = take.get("weight_kg", np.nan)
    out["weight_kg"] = weight_kg

    # Weight-derived outcomes:
    # only compute if both reference weight and selected weight are available.
    # Also avoid division by zero if reference weight happens to be 0.
    if pd.notna(reference_weight_kg) and reference_weight_kg != 0 and pd.notna(weight_kg):
        wl_kg = weight_kg - reference_weight_kg
        wl_pct = wl_kg / reference_weight_kg * 100

        # The rate denominator should reflect actual elapsed days, not the nominal target day.
        # If elapsed days are somehow missing or non-positive, fall back to 1 to avoid division by zero.
        days_for_rate = (
            out["days_to_measurement"]
            if pd.notna(out["days_to_measurement"]) and out["days_to_measurement"] > 0
            else 1
        )

        out["wl_kg"] = wl_kg
        out["wl_%"] = wl_pct
        out["wl_rate_kg_d"] = wl_kg / days_for_rate
        out["wl_rate_%_d"] = wl_pct / days_for_rate
    else:
        out["wl_kg"] = np.nan
        out["wl_%"] = np.nan
        out["wl_rate_kg_d"] = np.nan
        out["wl_rate_%_d"] = np.nan

    # BMI at the selected measurement, and its change relative to the reference.
    bmi = take.get("bmi", np.nan)
    out["bmi"] = bmi
    out["bmi_reduction"] = bmi - reference_bmi if pd.notna(bmi) and pd.notna(reference_bmi) else np.nan

    # Fat percentage at the selected measurement, and change relative to reference.
    fat_pct = take.get("fat_%", np.nan)
    out["fat_%"] = fat_pct
    out["fat_loss_%"] = fat_pct - reference_fat_pct if pd.notna(fat_pct) and pd.notna(reference_fat_pct) else np.nan

    # Muscle percentage at the selected measurement, and change relative to reference.
    muscle_pct = take.get("muscle_%", np.nan)
    out["muscle_%"] = muscle_pct
    out["muscle_change_%"] = (
        muscle_pct - reference_muscle_pct
        if pd.notna(muscle_pct) and pd.notna(reference_muscle_pct)
        else np.nan
    )

    # VAT percentage at the selected measurement, and change relative to reference.
    vat_pct = take.get("vat_%", np.nan)
    out["vat_%"] = vat_pct
    out["vat_loss_%"] = vat_pct - reference_vat_pct if pd.notna(vat_pct) and pd.notna(reference_vat_pct) else np.nan

    return out

def _select_closest_measurement_within_window(patient_record_measurements: pd.DataFrame,
                                              target_date,
                                              window_span: int):
    """
    This helper will select the measurement closest to a target date, within a symmetric ± window.
    It does not calculate any outcomes, only chooses which measurement row should
    represent that target timepoint.

    Example:
    If target_date is day 30 and window_span is 10,
    this function looks between day 20 and day 40 inclusive.
    Or if target is weight at genomics results (start of personalization), 
    genomics results date is looked up and all measurements within a window of 10 days before and after
    the genomics results date are considered.

    Within a window, the measurement closest to the target date is chosen.

    Why this exists:
    Both fixed-timepoint outcomes and genomics-timed outcomes use the same rule:
    find the observed measurement nearest to the intended timepoint.

    Returns:
    - The chosen measurement row (a pandas Series), if at least one measurement exists in the window.
    - None, if no measurement falls inside the allowed window.

    Important:
    This function does not calculate any outcomes.
    It only chooses WHICH measurement row should represent that target timepoint.
    """
    if pd.isna(target_date):
        return None

    lo = target_date - timedelta(days=window_span)
    hi = target_date + timedelta(days=window_span)

    # Restrict to measurements falling inside the allowed time window.
    window_meas = patient_record_measurements[
        (patient_record_measurements["measurement_date"] >= lo) &
        (patient_record_measurements["measurement_date"] <= hi)
    ]

    if window_meas.empty:
        return None

    # Among all valid candidates, choose the one closest to the target date.
    window_meas = window_meas.copy()
    window_meas["dist_to_center"] = (
        window_meas["measurement_date"] - target_date
    ).abs().dt.days

    return window_meas.loc[window_meas["dist_to_center"].idxmin()]

def calc_fixed_timepoints(patient_record_measurements: pd.DataFrame,
                          baseline_row_info: pd.Series,
                          config: timetoevent_config) -> dict:
    """
    Calculate standard fixed-timepoint outcomes relative to BASELINE.

    This function is for the regular time-to-event table structure:
    for each configured follow-up window (e.g. 30d, 60d, 120d),
    find the observed measurement closest to that target day,
    within the allowed ± window span.

    Reference (for _calc_intermediate_outcomes):
    All outcomes here are relative to the baseline measurement.

    Output naming:
    This function preserves the project's original naming convention:
        30d_weight_kg
        30d_wl_kg
        30d_wl_%
        30d_wl_rate_kg_d
        30d_wl_rate_%_d
        ...
    """
    out = {}

    # Baseline values serve as the reference for all standard fixed-timepoint outcomes.
    baseline_measurement_date = baseline_row_info["baseline_measurement_date"]
    baseline_weight_kg = baseline_row_info["baseline_weight_kg"]
    baseline_bmi = baseline_row_info["baseline_bmi"]
    baseline_fat_pct = baseline_row_info["baseline_fat_%"]
    baseline_muscle_pct = baseline_row_info["baseline_muscle_%"]
    baseline_vat_pct = baseline_row_info["baseline_vat_%"]

    # Repeat the same logic for each configured target day.
    for w in config.time_windows:
        target_date = baseline_measurement_date + timedelta(days=w)
        prefix = f"{w}d"

        # Step 1:
        # identify the closest observed measurement within the allowed ± window.
        take = _select_closest_measurement_within_window(
            patient_record_measurements, target_date, config.window_span
        )

        # If no measurement exists in the allowed window,
        # this timepoint is treated as missing/dropout for this patient-record.
        if take is None:
            out[f"{prefix}_dropout"] = 1
            out[f"{prefix}_weight_kg"] = np.nan
            out[f"{prefix}_wl_kg"] = np.nan
            out[f"{prefix}_wl_%"] = np.nan
            out[f"{prefix}_wl_rate_kg_d"] = np.nan
            out[f"{prefix}_wl_rate_%_d"] = np.nan
            out[f"{prefix}_bmi"] = np.nan
            out[f"{prefix}_bmi_reduction"] = np.nan
            out[f"{prefix}_fat_%"] = np.nan
            out[f"{prefix}_fat_loss_%"] = np.nan
            out[f"{prefix}_muscle_%"] = np.nan
            out[f"{prefix}_muscle_change_%"] = np.nan
            out[f"{prefix}_vat_%"] = np.nan
            out[f"{prefix}_vat_loss_%"] = np.nan
            out[f"{prefix}_date"] = pd.NaT
            out[f"days_to_{prefix}_measurement"] = np.nan
            continue

        # Step 2:
        # calculate outcomes for the chosen measurement relative to baseline.
        metrics = _calc_intermediate_outcomes(
            take=take,
            reference_date=baseline_measurement_date,
            reference_weight_kg=baseline_weight_kg,
            reference_bmi=baseline_bmi,
            reference_fat_pct=baseline_fat_pct,
            reference_muscle_pct=baseline_muscle_pct,
            reference_vat_pct=baseline_vat_pct,
        )

        # Step 3:
        # write the results into the output dictionary using the established naming scheme.
        out[f"{prefix}_dropout"] = 0
        out[f"{prefix}_weight_kg"] = metrics["weight_kg"]
        out[f"{prefix}_wl_kg"] = metrics["wl_kg"]
        out[f"{prefix}_wl_%"] = metrics["wl_%"]
        out[f"{prefix}_wl_rate_kg_d"] = metrics["wl_rate_kg_d"]
        out[f"{prefix}_wl_rate_%_d"] = metrics["wl_rate_%_d"]
        out[f"{prefix}_bmi"] = metrics["bmi"]
        out[f"{prefix}_bmi_reduction"] = metrics["bmi_reduction"]
        out[f"{prefix}_fat_%"] = metrics["fat_%"]
        out[f"{prefix}_fat_loss_%"] = metrics["fat_loss_%"]
        out[f"{prefix}_muscle_%"] = metrics["muscle_%"]
        out[f"{prefix}_muscle_change_%"] = metrics["muscle_change_%"]
        out[f"{prefix}_vat_%"] = metrics["vat_%"]
        out[f"{prefix}_vat_loss_%"] = metrics["vat_loss_%"]
        out[f"{prefix}_date"] = metrics["measurement_date"]
        out[f"days_to_{prefix}_measurement"] = metrics["days_to_measurement"]

    return out

# DOCSTRING MISSING 
def calc_time_to_targets(patient_record_measurements: pd.DataFrame, 
                         baseline_row_info: pd.Series, 
                         config: timetoevent_config) -> dict:
    out = {}
    baseline_measurement_date = baseline_row_info["baseline_measurement_date"]
    baseline_weight = baseline_row_info["baseline_weight_kg"]

    # Filter measurements strictly after baseline date; patient_record_measurements is already sorted
    group = patient_record_measurements[patient_record_measurements["measurement_date"] > baseline_measurement_date]

    if group.empty or baseline_weight == 0: # Added check for baseline_weight to prevent division by zero
        for t in config.weight_loss_targets:
            out[f"{t}%_wl_achieved"], out[f"{t}%_wl_%"] = 0, np.nan
            out[f"{t}%_wl_date"], out[f"days_to_{t}%_wl"] = pd.NaT, np.nan
        return out

    # Calculate weight loss percentage for all measurements in the group vectorially
    # Use .copy() to avoid SettingWithCopyWarning if 'group' is a slice
    group_copy = group.copy()
    group_copy["wl_pct_calculated"] = 100 * (baseline_weight - group_copy["weight_kg"]) / baseline_weight
    
    for t in config.weight_loss_targets:
        # Find the first measurement (due to sort order) where target is achieved
        achieved_measurements = group_copy[group_copy["wl_pct_calculated"] >= t]
        
        if not achieved_measurements.empty:
            first_achieved_event = achieved_measurements.iloc[0]
            out[f"{t}%_wl_achieved"] = 1
            out[f"{t}%_wl_%"] = first_achieved_event["wl_pct_calculated"]
            out[f"{t}%_wl_date"] = first_achieved_event["measurement_date"]
            out[f"days_to_{t}%_wl"] = (first_achieved_event["measurement_date"] - baseline_measurement_date).days + 1
        else:
            out[f"{t}%_wl_achieved"] = 0
            out[f"{t}%_wl_%"] = np.nan 
            out[f"{t}%_wl_date"] = pd.NaT
            out[f"days_to_{t}%_wl"] = np.nan
    return out

def calc_genomics_timing(record_info: pd.Series, overall_followup_dict: dict) -> dict:
    """
    Returns genomics timing flags based on record-level dates.
    Returns all NaN if genomics_results_date is missing or if record dates are missing.
    Calculations are based on 'genomics results date': 
    the date when the patient receives the results of their genetic test results, and from then on, the treatment is personalized. 
    This can be: 
    - within the medical record's validity
    - - 3 weeks or X defined days within the start of the record (ie. most of the record is personalized)
    - before the record's validity (ie. it was available from the start, so the whole record is personalized)
    - after the record's end: the record is not personalized, but its outcomes can be interpreted in light of the patient's genotype
    - special case: some patients did a genetics test, but the results date is not available - the temporal relationship is unknown
    
    The above is done both for the administrative validity period of a medical record, and the observed engagement period of a patient, 
    ie. the period between their first and last measurements within an administratively active medical record.

    Other variables are calculated to identify the time passed between the opening of a record to the first measurement, 
    the first measurement to the purchase of a genetic test, purchase to sampling, sampling to results, results to last measurement, 
    etc, to understand the dynamics of personalization reception within each individual treatment cycle. 
    This is used to propose and test better hypotheses regarding the effect of personalized prescription and its timing on adherence.         
    """

    out = {
        'no_genomics_results_date': np.nan,
        # Relationship of genomics timing with medical record validity period - the time when the record was open on paper
        'genomics_within_record': np.nan,
        'genomics_3wk_within_record_start': np.nan,
        'genomics_1mo_within_record_start': np.nan,
        'genomics_2mo_within_record_start': np.nan,
        'genomics_before_record': np.nan,
        'genomics_after_record': np.nan,
        # Relationship of genomics timing with period of observed active measurements - the time when the patient was actively measuring themselves
        'genomics_within_observation_period': np.nan,
        'genomics_3wk_within_1st_measurement': np.nan,
        'genomics_1mo_within_1st_measurement': np.nan,
        'genomics_2mo_within_1st_measurement': np.nan,
        'genomics_before_1st_measurement': np.nan,
        'genomics_after_last_measurement': np.nan,

        # Record administration to measurement gaps
        "record_start_to_first_measurement_d": np.nan,
        "record_start_to_genomics_results_d": np.nan,
        "first_measurement_to_genomics_results_d": np.nan,
        "last_measurement_to_record_end_d": np.nan,

        # Detailed genomics pipeline delivery gaps
        "record_start_to_genomics_prescription_d": np.nan,
        "genomics_prescription_to_purchase_d": np.nan,
        "genomics_purchase_to_sampling_d": np.nan,
        "genomics_sampling_to_lab_d": np.nan,
        "genomics_lab_to_results_d": np.nan,
        "genomics_sampling_to_results_d": np.nan,
        "first_measurement_to_genomics_purchase_d": np.nan,
        "genomics_results_to_last_measurement_d": np.nan,
    }

    # Get medical record, genomics, and measurement dates
    record_start = record_info.get('medical_record_creation_date') # administrative record start
    record_end = record_info.get('medical_record_closing_date') # administrative record end
    prescription_date = record_info.get('genomics_prescription_date') # prescription of genetic test - 99% on the day of record start
    purchase_date = record_info.get('genomics_purchase_date') # purchase of genetic test - depends on the patient
    sampling_date = record_info.get('genomics_sampling_date')  # taking the buccal swab - depends on both logistics and the patient
    lab_date = record_info.get('genomics_lab_date') # lab analysis of the sample - depends on logistics and the patient (if there are delays in sending the sample)
    results_date = record_info.get('genomics_results_date') # results available, start of personalization - 99% same as lab date
    baseline_meas = record_info.get('baseline_measurement_date') # first measurement's date
    final_meas = overall_followup_dict.get('final_measurement_date') # last measurement's date

    out['no_genomics_results_date'] = int(pd.isna(results_date))

    # Record administration to measurement gaps
    if pd.notna(record_start) and pd.notna(baseline_meas):
        out['record_start_to_first_measurement_d'] = (baseline_meas - record_start).days
        
    if pd.notna(record_start) and pd.notna(results_date):
        out['record_start_to_genomics_results_d'] = (results_date - record_start).days
        
    if pd.notna(baseline_meas) and pd.notna(results_date):
        out['first_measurement_to_genomics_results_d'] = (results_date - baseline_meas).days

    if pd.notna(final_meas) and pd.notna(record_end):
        out['last_measurement_to_record_end_d'] = (record_end - final_meas).days

    # Detailed genomics pipeline delivery gaps
    if pd.notna(record_start) and pd.notna(prescription_date):
        out['record_start_to_genomics_prescription_d'] = (prescription_date - record_start).days

    if pd.notna(prescription_date) and pd.notna(purchase_date):
        out['genomics_prescription_to_purchase_d'] = (purchase_date - prescription_date).days

    if pd.notna(purchase_date) and pd.notna(sampling_date):
        out['genomics_purchase_to_sampling_d'] = (sampling_date - purchase_date).days

    if pd.notna(sampling_date) and pd.notna(lab_date):
        out['genomics_sampling_to_lab_d'] = (lab_date - sampling_date).days

    if pd.notna(lab_date) and pd.notna(results_date):
        out['genomics_lab_to_results_d'] = (results_date - lab_date).days

    if pd.notna(sampling_date) and pd.notna(results_date):
        out['genomics_sampling_to_results_d'] = (results_date - sampling_date).days

    if pd.notna(baseline_meas) and pd.notna(purchase_date):
        out['first_measurement_to_genomics_purchase_d'] = (purchase_date - baseline_meas).days

    if pd.notna(results_date) and pd.notna(final_meas):
        out['genomics_results_to_last_measurement_d'] = (final_meas - results_date).days

    # If any required date is missing, return NaNs for the flags
    if pd.isna(results_date) or pd.isna(baseline_meas) or pd.isna(final_meas):
        return out

    within_record = (results_date >= record_start) and (results_date <= record_end)
    out['genomics_within_record'] = int(within_record)
    out['genomics_before_record'] = int(results_date < record_start)
    out['genomics_after_record'] = int(results_date > record_end)

    if within_record:
        cutoff_21d = record_start + pd.Timedelta(days=21)
        out['genomics_3wk_within_record_start'] = int(results_date <= cutoff_21d)
        
        cutoff_30d = record_start + pd.Timedelta(days=30)
        out['genomics_1mo_within_record_start'] = int(results_date <= cutoff_30d)

        cutoff_60d = record_start + pd.Timedelta(days=60)
        out['genomics_2mo_within_record_start'] = int(results_date <= cutoff_60d)

    within_observation_period = (results_date >= baseline_meas) and (results_date <= final_meas)
    out['genomics_within_observation_period'] = int(within_observation_period)  # Fixed: was using within_record
    out['genomics_before_1st_measurement'] = int(results_date < baseline_meas)
    out['genomics_after_last_measurement'] = int(results_date > final_meas)

    if within_observation_period:
        cutoff_21d = baseline_meas + pd.Timedelta(days=21)
        out['genomics_3wk_within_1st_measurement'] = int(results_date <= cutoff_21d)
        
        cutoff_30d = baseline_meas + pd.Timedelta(days=30)
        out['genomics_1mo_within_1st_measurement'] = int(results_date <= cutoff_30d)

        cutoff_60d = baseline_meas + pd.Timedelta(days=60)
        out['genomics_2mo_within_1st_measurement'] = int(results_date <= cutoff_60d)

    return out

def calc_genomics_timed_outcomes(patient_record_measurements: pd.DataFrame,
                                 baseline_row_info: pd.Series,
                                 config: timetoevent_config) -> dict:
    """
    Calculate intermediate outcomes relative to the timing of personalization, in two parts.

    PART 1 — Event snapshots
    ------------------------
    Find the measurement closest to each genomics event date:
    - genomics purchase (this is when the patient orders the genetic test, opting in for personalization)
    - genomics results (this is the start of personalization)

    These outputs describe the patient's observed weight around those event dates.

    PART 2 — Post-personalization fixed timepoints
    ----------------------------------------------
    If genomics results are available, treat the results date as the start of personalization,
    then calculate outcomes at fixed windows AFTER that date:
        post_personalization_30d_wl_kg
        post_personalization_60d_*
        etc.

    Key missing-data rule:
    - If an event date itself is missing, related outputs are left NULL.
    - If the event exists but no measurement is found inside the allowed ± window,
      then measurement_missing = 1 for that event/timepoint.

    This distinction matters:
    "event not observed" is not the same thing as
    "event observed, but no nearby measurement available."
    """
    out = {}

    # Baseline values are needed because:
    # 1) event snapshots are interpreted relative to baseline
    # 2) post-personalization "additional" outcomes are also baseline-referenced
    baseline_measurement_date = baseline_row_info.get("baseline_measurement_date")
    baseline_weight_kg = baseline_row_info.get("baseline_weight_kg")
    baseline_bmi = baseline_row_info.get("baseline_bmi")
    baseline_fat_pct = baseline_row_info.get("baseline_fat_%")
    baseline_muscle_pct = baseline_row_info.get("baseline_muscle_%")
    baseline_vat_pct = baseline_row_info.get("baseline_vat_%")

    # Without patient measurements or a baseline date, none of these calculations are meaningful.
    if patient_record_measurements.empty or pd.isna(baseline_measurement_date):
        return out

    # The two genomics events of interest.
    event_dates = {
        "purchase": baseline_row_info.get("genomics_purchase_date"),
        "results": baseline_row_info.get("genomics_results_date"),
    }

    # We store the full metric set for each event here,
    # because the "results" snapshot later becomes the anchor for post-personalization analyses.
    event_metrics = {}

    # ------------------------------------------------------------------
    # PART 1: outcomes at genomics purchase / genomics results
    # ------------------------------------------------------------------
    for event_name, event_date in event_dates.items():
        # If the event date itself is missing, leave all related outputs NULL.
        # We intentionally do NOT set measurement_missing = 1 here,
        # because this is not a measurement-window problem; the event itself is unknown.
        if pd.isna(event_date):
            continue

        # Find the observed measurement closest to the event date, within ± config.window_span days.
        take = _select_closest_measurement_within_window(
            patient_record_measurements, event_date, config.window_span
        )

        # Here measurement_missing means:
        # the event exists, but there is no measurement close enough to represent it.
        out[f"measurement_missing_at_genomics_{event_name}"] = 1 if take is None else 0

        if take is None:
            continue

        # Calculate event-linked outcomes relative to baseline.
        # This tells us the patient's state around the purchase/results event.
        metrics = _calc_intermediate_outcomes(
            take=take,
            reference_date=baseline_measurement_date,
            reference_weight_kg=baseline_weight_kg,
            reference_bmi=baseline_bmi,
            reference_fat_pct=baseline_fat_pct,
            reference_muscle_pct=baseline_muscle_pct,
            reference_vat_pct=baseline_vat_pct,
        )
        event_metrics[event_name] = metrics

        # Save the raw measurement date and values around the event.
        out[f"measurement_date_at_genomics_{event_name}"] = metrics["measurement_date"]
        out[f"weight_kg_at_genomics_{event_name}"] = metrics["weight_kg"]

        # Save weight-loss outcomes around the event.
        out[f"wl_kg_at_genomics_{event_name}"] = metrics["wl_kg"]
        out[f"wl_%_at_genomics_{event_name}"] = metrics["wl_%"]
        out[f"wl_rate_kg_d_at_genomics_{event_name}"] = metrics["wl_rate_kg_d"]
        out[f"wl_rate_%_d_at_genomics_{event_name}"] = metrics["wl_rate_%_d"]

        # Save BMI and body-composition values around the event.
        out[f"bmi_at_genomics_{event_name}"] = metrics["bmi"]
        out[f"bmi_reduction_at_genomics_{event_name}"] = metrics["bmi_reduction"]

        out[f"fat_%_at_genomics_{event_name}"] = metrics["fat_%"]
        out[f"fat_loss_%_at_genomics_{event_name}"] = metrics["fat_loss_%"]

        out[f"muscle_%_at_genomics_{event_name}"] = metrics["muscle_%"]
        out[f"muscle_change_%_at_genomics_{event_name}"] = metrics["muscle_change_%"]

        out[f"vat_%_at_genomics_{event_name}"] = metrics["vat_%"]
        out[f"vat_loss_%_at_genomics_{event_name}"] = metrics["vat_loss_%"]

    # ------------------------------------------------------------------
    # PART 2: post-personalization fixed timepoints relative to the start 
    # of personalization (reception of genomics results)
    # ------------------------------------------------------------------

    # Personalization is defined as starting at the reception of genomics results.
    results_date = event_dates["results"]

    # If results date is missing, post-personalization outcomes cannot be defined.
    if pd.isna(results_date):
        return out

    # To compute post-personalization change, we also need a usable "state at results".
    # That means we need an actual measurement near the results date.
    results_metrics = event_metrics.get("results")
    if results_metrics is None or pd.isna(results_metrics.get("weight_kg", np.nan)):
        return out

    # These "results_*" values become the reference for post-personalization outcomes.
    results_weight_kg = results_metrics["weight_kg"]
    results_bmi = results_metrics["bmi"]
    results_fat_pct = results_metrics["fat_%"]
    results_muscle_pct = results_metrics["muscle_%"]
    results_vat_pct = results_metrics["vat_%"]

    # Repeat the same fixed-window logic, but now the reference is genomics results date,
    # not baseline.
    for w in config.time_windows:
        prefix = f"post_personalization_{w}d"
        target_date = results_date + timedelta(days=w)

        take = _select_closest_measurement_within_window(
            patient_record_measurements, target_date, config.window_span
        )

        if take is None:
            out[f"{prefix}_measurement_missing"] = 1
            continue

        out[f"{prefix}_measurement_missing"] = 0

        # 1) Post-personalization metrics:
        # change from personalization date to P+X.
        post_metrics = _calc_intermediate_outcomes(
            take=take,
            reference_date=results_date,
            reference_weight_kg=results_weight_kg,
            reference_bmi=results_bmi,
            reference_fat_pct=results_fat_pct,
            reference_muscle_pct=results_muscle_pct,
            reference_vat_pct=results_vat_pct,
        )

        # 2) Total cumulative metrics at P+X:
        # change from baseline to the same later measurement.
        total_metrics = _calc_intermediate_outcomes(
            take=take,
            reference_date=baseline_measurement_date,
            reference_weight_kg=baseline_weight_kg,
            reference_bmi=baseline_bmi,
            reference_fat_pct=baseline_fat_pct,
            reference_muscle_pct=baseline_muscle_pct,
            reference_vat_pct=baseline_vat_pct,
        )

        # 3) Additional/incremental metrics:
        # difference between total WL at P+X and total WL at personalization,
        # both expressed relative to baseline.
        additional_wl_kg = np.nan
        additional_wl_pct = np.nan
        additional_wl_rate_kg_d = np.nan
        additional_wl_rate_pct_d = np.nan

        total_wl_at_personalization_kg = results_metrics.get("wl_kg", np.nan)
        total_wl_at_personalization_pct = results_metrics.get("wl_%", np.nan)

        if pd.notna(total_metrics.get("wl_kg", np.nan)) and pd.notna(total_wl_at_personalization_kg):
            additional_wl_kg = total_metrics["wl_kg"] - total_wl_at_personalization_kg

        if pd.notna(total_metrics.get("wl_%", np.nan)) and pd.notna(total_wl_at_personalization_pct):
            additional_wl_pct = total_metrics["wl_%"] - total_wl_at_personalization_pct

        days_after_personalization = post_metrics.get("days_to_measurement", np.nan)
        if pd.notna(days_after_personalization) and days_after_personalization > 0:
            if pd.notna(additional_wl_kg):
                additional_wl_rate_kg_d = additional_wl_kg / days_after_personalization
            if pd.notna(additional_wl_pct):
                additional_wl_rate_pct_d = additional_wl_pct / days_after_personalization

        # Save post-personalization outcomes relative to genomics results.
        out[f"{prefix}_measurement_date"] = post_metrics["measurement_date"]
        out[f"{prefix}_weight_kg"] = post_metrics["weight_kg"]
        out[f"{prefix}_wl_kg"] = post_metrics["wl_kg"]
        out[f"{prefix}_wl_%"] = post_metrics["wl_%"]
        out[f"{prefix}_wl_rate_kg_d"] = post_metrics["wl_rate_kg_d"]
        out[f"{prefix}_wl_rate_%_d"] = post_metrics["wl_rate_%_d"]
        out[f"{prefix}_bmi"] = post_metrics["bmi"]
        out[f"{prefix}_bmi_reduction"] = post_metrics["bmi_reduction"]
        out[f"{prefix}_fat_%"] = post_metrics["fat_%"]
        out[f"{prefix}_fat_loss_%"] = post_metrics["fat_loss_%"]
        out[f"{prefix}_muscle_%"] = post_metrics["muscle_%"]
        out[f"{prefix}_muscle_change_%"] = post_metrics["muscle_change_%"]
        out[f"{prefix}_vat_%"] = post_metrics["vat_%"]
        out[f"{prefix}_vat_loss_%"] = post_metrics["vat_loss_%"]
        out[f"days_to_{prefix}_measurement"] = post_metrics["days_to_measurement"]

        # Save total cumulative outcomes at P+X relative to baseline.
        out[f"{prefix}_total_wl_kg"] = total_metrics["wl_kg"]
        out[f"{prefix}_total_wl_%"] = total_metrics["wl_%"]
        out[f"{prefix}_total_wl_rate_kg_d"] = total_metrics["wl_rate_kg_d"]
        out[f"{prefix}_total_wl_rate_%_d"] = total_metrics["wl_rate_%_d"]

        # Save additional/incremental outcomes accumulated after personalization,
        # but still expressed relative to baseline-loss space.
        out[f"{prefix}_additional_wl_kg"] = additional_wl_kg
        out[f"{prefix}_additional_wl_%"] = additional_wl_pct
        out[f"{prefix}_additional_wl_rate_kg_d"] = additional_wl_rate_kg_d
        out[f"{prefix}_additional_wl_rate_%_d"] = additional_wl_rate_pct_d

    return out

def calc_post_personalization_final_outcomes(out_dict: dict) -> dict:
    """
    Derive post-personalization final endpoint outcomes from already computed values.
    These are derived from the weight at personalization and the last overall weight measured. 

    Uses:
    - overall follow-up final values / total cumulative changes
    - outcomes at genomics results (start of personalization)
    - time from genomics results to final measurement

    Returns only derived endpoint variables for the period from personalization
    to the last observed measurement.
    """
    out = {}

    days = out_dict.get("genomics_results_to_last_measurement_d", np.nan)

    final_weight = out_dict.get("final_weight_kg", np.nan)
    weight_at_personalization = out_dict.get("weight_kg_at_genomics_results", np.nan)

    total_wl_kg = out_dict.get("total_wl_kg", np.nan)
    wl_kg_at_personalization = out_dict.get("wl_kg_at_genomics_results", np.nan)

    total_wl_pct = out_dict.get("total_wl_%", np.nan)
    wl_pct_at_personalization = out_dict.get("wl_%_at_genomics_results", np.nan)

    final_bmi = out_dict.get("final_bmi", np.nan)
    bmi_at_personalization = out_dict.get("bmi_at_genomics_results", np.nan)
    total_bmi_reduction = out_dict.get("bmi_reduction", np.nan)
    bmi_reduction_at_personalization = out_dict.get("bmi_reduction_at_genomics_results", np.nan)

    final_fat = out_dict.get("final_fat_%", np.nan)
    fat_at_personalization = out_dict.get("fat_%_at_genomics_results", np.nan)
    total_fat_loss = out_dict.get("total_fat_loss_%", np.nan)
    fat_loss_at_personalization = out_dict.get("fat_loss_%_at_genomics_results", np.nan)

    final_muscle = out_dict.get("final_muscle_%", np.nan)
    muscle_at_personalization = out_dict.get("muscle_%_at_genomics_results", np.nan)
    total_muscle_change = out_dict.get("total_muscle_change_%", np.nan)
    muscle_change_at_personalization = out_dict.get("muscle_change_%_at_genomics_results", np.nan)

    final_vat = out_dict.get("final_vat_%", np.nan)
    vat_at_personalization = out_dict.get("vat_%_at_genomics_results", np.nan)
    total_vat_loss = out_dict.get("total_vat_loss_%", np.nan)
    vat_loss_at_personalization = out_dict.get("vat_loss_%_at_genomics_results", np.nan)

    if pd.isna(days) or days <= 0:
        return out
    if pd.isna(final_weight) or pd.isna(weight_at_personalization):
        return out

    # Post-personalization final change
    out["post_personalization_final_wl_kg"] = final_weight - weight_at_personalization
    out["post_personalization_final_wl_%"] = (
        100 * out["post_personalization_final_wl_kg"] / weight_at_personalization
        if pd.notna(weight_at_personalization) and weight_at_personalization != 0
        else np.nan
    )
    out["post_personalization_final_wl_rate_kg_d"] = out["post_personalization_final_wl_kg"] / days
    out["post_personalization_final_wl_rate_%_d"] = out["post_personalization_final_wl_%"] / days

    out["post_personalization_final_bmi_reduction"] = (
        final_bmi - bmi_at_personalization
        if pd.notna(final_bmi) and pd.notna(bmi_at_personalization)
        else np.nan
    )
    out["post_personalization_final_fat_loss_%"] = (
        final_fat - fat_at_personalization
        if pd.notna(final_fat) and pd.notna(fat_at_personalization)
        else np.nan
    )
    out["post_personalization_final_muscle_change_%"] = (
        final_muscle - muscle_at_personalization
        if pd.notna(final_muscle) and pd.notna(muscle_at_personalization)
        else np.nan
    )
    out["post_personalization_final_vat_loss_%"] = (
        final_vat - vat_at_personalization
        if pd.notna(final_vat) and pd.notna(vat_at_personalization)
        else np.nan
    )

    # Additional post-personalization change in baseline-referenced space: 
    # additional weight lost relative to what was already lost by the start of personalization
    out["post_personalization_final_additional_wl_kg"] = (
        total_wl_kg - wl_kg_at_personalization
        if pd.notna(total_wl_kg) and pd.notna(wl_kg_at_personalization)
        else np.nan
    )
    out["post_personalization_final_additional_wl_%"] = (
        total_wl_pct - wl_pct_at_personalization
        if pd.notna(total_wl_pct) and pd.notna(wl_pct_at_personalization)
        else np.nan
    )
    out["post_personalization_final_additional_bmi_reduction"] = (
        total_bmi_reduction - bmi_reduction_at_personalization
        if pd.notna(total_bmi_reduction) and pd.notna(bmi_reduction_at_personalization)
        else np.nan
    )
    out["post_personalization_final_additional_fat_loss_%"] = (
        total_fat_loss - fat_loss_at_personalization
        if pd.notna(total_fat_loss) and pd.notna(fat_loss_at_personalization)
        else np.nan
    )
    out["post_personalization_final_additional_muscle_change_%"] = (
        total_muscle_change - muscle_change_at_personalization
        if pd.notna(total_muscle_change) and pd.notna(muscle_change_at_personalization)
        else np.nan
    )
    out["post_personalization_final_additional_vat_loss_%"] = (
        total_vat_loss - vat_loss_at_personalization
        if pd.notna(total_vat_loss) and pd.notna(vat_loss_at_personalization)
        else np.nan
    )

    return out

def process_country_dummies(df: pd.DataFrame, output_column_order: list) -> tuple[pd.DataFrame, list, list]:
    """
    Processes country column to create dummy variables and updates the output_column_order.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the 'country' column.
        output_column_order (list): The list defining the desired order of output columns.
        
    Returns:
        tuple[pd.DataFrame, list, list]: The modified DataFrame, updated output_column_order, and a list of generated dummy column names.
    """
    dummy_cols = []
    if "country" in df.columns:
        dummies = pd.get_dummies(df["country"], prefix="country", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        dummy_cols = sorted(dummies.columns.tolist())
        
        # Inject dummy columns into output_column_order right after 'country'
        if "country" in output_column_order:
            idx = output_column_order.index("country")
            # Insert in reverse order so they end up in correct forward order A, B, C...
            for col in reversed(dummy_cols):
                output_column_order.insert(idx + 1, col)
        else:
            # If country not in output list for some reason, append them
            output_column_order.extend(dummy_cols)
            
    return df, output_column_order, dummy_cols


'''
2c. orchestration function - bring everything together
'''

def build_timetoevent_table(config: master_config):
    # 1. Get configs from the master object
    paths_config = config.paths
    timetoevent_config = config.timetoevent

    # Check if the required config is present
    if not timetoevent_config:
        print("Time-to-event configuration not provided. Skipping.")
        return

    print(f"Connecting to source database: {paths_config.source_db}")
    conn_in = sqlite3.connect(paths_config.source_db)

    # 2. Call refactored functions, passing the config
    all_measurements = load_measurements(conn_in, timetoevent_config)
    medical_records_data = load_med_records(conn_in, timetoevent_config)
    alleles_data = load_alleles(conn_in, timetoevent_config)
    conn_in.close()

    baseline_data = extract_baseline(all_measurements)
    merged_patient_records = merge_baseline_and_records(baseline_data, medical_records_data)

    # alleles_data_agg = alleles_data.groupby('patient_id').agg({
    #     'genomics_lab_date': 'first',  # assumes same date across SNPs
    # }).reset_index()
    # # Verify before merging:
    # assert alleles_data_agg["patient_id"].is_unique, \
    #     "alleles_data_agg has duplicate patient_ids — agg did not collapse correctly"

    # merged_patient_records = merged_patient_records.merge(alleles_data_agg, on=['patient_id'], how='left')


    # Collapse long-format allele rows to one row per patient/genomics sample.
    # This replaces the earlier patient-level genomics_lab_date-only aggregation.
    alleles_data_agg = collapse_cumulative_risk_load(alleles_data)

    # Verify before merging:
    assert alleles_data_agg[["patient_id", "genomics_sample_id"]].duplicated().sum() == 0, \
        "alleles_data_agg has duplicate (patient_id, genomics_sample_id) combinations — collapse did not work correctly"

    # Merge sample-level allele summary onto the record-centric scaffold.
    # The merge is many-to-one because one genomics sample may map to more than one medical record.
    merged_patient_records = merged_patient_records.merge(
        alleles_data_agg,
        on=["patient_id", "genomics_sample_id"],
        how="left",
        validate="many_to_one",
    )

    
    merged_patient_records_indexed = merged_patient_records.set_index(["patient_id", "medical_record_id"])
    assert merged_patient_records_indexed.index.is_unique, \
        "Duplicate (patient_id, medical_record_id) combinations after merge — check for fan-out in alleles merge"
    # Generate output column order dynamically
    output_column_order = make_column_order(timetoevent_config)

    output_rows = []
    # Group measurements ONCE, then iterate through groups
    # This is the core performance improvement for the loop structure.
    grouped_measurements = all_measurements.groupby(["patient_id", "medical_record_id"])

    for (patient_id, medical_record_id), patient_group_measurements in grouped_measurements:
        # patient_group_measurements is a DataFrame for the current patient-record, already sorted.   
        
        try: 
            # Efficiently get the corresponding baseline and medical record info (it's a Series)
            record_info_with_baseline = merged_patient_records_indexed.loc[(patient_id, medical_record_id)]
        except KeyError:
            # This happens if a patient-record group exists in measurements but not in merged_patient_records
            # (e.g. no baseline found, or no medical record entry)
            print(f"Warning: No baseline/record info for patient {patient_id}, record {medical_record_id}. Skipping.")
            continue # Skip this group if no baseline/record info

        # 1) overall follow‐up
        overall_followup_series = calc_overall_followup(patient_group_measurements, record_info_with_baseline)
        out_dict = overall_followup_series.to_dict() # Contains baseline fields + calculated overall fields
        # Manually add the patient_id, medical_record_id and genomics sample id from the groupby keys
        out_dict["patient_id"] = patient_id
        out_dict["medical_record_id"] = medical_record_id
        out_dict["genomics_sample_id"] = record_info_with_baseline.get("genomics_sample_id", None)

        # 2) fixed timepoints
        fixed_timepoints = calc_fixed_timepoints(patient_group_measurements, record_info_with_baseline, timetoevent_config)
        out_dict.update(fixed_timepoints)

        # 3) time-to-event
        time_to_targets = calc_time_to_targets(patient_group_measurements, record_info_with_baseline, timetoevent_config)
        out_dict.update(time_to_targets)

        # 4) genomics timing

        genomics_timing_dict = calc_genomics_timing(record_info_with_baseline, out_dict)
        out_dict.update(genomics_timing_dict)

        # 5) genomics-timed outcomes
        # These are outcome variables linked directly to:
        # - genomics purchase date
        # - genomics results date
        # - fixed post-personalization windows after genomics results
        #
        # This is separate from calc_genomics_timing(), which creates timing flags
        # and interval variables describing where genomics falls inside the record.
        genomics_timed_outcomes = calc_genomics_timed_outcomes(
            patient_group_measurements,
            record_info_with_baseline,
            timetoevent_config
        )
        out_dict.update(genomics_timed_outcomes)

        # 6) post-personalization final outcomes
        post_personalization_final_dict = calc_post_personalization_final_outcomes(out_dict)
        out_dict.update(post_personalization_final_dict)

        output_rows.append(out_dict)

    df_out = pd.DataFrame(output_rows)

    # --- NEW: PROCESS COUNTRY DUMMIES ---
    df_out, output_column_order, dummy_cols = process_country_dummies(df_out, output_column_order)

    # Reorder columns (ensure all columns produced are in output_column_order, or handle missing ones)
    
    # Filter output_column_order to only include columns that actually made it into df_out
    final_cols = [c for c in output_column_order if c in df_out.columns]
    
    df_out = df_out.reindex(columns=final_cols)

    # 4. Save to the same input database defined in the config
    conn_out = sqlite3.connect(paths_config.paper_in_db)
    df_out.to_sql(timetoevent_config.output_table, conn_out, if_exists="replace", index=False)
    conn_out.close()

    print(f"Saved {len(df_out)} rows to {paths_config.paper_in_db}::{timetoevent_config.output_table}")
    if dummy_cols:
        print(f"  - Generated {len(dummy_cols)} country dummy columns: {dummy_cols}")


def subset_timetoevent_table(config: master_config) -> None:
    """
    Creates subset tables from a source table based on a dictionary of definitions.
    """
    if not config.timetoevent_subsetting or not config.timetoevent_subsetting.definitions:
        print("Subsetting configuration not provided or is empty. Skipping.")
        return
    
    db_path = config.paths.paper_in_db
    source_table = config.timetoevent_subsetting.source_table
    subset_definitions = config.timetoevent_subsetting.definitions

    print(f"\n--- Starting Table Subsetting from '{source_table}' ---")
    
    try:
        with sqlite3.connect(db_path) as conn:
            source_df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

            # Iterate directly over the dictionary of definitions
            for output_table, conditions in subset_definitions.items():

                cols_to_check_existence = []
                cols_to_check_absence = []
                value_filters = []
                
                # for cond in conditions:
                #     if isinstance(cond, str):
                #         cols_to_check_existence.append(cond)
                #     elif isinstance(cond, (list, tuple)) and len(cond) == 2:
                #         if cond[1] == "IS_NULL":
                #             cols_to_check_absence.append(cond[0])
                #         else:
                #             value_filters.append(cond)
                #     else:
                #         print(f" - Warning: Invalid condition format '{cond}' for table '{output_table}'. Skipping.")

                                # Add a new list to collect custom operator filters
                operator_filters = []
                
                for cond in conditions:
                    if isinstance(cond, str):
                        cols_to_check_existence.append(cond)
                    elif isinstance(cond, (list, tuple)):
                        if len(cond) == 2:
                            if cond[1] == "IS_NULL":
                                cols_to_check_absence.append(cond[0])
                            else:
                                value_filters.append(cond)
                        elif len(cond) == 3:
                            # New format: ("column", "operator", value)
                            operator_filters.append(cond)
                        else:
                            print(f"      - Warning: Invalid condition format {cond} for table {output_table}. Skipping.")
                    else:
                        print(f"      - Warning: Invalid condition format {cond} for table {output_table}. Skipping.")
                
                # Check if all required columns exist in the DataFrame
                required_cols = cols_to_check_existence + cols_to_check_absence + [c for c, _ in value_filters]
                missing_cols = [col for col in required_cols if col not in source_df.columns]
                
                if missing_cols:
                    print(f" - Warning: Skipping table '{output_table}' because columns {missing_cols} were not found.")
                    continue

                # # Apply Filtering
                # subset_df = source_df.copy()

                # # 1. Existence Check (Not Null)
                # if cols_to_check_existence:
                #     subset_df = subset_df.dropna(subset=cols_to_check_existence)
                    
                # # 2. Absence Check (Is Null)
                # for col in cols_to_check_absence:
                #     subset_df = subset_df[subset_df[col].isna()]

                # # 3. Value Check (Equality)
                # for col, val in value_filters:
                #     subset_df = subset_df[subset_df[col] == val]

                # Apply Filtering
                subset_df = source_df.copy()
                
                # 1. Existence Check (Not Null)
                if cols_to_check_existence:
                    subset_df = subset_df.dropna(subset=cols_to_check_existence)
                    
                # 2. Absence Check (Is Null)
                for col in cols_to_check_absence:
                    subset_df = subset_df[subset_df[col].isna()]
                    
                # 3. Value Filters
                for col, val in value_filters:
                    subset_df = subset_df[subset_df[col] == val]

                # 4. Operator Filters (e.g. >= dates)
                for col, op, val in operator_filters:
                    if col not in subset_df.columns:
                        print(f"      - Warning: Column {col} missing for operator filter. Skipping filter.")
                        continue
                        
                    # If the value looks like a date string and the column isn't datetime, ensure it is converted
                    # For robustness, we let pandas compare strings directly if they are ISO format, 
                    # but converting to datetime guarantees correct comparison.
                    if isinstance(val, str) and "-" in val and op in [">=", "<=", ">", "<"]:
                        col_data = pd.to_datetime(subset_df[col], errors='coerce')
                        val_data = pd.to_datetime(val)
                    else:
                        col_data = subset_df[col]
                        val_data = val

                    if op == ">=":
                        subset_df = subset_df[col_data >= val_data]
                    elif op == "<=":
                        subset_df = subset_df[col_data <= val_data]
                    elif op == ">":
                        subset_df = subset_df[col_data > val_data]
                    elif op == "<":
                        subset_df = subset_df[col_data < val_data]
                    elif op == "!=":
                        subset_df = subset_df[col_data != val_data]
                    else:
                        print(f"      - Warning: Unsupported operator '{op}'. Skipping filter.")

                # Save result
                subset_df.to_sql(output_table, conn, if_exists="replace", index=False)
                
                print(f"  - Saved {len(subset_df)} rows to table '{output_table}'")
                print(f"    Filters applied: Existence={cols_to_check_existence}, Values={value_filters}")

    except Exception as e:
        print(f"An error occurred during subsetting: {e}")
    
    print("--- Table Subsetting Complete ---")

