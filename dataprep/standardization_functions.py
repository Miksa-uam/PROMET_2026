import pandas as pd
import os
import sqlite3
import unicodedata
import numpy as np
import warnings


"""HELPER TO NOT USE TEMP PANDAS DATA FRAMES ALL THE TIME - BUT LOAD DFS FROM SQL DBS"""

def load_table(db_path, table_name, fallback_df=None, parse_dates=None):
    """
    Load a table from SQLite if it exists there.
    Fall back to the in-memory dataframe only if the SQL table is not found.
    """
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            if cursor.fetchone() is not None:
                df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                conn.close()
                if parse_dates:
                    for col in parse_dates:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"  ✓ '{table_name}' loaded from SQL  ({len(df):,} rows)")
                return df
            conn.close()
        except Exception as e:
            print(f"  ⚠ SQL load failed for '{table_name}': {e}")

    if fallback_df is not None:
        print(f"  ↩ '{table_name}' not in SQL — using in-memory df  ({len(fallback_df):,} rows)")
        return fallback_df

    raise ValueError(
        f"'{table_name}' not found in '{db_path}' and no fallback_df was provided. "
        f"Run the upstream pipeline first."
    )


"""THE STANDARDIZATION FUNCTIONS, MODULARIZED"""

# Module-level helper — reusable across ALL standardize_* functions
def _normalize_string(value):
    value = unicodedata.normalize('NFKD', value).encode('ASCII', 'ignore').decode('utf-8')
    return value.lower()

# Standardize the prescriptions table
def standardize_prescriptions(df, column_names, value_dict, column_order):
    """
    Renames columns, translates values, enforces datetimes, reorders and sorts.
    All config (column_names, value_dict, column_order) is passed in from the notebook.
    """
    df = df.rename(columns=column_names)

    # Datetime coercion
    for col in ['prescription_creation_date', 'prescription_registration_date', 'prescription_validity_end_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Value translation
    normalized_dict = {_normalize_string(k): v for k, v in value_dict.items()}
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: normalized_dict.get(_normalize_string(x), x) if isinstance(x, str) else x
        )

    df = df[column_order]
    df = df.sort_values(by=['patient_id', 'prescription_creation_date'])
    return df


# Standardize the patients table
def standardize_patients(df, column_names, value_dict, column_order):
    """
    Renames columns, translates values, enforces datetimes, reorders and sorts.
    All config (column_names, value_dict, column_order) is passed in from the notebook.
    """
    df = df.rename(columns=column_names)

    for col in ['patient_record_creation_date', 'birth_date', 'gdpr4_date', 'gdpr10_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    normalized_dict = {_normalize_string(k): v for k, v in value_dict.items()}
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: normalized_dict.get(_normalize_string(x), x) if isinstance(x, str) else x
        )

    df = df[column_order]
    df = df.sort_values(by='patient_id', ascending=True)
    return df


# Standardize the comorbidities & drug prescriptions table
def standardize_comorbidities(df, column_names, value_dict, column_order):
    """
    Renames columns, translates Spanish values (comorbidities, drugs, doses, binary),
    enforces datetime, reorders and sorts.
    All config (column_names, value_dict, column_order) is passed in from the notebook.
    """
    df = df.rename(columns=column_names)

    if 'creation_date' in df.columns:
        df['creation_date'] = pd.to_datetime(df['creation_date'], errors='coerce')

    normalized_dict = {_normalize_string(k): v for k, v in value_dict.items()}
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: normalized_dict.get(_normalize_string(x), x) if isinstance(x, str) else x
        )

    df = df[column_order]
    df = df.sort_values(by=['patient_id', 'medical_record_id', 'creation_date'])
    return df

def pivot_comorbidities(comorbidities_colclean):
    """
    Identifies unique comorbidities and medications from the first prescription
    of each medical record and pivots them into wide format (one row per medical record).
    Duplicate comorbidities or drugs within the same first prescription are counted only once.
    No config needed — operates entirely on the data it receives.
    """

    comorbidities = comorbidities_colclean.copy()
    comorbidities['creation_date'] = pd.to_datetime(comorbidities['creation_date'])
    comorbidities.sort_values(
        by=['patient_id', 'medical_record_id', 'creation_date', 'comorbidity', 'drug'],
        inplace=True
    )

    # Step 1: Identify first prescription per medical record
    first_prescription_ids = (
        comorbidities
        .groupby(['patient_id', 'medical_record_id'])
        .first()
        .reset_index()
        [['patient_id', 'medical_record_id', 'prescription_id']]
    )

    # Step 2: Filter to first prescription rows only
    comorbidities_baseline_long = pd.merge(
        comorbidities, first_prescription_ids,
        on=['patient_id', 'medical_record_id', 'prescription_id'],
        how='inner'
    )

    if comorbidities_baseline_long.empty:
        return pd.DataFrame(columns=['patient_id', 'medical_record_id'])

    unique_ids = (
        comorbidities_baseline_long[['patient_id', 'medical_record_id']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Step 3: Pivot comorbidities
    max_com_rank = 0
    comorbidities_to_pivot = (
        comorbidities_baseline_long[['patient_id', 'medical_record_id', 'comorbidity']]
        .copy()
        .dropna(subset=['comorbidity'])
    )
    comorbidities_to_pivot = comorbidities_to_pivot[
        comorbidities_to_pivot['comorbidity'].astype(str).str.strip() != ''
    ]
    comorbidities_to_pivot.drop_duplicates(
        subset=['patient_id', 'medical_record_id', 'comorbidity'], keep='first', inplace=True
    )

    if not comorbidities_to_pivot.empty:
        comorbidities_to_pivot['entry_rank'] = (
            comorbidities_to_pivot.groupby(['patient_id', 'medical_record_id']).cumcount() + 1
        )
        comorbidities_wide = comorbidities_to_pivot.pivot_table(
            index=['patient_id', 'medical_record_id'],
            columns='entry_rank', values='comorbidity', aggfunc='first'
        ).reset_index()
        comorbidities_wide.rename(
            columns={col: f'comorbidity{col}' for col in comorbidities_wide.columns
                     if isinstance(col, (int, np.integer))},
            inplace=True
        )
        val = comorbidities_to_pivot['entry_rank'].max()
        max_com_rank = int(val) if pd.notna(val) else 0
    else:
        comorbidities_wide = pd.DataFrame(columns=['patient_id', 'medical_record_id'])

    # Step 4: Pivot drugs
    max_drug_rank = 0
    drugs_to_pivot = (
        comorbidities_baseline_long[['patient_id', 'medical_record_id', 'drug']]
        .copy()
        .dropna(subset=['drug'])
    )
    drugs_to_pivot = drugs_to_pivot[drugs_to_pivot['drug'].astype(str).str.strip() != '']
    drugs_to_pivot.drop_duplicates(
        subset=['patient_id', 'medical_record_id', 'drug'], keep='first', inplace=True
    )

    if not drugs_to_pivot.empty:
        drugs_to_pivot['entry_rank'] = (
            drugs_to_pivot.groupby(['patient_id', 'medical_record_id']).cumcount() + 1
        )
        drugs_wide = drugs_to_pivot.pivot_table(
            index=['patient_id', 'medical_record_id'],
            columns='entry_rank', values='drug', aggfunc='first'
        ).reset_index()
        drugs_wide.rename(
            columns={col: f'drug{col}' for col in drugs_wide.columns
                     if isinstance(col, (int, np.integer))},
            inplace=True
        )
        val = drugs_to_pivot['entry_rank'].max()
        max_drug_rank = int(val) if pd.notna(val) else 0
    else:
        drugs_wide = pd.DataFrame(columns=['patient_id', 'medical_record_id'])

    # Step 5: Merge the wide comorbidities and drugs tables
    # Ensure base IDs are present for merging, especially if one part is empty
    if not comorbidities_wide.empty and not drugs_wide.empty:
        comorbidities_pivoted = pd.merge(
            comorbidities_wide, drugs_wide,
            on=['patient_id', 'medical_record_id'], how='outer'
        )
    elif not comorbidities_wide.empty:
        comorbidities_pivoted = comorbidities_wide
    elif not drugs_wide.empty:
        comorbidities_pivoted = drugs_wide
    else: # Both are empty, start with unique IDs
        comorbidities_pivoted = unique_ids.copy()

    # Ensure all unique_ids are present in the final df, even if they had no comorbidities/drugs
    if not unique_ids.empty:
        comorbidities_pivoted = pd.merge(
            unique_ids, comorbidities_pivoted,
            on=['patient_id', 'medical_record_id'], how='left'
        )
    else: # If unique_ids is also empty (i.e. comorbidities_baseline_long was empty)
        return pd.DataFrame(columns=['patient_id', 'medical_record_id'])

    # Step 6: Order columns — interleave comorbidity/drug pairs
    final_ordered_columns = ['patient_id', 'medical_record_id']
    for i in range(1, max(max_com_rank, max_drug_rank) + 1):
        for col in [f'comorbidity{i}', f'drug{i}']:
            if col in comorbidities_pivoted.columns:
                final_ordered_columns.append(col)
    # Catch any remaining columns not yet included
    for col in comorbidities_pivoted.columns:
        if col not in final_ordered_columns:
            final_ordered_columns.append(col)

    return comorbidities_pivoted[final_ordered_columns]


# Standardize the medical records table
def standardize_medical_records(df, column_names, value_dict, column_order):
    """
    Renames columns, translates values, enforces datetimes, sets given 0 values to NULL,
    reorders and sorts. All config passed in from the notebook.
    """

    df = df.rename(columns=column_names)

    for col in ['medical_record_creation_date', 'medical_record_closing_date', 'birth_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    normalized_dict = {_normalize_string(k): v for k, v in value_dict.items()}
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(
            lambda x: normalized_dict.get(_normalize_string(x), x) if isinstance(x, str) else x
        )

    if 'wc_cm_confirm_time' in df.columns:
        df.loc[df['wc_cm_confirm_time'] == 0, 'wc_cm_confirm_time'] = np.nan

    df = df[column_order]
    df = df.sort_values(by=['medical_record_id', 'medical_record_creation_date'])
    return df

def complete_medical_records(
    medical_records_colclean,
    prescriptions_colclean,
    patients_colclean,
    comorbidities_wide,
    complete_column_order,
    wg_causes_db_path):
    """
    Enriches medical_records_colclean by merging in patient IDs, sex, country,
    genomics data, GDPR status, comorbidities, and weight-gain cause categories.
    Calculates record duration, sequence, and total records per patient.

    Args:
        medical_records_colclean:  cleaned medical records dataframe
        prescriptions_colclean:    cleaned prescriptions dataframe
        patients_colclean:         cleaned patients dataframe
        comorbidities_wide:        pivoted comorbidities dataframe
        complete_column_order:     final column order list (from notebook config)
        wg_causes_db_path:         path to the weight_gain_causes SQLite database (from notebook config)
    """
    import sqlite3

    df = medical_records_colclean.copy()

    # --- merge helpers (purely mechanical, no config needed) ---

    def add_patient_id(df, prescriptions_colclean):
        """Adds patient_id from prescriptions table based on medical_record_id."""
        patient_ids = (prescriptions_colclean[['patient_id', 'medical_record_id']]
                       .drop_duplicates(subset=['medical_record_id']))
        return df.merge(patient_ids, on='medical_record_id', how='left')

    def add_patient_sex(df, patients_colclean):
        """Adds sex data from patients table based on patient_id."""
        sex = patients_colclean[['patient_id', 'sex_f']].drop_duplicates(subset=['patient_id'])
        return df.merge(sex, on='patient_id', how='left')

    def add_patient_country(df, patients_colclean):
        """Adds country data from patients table based on patient_id."""
        country = patients_colclean[['patient_id', 'country']].drop_duplicates(subset=['patient_id'])
        return df.merge(country, on='patient_id', how='left')

    def add_genomics_sample_id(df, patients_colclean):
        """Adds genomics_sample_id from patients table based on patient_id."""
        genomics = (patients_colclean[['patient_id', 'genomics_sample_id']]
                    .drop_duplicates(subset=['patient_id']))
        return df.merge(genomics, on='patient_id', how='left')

    def add_genomics_dates(df, prescriptions_colclean):
        """Adds genomic test processing-related dates from prescriptions table based on medical_record_id."""
        genomics_dates = (prescriptions_colclean[[
            'medical_record_id',
            'genomics_prescription_date', 'genomics_purchase_date',
            'genomics_sampling_date', 'genomics_results_date'
        ]].drop_duplicates(subset=['medical_record_id']))
        return df.merge(genomics_dates, on='medical_record_id', how='left')

    def add_gdpr_data(df, patients_colclean):
        """Adds GDPR4 and GDPR10 true/false values and dates from patients table based on patient_id."""
        gdpr = (patients_colclean[['patient_id', 'gdpr4', 'gdpr4_date', 'gdpr10', 'gdpr10_date']]
                .drop_duplicates(subset=['patient_id']))
        return df.merge(gdpr, on='patient_id', how='left')

    def add_comorbidities(df, comorbidities_wide):
        """Adds baseline comorbidities from comorbidities_wide based on patient and medical record IDs.""" 
        return pd.merge(df, comorbidities_wide.copy(),
                        on=['patient_id', 'medical_record_id'], how='left')

    def add_wg_causes(df, wg_causes_db_path):
        """Adds categorized weight gain causes from the dedicated SQLite database - currently a hardcoded path, this could be improved."""
        with sqlite3.connect(wg_causes_db_path) as conn:
            df_wg = pd.read_sql_query("SELECT * FROM weight_gain_causes", conn)
        return pd.merge(df, df_wg,
                        on=['patient_id', 'medical_record_id', 'weight_gain_cause'],
                        how='left')

    def calculate_record_duration(df):
        """Calculates the duration of the medical record in days."""
        df['medical_record_duration_days'] = (
            df['medical_record_closing_date'] - df['medical_record_creation_date']
        ).dt.days
        return df

    def add_record_counts_and_sequence(df):
        """Adds columns for the total number of records per patient and the sequence of each record."""
        # Sort by patient and date to ensure correct sequencing
        df = df.sort_values(by=['patient_id', 'medical_record_creation_date'])
        # Calculate sequence number within each patient group
        df['medical_record_sequence'] = df.groupby('patient_id').cumcount() + 1
        # Calculate total number of records per patient
        df['nr_medical_records_patient'] = (
            df.groupby('patient_id')['medical_record_id'].transform('nunique')
        )
        return df

    # Execute completion steps
    df = add_patient_id(df, prescriptions_colclean)
    df = add_patient_sex(df, patients_colclean)
    df = add_patient_country(df, patients_colclean)
    df = add_genomics_sample_id(df, patients_colclean)
    df = add_genomics_dates(df, prescriptions_colclean)
    df = add_comorbidities(df, comorbidities_wide)
    df = add_wg_causes(df, wg_causes_db_path)
    df = add_gdpr_data(df, patients_colclean)
    df = calculate_record_duration(df)
    df = add_record_counts_and_sequence(df)

    # Ensure final column order and sort rows
    df = df[complete_column_order]
    df = df.sort_values(by=['patient_id', 'medical_record_creation_date'])
    return df

  
# Standardize and filter the measurements table
def standardize_measurements(df, column_names, column_order):
    """Renames columns to standardized English names, reorders, enforces datetime, sorts."""
    # Rename columns to standardized English names
    df = df.rename(columns=column_names)
    # Reorder columns
    df = df[column_order]
    # Ensure measurement_date is in datetime format
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    # Sort rows by patient_id and measurement_date
    df = df.sort_values(by=['patient_id', 'measurement_date'])
    return df

def rowclean_measurements(
    measurements_colclean,
    first_height_lookup,               # NEW — pd.DataFrame with [patient_id, height_m]
    bc_vars           = None,
    water_col         = "water_%",
    water_floor       = 30.0,
    bmi_disc_thresh   = 5.0,
    weight_floor      = 30.0,
    bmi_floor         = 15.0,):
    """
    Row-level cleaning for measurements. Operates in two stages:

    Stage 1 — BC quality and weight outlier filtering (before deduplication):
      Due to incorrect BMI entries, BIA measurement errors, or other causes, 
      BC values might be unreliable even if weight is correct. 
      These BC values are identified and set to NULL while keeping weight
      and a re-calculated, correct BMI, as the code: 
        - Joins first height per patient, recalculates BMI, calculates the difference between original and recalculated BMI.
        - 2a. water_% < water_floor → all BC → NULL
        - 2b. |bmi_recorded - bmi_calc| ≥ threshold → all BC → NULL
        - 2c. Any BC value still ≤ 0 in a row  → all BC in that row → NULL
      Irrespective of BC correctness, some entries can have implausibly low weight and BMI values. 
      These are filtered as the script: 
        - Drops rows where weight < 30 kg OR bmi_calculated < 15.
      Finally, the code drops transient height_m used in BMI recalculation.

    Stage 2 — Deduplication:
      If a patient has several measurement entries on the same day with the same rounded weight value, 
      the one with the lowest number of NULLified BC values is kept. 
      This is to ensure that if between duplicate entries there is a 'better', 
      more complete one, it is prioritized over the other.  
      If there are no differences in BC NULL count, the first entry is kept. 
    """
    if bc_vars is None:
        bc_vars = ["water_%", "muscle_%", "fat_%", "vat_%"]

    df = measurements_colclean.copy()
    n_start = len(df)

    """
    Row-level cleaning Stage 1: BC filtering - setting unreliable body composition entries to NULL 
    while keeping weight and BMI recalculated based on height (originally recorded BMI not always reliable)
    """

    df = df.merge(first_height_lookup, on="patient_id", how="left")
    df["bmi_calculated"]    = (df["weight_kg"] / df["height_m"] ** 2).round(2)
    df["bmi_rec_calc_diff"] = (df["bmi"] - df["bmi_calculated"]).abs()

    """!!! - originally recorded bmi is REPLACED here with a value recalculated on the spot from height and weight """ 
    # — Replace recorded BMI with recalculated; keep recorded as audit —
    df = df.rename(columns={"bmi": "bmi_recorded"})
    df["bmi"] = df["bmi_calculated"]
    df = df.drop(columns=["bmi_calculated"])  # no longer needed — bmi now holds it

    print(f"\n  [2a] {water_col} < {water_floor} → all BC → NULL")
    mask_low_water = df[water_col] < water_floor
    for col in bc_vars:
        if col in df.columns:
            n_nn = df.loc[mask_low_water, col].notna().sum()
            df.loc[mask_low_water, col] = None
            print(f"    {col:12s}  low-water rows nullified: {n_nn:>6,}")
    print(f"    Total rows affected: {int(mask_low_water.sum()):>6,}  "
          f"({mask_low_water.sum() / n_start * 100:.2f}%)")

    print(f"\n  [2b] bmi_rec_calc_diff ≥ {bmi_disc_thresh} → all BC → NULL")
    mask_disc = df["bmi_rec_calc_diff"] >= bmi_disc_thresh
    for col in bc_vars:
        if col in df.columns:
            n_nn = df.loc[mask_disc, col].notna().sum()
            df.loc[mask_disc, col] = None
            print(f"    {col:12s}  discordant rows nullified: {n_nn:>6,}")
    print(f"    Total rows affected: {int(mask_disc.sum()):>6,}  "
          f"({mask_disc.sum() / n_start * 100:.2f}%)")

    print("\n  [2c] Any zero / negative BC in row → all BC → NULL")
    present_bc   = [c for c in bc_vars if c in df.columns]
    mask_nonpos  = df[present_bc].le(0).any(axis=1)
    n_nonpos     = int(mask_nonpos.sum())
    for col in present_bc:
        n_nn = df.loc[mask_nonpos, col].notna().sum()
        df.loc[mask_nonpos, col] = None
        print(f"    {col:12s}  nonpos-row nullified: {n_nn:>6,}")
    print(f"    Total rows affected: {n_nonpos:>6,}  ({n_nonpos / n_start * 100:.2f}%)")

    print("\n  [row drop] weight < 30 kg OR bmi < 15")
    mask_drop = (df["weight_kg"] < weight_floor) | (df["bmi"] < bmi_floor)
    n_dropped = int(mask_drop.sum())
    df        = df[~mask_drop].copy()
    print(f"    Rows dropped: {n_dropped:>6,}  ({n_dropped / n_start * 100:.2f}%)")

    # height_m is a transient join key — drop it before deduplication.
    # It re-enters via medical_records_complete in complete_measurements.
    df = df.drop(columns=["height_m"], errors="ignore")

    """
    Row-level cleaning Stage 2: Deduplication
    How this works: 
    - identify duplicate measurements as measurements taken on the same day with the same rounded weight
    - for that, focus only on the date part of measurement datetime and the integer part of weight_kg
    - count the number of NULL values in the BC columns and rank the duplicate entries by BC NULL count
    - this is to ensure that if from two duplicates, only one lost BC values in the previous step, 
      the one with no missing data is preferred
    - if there is no difference in BC missingness, the first entry is kept
   """

    present_bc = [c for c in bc_vars if c in df.columns]

    df["_date"]          = df["measurement_date"].dt.date
    df["_weight_rounded"] = df["weight_kg"].round(0)
    df["_bc_null_count"] = df[present_bc].isna().sum(axis=1)

    # Sort best row to the top of each duplicate group (fewest BC nulls first).
    # Ties in null count are broken by original DataFrame order (stable sort).
    df = df.sort_values(
        by=["patient_id", "_date", "_weight_rounded", "_bc_null_count"],
        ascending=[True, True, True, True],
        kind="stable",
    )

    n_pre_dedup = len(df)
    df = df.drop_duplicates(
        subset=["patient_id", "_date", "_weight_rounded"],
        keep="first",
    )

    multi_weight = (
        df.groupby(["patient_id", "_date"])["_weight_rounded"]
        .nunique()
        .gt(1)
        .sum()
    )
    if multi_weight > 0:
        print(f"\n  ⚠  {multi_weight:,} patient-days have >1 distinct rounded weight "
            f"— all retained.")

    df = df.drop(columns=["_date", "_weight_rounded", "_bc_null_count"])
    df = df.sort_values(by=["patient_id", "measurement_date"])

    print(f"\n  Deduplication: {n_pre_dedup:,} → {len(df):,} rows "
        f"({n_pre_dedup - len(df):,} removed)")

    print(f"\n  rowclean_measurements total: {n_start:,} → {len(df):,} rows")
    return df 

def complete_measurements(
    measurements_rowclean,
    medical_records_complete,
    complete_column_order,
    columns_missing_from_out_range,
    permissivity_days=10):
    """
    During data collection, patients take at-home body weight measurements using electronic scales that collect their data using their patient ID. 
    However, medical record IDs are not used in this process, which is a problem, as many patients have multiple treatment cycles with different medical record IDs. 
    Thus, it is important to identify the medical record each measurement belongs to, so that data from different treatment cycles is not mixed during analysis. 
    Measurements and medical records can be linked via patient ID, and the correct measurement-record pairs can be identify by checking if the measurement falls within the medical record's validity period. 
    While unlikely, it is possible though that a patient does not take measurements right after the start or before the beginning of a medical record, for diverse reasons. 
    Thus, it is important to check for any measurements that have been taken right outside of the record's validity period, to make sure no valuable patient weight data is lost. 
    For example, if a patient weighs themselves two days before starting treatment, but fails to do so in the first two weeks of actually doing the diet, 
    this way, we still keep a baseline measurement to compare eventual weight loss outcomes to. 

    Args:
        measurements_rowclean:          deduplicated measurements dataframe
        medical_records_complete:       completed medical records dataframe
        complete_column_order:          final column order for measurements_complete_unfiltered (from notebook config)
        columns_missing_from_out_range: columns to drop from the out-of-range output (from notebook config)
        permissivity_days:              days window around record range to still consider a measurement linkable (default 10)

    Returns:
        tuple: (measurements_complete_unfiltered, out_range_measurements)
    """

    """
    Setup and data preparation: define permissivity days in the function arguments, ensure datetime format for date columns. 
    A permissivity window of 10 days is clinically reasonable;
    enough to catch measurements right before or after a record's validity period, but not too long to introduce bias or noise. 
    Upon discussion with stakeholders, we are settling with 10 days. 
    This causes the loss of about 100k out of 500k measurements, most of them are either between 11-1500 days far from a record's validity period, 
    or are derived from records with no registered end date. 
    """

    measurements_rowclean['measurement_date'] = pd.to_datetime(measurements_rowclean['measurement_date'])
    medical_records_complete['medical_record_creation_date'] = pd.to_datetime(
        medical_records_complete['medical_record_creation_date']
    )
    medical_records_complete['medical_record_closing_date'] = pd.to_datetime(
        medical_records_complete['medical_record_closing_date']
    )

    """
    Step 1: link every measurement to every medical record it can possibly belong to based on Patient ID. 

    We use the rowcleaned measurements that have duplicates removed, and the completed medical records, that have the patient IDs associated. 
    As a measurement can come from several record of a patient, we first need to find out, which are the possible record a measurement can belong to. 
    In later steps, these records are filtered to identify those measurement-record pairs where the measurement is actually within, or very close to, the time range of the record. 
    Mathematically, this dataframe where every possible link is established, is called a cartesian product, so the variable is named 'cartesian' accordingly. 
    """
    
    cartesian = pd.merge(
        measurements_rowclean,
        medical_records_complete,
        on="patient_id",
        how="left"
    )

    """
    Step 2: compute the time distance of a measurement's date from the start and end dates of each record it is linked to. 

    These distances will be used to filter for the measurement-record pairs that are the most probable to be true. 
    First, we calculate how many days before the record's start, and how many days after its end a measurement was taken. 
    In this calculation, negative values are allowed - for example, a record taken before start is taken 'negative days' 'after' its end.
    After calculating the distance of a measurement from both the start and the end dates of a record, a single absolute distance from record range is calculated. 
    This single distance metric is set to zero if the measurement falls between the record's start and end dates; 
    and contains a non-zero value if it was taken some days before or after the record's validity period. 
    This variable is especially interesting when investigating lost data, stored in the out_range_measurements dataframe: 
    it shows how far from any medical record's range that measurement was taken.
    This metric can inform on patient behavior and data collection patterns, and most relevantly, it can guide setting the permissivity days window. 
    After the single absolute 'distance from record range' indicator is calculated, each measurement is flagged with a boolean variable,
    that tells whether the measurement is within or only near the range of the record. 
    If the distance is 0, the measurement is flagged as 'in range'; 
    if the distance is between 1 and the indicated permissivity days (most likely 10), the measurement is flagged as 'near range'.
    Actual links are identified if the measurement is either in or near the range of a record.
    If none, it will be dropped as an 'out-of-range' measurement.
    """

    # First, calculate the days before start and after end dates of a record. 
    cartesian['days_before_record_start'] = (
        cartesian['medical_record_creation_date'] - cartesian['measurement_date']
    ).dt.days
    cartesian['days_after_record_end'] = (
        cartesian['measurement_date'] - cartesian['medical_record_closing_date']
    ).dt.days

    # Then, calculate the absolute distance of a measurement from a record's range. 
    # This value is 0 if the measurement is within the record's range, and the minimum positive gap otherwise.
    # In the case of medical records with no closing dates, this minimum gap can be infinity - as any date is 'infinitely far' from a non-existing end date. 
    # If infinite values are returned, they are set to a large number (99999) to avoid confusion with actual measurements.
    cartesian['measurement_distance_from_record_range'] = cartesian.apply(
        lambda row: 0
            if pd.notna(row['medical_record_creation_date'])
            and pd.notna(row['medical_record_closing_date'])
            and row['measurement_date'] >= row['medical_record_creation_date']
            and row['measurement_date'] <= row['medical_record_closing_date']
            else min(
                row['days_before_record_start']
                if pd.notna(row['days_before_record_start']) and row['days_before_record_start'] > 0
                else float('inf'),
                row['days_after_record_end']
                if pd.notna(row['days_after_record_end']) and row['days_after_record_end'] > 0
                else float('inf')
            ),
        axis=1
    )
    cartesian['measurement_distance_from_record_range'] = cartesian[
        'measurement_distance_from_record_range'
    ].replace(float('inf'), 99999)

    # Next, add within-and near-range flags to each measurement-record pair.
    cartesian['measurement_in_record_range'] = (
        pd.notna(cartesian['medical_record_creation_date']) &
        pd.notna(cartesian['medical_record_closing_date']) &
        (cartesian['measurement_date'] >= cartesian['medical_record_creation_date']) &
        (cartesian['measurement_date'] <= cartesian['medical_record_closing_date'])
    )
    cartesian['measurement_near_record_range'] = (
        ~cartesian['measurement_in_record_range'] &
        (
            (pd.notna(cartesian['days_before_record_start']) &
            (cartesian['days_before_record_start'] <= permissivity_days) &
            (cartesian['days_before_record_start'] > 0)
            ) |
            (pd.notna(cartesian['days_after_record_end']) &
            (cartesian['days_after_record_end'] <= permissivity_days) &
            (cartesian['days_after_record_end'] > 0)
            )
        )
    )

    """
    Step 3: filter the best-matching measurement-record pairs. 

    Within this operation, the first task is to flag the exact measurement-record pairs that are linkable, and also the measurements that have any potential links any record. 
    The 'measurement_record_pair_linkable' bool tells if the measurement in question is in or near the range of the record in question. 
    After, the 'measurement_linkable_to_any_record' bool is calculated by checking if a given measurement, identified as a patiend ID-measurement date group, 
    has any True values in the 'measurement_record_pair_linkable' column. If any True values are found, this is projected onto all the occurrences of that given patient ID-measurement date group. 
    After, we will actually filter for the best links based on the 'measurement_record_pair_linkable' variable, 
    sorting all the patient ID-measurement date groups by the 'measurement_distance_from_record_range' variable, keeping only the first occurrence of each group. 
    The creation and sorting of the 'measurement_linkable_to_any_record' variable might seem a bit redundant after creating the 'measurement_record_pair_linkable' variable, 
    but it is a good way of grouping together all the potential links of a measurement, and sorting through them in one place. 
    The 'measurement_linkable_to_any_record' variable is also used to determine out-of-range measurements: 
    any measurement where measurement_linkable_to_any_record is False, is considered out-of-range.
    """

    # Identify exact measurement-record pairs that are considered linkable
    cartesian['measurement_record_pair_linkable'] = (
        cartesian['measurement_in_record_range'] |
        cartesian['measurement_near_record_range']
    )
    # Based on that, identify those measurements that have any links
    measurement_group_keys = ['patient_id', 'measurement_date']
    cartesian['measurement_linkable_to_any_record'] = cartesian.groupby(
        measurement_group_keys
    )['measurement_record_pair_linkable'].transform('any')

    # Identify the best measurement-medical record links, ie. those where the measurement is within or closest to the record's range. 
    # First, subset the cartesian dataframe to only include measurements that are linkable to any record. 
    # Sort these measurements by their distance from the record's range. 
    # Group the sorted measurements by patient ID and measurement date,
    # and save the first occurrence of each group, ie. the measurement that is within or closest to a record, 
    # into the final output dataframe named measurements_complete_unfiltered (unfiltered, because measurements will later be filtered by baseline BMI and GDPR consent)
    potential_links = cartesian[cartesian['measurement_linkable_to_any_record']].copy()
    potential_links = potential_links.sort_values(
        measurement_group_keys + ['measurement_distance_from_record_range']
    )
    measurements_complete_unfiltered = potential_links.groupby(
        measurement_group_keys, observed=True
    ).head(1).copy()

    """
    Step 4: calculate the sequence of measurements within each record, the total number of measurements per record, 
    and the first and last measurement of each record. 

    These tags are important for various analytical and data processing purposes: 
    to be able to calculate the average number of measurements per patient as a potential adherence/engagement marker, 
    and to confidently identify the first measurement during baseline BMI filtering; as later, any record started with a low baseline BMI needs to be excluded. 
    """

    # First, sort the record ID-tagged measurements data frame by IDs and measurement date. 
    # Group by the IDs, to identify measurements from a specific record of a specific patient. 
    # Calculate the sequence of each measurement in the corresponding record, and based on that, calculate the total number of measurements per record.
    measurements_complete_unfiltered = measurements_complete_unfiltered.sort_values(
        ['patient_id', 'medical_record_id', 'measurement_date']
    )
    group_keys = ['patient_id', 'medical_record_id']
    measurements_complete_unfiltered['measurement_sequence'] = \
        measurements_complete_unfiltered.groupby(group_keys).cumcount() + 1
    measurements_complete_unfiltered['nr_total_measurements_record'] = \
        measurements_complete_unfiltered.groupby(group_keys)['measurement_date'].transform('size')

    # Then, add the first and last measurement tags to the measurements data frame.
    # Group by patient and record ID, and identify the first and last measurement of each record.
    # The first and last measurement tags are set to 0 by default, and then set to 1 for the first and last measurements of each record.
    measurements_complete_unfiltered['first_in_record'] = 0
    measurements_complete_unfiltered['last_in_record'] = 0
    first_idx = measurements_complete_unfiltered.groupby(
        group_keys, observed=True
    ).head(1).index
    last_idx = measurements_complete_unfiltered.groupby(
        group_keys, observed=True
    ).tail(1).index
    measurements_complete_unfiltered.loc[first_idx, 'first_in_record'] = 1
    measurements_complete_unfiltered.loc[last_idx, 'last_in_record'] = 1

    """
    Step 5: handle out‐of‐range measurements by linking them to the medical record they would be closest to. 

    This is good for analytical purposes, namely, to identify any patterns in measurements taken outside of record ranges. 
    The out of range candidate measurements are those rows of the cartesian dataframe where measurement_linkable_to_any_record is False. 
    These candidate measurements are sorted by patient ID, measurement date and distance from record range, grouped by patient ID, 
    and in each group, the first occurrence is kept - the measurement that is closest to any record, even though it might be way out of range. 
    """

    out_range_candidates = cartesian[~cartesian['measurement_linkable_to_any_record']].copy()
    out_range_candidates = out_range_candidates.sort_values(
        ['patient_id', 'measurement_date', 'measurement_distance_from_record_range']
    )
    out_range_measurements = out_range_candidates.groupby(
        ['patient_id', 'measurement_date'], observed=True
    ).head(1).copy()

    """
    Step 7: finalize column order in the completed measurements data frame. 
    Column order is defined in config. 
    In case of out_range_measurements, use a slightly modified column order (defined in config) 
    to account for the lack of the columns only created in measurements_complete_unfiltered. 
    """

    measurements_complete_unfiltered = measurements_complete_unfiltered[complete_column_order]
    out_range_column_order = [
        col for col in complete_column_order
        if col not in columns_missing_from_out_range and col in out_range_measurements.columns
    ]
    out_range_measurements = out_range_measurements[out_range_column_order]

    return measurements_complete_unfiltered, out_range_measurements

# # Standardize the alleles table
# def standardize_alleles(df, column_names, column_order):
#     """
#     Renames columns, enforces datetime, standardises lab names,
#     reorders and sorts. No value translation needed for this table.
#     column_order is passed in from the notebook.
#     Note: lab name standardisation (CG3 → CESGEN3) is a hardcoded
#     data-quality rule, not user config — it lives in the function intentionally.
#     """
#     df = df.copy()
#     df = df.rename(columns=column_names)

#     if 'genomics_lab_date' in df.columns:
#         df['genomics_lab_date'] = pd.to_datetime(df['genomics_lab_date'], errors='coerce')

#     if 'lab_name' in df.columns:
#         df['lab_name'] = df['lab_name'].str.upper().replace('CG3', 'CESGEN3')

#     cols_present = [col for col in column_order if col in df.columns]
#     df = df[cols_present]

#     if 'genomics_sample_id' in df.columns:
#         df = df.sort_values(by='genomics_sample_id')
#     return df

# def complete_alleles(
#     alleles_colclean,
#     patients_colclean,
#     prescriptions_colclean,
#     complete_column_order):
#     """
#     Enriches alleles_colclean with patient_id, GDPR10 data (from patients_colclean)
#     and genomics-related dates (from prescriptions_colclean).
#     Includes duplicate-safety checks and warnings before each merge.
#     complete_column_order is passed in from the notebook.
#     """

#     df = alleles_colclean.copy()
#     patients_df = patients_colclean.copy()
#     prescriptions_df = prescriptions_colclean.copy()

#     # --- Merge 1: patient_id + GDPR10 from patients ---
#     patient_cols = ['patient_id', 'genomics_sample_id', 'gdpr10', 'gdpr10_date']
#     patients_merge = patients_df[patient_cols].copy()

#     initial_rows = len(patients_merge)
#     patients_merge = patients_merge.drop_duplicates(subset=patient_cols, keep='first')
#     dropped = initial_rows - len(patients_merge)
#     if dropped > 0:
#         print(f"  Dropped {dropped} fully duplicated rows from patients merge data.")

#     key_dups = patients_merge[patients_merge.duplicated(subset=['genomics_sample_id'], keep=False)]
#     if not key_dups.empty:
#         print(f"  Warning: {len(key_dups)} rows with duplicated 'genomics_sample_id' after dedup — keeping first.")
#         patients_merge = patients_merge.drop_duplicates(subset=['genomics_sample_id'], keep='first')

#     patient_id_dups = patients_merge[patients_merge.duplicated(subset=['patient_id'], keep=False)]
#     if not patient_id_dups.empty:
#         differing = patient_id_dups.groupby('patient_id').filter(
#             lambda x: x.nunique().drop('patient_id').max() > 1
#         )
#         if not differing.empty:
#             print(f"  Found {differing['patient_id'].nunique()} patients with duplicated patient_id "
#                   f"but differing values in patients merge data.")

#     df = df.merge(patients_merge, on='genomics_sample_id', how='left')

#     # --- Merge 2: genomics dates from prescriptions ---
#     presc_cols = ['patient_id', 'genomics', 'genomics_purchase_date',
#                   'genomics_sampling_date', 'genomics_results_date']
#     prescriptions_merge = prescriptions_df[presc_cols].copy()

#     date_cols = ['genomics_purchase_date', 'genomics_sampling_date', 'genomics_results_date']
#     for col in date_cols:
#         prescriptions_merge[col] = pd.to_datetime(prescriptions_merge[col], errors='coerce')

#     agg_cols = ['genomics'] + date_cols
#     multiple_check = prescriptions_merge.groupby('patient_id')[agg_cols].agg(
#         lambda x: x.dropna().nunique()
#     )
#     multi_patients = multiple_check[(multiple_check > 1).any(axis=1)]
#     if not multi_patients.empty:
#         warnings.warn(
#             f"Multiple different non-NA genomics values in prescriptions for "
#             f"{len(multi_patients)} patients. Using first non-NA value. "
#             f"Patient IDs: {multi_patients.index.tolist()}"
#         )

#     prescriptions_agg = (
#         prescriptions_merge
#         .groupby('patient_id')[agg_cols]
#         .agg('first')
#         .reset_index()
#     )

#     if 'patient_id' in df.columns:
#         df = df.merge(prescriptions_agg, on='patient_id', how='left')
#     else:
#         warnings.warn("'patient_id' column not found after merging with patients data. Skipping merge with prescriptions data.")

#     # --- Final column order and sort ---
#     cols_present = [col for col in complete_column_order if col in df.columns]
#     for col in df.columns:
#         if col not in cols_present:
#             cols_present.append(col)
#     df = df[cols_present]

#     if 'patient_id' in df.columns:
#         df = df.sort_values(by='patient_id')
#     elif 'genomics_sample_id' in df.columns:
#         df = df.sort_values(by='genomics_sample_id')

#     return df

# Filter records by BMI threshold and GDPR consent

def filter_database_by_baseline_bmi(source_db_path, dest_db_path, bmi_threshold, tables_to_filter):
    """
    Filters tables in a source SQLite database based on a baseline BMI threshold
    and writes the filtered tables to a destination SQLite database.

    Args:
        source_db_path (str): Path to the source SQLite database.
        dest_db_path (str): Path to the destination SQLite database.
        bmi_threshold (float or int): The minimum baseline BMI required for inclusion.
        tables_to_filter (dict): Configuration dictionary defining tables to filter,
                               their ID columns, and destination table names.
    """

    # 2. IDENTIFY PATIENTS WITH BASELINE BMI OVER THRESHOLD
    # Connect to the source database
    conn_source = sqlite3.connect(source_db_path)

    # Read the measurements table
    unfiltered_measurements = pd.read_sql_query("SELECT * FROM measurements_complete_unfiltered", conn_source)

    # Create dataframe with only the first measurement of each record
    unfiltered_first_measurements = unfiltered_measurements.loc[
        unfiltered_measurements['first_in_record'] == 1,
        ['patient_id', 'medical_record_id', 'measurement_date', 'bmi', 'first_in_record']
    ].copy() # Use .copy() to avoid SettingWithCopyWarning later

    # Print how many first measurements were identified
    print(f"First measurements identified from {len(unfiltered_first_measurements)} medical records")

    # Filter for records meeting the BMI threshold
    over_treshold_starters = unfiltered_first_measurements[
        unfiltered_first_measurements['bmi'] >= bmi_threshold
    ].copy() # Use .copy()

    # For debugging/analytics: save the under-treshold starters too
    under_treshold_starters = unfiltered_first_measurements[
        unfiltered_first_measurements['bmi'] < bmi_threshold
    ].copy() # Use .copy()

    len(unfiltered_first_measurements) - len(over_treshold_starters)
    print(f"{len(over_treshold_starters)} medical records were started with a baseline BMI of at least {bmi_threshold}. "
          f"\n{len(under_treshold_starters)} medical records were started with a baseline BMI of under {bmi_threshold}.")

    # Close the source database connection (opened for initial ID finding)
    conn_source.close()


    # 3. CREATE THE FILTERED DB WITH ONLY OVER-TRESHOLD STARTERS

    # Extract unique IDs to keep
    over_treshold_record_ids = set(over_treshold_starters['medical_record_id'].unique())
    over_treshold_patient_ids = set(over_treshold_starters['patient_id'].unique())

    # Print number of unique IDs (optional check)
    print(f"\nFound {len(over_treshold_record_ids)} unique medical record IDs to keep.")
    print(f"Found {len(over_treshold_patient_ids)} unique patient IDs to keep.")

    # Connect to source and destination databases for filtering/writing
    conn_source = sqlite3.connect(source_db_path)
    conn_dest = sqlite3.connect(dest_db_path)

    # Iterate through the tables defined in the dictionary
    for source_table, config in tables_to_filter.items():
        id_column = config['id_column']
        dest_table = config['dest_table']

        # Determine which set of IDs to use for filtering
        if id_column == 'medical_record_id':
            ids_to_keep = over_treshold_record_ids
        elif id_column == 'patient_id':
            ids_to_keep = over_treshold_patient_ids

        # Convert set to list or tuple for SQL query parameter binding
        ids_list = list(ids_to_keep)

        # If there are no IDs to keep for this category, write an empty table
        if not ids_list:
             # Read schema from source to create an empty DataFrame with correct columns
            query_schema = f"SELECT * FROM {source_table} LIMIT 0"
            empty_df = pd.read_sql_query(query_schema, conn_source)
            empty_df.to_sql(dest_table, conn_dest, if_exists='replace', index=False)
            # print(f"No IDs found for '{id_column}'. Created empty table '{dest_table}'.") # Optional print
            continue # Skip to the next table

        # Construct the SQL query with parameter placeholders
        # The number of placeholders must match the number of IDs
        placeholders = ', '.join('?' * len(ids_list))
        query = f"SELECT * FROM {source_table} WHERE {id_column} IN ({placeholders})"

        # Read the filtered data from the source database
        # Using params argument for safe and efficient querying
        filtered_df = pd.read_sql_query(query, conn_source, params=ids_list)

        # Write the filtered data to the destination database
        # Replace the table if it already exists, do not write the DataFrame index
        filtered_df.to_sql(dest_table, conn_dest, if_exists='replace', index=False)        
        print(f"Filtered '{source_table}' based on '{id_column}' and saved to '{dest_table}' in the filtered database.")
        print(f"Filtered {len(filtered_df)} rows into {dest_table}.")

    # Close database connections
    conn_source.close()
    conn_dest.close()


def filter_database_by_gdpr(filtered_db_path, tables_to_filter):
    """
    Filters tables in the specified SQLite database based on GDPR consent rules.

    Keeps records with GDPR4=1.
    For records with GDPR10=1 but not GDPR4=1, keeps only those created
    on or after the GDPR10 consent date, or where the consent date falls
    within the record's duration, considering subsequent records valid once consent is established.
    Filters specified related tables based on the final list of valid
    patient_id and medical_record_id combinations.
    """

    print("\nFiltering database by GDPR consent...")
    conn = sqlite3.connect(filtered_db_path)
    # Removed unused cursor initialization

    # Step 1: identify records with GDPR4
    # Initialize a data frame named gdpr_filtered_records,
    # that will store the patient and medical record IDs of those records that have valid GDPR consent.
    # First, the records with GDPR4 consent are identified. Valid GDPR10-only records are identified in the following steps.
    # NOTE: Assumes 'medical_records_filtered' exists as a source for consent info.
    # If it's one of the tables *to be created*, this logic needs adjustment (e.g., read from 'medical_records_complete').
    # Assuming 'medical_records_filtered' is the correct source table name for consent lookup as per the original code.
    gdpr4_query = f"SELECT patient_id, medical_record_id FROM medical_records_filtered WHERE gdpr4 = 1"
    gdpr_filtered_records = pd.read_sql_query(gdpr4_query, conn)
    print(f"Found {len(gdpr_filtered_records)} records with GDPR4 consent.")

    # Step 2: identify records with GDPR10 but not GDPR4
    cols_to_fetch = [
        'patient_id', 'medical_record_id', 'gdpr4', 'gdpr4_date', 'gdpr10',
        'gdpr10_date', 'genomics_sample_id', 'genomics_prescription_date',
        'genomics_purchase_date', 'genomics_sampling_date', 'genomics_results_date',
        'medical_record_creation_date', 'medical_record_closing_date',
        'intervention_duration_days', 'nr_medical_records_patient',
        'medical_record_sequence'
    ]
    cols_str = ", ".join(f'"{col}"' for col in cols_to_fetch) # Quote column names
    no_gdpr4_query = f"""
        SELECT {cols_str}
        FROM medical_records_filtered
        WHERE (gdpr4 IS NULL OR gdpr4 != 1) AND gdpr10 = 1
    """
    no_gdpr4_records = pd.read_sql_query(no_gdpr4_query, conn)
    print(f"Found {len(no_gdpr4_records)} records with no GDPR4 consent.")

    # Step 3: filter no GDPR4 records to identify those with valid GDPR10 consent
    date_cols = ['gdpr10_date', 'medical_record_creation_date', 'medical_record_closing_date']
    for col in date_cols:
        no_gdpr4_records[col] = pd.to_datetime(no_gdpr4_records[col], errors='coerce')

    # Sort records by IDs and record creation date, group by patient_id
    no_gdpr4_records = no_gdpr4_records.sort_values(
        by=['patient_id', 'medical_record_creation_date', 'medical_record_id']
    ).reset_index(drop=True)  # Reset index after sort

    # Define conditions by which a medical record can be defined as having valid GDPR10 consent
    # Condition a: consent registered during the validity period of the medical record
    # Condition b: medical record created after the consent date
    condition_a = (
        no_gdpr4_records['gdpr10_date'].notna() &
        no_gdpr4_records['medical_record_creation_date'].notna() &
        no_gdpr4_records['medical_record_closing_date'].notna() &
        (no_gdpr4_records['gdpr10_date'] >= no_gdpr4_records['medical_record_creation_date']) &
        (no_gdpr4_records['gdpr10_date'] <= no_gdpr4_records['medical_record_closing_date'])
    )
    condition_b = (
        no_gdpr4_records['medical_record_creation_date'].notna() &
        no_gdpr4_records['gdpr10_date'].notna() &
        (no_gdpr4_records['medical_record_creation_date'] > no_gdpr4_records['gdpr10_date'])
    )

    # Filter records to only keep those with valid consent covering the record's period
    no_gdpr4_records['is_valid_consent_period'] = condition_a | condition_b

    # Within each patient, once a valid consent-covered record appears,
    # keep that record and all subsequent records (after sorting)
    no_gdpr4_records['keep_from_here_on'] = (
        no_gdpr4_records
        .groupby('patient_id')['is_valid_consent_period']
        .cummax()
    )

    no_gdpr4_yes_gdpr10_records = no_gdpr4_records[
        no_gdpr4_records['keep_from_here_on']
    ].copy()

    print(f"Kept {len(no_gdpr4_yes_gdpr10_records)} records after GDPR10 date filtering.")

    # Step 4: Combine IDs and create final sets of valid IDs
    gdpr10_ids = no_gdpr4_yes_gdpr10_records[['patient_id', 'medical_record_id']].drop_duplicates()
    # Combine GDPR4 and valid GDPR10 records
    gdpr_filtered_records = pd.concat([gdpr_filtered_records, gdpr10_ids]).drop_duplicates().reset_index(drop=True)

    # Extract unique IDs into sets for efficient filtering
    # Convert IDs to string to handle potential mixed types and simplify SQL query construction
    gdpr_filtered_record_ids = set(gdpr_filtered_records['medical_record_id'].astype(str).unique())
    gdpr_filtered_patient_ids = set(gdpr_filtered_records['patient_id'].astype(str).unique())
    print(f"Found {len(gdpr_filtered_record_ids)} unique medical record IDs with valid consent.")
    print(f"Found {len(gdpr_filtered_patient_ids)} unique patient IDs with valid consent.")

    # Step 5: Iterate through the tables defined in the dictionary and filter them
    for source_table, config in tables_to_filter.items():
        id_column = config['id_column']
        dest_table = config['dest_table']
        print(f"\nFiltering {source_table} into {dest_table} based on {id_column}...")

        # Determine which set of IDs to use for filtering
        if id_column == 'patient_id':
            valid_ids = gdpr_filtered_patient_ids
        elif id_column == 'medical_record_id':
            valid_ids = gdpr_filtered_record_ids

        # Check if there are any valid IDs to filter by
        if not valid_ids:
            print(f"No valid IDs ({id_column}) found. Creating empty table {dest_table}.")
            # Create an empty DataFrame with columns from the source table
            # Read schema using LIMIT 0, then write empty DataFrame
            empty_df_query = f'SELECT * FROM "{source_table}" LIMIT 0'
            filtered_df = pd.read_sql_query(empty_df_query, conn)
        else:
            # Format IDs for the SQL IN clause. Quote string IDs.
            # Assumes IDs are either numeric or strings that don't contain single quotes.
            ids_list_str = [f"'{str(id_val)}'" if isinstance(id_val, str) else str(id_val) for id_val in valid_ids]
            ids_string = ", ".join(ids_list_str)

            # Construct the SQL query to select rows with matching IDs
            # Quote the id_column name to handle potential special characters or keywords
            filter_query = f'SELECT * FROM "{source_table}" WHERE "{id_column}" IN ({ids_string})'

            # Execute the query to get the filtered data
            filtered_df = pd.read_sql_query(filter_query, conn)

        # Write the filtered DataFrame to the destination table, replacing it if it exists
        filtered_df.to_sql(dest_table, conn, if_exists='replace', index=False)
        print(f"Filtered {len(filtered_df)} rows into {dest_table}.")

    # Close the database connection
    conn.close()
    print("\nDatabase filtering complete and connection closed.")
