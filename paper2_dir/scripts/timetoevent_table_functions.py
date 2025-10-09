


import sqlite3
import pandas as pd
import numpy as np
from datetime import timedelta

"""
1. CONFIGURATION
"""

def make_column_order():
    cols = followup_columns.copy()
    for w in time_windows:
        prefix = f"{w}d"
        cols += [
            f"{prefix}_weight_kg", f"{prefix}_wl_kg", f"{prefix}_wl_%",
            f"{prefix}_bmi", f"{prefix}_bmi_reduction",
            f"{prefix}_fat_%", f"{prefix}_fat_loss_%",
            f"{prefix}_muscle_%", f"{prefix}_muscle_change_%",
            f"{prefix}_date", f"days_to_{prefix}_measurement", f"{prefix}_dropout"
        ]
    for t in weight_loss_targets:
        prefix = f"{t}%_wl"
        cols += [
            f"{prefix}_achieved", f"{prefix}_%", f"{prefix}_date", f"days_to_{prefix}"
        ]
    cols += predictor_columns
    return cols

"""
2. DATA LOADING
"""
def load_measurements(conn) -> pd.DataFrame:
    # Consider selecting only necessary columns if measurements_p1 has many unused ones
    cols_to_select = ["patient_id", "medical_record_id", "measurement_date", 
                      "first_in_record", "weight_kg", "bmi", "fat_%", "muscle_%"]
    cols_to_select = [f'"{col}"' for col in cols_to_select]  # Ensure proper quoting for SQL, as some columns have special characters
    # Check if any other columns from measurements_p1 are implicitly used. If not, this is safer.
    # If you are sure all columns are needed, use "SELECT *"
    sql_query = f"SELECT {','.join(cols_to_select)} FROM {input_measurements}"
    # sql_query = f"SELECT * FROM {input_measurements}" # Original version
    
    df = pd.read_sql(sql_query, conn, parse_dates=["measurement_date"])
    df = df.sort_values(["patient_id", "medical_record_id", "measurement_date"])
    return df

def load_med_records(conn) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT {','.join(fetch_from_records)} FROM {input_medical_records}", conn)
    return df

"""
3. BASELINE & MERGE (No major changes here, but ensure it's efficient)
"""
def extract_baseline(meas: pd.DataFrame) -> pd.DataFrame:
    base = meas[meas["first_in_record"] == 1].copy() # .copy() is good here
    base = base.rename(columns={
        "measurement_date": "baseline_date",
        "weight_kg": "baseline_weight_kg",
        "bmi": "baseline_bmi",
        "fat_%": "baseline_fat_%",
        "muscle_%": "baseline_muscle_%"
    })
    return base[[
        "patient_id", "medical_record_id",
        "baseline_date", "baseline_weight_kg", "baseline_bmi",
        "baseline_fat_%", "baseline_muscle_%"
    ]]

def merge_baseline_and_records(baseline, recs):
    df = baseline.merge(recs, on=["patient_id", "medical_record_id"], how="left")
    return df

"""
4. CALCULATIONS (Functions refactored for clarity and to work with grouped data)
"""
# calc_overall_followup: Largely similar, but operates on pre-grouped, pre-sorted data.
# Ensure baseline_row_info is a pd.Series (which it will be from indexed lookup)
def calc_overall_followup(patient_record_measurements: pd.DataFrame, baseline_row_info: pd.Series) -> pd.Series:
    baseline_date = baseline_row_info["baseline_date"]
    
    followup_measurements = patient_record_measurements[
        patient_record_measurements["measurement_date"] > baseline_date
    ] # patient_record_measurements is already sorted by date

    last = baseline_row_info.copy() # Start with baseline info (which is a Series)
    last["instant_dropout"] = 0 # Initialize instant_dropout

    if followup_measurements.empty:
        last["last_aval_date"] = baseline_row_info["baseline_date"]
        last["total_followup_days"] = 1
        last["instant_dropout"] = 1 # Set to 1 if only baseline measurement exists
        last["last_aval_weight_kg"] = baseline_row_info["baseline_weight_kg"]
        last["total_wl_kg"], last["total_wl_%"] = 0.0, 0.0
        last["final_bmi"] = baseline_row_info["baseline_bmi"]
        last["bmi_reduction"] = 0.0
        last["last_aval_fat_%"] = baseline_row_info["baseline_fat_%"]
        last["total_fat_loss_%"] = 0.0
        last["last_aval_muscle_%"] = baseline_row_info["baseline_muscle_%"]
        last["total_muscle_change_%"] = 0.0
    else:
        last_meas = followup_measurements.iloc[-1]
        dt = (last_meas.measurement_date - baseline_date).days + 1
        last["last_aval_date"] = last_meas.measurement_date
        last["total_followup_days"] = dt
        last["last_aval_weight_kg"] = last_meas.weight_kg
        last["total_wl_kg"] = last_meas.weight_kg - baseline_row_info["baseline_weight_kg"]
        last["total_wl_%"] = 100 * last["total_wl_kg"] / baseline_row_info["baseline_weight_kg"] if baseline_row_info["baseline_weight_kg"] else 0
        last["final_bmi"] = last_meas.bmi
        last["bmi_reduction"] = last_meas.bmi - baseline_row_info["baseline_bmi"]
        last["last_aval_fat_%"] = last_meas["fat_%"]
        last["total_fat_loss_%"] = last_meas["fat_%"] - baseline_row_info["baseline_fat_%"]
        last["last_aval_muscle_%"] = last_meas["muscle_%"]
        last["total_muscle_change_%"] = last_meas["muscle_%"] - baseline_row_info["baseline_muscle_%"]
    
    last["nr_total_measurements"] = len(patient_record_measurements)
    if last["nr_total_measurements"] > 1:
        # Denominator (last["nr_total_measurements"] - 1) cannot be zero here
        last["avg_days_between_measurements"] = \
            (last["total_followup_days"] - 1) / (last["nr_total_measurements"] - 1)
    else:
        last["avg_days_between_measurements"] = np.nan
    return last

# calc_fixed_timepoints: Operates on pre-grouped data.
def calc_fixed_timepoints(patient_record_measurements: pd.DataFrame, baseline_row_info: pd.Series) -> dict:
    out = {}
    baseline_date = baseline_row_info["baseline_date"]
    baseline_weight_kg = baseline_row_info["baseline_weight_kg"]
    baseline_bmi = baseline_row_info["baseline_bmi"]
    baseline_fat_pct = baseline_row_info["baseline_fat_%"]
    baseline_muscle_pct = baseline_row_info["baseline_muscle_%"]

    for w in time_windows:
        target_date = baseline_date + timedelta(days=w)
        lo = baseline_date + timedelta(days=w - window_span)
        hi = baseline_date + timedelta(days=w + window_span)
        
        window_meas = patient_record_measurements[
            (patient_record_measurements["measurement_date"] >= lo) &
            (patient_record_measurements["measurement_date"] <= hi)
        ]
        
        prefix = f"{w}d"
        if window_meas.empty:
            out[f"{prefix}_dropout"] = 1
            out[f"{prefix}_weight_kg"], out[f"{prefix}_wl_kg"], out[f"{prefix}_wl_%"] = np.nan, np.nan, np.nan
            out[f"{prefix}_bmi"], out[f"{prefix}_bmi_reduction"] = np.nan, np.nan
            out[f"{prefix}_fat_%"], out[f"{prefix}_fat_loss_%"] = np.nan, np.nan
            out[f"{prefix}_muscle_%"], out[f"{prefix}_muscle_change_%"] = np.nan, np.nan
            out[f"{prefix}_date"], out[f"days_to_{prefix}_measurement"] = pd.NaT, np.nan
        else:
            # Make a copy to safely add a column
            window_meas_copy = window_meas.copy()
            window_meas_copy["dist_to_center"] = (window_meas_copy["measurement_date"] - target_date).abs().dt.days
            take = window_meas_copy.loc[window_meas_copy["dist_to_center"].idxmin()]
            out[f"{prefix}_dropout"] = 0
            out[f"{prefix}_weight_kg"] = take.weight_kg
            wl_kg = take.weight_kg - baseline_weight_kg
            out[f"{prefix}_wl_kg"] = wl_kg
            out[f"{prefix}_wl_%"]  = 100 * wl_kg / baseline_weight_kg if baseline_weight_kg else 0
            out[f"{prefix}_bmi"] = take.bmi
            out[f"{prefix}_bmi_reduction"] = take.bmi - baseline_row_info["baseline_bmi"]
            out[f"{prefix}_fat_%"] = take["fat_%"]
            out[f"{prefix}_fat_loss_%"] = take["fat_%"] - baseline_fat_pct
            out[f"{prefix}_muscle_%"] = take["muscle_%"]
            out[f"{prefix}_muscle_change_%"] = take["muscle_%"] - baseline_muscle_pct
            out[f"{prefix}_date"] = take.measurement_date
            out[f"days_to_{prefix}_measurement"] = (take.measurement_date - baseline_date).days + 1
    return out

# calc_time_to_targets: CRITICAL REFACTOR - removed inner iterrows()
def calc_time_to_targets(patient_record_measurements: pd.DataFrame, baseline_row_info: pd.Series) -> dict:
    out = {}
    baseline_date = baseline_row_info["baseline_date"]
    baseline_weight = baseline_row_info["baseline_weight_kg"]

    # Filter measurements strictly after baseline date; patient_record_measurements is already sorted
    group = patient_record_measurements[patient_record_measurements["measurement_date"] > baseline_date]

    if group.empty or baseline_weight == 0: # Added check for baseline_weight to prevent division by zero
        for t in weight_loss_targets:
            out[f"{t}%_wl_achieved"], out[f"{t}%_wl_%"] = 0, np.nan
            out[f"{t}%_wl_date"], out[f"days_to_{t}%_wl"] = pd.NaT, np.nan
        return out

    # Calculate weight loss percentage for all measurements in the group vectorially
    # Use .copy() to avoid SettingWithCopyWarning if 'group' is a slice
    group_copy = group.copy()
    group_copy["wl_pct_calculated"] = 100 * (baseline_weight - group_copy["weight_kg"]) / baseline_weight
    
    for t in weight_loss_targets:
        # Find the first measurement (due to sort order) where target is achieved
        achieved_measurements = group_copy[group_copy["wl_pct_calculated"] >= t]
        
        if not achieved_measurements.empty:
            first_achieved_event = achieved_measurements.iloc[0]
            out[f"{t}%_wl_achieved"] = 1
            out[f"{t}%_wl_%"] = first_achieved_event["wl_pct_calculated"]
            out[f"{t}%_wl_date"] = first_achieved_event["measurement_date"]
            out[f"days_to_{t}%_wl"] = (first_achieved_event["measurement_date"] - baseline_date).days + 1
        else:
            out[f"{t}%_wl_achieved"] = 0
            out[f"{t}%_wl_%"] = np.nan 
            out[f"{t}%_wl_date"] = pd.NaT
            out[f"days_to_{t}%_wl"] = np.nan
    return out

"""
5. ORCHESTRATION (Refactored main loop)
"""
def build_timetoevent_table():
    conn_in  = sqlite3.connect(input_db)
    all_measurements = load_measurements(conn_in)
    medical_records_data = load_med_records(conn_in)
    conn_in.close()

    baseline_data = extract_baseline(all_measurements)
    merged_patient_records = merge_baseline_and_records(baseline_data, medical_records_data)

    # Set index for quick lookup of baseline/record info. This is crucial.
    # merged_patient_records has one row per (patient_id, medical_record_id)
    merged_patient_records_indexed = merged_patient_records.set_index(["patient_id", "medical_record_id"])

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
            # print(f"Warning: No baseline/record info for patient {patient_id}, record {medical_record_id}. Skipping.")
            continue # Skip this group if no baseline/record info

        # 1) overall follow‐up
        overall_followup_series = calc_overall_followup(patient_group_measurements, record_info_with_baseline)
        out_dict = overall_followup_series.to_dict() # Contains baseline fields + calculated overall fields
        
        # Manually add the patient_id and medical_record_id from the groupby keys
        out_dict["patient_id"] = patient_id
        out_dict["medical_record_id"] = medical_record_id
        
        # 2) fixed timepoints
        fixed_timepoints_dict = calc_fixed_timepoints(patient_group_measurements, record_info_with_baseline)
        out_dict.update(fixed_timepoints_dict)
        
        # 3) time-to-event
        time_to_event_dict = calc_time_to_targets(patient_group_measurements, record_info_with_baseline)
        out_dict.update(time_to_event_dict)
        
        output_rows.append(out_dict)

    df_out = pd.DataFrame(output_rows)
    
    # Reorder columns (ensure all columns produced are in output_column_order, or handle missing ones)
    # If some columns might not be generated for all rows (e.g. from record_info_with_baseline if it was skipped),
    # df_out.reindex might introduce NaNs. This should be fine.
    df_out = df_out.reindex(columns=output_column_order)

    conn_out = sqlite3.connect(output_db)
    df_out.to_sql(output_table, conn_out, if_exists="replace", index=False)
    conn_out.close()
    print(f"Saved {len(df_out)} rows to {output_db}::{output_table}")