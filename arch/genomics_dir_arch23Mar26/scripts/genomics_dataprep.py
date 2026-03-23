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
    # Create new table names measurements_p2 and medical_records_p2.
    table_mapping = {
        "medical_records_filtered": ("medical_records_genomics", record_ids),
        "measurements_filtered": ("measurements_genomics", record_ids)
    }

    # Create new SQLite database connection to write filtered data.
    gen_in_conn = sqlite3.connect(config.paths.paper_in_db)

    for src_table, (dst_table, mr_ids) in table_mapping.items():
        # Prepare filtering query for tables that include 'medical_record_id'
        if src_table in ("medical_records_filtered", "measurements_filtered"):
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
    """Generates the final column order based on the provided configuration."""

    # cols = config.followup_columns.copy()
    # cols += config.predictor_columns
    cols = config.predictor_columns.copy()
    cols += config.followup_columns
    for w in config.time_windows:
        prefix = f"{w}d"
        cols += [
            f"{prefix}_weight_kg", f"{prefix}_wl_kg", f"{prefix}_wl_%",
            f"{prefix}_bmi", f"{prefix}_bmi_reduction",
            f"{prefix}_fat_%", f"{prefix}_fat_loss_%",
            f"{prefix}_muscle_%", f"{prefix}_muscle_change_%",
            f"{prefix}_date", f"days_to_{prefix}_measurement", f"{prefix}_dropout"
        ]
    for t in config.weight_loss_targets:
        prefix = f"{t}%_wl"
        cols += [
            f"{prefix}_achieved", f"{prefix}_%", f"{prefix}_date", f"days_to_{prefix}"
        ]
    return cols

def load_measurements(conn, config: timetoevent_config) -> pd.DataFrame:
    # Consider selecting only necessary columns if measurements_p1 has many unused ones
    cols_to_fetch = ["patient_id", "medical_record_id", "measurement_date", 
                      "first_in_record", "weight_kg", "bmi", "fat_%", "muscle_%"]
    cols_to_fetch = [f'"{col}"' for col in cols_to_fetch]  # Ensure proper quoting for SQL, as some columns have special characters
    # Check if any other columns from measurements_p1 are implicitly used. If not, this is safer.
    # If you are sure all columns are needed, use "SELECT *"
    # Query measurements from SQL table, getting table name directly from config object
    sql_query = f"SELECT {','.join(cols_to_fetch)} FROM {config.input_measurements}"
    df = pd.read_sql(sql_query, conn, parse_dates=["measurement_date"])
    df = df.sort_values(["patient_id", "medical_record_id", "measurement_date"])
    return df

def load_med_records(conn, config: timetoevent_config) -> pd.DataFrame:
    # Gets table name and columns to fetch from the config object
    cols_to_fetch = ", ".join(config.fetch_from_records)
    sql_query = f"SELECT {cols_to_fetch} FROM {config.input_records}"
    df = pd.read_sql(sql_query, conn)
    return df

def extract_baseline(meas: pd.DataFrame) -> pd.DataFrame:
    base = meas[meas["first_in_record"] == 1].copy()
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

'''
2b. time-to-event type calculations - overall followup, outcome-at-timestamp, time-to-target data
'''

def calc_overall_followup(patient_record_measurements: pd.DataFrame, 
                          baseline_row_info: pd.Series
                          ) -> pd.Series:
    baseline_date = baseline_row_info["baseline_date"]
    
    followup_measurements = patient_record_measurements[
        patient_record_measurements["measurement_date"] > baseline_date
    ] # patient_record_measurements is already sorted by date

    last = baseline_row_info.copy() # Start with baseline info (which is a Series)
    last["instant_dropout"] = 0 # Initialize instant_dropout

    if followup_measurements.empty:
        # last["last_aval_date"] = baseline_row_info["baseline_date"]
        last["final_date"] = baseline_row_info["baseline_date"]
        last["total_followup_days"] = 1
        last["instant_dropout"] = 1 # Set to 1 if only baseline measurement exists
        # last["last_aval_weight_kg"] = baseline_row_info["baseline_weight_kg"]
        last["final_weight_kg"] = baseline_row_info["baseline_weight_kg"]
        last["total_wl_kg"], last["total_wl_%"] = 0.0, 0.0
        last["final_bmi"] = baseline_row_info["baseline_bmi"]
        last["bmi_reduction"] = 0.0
        # last["last_aval_fat_%"] = baseline_row_info["baseline_fat_%"]
        last["final_fat_%"] = baseline_row_info["baseline_fat_%"]
        last["total_fat_loss_%"] = 0.0
        # last["last_aval_muscle_%"] = baseline_row_info["baseline_muscle_%"]
        last["final_muscle_%"] = baseline_row_info["baseline_muscle_%"]
        last["total_muscle_change_%"] = 0.0
    else:
        last_meas = followup_measurements.iloc[-1]
        dt = (last_meas.measurement_date - baseline_date).days + 1
        # last["last_aval_date"] = last_meas.measurement_date
        last["final_date"] = last_meas.measurement_date
        last["total_followup_days"] = dt
        # last["last_aval_weight_kg"] = last_meas.weight_kg
        last["final_weight_kg"] = last_meas.weight_kg
        last["total_wl_kg"] = last_meas.weight_kg - baseline_row_info["baseline_weight_kg"]
        last["total_wl_%"] = 100 * last["total_wl_kg"] / baseline_row_info["baseline_weight_kg"] if baseline_row_info["baseline_weight_kg"] else 0
        last["final_bmi"] = last_meas.bmi
        last["bmi_reduction"] = last_meas.bmi - baseline_row_info["baseline_bmi"]
        # last["last_aval_fat_%"] = last_meas["fat_%"]
        last["final_fat_%"] = last_meas["fat_%"]
        last["total_fat_loss_%"] = last_meas["fat_%"] - baseline_row_info["baseline_fat_%"]
        # last["last_aval_muscle_%"] = last_meas["muscle_%"]
        last["final_muscle_%"] = last_meas["muscle_%"]
        last["total_muscle_change_%"] = last_meas["muscle_%"] - baseline_row_info["baseline_muscle_%"]
    
    last["nr_total_measurements"] = len(patient_record_measurements)
    if last["nr_total_measurements"] > 1:
        # Denominator (last["nr_total_measurements"] - 1) cannot be zero here
        last["avg_days_between_measurements"] = \
            (last["total_followup_days"] - 1) / (last["nr_total_measurements"] - 1)
    else:
        last["avg_days_between_measurements"] = np.nan
    return last

def calc_fixed_timepoints(patient_record_measurements: pd.DataFrame, 
                          baseline_row_info: pd.Series, 
                          config: timetoevent_config) -> dict:
    out = {}
    baseline_date = baseline_row_info["baseline_date"]
    baseline_weight_kg = baseline_row_info["baseline_weight_kg"]
    baseline_bmi = baseline_row_info["baseline_bmi"]
    baseline_fat_pct = baseline_row_info["baseline_fat_%"]
    baseline_muscle_pct = baseline_row_info["baseline_muscle_%"]

    for w in config.time_windows:
        target_date = baseline_date + timedelta(days=w)
        lo = baseline_date + timedelta(days=w - config.window_span)
        hi = baseline_date + timedelta(days=w + config.window_span)

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

def calc_time_to_targets(patient_record_measurements: pd.DataFrame, 
                         baseline_row_info: pd.Series, 
                         config: timetoevent_config) -> dict:
    out = {}
    baseline_date = baseline_row_info["baseline_date"]
    baseline_weight = baseline_row_info["baseline_weight_kg"]

    # Filter measurements strictly after baseline date; patient_record_measurements is already sorted
    group = patient_record_measurements[patient_record_measurements["measurement_date"] > baseline_date]

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
            out[f"days_to_{t}%_wl"] = (first_achieved_event["measurement_date"] - baseline_date).days + 1
        else:
            out[f"{t}%_wl_achieved"] = 0
            out[f"{t}%_wl_%"] = np.nan 
            out[f"{t}%_wl_date"] = pd.NaT
            out[f"days_to_{t}%_wl"] = np.nan
    return out

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
    conn_in.close()

    baseline_data = extract_baseline(all_measurements)
    merged_patient_records = merge_baseline_and_records(baseline_data, medical_records_data)
    merged_patient_records_indexed = merged_patient_records.set_index(["patient_id", "medical_record_id"])

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
        # Manually add the patient_id and medical_record_id from the groupby keys
        out_dict["patient_id"] = patient_id
        out_dict["medical_record_id"] = medical_record_id

        # 2) fixed timepoints
        fixed_timepoints = calc_fixed_timepoints(patient_group_measurements, record_info_with_baseline, timetoevent_config)
        out_dict.update(fixed_timepoints)

        # 3) time-to-event
        time_to_targets = calc_time_to_targets(patient_group_measurements, record_info_with_baseline, timetoevent_config)
        out_dict.update(time_to_targets)

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
                
                # Separate conditions into existence checks (columns must be non-null) and value checks (col == val)
                cols_to_check_existence = []
                value_filters = []

                for cond in conditions:
                    if isinstance(cond, str):
                        cols_to_check_existence.append(cond)
                    elif isinstance(cond, (list, tuple)) and len(cond) == 2:
                        value_filters.append(cond)
                    else:
                        print(f"  - Warning: Invalid condition format '{cond}' for table '{output_table}'. Skipping.")

                # Check if all required columns exist in the DataFrame
                required_cols = cols_to_check_existence + [c for c, _ in value_filters]
                missing_cols = [col for col in required_cols if col not in source_df.columns]
                
                if missing_cols:
                    print(f"  - Warning: Skipping table '{output_table}' because columns {missing_cols} were not found.")
                    continue

                # Apply Filtering
                subset_df = source_df.copy()

                # 1. Existence Check (Not Null)
                if cols_to_check_existence:
                    subset_df = subset_df.dropna(subset=cols_to_check_existence)

                # 2. Value Check (Equality)
                for col, val in value_filters:
                    subset_df = subset_df[subset_df[col] == val]

                # Save result
                subset_df.to_sql(output_table, conn, if_exists="replace", index=False)
                
                print(f"  - Saved {len(subset_df)} rows to table '{output_table}'")
                print(f"    Filters applied: Existence={cols_to_check_existence}, Values={value_filters}")

    except Exception as e:
        print(f"An error occurred during subsetting: {e}")
    
    print("--- Table Subsetting Complete ---")

