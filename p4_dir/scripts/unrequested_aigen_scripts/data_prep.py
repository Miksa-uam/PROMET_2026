import sqlite3
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from typing import Tuple, List

def load_and_filter_measurements(db_path: str) -> pd.DataFrame:
    """
    Loads patient measurement data from the SQLite database and applies initial quality filters.
    
    Methodological logic:
    - We extract only the columns relevant to our multivariate time series clustering:
      weight_kg, fat_%, and muscle_%.
    - DTW requires complete multivariate points at each time step, so we drop rows 
      with missing values in any of our 3 core variables. We use a strict complete-case 
      approach to avoid imputing synthetic clinical values that might skew longitudinal patterns.
    - Time series analysis requires sequences, so we enforce a minimum of 2 timepoints per patient.
    
    Args:
        db_path (str): Path to the SQLite database.
        
    Returns:
        pd.DataFrame: A cleaned pandas DataFrame with complete multivariate data 
                      and >=2 measurements per patient.
    """
    print(f"Loading data from {db_path}...")
    
    # 1. Connect to database and load relevant columns
    # Note: Double quotes are used to safely select column names containing '%'
    conn = sqlite3.connect(db_path)
    query = """
        SELECT patient_id, measurement_date, weight_kg, "fat_%", "muscle_%"
        FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    initial_rows = len(df)
    initial_patients = df['patient_id'].nunique()
    print(f"Initial data: {initial_rows} rows, {initial_patients} patients.")
    
    # 2. Filter out rows with missing data in any of the 3 key variables
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    df = df.dropna(subset=target_cols)
    print(f"After dropping missing values: {len(df)} rows.")
    
    # 3. Filter out patients with fewer than 2 weigh-ins
    counts = df.groupby('patient_id').size()
    valid_patients = counts[counts >= 2].index
    df = df[df['patient_id'].isin(valid_patients)]
    
    final_patients = df['patient_id'].nunique()
    print(f"After enforcing >=2 timepoints: {len(df)} rows, {final_patients} patients remaining.")
    
    return df

def calculate_percentage_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the data chronologically and converts absolute values into percentage change 
    from baseline for each patient.
    
    Methodological logic:
    - To cluster patients based on their *patterns* of weight loss (rather than absolute starting 
      weights which could group patients simply by their initial obesity level), we normalize 
      each trajectory to its own baseline.
    - Formula: (current - baseline) / baseline * 100.
      The baseline itself evaluates to 0.0.
    
    Args:
        df (pd.DataFrame): Cleaned measurement dataframe.
        
    Returns:
        pd.DataFrame: DataFrame with the absolute values replaced by percentage changes.
    """
    print("Calculating percentage change from baseline...")
    
    # 1. Ensure chronological order per patient
    df = df.sort_values(by=['patient_id', 'measurement_date']).copy()
    
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    
    # 2. Calculate percentage change
    # Group by patient, select the first chronological row as baseline
    baselines = df.groupby('patient_id')[target_cols].transform('first')
    
    # Apply the formula: (current - baseline) / baseline * 100
    df[target_cols] = ((df[target_cols] - baselines) / baselines) * 100
    
    print("Baseline normalization complete. Baseline values are now 0.0.")
    return df

def prepare_tslearn_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Pivots the longitudinal dataframe into the 3D numpy array format required by tslearn.
    
    Methodological logic:
    - tslearn algorithms (like TimeSeriesKMeans) expect a 3D array of shape 
      (n_patients, max_timepoints, n_dimensions).
    - Since patients have varying numbers of follow-up visits, we use `to_time_series_dataset`
      which naturally handles uneven sequence lengths by padding with NaNs at the end 
      of shorter sequences. DTW natively handles these NaN-padded unequal lengths.
      
    Args:
        df (pd.DataFrame): Normalized measurement dataframe.
        
    Returns:
        Tuple[np.ndarray, List[str]]: 
            - ts_dataset: The 3D numpy array formatted for tslearn.
            - patient_ids: A list of patient IDs corresponding to the first dimension of the array,
                           allowing us to map clusters back to clinical profiles later.
    """
    print("Reshaping data for tslearn...")
    
    # Ensure sorted order again for sequence extraction
    df = df.sort_values(by=['patient_id', 'measurement_date'])
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    
    # Extract sequences per patient into a list of 2D arrays (timepoints x dimensions)
    sequences = []
    patient_ids = []
    
    for patient_id, group in df.groupby('patient_id'):
        patient_ids.append(patient_id)
        # Extract only the 3 numerical columns as a 2D numpy array
        seq = group[target_cols].values
        sequences.append(seq)
        
    # Convert list of unequal-length 2D arrays into a padded 3D array
    ts_dataset = to_time_series_dataset(sequences)
    
    print(f"Dataset reshaped successfully. Shape: {ts_dataset.shape}")
    print(f"(patients: {ts_dataset.shape[0]}, max_timepoints: {ts_dataset.shape[1]}, dimensions: {ts_dataset.shape[2]})")
    
    return ts_dataset, patient_ids
