import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple

def extract_features(db_path: str, id_col: str = 'patient_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts trajectory, kinetics, body composition, and adherence features
    for clustering weight-loss trajectories.
    
    Filters out groups with less than 3 measurements OR less than 30 days of follow-up.
    """
    print(f"Fetching raw longitudinal measurements at the {id_col} level...")
    conn = sqlite3.connect(db_path)
    # Include medical_record_id and patient_id to allow dynamic grouping
    query = """
    SELECT patient_id, medical_record_id, measurement_date, weight_kg, "fat_%", "muscle_%"
    FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    initial_units = df[id_col].nunique()
    initial_pts = len(df)
    print(f"Initial data: {initial_units} {id_col}s, {initial_pts} measurements.")
    
    # Process numeric cols
    for col in ['weight_kg', 'fat_%', 'muscle_%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 1. Drop rows with missing weight, fat, or muscle
    df = df.dropna(subset=['weight_kg', 'fat_%', 'muscle_%'])
    
    clean_units = df[id_col].nunique()
    clean_pts = len(df)
    print(f"After dropping missing weight/fat/muscle entries: {clean_units} {id_col}s, {clean_pts} measurements.")
    
    # Convert body composition to absolute kg values
    df['fat_kg'] = (df['fat_%'] / 100) * df['weight_kg']
    df['muscle_kg'] = (df['muscle_%'] / 100) * df['weight_kg']
    
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df['calendar_date'] = df['measurement_date'].dt.normalize()
    
    # Sort by ID, Calendar Date, and Weight (descending) so that for duplicate dates, the highest weight is kept
    df = df.sort_values([id_col, 'calendar_date', 'weight_kg'], ascending=[True, True, False])
    
    # Drop duplicates on the same day for the same id_col to avoid LOESS division by zero
    df = df.drop_duplicates(subset=[id_col, 'calendar_date'], keep='first')
    dedup_units = df[id_col].nunique()
    dedup_pts = len(df)
    dropped_pts = clean_pts - dedup_pts
    if dropped_pts > 0:
        print(f"Dropped {dropped_pts} same-day duplicate measurements (kept highest weight). Remaining: {dedup_units} {id_col}s, {dedup_pts} measurements.")
    else:
        print(f"No same-day duplicate measurements found. Remaining: {dedup_units} {id_col}s, {dedup_pts} measurements.")
    
    features = []
    excluded = []
    
    grouped = df.groupby(id_col)
    print(f"Processing {len(grouped)} {id_col} units for feature extraction...")
    
    for group_id, group in grouped:
        group = group.copy()
        
        # Calculate days from baseline using the normalized calendar dates
        t0 = group['calendar_date'].iloc[0]
        group['days_from_baseline'] = (group['calendar_date'] - t0).dt.days
        
        n_meas = len(group)
        follow_up_days = group['days_from_baseline'].iloc[-1]
        
        # Exclusion criteria: Must have >= 3 measurements AND >= 30 days follow-up
        if n_meas < 3 or follow_up_days < 30:
            excluded.append({id_col: group_id, 'n_meas': n_meas, 'follow_up_days': follow_up_days})
            continue
            
        b_weight = group['weight_kg'].iloc[0]
        f_weight = group['weight_kg'].iloc[-1]
        
        # Find nadir (lowest weight)
        min_idx = group['weight_kg'].idxmin()
        nadir_wt = group.loc[min_idx, 'weight_kg']
        nadir_day = group.loc[min_idx, 'days_from_baseline']
        
        # Max loss (kg) and % - negative means loss!
        max_change_kg = nadir_wt - b_weight # will be negative if lost weight
        abs_max_loss_kg = abs(max_change_kg) # absolute amount for thresholds
        pct_max_loss = (max_change_kg / b_weight) * 100 if b_weight > 0 else 0.0
        
        # Regain % (always positive if regained)
        if abs_max_loss_kg < 2 and nadir_wt != f_weight:
            pct_regain = 100.0
        else:
            if abs_max_loss_kg > 0:
                pct_regain = ((f_weight - nadir_wt) / abs_max_loss_kg) * 100
            else:
                pct_regain = 0.0 # if no loss at all
                
        # Time to nadir
        time_to_nadir_ratio = nadir_day / follow_up_days if follow_up_days > 0 else 0
        
        # Early WL speed (% change / day)
        # 1. Calculate % change at each point relative to baseline (negative means loss)
        group['pct_change'] = ((group['weight_kg'] - b_weight) / b_weight) * 100
        
        # 2. Slice up to day 30
        early_data = group[group['days_from_baseline'] <= 30]
        if len(early_data) < 2:
            # Expand window to first available post-baseline measurement
            early_data = group.iloc[:2]
            
        if len(early_data) >= 2 and early_data['days_from_baseline'].var() > 0:
            X = sm.add_constant(early_data['days_from_baseline'])
            y = early_data['pct_change']
            model = sm.OLS(y, X).fit()
            early_wl_speed = model.params['days_from_baseline'] if 'days_from_baseline' in model.params else 0.0
        else:
            early_wl_speed = 0.0
        
        # Trajectory Volatility (RMSE around LOESS)
        # frac = 30 days / follow_up_days. 
        # Enforce a minimum fraction so at least 3 points fall in the local window.
        if follow_up_days > 0:
            min_frac = 3.0 / len(group)
            frac = max(30 / follow_up_days, min_frac)
        else:
            frac = 1.0
            
        # Ensure we don't pass duplicate x values to lowess, though we already dropped duplicate dates
        x_vals = group['days_from_baseline'].values
        y_vals = group['pct_change'].values
        
        if len(np.unique(x_vals)) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                try:
                    # return_sorted=False ensures the output matches the input index
                    loess_res = sm.nonparametric.lowess(y_vals, x_vals, frac=frac, return_sorted=False)
                    residuals = y_vals - loess_res
                    rmse_volatility = np.sqrt(np.mean(residuals**2))
                except Exception:
                    rmse_volatility = np.nan
        else:
            rmse_volatility = np.nan
        
        # Lean Loss Coefficient at nadir
        b_muscle = group['muscle_kg'].iloc[0]
        nadir_muscle = group.loc[min_idx, 'muscle_kg']
        
        if pd.isna(b_muscle) or pd.isna(nadir_muscle):
            llc = np.nan
        else:
            if abs_max_loss_kg < 2:
                llc = 0.0
            else:
                delta_muscle = nadir_muscle - b_muscle # negative if lost muscle
                llc_raw = delta_muscle / max_change_kg # neg / neg = positive ratio
                llc = np.clip(llc_raw, 0, 1) # Clip between 0 and 1
                
        # Muscle-fat correlation
        valid_comp = group.dropna(subset=['fat_kg', 'muscle_kg'])
        if len(valid_comp) > 1:
            # check variance to avoid pearsonr error on constant data
            if valid_comp['fat_kg'].var() == 0 or valid_comp['muscle_kg'].var() == 0:
                mf_corr = 0.0
            else:
                mf_corr, _ = pearsonr(valid_comp['fat_kg'], valid_comp['muscle_kg'])
        else:
            mf_corr = np.nan
            
        # Adherence
        gaps = group['days_from_baseline'].diff().dropna()
        longest_gap = gaps.max() if len(gaps) > 0 else 0
        n_30d_gaps = (gaps > 30).sum()
        
        features.append({
            id_col: group_id,
            'pct_max_loss': pct_max_loss,
            'pct_regain': pct_regain,
            'early_wl_speed': early_wl_speed,
            'time_to_nadir_ratio': time_to_nadir_ratio,
            'trajectory_volatility': rmse_volatility,
            'lean_loss_coeff': llc,
            'muscle_fat_corr': mf_corr,
            'observation_duration': follow_up_days,
            'longest_gap': longest_gap,
            'n_30d_gaps': n_30d_gaps
        })
        
    df_features = pd.DataFrame(features)
    df_excluded = pd.DataFrame(excluded)
    
    print(f"\n--- Feature Extraction Summary ---")
    print(f"Retained: {len(df_features)} {id_col} units for clustering.")
    print(f"Excluded: {len(df_excluded)} {id_col} units (due to early attrition: < 30 days follow-up OR < 3 total measurements).")
    return df_features, df_excluded

def eda_features(df_features: pd.DataFrame, id_col: str = 'patient_id'):
    """
    Performs Exploratory Data Analysis on the extracted features.
    Outputs correlation matrices, missingness counts, and distribution plots.
    """
    feature_cols = [c for c in df_features.columns if c != id_col]
    df_plot = df_features[feature_cols].copy()
    
    print("--- Missing Values Count per Feature ---")
    missing = df_plot.isna().sum()
    print(missing)
    print("-" * 40)
    
    # 1. Correlation Matrix
    corr = df_plot.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    # Check for high correlations (> 0.7 or < -0.7)
    print("--- Highly Correlated Pairs (|r| > 0.7) ---")
    high_corr_found = False
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if pd.notna(corr.iloc[i, j]) and abs(corr.iloc[i, j]) > 0.7:
                print(f"{corr.columns[i]} <--> {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
                high_corr_found = True
    if not high_corr_found:
        print("None found.")
        
    # 2. Distributions
    n_cols = 3
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        # Drop NaNs before plotting
        data = df_plot[col].dropna()
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
        
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    
    # 3. Descriptive Stats
    print("\\n--- Descriptive Statistics ---")
    from IPython.display import display
    display(df_plot.describe().T)
