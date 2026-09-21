import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict
import warnings

# Suppress warnings from empty statistical tests (when groups lack variance)
warnings.filterwarnings('ignore')

def fetch_measurements_metrics(db_path: str) -> pd.DataFrame:
    """
    Extracts raw measurements, converts body comp to absolute kg, and 
    calculates dynamic longitudinal variables for each medical record.
    """
    print("Fetching and processing longitudinal measurements...")
    conn = sqlite3.connect(db_path)
    # Using coalesce to handle missing columns if they don't exist in some versions,
    # but we assume the standard db structure here.
    query = """
    SELECT medical_record_id, measurement_date, weight_kg, "fat_%", "muscle_%", "vat_%"
    FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Safely convert to numeric
    for col in ['weight_kg', 'fat_%', 'muscle_%', 'vat_%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows without weight_kg as we can't absolutize without it
    df = df.dropna(subset=['weight_kg'])
    
    # Convert % to kg
    df['fat_kg'] = (df['fat_%'] / 100) * df['weight_kg']
    df['muscle_kg'] = (df['muscle_%'] / 100) * df['weight_kg']
    df['vat_kg'] = (df['vat_%'] / 100) * df['weight_kg']
    
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df = df.sort_values(['medical_record_id', 'measurement_date'])
    
    # Calculate days from baseline
    baseline_times = df.groupby('medical_record_id')['measurement_date'].transform('min')
    df['days_from_baseline'] = (df['measurement_date'] - baseline_times).dt.days
    
    metrics = []
    
    for mr_id, group in df.groupby('medical_record_id'):
        n_meas = len(group)
        if n_meas == 1:
            # Lost to follow-up cohort
            metrics.append({
                'medical_record_id': mr_id,
                'total_weight_loss_kg': np.nan,
                'total_fat_change_kg': np.nan,
                'total_muscle_change_kg': np.nan,
                'total_vat_change_kg': np.nan,
                'days_to_5pct_loss': np.nan,
                'days_to_10pct_loss': np.nan,
                'days_to_15pct_loss': np.nan,
                'dropout_30d': 1,
                'dropout_60d': 1,
                'dropout_90d': 1,
                'dropout_120d': 1,
                'dropout_150d': 1,
                'dropout_180d': 1,
                'dropout_360d': 1,
                'total_follow_up_days': 0,
                'n_meas': 1
            })
            continue
            
        first = group.iloc[0]
        last = group.iloc[-1]
        
        # Absolute changes
        w_change = last['weight_kg'] - first['weight_kg']
        f_change = last['fat_kg'] - first['fat_kg']
        m_change = last['muscle_kg'] - first['muscle_kg']
        v_change = last['vat_kg'] - first['vat_kg']
        
        follow_up_days = last['days_from_baseline']
        
        # Dropouts (1 if they dropped out before day X, 0 otherwise)
        dropouts = {f'dropout_{d}d': (1 if follow_up_days < d else 0) for d in [30, 60, 90, 120, 150, 180, 360]}
            
        # Time to X% weight loss
        group = group.copy()
        group['pct_weight_loss'] = ((first['weight_kg'] - group['weight_kg']) / first['weight_kg']) * 100
        
        t_5 = group[group['pct_weight_loss'] >= 5]['days_from_baseline'].min()
        t_10 = group[group['pct_weight_loss'] >= 10]['days_from_baseline'].min()
        t_15 = group[group['pct_weight_loss'] >= 15]['days_from_baseline'].min()
        
        metrics.append({
            'medical_record_id': mr_id,
            'total_weight_loss_kg': w_change,
            'total_fat_change_kg': f_change,
            'total_muscle_change_kg': m_change,
            'total_vat_change_kg': v_change,
            'days_to_5pct_loss': t_5,
            'days_to_10pct_loss': t_10,
            'days_to_15pct_loss': t_15,
            **dropouts,
            'total_follow_up_days': follow_up_days,
            'n_meas': n_meas
        })
        
    return pd.DataFrame(metrics)

def fetch_medical_records(db_path: str) -> pd.DataFrame:
    """
    Extracts cross-sectional medical records data and dummy-encodes multi-value 
    categoricals (comorbidities, drugs, survey variables).
    """
    print("Fetching and processing cross-sectional medical records...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT patient_id, medical_record_id, age_when_creating_record, sex_f, height_m,
           wc_cm_confirm_time, pnk_method, nr_medical_records_patient, dietitian_visits,
           physical_activity, physical_activity_frequency, physical_inactivity_cause,
           womens_health_and_pregnancy, mental_health, family_issues, medication_disease_injury,
           physical_inactivity, eating_habits, schedule, smoking_cessation, treatment_discontinuation_or_relapse,
           pandemic, lifestyle_circumstances, none_of_above, smoking_yn,
           hunger_yn, satiety_yn, emotional_eating_yn, emotional_eating_value_likert,
           quantity_control_likert, impulse_control_likert,
           comorbidity1, comorbidity2, comorbidity3, comorbidity4, comorbidity5, comorbidity6,
           drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8
    FROM medical_records_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Process comorbidities into binary flags
    como_cols = [f'comorbidity{i}' for i in range(1, 7)]
    all_comos = pd.unique(df[como_cols].values.ravel('K'))
    all_comos = [c for c in all_comos if pd.notna(c) and c != '']
    for c in all_comos:
        df[f'como_{c}'] = (df[como_cols] == c).any(axis=1).astype(float)
        
    # Process drugs into binary flags
    drug_cols = [f'drug{i}' for i in range(1, 9)]
    all_drugs = pd.unique(df[drug_cols].values.ravel('K'))
    all_drugs = [d for d in all_drugs if pd.notna(d) and d != '']
    for d in all_drugs:
        df[f'drug_{d}'] = (df[drug_cols] == d).any(axis=1).astype(float)
        
    df = df.drop(columns=como_cols + drug_cols)
    
    # One-hot encode string categoricals
    cat_cols = ['pnk_method', 'physical_activity', 'physical_activity_frequency', 'physical_inactivity_cause']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=float)
    
    return df

def fetch_alleles(db_path: str) -> pd.DataFrame:
    """
    Extracts genomics data and calculates the average risk load per gene per patient.
    """
    print("Fetching and processing genomics data...")
    conn = sqlite3.connect(db_path)
    query = "SELECT patient_id, gene, risk_load FROM alleles_filtered"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return pd.DataFrame(columns=['patient_id'])
        
    # Average risk load per gene per patient
    df_avg = df.groupby(['patient_id', 'gene'])['risk_load'].mean().reset_index()
    
    # Pivot to wide format
    df_wide = df_avg.pivot(index='patient_id', columns='gene', values='risk_load').reset_index()
    df_wide.columns = ['patient_id'] + [f'gene_{c}_avg_risk' for c in df_wide.columns if c != 'patient_id']
    return df_wide

def build_comparison_table(
    source_db_path: str, 
    out_db_path: str, 
    cluster_table: str, 
    target_table_name: str,
    cohort_names: Dict[int, str]
):
    """
    Main orchestration function to build the statistical comparison table.
    
    Inputs:
    - source_db_path: Path to pnk_db2_filtered.sqlite
    - out_db_path: Path to out_db.sqlite containing the cluster labels and target destination.
    - cluster_table: Name of the table with clustering labels.
    - target_table_name: Name of the output table to save the comparisons.
    - cohort_names: Dictionary mapping internal cluster IDs to display names 
      (e.g., {-1: 'lost_to_followup', 0: 'cl0_fast', 1: 'cl1_slow'}).
    """
    print("Starting Cluster Comparisons Pipeline...")
    
    # 1. Fetch Clusters
    conn = sqlite3.connect(out_db_path)
    df_clusters = pd.read_sql_query(f"SELECT medical_record_id, cluster_id FROM {cluster_table}", conn)
    conn.close()
    
    # 2. Extract Data
    df_meas = fetch_measurements_metrics(source_db_path)
    df_med = fetch_medical_records(source_db_path)
    df_gen = fetch_alleles(source_db_path)
    
    # 3. Merge Data
    print("Merging data...")
    df = df_med.merge(df_meas, on='medical_record_id', how='inner')
    df = df.merge(df_gen, on='patient_id', how='left')
    
    # 4. Map Clusters
    df_clusters = df_clusters.set_index('medical_record_id')
    df['cluster_id'] = df['medical_record_id'].map(df_clusters['cluster_id'])
    
    # Identify lost to follow-up (n=1) and assign special ID -1
    df.loc[df['n_meas'] == 1, 'cluster_id'] = -1
    
    # Filter to only the requested cohorts (drops unmapped highly adherent records and unwanted clusters like cluster 2)
    valid_cids = list(cohort_names.keys())
    df = df[df['cluster_id'].isin(valid_cids)]
    
    print(f"Data constructed. Total records for comparison: {len(df)}")
    
    # 5. Statistical Comparisons
    print("Running statistical comparisons...")
    skip_cols = ['medical_record_id', 'patient_id', 'cluster_id', 'n_meas']
    test_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    results = []
    
    for col in test_cols:
        # Determine if binary categorical (dummy) or continuous
        unique_vals = set(df[col].dropna().unique())
        is_categorical = unique_vals.issubset({0, 1, 0.0, 1.0})
        
        row = {'Variable': col}
        groups_data = []
        
        for cid in valid_cids:
            cname = cohort_names[cid]
            group_data = df[df['cluster_id'] == cid][col].dropna()
            groups_data.append(group_data)
            n = len(group_data)
            
            if is_categorical:
                count = int(group_data.sum()) if n > 0 else 0
                pct = (count / n * 100) if n > 0 else 0
                row[cname] = f"{count} ({pct:.1f}%)"
            else:
                mean = group_data.mean()
                std = group_data.std()
                row[cname] = f"{mean:.2f} ± {std:.2f} (n={n})"
                
        # Run tests
        if is_categorical:
            # Chi-Square
            ct = pd.crosstab(df['cluster_id'], df[col])
            # Valid chi-square requires at least a 2x2 structure and non-zero counts
            if ct.size > 0 and ct.shape[0] > 1 and ct.shape[1] > 1:
                try:
                    _, p, _, _ = stats.chi2_contingency(ct)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
        else:
            # Kruskal-Wallis
            valid_groups = [g for g in groups_data if len(g) > 0]
            if len(valid_groups) >= 2:
                try:
                    _, p = stats.kruskal(*valid_groups)
                except ValueError:
                    p = np.nan
            else:
                p = np.nan
                
        row['p_value_raw'] = p
        results.append(row)
        
    df_res = pd.DataFrame(results)
    
    # 6. FDR Correction (Benjamini-Hochberg)
    valid_p_mask = df_res['p_value_raw'].notna()
    df_res['p_value_fdr'] = np.nan
    
    if valid_p_mask.sum() > 0:
        _, p_adj, _, _ = multipletests(df_res.loc[valid_p_mask, 'p_value_raw'], method='fdr_bh')
        df_res.loc[valid_p_mask, 'p_value_fdr'] = p_adj
        
    # Format p-values for display
    df_res['p_value_fdr'] = df_res['p_value_fdr'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    df_res = df_res.drop(columns=['p_value_raw'])
    
    # 7. Save to SQLite
    print(f"Saving comparison table to {out_db_path} -> table '{target_table_name}'...")
    conn = sqlite3.connect(out_db_path)
    df_res.to_sql(target_table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    print("Done! Cluster comparison complete.")
