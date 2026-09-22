import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict
import warnings

# Suppress warnings from empty statistical tests (when groups lack variance)
warnings.filterwarnings('ignore')

def fetch_measurements_metrics(db_path: str, id_col: str = 'medical_record_id') -> pd.DataFrame:
    """
    Extracts raw measurements, converts body comp (except VAT) to absolute kg, and 
    calculates dynamic longitudinal variables grouped by the selected id_col.
    """
    print(f"Fetching and processing longitudinal measurements at the {id_col} level...")
    conn = sqlite3.connect(db_path)
    # Include both IDs to allow dynamic grouping, and bmi which was requested
    query = """
    SELECT patient_id, medical_record_id, measurement_date, weight_kg, bmi, "fat_%", "muscle_%", "vat_%"
    FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    for col in ['weight_kg', 'bmi', 'fat_%', 'muscle_%', 'vat_%']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['weight_kg'])
    
    # Convert fat and muscle to kg, but KEEP vat_% as is
    df['fat_kg'] = (df['fat_%'] / 100) * df['weight_kg']
    df['muscle_kg'] = (df['muscle_%'] / 100) * df['weight_kg']
    
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df = df.sort_values([id_col, 'measurement_date'])
    
    # Calculate days from baseline relative to the id_col group
    baseline_times = df.groupby(id_col)['measurement_date'].transform('min')
    df['days_from_baseline'] = (df['measurement_date'] - baseline_times).dt.days
    
    metrics = []
    
    for group_id, group in df.groupby(id_col):
        n_meas = len(group)
        if n_meas == 1:
            metrics.append({
                id_col: group_id,
                'baseline_weight_kg': group['weight_kg'].iloc[0],
                'baseline_bmi': group['bmi'].iloc[0],
                'baseline_fat_kg': group['fat_kg'].iloc[0],
                'baseline_muscle_kg': group['muscle_kg'].iloc[0],
                'baseline_vat_%': group['vat_%'].iloc[0],
                'total_weight_loss_kg': np.nan,
                'total_bmi_loss': np.nan,
                'total_fat_change_kg': np.nan,
                'total_muscle_change_kg': np.nan,
                'total_vat_change_%': np.nan,
                'weight_loss_nadir_kg': np.nan,
                'bmi_loss_nadir': np.nan,
                'fat_loss_nadir_kg': np.nan,
                'muscle_loss_nadir_kg': np.nan,
                'vat_loss_nadir_%': np.nan,
                'post_nadir_weight_regain_kg': np.nan,
                'post_nadir_bmi_regain': np.nan,
                'post_nadir_fat_regain_kg': np.nan,
                'post_nadir_muscle_regain_kg': np.nan,
                'post_nadir_vat_regain_%': np.nan,
                'days_to_5pct_loss': np.nan,
                'days_to_10pct_loss': np.nan,
                'days_to_15pct_loss': np.nan,
                'dropout_30d': 1, 'dropout_60d': 1, 'dropout_90d': 1, 
                'dropout_120d': 1, 'dropout_150d': 1, 'dropout_180d': 1, 'dropout_360d': 1,
                'total_follow_up_days': 0,
                'n_meas': 1
            })
            continue
            
        first = group.iloc[0]
        last = group.iloc[-1]
        
        # Baselines
        b_weight = first['weight_kg']
        b_bmi = first['bmi']
        b_fat = first['fat_kg']
        b_muscle = first['muscle_kg']
        b_vat = first['vat_%']
        
        # Absolute changes
        w_change = last['weight_kg'] - b_weight
        bmi_change = last['bmi'] - b_bmi
        f_change = last['fat_kg'] - b_fat
        m_change = last['muscle_kg'] - b_muscle
        v_change = last['vat_%'] - b_vat
        
        # Nadir losses (lowest point in trajectory - baseline, so negative means loss)
        w_loss_nadir = group['weight_kg'].min() - b_weight
        bmi_loss_nadir = group['bmi'].min() - b_bmi
        f_loss_nadir = group['fat_kg'].min() - b_fat
        m_loss_nadir = group['muscle_kg'].min() - b_muscle
        v_loss_nadir = group['vat_%'].min() - b_vat
        
        # Post-nadir regain (last point - lowest point)
        w_regain = last['weight_kg'] - group['weight_kg'].min()
        bmi_regain = last['bmi'] - group['bmi'].min()
        f_regain = last['fat_kg'] - group['fat_kg'].min()
        m_regain = last['muscle_kg'] - group['muscle_kg'].min()
        v_regain = last['vat_%'] - group['vat_%'].min()
        
        follow_up_days = last['days_from_baseline']
        
        dropouts = {f'dropout_{d}d': (1 if follow_up_days < d else 0) for d in [30, 60, 90, 120, 150, 180, 360]}
            
        group = group.copy()
        group['pct_weight_loss'] = ((b_weight - group['weight_kg']) / b_weight) * 100
        
        t_5 = group[group['pct_weight_loss'] >= 5]['days_from_baseline'].min()
        t_10 = group[group['pct_weight_loss'] >= 10]['days_from_baseline'].min()
        t_15 = group[group['pct_weight_loss'] >= 15]['days_from_baseline'].min()
        
        metrics.append({
            id_col: group_id,
            'baseline_weight_kg': b_weight,
            'baseline_bmi': b_bmi,
            'baseline_fat_kg': b_fat,
            'baseline_muscle_kg': b_muscle,
            'baseline_vat_%': b_vat,
            'total_weight_loss_kg': w_change,
            'total_bmi_loss': bmi_change,
            'total_fat_change_kg': f_change,
            'total_muscle_change_kg': m_change,
            'total_vat_change_%': v_change,
            'weight_loss_nadir_kg': w_loss_nadir,
            'bmi_loss_nadir': bmi_loss_nadir,
            'fat_loss_nadir_kg': f_loss_nadir,
            'muscle_loss_nadir_kg': m_loss_nadir,
            'vat_loss_nadir_%': v_loss_nadir,
            'post_nadir_weight_regain_kg': w_regain,
            'post_nadir_bmi_regain': bmi_regain,
            'post_nadir_fat_regain_kg': f_regain,
            'post_nadir_muscle_regain_kg': m_regain,
            'post_nadir_vat_regain_%': v_regain,
            'days_to_5pct_loss': t_5,
            'days_to_10pct_loss': t_10,
            'days_to_15pct_loss': t_15,
            **dropouts,
            'total_follow_up_days': follow_up_days,
            'n_meas': n_meas
        })
        
    return pd.DataFrame(metrics)

def fetch_medical_records(db_path: str, id_col: str = 'medical_record_id') -> pd.DataFrame:
    """
    Extracts cross-sectional medical records data. Aggregates data at the patient_id level 
    if requested by picking the first available chronological record.
    """
    print(f"Fetching cross-sectional medical records at the {id_col} level...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT patient_id, medical_record_id, medical_record_sequence, age_when_creating_record, sex_f, height_m,
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
    
    cat_cols = ['pnk_method', 'physical_activity', 'physical_activity_frequency', 'physical_inactivity_cause']
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False, dtype=float)
    
    if id_col == 'patient_id':
        print("Aggregating time-varying covariates to the patient level...")
        # Sort by sequence to ensure chronological order, then groupby patient_id and take first non-null
        df = df.sort_values(['patient_id', 'medical_record_sequence'])
        # .first() automatically skips NaNs in pandas
        df = df.groupby('patient_id').first().reset_index()
        # Drop medical_record_id since we aggregated across multiple
        if 'medical_record_id' in df.columns:
            df = df.drop(columns=['medical_record_id', 'medical_record_sequence'])
            
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
        
    df_avg = df.groupby(['patient_id', 'gene'])['risk_load'].mean().reset_index()
    df_wide = df_avg.pivot(index='patient_id', columns='gene', values='risk_load').reset_index()
    df_wide.columns = ['patient_id'] + [f'gene_{c}_avg_risk' for c in df_wide.columns if c != 'patient_id']
    return df_wide

def build_comparison_table(
    source_db_path: str, 
    out_db_path: str, 
    cluster_table: str, 
    target_table_name: str,
    cohort_names: Dict[int, str],
    id_col: str = 'medical_record_id'
):
    """
    Main orchestration function to build the statistical comparison table.
    
    Inputs:
    - source_db_path: Path to pnk_db2_filtered.sqlite
    - out_db_path: Path to out_db.sqlite containing the cluster labels and target destination.
    - cluster_table: Name of the table with clustering labels.
    - target_table_name: Name of the output table to save the comparisons.
    - cohort_names: Dictionary mapping internal cluster IDs to display names.
    - id_col: The unit of analysis ('medical_record_id' or 'patient_id').
    """
    print(f"Starting Cluster Comparisons Pipeline (Level: {id_col})...")
    
    # 1. Fetch Clusters
    conn = sqlite3.connect(out_db_path)
    df_clusters = pd.read_sql_query(f"SELECT {id_col}, cluster_id FROM {cluster_table}", conn)
    conn.close()
    
    # 2. Extract Data
    df_meas = fetch_measurements_metrics(source_db_path, id_col=id_col)
    df_med = fetch_medical_records(source_db_path, id_col=id_col)
    df_gen = fetch_alleles(source_db_path)
    
    # 3. Merge Data
    print("Merging data...")
    # Measurements and Medical Records always share id_col
    df = df_med.merge(df_meas, on=id_col, how='inner')
    
    # Merge Genomics based on patient_id (which is guaranteed to be in df_med)
    df = df.merge(df_gen, on='patient_id', how='left')
    
    # 4. Map Clusters
    df_clusters = df_clusters.set_index(id_col)
    df['cluster_id'] = df[id_col].map(df_clusters['cluster_id'])
    
    # Identify lost to follow-up (n=1) and assign special ID -1
    df.loc[df['n_meas'] == 1, 'cluster_id'] = -1
    
    valid_cids = list(cohort_names.keys())
    df = df[df['cluster_id'].isin(valid_cids)]
    
    print(f"Data constructed. Total records for comparison: {len(df)}")
    
    # 5. Statistical Comparisons
    print("Running statistical comparisons...")
    skip_cols = [id_col, 'patient_id', 'medical_record_id', 'medical_record_sequence', 'cluster_id', 'n_meas']
    test_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    results = []
    
    # First Row: N of clusters
    n_row = {'Variable': 'Cluster Size (N)'}
    for cid in valid_cids:
        cname = cohort_names[cid]
        n_row[cname] = str(len(df[df['cluster_id'] == cid]))
    n_row['p_value_raw'] = np.nan
    results.append(n_row)
    
    for col in test_cols:
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
                
        if is_categorical:
            ct = pd.crosstab(df['cluster_id'], df[col])
            if ct.size > 0 and ct.shape[0] > 1 and ct.shape[1] > 1:
                try:
                    _, p, _, _ = stats.chi2_contingency(ct)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
        else:
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
        
    df_res['p_value_fdr'] = df_res['p_value_fdr'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    df_res = df_res.drop(columns=['p_value_raw'])
    
    # 7. Save to SQLite
    print(f"Saving comparison table to {out_db_path} -> table '{target_table_name}'...")
    conn = sqlite3.connect(out_db_path)
    df_res.to_sql(target_table_name, conn, if_exists='replace', index=False)
    conn.close()
    
    print("Done! Cluster comparison complete.")
