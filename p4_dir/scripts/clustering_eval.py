import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics import silhouette_score, davies_bouldin_score
import statsmodels.api as sm
import warnings

def evaluate_clusters(X: pd.DataFrame, labels: np.ndarray, model=None, model_type: str = 'GMM') -> dict:
    """
    Computes internal clustering quality indices.
    """
    # Filter out noise points (-1) for metric calculations
    mask = labels != -1
    if sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {'Silhouette': np.nan, 'Davies-Bouldin': np.nan, 'BIC': np.nan}
        
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    metrics = {}
    metrics['Silhouette'] = silhouette_score(X_valid, labels_valid)
    metrics['Davies-Bouldin'] = davies_bouldin_score(X_valid, labels_valid)
    
    if model_type.upper() == 'GMM' and model is not None:
        metrics['BIC'] = model.bic(X)
    else:
        metrics['BIC'] = np.nan
        
    return metrics

def run_clustering_grid_search(df_features: pd.DataFrame, models=None, blocks=None, k_range=range(2, 9)):
    """
    Runs a comprehensive grid search over feature blocks, models, and hyperparameters (k or min_cluster_size).
    Evaluates internal clustering metrics and returns a DataFrame of results.
    """
    from clustering_models import prepare_data, fit_model, FEATURE_BLOCKS
    
    if isinstance(k_range, int):
        k_range = [k_range]
        
    if models is None:
        models = ['GMM', 'WARD', 'KMEANS', 'HDBSCAN', 'UMAP-HDBSCAN'] 
    if blocks is None:
        blocks = list(FEATURE_BLOCKS.keys())
        
    results = []
    
    print(f"Running grid search across {len(blocks)} blocks and {len(models)} models...")
    
    for block in blocks:
        try:
            X_scaled, _ = prepare_data(df_features, block=block)
        except Exception as e:
            print(f"Skipping block {block} due to error: {e}")
            continue
            
        for model_type in models:
            for k in k_range:
                try:
                    kwargs = {}
                    if model_type in ['HDBSCAN', 'UMAP_HDBSCAN']:
                        # For N > 10,000, min_cluster_size needs to be in the hundreds to find macro-phenotypes
                        kwargs['min_cluster_size'] = k * 100
                        kwargs['min_samples'] = max(15, int(k * 10))
                        param_val = f"size:{kwargs['min_cluster_size']}_samp:{kwargs['min_samples']}"
                    else:
                        param_val = str(k)
                        
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        labels, model = fit_model(X_scaled, model_type=model_type, k=k, **kwargs)
                    
                    metrics = evaluate_clusters(X_scaled, labels, model=model, model_type=model_type)
                    
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    noise_ratio = np.mean(labels == -1) if -1 in labels else 0.0
                    
                    results.append({
                        'Block': block,
                        'Model': model_type,
                        'Param (k/size)': param_val,
                        'Clusters Found': n_clusters_found,
                        'Noise Ratio': noise_ratio,
                        'Silhouette': metrics['Silhouette'],
                        'Davies-Bouldin': metrics['Davies-Bouldin'],
                        'BIC': metrics['BIC']
                    })
                except Exception as e:
                    # Silently skip failed setups
                    pass
                    
    df_res = pd.DataFrame(results)
    
    if len(df_res) > 0:
        # Filter out setups that only found 1 cluster or too many clusters
        valid_res = df_res[(df_res['Clusters Found'] > 1) & (df_res['Clusters Found'] < 15)]
        
        # Rank by Silhouette first (higher is better)
        df_top = valid_res.sort_values(by='Silhouette', ascending=False).head(5)
        print("\\n--- Top 5 Clustering Candidate Setups (by Silhouette) ---")
        from IPython.display import display
        display(df_top)
    
    return df_res

def evaluate_predictive_utility(db_path: str, df_clusters: pd.DataFrame, id_col: str = 'patient_id'):
    """
    Runs a quick multinomial logistic regression to predict cluster membership 
    from basic demographic predictors (age, sex, and baseline BMI).
    """
    print("Evaluating predictive utility (baseline characteristics vs cluster)...")
    
    print("\n--- CLUSTER SIZES ---")
    print(df_clusters['cluster_id'].value_counts().sort_index().to_string())
    print("-" * 21 + "\n")
    
    conn = sqlite3.connect(db_path)
    
    query_demo = """
    SELECT patient_id, medical_record_id, age_when_creating_record as age, sex_f as sex
    FROM medical_records_filtered
    """
    df_demo = pd.read_sql_query(query_demo, conn)
    
    query_meas = """
    SELECT patient_id, medical_record_id, measurement_date, bmi
    FROM measurements_filtered
    """
    df_meas = pd.read_sql_query(query_meas, conn)
    conn.close()
    
    df_demo = df_demo.drop_duplicates(subset=[id_col])
    
    df_meas['measurement_date'] = pd.to_datetime(df_meas['measurement_date'])
    df_meas = df_meas.sort_values([id_col, 'measurement_date'])
    df_bmi = df_meas.groupby(id_col).first().reset_index()
    
    df_baseline = df_demo.merge(df_bmi[[id_col, 'bmi']], on=id_col, how='inner')
    
    df = df_clusters.merge(df_baseline, on=id_col, how='inner')
    
    for col in ['age', 'sex', 'bmi']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['age', 'sex', 'bmi', 'cluster_id'])
    
    # Filter out noise clusters
    df = df[df['cluster_id'] != -1]
    
    unique_clusters = df['cluster_id'].unique()
    if len(unique_clusters) < 2:
        print("Not enough distinct clusters for regression.")
        return None
        
    remap = {c: i for i, c in enumerate(sorted(unique_clusters))}
    df['y'] = df['cluster_id'].map(remap)
    
    X = df[['age', 'sex', 'bmi']]
    X = sm.add_constant(X)
    y = df['y']
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.MNLogit(y, X).fit(disp=False)
        print(model.summary())
        return model
    except Exception as e:
        print(f"Regression failed: {e}")
        return None
