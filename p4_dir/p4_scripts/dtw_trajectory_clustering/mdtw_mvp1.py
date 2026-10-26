import sqlite3
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import Tuple, List

# -------------------------------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------------------------------

def load_and_filter_measurements(db_path: str) -> pd.DataFrame:
    """
    Loads patient measurement data from the SQLite database and applies initial quality filters.
    
    Methodological logic:
    - Extract only columns relevant to our multivariate clustering: weight_kg, fat_%, and muscle_%.
    - DTW requires complete multivariate points at each time step, so we drop rows 
      with missing values in any of our 3 core variables.
    - Time series analysis requires sequences, so we enforce >= 2 timepoints per patient.
    """
    print(f"Loading data from {db_path}...")
    
    conn = sqlite3.connect(db_path)
    query = """
        SELECT patient_id, measurement_date, weight_kg, "fat_%", "muscle_%"
        FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Initial data: {len(df)} rows, {df['patient_id'].nunique()} patients.")
    
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    df = df.dropna(subset=target_cols)
    print(f"After dropping missing values: {len(df)} rows.")
    
    counts = df.groupby('patient_id').size()
    valid_patients = counts[counts >= 2].index
    df = df[df['patient_id'].isin(valid_patients)]
    
    print(f"After enforcing >=2 timepoints: {len(df)} rows, {df['patient_id'].nunique()} patients remaining.")
    # NEW STEP: Remove extreme outliers to save DTW computation time
    counts = df.groupby('patient_id').size()
    
    # Check what the 95th percentile of visit counts is (likely around 20-30)
    # and filter out anyone with more than 50 weigh-ins.
    valid_patients_cap = counts[(counts >= 2) & (counts <= 50)].index
    df = df[df['patient_id'].isin(valid_patients_cap)]
    
    print(f"After capping at 50 timepoints: {len(df)} rows, {df['patient_id'].nunique()} patients.")
    
    return df

def calculate_days_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the elapsed time in days from the first measurement for each patient.
    This provides a continuous time axis for LOWESS smoothing plots.
    """
    print("Calculating days from baseline for time axis...")
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df = df.sort_values(by=['patient_id', 'measurement_date'])
    
    baselines_time = df.groupby('patient_id')['measurement_date'].transform('min')
    df['days_from_baseline'] = (df['measurement_date'] - baselines_time).dt.days
    
    return df

def calculate_percentage_from_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the data chronologically and converts absolute values into percentage change 
    from baseline for each patient.
    """
    print("Calculating percentage change from baseline...")
    df = df.sort_values(by=['patient_id', 'measurement_date']).copy()
    
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    baselines = df.groupby('patient_id')[target_cols].transform('first')
    
    df[target_cols] = ((df[target_cols] - baselines) / baselines) * 100
    
    return df

def prepare_tslearn_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Pivots the longitudinal dataframe into the 3D numpy array format required by tslearn.
    """
    print("Reshaping data for tslearn...")
    df = df.sort_values(by=['patient_id', 'measurement_date'])
    target_cols = ['weight_kg', 'fat_%', 'muscle_%']
    
    sequences = []
    patient_ids = []
    
    for patient_id, group in df.groupby('patient_id'):
        patient_ids.append(patient_id)
        seq = group[target_cols].values
        sequences.append(seq)
        
    ts_dataset = to_time_series_dataset(sequences)
    print(f"Dataset reshaped successfully. Shape: {ts_dataset.shape}")
    
    return ts_dataset, patient_ids

# -------------------------------------------------------------------------
# CLUSTERING MVP
# -------------------------------------------------------------------------

def run_dtw_clustering_search(
    ts_dataset: np.ndarray, 
    k_min: int = 2, 
    k_max: int = 7, 
    n_max_silhouette: int = 1000
) -> Tuple[int, TimeSeriesKMeans, np.ndarray]:
    """
    Runs DTW TimeSeriesKMeans for k in [k_min, k_max] and evaluates using silhouette score.
    To manage the O(N^2) complexity of DTW silhouette, it takes a stratified subsample 
    of up to `n_max_silhouette` points per cluster.
    """
    best_k = -1
    best_score = -1.0
    best_model = None
    best_labels = None
    
    print(f"Starting DTW KMeans search for k in range {k_min}-{k_max}...")
    
    for k in range(k_min, k_max + 1):
        print(f"\n--- Fitting k={k} ---")
        model = TimeSeriesKMeans(
            n_clusters=k, 
            metric="dtw", 
            max_iter=10,        # Kept low for MVP
            random_state=42, 
            n_jobs=-1           # Parallelize DTW
        )
        labels = model.fit_predict(ts_dataset)
        
        # Subsample for silhouette score
        sampled_indices = []
        for cluster_id in range(k):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > n_max_silhouette:
                sampled_idx = np.random.choice(cluster_indices, n_max_silhouette, replace=False)
                sampled_indices.extend(sampled_idx)
            else:
                sampled_indices.extend(cluster_indices)
                
        sampled_dataset = ts_dataset[sampled_indices]
        sampled_labels = labels[sampled_indices]
        
        # Compute DTW Silhouette
        print(f"Computing silhouette score on subset of {len(sampled_indices)} trajectories...")
        score = silhouette_score(sampled_dataset, sampled_labels, metric="dtw", n_jobs=-1)
        print(f"k={k} | Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels
            
    print(f"\n=> Best k selected: {best_k} (Score: {best_score:.4f})")
    return best_k, best_model, best_labels

def save_clusters_to_db(patient_ids: List[str], labels: np.ndarray, db_path: str, table_name: str):
    """
    Saves the final cluster assignments to a specified SQLite database and table.
    """
    print(f"Saving cluster assignments to {db_path} -> table '{table_name}'...")
    df_out = pd.DataFrame({
        'patient_id': patient_ids,
        'cluster_id': labels
    })
    
    conn = sqlite3.connect(db_path)
    df_out.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print("Database save complete.")

# -------------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------------

def plot_dtw_barycenters(model: TimeSeriesKMeans, output_path: str = None):
    """
    Plots the mathematical DTW cluster centers (barycenters) for the chosen model.
    Subplots for each cluster. Weight, Fat, and Muscle overlap with distinct colors.
    """
    k = model.n_clusters
    centers = model.cluster_centers_  # shape: (k, max_timepoints, 3)
    
    fig, axes = plt.subplots(k, 1, figsize=(10, 3 * k), sharex=True, sharey=True)
    if k == 1:
        axes = [axes]
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Weight % Change', 'Fat % Change', 'Muscle % Change']
    
    for i in range(k):
        ax = axes[i]
        center = centers[i]
        
        # Plot each dimension
        for dim in range(3):
            # center[:, dim] could have nans at the end due to padding in tslearn, 
            # we plot only the valid (non-nan) points for the barycenter.
            valid_idx = ~np.isnan(center[:, dim])
            ax.plot(center[valid_idx, dim], label=labels[dim], color=colors[dim], linewidth=2.5)
            
        ax.set_title(f"Cluster {i} - DTW Barycenter")
        ax.set_ylabel("% Change")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
            
    axes[-1].set_xlabel("DTW Warped Time Steps")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved DTW Barycenters plot to {output_path}")
        
    plt.show()

def plot_lowess_trajectories(df: pd.DataFrame, patient_ids: List[str], labels: np.ndarray, output_path: str = None):
    """
    Plots raw trajectories with a LOWESS-smoothed mean curve.
    X-axis is actual `days_from_baseline`.
    """
    k = len(np.unique(labels))
    
    # Map cluster IDs back to the dataframe
    cluster_map = dict(zip(patient_ids, labels))
    df['cluster_id'] = df['patient_id'].map(cluster_map)
    df = df.dropna(subset=['cluster_id']).copy()
    
    fig, axes = plt.subplots(k, 1, figsize=(10, 4 * k), sharex=True, sharey=True)
    if k == 1:
        axes = [axes]
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_labels = ['weight_kg', 'fat_%', 'muscle_%']
    display_labels = ['Weight', 'Fat', 'Muscle']
    
    for i in range(k):
        ax = axes[i]
        cluster_df = df[df['cluster_id'] == i]
        
        for dim_idx, dim_col in enumerate(dim_labels):
            color = colors[dim_idx]
            
            # Sort by days for LOWESS
            sorted_cluster = cluster_df.sort_values(by='days_from_baseline')
            x = sorted_cluster['days_from_baseline'].values
            y = sorted_cluster[dim_col].values
            
            # 1. Plot raw points faintly
            ax.scatter(x, y, color=color, alpha=0.05, s=10)
            
            # 2. Plot LOWESS smoothed line
            if len(x) > 10:
                lowess = sm.nonparametric.lowess(y, x, frac=0.3)
                ax.plot(lowess[:, 0], lowess[:, 1], color=color, linewidth=3, label=display_labels[dim_idx])
                
        ax.set_title(f"Cluster {i} - LOWESS Smoothed Mean (n={cluster_df['patient_id'].nunique()} patients)")
        ax.set_ylabel("% Change from Baseline")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
            
    axes[-1].set_xlabel("Days from Baseline")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved LOWESS plot to {output_path}")
        
    plt.show()
