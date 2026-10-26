import sqlite3
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import Tuple, List, Optional

# -------------------------------------------------------------------------
# DATA PREPARATION
# -------------------------------------------------------------------------

def load_and_filter_measurements(
    db_path: str, 
    id_col: str = 'patient_id', 
    target_cols: Optional[List[str]] = None,
    max_weighins: int = 50
) -> pd.DataFrame:
    """
    Loads measurement data from the SQLite database and applies initial quality filters.
    
    Inputs:
    - db_path (str): Path to the SQLite database.
    - id_col (str): The column used to identify the entity of analysis (e.g., 'patient_id' or 'medical_record_id').
    - target_cols (List[str]): The variables to cluster on (e.g., ['weight_kg', 'fat_%', 'muscle_%']).
    - max_weighins (int): Cutoff for the maximum number of weigh-ins to include, filtering out outliers.
    
    Outputs:
    - pd.DataFrame: Cleaned and filtered dataframe ready for processing.
    
    Methodological logic:
    - Extract only required columns (ID, date, and targets).
    - DTW requires complete multivariate points at each time step, so we drop rows 
      with missing values in any of our core target variables.
    - Time series analysis requires sequences, so we enforce >= 2 timepoints per ID.
    - Exclude extreme outliers with more weigh-ins than `max_weighins` to save DTW computation time.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    print(f"Loading data from {db_path}...")
    
    # Ensure weight_kg is queried if we have % columns to convert later
    query_cols = target_cols.copy()
    if any('%' in col for col in target_cols) and 'weight_kg' not in target_cols:
        query_cols.append('weight_kg')
        
    # Properly quote target columns in SQL if they contain special characters like '%'
    sql_target_cols = [f'"{col}"' if '%' in col or ' ' in col else col for col in query_cols]
    cols_to_select = [id_col, 'measurement_date'] + sql_target_cols
    
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT {', '.join(cols_to_select)}
        FROM measurements_filtered
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Initial data: {len(df)} rows, {df[id_col].nunique()} unique {id_col}s.")
    
    # Note: we drop missing values based on query_cols to ensure we have weight for the conversion
    df = df.dropna(subset=query_cols)
    print(f"After dropping missing values in targets: {len(df)} rows.")
    
    # --- BIOLOGICAL PLAUSIBILITY FILTER (Raw Data) ---
    # Filter out impossible raw values before any math is done to catch typos.
    
    # 1. Weight limits (e.g., no adults under 30kg, no one over 350kg)
    if 'weight_kg' in df.columns:
        df = df[df['weight_kg'].between(30.0, 350.0)]
    # 2. Percentage limits (e.g., body fat cannot be <3% or >80%)
    if "fat_%" in target_cols:
        df = df[df["fat_%"].between(3.0, 80.0)]
    # 3. Muscle limits (e.g., muscle cannot be <10% or >90% of total body weight)
    if "muscle_%" in target_cols:
        df = df[df["muscle_%"].between(10.0, 90.0)]
        
    print(f"After dropping biological raw outliers: {len(df)} rows.")
    
    counts = df.groupby(id_col).size()
    valid_ids = counts[counts >= 2].index
    df = df[df[id_col].isin(valid_ids)]
    
    print(f"After enforcing >=2 timepoints: {len(df)} rows, {df[id_col].nunique()} unique {id_col}s remaining.")
    
    # Calculate and print stats on weigh-ins
    weighins_per_id = df.groupby(id_col).size()
    print(f"95th percentile of weigh-ins per {id_col}: {np.percentile(weighins_per_id, 95):.1f}")
    print(f"Max weigh-ins per {id_col}: {weighins_per_id.max()}")
    
    # Remove extreme outliers to save DTW computation time
    counts = df.groupby(id_col).size()
    valid_ids_cap = counts[(counts >= 2) & (counts <= max_weighins)].index
    df = df[df[id_col].isin(valid_ids_cap)]
    
    print(f"After capping at {max_weighins} timepoints: {len(df)} rows, {df[id_col].nunique()} unique {id_col}s.")
    
    return df

def convert_percentages_to_absolute(
    df: pd.DataFrame, 
    target_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts relative body composition values (percentages) to absolute values (kg)
    based on the corresponding weight_kg value.
    Updates the target_cols list to reflect the new absolute column names.
    
    Methodological logic:
    - Percentage variables (like muscle_%) can artificially appear to grow if total body weight 
      drops significantly, even if absolute muscle mass is lost. 
    - Converting to absolute kilograms (kg) before calculating percentage change from baseline 
      provides a more biologically accurate reflection of body composition changes.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    print("Converting percentage variables to absolute kilograms...")
    new_target_cols = []
    
    # We must have weight_kg to do the conversion
    if 'weight_kg' not in df.columns and any('%' in col for col in target_cols):
        raise ValueError("Cannot convert percentages to kg without 'weight_kg' in the dataframe.")
        
    for col in target_cols:
        if '%' in col:
            new_col = col.replace('%', 'kg')
            # Calculate absolute value (e.g., (fat_% / 100) * weight_kg)
            df[new_col] = (df[col] / 100) * df['weight_kg']
            new_target_cols.append(new_col)
        else:
            new_target_cols.append(col)
            
    print(f"Updated target columns for clustering: {new_target_cols}")
    return df, new_target_cols

def calculate_days_from_baseline(df: pd.DataFrame, id_col: str = 'patient_id') -> pd.DataFrame:
    """
    Calculates the elapsed time in days from the first measurement for each entity (id_col).
    This provides a continuous time axis for LOWESS smoothing plots.
    """
    print(f"Calculating days from baseline for time axis using {id_col}...")
    df['measurement_date'] = pd.to_datetime(df['measurement_date'])
    df = df.sort_values(by=[id_col, 'measurement_date'])
    
    baselines_time = df.groupby(id_col)['measurement_date'].transform('min')
    df['days_from_baseline'] = (df['measurement_date'] - baselines_time).dt.days
    
    return df

def calculate_percentage_from_baseline(
    df: pd.DataFrame, 
    id_col: str = 'patient_id', 
    target_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Sorts the data chronologically and converts absolute values into percentage change 
    from baseline for each entity (id_col) across target columns.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    print("Calculating percentage change from baseline...")
    df = df.sort_values(by=[id_col, 'measurement_date']).copy()
    
    baselines = df.groupby(id_col)[target_cols].transform('first')
    
    # Replace zeros with NaN to avoid division by zero if necessary, though biological data baseline rarely zeroes
    df[target_cols] = ((df[target_cols] - baselines) / baselines) * 100
    
    return df

def prepare_tslearn_dataset(
    df: pd.DataFrame, 
    id_col: str = 'patient_id', 
    target_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Pivots the longitudinal dataframe into the 3D numpy array format required by tslearn.
    
    Outputs:
    - ts_dataset (np.ndarray): 3D array of shape (n_entities, max_timepoints, n_targets)
    - entity_ids (List[str]): List of IDs preserving order matching the 3D array.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    print("Reshaping data for tslearn...")
    df = df.sort_values(by=[id_col, 'measurement_date'])
    
    sequences = []
    entity_ids = []
    
    for entity_id, group in df.groupby(id_col):
        entity_ids.append(entity_id)
        seq = group[target_cols].values
        sequences.append(seq)
        
    ts_dataset = to_time_series_dataset(sequences)
    print(f"Dataset reshaped successfully. Shape: {ts_dataset.shape}")
    
    return ts_dataset, entity_ids

# -------------------------------------------------------------------------
# CLUSTERING MVP
# -------------------------------------------------------------------------

def run_dtw_clustering_search(
    ts_dataset: np.ndarray, 
    k_min: int = 2, 
    k_max: int = 7, 
    elbow_plot_path: Optional[str] = None
) -> Tuple[int, TimeSeriesKMeans, np.ndarray]:
    """
    Runs DTW TimeSeriesKMeans for k in [k_min, k_max] and evaluates using Inertia.
    Automatically detects the 'elbow' point using the maximum distance to the line 
    connecting the first and last points of the curve, and plots the result.
    
    Inputs:
    - ts_dataset (np.ndarray): The 3D tensor of time series data.
    - k_min (int): Minimum clusters to test.
    - k_max (int): Maximum clusters to test.
    - elbow_plot_path (str): Optional path to save the elbow plot.
    
    Outputs:
    - best_k (int): The optimal number of clusters determined by the elbow method.
    - best_model (TimeSeriesKMeans): The fitted model for best_k.
    - best_labels (np.ndarray): The cluster assignments for best_k.
    """
    models = {}
    inertias = []
    k_values = list(range(k_min, k_max + 1))
    
    print(f"Starting DTW KMeans search for k in range {k_min}-{k_max}...")
    
    for k in k_values:
        print(f"\n--- Fitting k={k} ---")
        model = TimeSeriesKMeans(
            n_clusters=k, 
            metric="dtw", 
            max_iter=10,        # Kept low for MVP
            random_state=42, 
            n_jobs=-1           # Parallelize DTW
        )
        labels = model.fit_predict(ts_dataset)
        
        # We extract inertia directly from the model object (zero extra computational cost!)
        inertia = model.inertia_
        inertias.append(inertia)
        models[k] = (model, labels)
        
        print(f"k={k} | Inertia: {inertia:.2f}")
        
    # Automatic Elbow Detection (Kneedle algorithm heuristic)
    # We find the point on the inertia curve furthest from the straight line 
    # connecting the first and last points (k_min and k_max).
    x1, y1 = k_values[0], inertias[0]
    x2, y2 = k_values[-1], inertias[-1]
    
    # Line equation: ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    
    distances = []
    for x0, y0 in zip(k_values, inertias):
        # Distance formula from point (x0, y0) to line ax + by + c = 0
        dist = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
        distances.append(dist)
        
    best_idx = np.argmax(distances)
    best_k = k_values[best_idx]
    best_model, best_labels = models[best_k]
    
    print(f"\n=> Best k selected via Elbow Method: {best_k} (Inertia: {inertias[best_idx]:.2f})")
    
    # Plotting the Elbow Curve
    print("Generating Elbow Curve plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='b', label='Inertia (WCSS)')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Selected Elbow (k={best_k})')
    plt.title("Elbow Method for Optimal k (DTW Inertia)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared DTW Distances)")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if elbow_plot_path:
        plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved Elbow plot to {elbow_plot_path}")
        
    plt.show()
    
    return best_k, best_model, best_labels
    
def save_clusters_to_db(
    ids: List[str], 
    labels: np.ndarray, 
    db_path: str, 
    table_name: str,
    id_col: str = 'patient_id'
):
    """
    Saves the final cluster assignments to a specified SQLite database and table.
    """
    print(f"Saving cluster assignments to {db_path} -> table '{table_name}'...")
    df_out = pd.DataFrame({
        id_col: ids,
        'cluster_id': labels
    })
    
    conn = sqlite3.connect(db_path)
    df_out.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print("Database save complete.")

# -------------------------------------------------------------------------
# VISUALIZATIONS
# -------------------------------------------------------------------------

def plot_dtw_barycenters(
    model: TimeSeriesKMeans, 
    target_cols: Optional[List[str]] = None,
    output_path: str = None
):
    """
    Plots the mathematical DTW cluster centers (barycenters) for the chosen model.
    Subplots for each cluster. Overlaps configured target variables with distinct colors.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    k = model.n_clusters
    centers = model.cluster_centers_  # shape: (k, max_timepoints, n_targets)
    
    fig, axes = plt.subplots(k, 1, figsize=(10, 3 * k), sharex=True, sharey=True)
    if k == 1:
        axes = [axes]
        
    # Standard matplotlib tab10 colors to handle variable number of targets
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(target_cols))]
    labels = [f"{col} % Change" for col in target_cols]
    
    for i in range(k):
        ax = axes[i]
        center = centers[i]
        
        # Plot each dimension
        for dim in range(len(target_cols)):
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

def plot_lowess_trajectories(
    df: pd.DataFrame, 
    ids: List[str], 
    labels: np.ndarray, 
    id_col: str = 'patient_id',
    target_cols: Optional[List[str]] = None,
    output_path: str = None
):
    """
    Plots raw trajectories with a LOWESS-smoothed mean curve.
    X-axis is actual `days_from_baseline`.
    """
    if target_cols is None:
        target_cols = ['weight_kg', 'fat_%', 'muscle_%']
        
    k = len(np.unique(labels))
    
    # Map cluster IDs back to the dataframe
    cluster_map = dict(zip(ids, labels))
    df['cluster_id'] = df[id_col].map(cluster_map)
    df = df.dropna(subset=['cluster_id']).copy()
    
    fig, axes = plt.subplots(k, 1, figsize=(10, 4 * k), sharex=True, sharey=True)
    if k == 1:
        axes = [axes]
        
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(target_cols))]
    
    for i in range(k):
        ax = axes[i]
        cluster_df = df[df['cluster_id'] == i]
        
        for dim_idx, dim_col in enumerate(target_cols):
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
                ax.plot(lowess[:, 0], lowess[:, 1], color=color, linewidth=3, label=dim_col)
                
        ax.set_title(f"Cluster {i} - LOWESS Smoothed Mean (n={cluster_df[id_col].nunique()} {id_col}s)")
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
