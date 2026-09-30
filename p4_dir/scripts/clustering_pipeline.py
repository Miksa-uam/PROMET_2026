import pandas as pd
import numpy as np
from clustering_models import prepare_data, fit_model
from clustering_eval import run_clustering_grid_search, evaluate_predictive_utility, evaluate_clusters, print_cluster_summary
from clustering_viz import (
    plot_pca, plot_umap, plot_tsne, 
    plot_random_trajectories, plot_random_trajectories_body_comp, plot_clinical_panels, 
    plot_elbows, plot_dendrogram
)

def run_clustering_pipeline(df_features: pd.DataFrame, db_path: str, blocks: list, models: list, k_range: list, id_col: str = 'patient_id'):
    """
    Unified entrypoint for clustering.
    If multiple blocks, models, or k's are provided, it runs a full grid search, plots elbows,
    selects the top model, and performs a deep visual characterization on it.
    If only one block, model, and k are provided, it simply characterizes that setup directly.
    """
    if isinstance(k_range, int):
        k_range = [k_range]
        
    is_grid = len(blocks) > 1 or len(models) > 1 or len(k_range) > 1
    
    if is_grid:
        print("================================================================")
        print("                 COMPREHENSIVE GRID SEARCH                      ")
        print("================================================================")
        df_res = run_clustering_grid_search(df_features, models=models, blocks=blocks, k_range=k_range)
        
        if df_res is None or len(df_res) == 0:
            print("Grid search failed or returned no results.")
            return
            
        # Plot elbows if applicable (only if k_range > 1)
        if len(k_range) > 1:
            plot_elbows(df_res)
            
        # Select best model (Rank 1 by Silhouette)
        valid_res = df_res[(df_res['Clusters Found'] > 1) & (df_res['Clusters Found'] < 15)]
        if len(valid_res) == 0:
            print("No valid clustering setup found.")
            return
            
        best_setup = valid_res.sort_values(by='Silhouette', ascending=False).iloc[0]
        
        best_block = best_setup['Block']
        best_model = best_setup['Model']
        best_param_str = str(best_setup['Param (k/size)'])
        
        print("\n================================================================")
        print(f"      DEEP CHARACTERIZATION: {best_model} on {best_block} ({best_param_str})")
        print("================================================================")
        
        # Parse params back
        kwargs = {}
        k = 4 # default fallback
        if best_model in ['HDBSCAN', 'UMAP_HDBSCAN']:
            # Param string looks like "size:20_samp:4"
            parts = best_param_str.split('_')
            kwargs['min_cluster_size'] = int(parts[0].split(':')[1])
            kwargs['min_samples'] = int(parts[1].split(':')[1])
        else:
            k = int(best_param_str)
            
        block = best_block
        model_type = best_model
        
    else:
        print("================================================================")
        print(f"               SINGLE RUN: {models[0]} on {blocks[0]}           ")
        print("================================================================")
        block = blocks[0]
        model_type = models[0]
        k = k_range[0]
        kwargs = {}
        if model_type in ['HDBSCAN', 'UMAP_HDBSCAN']:
            kwargs['min_cluster_size'] = k * 100
            kwargs['min_samples'] = max(15, int(k * 10))
            
    print(f"\n--- 1. PREPARING & FITTING ---")
    X_scaled, df_clean = prepare_data(df_features, block=block)
    
    print("\n--- Scaled Feature Variances (Sanity Check) ---")
    print(X_scaled.var().to_string())
    
    labels, model = fit_model(X_scaled, model_type=model_type, k=k, **kwargs)
    
    df_clusters = df_clean.copy()
    df_clusters['cluster_id'] = labels
    
    metrics = evaluate_clusters(X_scaled, labels, model=model, model_type=model_type)
    print(f"\nInternal Validity Metrics:")
    for k_metric, v_metric in metrics.items():
        print(f"  - {k_metric}: {v_metric:.4f}" if pd.notna(v_metric) else f"  - {k_metric}: N/A")
        
    # Dendrogram for WARD
    if model_type == 'WARD':
        print("\n--- Plotting Dendrogram for WARD clustering ---")
        plot_dendrogram(model, truncate_mode='level', p=4)
        
    print("\n--- 2. DIMENSIONALITY REDUCTION ---")
    plot_pca(X_scaled, labels, title=f"PCA - {model_type} ({block})")
    plot_tsne(X_scaled, labels, title=f"t-SNE - {model_type} ({block})")
    plot_umap(X_scaled, labels, title=f"UMAP - {model_type} ({block})")
    
    print("\n--- 3. CLINICAL CHARACTERIZATION (5th-95th Percentile Capped) ---")
    plot_clinical_panels(df_clean, labels)
    
    print("\n--- 4. TRAJECTORY HOMOGENEITY SAMPLES ---")
    plot_random_trajectories(db_path=db_path, df_clusters=df_clusters, id_col=id_col, k=15)
    plot_random_trajectories_body_comp(db_path=db_path, df_clusters=df_clusters, id_col=id_col, k=5)
    
    print("\n--- 5. PREDICTIVE UTILITY ---")
    evaluate_predictive_utility(db_path, df_clusters, id_col=id_col)
    
    print_cluster_summary(db_path, df_clusters, id_col=id_col)
