import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def plot_elbows(df_res: pd.DataFrame):
    """
    Plots internal validity metrics across different k values for each model-block combo.
    """
    df = df_res.copy()
    # Filter out HDBSCAN since it doesn't strictly scale on k
    df = df[df['Model'] != 'HDBSCAN']
    if len(df) == 0:
        return
        
    df['k'] = pd.to_numeric(df['Param (k/size)'], errors='coerce')
    df = df.dropna(subset=['k'])
    
    metrics = ['Silhouette', 'Davies-Bouldin', 'BIC']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(data=df, x='k', y=metric, hue='Model', style='Block', markers=True, ax=ax)
        ax.set_title(f"{metric} over k")
        ax.set_xticks(sorted(df['k'].unique()))
        
    plt.tight_layout()
    plt.show()

def plot_dendrogram(model, **kwargs):
    """Plots a dendrogram for Scikit-Learn AgglomerativeClustering models."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('Number of points in node (or index of point if no parenthesis).')
    plt.ylabel('Distance')
    plt.show()

def plot_pca(X_scaled: pd.DataFrame, labels: np.ndarray, title="PCA"):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
    counts = pd.Series(labels).value_counts()
    df_pca['Cluster'] = [f"{lbl} (n={counts[lbl]})" for lbl in labels]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=df_pca, s=50, alpha=0.7)
    plt.title(f"{title} - Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
    plt.show()
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=X_scaled.columns
    )
    print("\\n--- PCA Loadings (Feature Contributions to Principal Components) ---")
    from IPython.display import display
    display(loadings)

def plot_tsne(X_scaled: pd.DataFrame, labels: np.ndarray, title="t-SNE"):
    tsne = TSNE(n_components=2, random_state=42)
    components = tsne.fit_transform(X_scaled)
    
    df_tsne = pd.DataFrame(data=components, columns=['t-SNE1', 't-SNE2'])
    counts = pd.Series(labels).value_counts()
    df_tsne['Cluster'] = [f"{lbl} (n={counts[lbl]})" for lbl in labels]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Cluster', palette='tab10', data=df_tsne, s=50, alpha=0.7)
    plt.title(title)
    plt.show()

def plot_umap(X_scaled: pd.DataFrame, labels: np.ndarray, title="UMAP"):
    if not HAS_UMAP:
        print("UMAP is not installed (pip install umap-learn). Skipping UMAP visualization.")
        return
        
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    df_umap = pd.DataFrame(data=embedding, columns=['UMAP1', 'UMAP2'])
    counts = pd.Series(labels).value_counts()
    df_umap['Cluster'] = [f"{lbl} (n={counts[lbl]})" for lbl in labels]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', palette='tab10', data=df_umap, s=50, alpha=0.7)
    plt.title(title)
    plt.show()
    
def plot_clinical_panels(df_features: pd.DataFrame, labels: np.ndarray):
    """
    Plots a 4-panel composite plot detailing the clinical interpretability of clusters.
    Axes capped at 5th/95th percentiles (except where minimum is mathematically 0, or LLC is 0-1).
    """
    df_plot = df_features.copy()
    counts = pd.Series(labels).value_counts()
    df_plot['Cluster'] = [f"{lbl} (n={counts[lbl]})" for lbl in labels]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    def set_capped_limits(ax, data, x_feat, y_feat):
        def get_lims(series, feat):
            if feat == 'lean_loss_coeff':
                return (0.0, 1.0)
            
            vmin = series.min()
            vmax = series.max()
            p05 = series.quantile(0.05)
            p95 = series.quantile(0.95)
            
            low = vmin if np.isclose(vmin, 0) else p05
            high = vmax if np.isclose(vmax, 0) else p95
            
            margin = (high - low) * 0.05
            margin = margin if margin > 0 else 0.1
            return (low - margin, high + margin)
            
        if x_feat in data.columns:
            ax.set_xlim(*get_lims(data[x_feat], x_feat))
        if y_feat in data.columns:
            ax.set_ylim(*get_lims(data[y_feat], y_feat))
            
    # 1. Efficacy & Maintenance (The Yo-Yo Axis)
    if 'pct_max_loss' in df_plot.columns and 'pct_regain' in df_plot.columns:
        sns.scatterplot(x='pct_max_loss', y='pct_regain', hue='Cluster', palette='tab10', data=df_plot, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title("1. Efficacy & Maintenance (The Yo-Yo Axis)")
        axes[0,0].set_xlabel("Peak Efficacy (% Max Loss, negative is loss)")
        axes[0,0].set_ylabel("Long-term Maintenance (% Regain)")
        set_capped_limits(axes[0,0], df_plot, 'pct_max_loss', 'pct_regain')
    
    # 2. Body Composition Quality (The Sarcopenic Axis)
    if 'pct_max_loss' in df_plot.columns and 'lean_loss_coeff' in df_plot.columns:
        sns.scatterplot(x='pct_max_loss', y='lean_loss_coeff', hue='Cluster', palette='tab10', data=df_plot, ax=axes[0,1], alpha=0.7)
        axes[0,1].set_title("2. Body Composition Quality (The Sarcopenic Axis)")
        axes[0,1].set_xlabel("Peak Efficacy (% Max Loss)")
        axes[0,1].set_ylabel("Muscle Wasting (Lean Loss Coefficient)")
        set_capped_limits(axes[0,1], df_plot, 'pct_max_loss', 'lean_loss_coeff')
        
    # 3. Early Predictor (The Kinetics Axis)
    if 'early_wl_speed' in df_plot.columns and 'pct_max_loss' in df_plot.columns:
        sns.scatterplot(x='early_wl_speed', y='pct_max_loss', hue='Cluster', palette='tab10', data=df_plot, ax=axes[1,0], alpha=0.7)
        axes[1,0].set_title("3. Early Predictor (The Kinetics Axis)")
        axes[1,0].set_xlabel("Initial Velocity (Early WL Speed %/day)")
        axes[1,0].set_ylabel("Ultimate Success (% Max Loss)")
        set_capped_limits(axes[1,0], df_plot, 'early_wl_speed', 'pct_max_loss')
        
    # 4. Behavioral Engagement (The Digital Phenotype Axis)
    if 'observation_duration' in df_plot.columns:
        y_feat = 'longest_gap' if 'longest_gap' in df_plot.columns else 'trajectory_volatility'
        if y_feat in df_plot.columns:
            sns.scatterplot(x='observation_duration', y=y_feat, hue='Cluster', palette='tab10', data=df_plot, ax=axes[1,1], alpha=0.7)
            axes[1,1].set_title(f"4. Behavioral Engagement (The Digital Phenotype)")
            axes[1,1].set_xlabel("Longevity (Observation Duration in days)")
            axes[1,1].set_ylabel(f"Consistency ({y_feat})")
            set_capped_limits(axes[1,1], df_plot, 'observation_duration', y_feat)
        
    plt.tight_layout()
    plt.show()

def plot_random_trajectories(db_path: str, df_clusters: pd.DataFrame, id_col: str = 'patient_id', k: int = 15):
    """
    Plots k random weight-loss trajectories per cluster to visually validate intra-cluster homogeneity.
    """
    conn = sqlite3.connect(db_path)
    
    sampled_ids = []
    for cluster_id in df_clusters['cluster_id'].unique():
        cluster_df = df_clusters[df_clusters['cluster_id'] == cluster_id]
        sample = cluster_df.sample(min(k, len(cluster_df)), random_state=np.random.randint(10000))
        sampled_ids.extend(sample[id_col].tolist())
        
    placeholders = ','.join('?' for _ in sampled_ids)
    query = f"""
    SELECT {id_col}, measurement_date, weight_kg 
    FROM measurements_filtered 
    WHERE {id_col} IN ({placeholders})
    """
    df_meas = pd.read_sql_query(query, conn, params=sampled_ids)
    conn.close()
    
    df_meas['measurement_date'] = pd.to_datetime(df_meas['measurement_date'])
    df_meas['calendar_date'] = df_meas['measurement_date'].dt.normalize()
    
    unique_clusters = sorted(df_clusters['cluster_id'].unique())
    n_clusters = len(unique_clusters)
    
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
    if n_clusters == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for i, c_id in enumerate(unique_clusters):
        ax = axes[i]
        c_ids = df_clusters[df_clusters['cluster_id'] == c_id][id_col].values
        
        for p_id in c_ids:
            if p_id not in sampled_ids:
                continue
                
            p_data = df_meas[df_meas[id_col] == p_id].copy()
            if len(p_data) == 0:
                continue
                
            p_data = p_data.sort_values('calendar_date')
            t0 = p_data['calendar_date'].iloc[0]
            w0 = p_data['weight_kg'].iloc[0]
            
            p_data['days'] = (p_data['calendar_date'] - t0).dt.days
            p_data['pct_loss'] = ((p_data['weight_kg'] - w0) / w0) * 100
            
            ax.plot(p_data['days'], p_data['pct_loss'], alpha=0.6, linewidth=1.5)
            
        ax.set_title(f"Cluster {c_id}")
        ax.set_xlabel("Days from baseline")
        ax.set_ylabel("% Weight Change")
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.show()
