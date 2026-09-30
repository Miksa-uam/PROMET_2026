import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans
import warnings

# Attempt to import HDBSCAN
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    try:
        import hdbscan
        HAS_HDBSCAN = True
    except ImportError:
        HAS_HDBSCAN = False

FEATURE_BLOCKS = {
    'Kinetic': [
        'pct_max_loss', 
        'pct_regain', 
        # 'adj_pct_max_loss', 
        # 'adj_pct_regain', 
        'early_wl_speed', 
        'time_to_nadir_ratio',
        'trajectory_volatility'
        ],
    'Metabolic': [
        'lean_loss_coeff', 
        'muscle_fat_corr'
        ],
    'Physio': [
        'pct_max_loss', 
        'pct_regain', 
        # 'adj_pct_max_loss', 
        # 'adj_pct_regain', 
        'early_wl_speed', 
        'time_to_nadir_ratio', 
        'trajectory_volatility',
        'lean_loss_coeff', 
        'muscle_fat_corr'
    ],
    'Behavioral': [
        'observation_duration', 
        'observation_gap_ratio', 
        ],
    'Holistic raw': [
        'pct_max_loss', 
        'pct_regain', 
        # 'adj_pct_max_loss', 
        # 'adj_pct_regain', 
        'early_wl_speed', 
        'time_to_nadir_ratio', 
        'trajectory_volatility',
        'lean_loss_coeff', 
        'muscle_fat_corr',
        'observation_duration', 
        'observation_gap_ratio', 
    ],

    'Holistic adjusted': [
        # 'pct_max_loss', 
        # 'pct_regain', 
        'adj_pct_max_loss', 
        'adj_pct_regain', 
        'early_wl_speed', 
        'time_to_nadir_ratio', 
        'trajectory_volatility',
        'lean_loss_coeff', 
        'muscle_fat_corr',
        'observation_duration', 
        'observation_gap_ratio', 
    ]
}   

def prepare_data(df_features: pd.DataFrame, block: str = 'Holistic') -> pd.DataFrame:
    """
    Selects features based on the chosen block and scales them using RobustScaler.
    """
    if block not in FEATURE_BLOCKS:
        raise ValueError(f"Block must be one of {list(FEATURE_BLOCKS.keys())}")
        
    features = FEATURE_BLOCKS[block]
    
    # Drop rows with NaNs in the selected features
    df_clean = df_features.dropna(subset=features).copy()
    
    X_raw = df_clean[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    df_scaled = pd.DataFrame(X_scaled, columns=features, index=df_clean.index)
    return df_scaled, df_clean

def fit_model(X: pd.DataFrame, model_type: str = 'GMM', k: int = 4, **kwargs):
    """
    Fits a clustering model and returns the labels and the fitted model object.
    """
    model_type = model_type.upper().replace('-', '_')
    
    if model_type == 'GMM':
        model = GaussianMixture(n_components=k, covariance_type='diag', random_state=42, **kwargs)
        labels = model.fit_predict(X)
        
    elif model_type == 'HDBSCAN':
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN is not installed. Please install scikit-learn>=1.3.0 or the hdbscan package.")
        try:
            from sklearn.cluster import HDBSCAN
            model = HDBSCAN(
                min_cluster_size=kwargs.get('min_cluster_size', 50),
                min_samples=kwargs.get('min_samples', None),
                metric='manhattan',
                copy=True
            )
        except ImportError:
            import hdbscan
            model = hdbscan.HDBSCAN(
                min_cluster_size=kwargs.get('min_cluster_size', 50),
                min_samples=kwargs.get('min_samples', None),
                metric='manhattan'
            )
        labels = model.fit_predict(X)
        
    elif model_type == 'UMAP_HDBSCAN':
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN is not installed.")
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is not installed.")
        from sklearn.cluster import HDBSCAN
        
        # Project 10D/7D -> 5D (preserve global structure with high n_neighbors)
        reducer = umap.UMAP(n_components=5, n_neighbors=50, min_dist=0.0, random_state=42)
        X_umap = reducer.fit_transform(X.values if isinstance(X, pd.DataFrame) else X)
        
        model = HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 50),
            min_samples=kwargs.get('min_samples', None),
            metric='manhattan',
            copy=True
        )
        labels = model.fit_predict(X_umap)
        
    elif model_type == 'WARD':
        model = AgglomerativeClustering(n_clusters=k, linkage='ward', compute_distances=True, **kwargs)
        labels = model.fit_predict(X)
        
    elif model_type == 'KMEANS':
        model = KMeans(n_clusters=k, random_state=42, **kwargs)
        labels = model.fit_predict(X)
        
    elif model_type == 'SPECTRAL':
        from sklearn.cluster import SpectralClustering
        # Use nearest_neighbors affinity for manifold learning, similar to UMAP
        model = SpectralClustering(
            n_clusters=k, 
            affinity='nearest_neighbors', 
            random_state=42, 
            **kwargs
        )
        labels = model.fit_predict(X)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return labels, model
