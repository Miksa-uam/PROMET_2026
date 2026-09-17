# Design Document: Cluster Trajectories Module

## Overview

The `cluster_trajectories.py` module will consolidate cluster-based trajectory analysis functionality into a clean, reusable interface. It will integrate with the existing project structure, using the `cluster_config.json` for configuration and following patterns established in `cluster_descriptions.py` and `wgc_visualizations.py`.

## Architecture

### Module Structure

```
scripts/
├── cluster_trajectories.py      # Main module
├── cluster_config.json           # Configuration file (extended)
└── human_readable_variable_names.json  # Variable name mapping
```

### Key Components

1. **Data Loading Layer**: Functions to load and merge cluster labels with measurement data
2. **Trajectory Calculation Layer**: Functions to compute days from baseline and aggregate statistics
3. **Visualization Layer**: Functions to create spaghetti plots with various options
4. **Statistical Modeling Layer**: Functions to fit and visualize LMM results
5. **Configuration Layer**: Functions to load and validate configuration

## Components and Interfaces

### 1. Configuration Management

```python
class TrajectoryConfig:
    """Configuration for trajectory analysis"""
    
    # Database paths
    cluster_db_path: str
    measurements_db_path: str
    
    # Table names
    cluster_table: str
    measurements_table: str
    cluster_column: str
    
    # Column names
    patient_id_col: str = "patient_id"
    medical_record_id_col: str = "medical_record_id"
    measurement_date_col: str = "measurement_date"
    body_weight_col: str = "weight_kg"
    
    # Analysis parameters
    cutoff_days: Optional[int] = None
    smoothing_frac: float = 0.3
    
    # Visualization parameters
    colors: List[str] = DEFAULT_COLORS
    figure_size: Tuple[int, int] = (20, 12)
    dpi: int = 300
    
    # Output
    output_dir: str = "../outputs/cluster_trajectories"

def load_trajectory_config(config_path: str = "scripts/cluster_config.json") -> TrajectoryConfig:
    """Load trajectory configuration from JSON file"""
    pass
```

### 2. Data Loading Functions

```python
def load_cluster_labels(config: TrajectoryConfig) -> pd.DataFrame:
    """
    Load cluster labels from database.
    
    Returns:
        DataFrame with columns: patient_id, medical_record_id, cluster_id
    """
    pass

def load_measurements_for_patients(
    config: TrajectoryConfig,
    cluster_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Load measurement data for patients in cluster labels.
    
    Returns:
        DataFrame with columns: patient_id, medical_record_id, 
                               measurement_date, weight_kg
    """
    pass

def calculate_days_from_baseline(df: pd.DataFrame, config: TrajectoryConfig) -> pd.DataFrame:
    """
    Calculate days from baseline for each measurement.
    
    Returns:
        DataFrame with added 'days_from_baseline' column
    """
    pass

def merge_with_clusters(
    measurements: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    config: TrajectoryConfig
) -> pd.DataFrame:
    """
    Merge measurements with cluster labels.
    
    Returns:
        DataFrame with cluster_id column added
    """
    pass
```

### 3. Trajectory Visualization Functions

```python
def plot_whole_population(
    ax: plt.Axes,
    analysis_data: pd.DataFrame,
    config: TrajectoryConfig
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Plot whole population trajectories in a panel.
    
    Returns:
        (mean_followup, smoothed_trajectory, sem_array)
    """
    pass

def plot_single_cluster_trajectory(
    ax: plt.Axes,
    cluster_data: pd.DataFrame,
    cluster_id: int,
    config: TrajectoryConfig
) -> float:
    """
    Plot trajectories for a single cluster with length-adjusted smoothing.
    
    Returns:
        mean_followup for the cluster
    """
    pass

def create_spaghetti_plots_with_overlay(
    analysis_data: pd.DataFrame,
    config: TrajectoryConfig,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Create multi-panel spaghetti plot with overlay.
    
    Generates:
    - Panel 1: Whole population
    - Panels 2-N: Individual clusters
    - Panel N+1: Overlay of all smoothed means
    
    Returns:
        matplotlib Figure object
    """
    pass

def generate_spaghetti_plots(
    config: TrajectoryConfig,
    show_plots: bool = True,
    save_plots: bool = True
) -> Dict[str, plt.Figure]:
    """
    Main function to generate spaghetti plots.
    
    Generates two plots:
    1. Full timespan
    2. Cutoff timespan (if cutoff_days specified)
    
    Returns:
        Dictionary mapping plot names to Figure objects
    """
    pass
```

### 4. Statistical Modeling Functions

```python
class LMMConfig:
    """Configuration for Linear Mixed Model analysis"""
    use_deviation_coding: bool = True
    reference_cluster: int = 0
    adjust_for_covariates: bool = False
    show_both_adjusted_unadjusted: bool = False
    knot_quantiles: List[float] = [0.25, 0.5, 0.75]

def fit_lmm(
    data: pd.DataFrame,
    config: TrajectoryConfig,
    lmm_config: LMMConfig
) -> Any:  # statsmodels MixedLMResults
    """
    Fit linear mixed model to trajectory data.
    
    Model structure:
    - Fixed effects: cluster (with specified coding) * time_spline
    - Random effects: patient-level intercept and slope
    - Optional covariates: age, sex, baseline_bmi
    
    Returns:
        Fitted model results object
    """
    pass

def plot_predicted_trajectories(
    lmm_results: Any,
    data: pd.DataFrame,
    config: TrajectoryConfig,
    title_suffix: str = ""
) -> plt.Figure:
    """
    Plot model-predicted mean trajectories for each cluster.
    
    Returns:
        matplotlib Figure object
    """
    pass

def display_full_deviation_effects(
    lmm_results: Any,
    data: pd.DataFrame
) -> None:
    """
    Calculate and print effect for omitted cluster in deviation coding.
    """
    pass

def run_lmm_analysis(
    config: TrajectoryConfig,
    lmm_config: LMMConfig
) -> Dict[str, Any]:
    """
    Main function to run LMM analysis.
    
    Returns:
        Dictionary with keys:
        - 'model_results': fitted model object
        - 'summary': model summary
        - 'figure': predicted trajectories plot
    """
    pass
```

### 5. Utility Functions

```python
def create_fixed_time_cutoff_data(
    df: pd.DataFrame,
    cutoff_days: int
) -> pd.DataFrame:
    """Filter data to specified time cutoff"""
    pass

def get_cluster_colors(
    n_clusters: int,
    palette: Optional[List[str]] = None
) -> List[str]:
    """Get consistent colors for clusters"""
    pass

def validate_config(config: TrajectoryConfig) -> None:
    """Validate configuration parameters"""
    pass

def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist"""
    pass
```

## Data Models

### Input Data Structure

```python
# Cluster Labels DataFrame
cluster_labels = pd.DataFrame({
    'patient_id': int,
    'medical_record_id': int,
    'cluster_id': int  # -1 for outliers
})

# Measurements DataFrame
measurements = pd.DataFrame({
    'patient_id': int,
    'medical_record_id': int,
    'measurement_date': datetime,
    'weight_kg': float
})

# Analysis DataFrame (merged)
analysis_data = pd.DataFrame({
    'patient_id': int,
    'medical_record_id': int,
    'measurement_date': datetime,
    'weight_kg': float,
    'days_from_baseline': int,
    'cluster_id': int
})
```

### Configuration File Extension

Extend `cluster_config.json` to include trajectory-specific settings:

```json
{
  "cluster_algorithm": "pam",
  "n_clusters": 7,
  "cluster_db": "../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
  "measurements_db": "../dbs/pnk_db2_p2_in.sqlite",
  "cluster_table": "clust_labels_bl_nobc_bw_pam_goldstd",
  "cluster_column": "pam_k7",
  "measurements_table": "measurements_p2",
  
  "trajectory_analysis": {
    "cutoff_days": 365,
    "smoothing_frac": 0.3,
    "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"],
    "figure_size": [20, 12],
    "dpi": 300,
    "output_dir": "../outputs/cluster_trajectories"
  },
  
  "lmm_analysis": {
    "use_deviation_coding": true,
    "reference_cluster": 0,
    "adjust_for_covariates": false,
    "knot_quantiles": [0.25, 0.5, 0.75]
  }
}
```

## Error Handling

### Validation Strategy

1. **Configuration Validation**: Check all required fields present and valid types
2. **Database Validation**: Verify database files exist and tables are accessible
3. **Data Validation**: Check for required columns and sufficient data
4. **Model Validation**: Verify convergence and check for numerical issues

### Error Messages

Provide specific, actionable error messages:

```python
# Example error messages
"Configuration file not found at {path}. Please create cluster_config.json."
"Cluster column '{column}' not found in table '{table}'. Available columns: {columns}"
"Insufficient data for cluster {cluster_id}: only {n} patients found (minimum 10 required)"
"LMM did not converge after {max_iter} iterations. Consider adjusting model specification."
```

## Testing Strategy

### Unit Tests

1. Test configuration loading and validation
2. Test data loading functions with mock databases
3. Test trajectory calculation functions
4. Test smoothing algorithms
5. Test color palette generation

### Integration Tests

1. Test end-to-end spaghetti plot generation with test data
2. Test LMM fitting with synthetic trajectory data
3. Test configuration file parsing

### Visual Regression Tests

1. Generate reference plots with known data
2. Compare new plots against references
3. Flag significant visual differences

## Performance Considerations

### Optimization Strategies

1. **Vectorized Operations**: Use pandas/numpy vectorization for trajectory calculations
2. **Efficient Smoothing**: Cache LOWESS results where appropriate
3. **Lazy Loading**: Only load data needed for specific analysis
4. **Parallel Processing**: Consider parallelizing cluster-wise calculations if needed

### Memory Management

1. Process clusters sequentially to limit memory usage
2. Clear large intermediate DataFrames when no longer needed
3. Use appropriate dtypes (int32 vs int64, float32 vs float64)

## Integration with Existing Modules

### Consistency with cluster_descriptions.py

- Follow similar function naming conventions
- Use same configuration loading pattern
- Maintain consistent color palette usage
- Follow same output directory structure

### Consistency with wgc_visualizations.py

- Use similar plotting style and aesthetics
- Follow same figure saving conventions
- Use consistent variable name mapping approach

### Usage in Notebook

Simple notebook interface:

```python
from cluster_trajectories import generate_spaghetti_plots, run_lmm_analysis, load_trajectory_config

# Load configuration
config = load_trajectory_config()

# Generate spaghetti plots
figures = generate_spaghetti_plots(config)

# Run LMM analysis
lmm_results = run_lmm_analysis(config)
```

## Future Enhancements

1. **Interactive Plots**: Add plotly support for interactive exploration
2. **Additional Metrics**: Include velocity, acceleration of weight loss
3. **Subgroup Analysis**: Support stratified trajectory analysis
4. **Export Options**: Add data export for external statistical software
5. **Comparison Tools**: Functions to statistically compare trajectory parameters across clusters
