# Visualization Pipeline Refactor - Design

## Overview

This design extends the existing `descriptive_visualizations.py` module to support cluster-based analyses alongside the current WGC-based analyses. The approach focuses on minimal, targeted modifications to existing functions rather than creating new modules or abstractions. All changes maintain backward compatibility and follow the existing code style.

## Architecture

### File Structure (No New Files Created)

```
scripts/
├── descriptive_visualizations.py  # MODIFIED: Add cluster support to existing functions
├── descriptive_comparisons.py     # MODIFIED: Add cluster analysis functions  
├── cluster_config.json            # NEW: Cluster labels and colors
├── human_readable_variable_names.json  # UNCHANGED: Existing variable labels
├── fdr_correction_utils.py        # UNCHANGED: Existing FDR correction
└── paper2_notebook.ipynb          # MODIFIED: Add cluster visualization calls
```

### Design Principles

1. **Extend, Don't Replace**: Modify existing functions to accept optional parameters for cluster support
2. **Backward Compatible**: All existing code continues to work without changes
3. **Simple and Linear**: No classes, no abstractions, just straightforward procedural code
4. **Configuration Over Code**: Use JSON files for labels and colors, not hardcoded values
5. **Display After Save**: All plots show in notebooks after being saved to disk

## Components and Interfaces

### 1. Cluster Configuration File (`cluster_config.json`)

**Purpose**: Store cluster labels and colors in a centralized, editable file

**Structure**:
```json
{
  "cluster_labels": {
    "0": "Male-dominant, inactive",
    "1": "Women's health",
    "2": "Unspecified causes",
    "3": "External events",
    "4": "Medical issues",
    "5": "Unhealthy eating",
    "6": "Psyche and eating"
  },
  "cluster_colors": {
    "0": "#FF6700",
    "1": "#1f77b4",
    "2": "#2ca02c",
    "3": "#d62728",
    "4": "#9467bd",
    "5": "#8c564b",
    "6": "#e377c2"
  }
}
```

**Loading Function** (add to `descriptive_visualizations.py`):
```python
def load_cluster_config(json_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load cluster labels and colors from JSON file.
    Returns dict with 'cluster_labels' and 'cluster_colors' keys.
    Falls back to defaults if file doesn't exist.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: Cluster config file not found at '{json_path}'. Using defaults.")
        return {
            'cluster_labels': {},
            'cluster_colors': {}
        }
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load cluster config '{json_path}'. Error: {e}")
        return {'cluster_labels': {}, 'cluster_colors': {}}

def get_cluster_label(cluster_id: int, cluster_config: Dict) -> str:
    """Get human-readable label for cluster, with fallback to default"""
    labels = cluster_config.get('cluster_labels', {})
    return labels.get(str(cluster_id), f'Cluster {cluster_id}')

def get_cluster_color(cluster_id: int, cluster_config: Dict, default_palette: list) -> str:
    """Get color for cluster, with fallback to default palette"""
    colors = cluster_config.get('cluster_colors', {})
    if str(cluster_id) in colors:
        return colors[str(cluster_id)]
    # Fallback to default palette
    return default_palette[cluster_id % len(default_palette)]
```

### 2. Modified Visualization Functions

#### 2.1 `plot_distribution_comparison` (Violin Plot)

**Current Signature**:
```python
def plot_distribution_comparison(
    df: pd.DataFrame,
    population_df: pd.DataFrame,
    variable: str,
    group_col: str,
    output_filename: str,
    name_map_path: str,
    output_dir: str,
    significance_map_raw: Optional[Dict[Any, float]] = None,
    significance_map_fdr: Optional[Dict[Any, float]] = None,
    alpha: float = 0.05
):
```

**Modifications**:
- Add optional parameter: `cluster_config_path: Optional[str] = None`
- When `cluster_config_path` is provided, load cluster configuration
- Use cluster labels for x-axis tick labels
- Use cluster colors for the violin plot colors
- Add `plt.show()` before `plt.close()` to display in notebook

**Key Changes**:
```python
# At the beginning of the function
cluster_config = {}
if cluster_config_path:
    cluster_config = load_cluster_config(cluster_config_path)

# When creating group labels
groups = sorted(df[group_col].unique())
for group in groups:
    if cluster_config:
        group_label = get_cluster_label(group, cluster_config)
    else:
        group_label = f'Cluster {group}'  # or WGC label
    # ... rest of plotting logic

# Before plt.close()
plt.show()  # Display in notebook
plt.close()
```

#### 2.2 `plot_stacked_bar_comparison` (Stacked Bar Chart)

**Modifications**:
- Add optional parameter: `cluster_config_path: Optional[str] = None`
- Use cluster labels for x-axis
- Use cluster colors for bars (if applicable)
- Add `plt.show()` before `plt.close()`

#### 2.3 `plot_multi_lollipop` (Multi-Variable Lollipop)

**Modifications**:
- Add optional parameter: `cluster_config_path: Optional[str] = None`
- Use cluster labels in legend
- Use cluster colors for lollipop markers and lines
- Add `plt.show()` before `plt.close()`

**Key Changes**:
```python
# Load cluster config
cluster_config = {}
if cluster_config_path:
    cluster_config = load_cluster_config(cluster_config_path)

# When assigning colors
clusters = sorted(data_df['cluster'].unique(), key=lambda x: int(x.split(' ')[-1]))
colors = []
for i, cluster in enumerate(clusters):
    cluster_id = int(cluster.split(' ')[-1])
    if cluster_config:
        color = get_cluster_color(cluster_id, cluster_config, plt.cm.Set3(np.linspace(0, 1, len(clusters))))
    else:
        color = plt.cm.Set3(np.linspace(0, 1, len(clusters)))[i]
    colors.append(color)
```

#### 2.4 `plot_forest` (Forest Plot with Enhanced Display)

**Current Signature**:
```python
def plot_forest(
    results_df: pd.DataFrame,
    output_filename: str,
    name_map_path: str,
    output_dir: str,
    effect_col: str = 'effect',
    ci_lower_col: str = 'ci_lower',
    ci_upper_col: str = 'ci_upper',
    label_col: str = 'group',
    title: str = 'Forest Plot'
):
```

**Modifications**:
- Add optional parameter: `cluster_config_path: Optional[str] = None`
- Add optional parameter: `effect_type: str = 'RD'` (either 'RD' for risk difference or 'RR' for risk ratio)
- Create a secondary y-axis on the right showing effect sizes and CIs
- Remove significance asterisks (not needed for forest plots)
- Use cluster labels if cluster_config_path is provided
- Add `plt.show()` before `plt.close()`

**Key Changes**:
```python
# After creating the main plot
# Add secondary y-axis with effect sizes
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(y_pos)

# Format labels based on effect type
labels = []
for _, row in plot_data.iterrows():
    if effect_type == 'RR':
        label = f"RR: {row[effect_col]:.2f} [{row[ci_lower_col]:.2f}-{row[ci_upper_col]:.2f}]"
    else:  # RD
        label = f"RD: {row[effect_col]:.1f}% [{row[ci_lower_col]:.1f}-{row[ci_upper_col]:.1f}%]"
    labels.append(label)

ax2.set_yticklabels(labels)
ax2.set_ylabel('Effect Size [95% CI]', fontsize=12)

# Set reference line based on effect type
ref_value = 1 if effect_type == 'RR' else 0
ax.axvline(x=ref_value, color='red', linestyle='--', linewidth=2)
```

#### 2.5 `plot_wgc_cluster_heatmap` (NEW FUNCTION)

**Purpose**: Visualize WGC prevalence across clusters

**Signature**:
```python
def plot_wgc_cluster_heatmap(
    prevalence_df: pd.DataFrame,
    output_filename: str,
    name_map_path: str,
    cluster_config_path: str,
    output_dir: str,
    significance_map_raw: Optional[Dict[str, Dict[int, float]]] = None,
    significance_map_fdr: Optional[Dict[str, Dict[int, float]]] = None,
    alpha: float = 0.05
):
    """
    Create a heatmap showing WGC prevalence across clusters.
    
    Args:
        prevalence_df: DataFrame with columns ['wgc_variable', 'cluster_id', 'prevalence_%', 'n']
        output_filename: Name for saved plot file
        name_map_path: Path to human_readable_variable_names.json
        cluster_config_path: Path to cluster_config.json
        output_dir: Directory to save plot
        significance_map_raw: {wgc_variable: {cluster_id: p_value}}
        significance_map_fdr: {wgc_variable: {cluster_id: fdr_p_value}}
        alpha: Significance threshold
    """
```

**Implementation**:
```python
def plot_wgc_cluster_heatmap(
    prevalence_df: pd.DataFrame,
    output_filename: str,
    name_map_path: str,
    cluster_config_path: str,
    output_dir: str,
    significance_map_raw: Optional[Dict[str, Dict[int, float]]] = None,
    significance_map_fdr: Optional[Dict[str, Dict[int, float]]] = None,
    alpha: float = 0.05
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Pivot data to matrix format: rows=WGCs, columns=clusters
    matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='prevalence_%')
    
    # Get cluster labels for columns
    cluster_labels = [get_cluster_label(cid, cluster_config) for cid in matrix.columns]
    
    # Get WGC labels for rows
    wgc_labels = [get_nice_name(wgc, name_map) for wgc in matrix.index]
    
    # Also need n_matrix for sample counts
    n_matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='n')
    
    # Create annotations with "n (%)** " format
    annotations = np.empty_like(matrix, dtype=object)
    for i, wgc in enumerate(matrix.index):
        for j, cluster_id in enumerate(matrix.columns):
            pct = matrix.iloc[i, j]
            n = n_matrix.iloc[i, j]
            
            # Get significance marker
            sig_marker = ''
            if significance_map_fdr and wgc in significance_map_fdr:
                p_fdr = significance_map_fdr[wgc].get(cluster_id)
                if p_fdr and p_fdr < alpha:
                    sig_marker = '**'
            if not sig_marker and significance_map_raw and wgc in significance_map_raw:
                p_raw = significance_map_raw[wgc].get(cluster_id)
                if p_raw and p_raw < alpha:
                    sig_marker = '*'
            
            # Format: "n (%)** " where n is count, % is percentage, ** is significance
            annotations[i, j] = f"{int(n)} ({pct:.1f}%){sig_marker}"
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(cluster_labels) * 1.5), max(8, len(wgc_labels) * 0.6)))
    
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt='',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Prevalence (%)'},
        xticklabels=cluster_labels,
        yticklabels=wgc_labels,
        ax=ax
    )
    
    ax.set_title('Weight Gain Cause Prevalence by Cluster', fontsize=16, weight='bold')
    ax.set_xlabel('Clusters', fontsize=14)
    ax.set_ylabel('Weight Gain Causes', fontsize=14)
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Heatmap saved to: {output_path}")
```

### 3. New Analysis Function in `descriptive_comparisons.py`

#### 3.1 New Function: `cluster_vs_population_mean_analysis`

**Purpose**: Analyze WGC prevalence in each cluster compared to population mean, following the same pattern as `wgc_vs_population_mean_analysis`

**Signature**:
```python
def cluster_vs_population_mean_analysis(
    df: pd.DataFrame,
    config: descriptive_comparisons_config,
    conn,
    cluster_col: str = 'cluster_id'
):
    """
    Performs cluster vs population mean analysis for WGC variables.
    Analyzes the prevalence of each WGC in each cluster compared to the population.
    
    Args:
        df: DataFrame with cluster_id column and WGC binary variables
        config: Configuration object (reuse existing config structure)
        conn: Database connection for saving results
        cluster_col: Name of column containing cluster IDs (default: 'cluster_id')
    
    Output:
        Saves two tables to database:
        - {output_table}_detailed: Full table with p-values
        - {output_table}: Publication-ready table with asterisks
        
        Returns DataFrame for heatmap generation
    """
```

**Implementation Approach**:
- Follow the exact same pattern as `wgc_vs_population_mean_analysis`
- Instead of iterating over WGC variables as groups, iterate over clusters as groups
- For each cluster, calculate prevalence of each WGC variable
- Compare each cluster's WGC prevalence to the population mean prevalence
- Apply FDR correction if enabled
- Save results that can be used for heatmap generation

**Key Logic**:
```python
def cluster_vs_population_mean_analysis(df, config, conn, cluster_col='cluster_id'):
    output_table_name = getattr(config, 'cluster_vs_mean_output_table', None)
    if not isinstance(output_table_name, str) or not output_table_name.strip():
        return
    
    print("Running Cluster vs Population Mean Analysis...")
    
    # Get WGC columns from row_order
    row_order = config.row_order
    cause_cols = get_cause_cols(row_order)  # WGC variables
    var_types = get_variable_types(df, cause_cols)
    
    # Get unique clusters
    clusters = sorted(df[cluster_col].unique())
    cluster_labels = [f'Cluster {cid}' for cid in clusters]
    
    # Create groups dictionary
    groups = {}
    for cluster_id in clusters:
        label = f'Cluster {cluster_id}'
        groups[label] = df[df[cluster_col] == cluster_id]
    
    # Build summary rows - analyzing WGC variables across clusters
    summary_rows = []
    
    # N row
    n_row = {"Variable": "N", "Population Mean (±SD) or N (%)": len(df)}
    for label in cluster_labels:
        n_row[f"{label}: Mean/N"] = len(groups[label])
        n_row[f"{label}: p-value"] = "N/A"
    summary_rows.append(n_row)
    
    # Process each WGC variable (rows in the output)
    for i, (var, _) in enumerate(row_order):
        if var == "N" or var.startswith("delim_"):
            continue
        if var not in cause_cols:  # Only process WGC variables
            continue
        
        print(f"  Processing WGC variable {i}/{len(row_order)}: {var}")
        vtype = var_types.get(var, "categorical")  # WGCs are categorical
        row = {"Variable": var}
        
        for cluster_name, cluster_df in groups.items():
            row[cluster_name] = format_value(cluster_df, var, vtype)
            row[f"{cluster_name}: p-value"] = perform_comparison(df, cluster_df, var, vtype)
        
        summary_rows.append(row)
    
    # Create DataFrame and apply FDR correction
    summary_df = pd.DataFrame(add_empty_rows_and_pretty_names(summary_rows, row_order))
    
    if config.fdr_correction:
        # Apply FDR correction (same as wgc_stratification)
        pvalue_columns = [f"{cluster_name}: p-value" for cluster_name in groups.keys()]
        # ... FDR correction logic ...
    
    # Save to database
    summary_df.to_sql(config.cluster_output_table, conn, if_exists="replace", index=False)
    print(f"Cluster stratification table saved to {config.cluster_output_table}")
```

#### 3.2 Modified Function: `wgc_vs_population_mean_analysis`

**Modification**: Add support for cluster-based analysis

**Approach**:
- Add optional parameter: `group_col: Optional[str] = None`
- When `group_col='cluster_id'`, analyze clusters instead of WGCs
- Reuse existing logic but iterate over clusters instead of WGC variables
- This function will provide data for the heatmap

**Key Changes**:
```python
def wgc_vs_population_mean_analysis(
    df,
    config,
    conn,
    group_col: Optional[str] = None  # NEW PARAMETER
):
    # Determine if we're analyzing WGCs or clusters
    if group_col == 'cluster_id':
        # Cluster-based analysis
        groups = {}
        group_labels = []
        clusters = sorted(df[group_col].unique())
        for cluster_id in clusters:
            label = f'Cluster {cluster_id}'
            groups[label] = df[df[group_col] == cluster_id]
            group_labels.append(label)
    else:
        # WGC-based analysis (existing logic)
        cause_cols = get_cause_cols(config.row_order)
        groups = {}
        group_labels = []
        for cause in cause_cols:
            pretty_cause = next((p for v, p in config.row_order if v == cause), cause).replace(" (yes/no)", "")
            groups[pretty_cause] = df[df[cause] == 1]
            group_labels.append(pretty_cause)
    
    # Rest of the function remains the same
    # ... existing logic for calculating statistics and p-values ...
```

## Data Models

### Input Data Structures

#### For WGC-focused Analysis (Existing):
```python
df: pd.DataFrame with columns:
    - patient_id: str
    - medical_record_id: str
    - womens_health_and_pregnancy: int (0/1)
    - mental_health: int (0/1)
    - ... other WGC columns ...
    - total_wl_%: float
    - baseline_bmi: float
    - ... other outcome variables ...
```

#### For Cluster-focused Analysis (New):
```python
df: pd.DataFrame with columns:
    - patient_id: str
    - medical_record_id: str
    - cluster_id: int (0, 1, 2, ...)
    - womens_health_and_pregnancy: int (0/1)  # Still present for heatmap
    - mental_health: int (0/1)
    - ... other WGC columns ...
    - total_wl_%: float
    - baseline_bmi: float
    - ... other outcome variables ...
```

### Significance Data Structures

#### For Individual Plots:
```python
significance_map_raw: Dict[Any, float]
# Example: {0: 0.03, 1: 0.001, 2: 0.12}  # cluster_id: p_value

significance_map_fdr: Dict[Any, float]
# Example: {0: 0.06, 1: 0.005, 2: 0.18}  # cluster_id: fdr_p_value
```

#### For Heatmap:
```python
significance_map_raw: Dict[str, Dict[int, float]]
# Example: {
#     'mental_health': {0: 0.03, 1: 0.001, 2: 0.12},
#     'eating_habits': {0: 0.08, 1: 0.04, 2: 0.15}
# }

significance_map_fdr: Dict[str, Dict[int, float]]
# Example: {
#     'mental_health': {0: 0.06, 1: 0.005, 2: 0.18},
#     'eating_habits': {0: 0.12, 1: 0.08, 2: 0.20}
# }
```

### Heatmap Data Structure:
```python
prevalence_df: pd.DataFrame with columns:
    - wgc_variable: str (e.g., 'mental_health')
    - cluster_id: int (e.g., 0, 1, 2)
    - prevalence_%: float (e.g., 26.5)
    - n: int (sample size)

# Example:
#   wgc_variable  cluster_id  prevalence_%  n
#   mental_health      0          26.5      271
#   mental_health      1          99.6      500
#   eating_habits      0          33.4      271
#   eating_habits      1          54.0      500
```

## Error Handling

### Validation Checks

**In visualization functions**:
```python
# Check required columns exist
required_cols = [group_col, variable]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Check for empty data
if df.empty:
    print(f"⚠️ Warning: Empty DataFrame for variable '{variable}'. Skipping plot.")
    return

# Check cluster config matches data
if cluster_config_path:
    cluster_config = load_cluster_config(cluster_config_path)
    data_clusters = set(df[group_col].unique())
    config_clusters = set(int(k) for k in cluster_config.get('cluster_labels', {}).keys())
    missing_in_config = data_clusters - config_clusters
    if missing_in_config:
        print(f"⚠️ Warning: Clusters {missing_in_config} not found in config. Using default labels.")
```

**In analysis functions**:
```python
# Check cluster column exists
if cluster_col not in df.columns:
    raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

# Check for sufficient data
if len(df) < 10:
    print(f"⚠️ Warning: Very small sample size (n={len(df)}). Results may be unreliable.")

# Handle empty groups
for cluster_id, cluster_df in groups.items():
    if len(cluster_df) < 5:
        print(f"⚠️ Warning: Cluster {cluster_id} has only {len(cluster_df)} samples. Statistical tests may be unreliable.")
```

## Testing Strategy

### Manual Testing Approach

Since this is research code, formal unit tests are not required. Instead, use manual validation:

1. **Visual Inspection**: Generate plots and verify they look correct
2. **Comparison**: Compare cluster plots to WGC plots to ensure consistency
3. **Data Validation**: Check that numbers in plots match source data
4. **Edge Cases**: Test with small clusters, missing data, single cluster

### Test Scenarios

1. **Cluster analysis with full data**:
   - Load cluster assignments
   - Run cluster_stratification
   - Generate all plot types
   - Verify labels and colors match cluster_config.json

2. **Missing cluster config**:
   - Delete cluster_config.json temporarily
   - Verify default labels and colors are used
   - Verify warning message is printed

3. **Mixed WGC and cluster analysis**:
   - Run WGC analysis (existing code)
   - Run cluster analysis (new code)
   - Verify both work independently

4. **Heatmap with significance**:
   - Generate heatmap with p-values
   - Verify asterisks appear correctly
   - Verify percentages match source data

## Integration with Existing Workflow

### Notebook Usage Pattern

**Existing WGC Analysis** (unchanged):
```python
from descriptive_comparisons import run_descriptive_comparisons
from descriptive_visualizations import plot_distribution_comparison, plot_stacked_bar_comparison

# Run analysis
run_descriptive_comparisons(master_config)

# Generate plots
plot_distribution_comparison(
    df=wgc_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='mental_health',
    output_filename='wgc_mental_health_violin.png',
    name_map_path='human_readable_variable_names.json',
    output_dir='outputs',
    significance_map_raw=p_raw_dict,
    significance_map_fdr=p_fdr_dict
)
```

**New Cluster Analysis**:
```python
from descriptive_comparisons import cluster_vs_population_mean_analysis
from descriptive_visualizations import (
    plot_distribution_comparison,
    plot_stacked_bar_comparison,
    plot_multi_lollipop,
    plot_forest,
    plot_wgc_cluster_heatmap
)

# Assume cluster_df has a 'cluster_id' column from clustering analysis

# Run cluster vs population mean analysis (for WGC prevalence in clusters)
cluster_config = descriptive_comparisons_config(
    analysis_name='cluster_wgc_analysis',
    input_cohort_name='timetoevent_wgc_compl',
    mother_cohort_name='timetoevent_all',
    row_order=ROW_ORDER,  # Same as WGC analysis
    demographic_output_table='',  # Not used for this analysis
    demographic_strata=[],
    wgc_output_table='',  # Not used
    wgc_strata=[],
    cluster_vs_mean_output_table='cluster_wgc_vs_mean',  # NEW
    fdr_correction=True
)

with sqlite3.connect(paths.paper_out_db) as conn:
    # This function analyzes WGC prevalence across clusters
    results_df = cluster_vs_population_mean_analysis(
        cluster_df, 
        cluster_config, 
        conn, 
        cluster_col='cluster_id'
    )

# Generate cluster violin plot (for outcome variables like weight loss)
plot_distribution_comparison(
    df=cluster_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='cluster_id',
    output_filename='cluster_wl_violin.png',
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',  # NEW PARAMETER
    output_dir='outputs',
    significance_map_raw={0: 0.03, 1: 0.001, 2: 0.12},
    significance_map_fdr={0: 0.06, 1: 0.005, 2: 0.18}
)

# Generate WGC-cluster heatmap
# The data comes from cluster_vs_population_mean_analysis results
# Extract prevalence data from the results table
with sqlite3.connect(paths.paper_out_db) as conn:
    results_df = pd.read_sql_query(
        "SELECT * FROM cluster_wgc_vs_mean_detailed", 
        conn
    )

# Transform results into heatmap format
wgc_cols = get_cause_cols(ROW_ORDER)  # Get WGC variable names
clusters = sorted(cluster_df['cluster_id'].unique())

prevalence_data = []
sig_map_raw = {}
sig_map_fdr = {}

for _, row in results_df.iterrows():
    wgc_var = row['Variable']
    if wgc_var not in wgc_cols:
        continue
    
    sig_map_raw[wgc_var] = {}
    sig_map_fdr[wgc_var] = {}
    
    for cluster_id in clusters:
        cluster_label = f'Cluster {cluster_id}'
        
        # Extract n (%) from "Mean/N" column
        mean_n_str = row[f'{cluster_label}: Mean/N']
        # Parse "271 (26.5%)" format
        n = int(mean_n_str.split('(')[0].strip())
        pct = float(mean_n_str.split('(')[1].split('%')[0])
        
        prevalence_data.append({
            'wgc_variable': wgc_var,
            'cluster_id': cluster_id,
            'prevalence_%': pct,
            'n': n
        })
        
        # Extract p-values
        sig_map_raw[wgc_var][cluster_id] = row[f'{cluster_label}: p-value']
        sig_map_fdr[wgc_var][cluster_id] = row[f'{cluster_label}: p-value (FDR-corrected)']

prevalence_df = pd.DataFrame(prevalence_data)

plot_wgc_cluster_heatmap(
    prevalence_df=prevalence_df,
    output_filename='wgc_cluster_heatmap.png',
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    output_dir='outputs',
    significance_map_raw=sig_map_raw,
    significance_map_fdr=sig_map_fdr
)
```

## Summary of Changes

### Files Modified:
1. **descriptive_visualizations.py**:
   - Add `load_cluster_config()`, `get_cluster_label()`, `get_cluster_color()` helper functions
   - Add optional `cluster_config_path` parameter to existing plot functions
   - Modify plot functions to use cluster labels/colors when provided
   - Add `plt.show()` before `plt.close()` in all plot functions
   - Enhance `plot_forest()` with secondary y-axis for effect sizes
   - Add new `plot_wgc_cluster_heatmap()` function

2. **descriptive_comparisons.py**:
   - Add new `cluster_vs_population_mean_analysis()` function (follows pattern of `wgc_vs_population_mean_analysis`)

3. **cluster_config.json** (new file):
   - Create JSON file with cluster labels and colors

4. **paper2_notebook.ipynb**:
   - Add cells for cluster analysis
   - Add cells for cluster visualizations

### Backward Compatibility:
- All existing code continues to work without modification
- New parameters are optional
- Default behavior unchanged when new parameters not provided
