# Cluster Visualization Usage Guide

Quick reference for using the new cluster-based visualization features.

## Overview

The visualization pipeline now supports both WGC-based and cluster-based analyses with minimal code changes. All existing WGC analysis code continues to work without modifications.

## 1. Cluster Configuration

### Setup cluster_config.json

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

Location: `scripts/cluster_config.json`

## 2. Cluster vs Population Mean Analysis

Analyze WGC prevalence across clusters:

```python
from descriptive_comparisons import cluster_vs_population_mean_analysis
from paper12_config import descriptive_comparisons_config

# Configure analysis
config = descriptive_comparisons_config(
    analysis_name='cluster_wgc_analysis',
    input_cohort_name='your_cohort',
    mother_cohort_name='mother_cohort',
    row_order=ROW_ORDER,
    demographic_output_table='',
    demographic_strata=[],
    wgc_output_table='',
    wgc_strata=[],
    cluster_vs_mean_output_table='cluster_wgc_vs_mean',
    fdr_correction=True
)

# Run analysis
with sqlite3.connect(db_path) as conn:
    results_df = cluster_vs_population_mean_analysis(
        df=cluster_df,  # Must have 'cluster_id' column
        config=config,
        conn=conn,
        cluster_col='cluster_id'
    )
```

**Output Tables:**
- `cluster_wgc_vs_mean_detailed` - Full table with p-values
- `cluster_wgc_vs_mean` - Publication-ready with asterisks

## 3. Cluster Visualizations

### Violin Plot (Distribution Comparison)

```python
from descriptive_visualizations import plot_distribution_comparison

plot_distribution_comparison(
    df=cluster_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='cluster_id',
    output_filename='cluster_wl_violin.png',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',  # NEW
    output_dir='outputs',
    significance_map_raw={0: 0.03, 1: 0.001, 2: 0.12},
    significance_map_fdr={0: 0.06, 1: 0.005, 2: 0.18}
)
```

### Stacked Bar Plot

```python
from descriptive_visualizations import plot_stacked_bar_comparison

plot_stacked_bar_comparison(
    df=cluster_df,
    population_df=population_df,
    variable='mental_health',
    group_col='cluster_id',
    output_filename='cluster_mental_health_bar.png',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',  # NEW
    output_dir='outputs',
    significance_map_raw={0: 0.02, 1: 0.001},
    significance_map_fdr={0: 0.04, 1: 0.005}
)
```

### Lollipop Plot (Multi-Variable Comparison)

```python
from descriptive_visualizations import plot_multi_lollipop

# Prepare data: percent change vs population mean
lollipop_data = []
for cluster_id in cluster_df['cluster_id'].unique():
    cluster_subset = cluster_df[cluster_df['cluster_id'] == cluster_id]
    for variable in ['total_wl_%', 'baseline_bmi']:
        pop_mean = population_df[variable].mean()
        cluster_mean = cluster_subset[variable].mean()
        pct_change = ((cluster_mean - pop_mean) / pop_mean) * 100
        lollipop_data.append({
            'variable': variable,
            'cluster': f'Cluster {int(cluster_id)}',
            'value': pct_change
        })

lollipop_df = pd.DataFrame(lollipop_data)

plot_multi_lollipop(
    data_df=lollipop_df,
    output_filename='cluster_multi_lollipop.png',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',  # NEW
    output_dir='outputs'
)
```

### Forest Plot with Secondary Y-Axis

```python
from descriptive_visualizations import plot_forest

# Prepare forest plot data
forest_data = pd.DataFrame({
    'group': ['Cluster 0', 'Cluster 1', 'Cluster 2'],
    'effect': [1.2, 0.8, 1.5],  # Risk ratios
    'ci_lower': [1.0, 0.6, 1.2],
    'ci_upper': [1.4, 1.0, 1.8]
})

plot_forest(
    results_df=forest_data,
    output_filename='cluster_forest_rr.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir='outputs',
    cluster_config_path='scripts/cluster_config.json',  # NEW
    effect_type='RR'  # or 'RD' for risk difference
)
```

### Heatmap (WGC Prevalence Across Clusters)

```python
from descriptive_visualizations import plot_wgc_cluster_heatmap

# Extract prevalence data from cluster_vs_population_mean_analysis results
with sqlite3.connect(db_path) as conn:
    results_df = pd.read_sql_query(
        "SELECT * FROM cluster_wgc_vs_mean_detailed", 
        conn
    )

# Transform to heatmap format
wgc_cols = ['mental_health', 'eating_habits', 'womens_health_and_pregnancy']
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
        
        # Parse "n (%)" format from Mean/N column
        mean_n_str = row[f'{cluster_label}: Mean/N']
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
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',
    output_dir='outputs',
    significance_map_raw=sig_map_raw,
    significance_map_fdr=sig_map_fdr
)
```

## 4. Backward Compatibility

All existing WGC analysis code works without changes:

```python
# Existing code - no modifications needed
plot_distribution_comparison(
    df=wgc_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='mental_health',  # WGC variable
    output_filename='wgc_mental_health_violin.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir='outputs'
    # Note: No cluster_config_path - works as before
)
```

## 5. Key Features

### Significance Markers
- `*` = Raw p-value < 0.05, FDR p-value ≥ 0.05
- `**` = FDR-corrected p-value < 0.05
- No marker = Not significant

### Error Handling
- Missing cluster_config.json → Uses default labels ("Cluster 0", "Cluster 1", ...)
- Invalid JSON → Falls back to defaults with warning
- Missing cluster IDs in config → Uses defaults with warning
- Empty data → Skips plot with warning

### Figure Display
All plots automatically display in Jupyter notebooks after saving via `plt.show()`.

## 6. Data Requirements

### For Cluster Analysis
Your DataFrame must have:
- `cluster_id` column with integer cluster assignments (0, 1, 2, ...)
- WGC binary variables (0/1) for heatmap analysis
- Outcome variables for distribution comparisons

### For Visualizations
- Cluster IDs should be integers
- Significance maps use cluster IDs as keys: `{0: 0.03, 1: 0.001, ...}`
- For heatmaps, significance maps are nested: `{'wgc_var': {0: 0.03, 1: 0.001}}`

## 7. Common Patterns

### Complete Cluster Analysis Workflow

```python
import sqlite3
import pandas as pd
from descriptive_comparisons import cluster_vs_population_mean_analysis
from descriptive_visualizations import (
    plot_distribution_comparison,
    plot_wgc_cluster_heatmap
)

# 1. Run cluster analysis
with sqlite3.connect(db_path) as conn:
    results_df = cluster_vs_population_mean_analysis(
        df=cluster_df,
        config=config,
        conn=conn
    )

# 2. Generate outcome visualizations
plot_distribution_comparison(
    df=cluster_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='cluster_id',
    output_filename='cluster_wl_violin.png',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',
    output_dir='outputs'
)

# 3. Generate WGC prevalence heatmap
# (Extract and transform data as shown in section 3)
plot_wgc_cluster_heatmap(...)
```

## 8. Troubleshooting

**Issue:** Cluster labels not showing correctly  
**Solution:** Check that cluster_config.json has entries for all cluster IDs in your data

**Issue:** Colors not matching configuration  
**Solution:** Verify cluster_colors in cluster_config.json uses string keys ("0", "1", ...)

**Issue:** Figures not displaying in notebook  
**Solution:** Ensure you're running in Jupyter notebook environment (plt.show() requires interactive backend)

**Issue:** Missing cluster IDs warning  
**Solution:** Add missing cluster IDs to cluster_config.json or accept default labels

## 9. Tips

1. **Consistent Cluster IDs:** Use integers starting from 0 for cluster IDs
2. **Configuration First:** Set up cluster_config.json before running visualizations
3. **Test with Defaults:** If cluster_config.json is missing, code still works with default labels
4. **FDR Correction:** Always enable FDR correction for multiple comparisons
5. **Publication Tables:** Use the non-detailed table (without "_detailed" suffix) for publications

## 10. Example Notebook Cell

```python
# Complete example for notebook
import sqlite3
import pandas as pd
from descriptive_comparisons import cluster_vs_population_mean_analysis
from descriptive_visualizations import plot_distribution_comparison
from paper12_config import descriptive_comparisons_config

# Load data with cluster assignments
cluster_df = pd.read_sql_query("SELECT * FROM cohort_with_clusters", conn)
population_df = pd.read_sql_query("SELECT * FROM population_cohort", conn)

# Configure and run analysis
config = descriptive_comparisons_config(
    analysis_name='cluster_analysis',
    input_cohort_name='cohort_with_clusters',
    mother_cohort_name='population_cohort',
    row_order=ROW_ORDER,
    demographic_output_table='',
    demographic_strata=[],
    wgc_output_table='',
    wgc_strata=[],
    cluster_vs_mean_output_table='cluster_wgc_vs_mean',
    fdr_correction=True
)

with sqlite3.connect('database.db') as conn:
    cluster_vs_population_mean_analysis(cluster_df, config, conn)

# Generate visualization
plot_distribution_comparison(
    df=cluster_df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='cluster_id',
    output_filename='cluster_wl_distribution.png',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',
    output_dir='outputs'
)
```
