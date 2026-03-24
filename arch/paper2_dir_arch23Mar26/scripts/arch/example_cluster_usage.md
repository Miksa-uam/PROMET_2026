# Cluster Descriptions Module - Usage Examples

## Overview

The new `cluster_descriptions_new.py` module is:
- **Self-contained** - No dependencies on other visualization modules
- **Modular** - Task-specific functions for notebook use
- **Integrated** - Statistical testing built into visualizations
- **Configurable** - All labels, colors, and parameters customizable

## Quick Start

### 1. Load and Merge Data

```python
from cluster_descriptions_new import load_and_merge_cluster_data

cluster_df, pop_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all"
)
```

### 2. Generate Distribution Plots (Violin)

```python
from cluster_descriptions_new import plot_cluster_distributions

plot_cluster_distributions(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['total_wl_%', 'baseline_bmi', 'total_followup_days'],
    output_dir="../outputs/cluster_analysis",
    name_map_path="human_readable_variable_names.json",
    cluster_config_path="cluster_config.json",
    calculate_significance=True,
    fdr_correction=True
)
```

### 3. Generate Categorical Plots (Stacked Bar)

```python
from cluster_descriptions_new import plot_cluster_categorical

plot_cluster_categorical(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['sex_f', '5%_wl_achieved', '10%_wl_achieved'],
    output_dir="../outputs/cluster_analysis",
    name_map_path="human_readable_variable_names.json",
    cluster_config_path="cluster_config.json",
    calculate_significance=True,
    fdr_correction=True,
    legend_labels={'achieved': 'Achieved', 'not_achieved': 'Not Achieved'}
)
```

### 4. Statistical Analysis + Tables

```python
from cluster_descriptions_new import analyze_cluster_vs_population

results_df = analyze_cluster_vs_population(
    cluster_df=cluster_df,
    population_df=pop_df,
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity',
                   'womens_health_and_pregnancy', 'medication_disease_injury'],
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="cluster_k7_wgc_analysis",
    name_map_path="human_readable_variable_names.json",
    fdr_correction=True
)
```

This creates two tables:
- `cluster_k7_wgc_analysis_detailed` (with p-values)
- `cluster_k7_wgc_analysis` (publication-ready with asterisks)

### 5. Generate Heatmap

```python
from cluster_descriptions_new import plot_cluster_heatmap

plot_cluster_heatmap(
    results_df=results_df,
    output_filename="cluster_k7_wgc_heatmap.png",
    output_dir="../outputs/cluster_analysis",
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity',
                   'womens_health_and_pregnancy', 'medication_disease_injury'],
    name_map_path="human_readable_variable_names.json",
    cluster_config_path="cluster_config.json"
)
```

### 6. Generate Lollipop Plot

```python
from cluster_descriptions_new import plot_cluster_lollipop

plot_cluster_lollipop(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['total_wl_%', 'baseline_bmi', 'dietitian_visits'],
    output_filename="cluster_k7_lollipop.png",
    output_dir="../outputs/cluster_analysis",
    name_map_path="human_readable_variable_names.json",
    cluster_config_path="cluster_config.json"
)
```

## Complete Notebook Cell Example

```python
# Complete cluster analysis workflow
from cluster_descriptions_new import (
    load_and_merge_cluster_data,
    plot_cluster_distributions,
    plot_cluster_categorical,
    analyze_cluster_vs_population,
    plot_cluster_heatmap,
    plot_cluster_lollipop
)

# 1. Load data
cluster_df, pop_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all"
)

# 2. Continuous variables
plot_cluster_distributions(
    cluster_df, pop_df,
    variables=['total_wl_%', 'baseline_bmi'],
    output_dir="../outputs/cluster_k7",
    calculate_significance=True,
    fdr_correction=True
)

# 3. Categorical variables
plot_cluster_categorical(
    cluster_df, pop_df,
    variables=['sex_f', '5%_wl_achieved'],
    output_dir="../outputs/cluster_k7",
    calculate_significance=True,
    fdr_correction=True
)

# 4. WGC analysis
results_df = analyze_cluster_vs_population(
    cluster_df, pop_df,
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity'],
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="cluster_k7_wgc",
    fdr_correction=True
)

# 5. Heatmap
plot_cluster_heatmap(
    results_df,
    output_filename="wgc_heatmap.png",
    output_dir="../outputs/cluster_k7",
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity']
)

# 6. Lollipop
plot_cluster_lollipop(
    cluster_df, pop_df,
    variables=['total_wl_%', 'baseline_bmi'],
    output_filename="multi_lollipop.png",
    output_dir="../outputs/cluster_k7"
)
```

## Key Features

### Automatic Statistical Testing
- P-values calculated automatically for each cluster vs population
- FDR correction applied when requested
- Significance markers (* and **) added to plots

### Fixed Issues
✅ File paths resolved correctly (relative to script location)
✅ Cluster IDs converted to strings for config lookup
✅ Heatmap parsing fixed
✅ Statistical tests integrated into visualizations
✅ Self-contained (no external module dependencies)

### Customization Options

**Legend Labels:**
```python
legend_labels={'achieved': 'Yes', 'not_achieved': 'No'}
```

**Significance Settings:**
```python
calculate_significance=True,
fdr_correction=True,
alpha=0.05
```

**Different Cluster Solution:**
```python
cluster_column="pam_k5"  # Use k=5 instead of k=7
```

## Next Steps

1. **Test with your data** - Run the examples above
2. **Check outputs** - Verify plots and tables look correct
3. **Customize** - Adjust parameters as needed
4. **Replace old module** - Once tested, rename `cluster_descriptions_new.py` to `cluster_descriptions.py`
