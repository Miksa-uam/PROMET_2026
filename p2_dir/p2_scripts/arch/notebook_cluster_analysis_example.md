# Cluster Analysis Pipeline - Notebook Example

Copy this into a notebook cell to run the complete cluster analysis pipeline.

## Full Pipeline Example

```python
from cluster_descriptions import cluster_analysis_config, run_cluster_analysis_pipeline

# Define variables to analyze
CONTINUOUS_VARS = ['total_wl_%', 'baseline_bmi', 'total_followup_days']
CATEGORICAL_VARS = ['sex_f', '5%_wl_achieved', '10%_wl_achieved']
WGC_VARS = ['mental_health', 'eating_habits', 'physical_inactivity', 
            'womens_health_and_pregnancy', 'medication_disease_injury']

# Row order for tables
ROW_ORDER = [
    ("N", "N"),
    ("delim_wgc", "--- Weight Gain Causes ---"),
    ("mental_health", "Mental health (yes/no)"),
    ("eating_habits", "Eating habits (yes/no)"),
    ("physical_inactivity", "Physical inactivity (yes/no)"),
    ("womens_health_and_pregnancy", "Women's health (yes/no)"),
    ("medication_disease_injury", "Medical issues (yes/no)"),
]

# Configure analysis
config = cluster_analysis_config(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all",
    cluster_config_path="scripts/cluster_config.json",
    name_map_path="scripts/human_readable_variable_names.json",
    output_dir="../outputs/cluster_k7_analysis",
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_prefix="cluster_k7",
    variables_to_analyze=CONTINUOUS_VARS,
    categorical_variables=CATEGORICAL_VARS,
    wgc_variables=WGC_VARS,
    row_order=ROW_ORDER,
    fdr_correction=True,
    alpha=0.05
)

# Run complete pipeline
run_cluster_analysis_pipeline(config)
```

## What This Does

The pipeline will:

1. **Load Data**
   - Cluster assignments from cluster database
   - Outcome data from main database
   - Merge them on medical_record_id

2. **Statistical Analysis**
   - Run cluster vs population mean analysis
   - Apply FDR correction
   - Save detailed and publication-ready tables

3. **Generate Visualizations**
   - Violin plots for continuous variables
   - Stacked bar plots for categorical variables
   - Multi-variable lollipop plot
   - Forest plot (risk differences)
   - WGC prevalence heatmap

## Expected Output

**Console:**
```
============================================================
CLUSTER ANALYSIS PIPELINE
============================================================
Output directory: ../outputs/cluster_k7_analysis
Output database: ../dbs/pnk_db2_p2_out.sqlite
============================================================
Loading cluster data...
  ✓ Loaded 2463 cluster assignments
  ✓ Clusters: [0, 1, 2, 3, 4, 5, 6]
  ✓ Merged data: 2463 records with clusters
...
============================================================
PIPELINE COMPLETE
============================================================
```

**Files Created:**
- `cluster_k7_total_wl_%_violin.png`
- `cluster_k7_baseline_bmi_violin.png`
- `cluster_k7_sex_f_bar.png`
- `cluster_k7_5%_wl_achieved_bar.png`
- `cluster_k7_multi_lollipop.png`
- `cluster_k7_forest_rd.png`
- `cluster_k7_wgc_heatmap.png`

**Database Tables:**
- `cluster_k7_wgc_vs_mean_detailed` (with p-values)
- `cluster_k7_wgc_vs_mean` (publication-ready with asterisks)

## Customization Options

### Different Cluster Solution

```python
config = cluster_analysis_config(
    ...
    cluster_table="clust_labels_jaccard_wgc_pam_goldstd",  # Different clustering
    cluster_column="pam_k5",  # k=5 instead of k=7
    output_table_prefix="cluster_k5",
    ...
)
```

### Different Variables

```python
CONTINUOUS_VARS = ['total_wl_%', 'dietitian_visits', 'avg_days_between_measurements']
CATEGORICAL_VARS = ['instant_dropout', '15%_wl_achieved']
```

### Different Population Reference

```python
config = cluster_analysis_config(
    ...
    population_table="timetoevent_wgc_compl",  # Use WGC cohort as reference
    ...
)
```

## Troubleshooting

**Error: "Table not found"**
- Check that cluster_table name matches your database
- Verify cluster_column exists in that table

**Error: "Column not found"**
- Check that variables exist in outcome_table
- Use `df.columns.tolist()` to see available columns

**Warning: "Config file not found"**
- Check paths are relative to notebook location
- Use absolute paths if needed

**No plots generated**
- Check output_dir path is correct
- Verify you have write permissions
