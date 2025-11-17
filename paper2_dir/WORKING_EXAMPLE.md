# Complete Working Example: Table + Heatmap + Lollipop

This is a **complete, tested workflow** for generating statistical tables, heatmaps, and lollipop plots with significance markers.

## Full Working Code

```python
from cluster_descriptions import *

# =============================================================================
# STEP 1: Load Data
# =============================================================================

cluster_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
)

# =============================================================================
# STEP 2: Define Variables to Analyze
# =============================================================================

# Weight gain causes
wgc_vars = [
    "womens_health_and_pregnancy",
    "mental_health",
    "family_issues",
    "medication_disease_injury",
    "physical_inactivity",
    "eating_habits",
    "schedule",
    "smoking_cessation",
    "treatment_discontinuation_or_relapse",
    "pandemic",
    "lifestyle_circumstances",
    "none_of_above",
]

# Clinical/demographic variables for lollipop
clinical_vars = [
    "sex_f", 
    "age",
    "baseline_bmi",
    "baseline_weight_kg",
    "10%_wl_achieved", 
    "days_to_10%_wl",
    "60d_dropout",
    "total_followup_days",
]

# ALL variables for statistical analysis
all_vars = wgc_vars + clinical_vars

# =============================================================================
# STEP 3: Run Statistical Analysis (generates table)
# =============================================================================

results_df = analyze_cluster_vs_population(
    cluster_df,
    variables=all_vars,  # Analyze ALL variables
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="cluster_analysis",
    fdr_correction=True
)

# =============================================================================
# STEP 4: Extract P-Values for Lollipop Plot
# =============================================================================

# Use helper function to extract p-values
pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
    results_df, 
    clinical_vars,  # Variables to extract p-values for
    cluster_df
)

# =============================================================================
# STEP 5: Generate Heatmap (WGC variables only)
# =============================================================================

plot_cluster_heatmap(
    results_df,
    output_filename='wgc_heatmap.png',
    output_dir='../outputs/',
    variables=wgc_vars,  # Only WGC variables in heatmap
    title='Weight Gain Cause Prevalence by Cluster',
    xlabel='Patient Clusters',
    ylabel='Weight Gain Causes',
    cbar_label='Prevalence (%)'
)

# =============================================================================
# STEP 6: Generate Lollipop Plot with Significance (Clinical variables)
# =============================================================================

plot_cluster_lollipop(
    cluster_df,
    variables=clinical_vars,  # Clinical variables for lollipop
    output_filename='lollipop_with_sig.png',
    output_dir='../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr,
    alpha=0.05,
    title='Clinical Differences by Cluster',
    xlabel='Percent Change from Population Mean (%)'
)

print("\n✓ All visualizations complete!")
```

## Key Points

### 1. Variable Organization
- **`all_vars`**: ALL variables for statistical analysis
- **`wgc_vars`**: Subset for heatmap (weight gain causes)
- **`clinical_vars`**: Subset for lollipop plot (demographics, outcomes)

### 2. P-Value Extraction
The critical step that was missing from the quick reference:

```python
# Get cluster IDs from your data
clusters = sorted(cluster_df['cluster_id'].unique())

# Extract for each variable you want to plot
for var in clinical_vars:
    pvalues_raw[var] = {}
    pvalues_fdr[var] = {}
    
    var_row = results_df[results_df['Variable'] == var]
    
    for cluster_id in clusters:
        p_raw_col = f'Cluster {cluster_id}: p-value'
        p_fdr_col = f'Cluster {cluster_id}: p-value (FDR-corrected)'
        
        if p_raw_col in var_row.columns:
            pvalues_raw[var][cluster_id] = var_row[p_raw_col].values[0]
        if p_fdr_col in var_row.columns:
            pvalues_fdr[var][cluster_id] = var_row[p_fdr_col].values[0]
```

### 3. Workflow Summary

```
1. Load data → cluster_df
2. Define variable lists (all_vars, wgc_vars, clinical_vars)
3. Run analyze_cluster_vs_population(all_vars) → results_df
4. Extract p-values from results_df → pvalues_raw, pvalues_fdr
5. Generate heatmap(wgc_vars)
6. Generate lollipop(clinical_vars, pvalues)
```

## Troubleshooting

### Error: `NameError: name 'pvalues_raw' is not defined`
**Cause**: You didn't run Step 4 (p-value extraction)
**Fix**: Run the p-value extraction code before calling `plot_cluster_lollipop()`

### Error: `NameError: name 'cluster_df' is not defined`
**Cause**: You didn't run Step 1 (data loading)
**Fix**: Run `load_and_merge_cluster_data()` first

### Error: `NameError: name 'variables' is not defined`
**Cause**: Variable not defined in extraction loop
**Fix**: Use the actual variable list name (e.g., `clinical_vars`)

### Empty heatmap or lollipop
**Cause**: Variable names don't match between analysis and plotting
**Fix**: Ensure variable names in `wgc_vars`/`clinical_vars` match column names in `cluster_df`

## Minimal Example (Just Lollipop)

If you only want a lollipop plot with significance:

```python
from cluster_descriptions import *

# Load data
cluster_df = load_and_merge_cluster_data(...)

# Define variables
vars_to_plot = ['age', 'baseline_bmi', 'total_followup_days']

# Run analysis
results_df = analyze_cluster_vs_population(
    cluster_df, 
    variables=vars_to_plot,
    output_db_path='../outputs/results.db',
    output_table_name='analysis',
    fdr_correction=True
)

# Extract p-values
clusters = sorted(cluster_df['cluster_id'].unique())
pvalues_raw = {}
pvalues_fdr = {}

for var in vars_to_plot:
    pvalues_raw[var] = {}
    pvalues_fdr[var] = {}
    var_row = results_df[results_df['Variable'] == var]
    
    for cluster_id in clusters:
        p_raw_col = f'Cluster {cluster_id}: p-value'
        p_fdr_col = f'Cluster {cluster_id}: p-value (FDR-corrected)'
        
        if p_raw_col in var_row.columns:
            pvalues_raw[var][cluster_id] = var_row[p_raw_col].values[0]
        if p_fdr_col in var_row.columns:
            pvalues_fdr[var][cluster_id] = var_row[p_fdr_col].values[0]

# Plot
plot_cluster_lollipop(
    cluster_df,
    variables=vars_to_plot,
    output_filename='lollipop.png',
    output_dir='../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr
)
```
