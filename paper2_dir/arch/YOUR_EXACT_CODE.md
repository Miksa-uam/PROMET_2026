# Your Exact Use Case - Copy & Paste Ready

This is the **exact code** for your scenario: table + heatmap + lollipop with your specific variables.

## Complete Working Code (Copy This!)

```python
from cluster_descriptions import *

# =============================================================================
# LOAD DATA
# =============================================================================

cluster_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
)

# =============================================================================
# DEFINE VARIABLES
# =============================================================================

# Weight gain causes (for heatmap)
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

# Clinical variables (for lollipop)
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

# ALL variables for analysis
all_vars = wgc_vars + clinical_vars

# =============================================================================
# RUN STATISTICAL ANALYSIS (Creates table in database)
# =============================================================================

results_df = analyze_cluster_vs_population(
    cluster_df,
    variables=all_vars,
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="test_clusts",
    fdr_correction=True
)

# =============================================================================
# EXTRACT P-VALUES (For lollipop significance markers)
# =============================================================================

pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
    results_df, 
    clinical_vars,
    cluster_df
)

# =============================================================================
# GENERATE HEATMAP (WGC variables only)
# =============================================================================

plot_cluster_heatmap(
    results_df,
    output_filename='wgc_heatmap.png',
    output_dir='../outputs/',
    variables=wgc_vars,
    title='Weight Gain Cause Prevalence by Cluster',
    xlabel='Patient Clusters',
    ylabel='Weight Gain Causes',
    cbar_label='Prevalence (%)'
)

# =============================================================================
# GENERATE LOLLIPOP PLOT (Clinical variables with significance)
# =============================================================================

plot_cluster_lollipop(
    cluster_df,
    variables=clinical_vars,
    output_filename='lollipop_with_sig.png',
    output_dir='../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr,
    alpha=0.05,
    title='Clinical Differences by Cluster',
    xlabel='Percent Change from Population Mean (%)'
)

print("\n✅ All done! Check ../outputs/ for your plots.")
```

## What This Does

1. **Loads your data** from the SQLite databases
2. **Analyzes ALL variables** (WGC + clinical) and saves results to database
3. **Extracts p-values** automatically using the helper function
4. **Generates heatmap** showing WGC prevalence by cluster
5. **Generates lollipop plot** showing clinical differences with significance markers

## Key Changes from Your Original Code

### ❌ Your Original (Didn't Work)
```python
plot_cluster_lollipop(
    cluster_df,
    variables=['age', 'bmi', 'follow_up_days'],
    ...
    pvalues_raw=pvalues_raw,  # ❌ Not defined!
    pvalues_fdr=pvalues_fdr,  # ❌ Not defined!
)
```

### ✅ Fixed Version
```python
# First extract p-values
pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
    results_df, 
    clinical_vars,
    cluster_df
)

# Then plot
plot_cluster_lollipop(
    cluster_df,
    variables=clinical_vars,
    ...
    pvalues_raw=pvalues_raw,  # ✅ Now defined!
    pvalues_fdr=pvalues_fdr,  # ✅ Now defined!
)
```

## Customization Options

### Change Lollipop Variables
```python
# Only plot specific variables
clinical_vars = ["age", "baseline_bmi", "total_followup_days"]
```

### Change Heatmap Variables
```python
# Only show specific WGCs
wgc_vars = ["medication_disease_injury", "eating_habits", "physical_inactivity"]
```

### Custom Labels
```python
plot_cluster_lollipop(
    cluster_df,
    variables=clinical_vars,
    output_filename='lollipop_with_sig.png',
    output_dir='../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr,
    alpha=0.05,
    title='Key Clinical Differences Across Patient Clusters',  # Custom title
    xlabel='Relative Difference from Population Mean (%)'  # Custom x-label
)
```

## Troubleshooting

### "NameError: name 'pvalues_raw' is not defined"
**Fix**: Make sure you run the `extract_pvalues_for_lollipop()` line BEFORE calling `plot_cluster_lollipop()`

### "Variable 'X' not found in results_df"
**Fix**: Make sure the variable is in `all_vars` when you call `analyze_cluster_vs_population()`

### Empty plots
**Fix**: Check that variable names match exactly between your lists and the actual column names in `cluster_df`

## Run Order (Important!)

```
1. Load data (load_and_merge_cluster_data)
2. Define variables (wgc_vars, clinical_vars, all_vars)
3. Run analysis (analyze_cluster_vs_population)
4. Extract p-values (extract_pvalues_for_lollipop)
5. Generate plots (plot_cluster_heatmap, plot_cluster_lollipop)
```

**Don't skip steps or run them out of order!**
