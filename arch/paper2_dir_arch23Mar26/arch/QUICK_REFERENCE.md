# Quick Reference: New Visualization Features


## 2. Lollipop Plots with Significance Markers

### Complete Working Example
```python
from cluster_descriptions import *

# Define variables to analyze
variables = ['age', 'baseline_bmi', 'total_followup_days']

# Step 1: Run Statistical Analysis
results_df = analyze_cluster_vs_population(
    cluster_df,
    variables=variables,
    output_db_path='../outputs/results.db',
    output_table_name='cluster_analysis',
    fdr_correction=True
)

# Step 2: Extract P-Values
# Get cluster IDs from your data
clusters = sorted(cluster_df['cluster_id'].unique())

# Initialize dictionaries
pvalues_raw = {}
pvalues_fdr = {}

# Extract for each variable
for var in variables:
    pvalues_raw[var] = {}
    pvalues_fdr[var] = {}
    
    # Get row for this variable
    var_row = results_df[results_df['Variable'] == var]
    
    if var_row.empty:
        continue
    
    # Extract p-values for each cluster
    for cluster_id in clusters:
        p_raw_col = f'Cluster {cluster_id}: p-value'
        p_fdr_col = f'Cluster {cluster_id}: p-value (FDR-corrected)'
        
        if p_raw_col in var_row.columns:
            pvalues_raw[var][cluster_id] = var_row[p_raw_col].values[0]
        if p_fdr_col in var_row.columns:
            pvalues_fdr[var][cluster_id] = var_row[p_fdr_col].values[0]

# Step 3: Plot with Significance
plot_cluster_lollipop(
    cluster_df,
    variables=variables,
    output_filename='lollipop_with_sig.png',
    output_dir='../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr,
    alpha=0.05,
    title='Clinical Differences by Cluster',
    xlabel='Percent Change from Population Mean (%)'
)
```


### Heatmaps
```python
from cluster_descriptions import *

# Define WGC variables
wgc_vars = ['wgc_medication', 'wgc_medical', 'wgc_lifestyle']

# Run analysis (if not already done)
results_df = analyze_cluster_vs_population(
    cluster_df,
    variables=wgc_vars,
    output_db_path='../outputs/results.db',
    output_table_name='wgc_analysis',
    fdr_correction=True
)

# Generate heatmap
plot_cluster_heatmap(
    results_df,
    output_filename='wgc_heatmap.png',
    output_dir='../outputs/',
    variables=wgc_vars,  # Note: parameter is 'variables', not 'wgc_variables'
    title='Weight Gain Cause Prevalence by Cluster',
    xlabel='Patient Clusters',
    ylabel='Weight Gain Causes',
    cbar_label='Prevalence (%)'
)
```

## Placeholder Support

Use `{variable}` placeholder in titles and labels:
- `title='Distribution of {variable}'` → "Distribution of Age"
- `ylabel='{variable} (units)'` → "Age (units)"

## Legend Label Keys

For stacked bar plots, use these keys in `legend_labels`:
- `'achieved'`: Label for class 1 / positive outcome
- `'not_achieved'`: Label for class 0 / negative outcome

Example:
```python
legend_labels={
    'achieved': 'Female',      # For sex_f variable
    'not_achieved': 'Male'
}
```
## Tips

3. **Placeholders**: Use `{variable}` for dynamic titles
4. **Significance**: Link lollipop to `analyze_cluster_vs_population()` results
5. **Forest plots**: Use `effect_type='both'` to see RR and RD together
