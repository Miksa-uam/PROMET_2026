# Cluster Descriptions Module - Major Enhancement Summary

## Version 2.0 - Comprehensive Visualization Configurability

### Three Major Enhancements Implemented

#### 1. Forest Plot Function ✓
**New Function**: `plot_cluster_forest()`

- Calculates **Risk Ratios (RR)** and **Risk Differences (RD)** for binary outcomes
- Compares each cluster to entire clustered population
- Displays:
  - Cluster labels (from config) on **left y-axis**
  - Effect size with 95% CI on **right y-axis** (e.g., "RD: 5.0% [2.0-8.0%]")
  - Units on x-axis (Risk Difference or Risk Ratio)
  - Cluster colors from config
  - Red dashed reference line (0 for RD, 1 for RR)
- Supports `effect_type='both'` to generate both RR and RD plots

**Helper Function**: `calculate_risk_metrics()`
- Calculates risk ratios, risk differences, and 95% CIs
- Uses log transformation for RR confidence intervals
- Returns DataFrame with all metrics

#### 2. Lollipop Significance Markers ✓
**Enhanced Function**: `plot_cluster_lollipop()`

- Added parameters:
  - `pvalues_raw`: Dict mapping {variable: {cluster_id: p_value}}
  - `pvalues_fdr`: Dict mapping {variable: {cluster_id: p_value}}
  - `alpha`: Significance threshold (default: 0.05)
- Displays asterisks on lollipop markers:
  - `*` for raw p < 0.05
  - `**` for FDR-corrected p < 0.05
- Links directly to `analyze_cluster_vs_population()` results

#### 3. Configurable Labels (Option A2) ✓
**Applied to ALL visualization functions**:

##### Global + Per-Variable Override Pattern
```python
# Example: Stacked bar with global and per-variable configs
cluster_categorical_distributions(
    cluster_df, 
    ["sex_f", "10%_wl_achieved"], 
    "../outputs/",
    # Global defaults
    title="Distribution of {variable}",
    ylabel="Percentage (%)",
    xlabel="Groups",
    legend_labels={'achieved': 'Yes', 'not_achieved': 'No'},
    # Per-variable overrides
    variable_configs={
        'sex_f': {
            'title': 'Sex Distribution',
            'legend_labels': {'achieved': 'Female', 'not_achieved': 'Male'}
        }
    }
)
```

##### Functions Enhanced:
1. **`cluster_continuous_distributions()`** (Violin plots)
   - Parameters: `title`, `ylabel`, `xlabel`, `variable_configs`
   - Supports `{variable}` placeholder in title/ylabel

2. **`cluster_categorical_distributions()`** (Stacked bar plots)
   - Parameters: `title`, `ylabel`, `xlabel`, `legend_labels`, `variable_configs`
   - Per-variable legend label overrides

3. **`plot_cluster_lollipop()`**
   - Parameters: `title`, `ylabel`, `xlabel`
   - Plus significance markers (see #2)

4. **`plot_cluster_heatmap()`**
   - Parameters: `title`, `ylabel`, `xlabel`, `cbar_label`

5. **`plot_cluster_forest()`** (NEW)
   - Parameters: `title_rr`, `title_rd`, `ylabel`, `xlabel_rr`, `xlabel_rd`

### Key Design Decisions

1. **Option A2 Pattern**: Individual parameters + per-variable overrides
   - ✓ IDE autocomplete works
   - ✓ Type hints preserved
   - ✓ Backward compatible (all optional)
   - ✓ Flexible for complex scenarios

2. **Significance Integration**: Lollipop plot links to statistical analysis
   - Pass p-value dicts from `analyze_cluster_vs_population()`
   - Consistent asterisk notation across all plots

3. **Forest Plot Design**: Matches reference implementation
   - Dual y-axes (labels left, CIs right)
   - Cluster colors applied to error bars
   - Both RR and RD in single function call

### Usage Examples

#### Forest Plot
```python
plot_cluster_forest(
    cluster_df,
    outcome_variable='achieved_10pct_wl',
    output_dir='../outputs/forest/',
    effect_type='both',  # Generate both RR and RD
    title_rd='Risk Differences: 10% Weight Loss Achievement'
)
```

#### Lollipop with Significance
```python
# First, run statistical analysis
results_df = analyze_cluster_vs_population(
    cluster_df, variables, output_db, output_table,
    fdr_correction=True
)

# Extract p-values for lollipop
pvalues_raw = {}
pvalues_fdr = {}
for var in variables:
    pvalues_raw[var] = {cluster_id: p_val for cluster_id, p_val in ...}
    pvalues_fdr[var] = {cluster_id: p_val for cluster_id, p_val in ...}

# Plot with significance markers
plot_cluster_lollipop(
    cluster_df, variables, 'lollipop.png', '../outputs/',
    pvalues_raw=pvalues_raw,
    pvalues_fdr=pvalues_fdr,
    title='Clinical Differences by Cluster'
)
```

#### Categorical with Custom Labels
```python
cluster_categorical_distributions(
    cluster_df,
    ["sex_f", "10%_wl_achieved", "60d_dropout"],
    "../outputs/cat_test",
    calculate_significance=True,
    fdr_correction=True,
    variable_configs={
        'sex_f': {
            'title': 'Sex Distribution by Cluster',
            'legend_labels': {'achieved': 'Female', 'not_achieved': 'Male'}
        },
        '10%_wl_achieved': {
            'title': '10% Weight Loss Achievement',
            'legend_labels': {'achieved': 'Achieved', 'not_achieved': 'Not Achieved'}
        }
    }
)
```

### Backward Compatibility

All enhancements are **100% backward compatible**:
- All new parameters are optional
- Default behavior unchanged
- Existing notebook code will work without modification

### Testing Recommendations

1. Test forest plot with binary outcome
2. Test lollipop with p-values from `analyze_cluster_vs_population()`
3. Test per-variable configs with mixed scenarios
4. Verify cluster colors and labels from config JSON

### Files Modified

- `scripts/cluster_descriptions.py` - All enhancements (Version 1.1 → 2.0)

### Next Steps

1. Update notebook examples to demonstrate new features
2. Test with real data
3. Consider adding forest plot to `analyze_cluster_vs_population()` workflow
