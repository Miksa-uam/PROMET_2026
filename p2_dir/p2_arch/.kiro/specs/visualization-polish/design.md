# Design Document

## Overview

This design document outlines targeted enhancements to the `cluster_descriptions.py` module to improve publication readiness and visual polish. The approach focuses on minimal, surgical modifications to existing working code while maintaining backward compatibility. All changes are localized to specific functions without requiring architectural refactoring.

### Design Principles

1. **Minimal Change Philosophy**: Modify only what's necessary to meet requirements
2. **Backward Compatibility**: All existing function signatures remain unchanged
3. **Configuration-Driven**: Leverage existing JSON configuration files for labels and colors
4. **Localized Modifications**: Changes are isolated to specific functions without cascading effects
5. **Publication-Ready Output**: All visualizations and tables should be immediately usable in publications

### Scope

**In Scope:**
- Table header formatting and display enhancements
- Heatmap visual improvements (population column, cluster-specific colors, ordering)
- Lollipop plot refinements (significance markers, separators)
- Forest plot spacing adjustments
- Stacked bar and violin plot legend/label positioning
- Violin plot cluster-specific coloring

**Out of Scope:**
- Architectural refactoring or module reorganization
- Changes to statistical testing logic
- New visualization types
- Database schema modifications
- Changes to configuration file formats

## Architecture

### Current Architecture

The module follows a functional architecture with clear separation of concerns:

```
cluster_descriptions.py
├── Configuration Loading (load_name_map, load_cluster_config, get_nice_name, etc.)
├── Statistical Testing (mann_whitney_u_test, chi_squared_test, calculate_cluster_pvalues, etc.)
├── Visualization Functions (cluster_continuous_distributions, cluster_categorical_distributions, etc.)
├── Analysis Functions (analyze_cluster_vs_population)
└── Convenience Functions (load_and_merge_cluster_data, extract_pvalues_for_lollipop)
```

### Design Strategy

All enhancements will be implemented as **localized modifications** within existing functions. No new modules or architectural layers are required.

**Modification Approach:**
1. **Table Functions**: Modify column header generation and display logic in `analyze_cluster_vs_population()`
2. **Heatmap Functions**: Enhance `plot_cluster_heatmap()` with population column and cluster-specific color scales
3. **Lollipop Functions**: Adjust marker positioning and separator styling in `plot_cluster_lollipop()`
4. **Forest Functions**: Modify subplot spacing in `_plot_single_forest()`
5. **Bar/Violin Functions**: Adjust legend positioning and label alignment in respective functions

## Components and Interfaces

### Component 1: Table Header Formatting

**Location**: `analyze_cluster_vs_population()` function

**Current Behavior:**
- First column: `'Population Mean (±SD) or N (%)'`
- Cluster columns: `f'Cluster {cluster_id}: Mean/N'`
- P-value columns: `f'Cluster {cluster_id}: p-value'`

**Enhanced Behavior:**
- First column: `'Whole population: Mean (±SD) / N (%)'`
- Cluster columns: `f'{cluster_label}: Mean (±SD) / N (%)'` where `cluster_label` comes from config
- P-value columns: `f'{cluster_label}: p-value'`

**Implementation Details:**
```python
# Load cluster config at function start
cluster_config = load_cluster_config(cluster_config_path)

# When building column headers:
for cluster_id in clusters:
    cluster_label = get_cluster_label(cluster_id, cluster_config)
    row[f'{cluster_label}: Mean (±SD) / N (%)'] = ...
    row[f'{cluster_label}: p-value'] = ...
```

**Downstream Impact:**
- `plot_cluster_heatmap()` must parse new column format
- Column extraction logic must handle cluster labels instead of numeric IDs

**Interface Changes:**
- Add optional parameter: `cluster_config_path: str = 'cluster_config.json'`
- No breaking changes to existing calls (parameter is optional with default)

### Component 2: Table Variable Name Display

**Location**: `analyze_cluster_vs_population()` function

**Current Behavior:**
- Variables displayed with raw database column names

**Enhanced Behavior:**
- Variables displayed using `get_nice_name(variable, name_map)`

**Implementation Details:**
```python
# When building each variable row:
row = {'Variable': get_nice_name(variable, name_map)}
```

**Interface Changes:** None (uses existing `name_map_path` parameter)

### Component 3: Table Display Output

**Location**: `analyze_cluster_vs_population()` function

**Current Behavior:**
- Only prints confirmation messages
- No DataFrame display

**Enhanced Behavior:**
- Prints full DataFrame to console after saving
- Configures pandas display options for wide tables

**Implementation Details:**
```python
# Before returning results_df:
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print("\n" + "="*80)
print("RESULTS TABLE (Detailed)")
print("="*80)
print(results_df.to_string(index=False))
print("="*80 + "\n")
```

**Interface Changes:** None

### Component 4: Heatmap Population Column

**Location**: `plot_cluster_heatmap()` function

**Current Behavior:**
- Heatmap shows only cluster columns
- No population reference column

**Enhanced Behavior:**
- First column shows "Whole population" with overall prevalence
- Uses white-to-red color scale for population column

**Implementation Details:**
```python
# After extracting prevalence_data, add population data:
for wgc_var in variables:
    # Calculate population prevalence from results_df
    pop_row = results_df[results_df['Variable'] == wgc_var]
    pop_value = extract_percentage_from_string(pop_row['Whole population: Mean (±SD) / N (%)'])
    
    prevalence_data.insert(0, {
        'wgc_variable': wgc_var,
        'cluster_id': 'Population',
        'prevalence_%': pop_value,
        'n': pop_n
    })

# Modify pivot to include population:
matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='prevalence_%')
# Ensure 'Population' is first column
cols = ['Population'] + [c for c in matrix.columns if c != 'Population']
matrix = matrix[cols]
```

**Helper Function Required:**
```python
def _extract_percentage_from_table_cell(cell_value: str) -> float:
    """Extract percentage from 'N (X.X%)' format."""
    match = re.search(r'\((\d+\.?\d*)%\)', cell_value)
    return float(match.group(1)) if match else np.nan
```

**Interface Changes:** None

### Component 5: Heatmap Cluster-Specific Colors

**Location**: `plot_cluster_heatmap()` function

**Current Behavior:**
- Single colormap (YlOrRd) for entire heatmap
- Colorbar displayed on right side

**Enhanced Behavior:**
- Population column: white-to-red gradient
- Each cluster column: white-to-[cluster-color] gradient
- No colorbar displayed

**Implementation Details:**

This requires creating custom colormaps for each column and using matplotlib's `imshow` with multiple axes instead of seaborn's heatmap.

```python
from matplotlib.colors import LinearSegmentedColormap

# Create figure with subplots for each column
fig, axes = plt.subplots(1, len(matrix.columns), 
                         figsize=(...), 
                         gridspec_kw={'wspace': 0.01})

# For population column:
pop_cmap = LinearSegmentedColormap.from_list('pop', ['white', 'red'])
axes[0].imshow(matrix.iloc[:, 0:1], cmap=pop_cmap, vmin=0, vmax=100, aspect='auto')

# For each cluster column:
for i, cluster_id in enumerate(cluster_columns):
    cluster_color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
    cluster_cmap = LinearSegmentedColormap.from_list(f'cluster_{i}', ['white', cluster_color])
    axes[i+1].imshow(matrix.iloc[:, i+1:i+2], cmap=cluster_cmap, vmin=0, vmax=100, aspect='auto')
    
# Add annotations manually to each subplot
# Add row labels to leftmost axis only
# Add column labels to all axes
```

**Alternative Simpler Approach:**

Keep seaborn heatmap but create a custom colormap that interpolates based on column position:

```python
# This is complex and may not achieve desired effect
# Recommend the subplot approach above
```

**Interface Changes:** None

### Component 6: Heatmap Column Label Centering

**Location**: `plot_cluster_heatmap()` function

**Current Behavior:**
- Default seaborn alignment (may not be centered)

**Enhanced Behavior:**
- Column labels explicitly centered

**Implementation Details:**
```python
# After creating heatmap:
ax.set_xticklabels(cluster_labels, ha='center', rotation=45)
```

**Interface Changes:** None

### Component 7: Heatmap Variable Ordering

**Location**: `plot_cluster_heatmap()` function

**Current Behavior:**
- Variables may be sorted alphabetically during pivot

**Enhanced Behavior:**
- Variables displayed in exact order provided in `variables` parameter

**Implementation Details:**
```python
# After pivot, reindex rows to match input order:
matrix = matrix.reindex(variables)
n_matrix = n_matrix.reindex(variables)

# When creating annotations, iterate in input order:
for i, wgc in enumerate(variables):  # Use variables list, not matrix.index
    ...
```

**Interface Changes:** None

### Component 8: Lollipop Significance Marker Positioning

**Location**: `plot_cluster_lollipop()` function

**Current Behavior:**
- Markers positioned with fixed offset from point

**Enhanced Behavior:**
- Markers positioned on same horizontal line as lollipop head
- Minimal spacing after marker

**Implementation Details:**
```python
# Current code:
if sig_text:
    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02
    ax.text(x_val + x_offset, y_val, sig_text, ha='left', va='center', 
           fontsize=16, color='black', weight='bold')

# Enhanced code (reduce offset):
if sig_text:
    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01  # Reduced from 0.02
    ax.text(x_val + x_offset, y_val, sig_text, ha='left', va='center', 
           fontsize=16, color='black', weight='bold')
```

**Interface Changes:** None

### Component 9: Lollipop Variable Separator Enhancement

**Location**: `plot_cluster_lollipop()` function

**Current Behavior:**
- Separator lines drawn with linewidth=0.5

**Enhanced Behavior:**
- Variable separators drawn with bold linewidth (2.0)
- Cluster separators within variables remain thin (0.5)

**Implementation Details:**
```python
# Current code:
if len(variables_in_data) > 1 and var != variables_in_data[-1]:
    y_pos += 0.5
    ax.axhline(y=y_pos - 0.75, color='grey', linestyle='-', linewidth=0.5)

# Enhanced code:
if len(variables_in_data) > 1 and var != variables_in_data[-1]:
    y_pos += 0.5
    ax.axhline(y=y_pos - 0.75, color='grey', linestyle='-', linewidth=2.0)  # Bold
```

**Interface Changes:** None

### Component 10: Forest Plot Axis Spacing

**Location**: `_plot_single_forest()` function

**Current Behavior:**
- Default subplot spacing with `wspace=0.05`

**Enhanced Behavior:**
- Minimal spacing between main plot and effect size axis

**Implementation Details:**
```python
# Current code:
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(plot_data) * 0.5)), 
                               gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})

# Enhanced code:
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(plot_data) * 0.5)), 
                               gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.01})  # Reduced
```

**Interface Changes:** None

### Component 11: Stacked Bar Legend Positioning

**Location**: `_plot_single_stacked_bar()` function

**Current Behavior:**
- Legend positioned at `loc='upper right'`
- Y-axis limit set to 120

**Enhanced Behavior:**
- Legend positioned higher to avoid overlap
- Y-axis limit adjusted if needed

**Implementation Details:**
```python
# Current code:
ax.set_ylim(0, 120)
ax.legend(loc='upper right')

# Enhanced code:
ax.set_ylim(0, 120)
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.08))  # Move up
```

**Interface Changes:** None

### Component 12: Stacked Bar Y-Axis Range

**Location**: `_plot_single_stacked_bar()` function

**Current Behavior:**
- Y-axis shows 0-120 to accommodate labels above bars

**Enhanced Behavior:**
- Y-axis shows 0-100 for percentage scale
- Elements above 100% remain visible through plot margins

**Implementation Details:**
```python
# Current code:
ax.set_ylim(0, 120)

# Enhanced code:
ax.set_ylim(0, 100)
# Adjust figure margins to ensure top elements visible:
plt.subplots_adjust(top=0.85)  # Add space at top
```

**Note**: This may require adjusting where sample sizes and significance markers are positioned (currently at y=102 and y=108).

**Alternative Approach:**
```python
# Keep y-axis at 0-100 but use clip_on=False for text elements:
ax.text(i, 102, f'n={row["n"]}', ha='center', va='bottom', 
        fontweight='bold', clip_on=False)
```

**Interface Changes:** None

### Component 13: Violin Plot Legend Positioning

**Location**: `_plot_single_violin()` function

**Current Behavior:**
- Legend positioning not explicitly set (uses default)

**Enhanced Behavior:**
- Legend positioned above plot area

**Implementation Details:**
```python
# Add after creating violin plot:
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2)
```

**Interface Changes:** None

### Component 14: Violin Plot Label Centering

**Location**: `_plot_single_violin()` function

**Current Behavior:**
- X-axis labels rotated 45 degrees
- Alignment may not be centered

**Enhanced Behavior:**
- Labels explicitly centered on violins

**Implementation Details:**
```python
# Current code:
ax.tick_params(axis='x', rotation=45)

# Enhanced code:
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
```

**Interface Changes:** None

### Component 15: Violin Plot Cluster-Specific Colors

**Location**: `_plot_single_violin()` function

**Current Behavior:**
- Split violin with fixed colors: Population (POPULATION_COLOR), Cluster (CLUSTER_COLOR)
- Same cluster color used for all clusters

**Enhanced Behavior:**
- Each cluster gets its own subplot with cluster-specific color
- Population side maintains consistent color across all subplots

**Implementation Details:**

This requires a significant change from split violins to separate subplots:

```python
# Current approach: single plot with split violins
# New approach: one subplot per cluster

fig, axes = plt.subplots(1, n_clusters, figsize=(n_clusters * 4, 8), sharey=True)

for i, cluster_id in enumerate(clusters):
    ax = axes[i] if n_clusters > 1 else axes
    cluster_label = get_cluster_label(cluster_id, cluster_config)
    cluster_color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
    
    # Prepare data for this cluster
    plot_data_list = []
    # Population data (left side)
    for val in population_data:
        plot_data_list.append({
            'value': val,
            'status': 'Population'
        })
    # Cluster data (right side)
    for val in cluster_df[cluster_df[cluster_col] == cluster_id][variable].dropna():
        plot_data_list.append({
            'value': val,
            'status': 'Cluster'
        })
    
    plot_df = pd.DataFrame(plot_data_list)
    
    # Plot split violin with cluster-specific color
    palette = {'Population': POPULATION_COLOR, 'Cluster': cluster_color}
    sns.violinplot(data=plot_df, y='value', hue='status', split=True, 
                   inner='quart', palette=palette, ax=ax)
    
    # Add significance markers
    if sig_raw or sig_fdr:
        p_raw = sig_raw.get(cluster_id) if sig_raw else None
        p_fdr = sig_fdr.get(cluster_id) if sig_fdr else None
        y_max = plot_df['value'].max()
        _annotate_significance(ax, 0, y_max * 1.02, p_raw, p_fdr, alpha)
    
    # Set title and labels
    ax.set_title(cluster_label, fontsize=14, weight='bold')
    ax.set_xlabel('')
    if i == 0:
        ax.set_ylabel(plot_ylabel, fontsize=14)
    else:
        ax.set_ylabel('')
```

**Alternative Simpler Approach:**

Keep single plot but manually color each violin pair:

```python
# This is complex with seaborn's violinplot
# May require using matplotlib's violin directly or post-processing seaborn output
```

**Recommendation**: Use subplot approach for cleaner implementation and better visual separation.

**Interface Changes:** None (visual change only)

## Data Models

### Configuration Data Structures

**cluster_config.json:**
```json
{
  "cluster_labels": {
    "0": "Male-dominant, inactive",
    "1": "Women's health",
    ...
  },
  "cluster_colors": {
    "0": "#FF6700",
    "1": "#1f77b4",
    ...
  }
}
```

**human_readable_variable_names.json:**
```json
{
  "sex_f": "Sex (% of females)",
  "age": "Age (years)",
  ...
}
```

### DataFrame Structures

**Results DataFrame (analyze_cluster_vs_population output):**

Current structure:
```
| Variable | Population Mean (±SD) or N (%) | Cluster 0: Mean/N | Cluster 0: p-value | ... |
```

Enhanced structure:
```
| Variable | Whole population: Mean (±SD) / N (%) | Male-dominant, inactive: Mean (±SD) / N (%) | Male-dominant, inactive: p-value | ... |
```

**Prevalence DataFrame (for heatmap):**

Current structure:
```
| wgc_variable | cluster_id | prevalence_% | n |
```

Enhanced structure (with population):
```
| wgc_variable | cluster_id | prevalence_% | n |
| var1         | Population | 45.2         | 123 |
| var1         | 0          | 52.1         | 45  |
| var1         | 1          | 38.3         | 32  |
```

## Error Handling

### Graceful Degradation

All enhancements include fallback behavior when configuration is unavailable:

1. **Missing Cluster Config:**
   - Fallback to default labels: `f'Cluster {cluster_id}'`
   - Fallback to default colors: `DEFAULT_PALETTE[cluster_id % len(DEFAULT_PALETTE)]`

2. **Missing Name Map:**
   - Fallback to formatted variable name: `variable.replace('_', ' ').title()`

3. **Invalid Table Cell Format:**
   - Use regex with error handling to extract percentages
   - Return `np.nan` if parsing fails

4. **Empty Data:**
   - Existing validation checks remain in place
   - Print warnings and skip visualization

### Error Messages

Maintain existing warning patterns:
```python
print(f"⚠️ Warning: [specific issue]. Using default/Skipping.")
```

### Validation

Add validation for new functionality:

```python
# Validate column header format when parsing
def _parse_cluster_column_header(header: str) -> Optional[str]:
    """Extract cluster identifier from column header."""
    # Try new format: "[Cluster Name]: Mean (±SD) / N (%)"
    match = re.match(r'^(.+?):\s*Mean', header)
    if match:
        return match.group(1)
    # Fallback to old format: "Cluster X: Mean/N"
    match = re.match(r'^Cluster\s+(\d+):', header)
    if match:
        return f'Cluster {match.group(1)}'
    return None
```

## Testing Strategy

### Manual Testing Approach

Since this is a visualization module, testing will be primarily manual and visual:

1. **Table Output Testing:**
   - Run `analyze_cluster_vs_population()` with various datasets
   - Verify column headers use cluster labels from config
   - Verify variable names use human-readable format
   - Verify DataFrame prints to console with proper formatting
   - Test with missing config files (verify fallback behavior)

2. **Heatmap Testing:**
   - Generate heatmaps with various variable counts
   - Verify population column appears first
   - Verify cluster-specific color gradients (visual inspection)
   - Verify column labels are centered
   - Verify variable order matches input list
   - Test with missing cluster colors (verify fallback)

3. **Lollipop Testing:**
   - Generate plots with multiple variables and clusters
   - Verify significance markers align with lollipop heads
   - Verify variable separators are bold
   - Verify cluster separators are thin
   - Test with single variable (no separators)

4. **Forest Plot Testing:**
   - Generate RR and RD plots
   - Verify minimal spacing between axes
   - Verify effect size axis is close to main plot

5. **Stacked Bar Testing:**
   - Generate plots with various cluster counts
   - Verify legend positioned above plot
   - Verify y-axis shows 0-100%
   - Verify sample sizes and significance markers visible

6. **Violin Plot Testing:**
   - Generate plots with multiple clusters
   - Verify legend positioned above plot
   - Verify labels centered on violins
   - Verify cluster-specific colors applied
   - Verify population side maintains consistent color

### Test Cases

**Test Case 1: Full Configuration Available**
- Cluster config with all labels and colors
- Name map with all variables
- Expected: All enhancements active, publication-ready output

**Test Case 2: Missing Configuration**
- No cluster config file
- No name map file
- Expected: Fallback to defaults, functional output

**Test Case 3: Partial Configuration**
- Cluster config with labels but no colors
- Name map with some variables missing
- Expected: Mix of configured and default values

**Test Case 4: Edge Cases**
- Single cluster
- Single variable
- Empty data for some variables
- Expected: Graceful handling, appropriate warnings

### Validation Checklist

For each modified function:
- [ ] Backward compatibility maintained (existing calls work)
- [ ] Configuration loading works correctly
- [ ] Fallback behavior works when config missing
- [ ] Visual output matches requirements
- [ ] No errors or warnings for valid inputs
- [ ] Appropriate warnings for invalid inputs
- [ ] Output is publication-ready

### Integration Testing

Test complete workflow:
1. Load cluster data
2. Run `analyze_cluster_vs_population()` → verify table output
3. Generate all visualization types → verify visual consistency
4. Verify cluster colors consistent across all plots
5. Verify cluster labels consistent across all plots

## Implementation Notes

### Order of Implementation

Recommended implementation order (from simplest to most complex):

1. **Phase 1: Table Enhancements** (Requirements 1-3)
   - Modify `analyze_cluster_vs_population()`
   - Add cluster config parameter
   - Update column headers
   - Add display output
   - Test thoroughly before proceeding

2. **Phase 2: Simple Visual Adjustments** (Requirements 8-14)
   - Lollipop marker positioning and separators
   - Forest plot spacing
   - Stacked bar legend and y-axis
   - Violin plot legend and labels
   - These are small, isolated changes

3. **Phase 3: Heatmap Enhancements** (Requirements 4-7)
   - Add population column
   - Implement cluster-specific color scales (most complex)
   - Add column centering
   - Fix variable ordering

4. **Phase 4: Violin Plot Colors** (Requirement 15)
   - Implement subplot approach for cluster-specific colors
   - Most significant structural change

### Code Organization

All changes remain within `cluster_descriptions.py`. No new files needed.

Consider adding helper functions:
- `_extract_percentage_from_table_cell(cell_value: str) -> float`
- `_parse_cluster_column_header(header: str) -> Optional[str]`
- `_create_cluster_colormap(cluster_color: str) -> LinearSegmentedColormap`

### Dependencies

No new dependencies required. All functionality uses existing imports:
- `matplotlib.pyplot`
- `seaborn`
- `pandas`
- `numpy`
- `matplotlib.colors.LinearSegmentedColormap` (already available)

### Performance Considerations

All changes have minimal performance impact:
- Configuration loading: One-time at function start
- Color map creation: Negligible overhead
- Subplot creation: Slightly more memory for violin plots, but acceptable

### Backward Compatibility

All changes maintain backward compatibility:
- New parameters are optional with defaults
- Existing function signatures unchanged
- Output format enhanced but not breaking
- Configuration files optional (fallback behavior)

## Design Decisions and Rationales

### Decision 1: Subplot Approach for Violin Plot Colors

**Options Considered:**
1. Modify seaborn's violin plot output (complex, fragile)
2. Use matplotlib's violin directly (requires reimplementing seaborn features)
3. Create separate subplots for each cluster (clean, maintainable)

**Decision:** Use subplot approach (Option 3)

**Rationale:**
- Cleaner visual separation between clusters
- Easier to implement and maintain
- Better control over individual cluster styling
- Aligns with publication-quality visualization standards

### Decision 2: Heatmap Color Implementation

**Options Considered:**
1. Single heatmap with custom colormap (complex, may not achieve desired effect)
2. Multiple subplots with individual colormaps (clean, flexible)

**Decision:** Use subplot approach (Option 2)

**Rationale:**
- Allows true per-column color scales
- Easier to add population column with different scale
- More control over annotations and styling
- Standard matplotlib approach

### Decision 3: Table Header Format

**Options Considered:**
1. Keep numeric IDs, add separate label column
2. Replace numeric IDs with labels in headers
3. Use both: "Cluster 0 (Male-dominant, inactive)"

**Decision:** Replace with labels only (Option 2)

**Rationale:**
- Cleaner, more publication-ready
- Reduces redundancy
- Cluster IDs are internal implementation details
- Labels are more meaningful to readers

### Decision 4: Y-Axis Range for Stacked Bars

**Options Considered:**
1. Keep 0-120 range (current)
2. Change to 0-100, use clip_on=False for top elements
3. Change to 0-100, adjust figure margins

**Decision:** Use clip_on=False approach (Option 2)

**Rationale:**
- Maintains accurate percentage scale
- Elements remain visible without distorting axis
- Standard matplotlib technique for overflow elements
- No need to adjust figure-level margins

### Decision 5: Configuration Parameter Placement

**Options Considered:**
1. Add cluster_config_path to all functions
2. Load config globally at module level
3. Add only to functions that need it

**Decision:** Add parameter only where needed (Option 3)

**Rationale:**
- Minimal API changes
- Functions remain independent
- No global state
- Backward compatible (optional parameters)
