# Visualization Pipeline Refactor - Requirements

## Introduction

This specification defines improvements to the existing visualization functions in `descriptive_visualizations.py` to support both WGC-based and cluster-based analyses. The refactor focuses on enhancing existing code rather than creating new modules, maintaining compatibility with the current workflow in `descriptive_comparisons.py` and the notebook.

## Glossary

- **WGC**: Weight Gain Causes - categorical binary variables (0/1) indicating specific reasons for weight gain (e.g., mental_health, eating_habits)
- **Cluster**: Patient groups identified through clustering algorithms, represented by integer IDs (0, 1, 2, ...)
- **Population**: The reference cohort used for comparison (typically the full dataset or mother cohort)
- **Cluster-focused analysis**: Analysis where patients are grouped by cluster ID, comparing each cluster to the population
- **WGC-focused analysis**: Analysis where patients are grouped by presence/absence of specific WGCs, comparing WGC groups to the population
- **Visualization System**: The existing functions in `descriptive_visualizations.py` (plot_distribution_comparison, plot_stacked_bar_comparison, plot_multi_lollipop, plot_forest)

## Requirements

### Requirement 1: Cluster Configuration Storage

**User Story:** As a researcher, I want cluster labels and colors stored in an external configuration file, so that I can maintain consistent cluster styling across all visualizations without hardcoding values.

#### Acceptance Criteria

1. THE system SHALL store cluster metadata in a JSON file at `scripts/cluster_config.json` with structure:
   ```json
   {
     "cluster_labels": {"0": "Cluster A", "1": "Cluster B", ...},
     "cluster_colors": {"0": "#FF6700", "1": "#1f77b4", ...}
   }
   ```

2. THE system SHALL provide a utility function `load_cluster_config(json_path: str) -> Dict` that loads cluster configuration

3. THE cluster configuration file SHALL be optional - WHEN the file does not exist, THE system SHALL use default numeric labels ("Cluster 0", "Cluster 1") and a default color palette

4. THE system SHALL validate that all cluster IDs in the data have corresponding entries in the configuration

5. THE cluster configuration SHALL follow the same pattern as `human_readable_variable_names.json` for consistency

### Requirement 2: Extend Existing Plot Functions for Cluster Support

**User Story:** As a researcher, I want to use the existing visualization functions with cluster data, so that I can generate cluster-based plots without rewriting my analysis pipeline.

#### Acceptance Criteria

1. THE existing functions in `descriptive_visualizations.py` SHALL be extended to accept an optional `group_col` parameter that specifies the grouping column name (e.g., "cluster_id" for cluster analysis, or WGC variable names for WGC analysis)

2. WHEN `group_col` is provided, THE functions SHALL use it to identify groups instead of assuming WGC structure

3. THE functions SHALL load cluster labels and colors from `cluster_config.json` WHEN the `group_col` indicates cluster-based analysis

4. THE functions SHALL maintain backward compatibility - WHEN called with existing parameters, THE functions SHALL behave identically to the current implementation

5. THE functions SHALL display figures in the notebook after saving them using `plt.show()` or returning the figure object

### Requirement 3: Data Structure Compatibility

**User Story:** As a researcher, I want the visualization functions to work with my existing data structures from `descriptive_comparisons.py`, so that I don't need to transform data.

#### Acceptance Criteria

1. FOR WGC-focused analysis, THE system SHALL accept DataFrames with:
   - Individual WGC columns as binary variables (0/1)
   - Outcome variables as continuous or categorical columns
   - The structure currently used in `descriptive_comparisons.py`

2. FOR cluster-focused analysis, THE system SHALL accept DataFrames with:
   - A `cluster_id` column containing integer cluster assignments (0, 1, 2, ...)
   - Outcome variables as continuous or categorical columns
   - The same structure as WGC data but with cluster_id instead of multiple WGC columns

3. THE system SHALL accept pre-calculated significance values as dictionaries:
   ```python
   significance_map_raw: Dict[Any, float]  # {group_id: p_value}
   significance_map_fdr: Dict[Any, float]  # {group_id: fdr_corrected_p_value}
   ```

4. THE system SHALL handle missing values by dropping them with `dropna()` as currently implemented

5. THE system SHALL NOT require data transformation utilities - existing data formats SHALL work directly

### Requirement 4: Statistical Test Integration

**User Story:** As a researcher, I want to understand where statistical tests are calculated and how they integrate with visualizations, so that I can properly interpret significance markers.

#### Acceptance Criteria

1. THE statistical tests SHALL be calculated in `descriptive_comparisons.py` using existing functions (`welchs_ttest`, `mann_whitney_u_test`, `categorical_pvalue`)

2. THE statistical tests SHALL compare each group (cluster or WGC) against the population mean/distribution

3. THE FDR correction SHALL be applied in `descriptive_comparisons.py` using `fdr_correction_utils.py` before passing results to visualization functions

4. THE visualization functions SHALL accept pre-calculated p-values via `significance_map_raw` and `significance_map_fdr` parameters

5. THE visualization functions SHALL annotate plots with significance markers:
   - Single asterisk (*) WHEN raw p-value < alpha AND FDR-corrected p-value >= alpha
   - Double asterisk (**) WHEN FDR-corrected p-value < alpha
   - No marker WHEN raw p-value >= alpha

6. THE significance annotation SHALL use the existing `_annotate_significance` helper function

7. THE alpha threshold SHALL default to 0.05 and be configurable via function parameter

### Requirement 5: Forest Plot Improvements

**User Story:** As a researcher, I want forest plots to display effect sizes and confidence intervals on a secondary y-axis, so that I can see exact values alongside the visual representation.

#### Acceptance Criteria

1. THE `plot_forest` function SHALL add a right y-axis that displays the effect size and 95% CI for each group

2. THE right y-axis labels SHALL format values as:
   - For risk ratios (RR): "RR: 1.23 [1.05-1.45]"
   - For risk differences (RD): "RD: 5.2% [2.1-8.3%]"

3. THE forest plot SHALL NOT display significance asterisks, as crossing the reference line (RR=1 or RD=0%) indicates significance

4. THE reference line SHALL be clearly marked:
   - Vertical line at x=1 for risk ratios
   - Vertical line at x=0 for risk differences

5. THE forest plot SHALL maintain the existing error bar visualization on the main plot area

### Requirement 6: Heatmap Implementation for Cluster-WGC Analysis

**User Story:** As a researcher, I want to visualize WGC prevalence across clusters as a heatmap, so that I can identify which weight gain causes are associated with specific patient clusters.

#### Acceptance Criteria

1. THE system SHALL implement a new function `plot_wgc_cluster_heatmap` that displays:
   - Columns: Cluster IDs (using labels from cluster_config.json)
   - Rows: WGC variables (using labels from human_readable_variable_names.json)
   - Cell colors: Percentage prevalence (0-100%) of each WGC in each cluster

2. THE heatmap SHALL annotate each cell with the percentage value (e.g., "26.5%")

3. THE heatmap SHALL optionally annotate cells with significance markers (* or **) WHEN significance data is provided

4. THE heatmap SHALL use a sequential colormap (e.g., "YlOrRd") with range 0-100%

5. THE heatmap SHALL integrate with the `wgc_vs_population_mean_analysis` function in `descriptive_comparisons.py` to obtain:
   - Prevalence percentages for each WGC in each cluster
   - Statistical test results comparing each cluster-WGC combination to population mean

6. THE heatmap SHALL be saved as high-resolution PNG (300 dpi) and displayed in the notebook after saving

### Requirement 7: Figure Display in Notebooks

**User Story:** As a researcher, I want generated figures to display in my Jupyter notebook after being saved, so that I can immediately review visualizations without opening files.

#### Acceptance Criteria

1. AFTER saving a figure, THE visualization functions SHALL display the figure in the notebook using `plt.show()`

2. THE display SHALL occur before calling `plt.close()` to ensure the figure is visible

3. THE existing print statements (e.g., "✓ Split-violin plot saved to: {path}") SHALL remain to confirm file saving

4. THE figure display SHALL work in Jupyter notebook environments without requiring additional configuration

5. THE display behavior SHALL be consistent across all visualization functions (plot_distribution_comparison, plot_stacked_bar_comparison, plot_multi_lollipop, plot_forest, plot_wgc_cluster_heatmap)

### Requirement 8: Error Handling

**User Story:** As a researcher, I want clear error messages when data or parameters are invalid, so that I can quickly fix issues without debugging code.

#### Acceptance Criteria

1. THE visualization functions SHALL validate that required columns exist in the DataFrame before plotting

2. THE functions SHALL raise descriptive exceptions for:
   - Missing required columns (e.g., "Column 'cluster_id' not found in DataFrame")
   - Empty DataFrames after dropping missing values
   - Invalid file paths for output directories
   - Mismatched cluster IDs between data and cluster_config.json

3. THE functions SHALL print warning messages (using `print()`) for:
   - Missing cluster configuration file (falling back to defaults)
   - Variables with all missing values (skipping the plot)
   - Empty groups after filtering

4. THE functions SHALL NOT use logging modules - simple print statements SHALL be used for user feedback

5. THE error messages SHALL be concise and actionable, indicating what went wrong and how to fix it

### Requirement 9: Configurable Labels for All Figures

**User Story:** As a researcher, I want all axis labels, titles, and legends to use human-readable names from configuration files, so that figures are publication-ready without manual editing.

#### Acceptance Criteria

1. THE visualization functions SHALL load variable labels from `human_readable_variable_names.json` using the existing `load_name_map` and `get_nice_name` functions

2. THE visualization functions SHALL load cluster labels from `cluster_config.json` WHEN generating cluster-based plots

3. THE functions SHALL apply labels to:
   - Plot titles
   - X-axis and Y-axis labels
   - Legend entries
   - Tick labels
   - Heatmap row and column labels

4. THE functions SHALL handle missing labels gracefully by falling back to the raw variable/cluster name with underscores replaced by spaces and title case applied

5. THE label loading SHALL occur at the beginning of each visualization function to ensure consistency

### Requirement 10: Minimal Code Changes and Simplicity

**User Story:** As a researcher, I want the refactored code to be simple and maintainable, so that I can understand and modify it without extensive documentation.

#### Acceptance Criteria

1. THE refactor SHALL modify only the existing files:
   - `descriptive_visualizations.py` (extend existing functions)
   - `descriptive_comparisons.py` (add cluster support to analysis functions)
   - Create `cluster_config.json` (new configuration file)

2. THE refactor SHALL NOT create new Python modules or complex class hierarchies

3. THE code SHALL remain linear and procedural, following the existing style in `descriptive_visualizations.py`

4. THE functions SHALL have clear, simple signatures with explicit parameters rather than complex configuration objects

5. THE code SHALL avoid over-engineering - simple, direct implementations SHALL be preferred over abstract, flexible architectures

6. THE refactor SHALL maintain the existing workflow:
   - Data preparation in `descriptive_comparisons.py`
   - Statistical tests in `descriptive_comparisons.py`
   - Visualization in `descriptive_visualizations.py`
   - Notebook calls visualization functions with results from analysis functions
