# Requirements Document

## Introduction

This specification defines enhancements to the cluster visualization module to improve publication readiness, interpretability, and visual polish. The focus is on minimal, targeted improvements to existing working code rather than comprehensive refactoring. All changes maintain backward compatibility and the current functional architecture.

## Glossary

- **Cluster Descriptions Module**: The `cluster_descriptions.py` module providing statistical analysis and visualization functions
- **Results Table**: DataFrame output from `analyze_cluster_vs_population()` containing statistical comparisons
- **Cluster Config**: JSON file containing cluster labels and colors
- **Name Map**: JSON file containing human-readable variable names
- **WGC**: Weight Gain Cause variables
- **FDR**: False Discovery Rate correction

## Requirements

### Requirement 1: Table Column Header Enhancement

**User Story:** As a researcher, I want publication-ready table headers that use cluster names from configuration, so that tables are immediately interpretable without manual editing.

#### Acceptance Criteria

1. WHEN the Analysis Module generates results tables, THE System SHALL format the first column header as "Whole population: Mean (±SD) / N (%)"
2. WHEN the Analysis Module generates results tables, THE System SHALL format cluster column headers as "[Cluster Name]: Mean (±SD) / N (%)" WHERE [Cluster Name] is retrieved from cluster configuration
3. WHEN the Analysis Module generates detailed results tables, THE System SHALL format p-value column headers as "[Cluster Name]: p-value" WHERE [Cluster Name] is retrieved from cluster configuration
4. WHEN cluster configuration is unavailable, THE System SHALL use default format "Cluster [ID]: Mean (±SD) / N (%)"
5. WHEN downstream functions process results tables, THE System SHALL correctly parse data from columns with updated headers

### Requirement 2: Table Variable Name Display

**User Story:** As a researcher, I want variable names displayed in human-readable format in tables, so that results are publication-ready without manual formatting.

#### Acceptance Criteria

1. WHEN the Analysis Module generates results tables, THE System SHALL display variable names using human-readable mappings from the name map file
2. WHEN a variable name is not found in the name map, THE System SHALL display the variable name with underscores replaced by spaces and title case applied
3. WHEN the Variable column is populated, THE System SHALL maintain consistent formatting across all rows

### Requirement 3: Table Output Display

**User Story:** As a researcher, I want to see the generated table in my notebook output, so that I can immediately verify results without opening database files.

#### Acceptance Criteria

1. WHEN the Analysis Module saves results tables to database, THE System SHALL print the DataFrame to console output
2. WHEN the System prints the DataFrame, THE System SHALL configure pandas display options to prevent line wrapping
3. WHEN the System prints the DataFrame, THE System SHALL display all columns on a single line regardless of table width

### Requirement 4: Heatmap Population Column Addition

**User Story:** As a researcher, I want a "Whole population" column in heatmaps showing overall prevalence, so that I can compare cluster-specific values to the population baseline.

#### Acceptance Criteria

1. WHEN the Heatmap Function generates a heatmap, THE System SHALL add a first column labeled "Whole population"
2. WHEN the Heatmap Function populates the population column, THE System SHALL display prevalence data matching the first column of the results table
3. WHEN the Heatmap Function colors the population column, THE System SHALL use a white-to-red color scale
4. WHEN the Heatmap Function positions the population column, THE System SHALL place it as the leftmost column before cluster columns

### Requirement 5: Heatmap Cluster-Specific Color Scales

**User Story:** As a researcher, I want cluster columns in heatmaps to use cluster-specific colors, so that visual consistency is maintained across all visualizations.

#### Acceptance Criteria

1. WHEN the Heatmap Function generates cluster columns, THE System SHALL apply cluster-specific color scales WHERE cluster colors are defined in configuration
2. WHEN the Heatmap Function applies cluster color scales, THE System SHALL use white-to-[cluster-color] gradients for each cluster column
3. WHEN cluster colors are not available in configuration, THE System SHALL use the default yellow-to-red scale for all columns
4. WHEN the Heatmap Function applies color scales, THE System SHALL remove the colorbar from the right side

### Requirement 6: Heatmap Column Label Centering

**User Story:** As a researcher, I want column labels centered on their respective columns in heatmaps, so that labels are clearly associated with their data.

#### Acceptance Criteria

1. WHEN the Heatmap Function displays column labels, THE System SHALL center each label horizontally on its corresponding column
2. WHEN the Heatmap Function rotates column labels, THE System SHALL maintain centered alignment

### Requirement 7: Heatmap Variable Ordering

**User Story:** As a researcher, I want heatmap rows ordered as specified in my notebook call, so that variables appear in my preferred sequence.

#### Acceptance Criteria

1. WHEN the Heatmap Function receives a variables list, THE System SHALL display rows in the exact order provided
2. WHEN the Heatmap Function sorts data, THE System SHALL preserve the input variable order rather than applying alphabetical sorting

### Requirement 8: Lollipop Significance Marker Positioning

**User Story:** As a researcher, I want significance asterisks positioned consistently after lollipop markers, so that significance is immediately clear without visual confusion.

#### Acceptance Criteria

1. WHEN the Lollipop Function adds significance markers, THE System SHALL position asterisks on the same horizontal line as the lollipop marker
2. WHEN the Lollipop Function adds significance markers, THE System SHALL place asterisks immediately after the lollipop head with minimal spacing
3. WHEN the Lollipop Function adds significance markers, THE System SHALL ensure markers do not overlap with lollipop elements

### Requirement 9: Lollipop Variable Separator Enhancement

**User Story:** As a researcher, I want bold gridlines separating different variables in lollipop plots, so that I can easily distinguish which lollipops belong to which variable.

#### Acceptance Criteria

1. WHEN the Lollipop Function draws separator lines between variables, THE System SHALL render these lines with bold linewidth
2. WHEN the Lollipop Function draws separator lines, THE System SHALL maintain standard linewidth for cluster separators within variables
3. WHEN the Lollipop Function has only one variable, THE System SHALL not draw separator lines

### Requirement 10: Forest Plot Axis Spacing

**User Story:** As a researcher, I want the effect size axis positioned close to the forest plot, so that the visualization is compact and easy to read.

#### Acceptance Criteria

1. WHEN the Forest Plot Function creates dual y-axes, THE System SHALL minimize horizontal spacing between the main plot and the effect size axis
2. WHEN the Forest Plot Function adjusts subplot spacing, THE System SHALL ensure the effect size axis is visually connected to the plot
3. WHEN the Forest Plot Function renders the plot, THE System SHALL eliminate unnecessary whitespace between axes

### Requirement 11: Stacked Bar Legend Positioning

**User Story:** As a researcher, I want the legend box positioned above the plot area, so that it does not overlap with visualization elements.

#### Acceptance Criteria

1. WHEN the Stacked Bar Function displays the legend, THE System SHALL position the legend box one grid unit higher than current position
2. WHEN the Stacked Bar Function positions the legend, THE System SHALL ensure no overlap with bars or significance markers
3. WHEN the Stacked Bar Function adjusts y-axis limits, THE System SHALL account for legend positioning

### Requirement 12: Stacked Bar Y-Axis Range

**User Story:** As a researcher, I want the y-axis to display only up to 100%, so that the scale accurately represents percentage data without unnecessary extension.

#### Acceptance Criteria

1. WHEN the Stacked Bar Function sets y-axis limits, THE System SHALL set the maximum visible value to 100%
2. WHEN the Stacked Bar Function adds elements above 100%, THE System SHALL ensure these elements remain visible through appropriate plot margins
3. WHEN the Stacked Bar Function displays percentage ticks, THE System SHALL show values from 0% to 100% only

### Requirement 13: Violin Plot Legend Positioning

**User Story:** As a researcher, I want the legend box positioned above the violin plot area, so that it does not overlap with distributions.

#### Acceptance Criteria

1. WHEN the Violin Plot Function displays the legend, THE System SHALL position the legend box one grid unit higher than current position
2. WHEN the Violin Plot Function positions the legend, THE System SHALL ensure no overlap with violin distributions or significance markers

### Requirement 14: Violin Plot Label Centering

**User Story:** As a researcher, I want cluster labels centered on their respective violins, so that labels are clearly associated with their distributions.

#### Acceptance Criteria

1. WHEN the Violin Plot Function displays x-axis labels, THE System SHALL center each label horizontally on its corresponding violin
2. WHEN the Violin Plot Function rotates labels, THE System SHALL maintain centered alignment

### Requirement 15: Violin Plot Cluster-Specific Colors

**User Story:** As a researcher, I want cluster-specific colors applied to the cluster side of split violins, so that visual consistency is maintained across all visualizations.

#### Acceptance Criteria

1. WHEN the Violin Plot Function generates split violins, THE System SHALL apply cluster-specific colors to the cluster side WHERE cluster colors are defined in configuration
2. WHEN the Violin Plot Function applies colors, THE System SHALL maintain the population color for the population side
3. WHEN cluster colors are not available, THE System SHALL use default colors for cluster sides
4. WHEN the Violin Plot Function applies cluster colors, THE System SHALL create a separate colored violin for each cluster using its configured color
