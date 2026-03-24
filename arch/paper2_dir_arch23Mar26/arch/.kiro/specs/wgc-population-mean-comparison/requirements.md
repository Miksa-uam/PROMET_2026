# Requirements Document

## Introduction

This feature adds a new comparative analysis functionality to the existing descriptive analysis pipeline. The new functionality will create tables that compare individual Weight Gain Cause (WGC) category groups against the population mean for all variables currently defined in the pipeline. This extends the current stratified comparison approach by providing population-level baseline comparisons for each WGC group.

## Requirements

### Requirement 1

**User Story:** As a researcher analyzing weight gain causes, I want to compare each WGC group against the population mean on all variables, so that I can identify how each specific cause differs from the overall population baseline.

#### Acceptance Criteria

1. WHEN the comparative analysis pipeline runs THEN the system SHALL create a new table named '[input_cohort]_wgc_strt_vs_mean' in the input database
2. WHEN creating the table name THEN the system SHALL follow the existing naming convention (e.g., 'wgc_gen_compl_wgc_strt_vs_mean' for the wgc_gen_compl cohort)
3. WHEN processing variables THEN the system SHALL include all variables currently defined in the pipeline's row_order configuration
4. WHEN calculating population statistics THEN the system SHALL use the input cohort as the population baseline for mean and standard deviation calculations

### Requirement 2

**User Story:** As a researcher, I want the new table to have a specific column structure that matches the existing stratified comparison tables, so that I can easily interpret and compare results across different analysis types.

#### Acceptance Criteria

1. WHEN creating the table structure THEN the system SHALL include a 'variables' column as the first column containing variable names
2. WHEN adding population statistics THEN the system SHALL include a 'population mean (±std) or n (%)' column as the second column
3. WHEN processing each WGC group THEN the system SHALL create three consecutive columns: variable mean/n values, p-value, and FDR-corrected p-value
4. WHEN iterating through WGC groups THEN the system SHALL repeat the three-column pattern (mean/n, p-value, FDR-corrected p-value) for each WGC category
5. WHEN formatting continuous variables THEN the system SHALL display as 'mean ± std' format
6. WHEN formatting categorical variables THEN the system SHALL display as 'n (%)' format

### Requirement 3

**User Story:** As a researcher, I want statistical comparisons between each WGC group and the population mean using the existing statistical methods, so that I can identify statistically significant differences.

#### Acceptance Criteria

1. WHEN comparing continuous variables THEN the system SHALL use the existing Welch's t-test method for statistical comparison
2. WHEN comparing categorical variables THEN the system SHALL use the existing Chi-squared test method for statistical comparison
3. WHEN calculating p-values THEN the system SHALL store raw p-values in the p-value columns
4. WHEN FDR correction is enabled THEN the system SHALL apply Benjamini-Hochberg correction to all p-values within the table
5. WHEN FDR correction is disabled THEN the system SHALL not create the FDR-corrected p-value column

### Requirement 4

**User Story:** As a researcher, I want the new functionality to integrate seamlessly with the existing pipeline configuration, so that I can enable or disable it without disrupting current analyses.

#### Acceptance Criteria

1. WHEN the pipeline runs THEN the system SHALL execute the new WGC vs population mean analysis alongside existing demographic and WGC stratification analyses
2. WHEN the descriptive_comparisons_config is used THEN the system SHALL support the new analysis without requiring configuration changes
3. WHEN FDR correction is configured THEN the system SHALL apply the same FDR correction setting to the new analysis as used for other stratifications
4. WHEN the analysis completes THEN the system SHALL log the table creation and provide user feedback consistent with existing analyses

### Requirement 5

**User Story:** As a researcher, I want the new analysis to handle edge cases and data quality issues gracefully, so that the pipeline remains robust and reliable.

#### Acceptance Criteria

1. WHEN a variable has insufficient data for statistical testing THEN the system SHALL return NaN for p-values and handle gracefully
2. WHEN a WGC group is empty or has insufficient data THEN the system SHALL display appropriate placeholder values (e.g., "N/A") 
3. WHEN variables are missing from the dataset THEN the system SHALL skip those variables and continue processing
4. WHEN database operations fail THEN the system SHALL provide informative error messages and not crash the entire pipeline
5. WHEN the analysis encounters statistical test failures THEN the system SHALL log warnings and continue with remaining comparisons