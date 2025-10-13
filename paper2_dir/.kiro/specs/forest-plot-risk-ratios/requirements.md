# Requirements Document

## Introduction

This feature will create a forest plot pipeline that visualizes risk ratios for achieving 10% weight loss by weight gain causes. The pipeline will be implemented as a single, self-contained script that maximally reuses functionality from the existing descriptive_comparisons.py module. The script will accept configurable input table names and be callable from a notebook cell, using the existing FDR correction utilities and statistical functions.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to calculate risk ratios for 10% weight loss achievement across different weight gain causes, so that I can quantify the relative likelihood of success for each cause group.

#### Acceptance Criteria

1. WHEN the script is executed THEN the system SHALL load data from a configurable input table (default: timetoevent_wgc_compl)
2. WHEN processing weight gain causes THEN the system SHALL create 2x2 contingency tables for each of the 12 weight gain causes using the 10%_wl_achieved column
3. WHEN calculating risk ratios THEN the system SHALL compute the risk in the "cause present" group divided by the risk in the "cause absent" group
4. WHEN calculating confidence intervals THEN the system SHALL compute 95% confidence intervals using the standard error of the log risk ratio
5. WHEN any contingency table cell has count < 5 THEN the system SHALL use Fisher's exact test instead of Chi-squared test

### Requirement 2

**User Story:** As a researcher, I want to perform appropriate statistical testing with multiple comparison correction, so that I can ensure the validity of my findings across multiple weight gain causes.

#### Acceptance Criteria

1. WHEN performing statistical tests THEN the system SHALL reuse the categorical_pvalue function from descriptive_comparisons.py for Chi-squared tests
2. WHEN any contingency table cell has count < 5 THEN the system SHALL use Fisher's exact test from scipy.stats
3. WHEN all statistical tests are complete THEN the system SHALL apply Benjamini-Hochberg FDR correction using the existing fdr_correction_utils.apply_fdr_correction function
4. WHEN FDR correction is applied THEN the system SHALL store both raw and corrected p-values for each weight gain cause

### Requirement 3

**User Story:** As a researcher, I want to generate a publication-ready forest plot, so that I can visualize the risk ratios and confidence intervals in a clear, standardized format.

#### Acceptance Criteria

1. WHEN creating the forest plot THEN the system SHALL use a logarithmic x-axis scale for risk ratios
2. WHEN plotting data points THEN the system SHALL display each weight gain cause as a horizontal point with confidence interval bars
3. WHEN displaying the reference line THEN the system SHALL draw a vertical line at RR = 1.0 to indicate no effect
4. WHEN labeling axes THEN the system SHALL use "Risk Ratio (RR)" for the x-axis and weight gain cause names for the y-axis
5. WHEN indicating significance THEN the system SHALL add visual markers (asterisks) for statistically significant results after FDR correction

### Requirement 4

**User Story:** As a researcher, I want the pipeline to integrate with existing project infrastructure, so that I can maintain consistency with current analysis workflows and configurations.

#### Acceptance Criteria

1. WHEN loading configuration THEN the system SHALL use the existing paper12_config module for database paths and settings
2. WHEN performing FDR correction THEN the system SHALL reuse the existing fdr_correction_utils.apply_fdr_correction function
3. WHEN performing statistical tests THEN the system SHALL reuse the categorical_pvalue function from descriptive_comparisons.py
4. WHEN saving outputs THEN the system SHALL save results to the ../outputs directory following existing project conventions
5. WHEN identifying weight gain causes THEN the system SHALL reuse the get_cause_cols function from descriptive_comparisons.py

### Requirement 5

**User Story:** As a researcher, I want simple error handling with print statements, so that I can easily debug issues during development and execution.

#### Acceptance Criteria

1. WHEN the script encounters missing data THEN the system SHALL print warning messages and handle gracefully without crashing
2. WHEN database connections fail THEN the system SHALL print clear error messages with troubleshooting guidance
3. WHEN statistical calculations fail THEN the system SHALL print the specific cause and continue with remaining analyses
4. WHEN generating plots THEN the system SHALL validate data completeness and print warnings about any missing or invalid values
5. WHEN the pipeline completes THEN the system SHALL print a summary of results including sample sizes and number of significant findings

### Requirement 6

**User Story:** As a researcher, I want the script to be callable from a notebook cell with configurable parameters, so that I can integrate it seamlessly into my existing analysis workflow.

#### Acceptance Criteria

1. WHEN called from a notebook THEN the system SHALL accept configurable input table name as a parameter
2. WHEN executed THEN the system SHALL complete the entire pipeline from data loading to plot generation without manual intervention
3. WHEN outputs are generated THEN the system SHALL save both the forest plot image and a summary table of results
4. WHEN the analysis is repeated THEN the system SHALL produce identical results given the same input data
5. WHEN placed in the notebook THEN the system SHALL be callable after the "#### I/2.2. Risk ratio/risk difference analyses" section