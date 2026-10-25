# Requirements Document

## Introduction

This feature will create a comprehensive descriptive visualizations pipeline that serves as a "Swiss Army knife" for simple, exploratory, and descriptive visualizations. The pipeline will generate both risk ratio and risk difference forest plots for multiple outcomes including 10% weight loss achievement and 60-day dropout. The pipeline will be implemented as a single, self-contained script that maximally reuses functionality from the existing descriptive_comparisons.py module and provides a clean, simplified notebook interface.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to calculate risk ratios and risk differences for multiple outcomes across different weight gain causes, so that I can comprehensively assess the impact of each cause on various treatment outcomes.

#### Acceptance Criteria

1. WHEN the script is executed THEN the system SHALL load data from a configurable input table
2. WHEN processing outcomes THEN the system SHALL create 2x2 contingency tables for each weight gain cause using both 10%_wl_achieved and 60d_dropout columns
3. WHEN calculating risk ratios THEN the system SHALL compute the risk in the "cause present" group divided by the risk in the "cause absent" group
4. WHEN calculating risk differences THEN the system SHALL compute the absolute difference in risk between "cause present" and "cause absent" groups
5. WHEN calculating confidence intervals THEN the system SHALL compute 95% confidence intervals for both risk ratios and risk differences
6. WHEN any contingency table cell has count < 5 THEN the system SHALL use Fisher's exact test instead of Chi-squared test

### Requirement 2

**User Story:** As a researcher, I want to perform appropriate statistical testing with multiple comparison correction, so that I can ensure the validity of my findings across multiple weight gain causes.

#### Acceptance Criteria

1. WHEN performing statistical tests THEN the system SHALL reuse the categorical_pvalue function from descriptive_comparisons.py for Chi-squared tests
2. WHEN any contingency table cell has count < 5 THEN the system SHALL use Fisher's exact test from scipy.stats
3. WHEN all statistical tests are complete THEN the system SHALL apply Benjamini-Hochberg FDR correction using the existing fdr_correction_utils.apply_fdr_correction function
4. WHEN FDR correction is applied THEN the system SHALL store both raw and corrected p-values for each weight gain cause

### Requirement 3

**User Story:** As a researcher, I want to generate publication-ready forest plots for both risk ratios and risk differences, so that I can visualize effect sizes and confidence intervals in clear, standardized formats.

#### Acceptance Criteria

1. WHEN creating risk ratio forest plots THEN the system SHALL use a logarithmic x-axis scale
2. WHEN creating risk difference forest plots THEN the system SHALL use a linear x-axis scale
3. WHEN plotting data points THEN the system SHALL display each weight gain cause as a horizontal point with confidence interval bars
4. WHEN displaying reference lines THEN the system SHALL draw vertical lines at RR = 1.0 for risk ratios and RD = 0.0 for risk differences
5. WHEN labeling axes THEN the system SHALL use appropriate labels for each plot type and weight gain cause names for y-axes
6. WHEN displaying confidence intervals THEN the system SHALL rely on confidence intervals crossing reference lines to indicate statistical significance without additional markers

### Requirement 4

**User Story:** As a researcher, I want the pipeline to integrate with existing project infrastructure and provide organized output management, so that I can maintain consistency with current workflows while keeping visualizations well-organized.

#### Acceptance Criteria

1. WHEN loading configuration THEN the system SHALL use the existing paper12_config module for database paths and settings
2. WHEN performing FDR correction THEN the system SHALL reuse the existing fdr_correction_utils.apply_fdr_correction function
3. WHEN performing statistical tests THEN the system SHALL reuse the categorical_pvalue function from descriptive_comparisons.py
4. WHEN saving outputs THEN the system SHALL save all results to the ../outputs/descriptive_visualizations/ directory
5. WHEN identifying weight gain causes THEN the system SHALL reuse the get_cause_cols function from descriptive_comparisons.py
6. WHEN creating output directory THEN the system SHALL automatically create ../outputs/descriptive_visualizations/ if it doesn't exist

### Requirement 5

**User Story:** As a researcher, I want simple error handling with print statements, so that I can easily debug issues during development and execution.

#### Acceptance Criteria

1. WHEN the script encounters missing data THEN the system SHALL print warning messages and handle gracefully without crashing
2. WHEN database connections fail THEN the system SHALL print clear error messages with troubleshooting guidance
3. WHEN statistical calculations fail THEN the system SHALL print the specific cause and continue with remaining analyses
4. WHEN generating plots THEN the system SHALL validate data completeness and print warnings about any missing or invalid values
5. WHEN the pipeline completes THEN the system SHALL print a summary of results including sample sizes and number of significant findings

### Requirement 6

**User Story:** As a researcher, I want a clean and simplified notebook interface for calling the visualization pipeline, so that I can easily generate multiple types of plots with minimal code complexity.

#### Acceptance Criteria

1. WHEN called from a notebook THEN the system SHALL provide a simple, clean interface with minimal required parameters
2. WHEN executed THEN the system SHALL generate both risk ratio and risk difference plots for all specified outcomes
3. WHEN outputs are generated THEN the system SHALL save forest plots, summary tables, and statistical results in organized subdirectories
4. WHEN the analysis is repeated THEN the system SHALL produce identical results given the same input data
5. WHEN placed in the notebook THEN the system SHALL be callable after the "#### I/2.2. Risk ratio/risk difference analyses" section with a clean, professional appearance