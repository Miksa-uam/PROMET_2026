# Requirements Document: Cluster Trajectories Module

## Introduction

This specification defines the requirements for creating a unified `cluster_trajectories.py` module that consolidates cluster-based trajectory analysis functionality currently scattered across the notebook. The module will provide clean, reusable functions for visualizing and modeling weight loss trajectories across patient clusters.

## Glossary

- **Cluster Trajectory**: The pattern of weight change over time for patients within a specific cluster
- **Spaghetti Plot**: A visualization showing individual patient trajectories as thin lines with an overlaid smoothed mean trajectory
- **LMM (Linear Mixed Model)**: A statistical model that accounts for both fixed effects (cluster membership) and random effects (individual patient variation)
- **LOWESS Smoothing**: Locally weighted scatterplot smoothing technique used to create smooth trajectory curves
- **Mean Follow-up**: The average duration patients in a cluster were tracked
- **Length-Adjusted Trajectory**: A trajectory visualization that distinguishes between solid lines (within mean follow-up) and dashed lines (beyond mean follow-up)

## Requirements

### Requirement 1: Spaghetti Plot Generation

**User Story:** As a researcher, I want to generate publication-ready spaghetti plots showing weight loss trajectories by cluster, so that I can visualize how different patient groups respond to treatment over time.

#### Acceptance Criteria

1. THE System SHALL load cluster labels and measurement data from specified database tables
2. THE System SHALL calculate days from baseline for each patient measurement
3. THE System SHALL generate individual trajectory lines for each patient with configurable transparency
4. THE System SHALL compute and overlay smoothed mean trajectories using LOWESS smoothing
5. THE System SHALL create a multi-panel figure with one panel per cluster plus a population panel and overlay panel
6. THE System SHALL apply length-adjusted visualization where trajectories beyond mean follow-up are displayed as dashed lines
7. THE System SHALL include 95% confidence intervals around smoothed means
8. THE System SHALL support optional time cutoff filtering (e.g., 365-day cutoff)
9. THE System SHALL use consistent cluster colors across all visualizations
10. THE System SHALL display cluster statistics (n, average delta, mean follow-up) in panel titles

### Requirement 2: Configuration Management

**User Story:** As a researcher, I want to configure trajectory analysis parameters through a simple interface, so that I can easily adjust settings without modifying code.

#### Acceptance Criteria

1. THE System SHALL accept cluster configuration specifying algorithm, k value, and database paths
2. THE System SHALL support configurable smoothing parameters (LOWESS fraction)
3. THE System SHALL allow specification of time cutoffs for analysis
4. THE System SHALL accept custom color palettes for cluster visualization
5. THE System SHALL support configuration of figure dimensions and DPI
6. THE System SHALL validate configuration parameters before execution
7. THE System SHALL provide sensible defaults for all optional parameters

### Requirement 3: Linear Mixed Model Analysis

**User Story:** As a researcher, I want to fit linear mixed models to cluster trajectories, so that I can statistically compare trajectory slopes and intercepts across clusters.

#### Acceptance Criteria

1. THE System SHALL fit LMM with cluster as fixed effect and patient as random effect
2. THE System SHALL support both deviation coding (vs. grand mean) and reference coding (vs. specific cluster)
3. THE System SHALL use spline basis functions to model non-linear trajectories
4. THE System SHALL support optional covariate adjustment (age, sex, baseline BMI)
5. THE System SHALL generate model summary statistics and coefficient tables
6. THE System SHALL create predicted trajectory plots from fitted models
7. THE System SHALL report model convergence status
8. THE System SHALL calculate and display effects for omitted clusters in deviation coding

### Requirement 4: Data Integration

**User Story:** As a researcher, I want the module to seamlessly integrate with existing database structure, so that I can use it with minimal setup.

#### Acceptance Criteria

1. THE System SHALL load cluster labels from cluster database tables
2. THE System SHALL load measurement data from input database tables
3. THE System SHALL merge data on patient_id and medical_record_id
4. THE System SHALL handle missing data appropriately
5. THE System SHALL support filtering outliers based on cluster labels (-1 values)
6. THE System SHALL validate data integrity before analysis
7. THE System SHALL provide informative error messages for data issues

### Requirement 5: Visualization Quality

**User Story:** As a researcher, I want publication-quality visualizations, so that I can include them directly in manuscripts.

#### Acceptance Criteria

1. THE System SHALL generate high-resolution figures (configurable DPI)
2. THE System SHALL use consistent styling across all plots
3. THE System SHALL include proper axis labels with units
4. THE System SHALL display legends with clear cluster identification
5. THE System SHALL use color-blind friendly palettes by default
6. THE System SHALL support saving figures in multiple formats (PNG, SVG, PDF)
7. THE System SHALL apply professional grid styling and spacing
8. THE System SHALL ensure text is readable at publication sizes

### Requirement 6: Module Interface

**User Story:** As a researcher, I want a simple, intuitive API for trajectory analysis, so that I can generate analyses with minimal code in the notebook.

#### Acceptance Criteria

1. THE System SHALL provide a main function for spaghetti plot generation accepting cluster config
2. THE System SHALL provide a main function for LMM analysis accepting cluster config and model options
3. THE System SHALL return figure objects for further customization
4. THE System SHALL return model results as structured data
5. THE System SHALL support method chaining where appropriate
6. THE System SHALL provide clear docstrings for all public functions
7. THE System SHALL follow existing module patterns (cluster_descriptions.py, wgc_visualizations.py)

### Requirement 7: Performance and Scalability

**User Story:** As a researcher, I want trajectory analysis to complete in reasonable time, so that I can iterate on analyses efficiently.

#### Acceptance Criteria

1. THE System SHALL complete spaghetti plot generation for 7 clusters in under 30 seconds
2. THE System SHALL complete LMM fitting in under 2 minutes
3. THE System SHALL use vectorized operations where possible
4. THE System SHALL provide progress indicators for long-running operations
5. THE System SHALL handle datasets with up to 10,000 patients
6. THE System SHALL efficiently manage memory for large datasets

### Requirement 8: Error Handling and Validation

**User Story:** As a researcher, I want clear error messages when something goes wrong, so that I can quickly identify and fix issues.

#### Acceptance Criteria

1. THE System SHALL validate all input parameters before processing
2. THE System SHALL provide specific error messages for common issues
3. THE System SHALL handle database connection errors gracefully
4. THE System SHALL detect and report data quality issues
5. THE System SHALL validate cluster column existence in database
6. THE System SHALL check for sufficient data before fitting models
7. THE System SHALL warn about convergence issues in LMM fitting
