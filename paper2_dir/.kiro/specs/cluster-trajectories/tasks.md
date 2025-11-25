# Implementation Plan: Cluster Trajectories Module

## Task 1: Project Setup and Configuration

Create the foundational structure for the cluster trajectories module.

- [ ] 1.1 Create `scripts/cluster_trajectories.py` with module docstring and imports
  - Import required libraries: pandas, numpy, matplotlib, seaborn, sqlite3, statsmodels
  - Add module-level docstring explaining purpose and usage
  - Define module-level constants (DEFAULT_COLORS, DEFAULT_FIGURE_SIZE, etc.)
  - _Requirements: 6.1, 6.6_

- [ ] 1.2 Extend `scripts/cluster_config.json` with trajectory analysis settings
  - Add `trajectory_analysis` section with default parameters
  - Add `lmm_analysis` section with model configuration
  - Validate JSON structure
  - _Requirements: 2.1, 2.7_

- [ ] 1.3 Create output directory structure
  - Implement `ensure_output_dir()` function
  - Create subdirectories for spaghetti plots and LMM results
  - _Requirements: 2.1_

## Task 2: Configuration Management

Implement configuration loading and validation.

- [ ] 2.1 Create `TrajectoryConfig` dataclass
  - Define all configuration fields with types and defaults
  - Add validation methods for each field
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [ ] 2.2 Create `LMMConfig` dataclass
  - Define LMM-specific configuration fields
  - Add validation for model specification options
  - _Requirements: 3.2, 3.4_

- [ ] 2.3 Implement `load_trajectory_config()` function
  - Load and parse cluster_config.json
  - Create TrajectoryConfig and LMMConfig objects
  - Handle missing or invalid configuration gracefully
  - _Requirements: 2.1, 2.6, 8.1, 8.2_

- [ ] 2.4 Implement `validate_config()` function
  - Check database file existence
  - Validate parameter ranges (smoothing_frac, cutoff_days, etc.)
  - Verify color palette has sufficient colors
  - _Requirements: 2.6, 8.1, 8.5_

## Task 3: Data Loading and Preparation

Implement functions to load and prepare trajectory data.

- [ ] 3.1 Implement `load_cluster_labels()` function
  - Connect to cluster database
  - Load specified cluster table
  - Validate required columns exist
  - Return DataFrame with patient_id, medical_record_id, cluster_id
  - _Requirements: 4.1, 4.6, 8.3, 8.5_

- [ ] 3.2 Implement `load_measurements_for_patients()` function
  - Connect to measurements database
  - Create temporary table with cluster patient IDs
  - Join to load only relevant measurements
  - Validate required columns exist
  - _Requirements: 4.2, 4.6, 8.3_

- [ ] 3.3 Implement `calculate_days_from_baseline()` function
  - Convert measurement_date to datetime
  - Group by patient_id and medical_record_id
  - Calculate days from first measurement for each patient
  - Handle missing dates appropriately
  - _Requirements: 4.4, 8.4_

- [ ] 3.4 Implement `merge_with_clusters()` function
  - Merge measurements with cluster labels on patient identifiers
  - Validate merge success (no unexpected data loss)
  - Return combined DataFrame
  - _Requirements: 4.3, 4.6_

- [ ] 3.5 Implement `create_fixed_time_cutoff_data()` function
  - Filter data to specified cutoff_days
  - Preserve all patient identifiers
  - Return filtered DataFrame
  - _Requirements: 1.8_

## Task 4: Trajectory Statistics and Smoothing

Implement functions to calculate trajectory statistics and smoothing.

- [ ] 4.1 Implement cluster statistics calculation
  - Calculate mean follow-up time per cluster
  - Calculate average weight change (delta) per cluster
  - Calculate patient count per cluster
  - _Requirements: 1.10_

- [ ] 4.2 Implement LOWESS smoothing function
  - Apply statsmodels LOWESS with configurable fraction
  - Handle edge cases (insufficient data points)
  - Return smoothed trajectory and confidence intervals
  - _Requirements: 1.4, 1.7_

- [ ] 4.3 Implement length-adjusted trajectory splitting
  - Split smoothed trajectory at mean follow-up point
  - Create solid line data (within mean follow-up)
  - Create dashed line data (beyond mean follow-up)
  - _Requirements: 1.6_

## Task 5: Spaghetti Plot Visualization

Implement core spaghetti plot generation functions.

- [ ] 5.1 Implement `plot_whole_population()` function
  - Plot individual trajectories with low alpha
  - Calculate and plot population smoothed mean
  - Add 95% confidence interval shading
  - Apply length-adjusted styling (solid/dashed)
  - Add population statistics to title
  - Return mean follow-up and smoothed data for overlay
  - _Requirements: 1.3, 1.4, 1.6, 1.7, 1.10, 5.1, 5.2, 5.3, 5.7_

- [ ] 5.2 Implement `plot_single_cluster_trajectory()` function
  - Plot individual patient trajectories for cluster
  - Calculate and plot cluster smoothed mean
  - Add 95% confidence interval shading
  - Apply length-adjusted styling
  - Add cluster statistics to title
  - Use cluster-specific color from palette
  - Return mean follow-up for overlay
  - _Requirements: 1.3, 1.4, 1.6, 1.7, 1.9, 1.10, 5.1, 5.2, 5.3, 5.4, 5.7_

- [ ] 5.3 Implement overlay panel creation
  - Plot smoothed means for all clusters
  - Include population mean as reference
  - Apply length-adjusted styling to each trajectory
  - Use consistent cluster colors
  - Add legend with cluster identification
  - _Requirements: 1.5, 1.9, 5.4, 5.7_

- [ ] 5.4 Implement `create_spaghetti_plots_with_overlay()` function
  - Create multi-panel figure with appropriate layout
  - Call plot_whole_population() for first panel
  - Call plot_single_cluster_trajectory() for each cluster
  - Create overlay panel with all smoothed means
  - Add overall figure title
  - Apply consistent styling across panels
  - _Requirements: 1.5, 5.1, 5.2, 5.3, 5.7_

- [ ] 5.5 Implement `generate_spaghetti_plots()` main function
  - Load configuration
  - Load and prepare data
  - Generate full timespan plot
  - Generate cutoff timespan plot (if configured)
  - Save figures to output directory
  - Optionally display plots
  - Return dictionary of Figure objects
  - _Requirements: 1.8, 5.6, 6.1, 6.3, 7.1, 7.4_

## Task 6: Linear Mixed Model Analysis

Implement LMM fitting and visualization functions.

- [ ] 6.1 Implement spline basis function creation
  - Calculate knot positions from data quantiles
  - Create B-spline basis using patsy
  - Handle edge cases (insufficient data range)
  - _Requirements: 3.3_

- [ ] 6.2 Implement formula construction for LMM
  - Build formula string based on coding type (deviation vs reference)
  - Add covariate terms if configured
  - Add interaction terms (cluster * time_spline)
  - Validate formula syntax
  - _Requirements: 3.2, 3.4_

- [ ] 6.3 Implement `fit_lmm()` function
  - Prepare data (categorical cluster variable, spline terms)
  - Construct model formula
  - Fit mixed linear model using statsmodels
  - Check convergence status
  - Return fitted model results
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.7, 7.2, 8.7_

- [ ] 6.4 Implement `display_full_deviation_effects()` function
  - Extract cluster effects from model
  - Calculate omitted cluster effect (negative sum)
  - Print formatted results
  - _Requirements: 3.8_

- [ ] 6.5 Implement `plot_predicted_trajectories()` function
  - Create prediction grid across time range
  - Generate predictions for each cluster
  - Plot predicted trajectories with cluster colors
  - Add legend and labels
  - Apply consistent styling
  - _Requirements: 3.6, 5.1, 5.2, 5.3, 5.4, 5.7_

- [ ] 6.6 Implement `run_lmm_analysis()` main function
  - Load configuration
  - Load and prepare data
  - Fit LMM (unadjusted and/or adjusted)
  - Display model summaries
  - Display deviation effects if applicable
  - Generate predicted trajectory plots
  - Save results and figures
  - Return results dictionary
  - _Requirements: 3.1, 3.2, 3.4, 3.6, 3.7, 3.8, 6.2, 6.4, 7.2, 7.4_

## Task 7: Utility Functions and Helpers

Implement supporting utility functions.

- [ ] 7.1 Implement `get_cluster_colors()` function
  - Return default color palette or custom palette
  - Ensure sufficient colors for number of clusters
  - Support color-blind friendly palettes
  - _Requirements: 1.9, 5.5_

- [ ] 7.2 Implement progress indicator utilities
  - Add print statements for major steps
  - Include timing information for long operations
  - _Requirements: 7.4_

- [ ] 7.3 Implement data validation utilities
  - Check for minimum sample sizes per cluster
  - Validate date ranges
  - Check for missing critical values
  - _Requirements: 4.4, 4.6, 8.4, 8.6_

- [ ] 7.4 Implement figure saving utilities
  - Save figures in multiple formats (PNG, SVG, PDF)
  - Use consistent naming conventions
  - Create subdirectories as needed
  - _Requirements: 5.6_

## Task 8: Error Handling and Validation

Implement comprehensive error handling.

- [ ] 8.1 Add configuration validation errors
  - Raise specific exceptions for missing required fields
  - Validate parameter ranges with clear messages
  - _Requirements: 8.1, 8.2_

- [ ] 8.2 Add database connection error handling
  - Catch and re-raise database errors with context
  - Provide suggestions for common issues
  - _Requirements: 8.3_

- [ ] 8.3 Add data quality validation
  - Check for sufficient data before analysis
  - Warn about clusters with few patients
  - Detect and report anomalies
  - _Requirements: 8.4, 8.6_

- [ ] 8.4 Add model fitting error handling
  - Catch convergence failures
  - Provide diagnostic information
  - Suggest model adjustments
  - _Requirements: 8.7_

## Task 9: Documentation and Examples

Create comprehensive documentation.

- [ ] 9.1 Write module docstring
  - Explain module purpose and capabilities
  - Provide usage examples
  - Document configuration requirements
  - _Requirements: 6.6_

- [ ] 9.2 Write function docstrings
  - Document all parameters with types
  - Describe return values
  - Include usage examples for main functions
  - Note any important caveats or limitations
  - _Requirements: 6.6_

- [ ] 9.3 Create notebook usage examples
  - Add cell demonstrating spaghetti plot generation
  - Add cell demonstrating LMM analysis
  - Show how to customize configuration
  - Demonstrate accessing and customizing returned figures
  - _Requirements: 6.1, 6.2, 6.3_

- [ ] 9.4 Update existing notebook sections
  - Replace inline trajectory code with module calls
  - Clean up legacy code comments
  - Add references to new module
  - _Requirements: 6.7_

## Task 10: Testing and Validation

Validate module functionality.

- [ ] 10.1 Test with existing cluster configurations
  - Run with PAM k=7 configuration
  - Verify plots match expected output
  - Check LMM results for reasonableness
  - _Requirements: 7.1, 7.2, 7.5_

- [ ] 10.2 Test error handling
  - Test with missing configuration file
  - Test with invalid database paths
  - Test with insufficient data
  - Verify error messages are clear
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 10.3 Test edge cases
  - Test with single cluster
  - Test with very short follow-up times
  - Test with missing measurements
  - _Requirements: 4.4, 7.5_

- [ ] 10.4 Performance testing
  - Measure execution time for full analysis
  - Verify memory usage is reasonable
  - Optimize bottlenecks if needed
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

## Task 11: Integration and Cleanup

Finalize integration with existing codebase.

- [ ] 11.1 Ensure consistency with other modules
  - Match naming conventions from cluster_descriptions.py
  - Use same color palettes as wgc_visualizations.py
  - Follow same configuration patterns
  - _Requirements: 6.7_

- [ ] 11.2 Clean up notebook
  - Remove or comment out legacy trajectory code
  - Add clear section headers for trajectory analysis
  - Ensure all cells run without errors
  - _Requirements: 6.7_

- [ ] 11.3 Update cluster_config.json
  - Add all necessary trajectory configuration
  - Document configuration options with comments
  - Provide sensible defaults
  - _Requirements: 2.1, 2.7_

- [ ] 11.4 Final validation
  - Run complete notebook end-to-end
  - Verify all outputs are generated correctly
  - Check that figures are publication-quality
  - Confirm module is ready for use
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_
