# Implementation Plan

- [x] 1. Create forest plot analysis script structure





  - Create scripts/forest_plot_risk_ratios.py with main function signature
  - Import required modules (pandas, numpy, matplotlib, scipy.stats, sqlite3)
  - Import existing project modules (descriptive_comparisons, fdr_correction_utils, paper12_config)
  - Set up basic error handling with print statements
  - _Requirements: 4.1, 4.3, 5.1, 5.2, 6.1_

- [x] 2. Implement data loading and validation functionality





  - Create load_forest_plot_data function that accepts configurable input table name
  - Validate presence of required columns (10%_wl_achieved and weight gain cause columns)
  - Handle missing data gracefully with print warnings
  - Return cleaned DataFrame ready for analysis
  - _Requirements: 1.1, 5.1, 5.4_

- [x] 3. Implement risk ratio calculation engine





  - Create calculate_risk_ratios function that processes each weight gain cause
  - Build 2x2 contingency tables using 10%_wl_achieved as outcome
  - Calculate risk ratios: RR = (a/(a+b)) / (c/(c+d))
  - Compute 95% confidence intervals using log transformation and standard error
  - Handle edge cases (zero cells, infinite CIs) with print warnings
  - _Requirements: 1.2, 1.3, 1.4, 5.3, 5.4_

- [x] 4. Integrate statistical testing with existing functions





  - Reuse categorical_pvalue function from descriptive_comparisons.py for Chi-squared tests
  - Implement Fisher's exact test selection when any contingency table cell < 5
  - Create perform_statistical_tests function that returns p-values for each cause
  - Handle statistical test failures with print error messages
  - _Requirements: 1.5, 2.1, 2.2, 4.3, 5.3_

- [x] 5. Apply FDR correction using existing utilities





  - Import and use apply_fdr_correction from fdr_correction_utils
  - Apply Benjamini-Hochberg correction to collected p-values
  - Store both raw and FDR-corrected p-values in results
  - Print summary of correction results
  - _Requirements: 2.3, 2.4, 4.2, 5.5_

- [x] 6. Implement forest plot visualization





  - Create create_forest_plot function with matplotlib
  - Use logarithmic x-axis scale for risk ratios
  - Plot points with horizontal confidence interval error bars
  - Add vertical reference line at RR = 1.0
  - Apply proper axis labels and weight gain cause names
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 7. Add significance markers and styling





  - Add asterisks or other markers for FDR-significant results
  - Apply consistent styling following existing project conventions
  - Use seaborn-v0_8-whitegrid style for consistency with descriptive_comparisons.py
  - Set appropriate figure size and DPI for publication quality
  - _Requirements: 3.5, 4.4_
-

- [x] 8. Implement output generation and file handling




  - Save forest plot as PNG to ../outputs directory
  - Generate summary table with all calculated statistics
  - Save summary table as CSV file
  - Print completion summary with sample sizes and significant findings
  - Handle file save errors with clear error messages
  - _Requirements: 4.4, 5.2, 5.5, 6.3_
-

- [x] 9. Create notebook integration cell





  - Write notebook cell code that calls the forest plot analysis
  - Make input table name configurable from notebook parameters
  - Position cell after "#### I/2.2. Risk ratio/risk difference analyses" section
  - Include example usage with different input tables
  - Add configuration for output filename and paths
  - _Requirements: 6.1, 6.2, 6.5_

- [ ] 10. Integrate with existing project configuration
  - Reuse get_cause_cols function from descriptive_comparisons.py
  - Use existing ROW_ORDER configuration for weight gain cause identification
  - Apply existing database path configuration from paths_config
  - Ensure compatibility with master_config structure
  - Follow existing project conventions for directory structure
  - _Requirements: 4.1, 4.5, 6.4_

- [ ]* 11. Add comprehensive testing and validation
  - Test risk ratio calculations with known contingency tables
  - Verify confidence interval calculations against manual computations
  - Test statistical test selection logic (Chi-squared vs Fisher's exact)
  - Validate FDR correction integration with sample data
  - Test end-to-end pipeline with actual timetoevent_wgc_compl data
  - _Requirements: All requirements validation_