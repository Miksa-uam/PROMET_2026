# Implementation Plan

- [x] 1. Implement core WGC vs population mean analysis function





  - Create `wgc_vs_population_mean_analysis()` function in `descriptive_comparisons.py`
  - Implement population statistics calculation for continuous and categorical variables
  - Add WGC group extraction and comparison logic using existing statistical methods
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1, 3.2, 3.3_

- [x] 2. Implement dynamic table structure generation





  - Create dynamic column generation based on WGC groups from config.wgc_strata
  - Implement proper column naming following the specification pattern
  - Add conditional FDR-corrected column creation based on config.fdr_correction setting
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 3.5_

- [x] 3. Implement statistical comparison and formatting logic





  - Add population vs WGC group comparison using existing `perform_comparison()` function
  - Implement proper formatting for continuous variables (mean ± std) and categorical variables (n (%))
  - Add table row generation with proper variable ordering from config.row_order
  - _Requirements: 2.5, 2.6, 3.1, 3.2, 3.3_

- [x] 4. Integrate FDR correction functionality





  - Add FDR correction integration using existing `fdr_correction_utils` functions
  - Implement conditional FDR correction based on config.fdr_correction setting
  - Add proper p-value collection and correction application for the new table structure
  - _Requirements: 3.4, 3.5_

- [x] 5. Add error handling and edge case management





  - Implement graceful handling of missing variables and insufficient data
  - Add proper error handling for empty WGC groups and statistical test failures
  - Implement database operation error handling consistent with existing pipeline
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 6. Integrate with existing pipeline execution





  - Modify `run_descriptive_comparisons()` function to call new analysis
  - Add proper logging and user feedback consistent with existing analyses
  - Ensure seamless integration without disrupting existing functionality
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 7. Create unit tests for core functionality
  - Write unit tests for population statistics calculation
  - Test dynamic column generation with different WGC configurations
  - Test statistical comparison logic with known datasets
  - _Requirements: 1.1, 2.3, 3.1_

- [ ]* 8. Create integration tests
  - Test complete pipeline integration with new analysis function
  - Test database table creation and data insertion
  - Test FDR correction integration with various configurations
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 9. Add comprehensive logging and documentation





  - Add function documentation following existing code style
  - Implement logging statements consistent with current pipeline
  - Add inline comments explaining statistical comparison logic
  - _Requirements: 4.4, 5.4_

- [x] 10. Validate table naming convention implementation





  - Ensure table names follow '[input_cohort]_wgc_strt_vs_mean' pattern
  - Test with different input cohort names (e.g., wgc_gen_compl, timetoevent_wgc_compl)
  - Verify database table creation with correct naming
  - _Requirements: 1.2_