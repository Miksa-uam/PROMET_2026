# Implementation Plan

- [x] 1. Add variable type detection helper function





  - Create _infer_variable_type function after existing helper functions (after line 105)
  - Implement logic: ≤10 unique values = categorical, >10 = continuous
  - Handle edge cases: empty series, all NaN, single value (return 'categorical' as safe default)
  - _Requirements: 1.1, 4.2, 4.3, 4.4_

- [x] 2. Modify analyze_cluster_vs_population for correct test selection





  - Add optional variable_types parameter to function signature
  - Add type inference logic before processing loop when variable_types is None
  - Print warning when inferring types
  - Replace hardcoded chi_squared_test call (line 595) with conditional logic
  - Add print statements showing which test is used for each variable
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 4.1, 4.5_

- [x] 3. Implement dynamic FDR column insertion





  - Replace existing FDR correction logic (lines 607-619)
  - Build new column order list that inserts FDR columns immediately after raw p-value columns
  - Apply FDR correction and create columns during iteration
  - Reorder DataFrame columns using new column list
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4. Add test transparency to calculate_cluster_pvalues





  - Add print statement after line 177 showing test type being used
  - Format message to indicate continuous vs categorical test selection
  - _Requirements: 3.1, 3.2, 3.5_

- [x] 5. Update documentation





  - Update analyze_cluster_vs_population docstring to document variable_types parameter
  - Add docstring for _infer_variable_type function
  - Update module version from 1.0 to 1.1 in header comment
  - _Requirements: All requirements (documentation support)_

- [ ]* 6. Create comprehensive test suite
  - [ ]* 6.1 Write unit tests for _infer_variable_type function
    - Test continuous variable detection (>10 unique values)
    - Test categorical variable detection (≤10 unique values)
    - Test edge cases (empty, all NaN, single value)
    - _Requirements: 4.2, 4.3, 4.4_
  
  - [ ]* 6.2 Write integration tests for analyze_cluster_vs_population
    - Test with explicit variable_types parameter
    - Test with inferred variable_types (None)
    - Test mixed continuous and categorical variables
    - Verify correct test selection via captured print output
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2_
  
  - [ ]* 6.3 Write tests for FDR column positioning
    - Verify FDR columns appear immediately after raw p-value columns
    - Check column order pattern: Mean/N, p-value, p-value (FDR-corrected)
    - Test with multiple clusters
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [ ]* 6.4 Create integration test with real data
    - Run analyze_cluster_vs_population with continuous variables
    - Verify Mann-Whitney U test is used (check print output)
    - Verify no "Low expected frequencies" warnings for continuous variables
    - Verify FDR columns positioned correctly in saved database table
    - _Requirements: 1.2, 2.4, 3.1, 5.3, 5.4_

- [ ]* 7. Verify existing visualization functions
  - Confirm cluster_continuous_distributions uses is_categorical=False
  - Confirm cluster_categorical_distributions uses is_categorical=True
  - Run sample visualizations to verify print output shows correct tests
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
