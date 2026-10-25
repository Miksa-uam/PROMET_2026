# Implementation Plan

- [x] 1. Create reusable FDR correction utilities module





  - Create `scripts/fdr_correction_utils.py` with core FDR correction functions using statsmodels
  - Implement `apply_fdr_correction()` function that handles Benjamini-Hochberg correction with robust error handling
  - Implement `collect_pvalues_from_dataframe()` function for extracting p-values from DataFrame columns
  - Implement `integrate_corrected_pvalues()` function for adding corrected p-value columns with configurable naming
  - Add comprehensive docstrings explaining reusability across different analysis contexts
  - _Requirements: 1.2, 1.3, 4.1, 4.2, 4.3, 4.4_

- [ ]* 1.1 Write unit tests for FDR correction utilities
  - Create test cases for valid p-values, mixed valid/NaN p-values, all NaN p-values, and empty inputs
  - Test edge cases including all zeros, all ones, and single p-value scenarios
  - Validate statistical correctness against known Benjamini-Hochberg calculations
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 2. Extend configuration dataclass with FDR correction parameter






  - Add `fdr_correction: bool = False` parameter to `descriptive_comparisons_config` in `paper12_config.py`
  - Ensure backward compatibility by using default value of False
  - Update docstring to document the new parameter and its behavior
  - _Requirements: 1.1, 5.1, 5.2_


- [x] 3. Modify demographic stratification function for FDR support



  - Import FDR correction utilities in `descriptive_comparisons.py`
  - Modify `demographic_stratification()` function to collect p-values when FDR correction is enabled
  - Implement p-value collection for "Cohort comparison: p-value", "Age: p-value", "Gender: p-value", "BMI: p-value" columns
  - Apply FDR correction and integrate results with "(FDR-corrected)" suffix when enabled
  - Ensure original p-values remain unchanged and corrected columns are only added when FDR is enabled
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 3.4_

- [x] 4. Modify weight gain cause stratification function for FDR support


  - Modify `wgc_stratification()` function to collect p-values from all "{cause_name}: p-value" columns when FDR correction is enabled
  - Apply FDR correction separately to weight gain cause comparisons
  - Integrate corrected p-values with "{cause_name}: p-value (FDR-corrected)" naming convention
  - Ensure proper handling of dynamic cause column names based on configuration
  - _Requirements: 1.2, 1.3, 2.1, 2.3, 3.1, 3.3, 3.4_

- [x] 5. Add logging and error handling for FDR correction


  - Implement informational logging when FDR correction is applied (number of p-values corrected)
  - Add warning logging when FDR correction is skipped due to insufficient valid p-values
  - Add error logging with graceful fallback when FDR correction fails
  - Ensure all error conditions preserve original functionality without breaking the analysis pipeline
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 6. Create integration tests for FDR-enabled descriptive comparisons
  - Test demographic stratification with FDR correction enabled using sample data
  - Test weight gain cause stratification with FDR correction enabled
  - Verify output table structure includes both original and corrected p-value columns
  - Test backward compatibility by running existing analysis configurations without modification
  - Validate that corrected p-values are statistically appropriate (higher than or equal to raw p-values)
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 5.1, 5.2, 5.3_

- [ ] 7. Update main pipeline function to support FDR correction



  - Modify `run_descriptive_comparisons()` function to pass FDR correction configuration to stratification functions
  - Ensure the master config properly handles the new FDR correction parameter
  - Add validation to ensure FDR correction parameter is properly propagated through the analysis pipeline
  - _Requirements: 1.1, 1.2, 1.3, 5.1, 5.2, 5.3, 5.4_

- [ ]* 8. Create example notebook demonstrating FDR correction usage
  - Create example analysis configurations with `fdr_correction=True` and `fdr_correction=False`
  - Demonstrate the difference in output table structure between corrected and uncorrected analyses
  - Show how to interpret FDR-corrected p-values in the context of multiple testing
  - Document best practices for when to use FDR correction in descriptive comparisons
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3_