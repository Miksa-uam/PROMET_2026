# Implementation Plan

## Current Status Analysis
**Note**: A comprehensive `forest_plot_risk_ratios.py` script already exists with most core functionality implemented for single-outcome risk ratio analysis. The tasks below focus on extending this to create the full dual-outcome descriptive visualizations pipeline as specified in the requirements.

- [x] **Existing Implementation Review**: `forest_plot_risk_ratios.py` contains:
  - Complete data loading and validation (`load_forest_plot_data`)
  - Risk ratio calculation with confidence intervals (`calculate_risk_ratios`)
  - Statistical testing with Chi-squared/Fisher's exact selection (`perform_statistical_tests`)
  - FDR correction integration (`apply_fdr_correction_to_results`)
  - Forest plot generation (`create_forest_plot`)
  - Main analysis pipeline (`run_forest_plot_analysis`)

## Remaining Implementation Tasks

- [x] 1. Create comprehensive descriptive visualizations script





  - Create scripts/descriptive_visualizations.py by adapting existing forest_plot_risk_ratios.py
  - Extend data loading to handle both 10%_wl_achieved and 60d_dropout outcomes
  - Modify main function signature to match spec requirements
  - Set up organized output directory structure (../outputs/descriptive_visualizations/)
  - _Requirements: 4.1, 4.3, 4.6, 5.1, 5.2, 6.1_

- [x] 2. Extend effect size calculations for dual outcomes and risk differences







  - Adapt existing calculate_risk_ratios function to calculate_effect_sizes
  - Add risk difference calculations: RD = (a/(a+b)) - (c/(c+d))
  - Compute 95% confidence intervals for risk differences using appropriate formulas
  - Process both 10%_wl_achieved and 60d_dropout outcomes for each weight gain cause
  - Maintain existing edge case handling and print warnings
  - _Requirements: 1.3, 1.4, 1.5, 5.3, 5.4_

- [x] 3. Extend statistical testing for multiple outcomes





  - Adapt existing perform_statistical_tests to handle both outcomes
  - Maintain Chi-squared/Fisher's exact test selection logic
  - Return p-values organized by cause-outcome combinations
  - Preserve existing error handling and print messages
  - _Requirements: 1.6, 2.1, 2.2, 4.3, 5.3_

- [x] 4. Implement separate FDR correction for each outcome





  - Extend existing FDR correction to apply separately for each outcome
  - Maintain integration with fdr_correction_utils.apply_fdr_correction
  - Store results organized by outcome with both raw and corrected p-values
  - Print summary statistics for both outcomes
  - _Requirements: 2.3, 2.4, 4.2, 5.5_

- [x] 5. Create dual forest plot visualization system





  - Extend existing create_forest_plot to create_forest_plots (plural)
  - Generate separate risk ratio plots (log scale, RR=1.0 reference) for each outcome
  - Generate separate risk difference plots (linear scale, RD=0.0 reference) for each outcome
  - Maintain existing plot styling and confidence interval visualization
  - Remove significance markers and rely on confidence intervals crossing reference lines
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 6. Implement organized output structure





  - Generate descriptive filenames (e.g., risk_ratios_10pct_wl_achieved.png)
  - Export comprehensive summary tables with all statistics
  - Maintain existing error handling for file operations
  - _Requirements: 4.4, 4.6, 5.2, 5.5, 6.3_

- [x] 7. Create simplified notebook interface





  - Write clean notebook cell that calls run_descriptive_visualizations()
  - Make input_table the primary configurable parameter
  - Position for use after "#### I/2.2. Risk ratio/risk difference analyses" section
  - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [x] 8. Validate integration with existing project infrastructure





  - Ensure compatibility with existing get_cause_cols and categorical_pvalue functions
  - Verify database path configuration works with master_config
  - Test with actual project data (timetoevent_wgc_compl table)
  - Confirm output directory structure follows project conventions
  - _Requirements: 4.1, 4.5, 6.4_

- [ ]* 9. Comprehensive testing and validation
  - Test risk difference calculations alongside existing risk ratio tests
  - Validate dual-outcome processing with sample data
  - Test forest plot generation for both plot types and outcomes
  - Verify FDR correction works correctly for multiple outcomes
  - End-to-end testing with actual project data
  - _Requirements: All requirements validation_