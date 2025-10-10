# Implementation Plan

- [x] 1. Configuration management integration and cleanup



  - Merge RFAnalysisConfig from rf_config.py into paper2_rf_config class in paper12_config.py
  - Add new configuration fields for statistical testing and visualization options
  - Update rf_engine.py to import from paper12_config instead of rf_config
  - Remove rf_config.py file after successful migration
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement statistical significance testing infrastructure





- [x] 2.1 Create SignificanceResults dataclass and core testing framework

  - Implement SignificanceResults dataclass to store all significance test results
  - Create FeatureImportanceSignificanceTester class with initialization and configuration handling
  - Add error handling infrastructure for statistical testing failures

  - _Requirements: 2.1, 3.1_

- [x] 2.2 Implement shadow feature testing for Gini importance

  - Code create_shadow_features method to generate permuted shadow dataset
  - Implement test_gini_significance method with augmented dataset training
  - Add calculate_significance_threshold method using maximum shadow importance
  - Integrate shadow feature testing with existing Gini importance calculation
  - _Requirements: 2.2, 2.3, 2.4, 2.5_

- [x] 2.3 Implement SHAP significance testing with statistical rigor


  - Code test_shap_significance method using one-sample Wilcoxon signed-rank tests
  - Integrate with existing fdr_correction_utils for Benjamini-Hochberg correction
  - Add p-value collection and adjustment for multiple comparisons
  - Implement significance determination logic comparing adjusted p-values to alpha
  - _Requirements: 3.2, 3.3, 3.4_

- [ ]* 2.4 Write unit tests for statistical testing components
  - Create unit tests for shadow feature generation with synthetic datasets
  - Write tests for Wilcoxon test implementation and FDR correction integration
  - Test significance threshold calculations and edge cases
  - _Requirements: 2.1, 3.1_

- [x] 3. Implement enhanced visualization system



- [x] 3.1 Create EnhancedFeatureImportancePlotter class and base infrastructure


  - Implement EnhancedFeatureImportancePlotter class with configuration handling
  - Add dynamic figure sizing logic based on number of features
  - Create base plotting utilities for consistent styling and typography
  - Implement feature ordering logic for consistent cross-panel ordering
  - _Requirements: 4.4, 5.4_

- [x] 3.2 Implement primary composite plot (Gini + SHAP beeswarm)


  - Code _plot_primary_composite method creating two-panel figure layout
  - Implement Gini importance panel with significance threshold line visualization
  - Create SHAP beeswarm panel with significance annotations (asterisks)
  - Add proper axis labeling, titles, and legend for publication quality
  - _Requirements: 4.1, 4.2, 4.3, 4.6_

- [x] 3.3 Implement secondary composite plot (mean SHAP + permutation importance)


  - Code _plot_secondary_composite method for mean absolute SHAP and permutation importance
  - Implement left panel showing mean absolute SHAP values in descending order
  - Create right panel displaying permutation importance in descending order
  - Ensure consistent feature ordering and professional styling
  - _Requirements: 5.1, 5.2, 5.3, 5.6_

- [x] 3.4 Add significance annotation system


  - Implement _annotate_significance method for adding statistical markers
  - Code significance asterisk placement logic (* for p<0.05, ** for p<0.01)
  - Add vertical dashed line rendering for Gini significance thresholds
  - Create legend and annotation styling for clear significance indication
  - _Requirements: 4.6, 3.5_

- [ ]* 3.5 Write unit tests for visualization components
  - Create tests for plot generation with various feature counts and configurations
  - Test significance annotation placement and styling
  - Verify figure sizing algorithms and color scheme handling
  - _Requirements: 4.4, 5.4_

- [x] 4. Integrate enhanced components with existing RandomForestAnalyzer



- [x] 4.1 Extend RandomForestAnalyzer with significance testing integration


  - Add _test_feature_significance method to orchestrate all significance testing
  - Integrate FeatureImportanceSignificanceTester into analyzer workflow
  - Update results dictionary to store SignificanceResults and feature ordering
  - Add error handling for significance testing failures with graceful fallbacks
  - _Requirements: 2.6, 3.6_

- [x] 4.2 Update analyzer output pipeline with new visualization methods


  - Modify run_and_generate_outputs method to call new composite plotting methods
  - Replace existing plot_feature_importance_grid with new primary and secondary plots
  - Maintain ROC curve generation for backward compatibility
  - Update output file naming convention for new plot types
  - _Requirements: 6.2, 6.3, 6.4_

- [x] 4.3 Implement comprehensive error handling and logging


  - Add robust error handling for statistical testing and visualization failures
  - Implement informative logging for debugging and monitoring pipeline execution
  - Create fallback mechanisms when significance testing or enhanced plotting fails
  - Add configuration validation with clear error messages for invalid settings
  - _Requirements: 6.5_

- [ ]* 4.4 Write integration tests for complete pipeline
  - Create end-to-end tests with real WGC dataset to verify complete functionality
  - Test backward compatibility with existing analysis configurations
  - Verify output file generation, naming, and format consistency
  - Test memory usage and performance with large feature sets
  - _Requirements: 6.1, 6.6_

- [x] 5. Final validation and optimization



- [x] 5.1 Validate statistical accuracy and implementation correctness


  - Verify shadow feature testing produces expected significance thresholds
  - Validate SHAP significance testing against known statistical properties
  - Cross-check FDR correction implementation with established methods
  - Test significance results consistency across multiple runs
  - _Requirements: 2.5, 2.6, 3.4, 3.5_

- [x] 5.2 Optimize performance and memory usage


  - Profile memory usage during shadow feature testing with large datasets
  - Optimize SHAP significance calculation for computational efficiency
  - Implement memory management for large augmented datasets
  - Add progress indicators for long-running statistical computations
  - _Requirements: 4.4, 5.4_

- [x] 5.3 Create comprehensive documentation and usage examples


  - Document new configuration options and their effects on analysis output
  - Create example usage scripts demonstrating enhanced pipeline capabilities
  - Add inline code documentation for new methods and classes
  - Update existing documentation to reflect configuration changes
  - _Requirements: 1.4, 6.4_