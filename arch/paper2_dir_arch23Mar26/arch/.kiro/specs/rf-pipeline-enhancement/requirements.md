# Requirements Document

## Introduction

This specification outlines the enhancement of an existing Random Forest pipeline used for assessing feature importance of weight gain cause (WGC) variables in weight loss predictions. The enhancement focuses on three key areas: configuration management consolidation, robust statistical significance testing for feature importance scores, and improved visualization layouts for publication-ready outputs.

## Requirements

### Requirement 1: Configuration Management Integration

**User Story:** As a researcher, I want to use a unified configuration system across all pipeline components, so that I can maintain consistency and avoid configuration duplication.

#### Acceptance Criteria

1. WHEN the pipeline is initialized THEN the system SHALL use the paper12_config.py as the universal configuration file
2. WHEN integrating RFAnalysisConfig THEN the system SHALL merge it into the existing paper2_rf_config class in paper12_config.py
3. WHEN merging configurations THEN the system SHALL maintain backward compatibility with existing rf_engine.py functionality
4. WHEN the integration is complete THEN the system SHALL remove the standalone rf_config.py file
5. IF there are syntax differences between configurations THEN the system SHALL harmonize them to match the paper12_config.py architecture

### Requirement 2: Statistical Significance Testing for Gini Feature Importance

**User Story:** As a researcher, I want to determine which Gini importance scores are statistically significant compared to random chance, so that I can identify truly meaningful features using a data-driven approach.

#### Acceptance Criteria

1. WHEN calculating Gini importance THEN the system SHALL implement shadow feature permutation testing
2. WHEN creating shadow features THEN the system SHALL create a complete copy of training data with randomly shuffled values for each feature
3. WHEN training the model THEN the system SHALL combine original and shadow features into a single augmented dataset
4. WHEN establishing significance threshold THEN the system SHALL use the maximum importance score from shadow features as the cutoff
5. WHEN identifying significant features THEN the system SHALL mark any original feature with Gini score greater than the cutoff as statistically significant
6. WHEN visualizing results THEN the system SHALL add a vertical dashed line at the cutoff value on Gini importance plots

### Requirement 3: Statistical Significance Testing for SHAP Values

**User Story:** As a researcher, I want to test the statistical significance of SHAP value distributions, so that I can determine which features have meaningful impact on model predictions.

#### Acceptance Criteria

1. WHEN calculating SHAP significance THEN the system SHALL use one-sample Wilcoxon signed-rank tests for each feature
2. WHEN performing significance tests THEN the system SHALL test the null hypothesis that median SHAP values equal zero
3. WHEN handling multiple comparisons THEN the system SHALL apply Benjamini-Hochberg FDR correction to p-values
4. WHEN determining significance THEN the system SHALL compare adjusted p-values to alpha level (0.05)
5. WHEN visualizing results THEN the system SHALL annotate feature names with significance asterisks (* for p<0.05, ** for p<0.01)
6. IF SHAP distributions are non-normal THEN the system SHALL use non-parametric statistical tests

### Requirement 4: Enhanced Visualization Layout - Primary Feature Importance Plot

**User Story:** As a researcher, I want a composite plot showing Gini importance with significance testing alongside SHAP beeswarm plots, so that I can present comprehensive feature importance analysis in a publication-ready format.

#### Acceptance Criteria

1. WHEN creating the primary plot THEN the system SHALL generate a two-panel composite figure
2. WHEN displaying Gini importance THEN the system SHALL show it on the left panel with significance testing results and descending order
3. WHEN displaying SHAP beeswarm THEN the system SHALL show it on the right panel with significance annotations and descending order of contribution
4. WHEN sizing the figure THEN the system SHALL ensure all features are visible without collapsing into "sum of remaining features"
5. WHEN ordering features THEN the system SHALL use descending order of importance/contribution for both panels
6. WHEN indicating significance THEN the system SHALL clearly mark statistically significant features in both panels

### Requirement 5: Enhanced Visualization Layout - Secondary Feature Importance Plot

**User Story:** As a researcher, I want a secondary composite plot showing mean absolute SHAP values and permutation importance, so that I can provide additional feature importance perspectives without significance testing requirements.

#### Acceptance Criteria

1. WHEN creating the secondary plot THEN the system SHALL generate a two-panel composite figure
2. WHEN displaying mean absolute SHAP THEN the system SHALL show it on the left panel in descending order
3. WHEN displaying permutation importance THEN the system SHALL show it on the right panel in descending order
4. WHEN sizing the figure THEN the system SHALL ensure all features are visible without feature collapsing
5. WHEN generating this plot THEN the system SHALL NOT require significance testing for these metrics
6. WHEN ordering features THEN the system SHALL maintain consistent ordering within each panel based on respective importance values

### Requirement 6: Output Integration and Backward Compatibility

**User Story:** As a researcher, I want the enhanced pipeline to maintain existing ROC curve outputs while providing the new visualization formats, so that I can preserve current workflow compatibility.

#### Acceptance Criteria

1. WHEN running the complete pipeline THEN the system SHALL generate ROC curves as before
2. WHEN producing outputs THEN the system SHALL generate exactly three plots: ROC curve, primary FI composite, and secondary FI composite
3. WHEN replacing existing plots THEN the system SHALL substitute the old feature importance visualizations with the new two-panel layouts
4. WHEN maintaining compatibility THEN the system SHALL preserve all existing analysis functionality in rf_engine.py
5. IF the pipeline fails THEN the system SHALL provide clear error messages indicating which component failed
6. WHEN completing analysis THEN the system SHALL save all outputs to the configured output directory with appropriate filenames