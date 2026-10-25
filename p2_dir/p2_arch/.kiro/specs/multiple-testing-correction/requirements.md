# Requirements Document

## Introduction

The current statistical analysis workflow in the paper2_notebook performs multiple pairwise comparisons across demographic groups and weight gain cause (WGC) strata without applying multiple testing correction. This leads to inflated Type I error rates and potentially spurious significant findings. The feature will add configurable multiple testing correction methods to the descriptive_comparisons module to control family-wise error rate (FWER) or false discovery rate (FDR) across the multiple statistical tests performed.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to apply multiple testing correction to my statistical comparisons, so that I can control the family-wise error rate and avoid false positive findings.

#### Acceptance Criteria

1. WHEN the descriptive comparisons analysis runs THEN the system SHALL apply the configured multiple testing correction method to all p-values within each comparison family
2. WHEN multiple correction methods are available THEN the system SHALL support at least Bonferroni, Benjamini-Hochberg (FDR), and Holm-Bonferroni corrections
3. WHEN correction is applied THEN the system SHALL preserve the original raw p-values alongside the corrected p-values
4. WHEN no correction method is specified THEN the system SHALL default to no correction (current behavior)

### Requirement 2

**User Story:** As a researcher, I want to configure different correction methods for different comparison families, so that I can apply appropriate statistical rigor based on the analysis context.

#### Acceptance Criteria

1. WHEN configuring the analysis THEN the system SHALL allow specification of correction method per comparison type (demographic vs WGC stratification)
2. WHEN correction families are defined THEN the system SHALL group related comparisons together for correction (e.g., all age comparisons, all sex comparisons)
3. WHEN correction is applied THEN the system SHALL clearly indicate which correction method was used in the output tables
4. IF a comparison family has fewer than 2 valid p-values THEN the system SHALL skip correction for that family

### Requirement 3

**User Story:** As a researcher, I want to see both raw and corrected p-values in my output tables, so that I can understand the impact of multiple testing correction on my results.

#### Acceptance Criteria

1. WHEN correction is applied THEN the system SHALL display both raw p-values and corrected p-values in separate columns
2. WHEN formatting p-values THEN the system SHALL use consistent formatting (e.g., "<0.001" for very small values, "N/A" for missing)
3. WHEN correction results in p-values > 1 THEN the system SHALL cap corrected p-values at 1.0
4. WHEN exporting results THEN the system SHALL include metadata about which correction methods were applied

### Requirement 4

**User Story:** As a researcher, I want the correction to be applied efficiently without significantly impacting analysis runtime, so that I can maintain my current workflow speed.

#### Acceptance Criteria

1. WHEN processing large datasets THEN the system SHALL complete correction calculations within 10% of the original analysis time
2. WHEN correction fails for any reason THEN the system SHALL log the error and continue with raw p-values
3. WHEN invalid p-values are encountered THEN the system SHALL handle them gracefully without stopping the analysis
4. WHEN correction is disabled THEN the system SHALL maintain backward compatibility with existing configurations