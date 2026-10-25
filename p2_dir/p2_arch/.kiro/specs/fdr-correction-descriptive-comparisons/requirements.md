# Requirements Document

## Introduction

This feature adds optional False Discovery Rate (FDR) correction using the Benjamini-Hochberg method to the existing descriptive comparisons pipeline. The pipeline currently performs multiple statistical tests (t-tests and chi-squared tests) across demographic and weight gain cause stratifications without correcting for multiple testing. This enhancement will provide researchers with the option to apply FDR correction to control the expected proportion of false discoveries among rejected hypotheses.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to optionally enable FDR correction in my descriptive comparison analyses, so that I can control for multiple testing when performing numerous statistical comparisons.

#### Acceptance Criteria

1. WHEN I configure a descriptive comparison analysis THEN I SHALL have access to a boolean parameter `fdr_correction` in the `descriptive_comparisons_config` dataclass
2. WHEN `fdr_correction` is set to `True` THEN the system SHALL apply Benjamini-Hochberg FDR correction to all p-values within each analysis
3. WHEN `fdr_correction` is set to `False` or omitted THEN the system SHALL behave exactly as it currently does with raw p-values
4. WHEN FDR correction is applied THEN the system SHALL preserve the original p-values and add corrected p-values as additional columns

### Requirement 2

**User Story:** As a researcher, I want FDR correction to be applied separately to different comparison groups, so that the correction is appropriate for the statistical context of each set of tests.

#### Acceptance Criteria

1. WHEN FDR correction is enabled THEN the system SHALL apply correction separately to demographic stratification p-values and weight gain cause stratification p-values
2. WHEN processing demographic stratifications THEN the system SHALL collect all p-values from cohort comparisons, age group comparisons, gender comparisons, and BMI group comparisons for FDR correction
3. WHEN processing weight gain cause stratifications THEN the system SHALL collect all p-values from each weight gain cause comparison for FDR correction
4. WHEN applying FDR correction THEN the system SHALL exclude non-numeric p-values (NaN, "N/A") from the correction procedure

### Requirement 3

**User Story:** As a researcher, I want the corrected p-values to be clearly distinguished from raw p-values in the output tables, so that I can easily interpret the results and understand which values have been adjusted.

#### Acceptance Criteria

1. WHEN FDR correction is applied THEN the system SHALL add new columns with "(FDR-corrected)" suffix to distinguish corrected p-values from raw p-values
2. WHEN generating demographic stratification tables THEN corrected p-value columns SHALL be named "Cohort comparison: p-value (FDR-corrected)", "Age: p-value (FDR-corrected)", "Gender: p-value (FDR-corrected)", and "BMI: p-value (FDR-corrected)"
3. WHEN generating weight gain cause stratification tables THEN corrected p-value columns SHALL be named "{cause_name}: p-value (FDR-corrected)" for each weight gain cause
4. WHEN FDR correction is disabled THEN the system SHALL NOT create any "(FDR-corrected)" columns

### Requirement 4

**User Story:** As a researcher, I want the FDR correction implementation to be robust and handle edge cases appropriately, so that my analyses remain reliable even with unusual data conditions.

#### Acceptance Criteria

1. WHEN all p-values in a correction group are NaN or non-numeric THEN the system SHALL skip FDR correction for that group and log a warning
2. WHEN some p-values in a correction group are NaN or non-numeric THEN the system SHALL apply FDR correction only to valid numeric p-values and set corrected values for invalid p-values to NaN
3. WHEN FDR correction is applied THEN the system SHALL use the statsmodels implementation of the Benjamini-Hochberg method
4. WHEN FDR correction encounters an error THEN the system SHALL log the error and continue with raw p-values rather than failing the entire analysis

### Requirement 5

**User Story:** As a researcher, I want to maintain backward compatibility with existing analysis configurations, so that my current workflows continue to work without modification.

#### Acceptance Criteria

1. WHEN existing analysis configurations are used without the `fdr_correction` parameter THEN the system SHALL default to `fdr_correction=False`
2. WHEN `fdr_correction=False` THEN the output tables SHALL have identical structure and content to the current implementation
3. WHEN upgrading to the new version THEN existing notebook code SHALL continue to work without any required changes
4. WHEN the `fdr_correction` parameter is added to existing configurations THEN it SHALL not affect any other functionality or parameters