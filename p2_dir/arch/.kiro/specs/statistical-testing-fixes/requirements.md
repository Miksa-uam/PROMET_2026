# Requirements Document

## Introduction

This specification addresses critical statistical testing errors in the cluster_descriptions.py module. The current implementation incorrectly applies chi-squared tests to continuous variables in the `analyze_cluster_vs_population` function, and the FDR-corrected p-values are appended at the end of the table instead of being positioned adjacent to their corresponding raw p-values. Additionally, the module lacks transparency regarding which statistical tests are being applied to which variables.

## Glossary

- **System**: The cluster_descriptions.py module
- **Mann-Whitney U Test**: Non-parametric test for comparing distributions of continuous variables between two groups
- **Chi-Squared Test**: Statistical test for comparing categorical variable distributions between groups
- **Fisher's Exact Test**: Alternative to chi-squared test used when expected frequencies are low (< 5)
- **FDR Correction**: False Discovery Rate correction using Benjamini-Hochberg method
- **WGC Variables**: Weight Gain Cause variables (binary/categorical outcomes)
- **Continuous Variables**: Numeric variables with a range of values (e.g., age, BMI, follow-up days)
- **Categorical Variables**: Binary or discrete variables (e.g., yes/no, achieved/not achieved)
- **Column Insertion**: Dynamically placing FDR-corrected p-value columns immediately after their corresponding raw p-value columns

## Requirements

### Requirement 1: Correct Statistical Test Selection in analyze_cluster_vs_population

**User Story:** As a researcher, I want the analyze_cluster_vs_population function to apply the correct statistical test based on variable type, so that my statistical comparisons are theoretically valid.

#### Acceptance Criteria

1. WHEN the System processes a variable in analyze_cluster_vs_population, THE System SHALL determine whether the variable is continuous or categorical
2. WHEN the System identifies a continuous variable, THE System SHALL apply Mann-Whitney U test for cluster vs population comparisons
3. WHEN the System identifies a categorical variable, THE System SHALL apply chi-squared test (with Fisher's exact fallback) for cluster vs population comparisons
4. WHEN the System applies a statistical test, THE System SHALL print a message indicating which test was used for each variable
5. WHERE the function accepts a variable_types parameter, THE System SHALL use the provided mapping to determine test selection

### Requirement 2: Dynamic FDR Column Positioning in Results Tables

**User Story:** As a researcher, I want FDR-corrected p-values to appear immediately after their corresponding raw p-values in the detailed table, so that I can easily compare raw and corrected values side-by-side.

#### Acceptance Criteria

1. WHEN the System creates FDR-corrected p-value columns in analyze_cluster_vs_population, THE System SHALL insert each FDR column immediately after its corresponding raw p-value column
2. WHEN the System builds the results DataFrame, THE System SHALL maintain the column order: Mean/N, p-value, p-value (FDR-corrected) for each cluster
3. THE System SHALL NOT append all FDR columns at the end of the DataFrame
4. WHEN the System saves the detailed table, THE System SHALL preserve the adjacent positioning of raw and FDR-corrected p-values

### Requirement 3: Statistical Test Transparency and Logging

**User Story:** As a researcher, I want clear logging of which statistical tests are applied to which variables, so that I can verify the appropriateness of the analysis and troubleshoot issues.

#### Acceptance Criteria

1. WHEN the System applies Mann-Whitney U test, THE System SHALL print "Using Mann-Whitney U test (continuous variable)" with the variable name
2. WHEN the System applies chi-squared test, THE System SHALL print "Using chi-squared test (categorical variable)" with the variable name
3. WHEN the System applies Fisher's exact test fallback, THE System SHALL print "Using Fisher's exact test (low expected frequencies)" with the variable name
4. WHEN the System encounters a chi-squared test with low expected frequencies but table > 2x2, THE System SHALL print "⚠️ Low expected frequencies but table > 2x2, using chi-squared" with the variable name
5. THE System SHALL include test selection information in all functions that perform statistical comparisons

### Requirement 4: Variable Type Detection and Configuration

**User Story:** As a researcher, I want the system to automatically detect variable types or allow me to specify them, so that the correct statistical tests are applied without manual intervention for each analysis.

#### Acceptance Criteria

1. WHERE a variable_types parameter is provided to analyze_cluster_vs_population, THE System SHALL use the provided mapping to determine variable types
2. WHERE no variable_types parameter is provided, THE System SHALL infer variable type based on the number of unique values and data type
3. WHEN the System infers variable type, THE System SHALL classify variables with ≤ 5 unique values as categorical
4. WHEN the System infers variable type, THE System SHALL classify variables with > 5 unique values as continuous
5. THE System SHALL print a warning when inferring variable types to alert the user

### Requirement 5: Verification of Existing Visualization Functions

**User Story:** As a researcher, I want confirmation that the visualization functions (violin plots, stacked bar charts) use the correct statistical tests, so that I can trust the significance markers displayed on the plots.

#### Acceptance Criteria

1. THE System SHALL verify that cluster_continuous_distributions calls calculate_cluster_pvalues with is_categorical=False
2. THE System SHALL verify that cluster_categorical_distributions calls calculate_cluster_pvalues with is_categorical=True
3. THE System SHALL verify that calculate_cluster_pvalues correctly routes to Mann-Whitney U test for continuous variables
4. THE System SHALL verify that calculate_cluster_pvalues correctly routes to chi-squared test for categorical variables
5. WHERE verification identifies issues, THE System SHALL document and fix them

### Requirement 6: Enhanced Test Output Messages

**User Story:** As a researcher, I want detailed test output messages that include variable names and test types, so that I can audit the analysis process and identify any issues.

#### Acceptance Criteria

1. WHEN the System processes a variable in any statistical function, THE System SHALL print the variable name and test type being applied
2. WHEN the System encounters an error during statistical testing, THE System SHALL print the variable name, test type, and error message
3. WHEN the System completes statistical testing for a variable, THE System SHALL print a success indicator with the variable name
4. THE System SHALL format test output messages consistently across all functions
5. THE System SHALL use indentation to show hierarchical processing (e.g., variable → cluster → test)
