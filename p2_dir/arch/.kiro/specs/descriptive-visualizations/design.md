# Design Document

## Overview

The descriptive visualizations pipeline will be implemented as a comprehensive "Swiss Army knife" for exploratory and descriptive visualizations. It will generate both risk ratio and risk difference forest plots for multiple outcomes (10% weight loss achievement and 60-day dropout) across different weight gain causes. The design maximally reuses existing functionality from descriptive_comparisons.py and provides a clean, organized output structure.

## Architecture

### High-Level Flow
1. **Configuration & Data Loading**: Load data from configurable input table using existing config system
2. **Multi-Outcome Analysis**: Create 2x2 contingency tables for both 10%_wl_achieved and 60d_dropout outcomes
3. **Effect Size Calculation**: Calculate both risk ratios and risk differences with confidence intervals
4. **Statistical Testing**: Perform appropriate statistical tests with FDR correction
5. **Dual Visualization**: Generate paired forest plots (risk ratios with log scale, risk differences with linear scale)
6. **Organized Output**: Save plots and summary tables to structured output directory

### Module Dependencies
- **descriptive_comparisons.py**: Reuse categorical_pvalue, get_cause_cols functions
- **fdr_correction_utils.py**: Apply Benjamini-Hochberg FDR correction
- **paper12_config.py**: Database paths and configuration management
- **Standard libraries**: pandas, numpy, matplotlib, scipy.stats, sqlite3

## Components and Interfaces

### Main Function Interface
```python
def run_descriptive_visualizations(
    input_table: str,  # Configurable input table name (set from notebook)
    config: master_config = None
) -> dict
```

**Note**: The function generates multiple outputs automatically with standardized naming. All outputs are saved to ../outputs/descriptive_visualizations/ with organized subdirectories.

### Core Components

#### 1. Data Loader Component
```python
def load_forest_plot_data(input_table: str, db_path: str) -> pd.DataFrame
```
- Loads data from specified table
- Validates presence of required columns (10%_wl_achieved, weight gain cause columns)
- Returns cleaned DataFrame ready for analysis

#### 2. Effect Size Calculator Component
```python
def calculate_effect_sizes(df: pd.DataFrame, cause_columns: list, outcomes: list) -> pd.DataFrame
```
- Creates 2x2 contingency tables for each weight gain cause and outcome combination
- Calculates risk ratios: RR = (a/(a+b)) / (c/(c+d))
- Calculates risk differences: RD = (a/(a+b)) - (c/(c+d))
- Computes 95% confidence intervals for both measures
- Returns DataFrame with columns: cause, outcome, risk_ratio, rr_ci_lower, rr_ci_upper, risk_difference, rd_ci_lower, rd_ci_upper

#### 3. Statistical Testing Component
```python
def perform_statistical_tests(df: pd.DataFrame, cause_columns: list) -> list
```
- Reuses categorical_pvalue function from descriptive_comparisons.py
- Applies Fisher's exact test when any cell < 5
- Returns list of p-values for FDR correction

#### 4. Dual Forest Plot Generator Component
```python
def create_forest_plots(results_df: pd.DataFrame, output_dir: str) -> None
```
- Creates paired matplotlib figures for each outcome
- Risk ratio plots use logarithmic x-axis with reference line at RR = 1.0
- Risk difference plots use linear x-axis with reference line at RD = 0.0
- Plots points with confidence interval error bars
- No significance markers - relies on confidence intervals crossing reference lines

## Data Models

### Input Data Structure
```python
# Expected columns in input table
required_columns = [
    "10%_wl_achieved",  # Binary outcome (0/1)
    "60d_dropout",      # Binary outcome (0/1)
    "womens_health_and_pregnancy",  # Binary WGC (0/1)
    "mental_health",  # Binary WGC (0/1)
    # ... other WGC columns (total of 12)
]

# Outcomes to analyze
outcomes = ["10%_wl_achieved", "60d_dropout"]
```

### Results Structure
```python
results_schema = {
    "cause": str,  # Weight gain cause name
    "cause_pretty": str,  # Pretty name for plotting
    "outcome": str,  # Outcome variable name
    "outcome_pretty": str,  # Pretty outcome name
    "risk_ratio": float,  # Calculated risk ratio
    "rr_ci_lower": float,  # RR 95% CI lower bound
    "rr_ci_upper": float,  # RR 95% CI upper bound
    "risk_difference": float,  # Calculated risk difference
    "rd_ci_lower": float,  # RD 95% CI lower bound
    "rd_ci_upper": float,  # RD 95% CI upper bound
    "p_value": float,  # Raw p-value
    "p_value_fdr": float,  # FDR-corrected p-value
    "n_present": int,  # Sample size with cause present
    "n_absent": int,  # Sample size with cause absent
    "events_present": int,  # Events in cause present group
    "events_absent": int  # Events in cause absent group
}
```

### Contingency Table Structure
```python
# 2x2 table for each weight gain cause
contingency_table = {
    "achieved_wl_cause_present": int,  # a
    "no_wl_cause_present": int,       # b
    "achieved_wl_cause_absent": int,  # c
    "no_wl_cause_absent": int         # d
}
```

## Error Handling

### Data Validation
- Check for required columns in input table
- Validate binary nature of outcome and cause variables
- Handle missing values by excluding from analysis with printed warnings

### Statistical Edge Cases
- Zero cells in contingency tables: Print warning and skip that cause
- Infinite confidence intervals: Cap at reasonable bounds and print warning
- Failed statistical tests: Print error message and continue with remaining causes

### Plot Generation
- Empty results: Generate plot with message indicating no valid results
- Extreme confidence intervals: Adjust plot limits and print warning
- File save errors: Print clear error message with path information

## Testing Strategy

### Unit Testing Approach
- Test risk ratio calculations with known contingency tables
- Verify confidence interval calculations against manual computations
- Test statistical test selection logic (Chi-squared vs Fisher's exact)
- Validate FDR correction integration

### Integration Testing
- Test with actual timetoevent_wgc_compl data structure
- Verify compatibility with existing descriptive_comparisons functions
- Test end-to-end pipeline with various input configurations

### Data Quality Checks
- Validate against expected sample sizes from existing analyses
- Cross-check risk ratios with manual calculations for subset of causes
- Verify plot output matches expected format from provided example

## Implementation Notes

### Code Reuse Strategy
1. **get_cause_cols()**: Reuse from descriptive_comparisons.py to identify the 12 WGC columns from row_order config
2. **categorical_pvalue()**: Reuse for Chi-squared testing
3. **apply_fdr_correction()**: Direct import from fdr_correction_utils
4. **Database connection pattern**: Follow existing pattern from descriptive_comparisons.py
5. **Row order configuration**: Reuse the same ROW_ORDER structure from notebook to ensure consistent WGC identification

### Pretty Name Mapping
```python
# Reuse mapping logic from descriptive_comparisons.py row_order
cause_pretty_names = {
    "womens_health_and_pregnancy": "Women's health/pregnancy",
    "mental_health": "Mental health",
    "family_issues": "Family issues",
    # ... etc
}
```

### Plot Styling
- Follow existing matplotlib styling from descriptive_comparisons.py
- Use seaborn-v0_8-whitegrid style for consistency
- Save at 300 DPI for publication quality
- Use figure size (12, 8) for readability

### Output Structure
```
../outputs/descriptive_visualizations/
├── forest_plots/
│   ├── risk_ratios_10pct_wl_achieved.png
│   ├── risk_differences_10pct_wl_achieved.png
│   ├── risk_ratios_60d_dropout.png
│   └── risk_differences_60d_dropout.png
├── summary_tables/
│   ├── effect_sizes_summary.csv
│   └── statistical_tests_summary.csv
└── console_output_summary.txt
```

## Performance Considerations

### Memory Usage
- Process one weight gain cause at a time to minimize memory footprint
- Use pandas operations efficiently to avoid unnecessary data copying

### Computation Time
- Expected runtime < 30 seconds for typical dataset sizes
- Statistical tests are computationally lightweight
- Plot generation is the most time-consuming step but still fast

### Scalability
- Design supports easy addition of new weight gain causes
- Can handle varying sample sizes gracefully
- Configurable input tables allow use with different datasets