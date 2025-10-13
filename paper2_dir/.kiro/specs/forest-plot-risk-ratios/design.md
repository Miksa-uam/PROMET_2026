# Design Document

## Overview

The forest plot pipeline will be implemented as a single Python script that creates publication-ready forest plots showing risk ratios for 10% weight loss achievement across different weight gain causes. The design maximally reuses existing functionality from descriptive_comparisons.py and integrates seamlessly with the current project infrastructure.

## Architecture

### High-Level Flow
1. **Configuration & Data Loading**: Load data from configurable input table using existing config system
2. **Risk Ratio Calculation**: Create 2x2 contingency tables and calculate risk ratios with confidence intervals
3. **Statistical Testing**: Perform appropriate statistical tests with FDR correction
4. **Visualization**: Generate forest plot with logarithmic scale and significance markers
5. **Output**: Save plot and summary table to outputs directory

### Module Dependencies
- **descriptive_comparisons.py**: Reuse categorical_pvalue, get_cause_cols functions
- **fdr_correction_utils.py**: Apply Benjamini-Hochberg FDR correction
- **paper12_config.py**: Database paths and configuration management
- **Standard libraries**: pandas, numpy, matplotlib, scipy.stats, sqlite3

## Components and Interfaces

### Main Function Interface
```python
def run_forest_plot_analysis(
    input_table: str,  # Configurable input table name (set from notebook)
    output_filename: str = "forest_plot_10pct_wl_risk_ratios.png",
    config: master_config = None
) -> dict
```

**Note**: The input_table parameter is required and must be specified from the notebook cell, allowing flexibility to use different time-to-event tables (e.g., "timetoevent_wgc_compl", "timetoevent_wgc_gen_compl", etc.)

### Core Components

#### 1. Data Loader Component
```python
def load_forest_plot_data(input_table: str, db_path: str) -> pd.DataFrame
```
- Loads data from specified table
- Validates presence of required columns (10%_wl_achieved, weight gain cause columns)
- Returns cleaned DataFrame ready for analysis

#### 2. Risk Ratio Calculator Component
```python
def calculate_risk_ratios(df: pd.DataFrame, cause_columns: list) -> pd.DataFrame
```
- Creates 2x2 contingency tables for each weight gain cause
- Calculates risk ratios: RR = (a/(a+b)) / (c/(c+d))
- Computes 95% confidence intervals using log transformation
- Returns DataFrame with columns: cause, risk_ratio, ci_lower, ci_upper, n_present, n_absent

#### 3. Statistical Testing Component
```python
def perform_statistical_tests(df: pd.DataFrame, cause_columns: list) -> list
```
- Reuses categorical_pvalue function from descriptive_comparisons.py
- Applies Fisher's exact test when any cell < 5
- Returns list of p-values for FDR correction

#### 4. Forest Plot Generator Component
```python
def create_forest_plot(results_df: pd.DataFrame, output_path: str) -> None
```
- Creates matplotlib figure with logarithmic x-axis
- Plots points with confidence interval error bars
- Adds reference line at RR = 1.0
- Applies significance markers based on FDR-corrected p-values

## Data Models

### Input Data Structure
```python
# Expected columns in input table
required_columns = [
    "10%_wl_achieved",  # Binary outcome (0/1)
    "womens_health_and_pregnancy",  # Binary WGC (0/1)
    "mental_health",  # Binary WGC (0/1)
    # ... other WGC columns
]
```

### Risk Ratio Results Structure
```python
results_schema = {
    "cause": str,  # Weight gain cause name
    "cause_pretty": str,  # Pretty name for plotting
    "risk_ratio": float,  # Calculated risk ratio
    "ci_lower": float,  # 95% CI lower bound
    "ci_upper": float,  # 95% CI upper bound
    "p_value": float,  # Raw p-value
    "p_value_fdr": float,  # FDR-corrected p-value
    "significant": bool,  # True if p_value_fdr < 0.05
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

### Output Files
1. **Forest plot**: PNG file saved to ../outputs/
2. **Summary table**: CSV file with all calculated statistics
3. **Console output**: Summary statistics and warnings printed to console

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