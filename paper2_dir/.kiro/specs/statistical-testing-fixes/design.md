# Design Document

## Overview

This design addresses critical statistical testing errors in cluster_descriptions.py with minimal code changes. The solution focuses on three key fixes: (1) adding variable type detection to analyze_cluster_vs_population, (2) correcting test selection logic, and (3) dynamically inserting FDR columns adjacent to raw p-value columns. All changes maintain the existing code structure and minimize line count inflation.

## Architecture

### Current State
- `analyze_cluster_vs_population`: Hardcoded chi-squared test for all variables (line 595)
- FDR correction: Appends columns at end using simple column creation (lines 617-619)
- `calculate_cluster_pvalues`: Correctly implements test routing based on is_categorical flag
- Visualization functions: Correctly call calculate_cluster_pvalues with appropriate flags

### Proposed Changes
1. Add variable type detection helper function (10-15 lines)
2. Modify analyze_cluster_vs_population to detect types and select tests (5-10 line modification)
3. Replace FDR column creation with dynamic insertion logic (15-20 line modification)
4. Add print statements for test transparency (1 line per test call)

**Total estimated addition: ~30-40 lines maximum**

## Components and Interfaces

### 1. Variable Type Detection Helper

```python
def _infer_variable_type(series: pd.Series, threshold: int = 5) -> str:
    """
    Infer if variable is continuous or categorical.
    
    Args:
        series: Data series to analyze
        threshold: Max unique values for categorical classification
    
    Returns:
        'continuous' or 'categorical'
    """
    unique_count = series.nunique()
    return 'categorical' if unique_count <= threshold else 'continuous'
```

**Design rationale:**
- Simple heuristic: ≤5 unique values = categorical, >5 = continuous
- Single responsibility: only type detection
- No external dependencies
- Reusable across functions if needed

### 2. Modified analyze_cluster_vs_population Function

**Changes required:**

**A. Add variable_types parameter (optional)**
```python
def analyze_cluster_vs_population(
    cluster_df: pd.DataFrame,
    wgc_variables: List[str],
    output_db_path: str,
    output_table_name: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    variable_types: Optional[Dict[str, str]] = None,  # NEW PARAMETER
    fdr_correction: bool = True,
    alpha: float = 0.05
    ) -> pd.DataFrame:
```

**B. Detect variable types before processing loop**
```python
# After loading name_map, before building results
if variable_types is None:
    print("  ⚠️ No variable types provided, inferring from data...")
    variable_types = {}
    for variable in wgc_variables:
        if variable in cluster_df.columns:
            variable_types[variable] = _infer_variable_type(cluster_df[variable])
```

**C. Replace hardcoded chi-squared test (line 595)**
```python
# OLD (line 595):
p_val = chi_squared_test(cluster_subset[variable], cluster_df[variable])

# NEW:
var_type = variable_types.get(variable, 'categorical')
if var_type == 'continuous':
    print(f"    Using Mann-Whitney U test (continuous)")
    p_val = mann_whitney_u_test(cluster_subset[variable], cluster_df[variable])
else:
    print(f"    Using chi-squared test (categorical)")
    p_val = chi_squared_test(cluster_subset[variable], cluster_df[variable])
```

**D. Replace FDR column creation (lines 607-619)**

Current approach appends all FDR columns at end. New approach inserts each FDR column immediately after its raw p-value column.

```python
# Apply FDR correction with dynamic insertion
if fdr_correction:
    print("  Applying FDR correction...")
    p_cols = [col for col in results_df.columns if 'p-value' in col and 'FDR' not in col]
    
    # Build new column order with FDR columns inserted adjacently
    new_columns = []
    for col in results_df.columns:
        new_columns.append(col)
        # If this is a raw p-value column, insert FDR column after it
        if col in p_cols:
            # Extract valid p-values
            pvals = pd.to_numeric(results_df[col], errors='coerce')
            valid_mask = pvals.notna() & (pvals != 'N/A')
            
            if valid_mask.sum() > 0:
                valid_pvals = pvals[valid_mask]
                _, corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
                
                # Create FDR column
                fdr_col = col.replace('p-value', 'p-value (FDR-corrected)')
                results_df[fdr_col] = np.nan
                results_df.loc[valid_mask, fdr_col] = corrected
                new_columns.append(fdr_col)
    
    # Reorder columns
    results_df = results_df[new_columns]
```

### 3. Verification of Existing Functions

**No code changes needed** - verification confirms:
- `cluster_continuous_distributions` (line 333): calls `calculate_cluster_pvalues(..., is_categorical=False)` ✓
- `cluster_categorical_distributions` (line 417): calls `calculate_cluster_pvalues(..., is_categorical=True)` ✓
- `calculate_cluster_pvalues` (lines 169-189): correctly routes to mann_whitney_u_test or chi_squared_test ✓

**Action:** Add print statement to calculate_cluster_pvalues for transparency:

```python
def calculate_cluster_pvalues(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str = 'cluster_id',
    is_categorical: bool = False
    ) -> Dict[int, float]:
    """..."""
    pvalues = {}
    clusters = sorted(cluster_df[cluster_col].unique())
    population_data = cluster_df[variable]
    
    # ADD THIS LINE:
    test_type = "chi-squared (categorical)" if is_categorical else "Mann-Whitney U (continuous)"
    print(f"    Using {test_type} test")
    
    for cluster_id in clusters:
        cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id][variable]
        
        if is_categorical:
            p_val = chi_squared_test(cluster_subset, population_data)
        else:
            p_val = mann_whitney_u_test(cluster_subset, population_data)
        
        pvalues[cluster_id] = p_val
    
    return pvalues
```

## Data Models

### Variable Types Dictionary
```python
variable_types: Dict[str, str] = {
    'age': 'continuous',
    'bmi': 'continuous',
    'total_followup_days': 'continuous',
    'wgc_medication': 'categorical',
    'wgc_medical_condition': 'categorical',
    # ... etc
}
```

**Usage:**
```python
# Option 1: Provide explicit types
results_df = analyze_cluster_vs_population(
    cluster_df=data,
    wgc_variables=['wgc_medication', 'age', 'bmi'],
    variable_types={'age': 'continuous', 'bmi': 'continuous', 'wgc_medication': 'categorical'},
    ...
)

# Option 2: Let system infer (with warning)
results_df = analyze_cluster_vs_population(
    cluster_df=data,
    wgc_variables=['wgc_medication', 'age', 'bmi'],
    # variable_types=None (default)
    ...
)
```

## Error Handling

### Type Detection Edge Cases
- **Empty series**: Return 'categorical' as safe default
- **All NaN values**: Return 'categorical' as safe default
- **Single unique value**: Return 'categorical'

```python
def _infer_variable_type(series: pd.Series, threshold: int = 10) -> str:
    """Infer if variable is continuous or categorical."""
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 'categorical'  # Safe default
    unique_count = clean_series.nunique()
    return 'categorical' if unique_count <= threshold else 'continuous'
```

### Statistical Test Failures
- Existing error handling in mann_whitney_u_test and chi_squared_test remains unchanged
- Both functions return np.nan or 1.0 on failure
- No additional error handling needed

### FDR Correction Edge Cases
- **No valid p-values**: Skip FDR column creation (existing behavior)
- **Single p-value**: FDR correction still applied (existing behavior)
- **Column reordering**: Use list comprehension to build new column order safely

## Testing Strategy

### Unit Testing Approach
1. **Test _infer_variable_type**
   - Continuous variable (>10 unique values)
   - Categorical variable (≤10 unique values)
   - Edge cases (empty, all NaN, single value)

2. **Test analyze_cluster_vs_population with variable types**
   - Explicit variable_types provided
   - Inferred variable_types (None provided)
   - Mixed continuous and categorical variables
   - Verify correct test selection via print output

3. **Test FDR column positioning**
   - Verify FDR columns appear immediately after raw p-value columns
   - Check column order: Mean/N, p-value, p-value (FDR-corrected) pattern
   - Multiple clusters (verify pattern repeats correctly)

4. **Integration test with real data**
   - Run analyze_cluster_vs_population with continuous variables
   - Verify Mann-Whitney U test is used (check print output)
   - Verify no "Low expected frequencies" warnings for continuous variables
   - Verify FDR columns positioned correctly in output table

### Validation Criteria
- ✓ No chi-squared test applied to continuous variables
- ✓ FDR columns adjacent to raw p-value columns
- ✓ Print statements show correct test selection
- ✓ Existing visualization functions unchanged and working
- ✓ Total code addition < 50 lines

## Implementation Notes

### Code Organization
- Add _infer_variable_type immediately after existing helper functions (after line 105)
- Modify analyze_cluster_vs_population in place (lines 540-660)
- Add single print statement to calculate_cluster_pvalues (after line 177)

### Backward Compatibility
- New variable_types parameter is optional (defaults to None)
- Existing function calls work without modification
- Inferred types provide reasonable defaults with warning message

### Performance Considerations
- Type inference adds negligible overhead (single pass through unique values)
- Column reordering is O(n) where n = number of columns (typically < 20)
- No impact on statistical computation performance

### Documentation Updates
- Update analyze_cluster_vs_population docstring to document variable_types parameter
- Add docstring for _infer_variable_type
- Update module docstring version to 1.1
