# Design Document

## Overview

This design implements optional False Discovery Rate (FDR) correction using the Benjamini-Hochberg method for the descriptive comparisons pipeline. The solution extends the existing `descriptive_comparisons_config` dataclass with an optional boolean parameter and modifies the statistical comparison functions to collect, correct, and format p-values appropriately.

The design maintains full backward compatibility while providing researchers with the flexibility to apply multiple testing correction when needed. FDR correction is applied separately to different comparison contexts (demographic vs. weight gain cause stratifications) to ensure statistical appropriateness.

## Architecture

### Configuration Layer
- **Modified `descriptive_comparisons_config`**: Add optional `fdr_correction: bool = False` parameter
- **Backward Compatibility**: Default value ensures existing configurations continue working unchanged

### Statistical Processing Layer
- **P-value Collection**: Gather all p-values within each stratification context
- **FDR Correction Engine**: Apply Benjamini-Hochberg correction using statsmodels
- **Result Integration**: Merge corrected p-values back into result tables with clear labeling

### Output Layer
- **Dual Column Strategy**: Preserve original p-values and add corrected columns when enabled
- **Human-Readable Naming**: Use "(FDR-corrected)" suffix for clarity
- **Conditional Output**: Only create corrected columns when FDR correction is enabled

## Components and Interfaces

### 1. Configuration Component

**Modified `descriptive_comparisons_config` dataclass:**
```python
@dataclass
class descriptive_comparisons_config:
    # ... existing fields ...
    fdr_correction: bool = False  # New optional parameter
```

### 2. FDR Correction Engine

**New `fdr_correction_utils.py` module (Reusable across analysis contexts):**
```python
def apply_fdr_correction(p_values: List[float], method: str = 'fdr_bh') -> List[float]:
    """
    Apply FDR correction to a list of p-values using statsmodels.
    Designed for reuse across descriptive comparisons, regression analyses, etc.
    """
    
def collect_pvalues_from_dataframe(df: pd.DataFrame, pvalue_columns: List[str]) -> Dict[str, List[float]]:
    """
    Extract p-values from specified columns in a DataFrame.
    Generic utility for any analysis that stores p-values in DataFrame columns.
    """
    
def integrate_corrected_pvalues(df: pd.DataFrame, corrections: Dict[str, List[float]], 
                               suffix: str = " (FDR-corrected)") -> pd.DataFrame:
    """
    Add FDR-corrected p-value columns to DataFrame with configurable suffix.
    Supports different naming conventions for various analysis types.
    """

def format_pvalue_for_output(p_value: float, threshold: float = 0.001) -> str:
    """
    Format p-values for publication-ready output (e.g., '<0.001', '0.045').
    Useful for regression tables, comparison tables, etc.
    """
```

### 3. Modified Stratification Functions

**Enhanced `demographic_stratification()` function:**
- Collect p-values from: "Cohort comparison: p-value", "Age: p-value", "Gender: p-value", "BMI: p-value"
- Apply FDR correction if enabled
- Add corrected columns with "(FDR-corrected)" suffix

**Enhanced `wgc_stratification()` function:**
- Collect p-values from all "{cause_name}: p-value" columns
- Apply FDR correction if enabled
- Add corrected columns with "(FDR-corrected)" suffix

## Data Models

### P-value Collection Structure
```python
PValueCollection = Dict[str, List[float]]
# Example:
{
    "Cohort comparison: p-value": [0.05, 0.02, 0.001, ...],
    "Age: p-value": [0.1, 0.03, 0.08, ...],
    "Gender: p-value": [0.2, 0.01, 0.15, ...],
    "BMI: p-value": [0.07, 0.04, 0.12, ...]
}
```

### Correction Result Structure
```python
CorrectionResult = Dict[str, List[float]]
# Example:
{
    "Cohort comparison: p-value (FDR-corrected)": [0.08, 0.04, 0.002, ...],
    "Age: p-value (FDR-corrected)": [0.15, 0.06, 0.12, ...],
    "Gender: p-value (FDR-corrected)": [0.25, 0.03, 0.20, ...],
    "BMI: p-value (FDR-corrected)": [0.11, 0.07, 0.16, ...]
}
```

## Error Handling

### Robust P-value Processing
1. **NaN Handling**: Filter out NaN values before FDR correction, preserve NaN in corrected results
2. **Empty Collections**: Skip FDR correction and log warning if no valid p-values exist
3. **Statsmodels Errors**: Catch exceptions from FDR correction and fall back to raw p-values with error logging
4. **Data Type Validation**: Ensure p-values are numeric before processing

### Logging Strategy
- **Info Level**: Log when FDR correction is applied and number of p-values corrected
- **Warning Level**: Log when FDR correction is skipped due to insufficient data
- **Error Level**: Log when FDR correction fails and fallback is used

## Testing Strategy

### Unit Tests
1. **FDR Correction Engine Tests**:
   - Test with valid p-values
   - Test with mixed valid/NaN p-values
   - Test with all NaN p-values
   - Test with empty input
   - Test edge cases (all zeros, all ones)

2. **Configuration Tests**:
   - Test default behavior (fdr_correction=False)
   - Test enabled behavior (fdr_correction=True)
   - Test backward compatibility with existing configs

3. **Integration Tests**:
   - Test demographic stratification with FDR correction
   - Test weight gain cause stratification with FDR correction
   - Test output table structure and column naming

### Validation Tests
1. **Statistical Validation**:
   - Verify FDR correction produces expected results using known test cases
   - Compare with manual Benjamini-Hochberg calculations
   - Validate that corrected p-values are appropriately adjusted

2. **Data Integrity Tests**:
   - Ensure original p-values remain unchanged
   - Verify corrected columns are only added when enabled
   - Test that table structure matches expectations

### Performance Tests
1. **Scalability**: Test with large numbers of p-values (1000+ comparisons)
2. **Memory Usage**: Ensure FDR correction doesn't significantly increase memory footprint
3. **Processing Time**: Verify minimal impact on overall analysis runtime

## Implementation Approach

### Phase 1: Reusable FDR Infrastructure
1. Create `fdr_correction_utils.py` module with generic, reusable correction functions
2. Design functions to support multiple analysis contexts (descriptive comparisons, regressions, etc.)
3. Add `fdr_correction` parameter to `descriptive_comparisons_config`
4. Implement comprehensive unit tests for FDR correction engine

### Phase 2: Integration with Descriptive Comparisons
1. Modify `demographic_stratification()` to use the reusable FDR utilities
2. Modify `wgc_stratification()` to use the reusable FDR utilities
3. Implement p-value collection and integration logic specific to descriptive comparisons

### Phase 3: Testing and Validation
1. Create comprehensive test suite covering both generic utilities and specific integrations
2. Validate statistical correctness against known benchmarks
3. Test backward compatibility with existing notebooks
4. Test reusability by creating example usage for different analysis types

### Phase 4: Documentation and Future-Proofing
1. Document the reusable nature of `fdr_correction_utils.py` for future analysis extensions
2. Create example notebook demonstrating FDR correction usage in descriptive comparisons
3. Provide guidance for integrating FDR correction into other analysis pipelines (regressions, etc.)
4. Document performance characteristics and limitations

## Future Extensibility

The `fdr_correction_utils.py` module is designed to support FDR correction across the entire analysis pipeline:

- **Regression Analyses**: Can be used to correct p-values from multiple regression models
- **Subgroup Analyses**: Applicable to any analysis involving multiple statistical tests
- **Sensitivity Analyses**: Supports correction across different analytical approaches
- **Publication Tables**: Provides consistent p-value formatting across all analysis outputs

This modular approach ensures that FDR correction can be consistently applied across all statistical analyses in the research pipeline.