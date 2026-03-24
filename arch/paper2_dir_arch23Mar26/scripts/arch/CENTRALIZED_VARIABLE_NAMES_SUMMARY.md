# Centralized Variable Names System

## Overview

Implemented a centralized system for managing human-readable variable names across the entire project. This eliminates the need for maintaining multiple hardcoded dictionaries and ensures consistency across all scripts.

## Files Created/Modified

### 📁 New Files

1. **`human_readable_variable_names.json`**
   - Central dictionary with all variable name mappings
   - Organized by categories (demographics, treatment outcomes, time windows, etc.)
   - Uses comments for readability (filtered out programmatically)
   - Includes all dynamic variables for 40/60/80 day windows and 5%/10%/15% weight loss targets

2. **`variable_names_utils.py`**
   - Utility functions for loading and accessing variable names
   - Caching mechanism to avoid repeated file reads
   - Helper functions: `get_human_readable_name()`, `variable_exists()`, etc.

3. **`test_variable_names.py`**
   - Comprehensive test suite for the centralized system
   - Tests utility functions and RF integration

### 🔧 Modified Files

1. **`paper12_config.py`**
   - **REMOVED**: `nice_names` dictionary from `paper2_rf_config`
   - Now uses centralized system instead

2. **`rf_feature_importances.py`**
   - **ADDED**: Import for `get_human_readable_name`
   - **UPDATED**: `_get_nice_name()` method to use centralized dictionary
   - No more dependency on config's nice_names

## Key Benefits

### ✅ **Centralized Management**
- Single source of truth for all variable names
- Easy to update names across entire project
- No more scattered hardcoded dictionaries

### ✅ **Consistency**
- All scripts use the same variable names
- Eliminates discrepancies between different modules
- Standardized naming conventions

### ✅ **Maintainability**
- Edit JSON file once, changes apply everywhere
- Clear separation of data (JSON) and logic (Python)
- Easy to add new variables

### ✅ **Flexibility**
- Supports dynamic variables (time windows, weight loss targets)
- Graceful fallback to original name if not found
- Easy to extend for new variable categories

## Usage Examples

### Basic Usage
```python
from variable_names_utils import get_human_readable_name

# Get human-readable name
nice_name = get_human_readable_name("baseline_bmi")
# Returns: "Baseline BMI (kg/m²)"

# Check if variable exists
exists = variable_exists("some_variable")
```

### RF Pipeline Usage
```python
# OLD WAY (removed):
config = paper2_rf_config(
    # ... other params ...
    nice_names={
        "age": "Age (years)",
        "baseline_bmi": "Baseline BMI",
        # ... many more ...
    }
)

# NEW WAY (automatic):
config = paper2_rf_config(
    # ... other params ...
    # No nice_names needed - loaded automatically!
)
```

## Variable Categories Included

### 📊 Demographics & Baseline Anthropometry
- `age`, `sex_f`, `height_m`, `baseline_weight_kg`, `baseline_bmi`, etc.

### 📈 Treatment Outcomes
- `total_followup_days`, `dietitian_visits`, `total_wl_%`, `bmi_reduction`, etc.

### ⏰ Time Window Variables (40, 60, 80 days)
- `40d_dropout`, `40d_wl_%`, `60d_bmi_reduction`, `80d_fat_loss_%`, etc.

### 🎯 Weight Loss Targets (5%, 10%, 15%)
- `5%_wl_achieved`, `days_to_10%_wl`, `15%_wl_achieved`, etc.

### 🔍 Weight Gain Causes
- `womens_health_and_pregnancy`, `mental_health`, `family_issues`, etc.

## Testing

Run the test suite to verify everything works:

```bash
cd scripts
python test_variable_names.py
```

Expected output:
- ✅ Variable names utility tests passed
- ✅ RF integration tests passed
- 🎉 ALL TESTS PASSED

## Migration Guide for Other Scripts

To adapt other scripts to use this system:

1. **Add import**:
   ```python
   from variable_names_utils import get_human_readable_name
   ```

2. **Replace hardcoded dictionaries**:
   ```python
   # OLD:
   nice_names = {"age": "Age (years)", "baseline_bmi": "Baseline BMI"}
   display_name = nice_names.get(variable_name, variable_name)
   
   # NEW:
   display_name = get_human_readable_name(variable_name)
   ```

3. **Remove config parameters**:
   - Remove any `nice_names` or similar parameters from config classes
   - The centralized system handles this automatically

## Future Enhancements

- **Multi-language support**: Extend JSON to support multiple languages
- **Variable validation**: Add schema validation for the JSON file
- **Auto-generation**: Generate variable lists from database schemas
- **Documentation**: Auto-generate variable documentation from the JSON

## Troubleshooting

### Variable Not Found
If a variable returns its original name instead of a human-readable version:
1. Check if it exists in `human_readable_variable_names.json`
2. Verify spelling (case-sensitive)
3. Add it to the JSON file if missing

### Import Errors
Ensure `variable_names_utils.py` is in the same directory as your script, or adjust the import path accordingly.

### JSON Syntax Errors
Validate the JSON file if you get parsing errors:
```bash
python -m json.tool human_readable_variable_names.json
```