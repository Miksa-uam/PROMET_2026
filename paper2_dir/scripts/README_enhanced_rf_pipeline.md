# Enhanced Random Forest Pipeline Documentation

## Overview

The Enhanced Random Forest Pipeline provides a comprehensive framework for Random Forest feature importance analysis with robust statistical significance testing and publication-ready visualizations. This enhancement adds sophisticated statistical rigor while maintaining full backward compatibility with existing workflows.

## Key Features

### 🔬 Statistical Significance Testing
- **Shadow Feature Testing**: Data-driven significance thresholds for Gini importance using permutation-based shadow features
- **SHAP Significance Testing**: Wilcoxon signed-rank tests with Benjamini-Hochberg FDR correction for SHAP values
- **Robust Statistical Methods**: Non-parametric tests suitable for non-normal distributions

### 📊 Enhanced Visualizations
- **Primary Composite Plot**: Gini importance + SHAP beeswarm with significance annotations
- **Secondary Composite Plot**: Mean absolute SHAP + Permutation importance
- **Publication-Ready Quality**: Professional styling, dynamic sizing, clear significance indicators

### ⚙️ Unified Configuration
- **Single Configuration File**: All settings managed through `paper12_config.py`
- **Enhanced Validation**: Comprehensive parameter validation with informative error messages
- **Backward Compatibility**: Existing configurations continue to work unchanged

## Quick Start

### Basic Usage

```python
from paper12_config import paper2_rf_config
from rf_engine import RandomForestAnalyzer

# Configure analysis
config = paper2_rf_config(
    analysis_name='weight_loss_analysis',
    outcome_variable='weight_loss_10pct',
    model_type='classifier',
    predictors=['age', 'bmi', 'gender', 'medication'],
    classifier_threshold=0.1,
    threshold_direction='greater_than_or_equal',
    db_path='../dbs/my_database.sqlite',
    input_table='analysis_data'
)

# Run enhanced analysis
analyzer = RandomForestAnalyzer(config)
analyzer.run_and_generate_outputs()
```

### Advanced Configuration

```python
# Enhanced configuration with significance testing options
config = paper2_rf_config(
    analysis_name='advanced_analysis',
    outcome_variable='outcome',
    model_type='classifier',
    predictors=['feature_1', 'feature_2', 'feature_3'],
    
    # Statistical testing configuration
    enable_gini_significance=True,      # Enable shadow feature testing
    enable_shap_significance=True,      # Enable SHAP significance testing
    significance_alpha=0.05,            # Significance level
    
    # Visualization configuration
    figure_width_primary=16.0,          # Primary plot width
    figure_height_primary=10.0,         # Primary plot height
    figure_width_secondary=16.0,        # Secondary plot width
    figure_height_secondary=10.0,       # Secondary plot height
    max_features_display=None,          # Show all features (no collapsing)
    
    # Standard configuration
    classifier_threshold=0.5,
    threshold_direction='greater_than_or_equal',
    db_path='data.sqlite',
    input_table='my_table'
)
```

## Configuration Options

### Core Parameters
- `analysis_name`: Unique identifier for the analysis
- `outcome_variable`: Target variable column name
- `model_type`: 'classifier' or 'regressor'
- `predictors`: List of predictor variable names
- `covariates`: List of covariate variable names (optional)

### Statistical Testing Parameters
- `enable_gini_significance`: Enable shadow feature testing for Gini importance (default: True)
- `enable_shap_significance`: Enable Wilcoxon testing for SHAP values (default: True)
- `significance_alpha`: Significance level for statistical tests (default: 0.05)

### Visualization Parameters
- `figure_width_primary`: Width of primary composite plot (default: 16.0)
- `figure_height_primary`: Height of primary composite plot (default: 10.0)
- `figure_width_secondary`: Width of secondary composite plot (default: 16.0)
- `figure_height_secondary`: Height of secondary composite plot (default: 10.0)
- `max_features_display`: Maximum features to display (None = show all)

### Model Parameters
- `run_hyperparameter_tuning`: Enable hyperparameter optimization (default: False)
- `classifier_threshold`: Threshold for binary classification
- `threshold_direction`: 'greater_than_or_equal' or 'less_than_or_equal'

### Data Parameters
- `db_path`: Path to SQLite database
- `input_table`: Table name containing analysis data
- `output_dir`: Directory for saving outputs (default: '../outputs/rf_outputs')

## Output Files

The enhanced pipeline generates three main outputs:

### 1. ROC Curve (Classifiers Only)
- **File**: `{analysis_name}_roc_curve.png`
- **Content**: ROC curve with AUROC score
- **When**: Generated for classifier models only

### 2. Primary Composite Plot
- **File**: `{analysis_name}_primary_FI_composite.png`
- **Content**: 
  - Left panel: Gini importance with significance threshold line
  - Right panel: SHAP beeswarm plot with significance annotations
- **Features**: Statistical significance clearly marked with asterisks and threshold lines

### 3. Secondary Composite Plot
- **File**: `{analysis_name}_secondary_FI_composite.png`
- **Content**:
  - Left panel: Mean absolute SHAP values
  - Right panel: Permutation importance
- **Features**: Complementary importance metrics without significance testing

## Statistical Methods

### Shadow Feature Testing (Gini Importance)

The shadow feature method provides a data-driven approach to determine significance thresholds:

1. **Shadow Feature Creation**: Each original feature is duplicated and its values randomly shuffled
2. **Augmented Training**: Random Forest trained on combined original + shadow features
3. **Threshold Calculation**: Maximum shadow feature importance becomes the significance threshold
4. **Significance Determination**: Original features exceeding threshold are considered significant

**Advantages**:
- Data-driven threshold (no arbitrary cutoffs)
- Accounts for dataset-specific noise levels
- Robust to different feature scales and distributions

### SHAP Significance Testing

SHAP values are tested using rigorous statistical methods:

1. **Wilcoxon Signed-Rank Test**: Non-parametric test for each feature (H₀: median SHAP = 0)
2. **Multiple Comparison Correction**: Benjamini-Hochberg FDR correction applied
3. **Significance Determination**: Features with adjusted p-values < α are significant

**Advantages**:
- Non-parametric (no normality assumptions)
- Controls false discovery rate
- Appropriate for SHAP value distributions

## Usage Examples

### Example 1: Basic Weight Loss Analysis

```python
from paper12_config import paper2_rf_config
from rf_engine import RandomForestAnalyzer

# Configure for weight loss prediction
config = paper2_rf_config(
    analysis_name='weight_loss_10pct_analysis',
    outcome_variable='weight_loss_10pct',
    model_type='classifier',
    predictors=[
        'age', 'baseline_bmi', 'sex_f',
        'mental_health', 'eating_habits', 'physical_inactivity'
    ],
    classifier_threshold=0.1,
    threshold_direction='greater_than_or_equal',
    db_path='../dbs/weight_loss_data.sqlite',
    input_table='patient_outcomes',
    nice_names={
        'age': 'Age (years)',
        'baseline_bmi': 'Baseline BMI',
        'sex_f': 'Sex (Female)',
        'mental_health': 'Mental Health Issues',
        'eating_habits': 'Poor Eating Habits',
        'physical_inactivity': 'Physical Inactivity'
    }
)

# Run analysis
analyzer = RandomForestAnalyzer(config)
analyzer.run_and_generate_outputs()

# Results will show:
# - Which features are statistically significant for predicting weight loss
# - Publication-ready plots with significance annotations
# - Comprehensive statistical summary
```

### Example 2: Regression Analysis

```python
# Configure for continuous outcome
config = paper2_rf_config(
    analysis_name='weight_change_regression',
    outcome_variable='weight_change_kg',
    model_type='regressor',  # Note: regressor for continuous outcomes
    predictors=['age', 'baseline_bmi', 'treatment_duration'],
    covariates=['sex_f', 'clinic_site'],  # Control variables
    db_path='../dbs/treatment_data.sqlite',
    input_table='treatment_outcomes'
)

analyzer = RandomForestAnalyzer(config)
analyzer.run_and_generate_outputs()
```

### Example 3: Custom Visualization Settings

```python
# Configure with custom visualization parameters
config = paper2_rf_config(
    analysis_name='custom_viz_analysis',
    outcome_variable='outcome',
    model_type='classifier',
    predictors=['var1', 'var2', 'var3', 'var4', 'var5'],
    classifier_threshold=0.5,
    threshold_direction='greater_than_or_equal',
    
    # Custom visualization settings
    figure_width_primary=20.0,      # Wider plots for more features
    figure_height_primary=12.0,     # Taller plots for readability
    figure_width_secondary=18.0,
    figure_height_secondary=10.0,
    significance_alpha=0.01,        # More stringent significance level
    
    db_path='data.sqlite',
    input_table='analysis_data'
)

analyzer = RandomForestAnalyzer(config)
analyzer.run_and_generate_outputs()
```

## Backward Compatibility

### Legacy Method Available

For users who need the old visualization format:

```python
analyzer = RandomForestAnalyzer(config)
analyzer.run_and_generate_outputs_legacy()  # Generates old-style plots
```

### Existing Configurations

All existing `paper2_rf_config` configurations continue to work without modification. New features are enabled by default but can be disabled:

```python
config = paper2_rf_config(
    # ... existing configuration ...
    enable_gini_significance=False,  # Disable shadow feature testing
    enable_shap_significance=False   # Disable SHAP significance testing
)
```

## Performance Considerations

### Memory Usage
- **Shadow Features**: Doubles memory usage during Gini significance testing
- **Typical Usage**: 2-5 MB additional memory for datasets with <1000 samples
- **Large Datasets**: Consider sampling for shadow features if memory is constrained

### Computation Time
- **Shadow Feature Testing**: Adds ~0.5-1.0 seconds for typical datasets
- **SHAP Significance**: Minimal overhead (~0.01-0.1 seconds)
- **Visualization**: Comparable to original plotting time

### Optimization Tips
- For very large datasets (>10k samples), consider sampling for shadow features
- SHAP significance testing scales well with number of features
- Memory usage is dominated by shadow feature storage (2x original dataset)

## Troubleshooting

### Common Issues

**Issue**: "significance_alpha must be between 0 and 1"
**Solution**: Ensure `significance_alpha` is set to a value between 0 and 1 (e.g., 0.05)

**Issue**: "Primary figure dimensions must be positive"
**Solution**: Ensure `figure_width_primary` and `figure_height_primary` are positive numbers

**Issue**: Significance testing fails
**Solution**: The pipeline will automatically fall back to standard analysis without significance testing

### Error Handling

The enhanced pipeline includes comprehensive error handling:
- Statistical testing failures fall back to standard analysis
- Visualization failures fall back to individual plots
- Configuration errors provide clear guidance for fixes

### Getting Help

1. Check configuration parameters match the documentation
2. Verify database path and table names are correct
3. Ensure predictor variables exist in the specified table
4. Review console output for detailed error messages and warnings

## Advanced Usage

### Accessing Significance Results

```python
analyzer = RandomForestAnalyzer(config)
analyzer.run_analysis()  # Run analysis without generating plots
analyzer._test_feature_significance()  # Run significance testing

# Access results
sig_results = analyzer.results['significance_results']
print(f"Gini significant features: {sig_results.gini_significant_features}")
print(f"SHAP significant features: {sig_results.shap_significant_features}")
print(f"Significance threshold: {sig_results.gini_threshold}")
```

### Custom Plotting

```python
# Use enhanced plotter directly
from enhanced_visualization import EnhancedFeatureImportancePlotter

plotter = EnhancedFeatureImportancePlotter(config, config.nice_names)

# Create individual plots
plotter.plot_primary_composite(
    gini_importance=analyzer.results['gini_importance'],
    shap_explanation=analyzer.results['shap_explanation'],
    significance_results=analyzer.results['significance_results'],
    output_path='custom_primary_plot.png'
)
```

## Version History

### v2.0 (Enhanced Pipeline)
- Added statistical significance testing for Gini and SHAP importance
- Implemented shadow feature permutation method
- Added Wilcoxon signed-rank tests with FDR correction
- Created publication-ready composite visualizations
- Enhanced configuration system with validation
- Comprehensive error handling and logging

### v1.0 (Original Pipeline)
- Basic Random Forest analysis
- Standard feature importance calculations
- Individual plot generation
- Basic configuration system

---

For additional support or questions, please refer to the inline code documentation or contact the development team.