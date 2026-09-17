# Descriptive Visualizations Notebook Interface

## Overview

This document provides instructions for using the simplified notebook interface for the comprehensive descriptive visualizations pipeline.

## Usage Instructions

### 1. Copy the Notebook Cell

Copy the code from `notebook_interface_descriptive_viz.py` into a new cell in your Jupyter notebook, positioned after the "#### I/2.2. Risk ratio/risk difference analyses" section.

### 2. Configure Input Table

The primary configurable parameter is `input_table`. Common options:

- `"timetoevent_wgc_compl"` - Standard weight gain cause complete dataset
- `"timetoevent_wgc_gen_compl"` - Weight gain cause + genomics complete dataset
- `"timetoevent_all"` - Full dataset

### 3. Run the Cell

Execute the cell to generate:

- **Risk ratio forest plots** (logarithmic scale, RR=1.0 reference line)
- **Risk difference forest plots** (linear scale, RD=0.0 reference line)
- **Summary tables** with comprehensive statistics
- **Organized outputs** in `../outputs/descriptive_visualizations/`

## Generated Outputs

### Forest Plots
- `risk_ratios_10pct_wl_achieved.png`
- `risk_differences_10pct_wl_achieved.png`
- `risk_ratios_60d_dropout.png`
- `risk_differences_60d_dropout.png`

### Summary Tables
- `effect_sizes_summary.csv` - Risk ratios, risk differences, and confidence intervals
- `statistical_tests_summary.csv` - P-values, test types, and FDR corrections

## Requirements Satisfied

- **6.1**: Simple, clean interface with minimal required parameters ✓
- **6.2**: Generates both risk ratio and risk difference plots for all outcomes ✓
- **6.3**: Saves outputs in organized subdirectories ✓
- **6.4**: Produces identical results given same input data ✓
- **6.5**: Positioned after "#### I/2.2. Risk ratio/risk difference analyses" section ✓

## Example Usage

```python
# Basic usage with default configuration
results = run_descriptive_visualizations(
    input_table="timetoevent_wgc_compl",
    config=config
)

# Alternative input table
results = run_descriptive_visualizations(
    input_table="timetoevent_wgc_gen_compl", 
    config=config
)
```

## Integration Notes

- Reuses existing `paper12_config` for database paths
- Compatible with existing project infrastructure
- Follows established output directory conventions
- Maintains consistency with other analysis notebooks