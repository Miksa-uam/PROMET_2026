# Quick Test: WGC Visualization Pipeline

Copy and paste this code into a new notebook cell to test the WGC visualization pipeline with your actual data.

## Test Cell 1: Basic Setup and Violin Plot

```python
import sqlite3
import pandas as pd
import os
from descriptive_visualizations import plot_distribution_comparison

# Configuration
DB_PATH = "../dbs/pnk_db2_p2_in.sqlite"
OUTPUT_DIR = "../outputs/test_wgc_visualizations"
NAME_MAP_PATH = "scripts/human_readable_variable_names.json"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
    population_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)

print(f"✓ Loaded {len(df)} WGC records, {len(population_df)} population records")

# Test 1: Violin plot comparing mental health groups
print("\nGenerating violin plot...")
plot_distribution_comparison(
    df=df,
    population_df=population_df,
    variable='total_wl_%',
    group_col='mental_health',
    output_filename='test_mental_health_violin.png',
    name_map_path=NAME_MAP_PATH,
    output_dir=OUTPUT_DIR
)
print(f"✓ Plot saved to {OUTPUT_DIR}/test_mental_health_violin.png")
```

## Test Cell 2: Stacked Bar Plot

```python
from descriptive_visualizations import plot_stacked_bar_comparison

# Test 2: Stacked bar plot for 5% weight loss achievement
print("Generating stacked bar plot...")
plot_stacked_bar_comparison(
    df=df,
    population_df=population_df,
    variable='5%_wl_achieved',
    group_col='eating_habits',
    output_filename='test_eating_habits_bar.png',
    name_map_path=NAME_MAP_PATH,
    output_dir=OUTPUT_DIR
)
print(f"✓ Plot saved to {OUTPUT_DIR}/test_eating_habits_bar.png")
```

## Test Cell 3: Lollipop Plot (Multi-Variable)

```python
from descriptive_visualizations import plot_multi_lollipop

# Test 3: Lollipop plot showing percent change across multiple WGCs
print("Generating lollipop plot...")

# Prepare data
wgc_variables = ['mental_health', 'eating_habits', 'physical_inactivity']
outcome_variables = ['total_wl_%', 'baseline_bmi']

lollipop_data = []
for wgc_var in wgc_variables:
    for outcome_var in outcome_variables:
        wgc_yes = df[df[wgc_var] == 1]
        if len(wgc_yes) > 0:
            pop_mean = population_df[outcome_var].mean()
            wgc_mean = wgc_yes[outcome_var].mean()
            pct_change = ((wgc_mean - pop_mean) / pop_mean) * 100
            
            lollipop_data.append({
                'variable': outcome_var,
                'cluster': f'{wgc_var.replace("_", " ").title()}: Yes',
                'value': pct_change
            })

lollipop_df = pd.DataFrame(lollipop_data)

plot_multi_lollipop(
    data_df=lollipop_df,
    output_filename='test_wgc_lollipop.png',
    name_map_path=NAME_MAP_PATH,
    output_dir=OUTPUT_DIR
)
print(f"✓ Plot saved to {OUTPUT_DIR}/test_wgc_lollipop.png")
```

## What to Expect

After running these cells, you should see:

1. **Console Output:**
   - Data loading confirmation
   - Plot generation messages
   - File save confirmations

2. **Visual Output:**
   - Plots displayed directly in the notebook (via `plt.show()`)
   - Files saved to `../outputs/test_wgc_visualizations/`

3. **Files Created:**
   - `test_mental_health_violin.png` - Split violin plot
   - `test_eating_habits_bar.png` - Stacked bar chart
   - `test_wgc_lollipop.png` - Multi-variable lollipop plot

## Troubleshooting

**If you get "table not found" errors:**
- Check that `timetoevent_wgc_compl` and `timetoevent_all` tables exist in your database
- Run the data preparation cells in your notebook first

**If you get "column not found" errors:**
- Verify the column names match your database schema
- Check available columns with: `df.columns.tolist()`

**If plots don't display:**
- Make sure you're running in Jupyter notebook (not a script)
- Check that matplotlib is configured for inline display

## Next Steps

Once these basic tests work, you can:

1. **Test with different WGC variables:**
   - Replace `'mental_health'` with other WGC columns
   - Try: `'womens_health_and_pregnancy'`, `'medication_disease_injury'`, etc.

2. **Test with different outcome variables:**
   - Replace `'total_wl_%'` with other outcomes
   - Try: `'baseline_bmi'`, `'total_followup_days'`, `'dietitian_visits'`, etc.

3. **Add significance testing:**
   - Include `significance_map_raw` and `significance_map_fdr` parameters
   - These come from your statistical analysis results

4. **Move to cluster-based analysis:**
   - Once WGC visualizations work, test with cluster data
   - Add `cluster_config_path` parameter for cluster labels/colors
