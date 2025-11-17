# Practical Testing Guide: WGC Visualization Pipeline

## Quick Start (5 minutes)

### Step 1: Open Your Notebook
Open `scripts/paper2_notebook.ipynb`

### Step 2: Add a New Cell After Your Data Preparation
After the cell where you run `run_descriptive_comparisons`, add this test cell:

```python
# =============================================================================
# QUICK TEST: WGC Visualizations
# =============================================================================

import sqlite3
import pandas as pd
import os
from descriptive_visualizations import (
    plot_distribution_comparison,
    plot_stacked_bar_comparison
)

# Setup
OUTPUT_DIR = "../outputs/test_wgc_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
with sqlite3.connect("../dbs/pnk_db2_p2_in.sqlite") as conn:
    df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
    pop_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)

print(f"Data loaded: {len(df)} WGC records, {len(pop_df)} population records")

# Test 1: Violin plot
plot_distribution_comparison(
    df=df,
    population_df=pop_df,
    variable='total_wl_%',
    group_col='mental_health',
    output_filename='test_violin.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir=OUTPUT_DIR
)

# Test 2: Stacked bar
plot_stacked_bar_comparison(
    df=df,
    population_df=pop_df,
    variable='5%_wl_achieved',
    group_col='eating_habits',
    output_filename='test_bar.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir=OUTPUT_DIR
)

print(f"✓ Plots saved to {OUTPUT_DIR}")
```

### Step 3: Run the Cell
- Press `Shift + Enter`
- You should see plots displayed in the notebook
- Files will be saved to `../outputs/test_wgc_viz/`

### Step 4: Verify Success
Check for:
- ✓ Console output showing data loaded
- ✓ Two plots displayed in the notebook
- ✓ Two PNG files in the output directory
- ✓ No error messages

---

## Understanding What You're Testing

### The Visualization Functions

**1. `plot_distribution_comparison` (Violin Plot)**
- **Purpose:** Compare distributions of a continuous variable between groups
- **Use case:** "How does weight loss distribution differ between people with/without mental health issues?"
- **What it shows:** Split violin plot with population on left, group on right

**2. `plot_stacked_bar_comparison` (Stacked Bar)**
- **Purpose:** Compare proportions of a binary outcome between groups
- **Use case:** "What percentage achieved 5% weight loss in each WGC group?"
- **What it shows:** Stacked bars with achievement rates, population reference line

**3. `plot_multi_lollipop` (Lollipop Plot)**
- **Purpose:** Show percent change from population mean across multiple variables/groups
- **Use case:** "How much do different WGC groups deviate from population averages?"
- **What it shows:** Horizontal lollipop chart with percent changes

---

## Common Testing Scenarios

### Scenario 1: Test All WGC Variables

```python
# Test each WGC variable systematically
wgc_vars = [
    'mental_health',
    'eating_habits',
    'physical_inactivity',
    'womens_health_and_pregnancy',
    'medication_disease_injury',
    'family_issues'
]

for wgc in wgc_vars:
    print(f"\nTesting {wgc}...")
    
    plot_distribution_comparison(
        df=df,
        population_df=pop_df,
        variable='total_wl_%',
        group_col=wgc,
        output_filename=f'test_{wgc}_violin.png',
        name_map_path='scripts/human_readable_variable_names.json',
        output_dir=OUTPUT_DIR
    )
    
    print(f"✓ {wgc} plot created")
```

### Scenario 2: Test Multiple Outcome Variables

```python
# Test different outcomes for one WGC
outcomes = [
    'total_wl_%',
    'baseline_bmi',
    'total_followup_days',
    'dietitian_visits'
]

for outcome in outcomes:
    print(f"\nTesting outcome: {outcome}...")
    
    plot_distribution_comparison(
        df=df,
        population_df=pop_df,
        variable=outcome,
        group_col='mental_health',
        output_filename=f'test_mental_health_{outcome}.png',
        name_map_path='scripts/human_readable_variable_names.json',
        output_dir=OUTPUT_DIR
    )
```

### Scenario 3: Test Binary Outcomes (Achievement Rates)

```python
# Test achievement rates across WGCs
binary_outcomes = [
    '5%_wl_achieved',
    '10%_wl_achieved',
    '15%_wl_achieved',
    'instant_dropout'
]

for outcome in binary_outcomes:
    print(f"\nTesting {outcome}...")
    
    plot_stacked_bar_comparison(
        df=df,
        population_df=pop_df,
        variable=outcome,
        group_col='eating_habits',
        output_filename=f'test_eating_{outcome}.png',
        name_map_path='scripts/human_readable_variable_names.json',
        output_dir=OUTPUT_DIR
    )
```

---

## Integration with Your Existing Analysis

### Where to Add Visualizations in Your Workflow

Your current notebook structure:
1. Data preparation (database subsetting)
2. Time-to-event tables
3. **Block I - Comparative analyses** ← Add visualizations here
4. Other analyses...

### Example: Add After Descriptive Comparisons

```python
# Your existing code
from descriptive_comparisons import run_descriptive_comparisons
run_descriptive_comparisons(config)

# NEW: Add visualizations
print("\n" + "="*60)
print("Generating WGC Visualizations")
print("="*60)

# Load the analyzed data
with sqlite3.connect(paths.paper_in_db) as conn:
    wgc_df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
    pop_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)

# Generate key visualizations
viz_output = "../outputs/wgc_visualizations"
os.makedirs(viz_output, exist_ok=True)

# 1. Weight loss by mental health
plot_distribution_comparison(
    df=wgc_df,
    population_df=pop_df,
    variable='total_wl_%',
    group_col='mental_health',
    output_filename='wgc_mental_health_wl_distribution.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir=viz_output
)

# 2. Achievement rates by eating habits
plot_stacked_bar_comparison(
    df=wgc_df,
    population_df=pop_df,
    variable='5%_wl_achieved',
    group_col='eating_habits',
    output_filename='wgc_eating_habits_achievement.png',
    name_map_path='scripts/human_readable_variable_names.json',
    output_dir=viz_output
)

print(f"✓ Visualizations saved to {viz_output}")
```

---

## Checking Your Results

### What Good Output Looks Like

**Console Output:**
```
Data loaded: 2463 WGC records, 17680 population records
✓ Split-violin plot saved to: ../outputs/test_wgc_viz/test_violin.png
✓ Corrected stacked bar plot saved to: ../outputs/test_wgc_viz/test_bar.png
✓ Plots saved to ../outputs/test_wgc_viz
```

**Visual Output:**
- Plots appear in the notebook cell output
- Plots have clear titles and labels
- Colors are distinct and readable
- Axes are properly labeled with human-readable names

**File Output:**
- PNG files exist in the output directory
- Files are not empty (check file size > 50KB)
- Files open correctly in image viewer

### What to Check in the Plots

**Violin Plot:**
- ✓ Left side shows population distribution (blue)
- ✓ Right side shows group distribution (orange)
- ✓ Groups are labeled on x-axis
- ✓ Variable name is human-readable on y-axis
- ✓ Title describes what's being compared

**Stacked Bar Plot:**
- ✓ Bars show two colors (achieved vs not achieved)
- ✓ Reference line shows population mean
- ✓ Sample sizes (n=X) shown above bars
- ✓ Percentages add up to 100%
- ✓ Groups are clearly labeled

---

## Troubleshooting

### Error: "Table not found"
**Problem:** Database tables don't exist  
**Solution:** Run the data preparation cells first

```python
# Check what tables exist
with sqlite3.connect("../dbs/pnk_db2_p2_in.sqlite") as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", 
        conn
    )
    print("Available tables:")
    print(tables)
```

### Error: "Column not found"
**Problem:** Variable name doesn't match database  
**Solution:** Check available columns

```python
# Check columns in your data
with sqlite3.connect("../dbs/pnk_db2_p2_in.sqlite") as conn:
    df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl LIMIT 1", conn)
    print("Available columns:")
    print(df.columns.tolist())
```

### Warning: "No valid data"
**Problem:** Variable has all missing values  
**Solution:** Choose a different variable or check data quality

```python
# Check data availability
print(df['total_wl_%'].describe())
print(f"Missing values: {df['total_wl_%'].isna().sum()}")
```

### Plots Don't Display
**Problem:** Not running in Jupyter notebook  
**Solution:** Make sure you're in a notebook environment, not a script

---

## Next Steps After Successful Testing

Once your WGC visualizations work:

1. **Generate production visualizations** for your paper
2. **Add significance markers** from your statistical tests
3. **Test cluster-based visualizations** (if you have cluster data)
4. **Create publication-ready figures** with custom styling

See `arch/cluster_visualization_usage_guide.md` for cluster-based analysis examples.
