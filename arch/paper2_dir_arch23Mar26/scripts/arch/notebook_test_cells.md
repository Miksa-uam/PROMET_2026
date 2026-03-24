# Individual Test Cells for Notebook

Copy each cell below into your notebook and run them one at a time.

---

## Cell 1: Load Data

```python
from cluster_descriptions_new import load_and_merge_cluster_data

cluster_df, pop_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all"
)

print(f"✓ Loaded {len(cluster_df)} cluster records")
print(f"✓ Loaded {len(pop_df)} population records")
print(f"✓ Clusters: {sorted(cluster_df['cluster_id'].unique())}")
```

**Expected output:**
- Loading messages
- Cluster counts
- List of cluster IDs [0, 1, 2, 3, 4, 5, 6]

---

## Cell 2: Test Violin Plots

```python
from cluster_descriptions_new import plot_cluster_distributions

plot_cluster_distributions(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['total_wl_%', 'baseline_bmi'],
    output_dir="../outputs/test_cluster",
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    calculate_significance=True,
    fdr_correction=True
)
```

**Expected output:**
- Processing messages for each variable
- Plots displayed in notebook
- Files saved to `../outputs/test_cluster/`
- Significance markers (* or **) on plots

---

## Cell 3: Test Stacked Bar Plots

```python
from cluster_descriptions_new import plot_cluster_categorical

plot_cluster_categorical(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['sex_f', '5%_wl_achieved'],
    output_dir="../outputs/test_cluster",
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    calculate_significance=True,
    fdr_correction=True,
    legend_labels={'achieved': 'Yes', 'not_achieved': 'No'}
)
```

**Expected output:**
- Processing messages
- Stacked bar plots displayed
- Population reference line visible
- Sample sizes (n=X) above bars
- Significance markers on clusters

---

## Cell 4: Test Statistical Analysis

```python
from cluster_descriptions_new import analyze_cluster_vs_population

results_df = analyze_cluster_vs_population(
    cluster_df=cluster_df,
    population_df=pop_df,
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity'],
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="test_cluster_wgc",
    name_map_path='human_readable_variable_names.json',
    fdr_correction=True
)

# Display results
print("\nResults preview:")
print(results_df.head())
```

**Expected output:**
- Processing messages for each WGC
- FDR correction applied
- Two tables saved to database
- DataFrame displayed with cluster columns

---

## Cell 5: Test Heatmap

```python
from cluster_descriptions_new import plot_cluster_heatmap

plot_cluster_heatmap(
    results_df=results_df,
    output_filename="test_heatmap.png",
    output_dir="../outputs/test_cluster",
    wgc_variables=['mental_health', 'eating_habits', 'physical_inactivity'],
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json'
)
```

**Expected output:**
- Heatmap displayed in notebook
- Cells show "n (%)** " format
- Colors represent percentages (0-100%)
- Cluster labels are human-readable
- WGC labels are human-readable
- Significance markers (* or **) in cells

---

## Cell 6: Test Lollipop Plot

```python
from cluster_descriptions_new import plot_cluster_lollipop

plot_cluster_lollipop(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['total_wl_%', 'baseline_bmi', 'dietitian_visits'],
    output_filename="test_lollipop.png",
    output_dir="../outputs/test_cluster",
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json'
)
```

**Expected output:**
- Lollipop plot displayed
- Multiple variables shown (rows)
- Multiple clusters shown (colors)
- Percent change from population mean
- Cluster labels in legend
- Cluster colors from config

---

## Cell 7: Verify Database Tables

```python
import sqlite3
import pandas as pd

# Check what tables were created
with sqlite3.connect("../dbs/pnk_db2_p2_out.sqlite") as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'test_cluster%'",
        conn
    )
    print("Tables created:")
    print(tables)
    
    # Show detailed table
    print("\nDetailed table preview:")
    detailed = pd.read_sql_query("SELECT * FROM test_cluster_wgc_detailed LIMIT 5", conn)
    print(detailed)
    
    # Show publication table
    print("\nPublication table preview:")
    pub = pd.read_sql_query("SELECT * FROM test_cluster_wgc LIMIT 5", conn)
    print(pub)
```

**Expected output:**
- Two tables listed: `test_cluster_wgc_detailed` and `test_cluster_wgc`
- Detailed table has p-value columns
- Publication table has asterisks instead of p-values

---

## Cell 8: Check Output Files

```python
import os

output_dir = "../outputs/test_cluster"
files = os.listdir(output_dir) if os.path.exists(output_dir) else []

print(f"Files in {output_dir}:")
for f in sorted(files):
    size = os.path.getsize(os.path.join(output_dir, f)) / 1024
    print(f"  {f} ({size:.1f} KB)")
```

**Expected output:**
- List of PNG files
- Violin plots: `total_wl_%_violin.png`, `baseline_bmi_violin.png`
- Bar plots: `sex_f_bar.png`, `5%_wl_achieved_bar.png`
- Heatmap: `test_heatmap.png`
- Lollipop: `test_lollipop.png`

---

## Troubleshooting Cell

If you encounter errors, run this to check your setup:

```python
import os
import sys

print("Python version:", sys.version)
print("\nCurrent directory:", os.getcwd())
print("\nChecking files:")

files_to_check = [
    'human_readable_variable_names.json',
    'cluster_config.json',
    '../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite',
    '../dbs/pnk_db2_p2_in.sqlite'
]

for f in files_to_check:
    exists = "✓" if os.path.exists(f) else "✗"
    print(f"  {exists} {f}")

print("\nCluster data columns:")
print(cluster_df.columns.tolist()[:10], "...")

print("\nCluster IDs:")
print(sorted(cluster_df['cluster_id'].unique()))
```

---

## Complete Test (All in One)

If you want to run everything at once:

```python
from cluster_descriptions_new import *

# 1. Load
cluster_df, pop_df = load_and_merge_cluster_data(
    "../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    "../dbs/pnk_db2_p2_in.sqlite",
    "clust_labels_bl_nobc_bw_pam_goldstd",
    "pam_k7",
    "timetoevent_wgc_compl",
    "timetoevent_all"
)

# 2. Violin
plot_cluster_distributions(
    cluster_df, pop_df,
    ['total_wl_%', 'baseline_bmi'],
    "../outputs/test_all",
    calculate_significance=True,
    fdr_correction=True
)

# 3. Bar
plot_cluster_categorical(
    cluster_df, pop_df,
    ['sex_f', '5%_wl_achieved'],
    "../outputs/test_all",
    calculate_significance=True,
    fdr_correction=True
)

# 4. Analysis
results_df = analyze_cluster_vs_population(
    cluster_df, pop_df,
    ['mental_health', 'eating_habits', 'physical_inactivity'],
    "../dbs/pnk_db2_p2_out.sqlite",
    "test_all_wgc",
    fdr_correction=True
)

# 5. Heatmap
plot_cluster_heatmap(
    results_df,
    "test_heatmap.png",
    "../outputs/test_all",
    ['mental_health', 'eating_habits', 'physical_inactivity']
)

# 6. Lollipop
plot_cluster_lollipop(
    cluster_df, pop_df,
    ['total_wl_%', 'baseline_bmi'],
    "test_lollipop.png",
    "../outputs/test_all"
)

print("\n✓ ALL TESTS COMPLETE!")
```
