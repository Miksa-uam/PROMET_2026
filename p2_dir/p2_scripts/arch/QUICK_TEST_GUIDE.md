# Quick Test Guide - Cluster Descriptions Module

## Copy-Paste Test Cells

### 1️⃣ Load Data (Run First)
```python
from cluster_descriptions_new import load_and_merge_cluster_data

cluster_df = load_and_merge_cluster_data(
    "../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    "../dbs/pnk_db2_p2_in.sqlite",
    "clust_labels_bl_nobc_bw_pam_goldstd",
    "pam_k7",
    "timetoevent_wgc_compl"
)
```

### 2️⃣ Violin Plots
```python
from cluster_descriptions_new import plot_cluster_distributions

plot_cluster_distributions(
    cluster_df,
    ['total_wl_%', 'baseline_bmi'],
    "../outputs/test",
    calculate_significance=True,
    fdr_correction=True
)
```

### 3️⃣ Stacked Bar Plots
```python
from cluster_descriptions_new import plot_cluster_categorical

plot_cluster_categorical(
    cluster_df,
    ['sex_f', '5%_wl_achieved'],
    "../outputs/test",
    calculate_significance=True,
    fdr_correction=True
)
```

### 4️⃣ Statistical Analysis
```python
from cluster_descriptions_new import analyze_cluster_vs_population

results_df = analyze_cluster_vs_population(
    cluster_df,
    ['mental_health', 'eating_habits', 'physical_inactivity'],
    "../dbs/pnk_db2_p2_out.sqlite",
    "test_wgc",
    fdr_correction=True
)
```

### 5️⃣ Heatmap
```python
from cluster_descriptions_new import plot_cluster_heatmap

plot_cluster_heatmap(
    results_df,
    "heatmap.png",
    "../outputs/test",
    ['mental_health', 'eating_habits', 'physical_inactivity']
)
```

### 6️⃣ Lollipop Plot
```python
from cluster_descriptions_new import plot_cluster_lollipop

plot_cluster_lollipop(
    cluster_df,
    ['total_wl_%', 'baseline_bmi'],
    "lollipop.png",
    "../outputs/test"
)
```

---

## What to Check

✅ **Violin plots:** Split violins, cluster labels, significance markers  
✅ **Bar plots:** Stacked bars, population line, sample sizes, significance  
✅ **Analysis:** Two tables in database (detailed + publication)  
✅ **Heatmap:** Cell format "n (%)** ", colors, labels  
✅ **Lollipop:** Multiple variables, cluster colors, legend  

---

## Expected Output Locations

**Plots:** `../outputs/test/`  
**Tables:** `../dbs/pnk_db2_p2_out.sqlite`  
- `test_wgc_detailed` (with p-values)  
- `test_wgc` (with asterisks)

---

## If Something Fails

Check file paths:
```python
import os
print("Working dir:", os.getcwd())
print("Config exists:", os.path.exists('cluster_config.json'))
print("Name map exists:", os.path.exists('human_readable_variable_names.json'))
```

Check data:
```python
print("Cluster IDs:", sorted(cluster_df['cluster_id'].unique()))
print("Columns:", cluster_df.columns.tolist()[:10])
```
