"""
Test lollipop plot adjustments for Task 2:
- Significance marker positioning (x_offset reduced from 0.02 to 0.01)
- Variable separator linewidth (increased from 0.5 to 2.0)
"""
import sys
import os
sys.path.insert(0, 'scripts')

import sqlite3
import pandas as pd
from cluster_descriptions import (
    load_and_merge_cluster_data,
    analyze_cluster_vs_population,
    extract_pvalues_for_lollipop,
    plot_cluster_lollipop
)

print("="*80)
print("LOLLIPOP PLOT ADJUSTMENTS TEST - TASK 2")
print("="*80)

# Configuration
DB_PATH = "dbs/pnk_db2_p2_in.sqlite"
OUTPUT_DIR = "outputs/test_lollipop_task2"
NAME_MAP = "scripts/human_readable_variable_names.json"
CLUSTER_CONFIG = "scripts/cluster_config.json"

# Check files exist
if not os.path.exists(DB_PATH):
    print(f"✗ Database not found: {DB_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cluster data
print("\n1. Loading cluster data...")
try:
    cluster_df = load_and_merge_cluster_data(
        cluster_db_path=DB_PATH,
        main_db_path=DB_PATH,
        cluster_table='temp_cluster_patients',
        cluster_column='cluster_id',
        outcome_table='timetoevent_wgc_compl'
    )
    print(f"   ✓ Loaded {len(cluster_df)} records")
    print(f"   ✓ Clusters: {sorted(cluster_df['cluster_id'].unique())}")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    sys.exit(1)

# Test with multiple variables to verify separator lines
test_variables = ['age', 'bmi', 'total_wl_%']

print(f"\n2. Analyzing {len(test_variables)} variables...")
try:
    results_df = analyze_cluster_vs_population(
        cluster_df=cluster_df,
        variables=test_variables,
        output_db_path=os.path.join(OUTPUT_DIR, 'test_results.db'),
        output_table_name='test_lollipop',
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        fdr_correction=True
    )
    print(f"   ✓ Analysis complete")
except Exception as e:
    print(f"   ✗ Error in analysis: {e}")
    sys.exit(1)

# Extract p-values
print("\n3. Extracting p-values...")
try:
    pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
        results_df=results_df,
        variables=test_variables,
        cluster_df=cluster_df
    )
    print(f"   ✓ P-values extracted")
except Exception as e:
    print(f"   ✗ Error extracting p-values: {e}")
    sys.exit(1)

# Generate lollipop plot
print("\n4. Generating lollipop plot...")
print("   Expected changes:")
print("   - Significance markers (*/**) should be closer to lollipop heads")
print("   - Variable separator lines should be BOLD (linewidth=2.0)")
try:
    plot_cluster_lollipop(
        cluster_df=cluster_df,
        variables=test_variables,
        output_filename='test_lollipop_adjustments.png',
        output_dir=OUTPUT_DIR,
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        pvalues_raw=pvalues_raw,
        pvalues_fdr=pvalues_fdr,
        alpha=0.05
    )
    print("   ✓ Lollipop plot created")
except Exception as e:
    print(f"   ✗ Error creating plot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with single variable (no separators expected)
print("\n5. Testing with single variable (no separators)...")
try:
    single_var = ['age']
    results_single = analyze_cluster_vs_population(
        cluster_df=cluster_df,
        variables=single_var,
        output_db_path=os.path.join(OUTPUT_DIR, 'test_results_single.db'),
        output_table_name='test_lollipop_single',
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        fdr_correction=True
    )
    
    pvalues_raw_single, pvalues_fdr_single = extract_pvalues_for_lollipop(
        results_df=results_single,
        variables=single_var,
        cluster_df=cluster_df
    )
    
    plot_cluster_lollipop(
        cluster_df=cluster_df,
        variables=single_var,
        output_filename='test_lollipop_single_variable.png',
        output_dir=OUTPUT_DIR,
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        pvalues_raw=pvalues_raw_single,
        pvalues_fdr=pvalues_fdr_single,
        alpha=0.05
    )
    print("   ✓ Single variable plot created (no separators expected)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print(f"\nPlots saved to: {OUTPUT_DIR}/")
print("  - test_lollipop_adjustments.png (multiple variables)")
print("  - test_lollipop_single_variable.png (single variable)")
print("\nVERIFICATION CHECKLIST:")
print("  [ ] Significance markers (*/**) are positioned closer to lollipop heads")
print("  [ ] Markers align horizontally with lollipop heads")
print("  [ ] Variable separator lines are BOLD (thicker than before)")
print("  [ ] Single variable plot has no separator lines")
print("="*80)
