"""
Test cluster visualization with actual cluster data
"""
import sqlite3
import pandas as pd
import os
import sys
sys.path.insert(0, 'scripts')

from descriptive_visualizations import plot_distribution_comparison, plot_stacked_bar_comparison

OUTPUT_DIR = "outputs/test_cluster_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("CLUSTER VISUALIZATION TEST")
print("="*60)

# Step 1: Load cluster assignments
print("\n1. Loading cluster assignments...")
with sqlite3.connect("dbs/pnk_db2_p2_in.sqlite") as conn:
    cluster_df = pd.read_sql_query("SELECT * FROM temp_cluster_patients", conn)
print(f"   ✓ Loaded {len(cluster_df)} cluster assignments")
print(f"   ✓ Clusters: {sorted(cluster_df['cluster_id'].unique())}")

# Step 2: Load outcome data
print("\n2. Loading outcome data...")
with sqlite3.connect("dbs/pnk_db2_p2_in.sqlite") as conn:
    outcome_df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
    population_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)
print(f"   ✓ Loaded {len(outcome_df)} outcome records")
print(f"   ✓ Loaded {len(population_df)} population records")

# Step 3: Merge cluster assignments with outcome data
print("\n3. Merging cluster + outcome data...")
df_merged = outcome_df.merge(cluster_df[['medical_record_id', 'cluster_id']], 
                              on='medical_record_id', 
                              how='inner')
print(f"   ✓ Merged data: {len(df_merged)} records")
print(f"   ✓ Clusters in merged data: {sorted(df_merged['cluster_id'].unique())}")

# Step 4: Test violin plot
print("\n4. Generating violin plot...")
try:
    plot_distribution_comparison(
        df=df_merged,
        population_df=population_df,
        variable='total_wl_%',
        group_col='cluster_id',
        output_filename='test_cluster_violin.png',
        name_map_path='scripts/human_readable_variable_names.json',
        cluster_config_path='scripts/cluster_config.json',
        output_dir=OUTPUT_DIR
    )
    print("   ✓ Violin plot created!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Step 5: Test stacked bar plot
print("\n5. Generating stacked bar plot...")
try:
    plot_stacked_bar_comparison(
        df=df_merged,
        population_df=population_df,
        variable='5%_wl_achieved',
        group_col='cluster_id',
        output_filename='test_cluster_bar.png',
        name_map_path='scripts/human_readable_variable_names.json',
        cluster_config_path='scripts/cluster_config.json',
        output_dir=OUTPUT_DIR
    )
    print("   ✓ Stacked bar plot created!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print(f"\nPlots saved to: {OUTPUT_DIR}/")
print("  - test_cluster_violin.png")
print("  - test_cluster_bar.png")
print("\nCheck the plots to verify cluster labels and colors!")
print("="*60)
