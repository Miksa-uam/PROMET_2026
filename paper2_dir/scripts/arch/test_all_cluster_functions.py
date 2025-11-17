"""
Complete test script for all cluster_descriptions_new.py functions.
Run each section separately in your notebook to test individual functions.
"""

from cluster_descriptions_new import (
    load_and_merge_cluster_data,
    plot_cluster_distributions,
    plot_cluster_categorical,
    analyze_cluster_vs_population,
    plot_cluster_heatmap,
    plot_cluster_lollipop
)

# =============================================================================
# STEP 1: LOAD DATA (Run this first)
# =============================================================================

print("="*60)
print("STEP 1: Loading and merging cluster data")
print("="*60)

cluster_df, pop_df = load_and_merge_cluster_data(
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all"
)

print(f"\n✓ Data loaded successfully!")
print(f"  Cluster data: {len(cluster_df)} records")
print(f"  Population data: {len(pop_df)} records")
print(f"  Clusters: {sorted(cluster_df['cluster_id'].unique())}")

# =============================================================================
# STEP 2: TEST VIOLIN PLOTS (Continuous variables)
# =============================================================================

print("\n" + "="*60)
print("STEP 2: Testing violin plots (continuous variables)")
print("="*60)

plot_cluster_distributions(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['total_wl_%', 'baseline_bmi', 'total_followup_days'],
    output_dir="../outputs/test_cluster_all",
    cluster_col='cluster_id',
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    calculate_significance=True,
    fdr_correction=True,
    alpha=0.05
)

print("\n✓ Violin plots complete!")
print("  Check: ../outputs/test_cluster_all/")
print("  Files: total_wl_%_violin.png, baseline_bmi_violin.png, etc.")

# =============================================================================
# STEP 3: TEST STACKED BAR PLOTS (Categorical variables)
# =============================================================================

print("\n" + "="*60)
print("STEP 3: Testing stacked bar plots (categorical variables)")
print("="*60)

plot_cluster_categorical(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=['sex_f', '5%_wl_achieved', '10%_wl_achieved', 'instant_dropout'],
    output_dir="../outputs/test_cluster_all",
    cluster_col='cluster_id',
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    calculate_significance=True,
    fdr_correction=True,
    alpha=0.05,
    legend_labels={'achieved': 'Achieved', 'not_achieved': 'Not Achieved'}
)

print("\n✓ Stacked bar plots complete!")
print("  Check: ../outputs/test_cluster_all/")
print("  Files: sex_f_bar.png, 5%_wl_achieved_bar.png, etc.")

# =============================================================================
# STEP 4: TEST STATISTICAL ANALYSIS (WGC vs Population)
# =============================================================================

print("\n" + "="*60)
print("STEP 4: Testing statistical analysis (WGC prevalence)")
print("="*60)

results_df = analyze_cluster_vs_population(
    cluster_df=cluster_df,
    population_df=pop_df,
    wgc_variables=[
        'mental_health',
        'eating_habits',
        'physical_inactivity',
        'womens_health_and_pregnancy',
        'medication_disease_injury',
        'family_issues'
    ],
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_name="test_cluster_k7_wgc",
    cluster_col='cluster_id',
    name_map_path='human_readable_variable_names.json',
    fdr_correction=True,
    alpha=0.05
)

print("\n✓ Statistical analysis complete!")
print("  Database: ../dbs/pnk_db2_p2_out.sqlite")
print("  Tables created:")
print("    - test_cluster_k7_wgc_detailed (with p-values)")
print("    - test_cluster_k7_wgc (publication-ready with asterisks)")

# =============================================================================
# STEP 5: TEST HEATMAP (WGC Prevalence)
# =============================================================================

print("\n" + "="*60)
print("STEP 5: Testing heatmap (WGC prevalence across clusters)")
print("="*60)

plot_cluster_heatmap(
    results_df=results_df,
    output_filename="test_wgc_heatmap.png",
    output_dir="../outputs/test_cluster_all",
    wgc_variables=[
        'mental_health',
        'eating_habits',
        'physical_inactivity',
        'womens_health_and_pregnancy',
        'medication_disease_injury',
        'family_issues'
    ],
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json',
    alpha=0.05
)

print("\n✓ Heatmap complete!")
print("  Check: ../outputs/test_cluster_all/test_wgc_heatmap.png")

# =============================================================================
# STEP 6: TEST LOLLIPOP PLOT (Multi-variable comparison)
# =============================================================================

print("\n" + "="*60)
print("STEP 6: Testing lollipop plot (multi-variable comparison)")
print("="*60)

plot_cluster_lollipop(
    cluster_df=cluster_df,
    population_df=pop_df,
    variables=[
        'total_wl_%',
        'baseline_bmi',
        'total_followup_days',
        'dietitian_visits'
    ],
    output_filename="test_multi_lollipop.png",
    output_dir="../outputs/test_cluster_all",
    cluster_col='cluster_id',
    name_map_path='human_readable_variable_names.json',
    cluster_config_path='cluster_config.json'
)

print("\n✓ Lollipop plot complete!")
print("  Check: ../outputs/test_cluster_all/test_multi_lollipop.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("ALL TESTS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  Plots: ../outputs/test_cluster_all/")
print("  Tables: ../dbs/pnk_db2_p2_out.sqlite")
print("\nCheck each output to verify:")
print("  ✓ Plots display correctly")
print("  ✓ Cluster labels are human-readable")
print("  ✓ Cluster colors match config")
print("  ✓ Significance markers (* and **) appear")
print("  ✓ Tables have correct structure")
print("="*60)
