"""
Example: How to run cluster analysis pipeline

This script demonstrates how to configure and run a complete cluster analysis.
"""

from cluster_descriptions import cluster_analysis_config, run_cluster_analysis_pipeline

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define which variables to analyze
CONTINUOUS_VARS = [
    'total_wl_%',
    'baseline_bmi',
    'total_followup_days',
    'baseline_weight_kg'
]

CATEGORICAL_VARS = [
    'sex_f',
    '5%_wl_achieved',
    '10%_wl_achieved',
    'instant_dropout'
]

WGC_VARS = [
    'mental_health',
    'eating_habits',
    'physical_inactivity',
    'womens_health_and_pregnancy',
    'medication_disease_injury',
    'family_issues'
]

# Row order for tables (from your existing config)
ROW_ORDER = [
    ("N", "N"),
    ("delim_wgc", "--- Weight Gain Causes ---"),
    ("mental_health", "Mental health (yes/no)"),
    ("eating_habits", "Eating habits (yes/no)"),
    ("physical_inactivity", "Physical inactivity (yes/no)"),
    ("womens_health_and_pregnancy", "Women's health (yes/no)"),
    ("medication_disease_injury", "Medical issues (yes/no)"),
    ("family_issues", "Family issues (yes/no)"),
]

# =============================================================================
# CREATE CONFIGURATION
# =============================================================================

config = cluster_analysis_config(
    # Database paths
    cluster_db_path="../dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite",
    main_db_path="../dbs/pnk_db2_p2_in.sqlite",
    
    # Table names
    cluster_table="clust_labels_bl_nobc_bw_pam_goldstd",
    cluster_column="pam_k7",  # Using k=7 clusters
    outcome_table="timetoevent_wgc_compl",
    population_table="timetoevent_all",
    
    # Configuration files
    cluster_config_path="scripts/cluster_config.json",
    name_map_path="scripts/human_readable_variable_names.json",
    
    # Output settings
    output_dir="../outputs/cluster_k7_analysis",
    output_db_path="../dbs/pnk_db2_p2_out.sqlite",
    output_table_prefix="cluster_k7",
    
    # Analysis settings
    variables_to_analyze=CONTINUOUS_VARS,
    categorical_variables=CATEGORICAL_VARS,
    wgc_variables=WGC_VARS,
    row_order=ROW_ORDER,
    
    # Statistical settings
    fdr_correction=True,
    alpha=0.05
)

# =============================================================================
# RUN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    run_cluster_analysis_pipeline(config)
