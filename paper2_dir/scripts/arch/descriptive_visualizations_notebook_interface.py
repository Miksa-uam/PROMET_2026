# #### I/2.2. Risk ratio/risk difference analyses

# Import the descriptive visualizations pipeline
from descriptive_visualizations import run_descriptive_visualizations
from paper12_config import paths_config, master_config

# Configure paths (reuse existing configuration pattern)
paths = paths_config(
    source_dir = r"../dbs",
    source_db = r"../dbs/pnk_db2_filtered.sqlite", 
    paper_dir = r"..",
    paper_in_db = r"../dbs/pnk_db2_p2_in.sqlite",
    paper_out_db = r"../dbs/pnk_db2_p2_out.sqlite",
)

config = master_config(paths=paths)

# Run comprehensive descriptive visualizations
# This generates both risk ratio and risk difference forest plots 
# for 10% weight loss achievement and 60-day dropout outcomes
results = run_descriptive_visualizations(
    input_table="timetoevent_wgc_compl",  # Primary configurable parameter
    config=config
)

print("\n" + "="*60)
print("DESCRIPTIVE VISUALIZATIONS COMPLETE")
print("="*60)
print("Generated outputs:")
print("• Risk ratio forest plots for both outcomes")
print("• Risk difference forest plots for both outcomes") 
print("• Comprehensive summary tables with all statistics")
print("• All files saved to ../outputs/descriptive_visualizations/")
print("="*60)