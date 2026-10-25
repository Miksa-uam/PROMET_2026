"""
Simple test script for WGC visualization pipeline.
Run this in your notebook or as a standalone script to verify the pipeline works.
"""

import sqlite3
import pandas as pd
import os

# Import visualization functions
from descriptive_visualizations import (
    plot_distribution_comparison,
    plot_stacked_bar_comparison,
    plot_multi_lollipop
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database paths
DB_PATH = "../dbs/pnk_db2_p2_in.sqlite"
OUTPUT_DIR = "../outputs/test_wgc_visualizations"

# Configuration files
NAME_MAP_PATH = "scripts/human_readable_variable_names.json"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data from database...")
with sqlite3.connect(DB_PATH) as conn:
    # Load the WGC complete cohort
    df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
    
    # Load the full population as reference
    population_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)

print(f"✓ Loaded {len(df)} records from WGC cohort")
print(f"✓ Loaded {len(population_df)} records from population cohort")

# =============================================================================
# TEST 1: VIOLIN PLOT - Compare mental_health groups to population
# =============================================================================

print("\n" + "="*60)
print("TEST 1: Violin Plot - Mental Health Groups vs Population")
print("="*60)

try:
    plot_distribution_comparison(
        df=df,
        population_df=population_df,
        variable='total_wl_%',  # Weight loss percentage
        group_col='mental_health',  # Compare by mental health WGC
        output_filename='test_wgc_mental_health_violin.png',
        name_map_path=NAME_MAP_PATH,
        output_dir=OUTPUT_DIR
    )
    print("✓ Violin plot created successfully!")
except Exception as e:
    print(f"✗ Error creating violin plot: {e}")

# =============================================================================
# TEST 2: STACKED BAR PLOT - Compare 5% weight loss achievement
# =============================================================================

print("\n" + "="*60)
print("TEST 2: Stacked Bar Plot - 5% WL Achievement by Eating Habits")
print("="*60)

try:
    plot_stacked_bar_comparison(
        df=df,
        population_df=population_df,
        variable='5%_wl_achieved',  # Binary outcome
        group_col='eating_habits',  # Compare by eating habits WGC
        output_filename='test_wgc_eating_habits_bar.png',
        name_map_path=NAME_MAP_PATH,
        output_dir=OUTPUT_DIR
    )
    print("✓ Stacked bar plot created successfully!")
except Exception as e:
    print(f"✗ Error creating stacked bar plot: {e}")

# =============================================================================
# TEST 3: LOLLIPOP PLOT - Multi-variable comparison across WGCs
# =============================================================================

print("\n" + "="*60)
print("TEST 3: Lollipop Plot - Multiple Variables Across WGCs")
print("="*60)

try:
    # Prepare lollipop data: percent change vs population mean
    wgc_variables = ['mental_health', 'eating_habits', 'physical_inactivity']
    outcome_variables = ['total_wl_%', 'baseline_bmi']
    
    lollipop_data = []
    
    for wgc_var in wgc_variables:
        for outcome_var in outcome_variables:
            # Calculate for "Yes" group (WGC present)
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
        output_filename='test_wgc_multi_lollipop.png',
        name_map_path=NAME_MAP_PATH,
        output_dir=OUTPUT_DIR
    )
    print("✓ Lollipop plot created successfully!")
except Exception as e:
    print(f"✗ Error creating lollipop plot: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. test_wgc_mental_health_violin.png")
print("  2. test_wgc_eating_habits_bar.png")
print("  3. test_wgc_multi_lollipop.png")
print("\nCheck the output directory to view the visualizations!")
print("="*60)
