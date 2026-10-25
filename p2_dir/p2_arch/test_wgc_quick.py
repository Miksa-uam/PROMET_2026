"""
QUICK TEST: Run this script to verify WGC visualization pipeline works.
Usage: python test_wgc_quick.py
"""

import sys
import os
sys.path.insert(0, 'scripts')

import sqlite3
import pandas as pd
from descriptive_visualizations import plot_distribution_comparison, plot_stacked_bar_comparison

print("="*60)
print("QUICK WGC VISUALIZATION TEST")
print("="*60)

# Configuration
DB_PATH = "dbs/pnk_db2_p2_in.sqlite"
OUTPUT_DIR = "outputs/test_wgc_quick"
NAME_MAP = "scripts/human_readable_variable_names.json"

# Check files exist
if not os.path.exists(DB_PATH):
    print(f"✗ Database not found: {DB_PATH}")
    print("  Make sure you're running from the project root directory")
    sys.exit(1)

if not os.path.exists(NAME_MAP):
    print(f"✗ Name map not found: {NAME_MAP}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("\n1. Loading data...")
try:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM timetoevent_wgc_compl", conn)
        pop_df = pd.read_sql_query("SELECT * FROM timetoevent_all", conn)
    print(f"   ✓ Loaded {len(df)} WGC records")
    print(f"   ✓ Loaded {len(pop_df)} population records")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    sys.exit(1)

# Test 1: Violin plot
print("\n2. Testing violin plot...")
try:
    plot_distribution_comparison(
        df=df,
        population_df=pop_df,
        variable='total_wl_%',
        group_col='mental_health',
        output_filename='quick_test_violin.png',
        name_map_path=NAME_MAP,
        output_dir=OUTPUT_DIR
    )
    print("   ✓ Violin plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Stacked bar
print("\n3. Testing stacked bar plot...")
try:
    plot_stacked_bar_comparison(
        df=df,
        population_df=pop_df,
        variable='5%_wl_achieved',
        group_col='eating_habits',
        output_filename='quick_test_bar.png',
        name_map_path=NAME_MAP,
        output_dir=OUTPUT_DIR
    )
    print("   ✓ Stacked bar plot created")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print(f"\nPlots saved to: {OUTPUT_DIR}/")
print("  - quick_test_violin.png")
print("  - quick_test_bar.png")
print("\nIf you see ✓ marks above, the pipeline is working!")
print("="*60)
