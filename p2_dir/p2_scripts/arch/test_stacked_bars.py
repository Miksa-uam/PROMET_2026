#!/usr/bin/env python3
"""
Quick test script for WGC sex distribution stacked bar charts
"""

import os
import sys
import sqlite3
import pandas as pd

# Add the current directory to path so we can import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wgc_sex_stacked_bars import StackedBarConfig, get_wgc_columns, create_simple_proportional_stacked_bar


def quick_test():
    """Run a quick test with default settings"""
    
    # Default paths
    base_dir = r"C:\Users\Felhasználó\Desktop\Projects\PNK_DB2\paper2_dir"
    db_path = os.path.join(base_dir, "dbs", "pnk_db2_p2_in.sqlite")
    table_name = "timetoevent_wgc_compl"
    
    print("=== Quick Stacked Bar Chart Test ===")
    print(f"Looking for database at: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database not found! Please check the path.")
        return
    
    try:
        # Load data
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        print(f"Loaded {len(df)} records")
        
        # Check for sex column
        if 'sex_f' not in df.columns:
            print("Error: 'sex_f' column not found!")
            return
        
        # Show sex distribution
        sex_counts = df['sex_f'].value_counts()
        print(f"Sex distribution: {sex_counts[0]} males, {sex_counts[1]} females")
        
        # Get WGC columns
        wgc_cols = get_wgc_columns(df)
        print(f"Found {len(wgc_cols)} WGC columns: {wgc_cols}")
        
        # Create config
        config = StackedBarConfig(
            db_path=db_path,
            table_name=table_name,
            output_dir=os.path.join(base_dir, "outputs", "stacked_bars")
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        print(f"\n=== Generating Test Plots ===")
        print(f"Output directory: {config.output_dir}")
        
        # Generate simple proportional stacked bar chart (cause present only)
        print("Creating simple proportional stacked bar chart...")
        fig = create_simple_proportional_stacked_bar(df, wgc_cols, config)
        if fig:
            output_path = os.path.join(config.output_dir, "test_sex_wgc_simple_proportional.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved simple proportional chart: {output_path}")
        
        print(f"\n🎉 Test completed successfully!")
        print(f"Check your plots in: {config.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_test()