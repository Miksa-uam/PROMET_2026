#!/usr/bin/env python3
"""
Test script for enhanced forest plots with publication-ready improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from descriptive_visualizations import run_descriptive_visualizations

def test_enhanced_forest_plots():
    """Test the enhanced forest plots with correct database path"""
    
    print("=== Testing Enhanced Forest Plots ===")
    
    # Use the correct database path
    db_path = "dbs/pnk_db2_p2_in.sqlite"
    table_name = "timetoevent_wgc_compl"
    
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    
    try:
        # Run the pipeline with correct paths
        result = run_descriptive_visualizations(
            input_table=table_name,
            db_path=db_path
        )
        
        if result['status'] == 'success':
            print("✓ Enhanced forest plots generated successfully!")
            print(f"✓ Plots saved to: {result['output_directories']['forest_plots']}")
            
            # List generated files
            forest_plots_dir = result['output_directories']['forest_plots']
            if os.path.exists(forest_plots_dir):
                files = [f for f in os.listdir(forest_plots_dir) if f.endswith('.png')]
                print(f"✓ Generated {len(files)} forest plots:")
                for file in files:
                    print(f"  - {file}")
        else:
            print(f"❌ Error: {result.get('error_message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_forest_plots()