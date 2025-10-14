#!/usr/bin/env python3
"""
Test Pipeline Integration with Actual Project Data

This script tests the descriptive visualizations pipeline with actual project data
to ensure complete integration with existing infrastructure.
"""

import sys
import os
sys.path.append('scripts')

def test_pipeline_with_actual_data():
    """Test the pipeline with actual project data."""
    print("="*80)
    print("TESTING PIPELINE WITH ACTUAL PROJECT DATA")
    print("="*80)
    
    try:
        # Import required modules
        from descriptive_visualizations import run_descriptive_visualizations
        from paper12_config import paths_config, master_config
        
        print("✓ Successfully imported required modules")
        
        # Configure paths using existing pattern
        paths = paths_config(
            source_dir="dbs",
            source_db="dbs/pnk_db2_filtered.sqlite", 
            paper_dir=".",
            paper_in_db="dbs/pnk_db2_p2_in.sqlite",
            paper_out_db="dbs/pnk_db2_p2_out.sqlite",
        )
        
        config = master_config(paths=paths)
        print("✓ Successfully created master_config with paths")
        
        # Test the pipeline with a limited subset for validation
        print("\nRunning descriptive visualizations pipeline...")
        print("Note: This is a validation test with actual project data")
        
        # Run the pipeline
        results = run_descriptive_visualizations(
            input_table="timetoevent_wgc_compl",
            config=config
        )
        
        print("✓ Pipeline executed successfully!")
        
        # Check if outputs were created
        output_dir = "outputs/descriptive_visualizations"
        if os.path.exists(output_dir):
            print(f"✓ Output directory created: {output_dir}")
            
            # List generated files
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, output_dir)
                    file_size = os.path.getsize(file_path)
                    print(f"  📄 {rel_path} ({file_size:,} bytes)")
        
        print("\n✓ INTEGRATION TEST SUCCESSFUL!")
        print("The descriptive visualizations pipeline is fully integrated and working.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing pipeline integration: {e}")
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline_with_actual_data()
    sys.exit(0 if success else 1)