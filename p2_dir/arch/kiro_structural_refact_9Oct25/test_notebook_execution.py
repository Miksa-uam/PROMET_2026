#!/usr/bin/env python3
"""
Test script to verify that key notebook functionality works
with the reorganized directory structure.
"""

import sys
import os
import sqlite3

def test_notebook_imports():
    """Test that notebook can import required modules from scripts directory."""
    print("Testing notebook imports from reorganized structure...")
    print("=" * 60)
    
    # Change to scripts directory (where notebooks are now located)
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    os.chdir(scripts_dir)
    
    try:
        # Test imports that notebooks typically use
        from paper12_config import paths_config, paper2_rf_config
        from rf_engine import RandomForestAnalyzer
        import pandas as pd
        import sqlite3
        
        print("✅ All required modules imported successfully")
        
        # Test configuration with new paths
        test_paths = paths_config(
            source_dir="../dbs",
            source_db="../dbs/pnk_db2_filtered.sqlite",
            paper_dir="..",
            paper_in_db="../dbs/pnk_db2_p2_in.sqlite",
            paper_out_db="../dbs/pnk_db2_p2_out.sqlite"
        )
        
        print("✅ Configuration objects created successfully")
        
        # Test database connection with new paths
        if os.path.exists(test_paths.paper_in_db):
            with sqlite3.connect(test_paths.paper_in_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                table_count = cursor.fetchone()[0]
                print(f"✅ Database connection successful - found {table_count} tables")
        else:
            print("❌ Database file not found at expected location")
            return False
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"❌ Notebook import test failed: {e}")
        os.chdir(original_cwd)
        return False

def test_rf_config_functionality():
    """Test that RF configuration works with new paths."""
    print("\nTesting RF configuration functionality...")
    print("=" * 60)
    
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    os.chdir(scripts_dir)
    
    try:
        from paper12_config import paper2_rf_config
        
        # Create a test RF configuration
        test_config = paper2_rf_config(
            analysis_name="test_analysis",
            outcome_variable="test_outcome",
            model_type="classifier",
            predictors=["test_predictor1", "test_predictor2"],
            classifier_threshold=0.5,
            threshold_direction="greater_than_or_equal",
            db_path="../dbs/pnk_db2_p2_in.sqlite",
            output_dir="../outputs/rf_outputs"
        )
        
        print("✅ RF configuration created successfully")
        print(f"   Database path: {test_config.db_path}")
        print(f"   Output directory: {test_config.output_dir}")
        
        # Verify paths exist
        if os.path.exists(test_config.db_path):
            print("✅ Database path is accessible")
        else:
            print("❌ Database path not found")
            return False
            
        if os.path.exists(test_config.output_dir):
            print("✅ Output directory is accessible")
        else:
            print("❌ Output directory not found")
            return False
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"❌ RF configuration test failed: {e}")
        os.chdir(original_cwd)
        return False

def test_data_loading():
    """Test that data can be loaded from the database using new paths."""
    print("\nTesting data loading functionality...")
    print("=" * 60)
    
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    os.chdir(scripts_dir)
    
    try:
        import pandas as pd
        import sqlite3
        
        db_path = "../dbs/pnk_db2_p2_in.sqlite"
        
        # Try to load a sample of data
        with sqlite3.connect(db_path) as conn:
            # Get list of tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"✅ Found {len(tables)} tables in database")
            
            # Try to load a small sample from the first table
            if tables:
                sample_table = tables[0]
                df = pd.read_sql_query(f"SELECT * FROM {sample_table} LIMIT 5", conn)
                print(f"✅ Successfully loaded sample data from '{sample_table}' table")
                print(f"   Sample shape: {df.shape}")
            else:
                print("❌ No tables found in database")
                return False
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        os.chdir(original_cwd)
        return False

if __name__ == "__main__":
    print("📓 NOTEBOOK FUNCTIONALITY VALIDATION")
    print("=" * 60)
    
    # Run all tests
    import_success = test_notebook_imports()
    config_success = test_rf_config_functionality()
    data_success = test_data_loading()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"✅ Notebook Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ RF Configuration: {'PASS' if config_success else 'FAIL'}")
    print(f"✅ Data Loading: {'PASS' if data_success else 'FAIL'}")
    
    if import_success and config_success and data_success:
        print("\n🎉 ALL NOTEBOOK TESTS PASSED! Notebooks should work correctly with the new structure.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME NOTEBOOK TESTS FAILED. Please review the errors above.")
        sys.exit(1)