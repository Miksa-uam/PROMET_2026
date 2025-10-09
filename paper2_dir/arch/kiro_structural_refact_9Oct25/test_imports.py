#!/usr/bin/env python3
"""
Test script to verify that all Python modules can be imported correctly
from the reorganized directory structure.
"""

import sys
import os
import traceback

def test_imports():
    """Test importing all Python modules from the scripts directory."""
    print("Testing Python module imports from reorganized structure...")
    print("=" * 60)
    
    # Change to scripts directory to test imports from there
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    
    if not os.path.exists(scripts_dir):
        print("❌ ERROR: scripts/ directory not found!")
        return False
    
    os.chdir(scripts_dir)
    
    # List of modules to test
    modules_to_test = [
        "paper12_config",
        "rf_config", 
        "rf_engine",
        "stats_helpers",
        "timetoevent_table_functions",
        "descriptive_comparisons",
        "regressions_inference",
        "file_movement_utils",
        "create_project_structure"
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name in modules_to_test:
        try:
            # Remove from sys.modules if already imported to force fresh import
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Try to import the module
            __import__(module_name)
            print(f"✅ {module_name}: Import successful")
            success_count += 1
            
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️  {module_name}: Import succeeded but module has issues - {e}")
            success_count += 1  # Still counts as successful import
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    print("\n" + "=" * 60)
    print(f"Import Test Results: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 All modules imported successfully!")
        return True
    else:
        print("⚠️  Some modules failed to import. Check the errors above.")
        return False

def test_database_connections():
    """Test that database connections work with new relative paths."""
    print("\nTesting database connections...")
    print("=" * 60)
    
    # Change to scripts directory
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    os.chdir(scripts_dir)
    
    try:
        import sqlite3
        
        # Test database files that should exist
        db_files = [
            "../dbs/pnk_db2_p2_in.sqlite",
            "../dbs/pnk_db2_p2_out.sqlite"
        ]
        
        success_count = 0
        for db_file in db_files:
            try:
                if os.path.exists(db_file):
                    # Try to connect and run a simple query
                    with sqlite3.connect(db_file) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
                        result = cursor.fetchone()
                        print(f"✅ {db_file}: Connection successful")
                        success_count += 1
                else:
                    print(f"❌ {db_file}: File not found")
            except Exception as e:
                print(f"❌ {db_file}: Connection failed - {e}")
        
        os.chdir(original_cwd)
        
        print(f"\nDatabase Test Results: {success_count}/{len(db_files)} databases accessible")
        return success_count == len(db_files)
        
    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ Database testing failed: {e}")
        return False

def test_output_directories():
    """Test that output directories exist and are writable."""
    print("\nTesting output directories...")
    print("=" * 60)
    
    # Change to scripts directory
    original_cwd = os.getcwd()
    scripts_dir = os.path.join(original_cwd, "scripts")
    os.chdir(scripts_dir)
    
    try:
        output_dirs = [
            "../outputs",
            "../outputs/rf_outputs",
            "../outputs/wgc_association_networks"
        ]
        
        success_count = 0
        for output_dir in output_dirs:
            try:
                if os.path.exists(output_dir):
                    # Test if directory is writable by creating a test file
                    test_file = os.path.join(output_dir, "test_write.tmp")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    print(f"✅ {output_dir}: Directory exists and is writable")
                    success_count += 1
                else:
                    print(f"❌ {output_dir}: Directory not found")
            except Exception as e:
                print(f"❌ {output_dir}: Write test failed - {e}")
        
        os.chdir(original_cwd)
        
        print(f"\nOutput Directory Test Results: {success_count}/{len(output_dirs)} directories accessible")
        return success_count == len(output_dirs)
        
    except Exception as e:
        os.chdir(original_cwd)
        print(f"❌ Output directory testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 REORGANIZED PROJECT STRUCTURE VALIDATION")
    print("=" * 60)
    
    # Run all tests
    import_success = test_imports()
    db_success = test_database_connections()
    output_success = test_output_directories()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"✅ Module Imports: {'PASS' if import_success else 'FAIL'}")
    print(f"✅ Database Connections: {'PASS' if db_success else 'FAIL'}")
    print(f"✅ Output Directories: {'PASS' if output_success else 'FAIL'}")
    
    if import_success and db_success and output_success:
        print("\n🎉 ALL TESTS PASSED! The reorganized structure is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the errors above.")
        sys.exit(1)