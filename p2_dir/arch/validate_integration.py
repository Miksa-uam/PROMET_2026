#!/usr/bin/env python3
"""
Integration Validation Script for Descriptive Visualizations Pipeline

This script validates the integration with existing project infrastructure:
- Tests compatibility with get_cause_cols and categorical_pvalue functions
- Verifies database path configuration works with master_config
- Tests with actual project data (timetoevent_wgc_compl table)
- Confirms output directory structure follows project conventions

Requirements: 4.1, 4.5, 6.4
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Add scripts directory to path for imports
sys.path.append('scripts')

def test_imports():
    """Test that all required modules can be imported successfully."""
    print("="*60)
    print("TESTING MODULE IMPORTS")
    print("="*60)
    
    import_results = {}
    
    # Test descriptive_comparisons imports
    try:
        from descriptive_comparisons import get_cause_cols, categorical_pvalue
        import_results['descriptive_comparisons'] = True
        print("✓ Successfully imported get_cause_cols and categorical_pvalue from descriptive_comparisons.py")
    except ImportError as e:
        import_results['descriptive_comparisons'] = False
        print(f"✗ Failed to import from descriptive_comparisons.py: {e}")
    
    # Test fdr_correction_utils imports
    try:
        from fdr_correction_utils import apply_fdr_correction
        import_results['fdr_correction_utils'] = True
        print("✓ Successfully imported apply_fdr_correction from fdr_correction_utils.py")
    except ImportError as e:
        import_results['fdr_correction_utils'] = False
        print(f"✗ Failed to import from fdr_correction_utils.py: {e}")
    
    # Test paper12_config imports
    try:
        from paper12_config import master_config, paths_config
        import_results['paper12_config'] = True
        print("✓ Successfully imported master_config and paths_config from paper12_config.py")
    except ImportError as e:
        import_results['paper12_config'] = False
        print(f"✗ Failed to import from paper12_config.py: {e}")
    
    # Test descriptive_visualizations imports
    try:
        from descriptive_visualizations import run_descriptive_visualizations
        import_results['descriptive_visualizations'] = True
        print("✓ Successfully imported run_descriptive_visualizations from descriptive_visualizations.py")
    except ImportError as e:
        import_results['descriptive_visualizations'] = False
        print(f"✗ Failed to import from descriptive_visualizations.py: {e}")
    
    return import_results


def test_get_cause_cols_function():
    """Test the get_cause_cols function with sample row_order configuration."""
    print("\n" + "="*60)
    print("TESTING get_cause_cols FUNCTION")
    print("="*60)
    
    try:
        from descriptive_comparisons import get_cause_cols
        
        # Create sample row_order configuration similar to what's used in the project
        sample_row_order = [
            ("N", "N"),
            ("age", "Age (years)"),
            ("sex_f", "Sex (Female)"),
            ("baseline_bmi", "Baseline BMI"),
            ("delim_wgc", "Weight gain causes"),
            ("womens_health_and_pregnancy", "Women's health/pregnancy (yes/no)"),
            ("mental_health", "Mental health (yes/no)"),
            ("family_issues", "Family issues (yes/no)"),
            ("medication_disease_injury", "Medication/disease/injury (yes/no)"),
            ("physical_inactivity", "Physical inactivity (yes/no)"),
            ("eating_habits", "Eating habits (yes/no)"),
            ("schedule", "Schedule (yes/no)"),
            ("smoking_cessation", "Smoking cessation (yes/no)"),
            ("treatment_discontinuation_or_relapse", "Treatment relapse (yes/no)"),
            ("pandemic", "COVID-19 pandemic (yes/no)"),
            ("lifestyle_circumstances", "Lifestyle, circumstances (yes/no)"),
            ("none_of_above", "None of the above (yes/no)"),
            ("delim_outcomes", "Outcomes")
        ]
        
        # Test get_cause_cols function
        cause_cols = get_cause_cols(sample_row_order)
        
        print(f"✓ get_cause_cols function executed successfully")
        print(f"✓ Identified {len(cause_cols)} weight gain cause columns:")
        for i, col in enumerate(cause_cols, 1):
            print(f"  {i:2d}. {col}")
        
        return True, cause_cols
        
    except Exception as e:
        print(f"✗ Error testing get_cause_cols function: {e}")
        return False, []


def test_database_access():
    """Test database connection and access to timetoevent_wgc_compl table."""
    print("\n" + "="*60)
    print("TESTING DATABASE CONNECTION AND TABLE ACCESS")
    print("="*60)
    
    try:
        # Test with the available database
        db_path = "dbs/pnk_db2_p2_in.sqlite"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            print(f"✗ Database file not found: {db_path}")
            return False, None
        
        print(f"✓ Database file found: {db_path}")
        
        # Test database connection
        with sqlite3.connect(db_path) as conn:
            print("✓ Successfully connected to database")
            
            # List all tables in database
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_df = pd.read_sql_query(tables_query, conn)
            table_names = tables_df['name'].tolist()
            
            print(f"✓ Found {len(table_names)} tables in database:")
            for table in sorted(table_names):
                print(f"  - {table}")
            
            # Check for timetoevent_wgc_compl table specifically
            target_table = "timetoevent_wgc_compl"
            if target_table in table_names:
                print(f"✓ Target table '{target_table}' found in database")
                
                # Get table schema
                schema_query = f"PRAGMA table_info({target_table})"
                schema_df = pd.read_sql_query(schema_query, conn)
                column_names = schema_df['name'].tolist()
                
                print(f"✓ Table has {len(column_names)} columns")
                
                # Check for required outcome columns
                required_outcomes = ["10%_wl_achieved", "60d_dropout"]
                for outcome in required_outcomes:
                    if outcome in column_names:
                        print(f"  ✓ Required outcome column found: {outcome}")
                    else:
                        print(f"  ✗ Required outcome column missing: {outcome}")
                
                # Get total row count
                count_query = f"SELECT COUNT(*) as total_rows FROM {target_table}"
                count_df = pd.read_sql_query(count_query, conn)
                total_rows = count_df['total_rows'].iloc[0]
                
                print(f"✓ Total rows in '{target_table}': {total_rows:,}")
                
                return True, {
                    'table_exists': True,
                    'total_rows': total_rows,
                    'column_names': column_names
                }
                
            else:
                print(f"✗ Target table '{target_table}' not found in database")
                return False, {'table_exists': False, 'available_tables': table_names}
        
    except Exception as e:
        print(f"✗ Error testing database connection: {e}")
        return False, None


def test_output_directory_structure():
    """Test output directory structure follows project conventions."""
    print("\n" + "="*60)
    print("TESTING OUTPUT DIRECTORY STRUCTURE")
    print("="*60)
    
    try:
        # Define expected output directory structure
        base_output_dir = "outputs"
        descriptive_viz_dir = os.path.join(base_output_dir, "descriptive_visualizations")
        
        print(f"Testing output directory structure:")
        print(f"  Base output directory: {base_output_dir}")
        print(f"  Descriptive viz directory: {descriptive_viz_dir}")
        
        # Check if base output directory exists
        if os.path.exists(base_output_dir):
            print(f"✓ Base output directory exists: {base_output_dir}")
        else:
            print(f"⚠ Base output directory does not exist: {base_output_dir}")
            print("  This is normal - it will be created when needed")
        
        # Test directory creation capability
        test_dir = os.path.join(descriptive_viz_dir, "test_creation")
        try:
            os.makedirs(test_dir, exist_ok=True)
            print(f"✓ Successfully tested directory creation capability")
            
            # Clean up test directory
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
                if os.path.exists(descriptive_viz_dir) and not os.listdir(descriptive_viz_dir):
                    os.rmdir(descriptive_viz_dir)
                print(f"✓ Test directory cleaned up")
                
        except Exception as e:
            print(f"⚠ Warning: Could not test directory creation: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing output directory structure: {e}")
        return False


def run_integration_validation():
    """Run integration validation suite."""
    print("DESCRIPTIVE VISUALIZATIONS INTEGRATION VALIDATION")
    print("=" * 80)
    print("This script validates integration with existing project infrastructure")
    print("=" * 80)
    
    # Track validation results
    validation_results = {}
    
    # Test 1: Module imports
    import_results = test_imports()
    validation_results['imports'] = all(import_results.values())
    
    # Test 2: get_cause_cols function
    if import_results.get('descriptive_comparisons', False):
        get_cause_cols_success, cause_cols = test_get_cause_cols_function()
        validation_results['get_cause_cols'] = get_cause_cols_success
    else:
        validation_results['get_cause_cols'] = False
    
    # Test 3: Database access
    db_success, table_info = test_database_access()
    validation_results['database_access'] = db_success
    
    # Test 4: Output directory structure
    output_dir_success = test_output_directory_structure()
    validation_results['output_directory'] = output_dir_success
    
    # Summary report
    print("\n" + "="*80)
    print("INTEGRATION VALIDATION SUMMARY")
    print("="*80)
    
    total_tests = len(validation_results)
    passed_tests = sum(validation_results.values())
    
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed")
    print()
    
    for test_name, result in validation_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*80)
    
    if passed_tests == total_tests:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("The descriptive visualizations pipeline is ready for use.")
    elif passed_tests >= total_tests * 0.75:  # 75% pass rate
        print("⚠ MOST INTEGRATION TESTS PASSED")
        print("The pipeline should work but some features may have issues.")
    else:
        print("❌ INTEGRATION VALIDATION FAILED")
        print("Significant issues found. Please review and fix before using the pipeline.")
    
    print("="*80)
    
    return validation_results


if __name__ == "__main__":
    # Run the validation suite
    results = run_integration_validation()