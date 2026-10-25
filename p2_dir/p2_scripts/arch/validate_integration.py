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
        
        # Validate expected columns are present
        expected_cols = [
            "womens_health_and_pregnancy",
            "mental_health", 
            "family_issues",
            "medication_disease_injury",
            "physical_inactivity",
            "eating_habits",
            "schedule",
            "smoking_cessation",
            "treatment_discontinuation_or_relapse",
            "pandemic",
            "lifestyle_circumstances",
            "none_of_above"
        ]
        
        missing_cols = [col for col in expected_cols if col not in cause_cols]
        extra_cols = [col for col in cause_cols if col not in expected_cols]
        
        if not missing_cols and not extra_cols:
            print("✓ All expected weight gain cause columns identified correctly")
        else:
            if missing_cols:
                print(f"⚠ Missing expected columns: {missing_cols}")
            if extra_cols:
                print(f"⚠ Unexpected extra columns: {extra_cols}")
        
        return True, cause_cols
        
    except Exception as e:
        print(f"✗ Error testing get_cause_cols function: {e}")
        return False, []


def test_categorical_pvalue_function():
    """Test the categorical_pvalue function with sample data."""
    print("\n" + "="*60)
    print("TESTING categorical_pvalue FUNCTION")
    print("="*60)
    
    try:
        from descriptive_comparisons import categorical_pvalue
        
        # Create sample binary data for testing
        np.random.seed(42)  # For reproducible results
        
        # Group 1: 100 observations with 30% success rate
        group1 = np.random.binomial(1, 0.3, 100)
        
        # Group 2: 100 observations with 50% success rate  
        group2 = np.random.binomial(1, 0.5, 100)
        
        # Convert to pandas Series
        series1 = pd.Series(group1)
        series2 = pd.Series(group2)
        
        print(f"Testing with sample data:")
        print(f"  Group 1: {len(series1)} observations, {series1.sum()} successes ({series1.mean():.1%} rate)")
        print(f"  Group 2: {len(series2)} observations, {series2.sum()} successes ({series2.mean():.1%} rate)")
        
        # Test categorical_pvalue function
        p_value = categorical_pvalue(series1, series2)
        
        if pd.isna(p_value):
            print("⚠ categorical_pvalue returned NaN - this may indicate an issue")
            return False
        else:
            print(f"✓ categorical_pvalue function executed successfully")
            print(f"✓ Returned p-value: {p_value:.6f}")
            
            # Validate p-value is in expected range
            if 0 <= p_value <= 1:
                print("✓ P-value is in valid range [0, 1]")
            else:
                print(f"⚠ P-value {p_value} is outside valid range [0, 1]")
                return False
        
        # Test edge cases
        print("\nTesting edge cases:")
        
        # Test with identical groups (should give p ≈ 1)
        identical_p = categorical_pvalue(series1, series1)
        print(f"  Identical groups p-value: {identical_p:.6f}")
        
        # Test with empty series
        empty_series = pd.Series([], dtype=int)
        empty_p = categorical_pvalue(empty_series, series1)
        print(f"  Empty series p-value: {empty_p} (should be NaN)")
        
        # Test with constant series (no variation)
        constant_series = pd.Series([1] * 50)
        constant_p = categorical_pvalue(constant_series, series1)
        print(f"  Constant series p-value: {constant_p}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing categorical_pvalue function: {e}")
        return False


def test_master_config_database_paths():
    """Test database path configuration with master_config."""
    print("\n" + "="*60)
    print("TESTING MASTER_CONFIG DATABASE PATHS")
    print("="*60)
    
    try:
        from paper12_config import master_config, paths_config
        
        # Create paths configuration similar to project setup
        paths = paths_config(
            source_dir="../dbs",
            source_db="../dbs/pnk_db2_filtered.sqlite", 
            paper_dir="..",
            paper_in_db="../dbs/pnk_db2_p2_in.sqlite",
            paper_out_db="../dbs/pnk_db2_p2_out.sqlite",
        )
        
        # Create master config
        config = master_config(paths=paths)
        
        print("✓ Successfully created master_config with paths")
        print(f"✓ Input database path: {config.paths.paper_in_db}")
        print(f"✓ Output database path: {config.paths.paper_out_db}")
        
        # Check if database files exist
        input_db_exists = os.path.exists(config.paths.paper_in_db)
        output_db_exists = os.path.exists(config.paths.paper_out_db)
        
        print(f"✓ Input database exists: {input_db_exists}")
        print(f"✓ Output database exists: {output_db_exists}")
        
        if not input_db_exists:
            print(f"⚠ Warning: Input database not found at {config.paths.paper_in_db}")
            print("  This may be expected if the database hasn't been created yet")
        
        return True, config
        
    except Exception as e:
        print(f"✗ Error testing master_config database paths: {e}")
        return False, None


def test_database_connection_and_table_access(config):
    """Test database connection and access to timetoevent_wgc_compl table."""
    print("\n" + "="*60)
    print("TESTING DATABASE CONNECTION AND TABLE ACCESS")
    print("="*60)
    
    if config is None:
        print("✗ Cannot test database - no valid config provided")
        return False, None
    
    try:
        db_path = config.paths.paper_in_db
        
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
                
                # Get table schema and sample data
                schema_query = f"PRAGMA table_info({target_table})"
                schema_df = pd.read_sql_query(schema_query, conn)
                
                print(f"✓ Table schema for '{target_table}':")
                print(f"  Total columns: {len(schema_df)}")
                
                # Check for required outcome columns
                column_names = schema_df['name'].tolist()
                required_outcomes = ["10%_wl_achieved", "60d_dropout"]
                
                outcome_status = {}
                for outcome in required_outcomes:
                    if outcome in column_names:
                        outcome_status[outcome] = True
                        print(f"  ✓ Required outcome column found: {outcome}")
                    else:
                        outcome_status[outcome] = False
                        print(f"  ✗ Required outcome column missing: {outcome}")
                
                # Look for weight gain cause columns
                potential_wgc_cols = [col for col in column_names if any(keyword in col.lower() 
                                    for keyword in ['health', 'mental', 'family', 'medication', 
                                                  'physical', 'eating', 'schedule', 'smoking', 
                                                  'treatment', 'pandemic', 'lifestyle', 'none'])]
                
                print(f"  ✓ Potential weight gain cause columns ({len(potential_wgc_cols)}):")
                for col in potential_wgc_cols[:10]:  # Show first 10
                    print(f"    - {col}")
                if len(potential_wgc_cols) > 10:
                    print(f"    ... and {len(potential_wgc_cols) - 10} more")
                
                # Get sample data
                sample_query = f"SELECT * FROM {target_table} LIMIT 5"
                sample_df = pd.read_sql_query(sample_query, conn)
                
                print(f"✓ Successfully loaded sample data: {len(sample_df)} rows")
                
                # Get total row count
                count_query = f"SELECT COUNT(*) as total_rows FROM {target_table}"
                count_df = pd.read_sql_query(count_query, conn)
                total_rows = count_df['total_rows'].iloc[0]
                
                print(f"✓ Total rows in '{target_table}': {total_rows:,}")
                
                return True, {
                    'table_exists': True,
                    'total_rows': total_rows,
                    'total_columns': len(schema_df),
                    'column_names': column_names,
                    'outcome_status': outcome_status,
                    'potential_wgc_cols': potential_wgc_cols,
                    'sample_data': sample_df
                }
                
            else:
                print(f"✗ Target table '{target_table}' not found in database")
                print(f"  Available tables: {table_names}")
                return False, {
                    'table_exists': False,
                    'available_tables': table_names
                }
        
    except Exception as e:
        print(f"✗ Error testing database connection: {e}")
        return False, None


def test_output_directory_structure():
    """Test and validate output directory structure follows project conventions."""
    print("\n" + "="*60)
    print("TESTING OUTPUT DIRECTORY STRUCTURE")
    print("="*60)
    
    try:
        # Define expected output directory structure
        base_output_dir = "../outputs"
        descriptive_viz_dir = os.path.join(base_output_dir, "descriptive_visualizations")
        
        expected_subdirs = [
            "forest_plots",
            "summary_tables"
        ]
        
        print(f"Testing output directory structure:")
        print(f"  Base output directory: {base_output_dir}")
        print(f"  Descriptive viz directory: {descriptive_viz_dir}")
        
        # Check if base output directory exists
        if os.path.exists(base_output_dir):
            print(f"✓ Base output directory exists: {base_output_dir}")
        else:
            print(f"⚠ Base output directory does not exist: {base_output_dir}")
            print("  This is normal - it will be created when needed")
        
        # Check descriptive visualizations directory
        if os.path.exists(descriptive_viz_dir):
            print(f"✓ Descriptive visualizations directory exists: {descriptive_viz_dir}")
            
            # List existing contents
            existing_contents = os.listdir(descriptive_viz_dir)
            if existing_contents:
                print(f"  Existing contents ({len(existing_contents)} items):")
                for item in sorted(existing_contents):
                    item_path = os.path.join(descriptive_viz_dir, item)
                    if os.path.isdir(item_path):
                        print(f"    📁 {item}/")
                    else:
                        print(f"    📄 {item}")
            else:
                print("  Directory is empty")
        else:
            print(f"⚠ Descriptive visualizations directory does not exist: {descriptive_viz_dir}")
            print("  This is normal - it will be created when the pipeline runs")
        
        # Test directory creation capability
        test_dir = os.path.join(descriptive_viz_dir, "test_creation")
        try:
            os.makedirs(test_dir, exist_ok=True)
            print(f"✓ Successfully tested directory creation capability")
            
            # Clean up test directory
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
                print(f"✓ Test directory cleaned up")
                
        except Exception as e:
            print(f"⚠ Warning: Could not test directory creation: {e}")
        
        # Validate expected file naming conventions
        expected_files = [
            "forest_plots/risk_ratios_10pct_wl_achieved.png",
            "forest_plots/risk_differences_10pct_wl_achieved.png", 
            "forest_plots/risk_ratios_60d_dropout.png",
            "forest_plots/risk_differences_60d_dropout.png",
            "summary_tables/effect_sizes_summary.csv",
            "summary_tables/statistical_tests_summary.csv"
        ]
        
        print(f"\n✓ Expected output file structure:")
        for file_path in expected_files:
            print(f"  📄 {file_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing output directory structure: {e}")
        return False


def test_integration_with_actual_data(config, table_info):
    """Test integration with actual project data if available."""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH ACTUAL DATA")
    print("="*60)
    
    if config is None or table_info is None or not table_info.get('table_exists', False):
        print("✗ Cannot test with actual data - table not available")
        return False
    
    try:
        from descriptive_visualizations import load_forest_plot_data, calculate_effect_sizes, perform_statistical_tests
        from descriptive_comparisons import get_cause_cols
        
        # Create sample row_order for get_cause_cols
        sample_row_order = [
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
        
        print("Testing data loading with actual project data...")
        
        # Test load_forest_plot_data function
        df = load_forest_plot_data(
            input_table="timetoevent_wgc_compl",
            db_path=config.paths.paper_in_db,
            row_order=sample_row_order
        )
        
        print(f"✓ Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # Get cause columns
        cause_cols = get_cause_cols(sample_row_order)
        available_cause_cols = [col for col in cause_cols if col in df.columns]
        
        print(f"✓ Available weight gain cause columns: {len(available_cause_cols)}")
        
        # Test with a small subset for performance
        outcomes = ["10%_wl_achieved", "60d_dropout"]
        test_causes = available_cause_cols[:3]  # Test with first 3 causes
        
        print(f"Testing effect size calculations with {len(test_causes)} causes and {len(outcomes)} outcomes...")
        
        # Test calculate_effect_sizes function
        effect_sizes_df = calculate_effect_sizes(df, test_causes, outcomes)
        
        if len(effect_sizes_df) > 0:
            print(f"✓ Successfully calculated effect sizes: {len(effect_sizes_df)} results")
            print(f"  Columns: {list(effect_sizes_df.columns)}")
        else:
            print("⚠ Warning: No effect sizes calculated - this may indicate data issues")
        
        # Test perform_statistical_tests function
        print(f"Testing statistical tests with {len(test_causes)} causes and {len(outcomes)} outcomes...")
        
        stats_df = perform_statistical_tests(df, test_causes, outcomes)
        
        if len(stats_df) > 0:
            print(f"✓ Successfully performed statistical tests: {len(stats_df)} results")
            print(f"  Columns: {list(stats_df.columns)}")
        else:
            print("⚠ Warning: No statistical tests completed - this may indicate data issues")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing integration with actual data: {e}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        return False


def run_full_integration_validation():
    """Run complete integration validation suite."""
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
        cause_cols = []
    
    # Test 3: categorical_pvalue function  
    if import_results.get('descriptive_comparisons', False):
        categorical_pvalue_success = test_categorical_pvalue_function()
        validation_results['categorical_pvalue'] = categorical_pvalue_success
    else:
        validation_results['categorical_pvalue'] = False
    
    # Test 4: master_config database paths
    if import_results.get('paper12_config', False):
        config_success, config = test_master_config_database_paths()
        validation_results['master_config'] = config_success
    else:
        validation_results['master_config'] = False
        config = None
    
    # Test 5: Database connection and table access
    if config is not None:
        db_success, table_info = test_database_connection_and_table_access(config)
        validation_results['database_access'] = db_success
    else:
        validation_results['database_access'] = False
        table_info = None
    
    # Test 6: Output directory structure
    output_dir_success = test_output_directory_structure()
    validation_results['output_directory'] = output_dir_success
    
    # Test 7: Integration with actual data (if available)
    if (import_results.get('descriptive_visualizations', False) and 
        config is not None and 
        table_info is not None and 
        table_info.get('table_exists', False)):
        
        actual_data_success = test_integration_with_actual_data(config, table_info)
        validation_results['actual_data_integration'] = actual_data_success
    else:
        validation_results['actual_data_integration'] = False
        print("\n" + "="*60)
        print("SKIPPING ACTUAL DATA INTEGRATION TEST")
        print("="*60)
        print("Reason: Required modules not available or database/table not accessible")
    
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
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("⚠ MOST INTEGRATION TESTS PASSED")
        print("The pipeline should work but some features may have issues.")
    else:
        print("❌ INTEGRATION VALIDATION FAILED")
        print("Significant issues found. Please review and fix before using the pipeline.")
    
    print("="*80)
    
    return validation_results


if __name__ == "__main__":
    # Run the full validation suite
    results = run_full_integration_validation()
    
    # Exit with appropriate code
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    if passed_tests == total_tests:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed