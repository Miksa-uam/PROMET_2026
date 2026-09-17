#!/usr/bin/env python3
"""
Validation script for WGC vs Population Mean Analysis table naming convention.

This script validates that the table naming follows the '[input_cohort]_wgc_strt_vs_mean' pattern
as specified in requirement 1.2. It tests with different input cohort names and verifies
database table creation with correct naming.

Requirements tested:
- 1.2: Table names follow '[input_cohort]_wgc_strt_vs_mean' pattern
"""

import os
import sys
import sqlite3
import pandas as pd
import tempfile
import logging
from typing import List, Tuple, Dict

# Add the scripts directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper12_config import descriptive_comparisons_config
from descriptive_comparisons import wgc_vs_population_mean_analysis

# Configure logging for validation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data() -> pd.DataFrame:
    """
    Creates a minimal test dataset with the required structure for WGC analysis.
    
    Returns:
        pd.DataFrame: Test dataset with demographic variables and WGC indicators
    """
    # Create test data with required columns
    test_data = {
        'patient_id': range(1, 101),  # 100 test patients
        'age': [25 + (i % 50) for i in range(100)],  # Ages 25-74
        'sex_f': [i % 2 for i in range(100)],  # Alternating male/female
        'baseline_bmi': [20 + (i % 20) for i in range(100)],  # BMI 20-39
        'baseline_weight': [60 + (i % 40) for i in range(100)],  # Weight 60-99
        'mental_health': [1 if i < 20 else 0 for i in range(100)],  # 20% with mental health WGC
        'eating_habits': [1 if 20 <= i < 40 else 0 for i in range(100)],  # 20% with eating habits WGC
        'physical_inactivity': [1 if 40 <= i < 60 else 0 for i in range(100)],  # 20% with physical inactivity WGC
    }
    
    return pd.DataFrame(test_data)

def create_test_config(input_cohort_name: str) -> descriptive_comparisons_config:
    """
    Creates a test configuration for the specified input cohort name.
    
    Args:
        input_cohort_name: Name of the input cohort to test
        
    Returns:
        descriptive_comparisons_config: Test configuration object
    """
    # Minimal row order for testing
    test_row_order = [
        ("N", "N"),
        ("age", "Age (years)"),
        ("sex_f", "Female sex"),
        ("baseline_bmi", "Baseline BMI (kg/m²)"),
        ("baseline_weight", "Baseline weight (kg)"),
        ("delim_wgc", "Weight Gain Causes"),
        ("mental_health", "Mental Health (yes/no)"),
        ("eating_habits", "Eating Habits (yes/no)"),
        ("physical_inactivity", "Physical Inactivity (yes/no)"),
    ]
    
    # WGC strata for testing
    test_wgc_strata = ["mental_health", "eating_habits", "physical_inactivity"]
    
    return descriptive_comparisons_config(
        analysis_name=f"test_{input_cohort_name}",
        input_cohort_name=input_cohort_name,
        mother_cohort_name="test_mother",
        row_order=test_row_order,
        demographic_output_table=f"{input_cohort_name}_demo_test",
        demographic_strata=["age", "sex_f"],
        wgc_output_table=f"{input_cohort_name}_wgc_test",
        wgc_strata=test_wgc_strata,
        fdr_correction=False  # Disable FDR for simpler testing
    )

def validate_table_naming(input_cohort_name: str) -> Tuple[bool, str, str]:
    """
    Validates table naming convention for a specific input cohort name.
    
    Args:
        input_cohort_name: Name of the input cohort to test
        
    Returns:
        Tuple[bool, str, str]: (success, expected_table_name, actual_table_name)
    """
    logger.info(f"Validating table naming for input cohort: {input_cohort_name}")
    
    # Expected table name according to the specification
    expected_table_name = f"{input_cohort_name}_wgc_strt_vs_mean"
    
    # Create test data and configuration
    test_df = create_test_data()
    test_config = create_test_config(input_cohort_name)
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as temp_db:
        temp_db_path = temp_db.name
    
    try:
        # Connect to temporary database
        with sqlite3.connect(temp_db_path) as conn:
            # Run the WGC vs population mean analysis
            wgc_vs_population_mean_analysis(test_df, test_config, conn)
            
            # Check what tables were created
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Look for the expected table name
            if expected_table_name in tables:
                logger.info(f"✓ Table '{expected_table_name}' created successfully")
                return True, expected_table_name, expected_table_name
            else:
                # Find any table that might be the WGC vs mean table
                wgc_tables = [t for t in tables if 'wgc_strt_vs_mean' in t]
                actual_table = wgc_tables[0] if wgc_tables else "No WGC table found"
                logger.error(f"✗ Expected table '{expected_table_name}' not found. Found tables: {tables}")
                return False, expected_table_name, actual_table
                
    except Exception as e:
        logger.error(f"✗ Error during validation: {str(e)}")
        return False, expected_table_name, f"Error: {str(e)}"
    
    finally:
        # Clean up temporary database
        try:
            os.unlink(temp_db_path)
        except:
            pass

def validate_table_structure(input_cohort_name: str) -> Tuple[bool, List[str]]:
    """
    Validates that the created table has the expected structure.
    
    Args:
        input_cohort_name: Name of the input cohort to test
        
    Returns:
        Tuple[bool, List[str]]: (success, column_names)
    """
    logger.info(f"Validating table structure for input cohort: {input_cohort_name}")
    
    # Create test data and configuration
    test_df = create_test_data()
    test_config = create_test_config(input_cohort_name)
    expected_table_name = f"{input_cohort_name}_wgc_strt_vs_mean"
    
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as temp_db:
        temp_db_path = temp_db.name
    
    try:
        # Connect to temporary database
        with sqlite3.connect(temp_db_path) as conn:
            # Run the WGC vs population mean analysis
            wgc_vs_population_mean_analysis(test_df, test_config, conn)
            
            # Get table structure
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({expected_table_name});")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]  # Column name is at index 1
            
            # Expected columns based on the specification
            expected_columns = [
                "Variable",
                "Population Mean (±SD) or N (%)"
            ]
            
            # Add WGC group columns (each WGC should have mean/n and p-value columns)
            wgc_groups = ["Mental Health", "Eating Habits", "Physical Inactivity"]
            for wgc in wgc_groups:
                expected_columns.extend([
                    f"{wgc}: Mean/N",
                    f"{wgc}: p-value"
                ])
            
            # Check if all expected columns are present
            missing_columns = [col for col in expected_columns if col not in column_names]
            extra_columns = [col for col in column_names if col not in expected_columns]
            
            if not missing_columns and not extra_columns:
                logger.info(f"✓ Table structure is correct with {len(column_names)} columns")
                return True, column_names
            else:
                if missing_columns:
                    logger.error(f"✗ Missing columns: {missing_columns}")
                if extra_columns:
                    logger.warning(f"! Extra columns: {extra_columns}")
                return False, column_names
                
    except Exception as e:
        logger.error(f"✗ Error during structure validation: {str(e)}")
        return False, []
    
    finally:
        # Clean up temporary database
        try:
            os.unlink(temp_db_path)
        except:
            pass

def run_comprehensive_validation() -> Dict[str, Dict[str, any]]:
    """
    Runs comprehensive validation tests for table naming convention.
    
    Returns:
        Dict[str, Dict[str, any]]: Validation results for each test case
    """
    logger.info("Starting comprehensive table naming validation")
    
    # Test cases based on the task requirements and existing usage
    test_cases = [
        "wgc_gen_compl",           # From existing configuration
        "timetoevent_wgc_compl",   # From existing configuration  
        "timetoevent_wgc_gen_compl", # From existing configuration
        "test_cohort",             # Simple test case
        "my_cohort_123",           # Test with numbers
        "cohort_with_underscores", # Test with multiple underscores
    ]
    
    results = {}
    
    for cohort_name in test_cases:
        logger.info(f"\n--- Testing cohort: {cohort_name} ---")
        
        # Test table naming
        naming_success, expected_name, actual_name = validate_table_naming(cohort_name)
        
        # Test table structure
        structure_success, columns = validate_table_structure(cohort_name)
        
        results[cohort_name] = {
            'naming_success': naming_success,
            'expected_table_name': expected_name,
            'actual_table_name': actual_name,
            'structure_success': structure_success,
            'columns': columns,
            'overall_success': naming_success and structure_success
        }
        
        if results[cohort_name]['overall_success']:
            logger.info(f"✓ All tests passed for cohort: {cohort_name}")
        else:
            logger.error(f"✗ Some tests failed for cohort: {cohort_name}")
    
    return results

def print_validation_summary(results: Dict[str, Dict[str, any]]) -> None:
    """
    Prints a summary of validation results.
    
    Args:
        results: Validation results from run_comprehensive_validation()
    """
    print("\n" + "="*80)
    print("TABLE NAMING CONVENTION VALIDATION SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['overall_success'])
    
    print(f"Total test cases: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    print("-" * 80)
    
    for cohort_name, result in results.items():
        status = "✓ PASS" if result['overall_success'] else "✗ FAIL"
        print(f"{status} | {cohort_name}")
        print(f"      Expected table: {result['expected_table_name']}")
        print(f"      Actual table:   {result['actual_table_name']}")
        print(f"      Naming test:    {'✓' if result['naming_success'] else '✗'}")
        print(f"      Structure test: {'✓' if result['structure_success'] else '✗'}")
        if result['columns']:
            print(f"      Columns ({len(result['columns'])}): {', '.join(result['columns'][:3])}{'...' if len(result['columns']) > 3 else ''}")
        print()
    
    print("="*80)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Table naming convention is correctly implemented.")
    else:
        print("⚠️  SOME TESTS FAILED! Please review the implementation.")
    
    print("="*80)

def main():
    """Main function to run the validation."""
    try:
        # Run comprehensive validation
        results = run_comprehensive_validation()
        
        # Print summary
        print_validation_summary(results)
        
        # Return appropriate exit code
        all_passed = all(r['overall_success'] for r in results.values())
        return 0 if all_passed else 1
        
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        print(f"\n❌ VALIDATION FAILED: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)