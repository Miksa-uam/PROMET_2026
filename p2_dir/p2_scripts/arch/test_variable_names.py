#!/usr/bin/env python3
"""
Test script to verify the centralized variable names system works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from variable_names_utils import (
    get_human_readable_name, 
    get_all_variable_names, 
    variable_exists,
    print_all_variables
)
from rf_feature_importances import RandomForestAnalyzer
from paper12_config import paper2_rf_config

def test_variable_names_utility():
    """Test the variable names utility functions."""
    print("="*60)
    print("TESTING CENTRALIZED VARIABLE NAMES SYSTEM")
    print("="*60)
    
    # Test 1: Load all variables
    print("\n1. Testing variable loading...")
    all_vars = get_all_variable_names()
    print(f"Loaded {len(all_vars)} variable mappings")
    
    # Test 2: Test specific lookups
    print("\n2. Testing specific variable lookups...")
    test_variables = [
        "age", "baseline_bmi", "womens_health_and_pregnancy", 
        "40d_wl_%", "5%_wl_achieved", "nonexistent_variable"
    ]
    
    for var in test_variables:
        human_name = get_human_readable_name(var)
        exists = variable_exists(var)
        print(f"  {var:<35} -> {human_name:<40} (exists: {exists})")
    
    # Test 3: Show all variables
    print("\n3. All available variables:")
    print_all_variables()
    
    return True

def test_rf_integration():
    """Test that the RF pipeline works with the centralized dictionary."""
    print("\n" + "="*60)
    print("TESTING RF INTEGRATION WITH CENTRALIZED NAMES")
    print("="*60)
    
    try:
        # Create a minimal RF config (no nice_names parameter needed anymore)
        config = paper2_rf_config(
            analysis_name="Variable Names Integration Test",
            outcome_variable="10%_wl_achieved",
            model_type="classifier",
            classifier_threshold=0.5,
            threshold_direction="greater_than_or_equal",
            
            # Test with a few variables
            predictors=["womens_health_and_pregnancy", "mental_health", "family_issues"],
            covariates=["age", "sex_f", "baseline_bmi"],
            
            # Minimal settings
            enable_gini_significance=False,  # Skip significance for quick test
            enable_shap_significance=False,
            
            # Paths
            db_path="../dbs/pnk_db2_p2_in.sqlite",
            input_table="timetoevent_wgc_compl",
            output_dir="../outputs/rf_test_outputs",
            
            # Small figure for quick test
            figure_width_primary=12.0,
            figure_height_primary=6.0,
            max_features_display=6
        )
        
        print("✓ RF config created successfully (no nice_names parameter needed)")
        
        # Test the _get_nice_name method
        analyzer = RandomForestAnalyzer(config)
        
        test_vars = ["age", "baseline_bmi", "womens_health_and_pregnancy"]
        print("\nTesting _get_nice_name method:")
        for var in test_vars:
            nice_name = analyzer._get_nice_name(var)
            print(f"  {var:<35} -> {nice_name}")
        
        print("✓ RF integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ RF integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing centralized variable names system...")
    
    success = True
    
    # Test the utility functions
    try:
        test_variable_names_utility()
        print("✓ Variable names utility tests passed")
    except Exception as e:
        print(f"✗ Variable names utility tests failed: {e}")
        success = False
    
    # Test RF integration
    try:
        rf_success = test_rf_integration()
        if rf_success:
            print("✓ RF integration tests passed")
        else:
            success = False
    except Exception as e:
        print(f"✗ RF integration tests failed: {e}")
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED - Centralized variable names system is working!")
    else:
        print("❌ SOME TESTS FAILED - Check the output above")
    print("="*60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)