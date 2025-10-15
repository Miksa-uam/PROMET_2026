#!/usr/bin/env python3
"""
Test script specifically for SHAP ordering bug fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rf_feature_importances import RandomForestAnalyzer
from paper12_config import paper2_rf_config

def test_shap_ordering():
    """Test that SHAP feature ordering is correctly handled."""
    
    # Create a minimal test configuration
    config = paper2_rf_config(
        analysis_name="SHAP Ordering Test",
        outcome_variable="10%_wl_achieved",
        model_type="classifier",
        classifier_threshold=0.5,
        threshold_direction="greater_than_or_equal",
        
        # Just a few features to make debugging easier
        predictors=[
            "womens_health_and_pregnancy", "mental_health", "family_issues",
            "medication_disease_injury", "physical_inactivity"
        ],
        covariates=["age", "sex_f", "baseline_bmi"],
        
        # Testing configuration
        enable_gini_significance=True,
        enable_shap_significance=True,
        significance_alpha=0.05,
        
        # Paths
        db_path="../dbs/pnk_db2_p2_in.sqlite",
        input_table="timetoevent_wgc_compl",
        output_dir="../outputs/rf_test_outputs",
        
        # Visualization
        figure_width_primary=16.0,
        figure_height_primary=8.0,
        max_features_display=8
    )
    
    print("="*60)
    print("TESTING SHAP ORDERING FIX")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = RandomForestAnalyzer(config)
        
        # Run the analysis
        print("\n1. Running analysis...")
        analyzer.run_analysis()
        
        # Test significance testing
        print("\n2. Testing significance...")
        analyzer._test_feature_significance()
        
        # Generate plots - this is where the bug should be fixed
        print("\n3. Generating plots (testing SHAP ordering)...")
        analyzer._plot_primary_composite()
        
        print("\n" + "="*60)
        print("SHAP ORDERING TEST COMPLETED!")
        print("="*60)
        
        # Check if the debug output shows correct mapping
        print("\nPlease check the debug output above for:")
        print("- No 'WARNING: Label mismatch' messages in SHAP panel")
        print("- Correct feature-to-label mapping")
        print("- Proper asterisk placement")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_shap_ordering()
    sys.exit(0 if success else 1)