#!/usr/bin/env python3
"""
Quick test to verify the layout fixes for longer variable names.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rf_feature_importances import RandomForestAnalyzer
from paper12_config import paper2_rf_config

def test_layout_fix():
    """Test the improved layout for longer variable names."""
    
    print("="*60)
    print("TESTING LAYOUT FIX FOR LONGER VARIABLE NAMES")
    print("="*60)
    
    try:
        # Create a test configuration with a few variables
        config = paper2_rf_config(
            analysis_name="Layout Test - Longer Names",
            outcome_variable="10%_wl_achieved",
            model_type="classifier",
            classifier_threshold=0.5,
            threshold_direction="greater_than_or_equal",
            
            # Test with variables that have long descriptive names
            predictors=[
                "womens_health_and_pregnancy", 
                "treatment_discontinuation_or_relapse",
                "medication_disease_injury",
                "lifestyle_circumstances"
            ],
            covariates=["age", "sex_f", "baseline_bmi"],
            
            # Quick test settings
            enable_gini_significance=True,
            enable_shap_significance=True,
            run_hyperparameter_tuning=False,  # Skip for speed
            
            # Paths
            db_path="../dbs/pnk_db2_p2_in.sqlite",
            input_table="timetoevent_wgc_compl",
            output_dir="../outputs/rf_test_outputs",
            
            # Test with smaller figure initially
            figure_width_primary=18.0,  # Will be increased automatically
            figure_height_primary=8.0,
            max_features_display=7
        )
        
        print("✓ Configuration created")
        
        # Initialize and run quick analysis
        analyzer = RandomForestAnalyzer(config)
        print("✓ Analyzer initialized")
        
        # Run analysis
        analyzer.run_analysis()
        print("✓ Analysis completed")
        
        # Test significance
        analyzer._test_feature_significance()
        print("✓ Significance testing completed")
        
        # Generate the plot with improved layout
        analyzer._plot_primary_composite()
        print("✓ Primary composite plot generated with improved layout")
        
        print("\n" + "="*60)
        print("🎉 LAYOUT TEST COMPLETED SUCCESSFULLY!")
        print("Check the generated plot for:")
        print("- Proper spacing in Gini panel (left side)")
        print("- Readable variable names")
        print("- Correct asterisk placement")
        print("- Balanced proportions between panels")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Layout test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_layout_fix()
    sys.exit(0 if success else 1)