#!/usr/bin/env python3
"""
Test script to verify the Random Forest pipeline fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rf_feature_importances import RandomForestAnalyzer
from paper12_config import paper2_rf_config

def test_rf_pipeline():
    """Test the fixed RF pipeline with a sample configuration."""
    
    # Create a test configuration
    config = paper2_rf_config(
        analysis_name="Test RF Pipeline - 10% WL Achievement",
        outcome_variable="10%_wl_achieved",
        model_type="classifier",
        classifier_threshold=0.5,
        threshold_direction="greater_than_or_equal",
        
        # Features to test
        predictors=[
            "womens_health_and_pregnancy", "mental_health", "family_issues",
            "medication_disease_injury", "physical_inactivity", "eating_habits",
            "schedule", "smoking_cessation", "treatment_discontinuation_or_relapse",
            "pandemic", "lifestyle_circumstances", "none_of_above"
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
        figure_height_primary=10.0,
        max_features_display=15
    )
    
    print("="*60)
    print("TESTING RANDOM FOREST PIPELINE FIXES")
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
        
        # Generate plots
        print("\n3. Generating plots...")
        analyzer._plot_primary_composite()
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Print summary
        if 'significance_results' in analyzer.results:
            sig_results = analyzer.results['significance_results']
            print(f"\nSUMMARY:")
            print(f"- Gini significant features (95th percentile method): {len(sig_results.gini_significant_features)}")
            print(f"- SHAP significant features (Wilcoxon + FDR): {len(sig_results.shap_significant_features)}")
            print(f"- Gini threshold (95th percentile): {sig_results.gini_threshold:.6f}")
            print(f"- Gini significant: {sig_results.gini_significant_features}")
            print(f"- SHAP significant: {sig_results.shap_significant_features}")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_rf_pipeline()
    sys.exit(0 if success else 1)