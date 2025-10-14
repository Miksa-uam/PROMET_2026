#!/usr/bin/env python3
"""
Regenerate forest plots with enhanced publication-ready styling
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from descriptive_visualizations import _create_single_forest_plot

def create_sample_forest_plot_data():
    """Create sample data to test the enhanced forest plot styling"""
    
    # Sample data structure based on the existing format
    causes = [
        "Womens Health And Pregnancy",
        "Mental Health", 
        "Family Issues",
        "Medication Disease Injury",
        "Physical Inactivity",
        "Eating Habits",
        "Schedule",
        "Smoking Cessation",
        "Treatment Discontinuation Or Relapse",
        "Pandemic",
        "Lifestyle Circumstances",
        "None Of Above"
    ]
    
    # Generate sample risk ratios and confidence intervals
    np.random.seed(42)  # For reproducible results
    
    data = []
    for i, cause in enumerate(causes):
        # Generate realistic risk ratios around 1.0
        rr = np.random.normal(1.0, 0.3)
        rr = max(0.3, min(3.0, rr))  # Keep within reasonable bounds
        
        # Generate confidence intervals
        ci_width = np.random.uniform(0.2, 0.8)
        ci_lower = max(0.1, rr - ci_width/2)
        ci_upper = min(5.0, rr + ci_width/2)
        
        # Sample sizes
        n_present = np.random.randint(50, 500)
        n_absent = np.random.randint(1000, 2000)
        
        data.append({
            'cause': cause,
            'cause_pretty': cause,
            'risk_ratio': rr,
            'rr_ci_lower': ci_lower,
            'rr_ci_upper': ci_upper,
            'risk_difference': np.random.normal(0, 5),  # Risk difference in %
            'rd_ci_lower': np.random.normal(-8, 3),
            'rd_ci_upper': np.random.normal(8, 3),
            'n_present': n_present,
            'n_absent': n_absent
        })
    
    return pd.DataFrame(data)

def test_enhanced_forest_plots():
    """Generate test forest plots with enhanced styling"""
    
    print("=== Generating Enhanced Forest Plots ===")
    
    # Create sample data
    df = create_sample_forest_plot_data()
    print(f"✓ Created sample data with {len(df)} causes")
    
    # Create output directory
    output_dir = "outputs/enhanced_forest_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Risk Ratio plot
    print("Creating enhanced Risk Ratio plot...")
    rr_output = os.path.join(output_dir, "enhanced_risk_ratios_test.png")
    
    _create_single_forest_plot(
        data=df,
        output_path=rr_output,
        effect_col='risk_ratio',
        ci_lower_col='rr_ci_lower',
        ci_upper_col='rr_ci_upper',
        effect_label='Risk Ratio (RR)',
        outcome_label='10% Weight Loss Achievement',
        reference_value=1.0,
        use_log_scale=True
    )
    
    print(f"✓ Risk Ratio plot saved: {rr_output}")
    
    # Generate Risk Difference plot
    print("Creating enhanced Risk Difference plot...")
    rd_output = os.path.join(output_dir, "enhanced_risk_differences_test.png")
    
    _create_single_forest_plot(
        data=df,
        output_path=rd_output,
        effect_col='risk_difference',
        ci_lower_col='rd_ci_lower',
        ci_upper_col='rd_ci_upper',
        effect_label='Risk Difference (RD)',
        outcome_label='10% Weight Loss Achievement',
        reference_value=0.0,
        use_log_scale=False
    )
    
    print(f"✓ Risk Difference plot saved: {rd_output}")
    
    print("\n🎉 Enhanced forest plots generated successfully!")
    print(f"📁 Check your plots in: {output_dir}")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_forest_plots()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()