#!/usr/bin/env python3
"""
Enhanced Random Forest Pipeline - Usage Example

This script demonstrates how to use the enhanced Random Forest pipeline
with statistical significance testing and publication-ready visualizations.

Author: Enhanced RF Pipeline Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from sklearn.datasets import make_classification

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paper12_config import paper2_rf_config
from rf_engine import RandomForestAnalyzer


def create_example_dataset():
    """
    Create a synthetic dataset for demonstration purposes.
    
    This simulates a weight loss prediction dataset with realistic
    feature names and relationships.
    """
    print("Creating synthetic weight loss dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    # Define realistic feature names
    feature_names = [
        'age',
        'baseline_bmi', 
        'sex_f',
        'mental_health',
        'eating_habits',
        'physical_inactivity',
        'medication_use',
        'family_support'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add some realistic transformations
    df['age'] = (df['age'] * 20 + 45).clip(18, 80)  # Age 18-80
    df['baseline_bmi'] = (df['baseline_bmi'] * 10 + 30).clip(18, 50)  # BMI 18-50
    df['sex_f'] = (df['sex_f'] > 0).astype(int)  # Binary sex variable
    
    # Binary outcome: 10% weight loss achieved
    df['weight_loss_10pct'] = y
    
    print(f"✓ Created dataset with {len(df)} samples and {len(feature_names)} features")
    print(f"✓ Outcome distribution: {df['weight_loss_10pct'].value_counts().to_dict()}")
    
    return df, feature_names


def create_example_database(df, db_path='example_data.sqlite'):
    """Create SQLite database with example data."""
    print(f"Creating example database: {db_path}")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create new database
    with sqlite3.connect(db_path) as conn:
        df.to_sql('weight_loss_analysis', conn, index=False, if_exists='replace')
    
    print(f"✓ Database created with table 'weight_loss_analysis'")
    return db_path


def example_basic_analysis():
    """
    Example 1: Basic enhanced analysis with default settings.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Enhanced Analysis")
    print("="*60)
    
    # Create example data
    df, feature_names = create_example_dataset()
    db_path = create_example_database(df)
    
    # Configure analysis with nice feature names
    nice_names = {
        'age': 'Age (years)',
        'baseline_bmi': 'Baseline BMI',
        'sex_f': 'Sex (Female)',
        'mental_health': 'Mental Health Issues',
        'eating_habits': 'Poor Eating Habits',
        'physical_inactivity': 'Physical Inactivity',
        'medication_use': 'Medication Use',
        'family_support': 'Family Support'
    }
    
    config = paper2_rf_config(
        analysis_name='basic_weight_loss_analysis',
        outcome_variable='weight_loss_10pct',
        model_type='classifier',
        predictors=feature_names,
        classifier_threshold=0.5,
        threshold_direction='greater_than_or_equal',
        db_path=db_path,
        input_table='weight_loss_analysis',
        output_dir='example_outputs',
        nice_names=nice_names
    )
    
    # Run enhanced analysis
    print("\nRunning enhanced Random Forest analysis...")
    analyzer = RandomForestAnalyzer(config)
    analyzer.run_and_generate_outputs()
    
    print("\n✓ Basic analysis complete!")
    print("✓ Check 'example_outputs' directory for results:")
    print("  - basic_weight_loss_analysis_roc_curve.png")
    print("  - basic_weight_loss_analysis_primary_FI_composite.png")
    print("  - basic_weight_loss_analysis_secondary_FI_composite.png")
    
    # Cleanup
    os.remove(db_path)


def example_custom_configuration():
    """
    Example 2: Custom configuration with specific significance settings.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create example data
    df, feature_names = create_example_dataset()
    db_path = create_example_database(df)
    
    # Configure with custom settings
    config = paper2_rf_config(
        analysis_name='custom_analysis',
        outcome_variable='weight_loss_10pct',
        model_type='classifier',
        predictors=feature_names[:6],  # Use only first 6 features
        
        # Custom significance testing settings
        enable_gini_significance=True,
        enable_shap_significance=True,
        significance_alpha=0.01,  # More stringent significance level
        
        # Custom visualization settings
        figure_width_primary=18.0,
        figure_height_primary=12.0,
        figure_width_secondary=16.0,
        figure_height_secondary=10.0,
        
        # Standard settings
        classifier_threshold=0.5,
        threshold_direction='greater_than_or_equal',
        db_path=db_path,
        input_table='weight_loss_analysis',
        output_dir='example_outputs'
    )
    
    print("\nRunning custom configured analysis...")
    analyzer = RandomForestAnalyzer(config)
    analyzer.run_and_generate_outputs()
    
    print("\n✓ Custom analysis complete!")
    print("✓ Used more stringent significance level (α = 0.01)")
    print("✓ Custom figure dimensions for better readability")
    
    # Cleanup
    os.remove(db_path)


def example_accessing_results():
    """
    Example 3: Accessing and interpreting significance results.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Accessing Significance Results")
    print("="*60)
    
    # Create example data
    df, feature_names = create_example_dataset()
    db_path = create_example_database(df)
    
    config = paper2_rf_config(
        analysis_name='results_access_example',
        outcome_variable='weight_loss_10pct',
        model_type='classifier',
        predictors=feature_names,
        classifier_threshold=0.5,
        threshold_direction='greater_than_or_equal',
        db_path=db_path,
        input_table='weight_loss_analysis',
        output_dir='example_outputs'
    )
    
    # Run analysis and access results
    print("\nRunning analysis and extracting results...")
    analyzer = RandomForestAnalyzer(config)
    analyzer.run_analysis()
    analyzer._test_feature_significance()
    
    # Access significance results
    sig_results = analyzer.results['significance_results']
    
    print(f"\n--- Significance Testing Results ---")
    print(f"Significance level (α): {sig_results.alpha_level}")
    print(f"Features tested: {sig_results.n_features_tested}")
    print(f"Shadow features created: {sig_results.n_shadow_features}")
    print(f"Gini significance threshold: {sig_results.gini_threshold:.6f}")
    
    print(f"\n--- Gini Importance Significance ---")
    print(f"Significant features ({len(sig_results.gini_significant_features)}):")
    for feat in sig_results.gini_significant_features:
        gini_importance = analyzer.results['gini_importance'][feat]
        print(f"  • {feat}: {gini_importance:.6f} (> {sig_results.gini_threshold:.6f})")
    
    print(f"\n--- SHAP Value Significance ---")
    print(f"Significant features ({len(sig_results.shap_significant_features)}):")
    for feat in sig_results.shap_significant_features:
        raw_p = sig_results.shap_pvalues.get(feat, 'N/A')
        adj_p = sig_results.shap_adjusted_pvalues.get(feat, 'N/A')
        print(f"  • {feat}: raw p={raw_p:.6f}, adjusted p={adj_p:.6f}")
    
    # Access other results
    print(f"\n--- Model Performance ---")
    if 'auroc' in analyzer.results:
        print(f"AUROC: {analyzer.results['auroc']:.3f}")
    if 'f1_score' in analyzer.results:
        print(f"F1 Score: {analyzer.results['f1_score']:.3f}")
    
    print("\n✓ Results access example complete!")
    
    # Cleanup
    os.remove(db_path)


def example_legacy_compatibility():
    """
    Example 4: Using legacy method for backward compatibility.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Legacy Compatibility")
    print("="*60)
    
    # Create example data
    df, feature_names = create_example_dataset()
    db_path = create_example_database(df)
    
    config = paper2_rf_config(
        analysis_name='legacy_example',
        outcome_variable='weight_loss_10pct',
        model_type='classifier',
        predictors=feature_names,
        classifier_threshold=0.5,
        threshold_direction='greater_than_or_equal',
        db_path=db_path,
        input_table='weight_loss_analysis',
        output_dir='example_outputs'
    )
    
    print("\nRunning analysis with legacy output format...")
    analyzer = RandomForestAnalyzer(config)
    analyzer.run_and_generate_outputs_legacy()  # Use legacy method
    
    print("\n✓ Legacy analysis complete!")
    print("✓ Generated old-style plots for backward compatibility")
    print("✓ Check 'example_outputs' directory for legacy format results")
    
    # Cleanup
    os.remove(db_path)


def main():
    """
    Main function to run all examples.
    """
    print("Enhanced Random Forest Pipeline - Usage Examples")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('example_outputs', exist_ok=True)
    
    try:
        # Run all examples
        example_basic_analysis()
        example_custom_configuration()
        example_accessing_results()
        example_legacy_compatibility()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files in 'example_outputs' directory:")
        
        # List generated files
        if os.path.exists('example_outputs'):
            files = os.listdir('example_outputs')
            for file in sorted(files):
                print(f"  • {file}")
        
        print(f"\nTotal files generated: {len(files) if 'files' in locals() else 0}")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)