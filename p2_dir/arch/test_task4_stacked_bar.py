"""
Test script for Task 4: Stacked bar plot legend and y-axis adjustments
"""
import sys
import os
import pandas as pd
import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from cluster_descriptions import cluster_categorical_distributions

# Create test data
np.random.seed(42)
n_samples = 200

test_data = pd.DataFrame({
    'cluster_id': np.random.choice([0, 1, 2], size=n_samples),
    'binary_var1': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
    'binary_var2': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
})

# Create output directory
output_dir = 'outputs/test_task4'
os.makedirs(output_dir, exist_ok=True)

print("Testing stacked bar plot with Task 4 enhancements...")
print("=" * 60)
print("Expected changes:")
print("1. Legend positioned above plot with bbox_to_anchor=(1.0, 1.08)")
print("2. Y-axis limit changed from 120 to 100")
print("3. Sample sizes and significance markers use clip_on=False")
print("=" * 60)

# Generate stacked bar plots
cluster_categorical_distributions(
    cluster_df=test_data,
    variables=['binary_var1', 'binary_var2'],
    output_dir=output_dir,
    cluster_col='cluster_id',
    calculate_significance=True,
    fdr_correction=True,
    alpha=0.05
)

print("\n" + "=" * 60)
print("✓ Test completed successfully!")
print(f"✓ Plots saved to: {output_dir}")
print("=" * 60)
print("\nVerification checklist:")
print("1. [ ] Legend is positioned above the plot area")
print("2. [ ] Y-axis shows 0-100% scale (not 0-120%)")
print("3. [ ] Sample sizes (n=X) are visible above 100%")
print("4. [ ] Significance markers (*/**) are visible above 100%")
print("5. [ ] No overlap between legend and plot elements")
