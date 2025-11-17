"""
Test script for Task 5: Violin plot legend and label positioning
Tests the legend positioning and x-axis label alignment changes.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from cluster_descriptions import cluster_continuous_distributions

# Create test data
np.random.seed(42)
n_samples = 200

test_data = pd.DataFrame({
    'cluster_id': np.random.choice([0, 1, 2], size=n_samples),
    'age': np.random.normal(50, 15, n_samples),
    'bmi': np.random.normal(25, 5, n_samples)
})

# Create output directory
output_dir = 'outputs/test_task5_violin'
os.makedirs(output_dir, exist_ok=True)

print("Testing Task 5: Violin plot legend and label positioning")
print("=" * 60)

# Test with age variable
print("\nGenerating violin plot for 'age' variable...")
cluster_continuous_distributions(
    cluster_df=test_data,
    variables=['age'],
    output_dir=output_dir,
    cluster_col='cluster_id',
    name_map_path='scripts/human_readable_variable_names.json',
    cluster_config_path='scripts/cluster_config.json',
    calculate_significance=True,
    fdr_correction=True,
    alpha=0.05
)

print("\n" + "=" * 60)
print("✓ Test completed successfully!")
print(f"✓ Output saved to: {output_dir}")
print("\nVerification checklist:")
print("  [ ] Legend is positioned above the plot area")
print("  [ ] Legend uses loc='upper center' with bbox_to_anchor=(0.5, 1.08)")
print("  [ ] Legend displays in 2 columns (ncol=2)")
print("  [ ] X-axis labels are centered on violin distributions")
print("  [ ] X-axis labels are rotated 45 degrees")
print("\nPlease visually inspect the generated plot to confirm these requirements.")
