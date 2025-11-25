"""
Test script for Task 9: Violin plot cluster-specific colors
Tests the subplot approach with cluster-specific colors for each violin.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from cluster_descriptions import cluster_continuous_distributions

# Create test data with multiple clusters
np.random.seed(42)
n_samples = 300

test_data = pd.DataFrame({
    'cluster_id': np.random.choice([0, 1, 2, 3], size=n_samples),
    'age': np.random.normal(50, 15, n_samples),
    'bmi': np.random.normal(25, 5, n_samples),
    'total_wl_%': np.random.normal(10, 5, n_samples)
})

# Create output directory
output_dir = 'outputs/test_task9_violin_colors'
os.makedirs(output_dir, exist_ok=True)

print("Testing Task 9: Violin plot cluster-specific colors")
print("=" * 60)

# Test with multiple variables
print("\nGenerating violin plots with cluster-specific colors...")
print("Variables: age, bmi, total_wl_%")
print("Clusters: 0, 1, 2, 3")

cluster_continuous_distributions(
    cluster_df=test_data,
    variables=['age', 'bmi', 'total_wl_%'],
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
print("  [ ] Each cluster has its own subplot")
print("  [ ] Each cluster uses its configured color from cluster_config.json")
print("  [ ] Population side (left) uses consistent purple color across all subplots")
print("  [ ] Cluster side (right) uses cluster-specific color")
print("  [ ] Cluster labels appear as subplot titles")
print("  [ ] Y-axis label appears on leftmost subplot only")
print("  [ ] Legend appears on first subplot only")
print("  [ ] Significance markers appear above each subplot")
print("  [ ] Figure size adjusts based on number of clusters")
print("\nPlease visually inspect the generated plots to confirm these requirements.")
print("\nExpected colors from cluster_config.json:")
print("  - Cluster 0 (Male-dominant, inactive): #FF6700 (orange)")
print("  - Cluster 1 (Women's health): #1f77b4 (blue)")
print("  - Cluster 2 (Metabolic): #2ca02c (green)")
print("  - Cluster 3 (Musculoskeletal): #d62728 (red)")
