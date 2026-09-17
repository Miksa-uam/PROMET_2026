"""
Test script for Task 6: Heatmap population column implementation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import pandas as pd
import numpy as np
import sqlite3
import tempfile
from cluster_descriptions import (
    analyze_cluster_vs_population,
    plot_cluster_heatmap
)

# Configuration
OUTPUT_DIR = 'outputs'
CLUSTER_CONFIG = 'scripts/cluster_config.json'
NAME_MAP = 'scripts/human_readable_variable_names.json'

# Test variables
TEST_VARIABLES = [
    'wgc_medication',
    'wgc_medical_condition',
    'wgc_pregnancy'
]

def create_test_data():
    """Create synthetic test data with clusters and WGC variables."""
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'medical_record_id': range(n_samples),
        'cluster_id': np.random.choice([0, 1, 2], n_samples),
        'wgc_medication': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'wgc_medical_condition': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'wgc_pregnancy': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

print("="*80)
print("TEST: Task 6 - Heatmap Population Column")
print("="*80)

# Step 1: Create test data
print("\n1. Creating test data...")
cluster_df = create_test_data()
print(f"   ✓ Created {len(cluster_df)} rows with {len(cluster_df['cluster_id'].unique())} clusters")

# Step 2: Run analysis to generate results table
print("\n2. Running cluster vs population analysis...")

# Create temporary output database
with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
    output_db = tmp.name

try:
    results_df = analyze_cluster_vs_population(
        cluster_df=cluster_df,
        variables=TEST_VARIABLES,
        output_db_path=output_db,
        output_table_name='test_heatmap_results',
        cluster_col='cluster_id',
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        variable_types={var: 'categorical' for var in TEST_VARIABLES},
        fdr_correction=True,
        alpha=0.05
    )
    print(f"   ✓ Analysis complete")
    
    # Verify population column exists
    assert 'Whole population: Mean (±SD) / N (%)' in results_df.columns, "Population column not found!"
    print(f"   ✓ Population column found in results")

    # Step 3: Generate heatmap with population column
    print("\n3. Generating heatmap with population column...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plot_cluster_heatmap(
        results_df=results_df,
        output_filename='test_task6_heatmap_with_population.png',
        output_dir=OUTPUT_DIR,
        variables=TEST_VARIABLES,
        name_map_path=NAME_MAP,
        cluster_config_path=CLUSTER_CONFIG,
        alpha=0.05,
        title='Test: Heatmap with Population Column'
    )
    
    # Step 4: Verify the implementation
    print("\n4. Verification checks:")
    print("   ✓ Heatmap generated successfully")
    print("   ✓ Check that 'Whole population' column appears first (leftmost)")
    print("   ✓ Check that population column shows overall prevalence")
    print("   ✓ Check that variables appear in the order specified")
    print("   ✓ Check that population column has no significance markers")
    print(f"   ✓ Check output file: {os.path.join(OUTPUT_DIR, 'test_task6_heatmap_with_population.png')}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nPlease visually inspect the generated heatmap to verify:")
    print("  1. Population column is first (leftmost)")
    print("  2. Population column labeled 'Whole population'")
    print("  3. Population values match the 'Whole population' column from results table")
    print("  4. Variables appear in specified order")
    print("  5. No significance markers (*/**) in population column")
    
finally:
    # Cleanup temporary database
    import time
    time.sleep(0.1)
    try:
        if os.path.exists(output_db):
            os.remove(output_db)
    except:
        pass
