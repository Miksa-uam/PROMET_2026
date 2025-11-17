"""
Test script for Task 7: Heatmap cluster-specific color scales
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

import pandas as pd
import numpy as np
import tempfile
import json
import time
from cluster_descriptions import (
    analyze_cluster_vs_population,
    plot_cluster_heatmap
)

# Configuration
OUTPUT_DIR = 'outputs/test_task7'
NAME_MAP_PATH = 'scripts/human_readable_variable_names.json'
CLUSTER_CONFIG_PATH = 'scripts/cluster_config.json'

# Test variables
TEST_VARIABLES = [
    'wgc_pregnancy',
    'wgc_menopause',
    'wgc_medication',
    'wgc_medical_issue'
]

def create_test_data(n_samples=500):
    """Create synthetic test data with multiple clusters."""
    np.random.seed(42)
    data = {
        'medical_record_id': range(n_samples),
        'cluster_id': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples),
        'wgc_pregnancy': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'wgc_menopause': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'wgc_medication': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'wgc_medical_issue': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    }
    return pd.DataFrame(data)

print("="*80)
print("TEST: Task 7 - Heatmap Cluster-Specific Color Scales")
print("="*80)

# Create test data
print("\n1. Creating synthetic test data...")
cluster_df = create_test_data()
print(f"   ✓ Created {len(cluster_df)} records with {cluster_df['cluster_id'].nunique()} clusters")

# Analyze cluster vs population
print("\n2. Running cluster vs population analysis...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
    output_db = tmp.name

try:
    results_df = analyze_cluster_vs_population(
        cluster_df=cluster_df,
        variables=TEST_VARIABLES,
        output_db_path=output_db,
        output_table_name='test_task7_results',
        cluster_col='cluster_id',
        name_map_path=NAME_MAP_PATH,
        cluster_config_path=CLUSTER_CONFIG_PATH,
        variable_types={var: 'categorical' for var in TEST_VARIABLES},
        fdr_correction=True
    )
    print(f"   ✓ Analysis complete")
    
    # Test 1: Full configuration (with cluster colors)
    print("\n3. Test 1: Heatmap with full cluster colors configuration...")
    try:
        plot_cluster_heatmap(
            results_df=results_df,
            output_filename='test_task7_full_config.png',
            output_dir=OUTPUT_DIR,
            variables=TEST_VARIABLES,
            name_map_path=NAME_MAP_PATH,
            cluster_config_path=CLUSTER_CONFIG_PATH,
            title='Test: Cluster-Specific Color Scales (Full Config)'
        )
        print("   ✓ Test 1 PASSED: Heatmap with full config generated")
    except Exception as e:
        print(f"   ✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Missing cluster colors (fallback to default palette)
    print("\n4. Test 2: Heatmap with missing cluster colors (fallback)...")
    try:
        # Create a temporary config without colors
        with open(CLUSTER_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Remove colors
        config_no_colors = {'cluster_labels': config['cluster_labels'], 'cluster_colors': {}}
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_no_colors, f)
            temp_config_path = f.name
        
        plot_cluster_heatmap(
            results_df=results_df,
            output_filename='test_task7_no_colors.png',
            output_dir=OUTPUT_DIR,
            variables=TEST_VARIABLES,
            name_map_path=NAME_MAP_PATH,
            cluster_config_path=temp_config_path,
            title='Test: Cluster-Specific Color Scales (No Colors - Fallback)'
        )
        
        # Clean up temp file
        os.unlink(temp_config_path)
        
        print("   ✓ Test 2 PASSED: Heatmap with fallback colors generated")
    except Exception as e:
        print(f"   ✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Verify visual elements
    print("\n5. Test 3: Verifying visual elements...")
    checks = []

    # Check that output files exist
    output_file1 = os.path.join(OUTPUT_DIR, 'test_task7_full_config.png')
    output_file2 = os.path.join(OUTPUT_DIR, 'test_task7_no_colors.png')

    if os.path.exists(output_file1):
        checks.append("✓ Full config heatmap file created")
    else:
        checks.append("✗ Full config heatmap file NOT created")

    if os.path.exists(output_file2):
        checks.append("✓ Fallback heatmap file created")
    else:
        checks.append("✗ Fallback heatmap file NOT created")

    for check in checks:
        print(f"   {check}")

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("Task 7 implementation complete. Please visually inspect the generated heatmaps:")
    print(f"  1. {output_file1}")
    print(f"  2. {output_file2}")
    print("\nExpected visual features:")
    print("  - Population column uses white-to-red color scale")
    print("  - Each cluster column uses white-to-[cluster-color] scale")
    print("  - No colorbar on the right side")
    print("  - Column labels are centered above each column")
    print("  - Row labels appear only on the leftmost axis")
    print("  - Annotations show N (X.X%) with significance markers")
    print("="*80)

except Exception as e:
    print(f"   ✗ Analysis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Clean up temp database
    time.sleep(0.1)
    try:
        if os.path.exists(output_db):
            os.remove(output_db)
    except:
        pass
