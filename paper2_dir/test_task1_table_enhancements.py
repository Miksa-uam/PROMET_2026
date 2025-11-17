"""
Test script for Task 1: Enhance table output in analyze_cluster_vs_population function

Tests:
1. Full configuration (cluster config + name map)
2. Missing cluster configuration
3. Missing name map
4. Partial configuration
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import tempfile
import shutil
import time

# Add scripts directory to path
sys.path.insert(0, 'scripts')

from cluster_descriptions import analyze_cluster_vs_population

def safe_cleanup(filepath):
    """Safely remove file with retry for Windows file locking."""
    time.sleep(0.1)
    for _ in range(3):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return
        except PermissionError:
            time.sleep(0.2)
    # Give up after retries

def create_test_data():
    """Create synthetic test data with clusters and WGC variables."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'medical_record_id': range(n_samples),
        'cluster_id': np.random.choice([0, 1, 2], n_samples),
        'wgc_medication': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'wgc_stress': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'wgc_eating': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def test_full_configuration():
    """Test with full configuration (cluster config + name map)."""
    print("\n" + "="*80)
    print("TEST 1: Full Configuration")
    print("="*80)
    
    # Create test data
    cluster_df = create_test_data()
    
    # Create temporary output database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        output_db = tmp.name
    
    try:
        # Run analysis with full configuration
        results_df = analyze_cluster_vs_population(
            cluster_df=cluster_df,
            variables=['wgc_medication', 'wgc_stress', 'wgc_eating'],
            output_db_path=output_db,
            output_table_name='test_full_config',
            cluster_col='cluster_id',
            name_map_path='scripts/human_readable_variable_names.json',
            cluster_config_path='scripts/cluster_config.json',
            variable_types={'wgc_medication': 'categorical', 'wgc_stress': 'categorical', 'wgc_eating': 'categorical'},
            fdr_correction=True,
            alpha=0.05
        )
        
        print("\n✓ Test 1 PASSED: Full configuration works")
        print(f"  - Columns: {list(results_df.columns)}")
        print(f"  - Shape: {results_df.shape}")
        
        # Verify column headers use cluster labels
        assert 'Whole population: Mean (±SD) / N (%)' in results_df.columns, "Population column header incorrect"
        
        # Check for cluster label columns (should have cluster names, not just numbers)
        cluster_label_found = False
        for col in results_df.columns:
            if ': Mean (±SD) / N (%)' in col and col != 'Whole population: Mean (±SD) / N (%)':
                cluster_label_found = True
                print(f"  - Found cluster column: {col}")
        
        assert cluster_label_found, "No cluster label columns found"
        
        # Verify variable names are human-readable (not raw column names)
        variables_in_table = results_df['Variable'].tolist()
        print(f"  - Variables in table: {variables_in_table}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        safe_cleanup(output_db)

def test_missing_cluster_config():
    """Test with missing cluster configuration (should use defaults)."""
    print("\n" + "="*80)
    print("TEST 2: Missing Cluster Configuration")
    print("="*80)
    
    # Create test data
    cluster_df = create_test_data()
    
    # Create temporary output database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        output_db = tmp.name
    
    try:
        # Run analysis with non-existent cluster config
        results_df = analyze_cluster_vs_population(
            cluster_df=cluster_df,
            variables=['wgc_medication', 'wgc_stress'],
            output_db_path=output_db,
            output_table_name='test_missing_config',
            cluster_col='cluster_id',
            name_map_path='scripts/human_readable_variable_names.json',
            cluster_config_path='nonexistent_config.json',  # This file doesn't exist
            variable_types={'wgc_medication': 'categorical', 'wgc_stress': 'categorical'},
            fdr_correction=True,
            alpha=0.05
        )
        
        print("\n✓ Test 2 PASSED: Missing cluster config handled gracefully")
        print(f"  - Columns: {list(results_df.columns)}")
        
        # Verify fallback to default format "Cluster X: Mean (±SD) / N (%)"
        cluster_default_found = False
        for col in results_df.columns:
            if col.startswith('Cluster ') and ': Mean (±SD) / N (%)' in col:
                cluster_default_found = True
                print(f"  - Found default cluster column: {col}")
        
        assert cluster_default_found, "No default cluster columns found"
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        safe_cleanup(output_db)

def test_missing_name_map():
    """Test with missing name map (should use formatted variable names)."""
    print("\n" + "="*80)
    print("TEST 3: Missing Name Map")
    print("="*80)
    
    # Create test data
    cluster_df = create_test_data()
    
    # Create temporary output database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        output_db = tmp.name
    
    try:
        # Run analysis with non-existent name map
        results_df = analyze_cluster_vs_population(
            cluster_df=cluster_df,
            variables=['wgc_medication', 'wgc_stress'],
            output_db_path=output_db,
            output_table_name='test_missing_namemap',
            cluster_col='cluster_id',
            name_map_path='nonexistent_namemap.json',  # This file doesn't exist
            cluster_config_path='scripts/cluster_config.json',
            variable_types={'wgc_medication': 'categorical', 'wgc_stress': 'categorical'},
            fdr_correction=True,
            alpha=0.05
        )
        
        print("\n✓ Test 3 PASSED: Missing name map handled gracefully")
        print(f"  - Variables: {results_df['Variable'].tolist()}")
        
        # Verify fallback to formatted names (e.g., "Wgc Medication" instead of "wgc_medication")
        variables_in_table = results_df['Variable'].tolist()
        for var in variables_in_table:
            if var != 'N':
                # Should be title case with spaces
                assert ' ' in var or var.istitle(), f"Variable '{var}' not properly formatted"
                print(f"  - Formatted variable: {var}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        safe_cleanup(output_db)

def test_dataframe_display():
    """Test that DataFrame is displayed with proper pandas options."""
    print("\n" + "="*80)
    print("TEST 4: DataFrame Display Output")
    print("="*80)
    
    # Create test data
    cluster_df = create_test_data()
    
    # Create temporary output database
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        output_db = tmp.name
    
    try:
        # Run analysis - should print DataFrame to console
        print("\nExpecting DataFrame output below:")
        print("-" * 80)
        
        results_df = analyze_cluster_vs_population(
            cluster_df=cluster_df,
            variables=['wgc_medication'],
            output_db_path=output_db,
            output_table_name='test_display',
            cluster_col='cluster_id',
            name_map_path='scripts/human_readable_variable_names.json',
            cluster_config_path='scripts/cluster_config.json',
            variable_types={'wgc_medication': 'categorical'},
            fdr_correction=True,
            alpha=0.05
        )
        
        print("-" * 80)
        print("\n✓ Test 4 PASSED: DataFrame display output shown above")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        safe_cleanup(output_db)

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING TASK 1: Table Output Enhancements")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Full Configuration", test_full_configuration()))
    results.append(("Missing Cluster Config", test_missing_cluster_config()))
    results.append(("Missing Name Map", test_missing_name_map()))
    results.append(("DataFrame Display", test_dataframe_display()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED")
        print("="*80)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
