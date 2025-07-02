# stats_helpers.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from multiprocessing import Pool, cpu_count
import itertools

# ==============================================================================
# 1. ORIGINAL FORMATTING AND BASIC STATS HELPERS
# ==============================================================================

def format_mean_sd(series):
    """Formats a series into 'mean ± SD' string, handling non-numeric data."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    mean = series.mean()
    sd = series.std()
    return f"{mean:.2f} ± {sd:.2f}"

def format_n_perc(series):
    """Formats a binary categorical series into 'N (%)' for the positive class (1)."""
    series = series.dropna()
    if series.empty:
        return "0 (0.0%)"
    n_positive = series.sum()
    n_total = len(series)
    perc = (n_positive / n_total) * 100 if n_total > 0 else 0
    return f"{int(n_positive)} ({perc:.1f}%)"

def format_median_iqr(series):
    """Formats a series into 'median [IQR]' string."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.2f} [{q1:.2f}–{q3:.2f}]"

def format_availability(series):
    """Counts non-null values and formats as 'N (%)'."""
    total_count = len(series)
    available_count = series.notna().sum()
    perc = (available_count / total_count) * 100 if total_count > 0 else 0
    return f"{available_count} ({perc:.1f}%)"

def welchs_ttest(series1, series2):
    """Performs Welch's t-test and returns a formatted p-value."""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 2 or len(s2) < 2:
        return "N/A"
    _, p_val = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
    return f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"

def categorical_pvalue(series1, series2):
    """Performs Chi-squared test and returns a formatted p-value."""
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return "N/A"
    contingency_table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return "N/A" # Not enough data for a 2x2 table
    try:
        chi2, p_val, _, _ = chi2_contingency(contingency_table)
        return f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
    except ValueError: # Catches errors with all 0s in a row/col
        return "N/A"

# ==============================================================================
# 2. BOOTSTRAPPING WORKER FUNCTIONS (FOR PARALLELIZATION)
# These must be top-level functions to be pickled by multiprocessing.
# ==============================================================================

def _bootstrap_ci_worker(data_tuple, var_type):
    """Worker for a single standard bootstrap iteration."""
    g1_data, g2_data = data_tuple
    # Resample with replacement
    g1_sample = g1_data.sample(n=len(g1_data), replace=True)
    g2_sample = g2_data.sample(n=len(g2_data), replace=True)
    
    if var_type == 'continuous':
        return g1_sample.mean() - g2_sample.mean()
    else: # categorical
        return g1_sample.mean() - g2_sample.mean() # .mean() on a 0/1 series is the proportion

def _permutation_p_worker(data_tuple, var_type):
    """Worker for a single permutation bootstrap iteration."""
    pooled_data, n1, n2 = data_tuple
    # Shuffle the pooled data and create new fake groups
    shuffled = pooled_data.sample(frac=1, replace=False)
    g1_fake = shuffled.iloc[:n1]
    g2_fake = shuffled.iloc[n1:]
    
    if var_type == 'continuous':
        return g1_fake.mean() - g2_fake.mean()
    else: # categorical
        return g1_fake.mean() - g2_fake.mean()

# ==============================================================================
# 3. MAIN BOOTSTRAP COMPARISON FUNCTION
# ==============================================================================
# In stats_helpers.py (AFTER the fix)
def run_bootstrap_comparison(series1, series2, vtype, n_iterations=5000, run_bootstrap=True):
    """
    Compares two series using standard statistical tests and optional bootstrapping.
    
    ... (your docstring) ...
    
    Args:
        ...
        run_bootstrap (bool): If True, performs bootstrapping. If False, only runs the standard test.
    """
    # --- Standard p-value calculation (this part always runs) ---
    # ... your existing code for t-test, chi-squared, etc. ...
    std_p_value = ... # Calculate this as before

    results = {
        'std_p': std_p_value,
        'boot_p': "N/A",
        'boot_ci_low': "N/A",
        'boot_ci_high': "N/A"
    }

    # --- Conditional bootstrapping ---
    if run_bootstrap:
        # ... your existing bootstrapping logic goes here ...
        # For example:
        # diffs = []
        # for _ in range(n_iterations):
        #     ...
        # boot_p = ...
        # boot_ci_low, boot_ci_high = ...
        
        # Update the results dictionary
        results['boot_p'] = boot_p
        results['boot_ci_low'] = boot_ci_low
        results['boot_ci_high'] = boot_ci_high

    return results