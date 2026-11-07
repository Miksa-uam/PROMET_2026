# stats_helpers.py

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.multitest import multipletests
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
    """Performs Welch's t-test and returns the raw p-value."""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 2 or len(s2) < 2:
        return np.nan
    _, p_val = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
    return p_val

def categorical_pvalue(series1, series2):
    """Performs Chi-squared test and returns the raw p-value."""
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    contingency_table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return np.nan # Not enough data for a 2x2 table
    try:
        _, p_val, _, _ = chi2_contingency(contingency_table)
        return p_val
    except ValueError: # Catches errors with all 0s in a row/col
        return np.nan

def format_pval(pval):
    """
    Formats a p-value to 3 decimal places or '<0.001' if p < 0.001.
    Handles NaN/None gracefully by returning 'N/A'.
    """
    if pd.isna(pval) or pval is None:
        return "N/A"
    try:
        p = float(pval)
    except (ValueError, TypeError):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def adjust_pvalues_fdr(pvalues, method="fdr_bh"):
    """
    Applies Benjamini-Hochberg False Discovery Rate (FDR) multiple testing correction.
    
    Parameters:
    -----------
    pvalues : list or np.ndarray or pd.Series
        Collection of raw nominal p-values. May contain NaN values.
    method : str
        Multiple testing correction method (default: 'fdr_bh').
        
    Returns:
    --------
    list:
        List of FDR-adjusted p-values (q-values) with NaNs preserved at original indices.
    """
    p_array = np.array(pvalues, dtype=float)
    valid_mask = ~np.isnan(p_array)
    adjusted = np.full_like(p_array, np.nan)
    if np.any(valid_mask):
        _, corrected, _, _ = multipletests(p_array[valid_mask], method=method)
        adjusted[valid_mask] = corrected
    return adjusted.tolist()

def calculate_effect_size(series1, series2, vtype="continuous"):
    """
    Calculates unified Standardized Mean Differences (SMD / Cohen's d / Austin's standardized difference)
    along with analytical 95% Confidence Intervals for continuous and binary/categorical variables.

    Statistical Methodology:
    ------------------------
    1. Continuous variables (Cohen's d):
       - Let group 1 have sample size n1, mean m1, sample variance s1^2.
       - Let group 2 have sample size n2, mean m2, sample variance s2^2.
       - Pooled standard deviation:
           s_pooled = sqrt( ((n1 - 1)*s1^2 + (n2 - 1)*s2^2) / (n1 + n2 - 2) )
       - Cohen's d:
           d = (m1 - m2) / s_pooled
       - Standard Error (Hedges & Olkin 1985 / Borenstein et al. 2009):
           SE(d) = sqrt( (n1 + n2)/(n1 * n2) + d^2 / (2 * (n1 + n2 - 2)) )
       - 95% CI: [d - 1.96 * SE(d), d + 1.96 * SE(d)]

    2. Binary/Categorical variables (Austin 2009 Standardized Difference):
       - Evaluated on binary 0/1 indicator distributions.
       - Proportions p1 and p2 in groups 1 and 2.
       - Pooled variance:
           s_p^2 = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
       - Standardized Difference:
           d = (p1 - p2) / sqrt(s_p^2)
       - Standard Error (Austin 2009 / Borenstein 2009 delta method):
           SE(d) = sqrt( (n1 + n2)/(n1 * n2) + d^2 / (2 * (n1 + n2)) )
       - 95% CI: [d - 1.96 * SE(d), d + 1.96 * SE(d)]

    3. Availability variables:
       - Transformed to binary indicator (1 = not null, 0 = null) and evaluated via Austin's SMD.

    Parameters:
    -----------
    series1 : pd.Series or array-like
        Values for group 1 (e.g., daughter / subset cohort).
    series2 : pd.Series or array-like
        Values for group 2 (e.g., mother-excluding-daughter / rest cohort).
    vtype : str
        Variable type: 'continuous', 'categorical', or 'availability'.

    Returns:
    --------
    dict:
        {
            'smd': float,
            'se': float,
            'ci_lower': float,
            'ci_upper': float
        }
    """
    if vtype == "availability":
        s1 = series1.notna().astype(float)
        s2 = series2.notna().astype(float)
        effective_vtype = "categorical"
    else:
        s1 = pd.to_numeric(series1, errors="coerce").dropna()
        s2 = pd.to_numeric(series2, errors="coerce").dropna()
        effective_vtype = vtype

    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return {"smd": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    if effective_vtype == "continuous":
        m1, m2 = float(s1.mean()), float(s2.mean())
        v1, v2 = float(s1.var(ddof=1)), float(s2.var(ddof=1))

        denom = n1 + n2 - 2
        s_pooled = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / denom)) if denom > 0 else 0.0

        if s_pooled == 0.0:
            return {"smd": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        smd = float((m1 - m2) / s_pooled)
        se = float(np.sqrt((n1 + n2) / (n1 * n2) + (smd ** 2) / (2.0 * (n1 + n2 - 2))))
        ci_lower = float(smd - 1.96 * se)
        ci_upper = float(smd + 1.96 * se)

    else: # categorical / binary
        p1 = float(s1.mean())
        p2 = float(s2.mean())

        pooled_var = float((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / 2.0)
        if pooled_var <= 0.0:
            return {"smd": 0.0, "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        smd = float((p1 - p2) / np.sqrt(pooled_var))
        se = float(np.sqrt((n1 + n2) / (n1 * n2) + (smd ** 2) / (2.0 * (n1 + n2))))
        ci_lower = float(smd - 1.96 * se)
        ci_upper = float(smd + 1.96 * se)

    return {"smd": smd, "se": se, "ci_lower": ci_lower, "ci_upper": ci_upper}

def format_effect_size(smd_dict_or_val, ci_lower=None, ci_upper=None, decimals=2):
    """
    Formats standardized effect size into '±X.XX [Y.YY, Z.ZZ]' string.
    Accepts either a dictionary from calculate_effect_size or separate numerical values.
    """
    if isinstance(smd_dict_or_val, dict):
        smd = smd_dict_or_val.get("smd")
        ci_lower = smd_dict_or_val.get("ci_lower")
        ci_upper = smd_dict_or_val.get("ci_upper")
    else:
        smd = smd_dict_or_val

    if pd.isna(smd) or pd.isna(ci_lower) or pd.isna(ci_upper):
        return "N/A"

    sign = "+" if smd > 0 else ""
    return f"{sign}{smd:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"


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
    Returns a dictionary containing raw numerical results.
    """
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()

    # --- Standard p-value calculation (always runs) ---
    if vtype == 'continuous':
        std_p_value = welchs_ttest(s1, s2)
    else: # categorical
        std_p_value = categorical_pvalue(s1, s2)

    # Calculate effect size (SMD)
    es = calculate_effect_size(s1, s2, vtype)

    results = {
        'std_p': std_p_value,
        'boot_p': np.nan,
        'boot_ci_low': np.nan,
        'boot_ci_high': np.nan,
        'obs_diff': np.nan,
        'smd': es['smd'],
        'smd_ci_low': es['ci_lower'],
        'smd_ci_high': es['ci_upper'],
        'smd_formatted': format_effect_size(es)
    }

    # --- Conditional bootstrapping ---
    if run_bootstrap:
        if s1.empty or s2.empty:
            return results # Not enough data for bootstrapping

        # Observed difference
        obs_diff = s1.mean() - s2.mean()
        results['obs_diff'] = obs_diff

        # Setup for parallel processing
        n_cpu = cpu_count()
        pool = Pool(processes=n_cpu)

        # Permutation test for p-value
        pooled_data = pd.concat([s1, s2], ignore_index=True)
        perm_args = [(pooled_data, len(s1), len(s2))] * n_iterations
        perm_diffs = pool.starmap(_permutation_p_worker, [(arg, vtype) for arg in perm_args])
        perm_diffs = np.array(perm_diffs)
        
        # Calculate p-value
        if obs_diff >= 0:
            p_val_boot = (np.sum(perm_diffs >= obs_diff) + 1) / (n_iterations + 1)
        else:
            p_val_boot = (np.sum(perm_diffs <= obs_diff) + 1) / (n_iterations + 1)
        results['boot_p'] = p_val_boot

        # Bootstrap for Confidence Interval
        ci_args = [(s1, s2)] * n_iterations
        ci_diffs = pool.starmap(_bootstrap_ci_worker, [(arg, vtype) for arg in ci_args])
        ci_diffs = np.array(ci_diffs)
        
        # Calculate CI
        ci_low, ci_high = np.nanpercentile(ci_diffs, [2.5, 97.5])
        results['boot_ci_low'] = ci_low
        results['boot_ci_high'] = ci_high
        
        pool.close()
        pool.join()

    return results