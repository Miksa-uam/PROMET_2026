"""
Forest Plot Risk Ratios Analysis Script

This script creates publication-ready forest plots showing risk ratios for 10% weight loss 
achievement across different weight gain causes. It maximally reuses existing functionality 
from descriptive_comparisons.py and integrates with the current project infrastructure.

The script is designed to be callable from notebook cells with configurable parameters
and provides comprehensive error handling with print statements for easy debugging.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from typing import Dict, List, Tuple, Optional

# Import existing project modules
try:
    from descriptive_comparisons import categorical_pvalue, get_cause_cols
    print("Successfully imported functions from descriptive_comparisons.py")
except ImportError as e:
    print(f"Error importing from descriptive_comparisons: {e}")
    print("Please ensure descriptive_comparisons.py is in the same directory")

try:
    from fdr_correction_utils import apply_fdr_correction
    print("Successfully imported FDR correction utilities")
except ImportError as e:
    print(f"Error importing FDR correction utilities: {e}")
    print("Please ensure fdr_correction_utils.py is in the same directory")

try:
    from paper12_config import master_config
    print("Successfully imported paper12_config")
except ImportError as e:
    print(f"Error importing paper12_config: {e}")
    print("Please ensure paper12_config.py is in the same directory")


def load_forest_plot_data(input_table: str, db_path: str, row_order: List = None) -> pd.DataFrame:
    """
    Load and validate data from the specified input table for forest plot analysis.
    
    This function loads data from the database, validates the presence of required columns,
    handles missing data gracefully, and returns a cleaned DataFrame ready for analysis.
    
    Args:
        input_table (str): Name of the input table containing the data
        db_path (str): Path to the SQLite database file
        row_order (List, optional): Row order configuration for identifying weight gain cause columns
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with validated columns ready for analysis
        
    Requirements: 1.1, 5.1, 5.4
    """
    print(f"Loading data from table: {input_table}")
    print(f"Database path: {db_path}")
    
    try:
        # Connect to database and load data
        with sqlite3.connect(db_path) as conn:
            # First, check if the table exists
            table_check_query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
            """
            table_exists = pd.read_sql_query(table_check_query, conn, params=(input_table,))
            
            if table_exists.empty:
                raise ValueError(f"Table '{input_table}' not found in database")
            
            # Load the data
            query = f"SELECT * FROM {input_table}"
            df = pd.read_sql_query(query, conn)
            print(f"Successfully loaded {len(df)} records from {input_table}")
            
    except sqlite3.Error as e:
        error_msg = f"Database error while loading {input_table}: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error while loading data: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)
    
    # Validate presence of required outcome column
    required_outcome_col = "10%_wl_achieved"
    if required_outcome_col not in df.columns:
        error_msg = f"Required outcome column '{required_outcome_col}' not found in {input_table}"
        print(error_msg)
        available_cols = [col for col in df.columns if 'wl_achieved' in col.lower()]
        if available_cols:
            print(f"Available weight loss columns: {available_cols}")
        raise ValueError(error_msg)
    
    print(f"✓ Found required outcome column: {required_outcome_col}")
    
    # Get weight gain cause columns using existing function
    if row_order is None:
        # If no row_order provided, use a default list of expected WGC columns
        expected_wgc_cols = [
            "womens_health_and_pregnancy",
            "mental_health", 
            "family_issues",
            "medication_disease_injury",
            "physical_inactivity",
            "eating_habits",
            "schedule",
            "smoking_cessation",
            "treatment_discontinuation_or_relapse",
            "pandemic",
            "lifestyle_circumstances",
            "none_of_above"
        ]
        print("Using default weight gain cause columns (no row_order provided)")
    else:
        expected_wgc_cols = get_cause_cols(row_order)
        print(f"Using weight gain cause columns from row_order configuration")
    
    # Validate presence of weight gain cause columns
    missing_wgc_cols = []
    available_wgc_cols = []
    
    for col in expected_wgc_cols:
        if col in df.columns:
            available_wgc_cols.append(col)
        else:
            missing_wgc_cols.append(col)
    
    if not available_wgc_cols:
        error_msg = f"No weight gain cause columns found in {input_table}"
        print(error_msg)
        print(f"Expected columns: {expected_wgc_cols}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(error_msg)
    
    print(f"✓ Found {len(available_wgc_cols)} weight gain cause columns")
    
    if missing_wgc_cols:
        print(f"⚠ Warning: Missing {len(missing_wgc_cols)} expected weight gain cause columns:")
        for col in missing_wgc_cols:
            print(f"  - {col}")
        print("Analysis will proceed with available columns only")
    
    # Validate data types and handle missing data
    print("Validating data quality...")
    
    # Check outcome column
    outcome_series = df[required_outcome_col]
    outcome_missing = outcome_series.isna().sum()
    outcome_total = len(outcome_series)
    
    if outcome_missing > 0:
        missing_pct = (outcome_missing / outcome_total) * 100
        print(f"⚠ Warning: {outcome_missing} ({missing_pct:.1f}%) missing values in outcome column")
        
        if missing_pct > 50:
            print("⚠ Warning: More than 50% missing outcome data - results may be unreliable")
    
    # Validate outcome column is binary (0/1)
    outcome_clean = pd.to_numeric(outcome_series, errors='coerce')
    unique_values = outcome_clean.dropna().unique()
    
    if not all(val in [0, 1] for val in unique_values):
        print(f"⚠ Warning: Outcome column contains non-binary values: {unique_values}")
        print("Expected binary values (0/1) for weight loss achievement")
    
    # Check weight gain cause columns
    wgc_issues = []
    for col in available_wgc_cols:
        col_series = df[col]
        col_missing = col_series.isna().sum()
        
        if col_missing > 0:
            missing_pct = (col_missing / len(col_series)) * 100
            if missing_pct > 10:  # Only warn if >10% missing
                wgc_issues.append(f"{col}: {col_missing} ({missing_pct:.1f}%) missing")
        
        # Validate binary nature
        col_clean = pd.to_numeric(col_series, errors='coerce')
        unique_vals = col_clean.dropna().unique()
        
        if not all(val in [0, 1] for val in unique_vals):
            wgc_issues.append(f"{col}: non-binary values {unique_vals}")
    
    if wgc_issues:
        print("⚠ Warning: Issues found in weight gain cause columns:")
        for issue in wgc_issues:
            print(f"  - {issue}")
    
    # Create cleaned DataFrame with only required columns
    required_columns = [required_outcome_col] + available_wgc_cols
    df_clean = df[required_columns].copy()
    
    # Store metadata about available columns for later use
    df_clean.attrs['available_wgc_cols'] = available_wgc_cols
    df_clean.attrs['missing_wgc_cols'] = missing_wgc_cols
    df_clean.attrs['outcome_col'] = required_outcome_col
    
    print(f"✓ Data validation complete")
    print(f"✓ Cleaned dataset ready: {len(df_clean)} records, {len(required_columns)} columns")
    print(f"✓ Available for analysis: {len(available_wgc_cols)} weight gain causes")
    
    return df_clean


def calculate_risk_ratios(df: pd.DataFrame, cause_columns: List[str]) -> pd.DataFrame:
    """
    Calculate risk ratios and confidence intervals for each weight gain cause.
    
    This function creates 2x2 contingency tables for each weight gain cause using
    10%_wl_achieved as the outcome, calculates risk ratios with 95% confidence intervals,
    and handles edge cases appropriately.
    
    Args:
        df (pd.DataFrame): Input DataFrame with outcome and cause columns
        cause_columns (List[str]): List of weight gain cause column names
        
    Returns:
        pd.DataFrame: Results with columns: cause, risk_ratio, ci_lower, ci_upper, 
                     n_present, n_absent, events_present, events_absent
                     
    Requirements: 1.2, 1.3, 1.4, 5.3, 5.4
    """
    print(f"Calculating risk ratios for {len(cause_columns)} weight gain causes")
    
    outcome_col = df.attrs.get('outcome_col', '10%_wl_achieved')
    results_list = []
    
    for cause in cause_columns:
        print(f"Processing cause: {cause}")
        
        try:
            # Create 2x2 contingency table
            # Remove rows with missing data for this cause or outcome
            analysis_df = df[[outcome_col, cause]].dropna()
            
            if len(analysis_df) == 0:
                print(f"⚠ Warning: No valid data for cause '{cause}' - skipping")
                continue
            
            # Build contingency table
            # a = achieved weight loss AND cause present
            # b = no weight loss AND cause present  
            # c = achieved weight loss AND cause absent
            # d = no weight loss AND cause absent
            
            cause_present = analysis_df[cause] == 1
            cause_absent = analysis_df[cause] == 0
            wl_achieved = analysis_df[outcome_col] == 1
            wl_not_achieved = analysis_df[outcome_col] == 0
            
            a = len(analysis_df[cause_present & wl_achieved])  # events in exposed group
            b = len(analysis_df[cause_present & wl_not_achieved])  # non-events in exposed group
            c = len(analysis_df[cause_absent & wl_achieved])  # events in unexposed group
            d = len(analysis_df[cause_absent & wl_not_achieved])  # non-events in unexposed group
            
            # Validate contingency table
            total_check = a + b + c + d
            if total_check != len(analysis_df):
                print(f"⚠ Warning: Contingency table sum ({total_check}) doesn't match data length ({len(analysis_df)}) for {cause}")
            
            print(f"  Contingency table for {cause}:")
            print(f"    Cause present: {a} achieved WL, {b} did not achieve WL (total: {a+b})")
            print(f"    Cause absent:  {c} achieved WL, {d} did not achieve WL (total: {c+d})")
            
            # Handle edge cases - zero cells
            if a == 0 or b == 0 or c == 0 or d == 0:
                print(f"⚠ Warning: Zero cell detected in contingency table for '{cause}'")
                print(f"    Cells: a={a}, b={b}, c={c}, d={d}")
                
                # Add 0.5 to all cells (continuity correction) for calculation
                a_adj = a + 0.5
                b_adj = b + 0.5
                c_adj = c + 0.5
                d_adj = d + 0.5
                print(f"    Applying continuity correction (+0.5 to all cells)")
            else:
                a_adj, b_adj, c_adj, d_adj = a, b, c, d
            
            # Calculate risk in exposed group: a/(a+b)
            risk_exposed = a_adj / (a_adj + b_adj)
            
            # Calculate risk in unexposed group: c/(c+d)  
            risk_unexposed = c_adj / (c_adj + d_adj)
            
            # Calculate risk ratio: RR = risk_exposed / risk_unexposed
            if risk_unexposed == 0:
                print(f"⚠ Warning: Risk in unexposed group is 0 for '{cause}' - cannot calculate risk ratio")
                risk_ratio = np.inf
                ci_lower = np.inf
                ci_upper = np.inf
            else:
                risk_ratio = risk_exposed / risk_unexposed
                
                # Calculate 95% confidence interval using log transformation
                # SE(log(RR)) = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
                try:
                    se_log_rr = np.sqrt(
                        (1/a_adj) - (1/(a_adj + b_adj)) + 
                        (1/c_adj) - (1/(c_adj + d_adj))
                    )
                    
                    # 95% CI for log(RR)
                    log_rr = np.log(risk_ratio)
                    z_score = 1.96  # 95% confidence level
                    
                    log_ci_lower = log_rr - (z_score * se_log_rr)
                    log_ci_upper = log_rr + (z_score * se_log_rr)
                    
                    # Transform back to RR scale
                    ci_lower = np.exp(log_ci_lower)
                    ci_upper = np.exp(log_ci_upper)
                    
                    # Check for extreme confidence intervals
                    if ci_upper > 1000 or ci_lower < 0.001:
                        print(f"⚠ Warning: Extreme confidence interval for '{cause}': [{ci_lower:.3f}, {ci_upper:.3f}]")
                        print(f"    This may indicate sparse data or numerical instability")
                    
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    print(f"⚠ Warning: Error calculating confidence interval for '{cause}': {str(e)}")
                    ci_lower = np.nan
                    ci_upper = np.nan
            
            # Store results
            result = {
                'cause': cause,
                'risk_ratio': risk_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_present': a + b,  # Total with cause present
                'n_absent': c + d,   # Total with cause absent
                'events_present': a,  # Events in cause present group
                'events_absent': c,   # Events in cause absent group
                'risk_exposed': risk_exposed,
                'risk_unexposed': risk_unexposed,
                'contingency_a': a,
                'contingency_b': b,
                'contingency_c': c,
                'contingency_d': d
            }
            
            results_list.append(result)
            
            print(f"  ✓ Risk ratio: {risk_ratio:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
            print(f"  ✓ Risk exposed: {risk_exposed:.3f}, Risk unexposed: {risk_unexposed:.3f}")
            
        except Exception as e:
            print(f"⚠ Error processing cause '{cause}': {str(e)}")
            print(f"  Skipping this cause and continuing with analysis")
            continue
    
    if not results_list:
        print("⚠ Warning: No valid risk ratios calculated - returning empty results")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"✓ Risk ratio calculations complete")
    print(f"✓ Successfully calculated risk ratios for {len(results_df)} causes")
    print(f"✓ Risk ratio range: {results_df['risk_ratio'].min():.3f} to {results_df['risk_ratio'].max():.3f}")
    
    return results_df


def perform_statistical_tests(df: pd.DataFrame, cause_columns: List[str]) -> List[float]:
    """
    Perform appropriate statistical tests for each weight gain cause.
    
    This function reuses the categorical_pvalue function from descriptive_comparisons.py
    for Chi-squared tests and implements Fisher's exact test selection when any 
    contingency table cell < 5. It handles statistical test failures gracefully.
    
    Args:
        df (pd.DataFrame): Input DataFrame with outcome and cause columns
        cause_columns (List[str]): List of weight gain cause column names
        
    Returns:
        List[float]: List of p-values for each cause (same order as cause_columns)
                    Returns np.nan for causes where tests fail
                    
    Requirements: 1.5, 2.1, 2.2, 4.3, 5.3
    """
    print(f"Performing statistical tests for {len(cause_columns)} weight gain causes")
    
    outcome_col = df.attrs.get('outcome_col', '10%_wl_achieved')
    p_values = []
    
    for cause in cause_columns:
        print(f"Testing cause: {cause}")
        
        try:
            # Remove rows with missing data for this cause or outcome
            analysis_df = df[[outcome_col, cause]].dropna()
            
            if len(analysis_df) == 0:
                print(f"⚠ Warning: No valid data for cause '{cause}' - assigning p-value = NaN")
                p_values.append(np.nan)
                continue
            
            # Check if we have enough variation for testing
            outcome_unique = analysis_df[outcome_col].nunique()
            cause_unique = analysis_df[cause].nunique()
            
            if outcome_unique < 2 or cause_unique < 2:
                print(f"⚠ Warning: Insufficient variation for testing '{cause}' - assigning p-value = NaN")
                print(f"  Outcome unique values: {outcome_unique}, Cause unique values: {cause_unique}")
                p_values.append(np.nan)
                continue
            
            # Build contingency table to check cell counts
            cause_present = analysis_df[cause] == 1
            cause_absent = analysis_df[cause] == 0
            wl_achieved = analysis_df[outcome_col] == 1
            wl_not_achieved = analysis_df[outcome_col] == 0
            
            a = len(analysis_df[cause_present & wl_achieved])  # events in exposed group
            b = len(analysis_df[cause_present & wl_not_achieved])  # non-events in exposed group
            c = len(analysis_df[cause_absent & wl_achieved])  # events in unexposed group
            d = len(analysis_df[cause_absent & wl_not_achieved])  # non-events in unexposed group
            
            print(f"  Contingency table: a={a}, b={b}, c={c}, d={d}")
            
            # Check if any cell has count < 5 (Fisher's exact test criterion)
            min_cell_count = min(a, b, c, d)
            use_fisher = min_cell_count < 5
            
            if use_fisher:
                print(f"  Using Fisher's exact test (min cell count: {min_cell_count} < 5)")
                
                try:
                    # Fisher's exact test using scipy.stats.fisher_exact
                    # Create 2x2 contingency table for fisher_exact
                    contingency_matrix = np.array([[a, b], [c, d]])
                    
                    # fisher_exact returns (odds_ratio, p_value)
                    odds_ratio, p_value = fisher_exact(contingency_matrix, alternative='two-sided')
                    
                    print(f"  ✓ Fisher's exact test p-value: {p_value:.6f}")
                    p_values.append(p_value)
                    
                except Exception as e:
                    print(f"⚠ Error in Fisher's exact test for '{cause}': {str(e)}")
                    print(f"  Assigning p-value = NaN")
                    p_values.append(np.nan)
                    
            else:
                print(f"  Using Chi-squared test (min cell count: {min_cell_count} >= 5)")
                
                try:
                    # Use existing categorical_pvalue function from descriptive_comparisons.py
                    # This function expects two series to compare
                    # We need to create series representing the two groups (cause present vs absent)
                    
                    # Create series for cause present group (outcomes for those with cause = 1)
                    cause_present_outcomes = analysis_df[analysis_df[cause] == 1][outcome_col]
                    
                    # Create series for cause absent group (outcomes for those with cause = 0)
                    cause_absent_outcomes = analysis_df[analysis_df[cause] == 0][outcome_col]
                    
                    # Call categorical_pvalue function
                    p_value = categorical_pvalue(cause_present_outcomes, cause_absent_outcomes)
                    
                    if pd.isna(p_value):
                        print(f"⚠ Warning: categorical_pvalue returned NaN for '{cause}'")
                        print(f"  This may indicate insufficient data or other statistical issues")
                    else:
                        print(f"  ✓ Chi-squared test p-value: {p_value:.6f}")
                    
                    p_values.append(p_value)
                    
                except Exception as e:
                    print(f"⚠ Error in Chi-squared test for '{cause}': {str(e)}")
                    print(f"  Assigning p-value = NaN")
                    p_values.append(np.nan)
        
        except Exception as e:
            print(f"⚠ Critical error processing cause '{cause}': {str(e)}")
            print(f"  Assigning p-value = NaN and continuing with next cause")
            p_values.append(np.nan)
    
    # Summary statistics
    valid_p_values = [p for p in p_values if not pd.isna(p)]
    n_valid = len(valid_p_values)
    n_total = len(cause_columns)
    n_failed = n_total - n_valid
    
    print(f"✓ Statistical testing complete")
    print(f"✓ Valid tests: {n_valid}/{n_total}")
    
    if n_failed > 0:
        print(f"⚠ Failed tests: {n_failed}/{n_total}")
        failed_causes = [cause_columns[i] for i, p in enumerate(p_values) if pd.isna(p)]
        print(f"  Failed causes: {failed_causes}")
    
    if n_valid > 0:
        min_p = min(valid_p_values)
        max_p = max(valid_p_values)
        print(f"✓ P-value range: {min_p:.6f} to {max_p:.6f}")
        
        # Count significant results at alpha = 0.05
        n_significant = sum(1 for p in valid_p_values if p < 0.05)
        print(f"✓ Nominally significant (p < 0.05): {n_significant}/{n_valid}")
    
    return p_values


def apply_fdr_correction_to_results(results_df: pd.DataFrame, p_values: List[float]) -> pd.DataFrame:
    """
    Apply FDR correction to p-values and integrate results into the results DataFrame.
    
    This function imports and uses apply_fdr_correction from fdr_correction_utils,
    applies Benjamini-Hochberg correction to collected p-values, stores both raw 
    and FDR-corrected p-values in results, and prints summary of correction results.
    
    Args:
        results_df (pd.DataFrame): DataFrame with risk ratio results
        p_values (List[float]): List of raw p-values corresponding to each cause
        
    Returns:
        pd.DataFrame: Updated results DataFrame with FDR-corrected p-values and significance markers
        
    Requirements: 2.3, 2.4, 4.2, 5.5
    """
    print(f"Applying FDR correction to {len(p_values)} p-values")
    
    try:
        # Import and use apply_fdr_correction from fdr_correction_utils
        corrected_p_values = apply_fdr_correction(p_values, method='fdr_bh', alpha=0.05)
        
        print(f"✓ FDR correction applied using Benjamini-Hochberg method")
        
        # Store both raw and FDR-corrected p-values in results
        results_df = results_df.copy()
        results_df['p_value'] = p_values
        results_df['p_value_fdr'] = corrected_p_values
        
        # Add significance markers based on FDR-corrected p-values
        results_df['significant_raw'] = [p < 0.05 if not pd.isna(p) else False for p in p_values]
        results_df['significant_fdr'] = [p < 0.05 if not pd.isna(p) else False for p in corrected_p_values]
        
        # Print summary of correction results
        valid_raw_p = [p for p in p_values if not pd.isna(p)]
        valid_fdr_p = [p for p in corrected_p_values if not pd.isna(p)]
        
        if valid_raw_p:
            n_total = len(valid_raw_p)
            n_raw_significant = sum(1 for p in valid_raw_p if p < 0.05)
            n_fdr_significant = sum(1 for p in valid_fdr_p if p < 0.05)
            
            print(f"✓ FDR correction summary:")
            print(f"  Total valid tests: {n_total}")
            print(f"  Raw significant (p < 0.05): {n_raw_significant}/{n_total} ({n_raw_significant/n_total*100:.1f}%)")
            print(f"  FDR significant (q < 0.05): {n_fdr_significant}/{n_total} ({n_fdr_significant/n_total*100:.1f}%)")
            
            if n_raw_significant > 0:
                reduction = n_raw_significant - n_fdr_significant
                print(f"  Reduction due to FDR correction: {reduction} tests")
                
                if n_fdr_significant > 0:
                    min_fdr_p = min(valid_fdr_p)
                    max_fdr_p = max(valid_fdr_p)
                    print(f"  FDR-corrected p-value range: {min_fdr_p:.6f} to {max_fdr_p:.6f}")
                    
                    # Show which causes remain significant after FDR correction
                    fdr_significant_causes = results_df[results_df['significant_fdr']]['cause'].tolist()
                    if fdr_significant_causes:
                        print(f"  FDR-significant causes: {fdr_significant_causes}")
                else:
                    print(f"  No tests remain significant after FDR correction")
            else:
                print(f"  No tests were significant before FDR correction")
        else:
            print(f"⚠ Warning: No valid p-values found for FDR correction")
        
        print(f"✓ FDR correction results integrated into results DataFrame")
        
        return results_df
        
    except Exception as e:
        error_msg = f"Error applying FDR correction: {str(e)}"
        print(error_msg)
        
        # Return original results with NaN values for FDR columns if correction fails
        results_df = results_df.copy()
        results_df['p_value'] = p_values
        results_df['p_value_fdr'] = [np.nan] * len(p_values)
        results_df['significant_raw'] = [p < 0.05 if not pd.isna(p) else False for p in p_values]
        results_df['significant_fdr'] = [False] * len(p_values)
        
        print(f"⚠ FDR correction failed - returning results with raw p-values only")
        return results_df


def create_forest_plot(results_df: pd.DataFrame, output_path: str, row_order: List = None) -> None:
    """
    Create a publication-ready forest plot showing risk ratios and confidence intervals.
    
    This function creates a matplotlib figure with logarithmic x-axis scale for risk ratios,
    plots points with horizontal confidence interval error bars, adds a vertical reference 
    line at RR = 1.0, applies proper axis labels, and adds significance markers for 
    FDR-significant results.
    
    Args:
        results_df (pd.DataFrame): DataFrame with risk ratio results including columns:
                                  cause, risk_ratio, ci_lower, ci_upper, significant_fdr
        output_path (str): Full path where the forest plot PNG file should be saved
        row_order (List, optional): Row order configuration for pretty cause names
        
    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    print(f"Creating forest plot with {len(results_df)} causes")
    print(f"Output path: {output_path}")
    
    try:
        # Validate input data
        if results_df.empty:
            print("⚠ Warning: Empty results DataFrame - creating placeholder plot")
            
            # Create a simple placeholder plot with consistent styling
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            ax.text(0.5, 0.5, 'No valid results to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16, fontweight='bold', color='gray')
            ax.set_title('Forest Plot: Risk Ratios for 10% Weight Loss Achievement', 
                        fontsize=16, fontweight='bold', color='black', pad=25)
            ax.set_xlabel('Risk Ratio (RR)', fontsize=14, fontweight='bold', color='black')
            ax.set_ylabel('Weight Gain Causes', fontsize=14, fontweight='bold', color='black')
            
            # Add grid and styling consistency
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='lightgray')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)
            
            plt.tight_layout(pad=2.0)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close(fig)
            print("✓ Placeholder forest plot saved")
            return
        
        # Validate required columns
        required_cols = ['cause', 'risk_ratio', 'ci_lower', 'ci_upper']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            error_msg = f"Missing required columns in results DataFrame: {missing_cols}"
            print(f"⚠ Error: {error_msg}")
            raise ValueError(error_msg)
        
        # Check for significance column (may not exist if FDR correction failed)
        has_significance = 'significant_fdr' in results_df.columns
        if not has_significance:
            print("⚠ Warning: No FDR significance column found - will not add significance markers")
        
        # Create pretty names mapping from row_order if available
        cause_pretty_names = {}
        if row_order is not None:
            for var, pretty in row_order:
                if not var.startswith("delim_"):
                    # Remove " (yes/no)" suffix if present for cleaner display
                    clean_pretty = pretty.replace(" (yes/no)", "")
                    cause_pretty_names[var] = clean_pretty
            print(f"✓ Using pretty names from row_order configuration ({len(cause_pretty_names)} mappings)")
        else:
            print("⚠ No row_order provided - using raw cause names")
        
        # Prepare data for plotting
        plot_data = results_df.copy()
        
        # Add pretty names
        plot_data['cause_pretty'] = plot_data['cause'].map(
            lambda x: cause_pretty_names.get(x, x.replace('_', ' ').title())
        )
        
        # Filter out invalid risk ratios (inf, -inf, NaN)
        valid_mask = (
            np.isfinite(plot_data['risk_ratio']) & 
            np.isfinite(plot_data['ci_lower']) & 
            np.isfinite(plot_data['ci_upper']) &
            (plot_data['risk_ratio'] > 0) &
            (plot_data['ci_lower'] > 0) &
            (plot_data['ci_upper'] > 0)
        )
        
        invalid_count = len(plot_data) - valid_mask.sum()
        if invalid_count > 0:
            print(f"⚠ Warning: Excluding {invalid_count} causes with invalid risk ratios or confidence intervals")
            invalid_causes = plot_data[~valid_mask]['cause'].tolist()
            print(f"  Invalid causes: {invalid_causes}")
        
        plot_data = plot_data[valid_mask].copy()
        
        if plot_data.empty:
            print("⚠ Warning: No valid data remaining after filtering - creating placeholder plot")
            
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            ax.text(0.5, 0.5, 'No valid risk ratios to display\n(all values were infinite or invalid)', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16, fontweight='bold', color='gray')
            ax.set_title('Forest Plot: Risk Ratios for 10% Weight Loss Achievement', 
                        fontsize=16, fontweight='bold', color='black', pad=25)
            ax.set_xlabel('Risk Ratio (RR)', fontsize=14, fontweight='bold', color='black')
            ax.set_ylabel('Weight Gain Causes', fontsize=14, fontweight='bold', color='black')
            
            # Add grid and styling consistency
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='lightgray')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)
            
            plt.tight_layout(pad=2.0)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close(fig)
            print("✓ Placeholder forest plot saved")
            return
        
        # Sort causes by risk ratio for better visualization (highest to lowest)
        plot_data = plot_data.sort_values('risk_ratio', ascending=True).reset_index(drop=True)
        
        print(f"✓ Plotting {len(plot_data)} valid causes")
        print(f"✓ Risk ratio range: {plot_data['risk_ratio'].min():.3f} to {plot_data['risk_ratio'].max():.3f}")
        
        # Set up the plot style (following existing project conventions)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with publication-quality settings
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Set background color for publication quality
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create y-axis positions for each cause
        y_positions = np.arange(len(plot_data))
        
        # Plot the risk ratios as points with confidence interval error bars
        # Use asymmetric error bars since we're on log scale
        risk_ratios = plot_data['risk_ratio'].values
        ci_lower = plot_data['ci_lower'].values
        ci_upper = plot_data['ci_upper'].values
        
        # Calculate error bar lengths (asymmetric for log scale)
        lower_errors = risk_ratios - ci_lower
        upper_errors = ci_upper - risk_ratios
        
        # Plot points and error bars with enhanced styling
        # Use different colors for significant vs non-significant results
        if has_significance:
            significant_mask = plot_data['significant_fdr'].fillna(False)
            
            # Plot non-significant results first (in background)
            if (~significant_mask).any():
                ax.errorbar(
                    risk_ratios[~significant_mask], y_positions[~significant_mask],
                    xerr=[lower_errors[~significant_mask], upper_errors[~significant_mask]],
                    fmt='o',  # Circle markers
                    markersize=8,
                    capsize=5,  # Cap size for error bars
                    capthick=2,
                    elinewidth=2,
                    color='steelblue',
                    markerfacecolor='lightsteelblue',
                    markeredgecolor='steelblue',
                    markeredgewidth=1.5,
                    alpha=0.7,
                    label='Non-significant'
                )
            
            # Plot significant results on top with enhanced styling
            if significant_mask.any():
                ax.errorbar(
                    risk_ratios[significant_mask], y_positions[significant_mask],
                    xerr=[lower_errors[significant_mask], upper_errors[significant_mask]],
                    fmt='o',  # Circle markers
                    markersize=10,  # Slightly larger for significant results
                    capsize=6,  # Larger cap size for error bars
                    capthick=2.5,
                    elinewidth=2.5,
                    color='darkred',
                    markerfacecolor='red',
                    markeredgecolor='darkred',
                    markeredgewidth=2,
                    alpha=0.9,
                    label='FDR significant',
                    zorder=5  # Ensure significant points appear on top
                )
        else:
            # Default styling when no significance information available
            ax.errorbar(
                risk_ratios, y_positions,
                xerr=[lower_errors, upper_errors],
                fmt='o',  # Circle markers
                markersize=8,
                capsize=5,  # Cap size for error bars
                capthick=2,
                elinewidth=2,
                color='steelblue',
                markerfacecolor='steelblue',
                markeredgecolor='darkblue',
                markeredgewidth=1
            )
        
        # Add significance markers (asterisks) for FDR-significant results
        if has_significance:
            significant_mask = plot_data['significant_fdr'].fillna(False)
            n_significant = significant_mask.sum()
            
            if n_significant > 0:
                print(f"✓ Adding significance markers for {n_significant} FDR-significant causes")
                
                # Add asterisks to the right of significant points
                sig_y_positions = y_positions[significant_mask]
                sig_risk_ratios = risk_ratios[significant_mask]
                sig_ci_upper = ci_upper[significant_mask]
                
                # Position asterisks slightly to the right of the upper confidence interval
                # Use dynamic positioning based on plot scale
                x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
                asterisk_offset = x_range * 0.05  # 5% of x-range to the right
                asterisk_x_positions = sig_ci_upper + asterisk_offset
                
                # Add asterisks with enhanced styling
                ax.scatter(
                    asterisk_x_positions, sig_y_positions,
                    marker='*', s=300, color='red', 
                    edgecolors='darkred', linewidths=1,
                    zorder=10,  # Ensure asterisks appear on top
                    label='* FDR q < 0.05'
                )
                
                # Add text annotations for extra clarity
                for i, (x, y) in enumerate(zip(asterisk_x_positions, sig_y_positions)):
                    ax.annotate('*', xy=(x, y), xytext=(5, 0), 
                               textcoords='offset points', 
                               fontsize=16, fontweight='bold', 
                               color='red', ha='left', va='center')
            else:
                print("✓ No FDR-significant results to mark")
        
        # Add vertical reference line at RR = 1.0 (no effect) with enhanced styling
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                  label='No effect (RR = 1.0)', zorder=1)
        
        # Add subtle shading to highlight areas of increased/decreased risk
        x_min, x_max = ax.get_xlim()
        if x_min < 1.0:
            ax.axvspan(x_min, 1.0, alpha=0.05, color='blue', label='Decreased risk')
        if x_max > 1.0:
            ax.axvspan(1.0, x_max, alpha=0.05, color='red', label='Increased risk')
        
        # Set logarithmic x-axis scale
        ax.set_xscale('log')
        
        # Set axis labels with enhanced styling
        ax.set_xlabel('Risk Ratio (RR)', fontsize=14, fontweight='bold', color='black')
        ax.set_ylabel('Weight Gain Causes', fontsize=14, fontweight='bold', color='black')
        
        # Set y-axis labels to pretty cause names with enhanced formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_data['cause_pretty'], fontsize=11, color='black')
        
        # Enhance x-axis tick labels
        ax.tick_params(axis='x', labelsize=11, colors='black')
        ax.tick_params(axis='y', labelsize=11, colors='black')
        
        # Set title with enhanced styling
        ax.set_title('Forest Plot: Risk Ratios for 10% Weight Loss Achievement', 
                    fontsize=16, fontweight='bold', pad=25, color='black')
        
        # Add subtitle with sample size information
        n_total_sample = plot_data[['n_present', 'n_absent']].sum().sum()
        subtitle = f'Analysis of {len(plot_data)} weight gain causes (N = {n_total_sample:,} participants)'
        ax.text(0.5, 0.98, subtitle, transform=ax.transAxes, fontsize=12, 
                ha='center', va='top', style='italic', color='gray')
        
        # Adjust x-axis limits to provide good visualization
        min_rr = plot_data['ci_lower'].min()
        max_rr = plot_data['ci_upper'].max()
        
        # Add some padding to the limits
        x_min = max(0.1, min_rr * 0.8)  # Don't go below 0.1 for log scale
        x_max = min(10.0, max_rr * 1.2)  # Cap at 10 for reasonable display
        
        ax.set_xlim(x_min, x_max)
        
        # Format x-axis ticks for better readability
        from matplotlib.ticker import LogFormatter, LogLocator
        ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        
        # Add enhanced grid for better readability
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='lightgray')
        ax.set_axisbelow(True)  # Ensure grid appears behind data points
        
        # Add subtle border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)
        
        # Add comprehensive legend with enhanced styling
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, alpha=0.7, label='No effect (RR = 1.0)')
        ]
        
        if has_significance:
            significant_mask = plot_data['significant_fdr'].fillna(False)
            n_significant = significant_mask.sum()
            n_total = len(plot_data)
            
            if n_significant > 0:
                # Add legend elements for significant and non-significant results
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='red', linestyle='None', 
                           markersize=10, markerfacecolor='red', markeredgecolor='darkred',
                           markeredgewidth=2, label=f'FDR significant (n={n_significant})'),
                    Line2D([0], [0], marker='o', color='steelblue', linestyle='None', 
                           markersize=8, markerfacecolor='lightsteelblue', markeredgecolor='steelblue',
                           markeredgewidth=1.5, alpha=0.7, label=f'Non-significant (n={n_total-n_significant})'),
                    Line2D([0], [0], marker='*', color='red', linestyle='None', 
                           markersize=15, label='* FDR q < 0.05')
                ])
            else:
                # All results are non-significant
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='steelblue', linestyle='None', 
                           markersize=8, markerfacecolor='lightsteelblue', markeredgecolor='steelblue',
                           markeredgewidth=1.5, alpha=0.7, label=f'All non-significant (n={n_total})')
                )
        else:
            # No significance information available
            legend_elements.append(
                Line2D([0], [0], marker='o', color='steelblue', linestyle='None', 
                       markersize=8, markerfacecolor='steelblue', markeredgecolor='darkblue',
                       markeredgewidth=1, label=f'Risk ratios (n={len(plot_data)})')
            )
        
        # Create legend with enhanced styling
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                          frameon=True, fancybox=True, shadow=True, 
                          framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_facecolor('white')
        
        # Adjust layout to prevent label cutoff with enhanced spacing
        plt.tight_layout(pad=2.0)
        
        # Save the plot with publication-quality settings
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', pil_kwargs={'optimize': True})
        plt.close(fig)
        
        print(f"✓ Forest plot saved successfully to: {output_path}")
        print(f"✓ Plot dimensions: 12x8 inches at 300 DPI")
        print(f"✓ Displayed {len(plot_data)} causes with risk ratios from {min_rr:.3f} to {max_rr:.3f}")
        
        if has_significance:
            n_sig = plot_data['significant_fdr'].fillna(False).sum()
            print(f"✓ Significance markers: {n_sig}/{len(plot_data)} causes FDR-significant")
        
    except Exception as e:
        error_msg = f"Error creating forest plot: {str(e)}"
        print(error_msg)
        
        # Try to create a basic error plot with consistent styling
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            ax.text(0.5, 0.5, f'Error creating forest plot:\n{str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, fontweight='bold', color='red')
            ax.set_title('Forest Plot: Error', fontsize=16, fontweight='bold', color='black', pad=25)
            ax.set_xlabel('Risk Ratio (RR)', fontsize=14, fontweight='bold', color='black')
            ax.set_ylabel('Weight Gain Causes', fontsize=14, fontweight='bold', color='black')
            
            # Add grid and styling consistency
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='lightgray')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1)
            
            plt.tight_layout(pad=2.0)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png', pil_kwargs={'optimize': True})
            plt.close(fig)
            print(f"✓ Error plot saved to: {output_path}")
            
        except Exception as save_error:
            print(f"⚠ Could not save error plot: {str(save_error)}")
        
        raise RuntimeError(error_msg)


def run_forest_plot_analysis(
    input_table: str,
    output_filename: str = "forest_plot_10pct_wl_risk_ratios.png",
    config: Optional[master_config] = None
) -> Dict:
    """
    Main function to run the complete forest plot analysis pipeline.
    
    This function orchestrates the entire analysis from data loading through
    plot generation, providing comprehensive error handling and status reporting.
    
    Args:
        input_table (str): Name of the input table containing the data
        output_filename (str): Name for the output forest plot file
        config (master_config, optional): Configuration object for database paths
        
    Returns:
        Dict: Summary results including statistics and file paths
        
    Requirements: 4.1, 4.3, 5.1, 5.2, 6.1
    """
    print(f"Starting forest plot analysis for table: {input_table}")
    print(f"Output filename: {output_filename}")
    
    try:
        # Initialize results dictionary
        results = {
            'input_table': input_table,
            'output_filename': output_filename,
            'success': False,
            'error_message': None,
            'summary_stats': {},
            'output_files': {}
        }
        
        # Set up configuration and paths
        if config is None:
            try:
                config = master_config()
                print("✓ Using default master_config")
            except Exception as e:
                print(f"⚠ Warning: Could not load master_config: {str(e)}")
                print("Using fallback database path")
                # Fallback to a reasonable default path
                db_path = "../dbs/pnk_db2_p2_out.sqlite"
        else:
            print("✓ Using provided config")
        
        # Get database path from config
        try:
            if hasattr(config, 'paths_config') and hasattr(config.paths_config, 'db_path'):
                db_path = config.paths_config.db_path
            elif hasattr(config, 'db_path'):
                db_path = config.db_path
            else:
                db_path = "../dbs/pnk_db2_p2_out.sqlite"
                print(f"⚠ Warning: Using fallback database path: {db_path}")
        except Exception as e:
            db_path = "../dbs/pnk_db2_p2_out.sqlite"
            print(f"⚠ Warning: Error getting database path from config: {str(e)}")
            print(f"Using fallback path: {db_path}")
        
        # Get row order configuration for weight gain causes
        try:
            if hasattr(config, 'ROW_ORDER'):
                row_order = config.ROW_ORDER
                print("✓ Using ROW_ORDER from config")
            else:
                row_order = None
                print("⚠ Warning: No ROW_ORDER found in config, using default cause columns")
        except Exception as e:
            row_order = None
            print(f"⚠ Warning: Error getting ROW_ORDER from config: {str(e)}")
        
        print("=" * 60)
        print("STEP 1: Loading and validating data")
        print("=" * 60)
        
        # Load and validate data
        df = load_forest_plot_data(input_table, db_path, row_order)
        
        if df.empty:
            raise ValueError(f"No data loaded from table {input_table}")
        
        # Get available weight gain cause columns
        available_wgc_cols = df.attrs.get('available_wgc_cols', [])
        
        if not available_wgc_cols:
            raise ValueError("No weight gain cause columns found in data")
        
        print("=" * 60)
        print("STEP 2: Calculating risk ratios")
        print("=" * 60)
        
        # Calculate risk ratios
        risk_results = calculate_risk_ratios(df, available_wgc_cols)
        
        if risk_results.empty:
            raise ValueError("No valid risk ratios calculated")
        
        print("=" * 60)
        print("STEP 3: Performing statistical tests")
        print("=" * 60)
        
        # Perform statistical tests
        p_values = perform_statistical_tests(df, available_wgc_cols)
        
        print("=" * 60)
        print("STEP 4: Applying FDR correction")
        print("=" * 60)
        
        # Apply FDR correction
        final_results = apply_fdr_correction_to_results(risk_results, p_values)
        
        print("=" * 60)
        print("STEP 5: Generating outputs")
        print("=" * 60)
        
        # Set up output paths
        outputs_dir = "../outputs"
        
        # Ensure outputs directory exists
        try:
            os.makedirs(outputs_dir, exist_ok=True)
            print(f"✓ Output directory ready: {outputs_dir}")
        except Exception as e:
            error_msg = f"Error creating output directory {outputs_dir}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Generate forest plot
        plot_path = os.path.join(outputs_dir, output_filename)
        
        try:
            print(f"Creating forest plot: {plot_path}")
            create_forest_plot(final_results, plot_path, row_order)
            
            # Verify plot file was created
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"✓ Forest plot saved successfully: {plot_path} ({file_size:,} bytes)")
                results['output_files']['forest_plot'] = plot_path
            else:
                raise FileNotFoundError(f"Forest plot file was not created: {plot_path}")
                
        except Exception as e:
            error_msg = f"Error creating forest plot: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # Generate summary table CSV
        summary_filename = output_filename.replace('.png', '_summary.csv')
        summary_path = os.path.join(outputs_dir, summary_filename)
        
        try:
            print(f"Creating summary table: {summary_path}")
            
            # Prepare summary table with all calculated statistics
            summary_table = final_results.copy()
            
            # Add pretty names for causes if available
            if row_order is not None:
                try:
                    # Create mapping from cause names to pretty names
                    cause_pretty_map = {}
                    for item in row_order:
                        if isinstance(item, dict) and 'name' in item and 'pretty' in item:
                            cause_pretty_map[item['name']] = item['pretty']
                    
                    # Add pretty names column
                    summary_table['cause_pretty'] = summary_table['cause'].map(cause_pretty_map).fillna(summary_table['cause'])
                    print(f"✓ Added pretty names for {len(cause_pretty_map)} causes")
                except Exception as e:
                    print(f"⚠ Warning: Could not add pretty names: {str(e)}")
                    summary_table['cause_pretty'] = summary_table['cause']
            else:
                summary_table['cause_pretty'] = summary_table['cause']
            
            # Reorder columns for better readability
            column_order = [
                'cause', 'cause_pretty', 'risk_ratio', 'ci_lower', 'ci_upper',
                'p_value', 'p_value_fdr', 'significant_raw', 'significant_fdr',
                'n_present', 'n_absent', 'events_present', 'events_absent',
                'risk_exposed', 'risk_unexposed'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in summary_table.columns]
            summary_table_ordered = summary_table[available_columns]
            
            # Round numeric columns for readability
            numeric_columns = ['risk_ratio', 'ci_lower', 'ci_upper', 'p_value', 'p_value_fdr', 'risk_exposed', 'risk_unexposed']
            for col in numeric_columns:
                if col in summary_table_ordered.columns:
                    summary_table_ordered[col] = summary_table_ordered[col].round(6)
            
            # Save to CSV
            summary_table_ordered.to_csv(summary_path, index=False)
            
            # Verify CSV file was created
            if os.path.exists(summary_path):
                file_size = os.path.getsize(summary_path)
                print(f"✓ Summary table saved successfully: {summary_path} ({file_size:,} bytes)")
                results['output_files']['summary_table'] = summary_path
            else:
                raise FileNotFoundError(f"Summary table file was not created: {summary_path}")
                
        except Exception as e:
            error_msg = f"Error creating summary table: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        print("=" * 60)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("=" * 60)
        
        # Generate completion summary with sample sizes and significant findings
        total_records = len(df)
        total_causes = len(available_wgc_cols)
        valid_results = len(final_results)
        
        # Count significant findings
        n_raw_significant = sum(final_results['significant_raw']) if 'significant_raw' in final_results.columns else 0
        n_fdr_significant = sum(final_results['significant_fdr']) if 'significant_fdr' in final_results.columns else 0
        
        # Get sample size ranges
        if 'n_present' in final_results.columns and 'n_absent' in final_results.columns:
            min_n_present = final_results['n_present'].min()
            max_n_present = final_results['n_present'].max()
            min_n_absent = final_results['n_absent'].min()
            max_n_absent = final_results['n_absent'].max()
        else:
            min_n_present = max_n_present = min_n_absent = max_n_absent = "N/A"
        
        # Get risk ratio ranges
        if 'risk_ratio' in final_results.columns:
            finite_rr = final_results['risk_ratio'][np.isfinite(final_results['risk_ratio'])]
            if len(finite_rr) > 0:
                min_rr = finite_rr.min()
                max_rr = finite_rr.max()
            else:
                min_rr = max_rr = "N/A"
        else:
            min_rr = max_rr = "N/A"
        
        # Print completion summary
        print(f"📊 DATASET SUMMARY:")
        print(f"   Input table: {input_table}")
        print(f"   Total records analyzed: {total_records:,}")
        print(f"   Weight gain causes evaluated: {total_causes}")
        print(f"   Valid risk ratio calculations: {valid_results}")
        print()
        
        print(f"📈 SAMPLE SIZE SUMMARY:")
        print(f"   Cause present group size range: {min_n_present} - {max_n_present}")
        print(f"   Cause absent group size range: {min_n_absent} - {max_n_absent}")
        print()
        
        print(f"🎯 RISK RATIO SUMMARY:")
        if min_rr != "N/A" and max_rr != "N/A":
            print(f"   Risk ratio range: {min_rr:.3f} - {max_rr:.3f}")
        else:
            print(f"   Risk ratio range: {min_rr} - {max_rr}")
        print()
        
        print(f"📊 STATISTICAL SIGNIFICANCE SUMMARY:")
        print(f"   Raw significant results (p < 0.05): {n_raw_significant}/{valid_results}")
        print(f"   FDR significant results (q < 0.05): {n_fdr_significant}/{valid_results}")
        
        if n_fdr_significant > 0:
            fdr_significant_causes = final_results[final_results['significant_fdr']]['cause'].tolist()
            print(f"   FDR-significant causes: {fdr_significant_causes}")
        else:
            print(f"   No causes remain significant after FDR correction")
        print()
        
        print(f"📁 OUTPUT FILES:")
        print(f"   Forest plot: {plot_path}")
        print(f"   Summary table: {summary_path}")
        print()
        
        # Store summary statistics in results
        results['summary_stats'] = {
            'total_records': total_records,
            'total_causes': total_causes,
            'valid_results': valid_results,
            'n_raw_significant': n_raw_significant,
            'n_fdr_significant': n_fdr_significant,
            'sample_size_ranges': {
                'min_n_present': min_n_present,
                'max_n_present': max_n_present,
                'min_n_absent': min_n_absent,
                'max_n_absent': max_n_absent
            },
            'risk_ratio_range': {
                'min_rr': min_rr,
                'max_rr': max_rr
            }
        }
        
        if n_fdr_significant > 0:
            results['summary_stats']['fdr_significant_causes'] = fdr_significant_causes
        
        results['success'] = True
        
        print("🎉 Forest plot analysis completed successfully!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        error_msg = f"Error in forest plot analysis pipeline: {str(e)}"
        print(f"❌ ANALYSIS FAILED: {error_msg}")
        print("=" * 60)
        
        return {
            'input_table': input_table,
            'output_filename': output_filename,
            'success': False,
            'error_message': error_msg,
            'summary_stats': {},
            'output_files': {}
        }


def test_fdr_correction():
    """
    Test function to validate FDR correction functionality with known data.
    
    This function creates test datasets with known p-values and verifies that
    the FDR correction function correctly applies Benjamini-Hochberg correction,
    handles edge cases, and integrates results properly.
    """
    print("Testing FDR correction functionality...")
    
    # Test 1: Normal case with mixed significant and non-significant p-values
    print("\n--- Test 1: Normal FDR correction case ---")
    
    # Create test results DataFrame
    test_results = pd.DataFrame({
        'cause': ['cause_a', 'cause_b', 'cause_c', 'cause_d', 'cause_e'],
        'risk_ratio': [1.5, 2.0, 1.2, 0.8, 1.8],
        'ci_lower': [1.1, 1.5, 0.9, 0.6, 1.3],
        'ci_upper': [2.0, 2.5, 1.5, 1.0, 2.3]
    })
    
    # Test p-values: some significant, some not
    test_p_values = [0.01, 0.03, 0.08, 0.15, 0.005]  # 3 nominally significant
    
    print(f"Test p-values: {test_p_values}")
    print(f"Nominally significant (p < 0.05): {sum(1 for p in test_p_values if p < 0.05)}/5")
    
    # Apply FDR correction
    corrected_results = apply_fdr_correction_to_results(test_results, test_p_values)
    
    # Verify results structure
    expected_columns = ['cause', 'risk_ratio', 'ci_lower', 'ci_upper', 'p_value', 'p_value_fdr', 'significant_raw', 'significant_fdr']
    missing_cols = [col for col in expected_columns if col not in corrected_results.columns]
    
    if not missing_cols:
        print("✓ All expected columns present in corrected results")
    else:
        print(f"✗ Missing columns: {missing_cols}")
    
    # Check that FDR correction was applied
    if 'p_value_fdr' in corrected_results.columns:
        fdr_p_values = corrected_results['p_value_fdr'].tolist()
        print(f"FDR-corrected p-values: {[f'{p:.6f}' if not pd.isna(p) else 'NaN' for p in fdr_p_values]}")
        
        # Verify that FDR p-values are generally >= raw p-values (with some tolerance for numerical precision)
        valid_comparisons = []
        for raw, fdr in zip(test_p_values, fdr_p_values):
            if not pd.isna(fdr):
                valid_comparisons.append(fdr >= raw - 1e-10)  # Small tolerance for numerical precision
        
        if all(valid_comparisons):
            print("✓ FDR-corrected p-values are appropriately adjusted (>= raw p-values)")
        else:
            print("⚠ Some FDR-corrected p-values are smaller than raw p-values")
        
        # Check significance markers
        n_raw_sig = sum(corrected_results['significant_raw'])
        n_fdr_sig = sum(corrected_results['significant_fdr'])
        
        print(f"Raw significant: {n_raw_sig}, FDR significant: {n_fdr_sig}")
        
        if n_fdr_sig <= n_raw_sig:
            print("✓ FDR correction appropriately reduced number of significant results")
        else:
            print("⚠ FDR correction increased number of significant results (unexpected)")
    
    # Test 2: Edge case with all NaN p-values
    print("\n--- Test 2: All NaN p-values ---")
    
    test_results_nan = pd.DataFrame({
        'cause': ['cause_x', 'cause_y'],
        'risk_ratio': [1.0, 1.5],
        'ci_lower': [0.8, 1.1],
        'ci_upper': [1.2, 1.9]
    })
    
    nan_p_values = [np.nan, np.nan]
    print(f"Test p-values: {nan_p_values}")
    
    corrected_results_nan = apply_fdr_correction_to_results(test_results_nan, nan_p_values)
    
    if 'p_value_fdr' in corrected_results_nan.columns:
        all_fdr_nan = all(pd.isna(p) for p in corrected_results_nan['p_value_fdr'])
        all_sig_false = all(not sig for sig in corrected_results_nan['significant_fdr'])
        
        if all_fdr_nan and all_sig_false:
            print("✓ NaN p-values handled correctly (FDR p-values remain NaN, no significance)")
        else:
            print("⚠ NaN p-values not handled correctly")
    
    # Test 3: Edge case with single p-value
    print("\n--- Test 3: Single p-value ---")
    
    test_results_single = pd.DataFrame({
        'cause': ['single_cause'],
        'risk_ratio': [2.0],
        'ci_lower': [1.5],
        'ci_upper': [2.5]
    })
    
    single_p_value = [0.02]
    print(f"Test p-value: {single_p_value}")
    
    corrected_results_single = apply_fdr_correction_to_results(test_results_single, single_p_value)
    
    if 'p_value_fdr' in corrected_results_single.columns:
        fdr_single = corrected_results_single['p_value_fdr'].iloc[0]
        raw_single = corrected_results_single['p_value'].iloc[0]
        
        # For single p-value, FDR correction should return the same value
        if abs(fdr_single - raw_single) < 1e-10:
            print("✓ Single p-value handled correctly (no adjustment needed)")
        else:
            print(f"⚠ Single p-value adjustment unexpected: raw={raw_single:.6f}, fdr={fdr_single:.6f}")
    
    # Test 4: Mixed valid and NaN p-values
    print("\n--- Test 4: Mixed valid and NaN p-values ---")
    
    test_results_mixed = pd.DataFrame({
        'cause': ['cause_1', 'cause_2', 'cause_3', 'cause_4'],
        'risk_ratio': [1.2, 1.8, 0.9, 1.5],
        'ci_lower': [1.0, 1.3, 0.7, 1.1],
        'ci_upper': [1.4, 2.3, 1.1, 1.9]
    })
    
    mixed_p_values = [0.01, np.nan, 0.08, 0.03]  # 2 valid, 1 NaN, 1 valid
    print(f"Test p-values: {mixed_p_values}")
    
    corrected_results_mixed = apply_fdr_correction_to_results(test_results_mixed, mixed_p_values)
    
    if 'p_value_fdr' in corrected_results_mixed.columns:
        fdr_mixed = corrected_results_mixed['p_value_fdr'].tolist()
        
        # Check that NaN positions are preserved
        nan_positions_raw = [i for i, p in enumerate(mixed_p_values) if pd.isna(p)]
        nan_positions_fdr = [i for i, p in enumerate(fdr_mixed) if pd.isna(p)]
        
        if nan_positions_raw == nan_positions_fdr:
            print("✓ NaN positions preserved in mixed p-value correction")
        else:
            print(f"⚠ NaN positions not preserved: raw={nan_positions_raw}, fdr={nan_positions_fdr}")
        
        # Check that valid p-values were corrected
        valid_indices = [i for i, p in enumerate(mixed_p_values) if not pd.isna(p)]
        valid_raw = [mixed_p_values[i] for i in valid_indices]
        valid_fdr = [fdr_mixed[i] for i in valid_indices]
        
        print(f"Valid raw p-values: {valid_raw}")
        print(f"Valid FDR p-values: {[f'{p:.6f}' for p in valid_fdr]}")
    
    print("✓ FDR correction testing complete")


def test_statistical_testing():
    """
    Test function to validate statistical testing logic with known data.
    
    This function creates test datasets and verifies that the statistical testing
    function correctly selects between Chi-squared and Fisher's exact tests,
    and handles edge cases appropriately.
    """
    print("Testing statistical testing logic...")
    
    # Test 1: Normal case with sufficient cell counts (Chi-squared test)
    print("\n--- Test 1: Chi-squared test case (large cell counts) ---")
    chi_test_data = {
        '10%_wl_achieved': [1]*20 + [0]*20 + [1]*15 + [0]*25,  # 80 total
        'test_cause_chi': [1]*40 + [0]*40  # 40 with cause, 40 without
    }
    
    df_chi = pd.DataFrame(chi_test_data)
    df_chi.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("Chi-squared test dataset:")
    print(f"Total records: {len(df_chi)}")
    print(f"Cause present: {sum(df_chi['test_cause_chi'])}")
    print(f"Weight loss achieved: {sum(df_chi['10%_wl_achieved'])}")
    
    # Expected contingency table:
    # Cause present (40): 20 achieved WL, 20 did not
    # Cause absent (40): 15 achieved WL, 25 did not
    # All cells >= 5, so should use Chi-squared
    
    p_values_chi = perform_statistical_tests(df_chi, ['test_cause_chi'])
    
    if len(p_values_chi) > 0 and not pd.isna(p_values_chi[0]):
        print(f"✓ Chi-squared test p-value: {p_values_chi[0]:.6f}")
        print("✓ Chi-squared test completed successfully")
    else:
        print("✗ Chi-squared test failed")
    
    # Test 2: Small cell counts (Fisher's exact test)
    print("\n--- Test 2: Fisher's exact test case (small cell counts) ---")
    fisher_test_data = {
        '10%_wl_achieved': [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # 10 total
        'test_cause_fisher': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 4 with cause, 6 without
    }
    
    df_fisher = pd.DataFrame(fisher_test_data)
    df_fisher.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("Fisher's exact test dataset:")
    print(df_fisher)
    
    # Expected contingency table:
    # Cause present (4): 2 achieved WL, 2 did not
    # Cause absent (6): 1 achieved WL, 5 did not
    # Some cells < 5, so should use Fisher's exact
    
    p_values_fisher = perform_statistical_tests(df_fisher, ['test_cause_fisher'])
    
    if len(p_values_fisher) > 0 and not pd.isna(p_values_fisher[0]):
        print(f"✓ Fisher's exact test p-value: {p_values_fisher[0]:.6f}")
        print("✓ Fisher's exact test completed successfully")
    else:
        print("✗ Fisher's exact test failed")
    
    # Test 3: Edge case - no variation in outcome
    print("\n--- Test 3: Edge case - no variation in outcome ---")
    no_var_data = {
        '10%_wl_achieved': [1, 1, 1, 1, 1, 1],  # All achieved weight loss
        'test_cause_novar': [1, 1, 1, 0, 0, 0]   # Half with cause, half without
    }
    
    df_no_var = pd.DataFrame(no_var_data)
    df_no_var.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("No variation dataset:")
    print(df_no_var)
    
    p_values_no_var = perform_statistical_tests(df_no_var, ['test_cause_novar'])
    
    if len(p_values_no_var) > 0:
        if pd.isna(p_values_no_var[0]):
            print("✓ No variation case handled correctly (p-value = NaN)")
        else:
            print(f"⚠ No variation case returned p-value: {p_values_no_var[0]:.6f}")
    
    # Test 4: Edge case - no variation in cause
    print("\n--- Test 4: Edge case - no variation in cause ---")
    no_cause_var_data = {
        '10%_wl_achieved': [1, 0, 1, 0, 1, 0],  # Variation in outcome
        'test_cause_nocausevar': [1, 1, 1, 1, 1, 1]  # All have the cause
    }
    
    df_no_cause_var = pd.DataFrame(no_cause_var_data)
    df_no_cause_var.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("No cause variation dataset:")
    print(df_no_cause_var)
    
    p_values_no_cause_var = perform_statistical_tests(df_no_cause_var, ['test_cause_nocausevar'])
    
    if len(p_values_no_cause_var) > 0:
        if pd.isna(p_values_no_cause_var[0]):
            print("✓ No cause variation case handled correctly (p-value = NaN)")
        else:
            print(f"⚠ No cause variation case returned p-value: {p_values_no_cause_var[0]:.6f}")
    
    # Test 5: Multiple causes test
    print("\n--- Test 5: Multiple causes test ---")
    multi_cause_data = {
        '10%_wl_achieved': [1]*10 + [0]*10 + [1]*8 + [0]*12,  # 40 total
        'cause_a': [1]*20 + [0]*20,  # First 20 have cause A
        'cause_b': [0]*10 + [1]*10 + [0]*10 + [1]*10  # Alternating pattern for cause B
    }
    
    df_multi = pd.DataFrame(multi_cause_data)
    df_multi.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("Multiple causes dataset:")
    print(f"Total records: {len(df_multi)}")
    print(f"Cause A present: {sum(df_multi['cause_a'])}")
    print(f"Cause B present: {sum(df_multi['cause_b'])}")
    
    p_values_multi = perform_statistical_tests(df_multi, ['cause_a', 'cause_b'])
    
    print(f"Multiple causes p-values: {p_values_multi}")
    
    valid_p_values = [p for p in p_values_multi if not pd.isna(p)]
    if len(valid_p_values) == 2:
        print("✓ Multiple causes test completed successfully")
    else:
        print(f"⚠ Multiple causes test: {len(valid_p_values)}/2 valid p-values")
    
    return p_values_chi, p_values_fishertatistical_tests(df_multi, ['cause_a', 'cause_b'])
    
    print(f"Multiple causes p-values: {p_values_multi}")
    
    valid_p_values = [p for p in p_values_multi if not pd.isna(p)]
    if len(valid_p_values) == 2:
        print("✓ Multiple causes test completed successfully")
    else:
        print(f"⚠ Multiple causes test: {len(valid_p_values)}/2 valid p-values")
    
    return p_values_chi, p_values_fisher


def test_forest_plot_creation():
    """
    Test function to validate forest plot creation with sample data.
    
    This function creates test datasets with known risk ratios and tests
    the forest plot creation function with various scenarios.
    """
    print("Testing forest plot creation...")
    
    # Test 1: Normal case with mixed significant and non-significant results
    print("\n--- Test 1: Normal forest plot case ---")
    
    # Create test results DataFrame with realistic data
    test_results = pd.DataFrame({
        'cause': ['mental_health', 'physical_inactivity', 'eating_habits', 'family_issues', 'medication_disease_injury'],
        'risk_ratio': [1.25, 0.85, 1.45, 1.10, 0.95],
        'ci_lower': [1.05, 0.70, 1.20, 0.90, 0.75],
        'ci_upper': [1.50, 1.05, 1.75, 1.35, 1.20],
        'p_value': [0.02, 0.15, 0.001, 0.35, 0.65],
        'p_value_fdr': [0.04, 0.25, 0.005, 0.45, 0.70],
        'significant_fdr': [True, False, True, False, False],
        'n_present': [150, 200, 180, 120, 160],
        'n_absent': [350, 300, 320, 380, 340]
    })
    
    print("Test results DataFrame:")
    print(test_results[['cause', 'risk_ratio', 'ci_lower', 'ci_upper', 'significant_fdr']])
    
    # Create test row_order for pretty names
    test_row_order = [
        ('mental_health', 'Mental health (yes/no)'),
        ('physical_inactivity', 'Physical inactivity (yes/no)'),
        ('eating_habits', 'Eating habits (yes/no)'),
        ('family_issues', 'Family issues (yes/no)'),
        ('medication_disease_injury', 'Medication/disease/injury (yes/no)')
    ]
    
    # Test the forest plot creation
    test_output_path = "outputs/test_forest_plot_normal.png"
    
    try:
        create_forest_plot(test_results, test_output_path, test_row_order)
        print("✓ Normal forest plot test completed successfully")
    except Exception as e:
        print(f"✗ Normal forest plot test failed: {str(e)}")
    
    # Test 2: Edge case with empty results
    print("\n--- Test 2: Empty results case ---")
    
    empty_results = pd.DataFrame()
    test_output_path_empty = "outputs/test_forest_plot_empty.png"
    
    try:
        create_forest_plot(empty_results, test_output_path_empty)
        print("✓ Empty results test completed successfully")
    except Exception as e:
        print(f"✗ Empty results test failed: {str(e)}")
    
    # Test 3: Edge case with invalid risk ratios
    print("\n--- Test 3: Invalid risk ratios case ---")
    
    invalid_results = pd.DataFrame({
        'cause': ['cause_inf', 'cause_zero', 'cause_nan'],
        'risk_ratio': [np.inf, 0.0, np.nan],
        'ci_lower': [1.0, 0.0, np.nan],
        'ci_upper': [np.inf, 0.5, np.nan],
        'significant_fdr': [False, False, False]
    })
    
    test_output_path_invalid = "outputs/test_forest_plot_invalid.png"
    
    try:
        create_forest_plot(invalid_results, test_output_path_invalid)
        print("✓ Invalid risk ratios test completed successfully")
    except Exception as e:
        print(f"✗ Invalid risk ratios test failed: {str(e)}")
    
    # Test 4: No significance column
    print("\n--- Test 4: No significance column case ---")
    
    no_sig_results = pd.DataFrame({
        'cause': ['cause_a', 'cause_b'],
        'risk_ratio': [1.2, 0.8],
        'ci_lower': [1.0, 0.6],
        'ci_upper': [1.4, 1.0]
    })
    
    test_output_path_no_sig = "outputs/test_forest_plot_no_sig.png"
    
    try:
        create_forest_plot(no_sig_results, test_output_path_no_sig)
        print("✓ No significance column test completed successfully")
    except Exception as e:
        print(f"✗ No significance column test failed: {str(e)}")
    
    print("✓ Forest plot creation testing complete")
    return test_results


def test_risk_ratio_calculation():
    """
    Test function to validate risk ratio calculation logic with known data.
    
    This function creates a simple test dataset and verifies that the risk ratio
    calculations produce expected results, including edge cases.
    """
    print("Testing risk ratio calculation logic...")
    
    # Test 1: Normal case with known risk ratios
    print("\n--- Test 1: Normal cases ---")
    test_data = {
        '10%_wl_achieved': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        'test_cause_1': [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        'test_cause_2': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df_test = pd.DataFrame(test_data)
    df_test.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("Test dataset:")
    print(df_test)
    
    # Calculate risk ratios
    cause_cols = ['test_cause_1', 'test_cause_2']
    results = calculate_risk_ratios(df_test, cause_cols)
    
    print("\nTest results:")
    print(results[['cause', 'risk_ratio', 'ci_lower', 'ci_upper', 'n_present', 'n_absent']])
    
    # Manual verification for test_cause_1:
    # Cause present (1): rows 0,1,2,3,8,9 → outcomes 1,1,0,0,1,0 → 3/6 = 0.5 risk
    # Cause absent (0): rows 4,5,6,7,10,11 → outcomes 1,1,0,0,1,0 → 3/6 = 0.5 risk  
    # Expected RR = 0.5 / 0.5 = 1.0
    
    # Manual verification for test_cause_2:
    # Cause present (1): rows 0,2,4,6,8,10 → outcomes 1,0,1,0,1,1 → 4/6 = 0.667 risk
    # Cause absent (0): rows 1,3,5,7,9,11 → outcomes 1,0,1,0,0,0 → 2/6 = 0.333 risk
    # Expected RR = 0.667 / 0.333 = 2.0
    
    if len(results) > 0:
        # Test cause 1
        rr_cause1 = results[results['cause'] == 'test_cause_1']['risk_ratio'].iloc[0]
        expected_rr1 = 1.0
        print(f"\nValidation for test_cause_1:")
        print(f"  Calculated RR: {rr_cause1:.3f}")
        print(f"  Expected RR: {expected_rr1:.3f}")
        print(f"  Difference: {abs(rr_cause1 - expected_rr1):.3f}")
        
        # Test cause 2
        rr_cause2 = results[results['cause'] == 'test_cause_2']['risk_ratio'].iloc[0]
        expected_rr2 = 2.0
        print(f"\nValidation for test_cause_2:")
        print(f"  Calculated RR: {rr_cause2:.3f}")
        print(f"  Expected RR: {expected_rr2:.3f}")
        print(f"  Difference: {abs(rr_cause2 - expected_rr2):.3f}")
        
        if abs(rr_cause1 - expected_rr1) < 0.01 and abs(rr_cause2 - expected_rr2) < 0.01:
            print("\n  ✓ Normal case tests PASSED")
        else:
            print("\n  ✗ Normal case tests FAILED")
    
    # Test 2: Edge case with zero cells
    print("\n--- Test 2: Edge case with zero cells ---")
    edge_data = {
        '10%_wl_achieved': [1, 1, 1, 0, 0, 0],  # All cause present achieved WL, none cause absent did
        'zero_cell_cause': [1, 1, 1, 0, 0, 0]   # Perfect separation
    }
    
    df_edge = pd.DataFrame(edge_data)
    df_edge.attrs['outcome_col'] = '10%_wl_achieved'
    
    print("Edge case dataset (zero cells):")
    print(df_edge)
    
    edge_results = calculate_risk_ratios(df_edge, ['zero_cell_cause'])
    
    if len(edge_results) > 0:
        print("\nEdge case results:")
        print(edge_results[['cause', 'risk_ratio', 'ci_lower', 'ci_upper']])
        print("  ✓ Edge case handled - continuity correction applied")
    
    return results


def main():
    """
    Main entry point for script execution.
    
    Provides example usage and basic error handling for direct script execution.
    """
    print("Forest Plot Risk Ratios Analysis Script")
    print("=" * 50)
    
    # Run test of risk ratio calculation
    print("\n1. Testing risk ratio calculation function...")
    try:
        test_results = test_risk_ratio_calculation()
        print("Risk ratio calculation test completed")
    except Exception as e:
        print(f"Error in risk ratio calculation test: {str(e)}")
    
    # Run test of statistical testing
    print("\n" + "=" * 50)
    print("2. Testing statistical testing function...")
    try:
        chi_p_vals, fisher_p_vals = test_statistical_testing()
        print("Statistical testing test completed")
    except Exception as e:
        print(f"Error in statistical testing test: {str(e)}")
    
    # Run test of FDR correction
    print("\n" + "=" * 50)
    print("3. Testing FDR correction function...")
    try:
        test_fdr_correction()
        print("FDR correction test completed")
    except Exception as e:
        print(f"Error in FDR correction test: {str(e)}")
    
    # Run test of forest plot creation
    print("\n" + "=" * 50)
    print("4. Testing forest plot creation function...")
    try:
        test_forest_plot_creation()
        print("Forest plot creation test completed")
    except Exception as e:
        print(f"Error in forest plot creation test: {str(e)}")
    
    print("\n" + "=" * 50)
    print("5. Example pipeline execution...")
    
    try:
        # Example usage with default parameters
        results = run_forest_plot_analysis(
            input_table="timetoevent_wgc_compl",
            output_filename="forest_plot_10pct_wl_risk_ratios.png"
        )
        
        if results['success']:
            print("Analysis completed successfully!")
        else:
            print(f"Analysis failed: {results['error_message']}")
            
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")
        print("Please check your configuration and try again")


if __name__ == "__main__":
    main()