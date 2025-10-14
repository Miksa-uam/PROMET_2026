"""
Comprehensive Descriptive Visualizations Pipeline

This script creates a "Swiss Army knife" for exploratory and descriptive visualizations.
It generates both risk ratio and risk difference forest plots for multiple outcomes 
(10% weight loss achievement and 60-day dropout) across different weight gain causes.

The pipeline maximally reuses existing functionality from descriptive_comparisons.py
and provides a clean, organized output structure with simplified notebook interface.
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
    Load and validate data from the specified input table for dual-outcome analysis.
    
    This function loads data from the database, validates the presence of required columns
    for both 10%_wl_achieved and 60d_dropout outcomes, handles missing data gracefully,
    and returns a cleaned DataFrame ready for dual-outcome analysis.
    
    Args:
        input_table (str): Name of the input table containing the data
        db_path (str): Path to the SQLite database file
        row_order (List, optional): Row order configuration for identifying weight gain cause columns
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with validated columns ready for dual-outcome analysis
        
    Requirements: 4.1, 5.1, 5.2
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
    
    # Validate presence of required outcome columns for dual-outcome analysis
    required_outcome_cols = ["10%_wl_achieved", "60d_dropout"]
    missing_outcome_cols = []
    
    for outcome_col in required_outcome_cols:
        if outcome_col not in df.columns:
            missing_outcome_cols.append(outcome_col)
    
    if missing_outcome_cols:
        error_msg = f"Required outcome columns not found in {input_table}: {missing_outcome_cols}"
        print(error_msg)
        available_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['wl_achieved', 'dropout'])]
        if available_cols:
            print(f"Available outcome-related columns: {available_cols}")
        raise ValueError(error_msg)
    
    print(f"✓ Found required outcome columns: {required_outcome_cols}")
    
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
    
    # Validate data quality for both outcomes
    print("Validating data quality for dual outcomes...")
    
    for outcome_col in required_outcome_cols:
        outcome_series = df[outcome_col]
        outcome_missing = outcome_series.isna().sum()
        outcome_total = len(outcome_series)
        
        if outcome_missing > 0:
            missing_pct = (outcome_missing / outcome_total) * 100
            print(f"⚠ Warning: {outcome_missing} ({missing_pct:.1f}%) missing values in {outcome_col}")
            
            if missing_pct > 50:
                print(f"⚠ Warning: More than 50% missing data in {outcome_col} - results may be unreliable")
        
        # Validate outcome column is binary (0/1)
        outcome_clean = pd.to_numeric(outcome_series, errors='coerce')
        unique_values = outcome_clean.dropna().unique()
        
        if not all(val in [0, 1] for val in unique_values):
            print(f"⚠ Warning: {outcome_col} contains non-binary values: {unique_values}")
            print("Expected binary values (0/1)")
    
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
    
    # Create cleaned DataFrame with required columns for dual-outcome analysis
    required_columns = required_outcome_cols + available_wgc_cols
    df_clean = df[required_columns].copy()
    
    # Store metadata about available columns for later use
    df_clean.attrs['available_wgc_cols'] = available_wgc_cols
    df_clean.attrs['missing_wgc_cols'] = missing_wgc_cols
    df_clean.attrs['outcome_cols'] = required_outcome_cols
    
    print(f"✓ Data validation complete")
    print(f"✓ Cleaned dataset ready: {len(df_clean)} records, {len(required_columns)} columns")
    print(f"✓ Available for analysis: {len(available_wgc_cols)} weight gain causes, {len(required_outcome_cols)} outcomes")
    
    return df_clean


def calculate_effect_sizes(df: pd.DataFrame, cause_columns: List[str], outcomes: List[str]) -> pd.DataFrame:
    """
    Calculate risk ratios and risk differences with confidence intervals for dual outcomes.
    
    This function creates 2x2 contingency tables for each weight gain cause and outcome combination,
    calculates both risk ratios and risk differences with 95% confidence intervals,
    and handles edge cases appropriately. Processes both 10%_wl_achieved and 60d_dropout outcomes.
    
    Args:
        df (pd.DataFrame): Input DataFrame with outcome and cause columns
        cause_columns (List[str]): List of weight gain cause column names
        outcomes (List[str]): List of outcome column names (e.g., ['10%_wl_achieved', '60d_dropout'])
        
    Returns:
        pd.DataFrame: Results with columns: cause, outcome, risk_ratio, rr_ci_lower, rr_ci_upper,
                     risk_difference, rd_ci_lower, rd_ci_upper, n_present, n_absent, 
                     events_present, events_absent
                     
    Requirements: 1.3, 1.4, 1.5, 5.3, 5.4
    """
    print(f"Calculating effect sizes for {len(cause_columns)} weight gain causes and {len(outcomes)} outcomes")
    print(f"Outcomes: {outcomes}")
    
    results_list = []
    
    for outcome_col in outcomes:
        print(f"\nProcessing outcome: {outcome_col}")
        
        if outcome_col not in df.columns:
            print(f"⚠ Warning: Outcome column '{outcome_col}' not found in data - skipping")
            continue
        
        for cause in cause_columns:
            print(f"  Processing cause: {cause} for outcome: {outcome_col}")
            
            try:
                # Create 2x2 contingency table
                # Remove rows with missing data for this cause or outcome
                analysis_df = df[[outcome_col, cause]].dropna()
                
                if len(analysis_df) == 0:
                    print(f"    ⚠ Warning: No valid data for cause '{cause}' and outcome '{outcome_col}' - skipping")
                    continue
                
                # Build contingency table
                # a = outcome achieved AND cause present
                # b = outcome not achieved AND cause present  
                # c = outcome achieved AND cause absent
                # d = outcome not achieved AND cause absent
                
                cause_present = analysis_df[cause] == 1
                cause_absent = analysis_df[cause] == 0
                outcome_achieved = analysis_df[outcome_col] == 1
                outcome_not_achieved = analysis_df[outcome_col] == 0
                
                a = len(analysis_df[cause_present & outcome_achieved])  # events in exposed group
                b = len(analysis_df[cause_present & outcome_not_achieved])  # non-events in exposed group
                c = len(analysis_df[cause_absent & outcome_achieved])  # events in unexposed group
                d = len(analysis_df[cause_absent & outcome_not_achieved])  # non-events in unexposed group
                
                # Validate contingency table
                total_check = a + b + c + d
                if total_check != len(analysis_df):
                    print(f"    ⚠ Warning: Contingency table sum ({total_check}) doesn't match data length ({len(analysis_df)}) for {cause}-{outcome_col}")
                
                print(f"    Contingency table for {cause} vs {outcome_col}:")
                print(f"      Cause present: {a} achieved outcome, {b} did not achieve outcome (total: {a+b})")
                print(f"      Cause absent:  {c} achieved outcome, {d} did not achieve outcome (total: {c+d})")
                
                # Handle edge cases - zero cells
                if a == 0 or b == 0 or c == 0 or d == 0:
                    print(f"    ⚠ Warning: Zero cell detected in contingency table for '{cause}' vs '{outcome_col}'")
                    print(f"      Cells: a={a}, b={b}, c={c}, d={d}")
                    
                    # Add 0.5 to all cells (continuity correction) for calculation
                    a_adj = a + 0.5
                    b_adj = b + 0.5
                    c_adj = c + 0.5
                    d_adj = d + 0.5
                    print(f"      Applying continuity correction (+0.5 to all cells)")
                else:
                    a_adj, b_adj, c_adj, d_adj = a, b, c, d
                
                # Calculate risks
                # Risk in exposed group: a/(a+b)
                risk_exposed = a_adj / (a_adj + b_adj)
                
                # Risk in unexposed group: c/(c+d)  
                risk_unexposed = c_adj / (c_adj + d_adj)
                
                # Calculate risk ratio: RR = risk_exposed / risk_unexposed
                if risk_unexposed == 0:
                    print(f"    ⚠ Warning: Risk in unexposed group is 0 for '{cause}' vs '{outcome_col}' - cannot calculate risk ratio")
                    risk_ratio = np.inf
                    rr_ci_lower = np.inf
                    rr_ci_upper = np.inf
                else:
                    risk_ratio = risk_exposed / risk_unexposed
                    
                    # Calculate 95% confidence interval for risk ratio using log transformation
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
                        rr_ci_lower = np.exp(log_ci_lower)
                        rr_ci_upper = np.exp(log_ci_upper)
                        
                        # Check for extreme confidence intervals
                        if rr_ci_upper > 1000 or rr_ci_lower < 0.001:
                            print(f"    ⚠ Warning: Extreme RR confidence interval for '{cause}' vs '{outcome_col}': [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]")
                            print(f"      This may indicate sparse data or numerical instability")
                        
                    except (ValueError, ZeroDivisionError, OverflowError) as e:
                        print(f"    ⚠ Warning: Error calculating RR confidence interval for '{cause}' vs '{outcome_col}': {str(e)}")
                        rr_ci_lower = np.nan
                        rr_ci_upper = np.nan
                
                # Calculate risk difference: RD = risk_exposed - risk_unexposed
                # RD = (a/(a+b)) - (c/(c+d))
                risk_difference = risk_exposed - risk_unexposed
                
                # Calculate 95% confidence interval for risk difference
                # SE(RD) = sqrt((a*b)/((a+b)^3) + (c*d)/((c+d)^3))
                try:
                    se_rd = np.sqrt(
                        (a_adj * b_adj) / ((a_adj + b_adj)**3) + 
                        (c_adj * d_adj) / ((c_adj + d_adj)**3)
                    )
                    
                    # 95% CI for RD
                    z_score = 1.96  # 95% confidence level
                    
                    rd_ci_lower = risk_difference - (z_score * se_rd)
                    rd_ci_upper = risk_difference + (z_score * se_rd)
                    
                    # Check for extreme confidence intervals
                    if rd_ci_upper > 1.0 or rd_ci_lower < -1.0:
                        print(f"    ⚠ Warning: RD confidence interval outside [-1,1] for '{cause}' vs '{outcome_col}': [{rd_ci_lower:.3f}, {rd_ci_upper:.3f}]")
                        print(f"      This may indicate sparse data or numerical instability")
                    
                except (ValueError, ZeroDivisionError, OverflowError) as e:
                    print(f"    ⚠ Warning: Error calculating RD confidence interval for '{cause}' vs '{outcome_col}': {str(e)}")
                    rd_ci_lower = np.nan
                    rd_ci_upper = np.nan
                
                # Store results
                result = {
                    'cause': cause,
                    'outcome': outcome_col,
                    'risk_ratio': risk_ratio,
                    'rr_ci_lower': rr_ci_lower,
                    'rr_ci_upper': rr_ci_upper,
                    'risk_difference': risk_difference,
                    'rd_ci_lower': rd_ci_lower,
                    'rd_ci_upper': rd_ci_upper,
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
                
                print(f"    ✓ Risk ratio: {risk_ratio:.3f} (95% CI: {rr_ci_lower:.3f}-{rr_ci_upper:.3f})")
                print(f"    ✓ Risk difference: {risk_difference:.3f} (95% CI: {rd_ci_lower:.3f}-{rd_ci_upper:.3f})")
                print(f"    ✓ Risk exposed: {risk_exposed:.3f}, Risk unexposed: {risk_unexposed:.3f}")
                
            except Exception as e:
                print(f"    ⚠ Error processing cause '{cause}' for outcome '{outcome_col}': {str(e)}")
                print(f"      Skipping this cause-outcome combination and continuing with analysis")
                continue
    
    if not results_list:
        print("⚠ Warning: No valid effect sizes calculated - returning empty results")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"\n✓ Effect size calculations complete")
    print(f"✓ Successfully calculated effect sizes for {len(results_df)} cause-outcome combinations")
    
    # Summary statistics by outcome
    for outcome in outcomes:
        outcome_results = results_df[results_df['outcome'] == outcome]
        if len(outcome_results) > 0:
            print(f"✓ {outcome}: {len(outcome_results)} causes analyzed")
            print(f"  Risk ratio range: {outcome_results['risk_ratio'].min():.3f} to {outcome_results['risk_ratio'].max():.3f}")
            print(f"  Risk difference range: {outcome_results['risk_difference'].min():.3f} to {outcome_results['risk_difference'].max():.3f}")
    
    return results_df


def perform_statistical_tests(df: pd.DataFrame, cause_columns: List[str], outcomes: List[str]) -> pd.DataFrame:
    """
    Perform appropriate statistical tests for multiple outcomes across weight gain causes.
    
    This function adapts the existing perform_statistical_tests to handle both outcomes,
    maintains Chi-squared/Fisher's exact test selection logic, returns p-values organized 
    by cause-outcome combinations, and preserves existing error handling and print messages.
    
    Args:
        df (pd.DataFrame): Input DataFrame with outcome and cause columns
        cause_columns (List[str]): List of weight gain cause column names
        outcomes (List[str]): List of outcome column names (e.g., ['10%_wl_achieved', '60d_dropout'])
        
    Returns:
        pd.DataFrame: Results with columns: cause, outcome, p_value, test_used, 
                     contingency_a, contingency_b, contingency_c, contingency_d
                     
    Requirements: 1.6, 2.1, 2.2, 4.3, 5.3
    """
    print(f"Performing statistical tests for {len(cause_columns)} weight gain causes and {len(outcomes)} outcomes")
    print(f"Outcomes: {outcomes}")
    
    results_list = []
    
    for outcome_col in outcomes:
        print(f"\nProcessing outcome: {outcome_col}")
        
        if outcome_col not in df.columns:
            print(f"⚠ Warning: Outcome column '{outcome_col}' not found in data - skipping")
            continue
        
        for cause in cause_columns:
            print(f"  Testing cause: {cause} for outcome: {outcome_col}")
            
            try:
                # Remove rows with missing data for this cause or outcome
                analysis_df = df[[outcome_col, cause]].dropna()
                
                if len(analysis_df) == 0:
                    print(f"    ⚠ Warning: No valid data for cause '{cause}' and outcome '{outcome_col}' - assigning p-value = NaN")
                    result = {
                        'cause': cause,
                        'outcome': outcome_col,
                        'p_value': np.nan,
                        'test_used': 'none',
                        'contingency_a': 0,
                        'contingency_b': 0,
                        'contingency_c': 0,
                        'contingency_d': 0,
                        'min_cell_count': 0,
                        'n_total': 0
                    }
                    results_list.append(result)
                    continue
                
                # Check if we have enough variation for testing
                outcome_unique = analysis_df[outcome_col].nunique()
                cause_unique = analysis_df[cause].nunique()
                
                if outcome_unique < 2 or cause_unique < 2:
                    print(f"    ⚠ Warning: Insufficient variation for testing '{cause}' vs '{outcome_col}' - assigning p-value = NaN")
                    print(f"      Outcome unique values: {outcome_unique}, Cause unique values: {cause_unique}")
                    result = {
                        'cause': cause,
                        'outcome': outcome_col,
                        'p_value': np.nan,
                        'test_used': 'insufficient_variation',
                        'contingency_a': 0,
                        'contingency_b': 0,
                        'contingency_c': 0,
                        'contingency_d': 0,
                        'min_cell_count': 0,
                        'n_total': len(analysis_df)
                    }
                    results_list.append(result)
                    continue
                
                # Build contingency table to check cell counts
                cause_present = analysis_df[cause] == 1
                cause_absent = analysis_df[cause] == 0
                outcome_achieved = analysis_df[outcome_col] == 1
                outcome_not_achieved = analysis_df[outcome_col] == 0
                
                a = len(analysis_df[cause_present & outcome_achieved])  # events in exposed group
                b = len(analysis_df[cause_present & outcome_not_achieved])  # non-events in exposed group
                c = len(analysis_df[cause_absent & outcome_achieved])  # events in unexposed group
                d = len(analysis_df[cause_absent & outcome_not_achieved])  # non-events in unexposed group
                
                print(f"    Contingency table: a={a}, b={b}, c={c}, d={d}")
                
                # Check if any cell has count < 5 (Fisher's exact test criterion)
                min_cell_count = min(a, b, c, d)
                use_fisher = min_cell_count < 5
                
                if use_fisher:
                    print(f"    Using Fisher's exact test (min cell count: {min_cell_count} < 5)")
                    
                    try:
                        # Fisher's exact test using scipy.stats.fisher_exact
                        # Create 2x2 contingency table for fisher_exact
                        contingency_matrix = np.array([[a, b], [c, d]])
                        
                        # fisher_exact returns (odds_ratio, p_value)
                        odds_ratio, p_value = fisher_exact(contingency_matrix, alternative='two-sided')
                        
                        print(f"    ✓ Fisher's exact test p-value: {p_value:.6f}")
                        
                        result = {
                            'cause': cause,
                            'outcome': outcome_col,
                            'p_value': p_value,
                            'test_used': 'fisher_exact',
                            'contingency_a': a,
                            'contingency_b': b,
                            'contingency_c': c,
                            'contingency_d': d,
                            'min_cell_count': min_cell_count,
                            'n_total': len(analysis_df)
                        }
                        results_list.append(result)
                        
                    except Exception as e:
                        print(f"    ⚠ Error in Fisher's exact test for '{cause}' vs '{outcome_col}': {str(e)}")
                        print(f"      Assigning p-value = NaN")
                        result = {
                            'cause': cause,
                            'outcome': outcome_col,
                            'p_value': np.nan,
                            'test_used': 'fisher_exact_failed',
                            'contingency_a': a,
                            'contingency_b': b,
                            'contingency_c': c,
                            'contingency_d': d,
                            'min_cell_count': min_cell_count,
                            'n_total': len(analysis_df)
                        }
                        results_list.append(result)
                        
                else:
                    print(f"    Using Chi-squared test (min cell count: {min_cell_count} >= 5)")
                    
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
                            print(f"    ⚠ Warning: categorical_pvalue returned NaN for '{cause}' vs '{outcome_col}'")
                            print(f"      This may indicate insufficient data or other statistical issues")
                        else:
                            print(f"    ✓ Chi-squared test p-value: {p_value:.6f}")
                        
                        result = {
                            'cause': cause,
                            'outcome': outcome_col,
                            'p_value': p_value,
                            'test_used': 'chi_squared',
                            'contingency_a': a,
                            'contingency_b': b,
                            'contingency_c': c,
                            'contingency_d': d,
                            'min_cell_count': min_cell_count,
                            'n_total': len(analysis_df)
                        }
                        results_list.append(result)
                        
                    except Exception as e:
                        print(f"    ⚠ Error in Chi-squared test for '{cause}' vs '{outcome_col}': {str(e)}")
                        print(f"      Assigning p-value = NaN")
                        result = {
                            'cause': cause,
                            'outcome': outcome_col,
                            'p_value': np.nan,
                            'test_used': 'chi_squared_failed',
                            'contingency_a': a,
                            'contingency_b': b,
                            'contingency_c': c,
                            'contingency_d': d,
                            'min_cell_count': min_cell_count,
                            'n_total': len(analysis_df)
                        }
                        results_list.append(result)
            
            except Exception as e:
                print(f"    ⚠ Critical error processing cause '{cause}' for outcome '{outcome_col}': {str(e)}")
                print(f"      Assigning p-value = NaN and continuing with next cause-outcome combination")
                result = {
                    'cause': cause,
                    'outcome': outcome_col,
                    'p_value': np.nan,
                    'test_used': 'critical_error',
                    'contingency_a': 0,
                    'contingency_b': 0,
                    'contingency_c': 0,
                    'contingency_d': 0,
                    'min_cell_count': 0,
                    'n_total': 0
                }
                results_list.append(result)
    
    if not results_list:
        print("⚠ Warning: No statistical tests performed - returning empty results")
        return pd.DataFrame()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Summary statistics by outcome
    print(f"\n✓ Statistical testing complete")
    print(f"✓ Total cause-outcome combinations tested: {len(results_df)}")
    
    for outcome in outcomes:
        outcome_results = results_df[results_df['outcome'] == outcome]
        if len(outcome_results) > 0:
            valid_p_values = outcome_results['p_value'].dropna()
            n_valid = len(valid_p_values)
            n_total = len(outcome_results)
            n_failed = n_total - n_valid
            
            print(f"\n✓ {outcome}: {n_total} causes tested")
            print(f"  Valid tests: {n_valid}/{n_total}")
            
            if n_failed > 0:
                print(f"  Failed tests: {n_failed}/{n_total}")
                failed_causes = outcome_results[outcome_results['p_value'].isna()]['cause'].tolist()
                print(f"    Failed causes: {failed_causes}")
            
            if n_valid > 0:
                min_p = valid_p_values.min()
                max_p = valid_p_values.max()
                print(f"  P-value range: {min_p:.6f} to {max_p:.6f}")
                
                # Count significant results at alpha = 0.05
                n_significant = sum(1 for p in valid_p_values if p < 0.05)
                print(f"  Nominally significant (p < 0.05): {n_significant}/{n_valid}")
                
                # Count test types used
                test_counts = outcome_results['test_used'].value_counts()
                print(f"  Test types used: {dict(test_counts)}")
    
    # Overall summary
    all_valid_p = results_df['p_value'].dropna()
    if len(all_valid_p) > 0:
        overall_significant = sum(1 for p in all_valid_p if p < 0.05)
        print(f"\n✓ Overall summary:")
        print(f"  Total valid tests across all outcomes: {len(all_valid_p)}")
        print(f"  Overall nominally significant (p < 0.05): {overall_significant}/{len(all_valid_p)}")
    
    return results_df


def apply_fdr_correction_to_results(results_df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """
    Apply separate FDR correction for each outcome using existing fdr_correction_utils.
    
    This function extends existing FDR correction to apply separately for each outcome,
    maintains integration with fdr_correction_utils.apply_fdr_correction, stores results
    organized by outcome with both raw and corrected p-values, and prints summary 
    statistics for both outcomes.
    
    Args:
        results_df (pd.DataFrame): Statistical test results with columns: cause, outcome, p_value
        outcomes (List[str]): List of outcome names to process separately
        
    Returns:
        pd.DataFrame: Results with added p_value_fdr column containing FDR-corrected p-values
                     organized by outcome
                     
    Requirements: 2.3, 2.4, 4.2, 5.5
    """
    print(f"Applying FDR correction separately for {len(outcomes)} outcomes")
    print(f"Outcomes: {outcomes}")
    
    if results_df.empty:
        print("⚠ Warning: Empty results DataFrame - returning without FDR correction")
        return results_df
    
    # Create a copy to avoid modifying the original
    results_with_fdr = results_df.copy()
    
    # Initialize FDR-corrected p-value column
    results_with_fdr['p_value_fdr'] = np.nan
    
    # Track summary statistics for each outcome
    fdr_summary = {}
    
    for outcome in outcomes:
        print(f"\nProcessing FDR correction for outcome: {outcome}")
        
        # Filter results for this outcome
        outcome_mask = results_with_fdr['outcome'] == outcome
        outcome_results = results_with_fdr[outcome_mask]
        
        if len(outcome_results) == 0:
            print(f"  ⚠ Warning: No results found for outcome '{outcome}' - skipping FDR correction")
            fdr_summary[outcome] = {
                'n_total': 0,
                'n_valid_pvals': 0,
                'n_corrected': 0,
                'raw_significant': 0,
                'fdr_significant': 0,
                'min_raw_pval': np.nan,
                'min_fdr_pval': np.nan
            }
            continue
        
        # Extract p-values for this outcome
        outcome_pvals = outcome_results['p_value'].tolist()
        
        # Count valid p-values (non-NaN)
        valid_pvals = [p for p in outcome_pvals if not pd.isna(p)]
        n_valid = len(valid_pvals)
        n_total = len(outcome_pvals)
        n_missing = n_total - n_valid
        
        print(f"  Total tests for {outcome}: {n_total}")
        print(f"  Valid p-values: {n_valid}")
        
        if n_missing > 0:
            print(f"  Missing/invalid p-values: {n_missing}")
            missing_causes = outcome_results[outcome_results['p_value'].isna()]['cause'].tolist()
            print(f"    Causes with missing p-values: {missing_causes}")
        
        if n_valid == 0:
            print(f"  ⚠ Warning: No valid p-values for outcome '{outcome}' - skipping FDR correction")
            fdr_summary[outcome] = {
                'n_total': n_total,
                'n_valid_pvals': 0,
                'n_corrected': 0,
                'raw_significant': 0,
                'fdr_significant': 0,
                'min_raw_pval': np.nan,
                'min_fdr_pval': np.nan
            }
            continue
        
        if n_valid == 1:
            print(f"  ℹ Single valid p-value for outcome '{outcome}' - FDR correction equals original p-value")
        
        try:
            # Apply FDR correction using existing utility function
            print(f"  Applying Benjamini-Hochberg FDR correction to {n_valid} p-values...")
            
            corrected_pvals = apply_fdr_correction(outcome_pvals, method='fdr_bh', alpha=0.05)
            
            # Update the results DataFrame with corrected p-values for this outcome
            results_with_fdr.loc[outcome_mask, 'p_value_fdr'] = corrected_pvals
            
            # Calculate summary statistics
            valid_raw_pvals = [p for p in outcome_pvals if not pd.isna(p)]
            valid_corrected_pvals = [p for p in corrected_pvals if not pd.isna(p)]
            
            raw_significant = sum(1 for p in valid_raw_pvals if p < 0.05)
            fdr_significant = sum(1 for p in valid_corrected_pvals if p < 0.05)
            
            min_raw_pval = min(valid_raw_pvals) if valid_raw_pvals else np.nan
            min_fdr_pval = min(valid_corrected_pvals) if valid_corrected_pvals else np.nan
            
            # Store summary for this outcome
            fdr_summary[outcome] = {
                'n_total': n_total,
                'n_valid_pvals': n_valid,
                'n_corrected': len(valid_corrected_pvals),
                'raw_significant': raw_significant,
                'fdr_significant': fdr_significant,
                'min_raw_pval': min_raw_pval,
                'min_fdr_pval': min_fdr_pval
            }
            
            print(f"  ✓ FDR correction completed for {outcome}")
            print(f"    Raw p-values < 0.05: {raw_significant}/{n_valid}")
            print(f"    FDR-corrected p-values < 0.05: {fdr_significant}/{n_valid}")
            print(f"    Minimum raw p-value: {min_raw_pval:.6f}" if not pd.isna(min_raw_pval) else "    Minimum raw p-value: N/A")
            print(f"    Minimum FDR-corrected p-value: {min_fdr_pval:.6f}" if not pd.isna(min_fdr_pval) else "    Minimum FDR-corrected p-value: N/A")
            
            # Show examples of correction for most significant results
            if n_valid > 0:
                outcome_results_updated = results_with_fdr[outcome_mask].copy()
                valid_results = outcome_results_updated[outcome_results_updated['p_value'].notna()]
                
                if len(valid_results) > 0:
                    # Sort by raw p-value to show most significant
                    valid_results_sorted = valid_results.sort_values('p_value')
                    n_examples = min(3, len(valid_results_sorted))
                    
                    print(f"    Examples of FDR correction (top {n_examples} most significant):")
                    for i, (_, row) in enumerate(valid_results_sorted.head(n_examples).iterrows()):
                        cause = row['cause']
                        raw_p = row['p_value']
                        fdr_p = row['p_value_fdr']
                        print(f"      {i+1}. {cause}: {raw_p:.6f} → {fdr_p:.6f}")
            
        except Exception as e:
            print(f"  ⚠ Error applying FDR correction for outcome '{outcome}': {str(e)}")
            print(f"    FDR-corrected p-values will remain as NaN for this outcome")
            
            # Store error summary
            fdr_summary[outcome] = {
                'n_total': n_total,
                'n_valid_pvals': n_valid,
                'n_corrected': 0,
                'raw_significant': sum(1 for p in valid_pvals if p < 0.05) if valid_pvals else 0,
                'fdr_significant': 0,
                'min_raw_pval': min(valid_pvals) if valid_pvals else np.nan,
                'min_fdr_pval': np.nan,
                'error': str(e)
            }
    
    # Print overall summary statistics for both outcomes
    print(f"\n" + "=" * 50)
    print("FDR CORRECTION SUMMARY")
    print("=" * 50)
    
    total_tests = 0
    total_valid = 0
    total_corrected = 0
    total_raw_sig = 0
    total_fdr_sig = 0
    
    for outcome, summary in fdr_summary.items():
        print(f"\n{outcome}:")
        print(f"  Total tests: {summary['n_total']}")
        print(f"  Valid p-values: {summary['n_valid_pvals']}")
        print(f"  Successfully corrected: {summary['n_corrected']}")
        print(f"  Raw significant (p < 0.05): {summary['raw_significant']}")
        print(f"  FDR significant (p < 0.05): {summary['fdr_significant']}")
        
        if not pd.isna(summary['min_raw_pval']):
            print(f"  Minimum raw p-value: {summary['min_raw_pval']:.6f}")
        if not pd.isna(summary['min_fdr_pval']):
            print(f"  Minimum FDR p-value: {summary['min_fdr_pval']:.6f}")
        
        if 'error' in summary:
            print(f"  ⚠ Error: {summary['error']}")
        
        # Add to totals
        total_tests += summary['n_total']
        total_valid += summary['n_valid_pvals']
        total_corrected += summary['n_corrected']
        total_raw_sig += summary['raw_significant']
        total_fdr_sig += summary['fdr_significant']
    
    # Overall summary across all outcomes
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total tests across all outcomes: {total_tests}")
    print(f"  Total valid p-values: {total_valid}")
    print(f"  Total successfully corrected: {total_corrected}")
    print(f"  Total raw significant (p < 0.05): {total_raw_sig}")
    print(f"  Total FDR significant (p < 0.05): {total_fdr_sig}")
    
    if total_valid > 0:
        raw_sig_rate = (total_raw_sig / total_valid) * 100
        fdr_sig_rate = (total_fdr_sig / total_valid) * 100
        print(f"  Raw significance rate: {raw_sig_rate:.1f}%")
        print(f"  FDR significance rate: {fdr_sig_rate:.1f}%")
        
        if total_raw_sig > 0:
            fdr_reduction = ((total_raw_sig - total_fdr_sig) / total_raw_sig) * 100
            print(f"  FDR reduction in significant findings: {fdr_reduction:.1f}%")
    
    # Store summary in DataFrame attributes for later use
    results_with_fdr.attrs['fdr_summary'] = fdr_summary
    
    print(f"\n✓ FDR correction complete for all outcomes")
    print(f"✓ Results organized by outcome with both raw and corrected p-values")
    
    return results_with_fdr


def create_forest_plots(results_df: pd.DataFrame, output_dir: str, row_order: List = None) -> None:
    """
    Create dual forest plot visualization system for both risk ratios and risk differences.
    
    This function extends the existing create_forest_plot to create_forest_plots (plural)
    and generates separate plots for each outcome and effect measure combination:
    - Risk ratio plots (log scale, RR=1.0 reference) for each outcome
    - Risk difference plots (linear scale, RD=0.0 reference) for each outcome
    
    Maintains existing plot styling and confidence interval visualization while
    removing significance markers and relying on confidence intervals crossing 
    reference lines to indicate statistical significance.
    
    Args:
        results_df (pd.DataFrame): DataFrame with effect size results including columns:
                                  cause, outcome, risk_ratio, rr_ci_lower, rr_ci_upper,
                                  risk_difference, rd_ci_lower, rd_ci_upper
        output_dir (str): Directory path where forest plot files should be saved
        row_order (List, optional): Row order configuration for pretty cause names
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
    """
    print(f"Creating dual forest plot visualization system")
    print(f"Output directory: {output_dir}")
    
    try:
        # Validate input data
        if results_df.empty:
            print("⚠ Warning: Empty results DataFrame - creating placeholder plots")
            
            # Create placeholder plots for both outcomes and effect measures
            outcomes = ["10%_wl_achieved", "60d_dropout"]
            effect_measures = [("risk_ratios", "Risk Ratio (RR)", True), 
                             ("risk_differences", "Risk Difference (RD)", False)]
            
            for outcome in outcomes:
                for measure_name, measure_label, use_log_scale in effect_measures:
                    filename = f"{measure_name}_{outcome.replace('%', 'pct').replace('_', '_')}.png"
                    output_path = os.path.join(output_dir, "forest_plots", filename)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    _create_placeholder_plot(output_path, measure_label, outcome)
            
            print("✓ Placeholder forest plots created for all outcome-measure combinations")
            return
        
        # Validate required columns
        required_cols = ['cause', 'outcome', 'risk_ratio', 'rr_ci_lower', 'rr_ci_upper',
                        'risk_difference', 'rd_ci_lower', 'rd_ci_upper']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            error_msg = f"Missing required columns in results DataFrame: {missing_cols}"
            print(f"⚠ Error: {error_msg}")
            raise ValueError(error_msg)
        
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
        
        # Create forest_plots subdirectory
        forest_plots_dir = os.path.join(output_dir, "forest_plots")
        os.makedirs(forest_plots_dir, exist_ok=True)
        print(f"✓ Created forest plots directory: {forest_plots_dir}")
        
        # Get unique outcomes from results
        outcomes = results_df['outcome'].unique()
        print(f"✓ Found {len(outcomes)} outcomes: {list(outcomes)}")
        
        # Generate plots for each outcome and effect measure combination
        for outcome in outcomes:
            print(f"\nGenerating plots for outcome: {outcome}")
            
            # Filter data for this outcome
            outcome_data = results_df[results_df['outcome'] == outcome].copy()
            
            if outcome_data.empty:
                print(f"⚠ Warning: No data for outcome '{outcome}' - skipping")
                continue
            
            # Add pretty names
            outcome_data['cause_pretty'] = outcome_data['cause'].map(
                lambda x: cause_pretty_names.get(x, x.replace('_', ' ').title())
            )
            
            # Create outcome pretty name for titles
            outcome_pretty = outcome.replace('10%_wl_achieved', '10% Weight Loss Achievement').replace('60d_dropout', '60-Day Dropout')
            
            # Generate Risk Ratio plot (log scale)
            print(f"  Creating risk ratio plot for {outcome}")
            rr_filename = f"risk_ratios_{outcome.replace('%', 'pct')}.png"
            rr_output_path = os.path.join(forest_plots_dir, rr_filename)
            
            _create_single_forest_plot(
                data=outcome_data,
                output_path=rr_output_path,
                effect_col='risk_ratio',
                ci_lower_col='rr_ci_lower',
                ci_upper_col='rr_ci_upper',
                effect_label='Risk Ratio (RR)',
                outcome_label=outcome_pretty,
                reference_value=1.0,
                use_log_scale=True
            )
            
            # Generate Risk Difference plot (linear scale)
            print(f"  Creating risk difference plot for {outcome}")
            rd_filename = f"risk_differences_{outcome.replace('%', 'pct')}.png"
            rd_output_path = os.path.join(forest_plots_dir, rd_filename)
            
            _create_single_forest_plot(
                data=outcome_data,
                output_path=rd_output_path,
                effect_col='risk_difference',
                ci_lower_col='rd_ci_lower',
                ci_upper_col='rd_ci_upper',
                effect_label='Risk Difference (RD)',
                outcome_label=outcome_pretty,
                reference_value=0.0,
                use_log_scale=False
            )
        
        print(f"\n✓ Dual forest plot visualization system complete")
        print(f"✓ Generated plots for {len(outcomes)} outcomes with both risk ratios and risk differences")
        print(f"✓ All plots saved to: {forest_plots_dir}")
        
    except Exception as e:
        error_msg = f"Error creating dual forest plots: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)


def _create_placeholder_plot(output_path: str, measure_label: str, outcome: str) -> None:
    """Create a placeholder plot when no data is available."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    outcome_pretty = outcome.replace('10%_wl_achieved', '10% Weight Loss Achievement').replace('60d_dropout', '60-Day Dropout')
    
    ax.text(0.5, 0.5, 'No valid results to display', 
           horizontalalignment='center', verticalalignment='center',
           transform=ax.transAxes, fontsize=16, fontweight='bold', color='gray')
    ax.set_title(f'Forest Plot: {measure_label} for {outcome_pretty}', 
                fontsize=16, fontweight='bold', color='black', pad=25)
    ax.set_xlabel(measure_label, fontsize=14, fontweight='bold', color='black')
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


def _create_single_forest_plot(data: pd.DataFrame, output_path: str, effect_col: str, 
                              ci_lower_col: str, ci_upper_col: str, effect_label: str,
                              outcome_label: str, reference_value: float, use_log_scale: bool) -> None:
    """
    Create a single forest plot for one effect measure and outcome combination.
    
    This helper function creates individual forest plots with appropriate scaling
    and reference lines, maintaining consistent styling across all plots while
    removing significance markers and relying on confidence intervals.
    """
    print(f"    Creating {effect_label} plot: {os.path.basename(output_path)}")
    
    try:
        # Filter out invalid effect sizes (inf, -inf, NaN)
        if use_log_scale:
            # For log scale, all values must be positive
            valid_mask = (
                np.isfinite(data[effect_col]) & 
                np.isfinite(data[ci_lower_col]) & 
                np.isfinite(data[ci_upper_col]) &
                (data[effect_col] > 0) &
                (data[ci_lower_col] > 0) &
                (data[ci_upper_col] > 0)
            )
        else:
            # For linear scale, just check for finite values
            valid_mask = (
                np.isfinite(data[effect_col]) & 
                np.isfinite(data[ci_lower_col]) & 
                np.isfinite(data[ci_upper_col])
            )
        
        invalid_count = len(data) - valid_mask.sum()
        if invalid_count > 0:
            print(f"    ⚠ Warning: Excluding {invalid_count} causes with invalid {effect_label.lower()} values")
            invalid_causes = data[~valid_mask]['cause'].tolist()
            print(f"      Invalid causes: {invalid_causes}")
        
        plot_data = data[valid_mask].copy()
        
        if plot_data.empty:
            print(f"    ⚠ Warning: No valid data for {effect_label} - creating placeholder")
            _create_placeholder_plot(output_path, effect_label, outcome_label)
            return
        
        # Sort causes by effect size for better visualization
        plot_data = plot_data.sort_values(effect_col, ascending=True).reset_index(drop=True)
        
        print(f"    ✓ Plotting {len(plot_data)} valid causes")
        effect_values = plot_data[effect_col].values
        print(f"    ✓ {effect_label} range: {effect_values.min():.3f} to {effect_values.max():.3f}")
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with publication-quality settings (wider for enhanced labels)
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Create y-axis positions for each cause
        y_positions = np.arange(len(plot_data))
        
        # Get effect sizes and confidence intervals
        ci_lower = plot_data[ci_lower_col].values
        ci_upper = plot_data[ci_upper_col].values
        
        # Calculate error bar lengths (ensure no negative values)
        if use_log_scale:
            # For log scale, use asymmetric error bars
            lower_errors = np.abs(effect_values - ci_lower)
            upper_errors = np.abs(ci_upper - effect_values)
        else:
            # For linear scale, use symmetric error bars
            lower_errors = np.abs(effect_values - ci_lower)
            upper_errors = np.abs(ci_upper - effect_values)
        
        # Plot points and error bars with consistent styling (no significance markers)
        ax.errorbar(
            effect_values, y_positions,
            xerr=[lower_errors, upper_errors],
            fmt='o',  # Circle markers
            markersize=8,
            capsize=5,  # Cap size for error bars
            capthick=2,
            elinewidth=2,
            color='steelblue',
            markerfacecolor='steelblue',
            markeredgecolor='darkblue',
            markeredgewidth=1.5,
            alpha=0.8
        )
        
        # Add vertical reference line with enhanced styling
        ref_label = f'No effect ({effect_label.split()[0]} = {reference_value})'
        ax.axvline(x=reference_value, color='red', linestyle='--', linewidth=3, alpha=0.8, 
                  label=ref_label, zorder=1)
        
        # Set appropriate axis scale
        if use_log_scale:
            ax.set_xscale('log')
            
            # Adjust x-axis limits for log scale
            min_val = min(ci_lower.min(), effect_values.min())
            max_val = max(ci_upper.max(), effect_values.max())
            
            x_min = max(0.1, min_val * 0.8)  # Don't go below 0.1 for log scale
            x_max = min(10.0, max_val * 1.2)  # Cap at 10 for reasonable display
            ax.set_xlim(x_min, x_max)
            
            # Format x-axis ticks for log scale
            from matplotlib.ticker import LogFormatter, LogLocator
            ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        else:
            # Linear scale
            min_val = min(ci_lower.min(), effect_values.min())
            max_val = max(ci_upper.max(), effect_values.max())
            
            # Add padding to limits
            val_range = max_val - min_val
            padding = val_range * 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
            
            # For risk differences, format x-axis as percentages if values are small decimals
            if 'Risk Difference' in effect_label and max(abs(min_val), abs(max_val)) < 1:
                # Values are likely in decimal form (0.05 = 5%), convert to percentage display
                from matplotlib.ticker import FuncFormatter
                def percent_formatter(x, pos):
                    return f'{x*100:.1f}%'
                ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        
        # Enhanced x-axis labels with publication-friendly descriptions and units
        if 'Risk Ratio' in effect_label:
            if use_log_scale:
                x_label = 'Risk Ratio (Relative likelihood of outcome occurrence)\nLog scale: RR=1.0 means no effect'
            else:
                x_label = 'Risk Ratio (Relative likelihood of outcome occurrence)'
        elif 'Risk Difference' in effect_label:
            x_label = 'Risk Difference (%)\n(Absolute difference in outcome probability)'
        else:
            x_label = effect_label
            
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold', color='black')
        ax.set_ylabel('Weight Gain Causes', fontsize=14, fontweight='bold', color='black')
        
        # Enhanced y-axis labels with sample sizes
        y_labels_with_n = []
        for idx, row in plot_data.iterrows():
            cause_name = row['cause_pretty']
            n_present = row['n_present']
            n_absent = row['n_absent']
            # Add sample sizes on new lines below cause names
            label_with_n = f"{cause_name}\n(n={n_present}|{n_absent})"
            y_labels_with_n.append(label_with_n)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels_with_n, fontsize=10, color='black')
        
        # Enhance tick labels
        ax.tick_params(axis='x', labelsize=11, colors='black')
        ax.tick_params(axis='y', labelsize=10, colors='black')
        
        # Set title with enhanced styling (remove subtitle from middle)
        ax.set_title(f'Forest Plot: {effect_label} for {outcome_label}', 
                    fontsize=16, fontweight='bold', pad=25, color='black')
        
        # Add enhanced grid for better readability
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='lightgray')
        ax.set_axisbelow(True)  # Ensure grid appears behind data points
        
        # Add subtle border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)
        
        # Add clean legend without redundant sample size info
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, alpha=0.7, label=ref_label),
            Line2D([0], [0], marker='o', color='steelblue', linestyle='None', 
                   markersize=8, markerfacecolor='steelblue', markeredgecolor='darkblue',
                   markeredgewidth=1.5, alpha=0.8, label=f'{effect_label}s (n={len(plot_data)})')
        ]
        
        # Create legend with enhanced styling
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                          frameon=True, fancybox=True, shadow=True, 
                          framealpha=0.9, edgecolor='gray')
        legend.get_frame().set_facecolor('white')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout(pad=2.0)
        
        # Save the plot with publication-quality settings
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', format='png', pil_kwargs={'optimize': True})
        plt.close(fig)
        
        print(f"    ✓ {effect_label} plot saved: {os.path.basename(output_path)}")
        print(f"    ✓ Displayed {len(plot_data)} causes with {effect_label.lower()}s from {effect_values.min():.3f} to {effect_values.max():.3f}")
        
    except Exception as e:
        error_msg = f"Error creating {effect_label} plot: {str(e)}"
        print(f"    ⚠ {error_msg}")
        
        # Try to create a basic error plot
        try:
            _create_placeholder_plot(output_path, effect_label, outcome_label)
            print(f"    ✓ Error placeholder saved: {os.path.basename(output_path)}")
        except Exception as save_error:
            print(f"    ⚠ Could not save error plot: {str(save_error)}")
        
        raise RuntimeError(error_msg)


def export_comprehensive_summary_tables(
    merged_results: pd.DataFrame,
    statistical_results: pd.DataFrame, 
    effect_sizes: pd.DataFrame,
    output_dir: str,
    input_table: str
) -> dict:
    """
    Export comprehensive summary tables with all statistics and organized output structure.
    
    This function generates descriptive filenames and exports comprehensive summary tables
    with all statistics, maintaining existing error handling for file operations.
    
    Args:
        merged_results (pd.DataFrame): Combined effect sizes and statistical results
        statistical_results (pd.DataFrame): Statistical test results with FDR correction
        effect_sizes (pd.DataFrame): Effect size calculations
        output_dir (str): Directory path for summary tables
        input_table (str): Name of input table for filename generation
        
    Returns:
        dict: Dictionary of exported file paths and metadata
        
    Requirements: 4.4, 4.6, 5.2, 5.5, 6.3
    """
    print(f"Exporting comprehensive summary tables to: {output_dir}")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Summary tables directory ready: {output_dir}")
        
        exported_files = {}
        
        # 1. Export comprehensive effect sizes summary
        print("  Generating effect sizes summary table...")
        
        effect_sizes_filename = f"effect_sizes_summary_{input_table}.csv"
        effect_sizes_path = os.path.join(output_dir, effect_sizes_filename)
        
        # Create comprehensive effect sizes table with descriptive columns
        effect_sizes_export = effect_sizes.copy()
        
        # Add descriptive columns
        effect_sizes_export['analysis_table'] = input_table
        effect_sizes_export['effect_measure_type'] = 'Both Risk Ratio and Risk Difference'
        
        # Round numerical columns for readability
        numerical_cols = ['risk_ratio', 'rr_ci_lower', 'rr_ci_upper', 
                         'risk_difference', 'rd_ci_lower', 'rd_ci_upper',
                         'risk_exposed', 'risk_unexposed']
        
        for col in numerical_cols:
            if col in effect_sizes_export.columns:
                effect_sizes_export[col] = effect_sizes_export[col].round(4)
        
        # Reorder columns for better readability
        column_order = [
            'analysis_table', 'cause', 'outcome', 'effect_measure_type',
            'n_present', 'n_absent', 'events_present', 'events_absent',
            'risk_exposed', 'risk_unexposed',
            'risk_ratio', 'rr_ci_lower', 'rr_ci_upper',
            'risk_difference', 'rd_ci_lower', 'rd_ci_upper',
            'contingency_a', 'contingency_b', 'contingency_c', 'contingency_d'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in effect_sizes_export.columns]
        effect_sizes_export = effect_sizes_export[available_columns]
        
        # Export with error handling
        try:
            effect_sizes_export.to_csv(effect_sizes_path, index=False)
            exported_files['effect_sizes_summary'] = effect_sizes_path
            print(f"    ✓ Effect sizes summary saved: {effect_sizes_filename}")
            print(f"    ✓ Contains {len(effect_sizes_export)} cause-outcome combinations")
        except Exception as e:
            print(f"    ⚠ Error saving effect sizes summary: {str(e)}")
            raise
        
        # 2. Export comprehensive statistical tests summary
        print("  Generating statistical tests summary table...")
        
        statistical_filename = f"statistical_tests_summary_{input_table}.csv"
        statistical_path = os.path.join(output_dir, statistical_filename)
        
        # Create comprehensive statistical results table
        statistical_export = statistical_results.copy()
        
        # Add descriptive columns
        statistical_export['analysis_table'] = input_table
        statistical_export['multiple_testing_correction'] = 'Benjamini-Hochberg FDR'
        
        # Add significance indicators
        statistical_export['raw_significant'] = (statistical_export['p_value'] < 0.05).astype(str)
        statistical_export['fdr_significant'] = (statistical_export['p_value_fdr'] < 0.05).astype(str)
        
        # Replace boolean strings with more descriptive text
        statistical_export['raw_significant'] = statistical_export['raw_significant'].replace({
            'True': 'Significant (p < 0.05)', 
            'False': 'Not Significant (p >= 0.05)',
            'nan': 'Unable to determine'
        })
        statistical_export['fdr_significant'] = statistical_export['fdr_significant'].replace({
            'True': 'Significant (FDR < 0.05)', 
            'False': 'Not Significant (FDR >= 0.05)',
            'nan': 'Unable to determine'
        })
        
        # Round p-values for readability
        p_value_cols = ['p_value', 'p_value_fdr']
        for col in p_value_cols:
            if col in statistical_export.columns:
                statistical_export[col] = statistical_export[col].round(6)
        
        # Reorder columns for better readability
        stat_column_order = [
            'analysis_table', 'cause', 'outcome', 'multiple_testing_correction',
            'n_total', 'min_cell_count', 'test_used',
            'p_value', 'raw_significant', 'p_value_fdr', 'fdr_significant',
            'contingency_a', 'contingency_b', 'contingency_c', 'contingency_d'
        ]
        
        # Only include columns that exist
        available_stat_columns = [col for col in stat_column_order if col in statistical_export.columns]
        statistical_export = statistical_export[available_stat_columns]
        
        # Export with error handling
        try:
            statistical_export.to_csv(statistical_path, index=False)
            exported_files['statistical_tests_summary'] = statistical_path
            print(f"    ✓ Statistical tests summary saved: {statistical_filename}")
            print(f"    ✓ Contains {len(statistical_export)} statistical tests")
        except Exception as e:
            print(f"    ⚠ Error saving statistical tests summary: {str(e)}")
            raise
        
        # 3. Export combined comprehensive results
        print("  Generating combined comprehensive results table...")
        
        combined_filename = f"comprehensive_results_{input_table}.csv"
        combined_path = os.path.join(output_dir, combined_filename)
        
        # Create comprehensive combined table
        combined_export = merged_results.copy()
        
        # Add metadata columns
        combined_export['analysis_table'] = input_table
        combined_export['analysis_type'] = 'Dual-Outcome Risk Analysis'
        combined_export['effect_measures'] = 'Risk Ratio and Risk Difference'
        combined_export['statistical_correction'] = 'Benjamini-Hochberg FDR'
        
        # Add significance indicators
        combined_export['raw_significant'] = (combined_export['p_value'] < 0.05).map({
            True: 'Significant (p < 0.05)', 
            False: 'Not Significant (p >= 0.05)'
        }).fillna('Unable to determine')
        
        combined_export['fdr_significant'] = (combined_export['p_value_fdr'] < 0.05).map({
            True: 'Significant (FDR < 0.05)', 
            False: 'Not Significant (FDR >= 0.05)'
        }).fillna('Unable to determine')
        
        # Round numerical columns
        numerical_cols_combined = ['risk_ratio', 'rr_ci_lower', 'rr_ci_upper', 
                                  'risk_difference', 'rd_ci_lower', 'rd_ci_upper',
                                  'risk_exposed', 'risk_unexposed', 'p_value', 'p_value_fdr']
        
        for col in numerical_cols_combined:
            if col in combined_export.columns:
                combined_export[col] = combined_export[col].round(6)
        
        # Reorder columns for comprehensive view
        combined_column_order = [
            'analysis_table', 'analysis_type', 'cause', 'outcome',
            'effect_measures', 'statistical_correction',
            'n_present', 'n_absent', 'events_present', 'events_absent',
            'risk_exposed', 'risk_unexposed',
            'risk_ratio', 'rr_ci_lower', 'rr_ci_upper',
            'risk_difference', 'rd_ci_lower', 'rd_ci_upper',
            'test_used', 'p_value', 'raw_significant', 
            'p_value_fdr', 'fdr_significant'
        ]
        
        # Only include columns that exist
        available_combined_columns = [col for col in combined_column_order if col in combined_export.columns]
        combined_export = combined_export[available_combined_columns]
        
        # Export with error handling
        try:
            combined_export.to_csv(combined_path, index=False)
            exported_files['comprehensive_results'] = combined_path
            print(f"    ✓ Combined comprehensive results saved: {combined_filename}")
            print(f"    ✓ Contains {len(combined_export)} complete cause-outcome analyses")
        except Exception as e:
            print(f"    ⚠ Error saving combined comprehensive results: {str(e)}")
            raise
        
        # 4. Export analysis metadata and summary statistics
        print("  Generating analysis metadata and summary statistics...")
        
        metadata_filename = f"analysis_metadata_{input_table}.csv"
        metadata_path = os.path.join(output_dir, metadata_filename)
        
        # Create metadata summary
        outcomes = merged_results['outcome'].unique()
        causes = merged_results['cause'].unique()
        
        metadata_rows = []
        
        # Overall analysis metadata
        metadata_rows.append({
            'metric': 'Analysis Type',
            'value': 'Dual-Outcome Descriptive Visualizations',
            'description': 'Risk ratio and risk difference analysis for multiple outcomes'
        })
        
        metadata_rows.append({
            'metric': 'Input Table',
            'value': input_table,
            'description': 'Source data table for analysis'
        })
        
        metadata_rows.append({
            'metric': 'Number of Outcomes',
            'value': len(outcomes),
            'description': f'Outcomes analyzed: {", ".join(outcomes)}'
        })
        
        metadata_rows.append({
            'metric': 'Number of Weight Gain Causes',
            'value': len(causes),
            'description': f'Total weight gain causes analyzed'
        })
        
        metadata_rows.append({
            'metric': 'Total Cause-Outcome Combinations',
            'value': len(merged_results),
            'description': 'Total number of statistical comparisons performed'
        })
        
        # Statistical testing summary
        valid_tests = len(merged_results[merged_results['p_value'].notna()])
        metadata_rows.append({
            'metric': 'Valid Statistical Tests',
            'value': valid_tests,
            'description': 'Number of successful statistical tests'
        })
        
        # FDR correction summary
        valid_fdr = len(merged_results[merged_results['p_value_fdr'].notna()])
        metadata_rows.append({
            'metric': 'Valid FDR Corrections',
            'value': valid_fdr,
            'description': 'Number of successful FDR corrections'
        })
        
        # Significance summary
        if valid_tests > 0:
            raw_sig = len(merged_results[merged_results['p_value'] < 0.05])
            metadata_rows.append({
                'metric': 'Raw Significant Results (p < 0.05)',
                'value': raw_sig,
                'description': f'{raw_sig}/{valid_tests} tests significant before correction'
            })
        
        if valid_fdr > 0:
            fdr_sig = len(merged_results[merged_results['p_value_fdr'] < 0.05])
            metadata_rows.append({
                'metric': 'FDR Significant Results (FDR < 0.05)',
                'value': fdr_sig,
                'description': f'{fdr_sig}/{valid_fdr} tests significant after FDR correction'
            })
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)
        
        # Export with error handling
        try:
            metadata_df.to_csv(metadata_path, index=False)
            exported_files['analysis_metadata'] = metadata_path
            print(f"    ✓ Analysis metadata saved: {metadata_filename}")
            print(f"    ✓ Contains {len(metadata_df)} metadata entries")
        except Exception as e:
            print(f"    ⚠ Error saving analysis metadata: {str(e)}")
            raise
        
        # 5. Generate console output summary file
        print("  Generating console output summary...")
        
        console_filename = f"console_output_summary_{input_table}.txt"
        console_path = os.path.join(output_dir, console_filename)
        
        # Create console output summary
        console_summary = []
        console_summary.append("=" * 60)
        console_summary.append("DESCRIPTIVE VISUALIZATIONS PIPELINE - OUTPUT SUMMARY")
        console_summary.append("=" * 60)
        console_summary.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console_summary.append(f"Input Table: {input_table}")
        console_summary.append("")
        
        console_summary.append("ANALYSIS OVERVIEW:")
        console_summary.append(f"- Outcomes analyzed: {len(outcomes)} ({', '.join(outcomes)})")
        console_summary.append(f"- Weight gain causes: {len(causes)}")
        console_summary.append(f"- Total comparisons: {len(merged_results)}")
        console_summary.append("")
        
        console_summary.append("STATISTICAL TESTING SUMMARY:")
        console_summary.append(f"- Valid statistical tests: {valid_tests}/{len(merged_results)}")
        console_summary.append(f"- Valid FDR corrections: {valid_fdr}/{len(merged_results)}")
        
        if valid_tests > 0:
            raw_sig = len(merged_results[merged_results['p_value'] < 0.05])
            console_summary.append(f"- Raw significant results (p < 0.05): {raw_sig}/{valid_tests}")
        
        if valid_fdr > 0:
            fdr_sig = len(merged_results[merged_results['p_value_fdr'] < 0.05])
            console_summary.append(f"- FDR significant results (FDR < 0.05): {fdr_sig}/{valid_fdr}")
        
        console_summary.append("")
        
        console_summary.append("OUTPUT FILES GENERATED:")
        for file_type, file_path in exported_files.items():
            filename = os.path.basename(file_path)
            console_summary.append(f"- {file_type}: {filename}")
        
        console_summary.append("")
        console_summary.append("FOREST PLOTS GENERATED:")
        for outcome in outcomes:
            rr_filename = f"risk_ratios_{outcome.replace('%', 'pct')}.png"
            rd_filename = f"risk_differences_{outcome.replace('%', 'pct')}.png"
            console_summary.append(f"- {outcome}: {rr_filename}, {rd_filename}")
        
        console_summary.append("")
        console_summary.append("=" * 60)
        console_summary.append("END OF SUMMARY")
        console_summary.append("=" * 60)
        
        # Export console summary with error handling
        try:
            with open(console_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(console_summary))
            exported_files['console_output_summary'] = console_path
            print(f"    ✓ Console output summary saved: {console_filename}")
        except Exception as e:
            print(f"    ⚠ Error saving console output summary: {str(e)}")
            raise
        
        print(f"\n✓ Comprehensive summary tables export complete")
        print(f"✓ Generated {len(exported_files)} summary files with descriptive filenames")
        print(f"✓ All files saved to: {output_dir}")
        
        # Print file summary
        print(f"\nEXPORTED FILES:")
        for file_type, file_path in exported_files.items():
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"  - {file_type}: {filename} ({file_size:,} bytes)")
        
        return exported_files
        
    except Exception as e:
        error_msg = f"Error exporting comprehensive summary tables: {str(e)}"
        print(f"⚠ {error_msg}")
        raise RuntimeError(error_msg)


def run_descriptive_visualizations(
    input_table: str,
    config: master_config = None
) -> dict:
    """
    Main function to run comprehensive descriptive visualizations pipeline.
    
    This function serves as the primary entry point for generating both risk ratio 
    and risk difference forest plots for multiple outcomes. It provides a clean,
    simplified interface for notebook usage while maintaining comprehensive 
    functionality and organized output structure.
    
    Args:
        input_table (str): Name of the input table containing the data
        config (master_config, optional): Configuration object with database paths
        
    Returns:
        dict: Summary of results including file paths and statistics
        
    Requirements: 6.1, 6.2, 4.1, 4.6
    """
    print("=" * 60)
    print("COMPREHENSIVE DESCRIPTIVE VISUALIZATIONS PIPELINE")
    print("=" * 60)
    print(f"Input table: {input_table}")
    
    # Set up output directory structure
    output_base_dir = "../outputs/descriptive_visualizations"
    forest_plots_dir = os.path.join(output_base_dir, "forest_plots")
    summary_tables_dir = os.path.join(output_base_dir, "summary_tables")
    
    # Create output directories if they don't exist
    for directory in [output_base_dir, forest_plots_dir, summary_tables_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")
    
    try:
        # Get database path from config - use INPUT database for raw data
        if config is None or config.paths is None:
            # Use default INPUT database path if no config provided
            db_path = "../dbs/pnk_db2_p2_in.sqlite"
            print(f"Using default INPUT database path: {db_path}")
        else:
            db_path = config.paths.paper_in_db
            print(f"Using configured INPUT database path: {db_path}")
        
        # Load and validate data for dual-outcome analysis
        print("\n" + "=" * 40)
        print("STEP 1: DATA LOADING AND VALIDATION")
        print("=" * 40)
        
        # For now, we'll use None for row_order - this will be extended in future tasks
        df = load_forest_plot_data(input_table, db_path, row_order=None)
        
        # Get available columns from metadata
        available_wgc_cols = df.attrs['available_wgc_cols']
        outcome_cols = df.attrs['outcome_cols']
        
        print(f"\n✓ Data loading complete")
        print(f"✓ Ready to analyze {len(available_wgc_cols)} weight gain causes")
        print(f"✓ Ready to analyze {len(outcome_cols)} outcomes: {outcome_cols}")
        
        # Step 2: Calculate effect sizes for dual outcomes
        print("\n" + "=" * 40)
        print("STEP 2: EFFECT SIZE CALCULATIONS")
        print("=" * 40)
        
        effect_sizes_df = calculate_effect_sizes(df, available_wgc_cols, outcome_cols)
        
        if effect_sizes_df.empty:
            print("❌ No effect sizes calculated - cannot proceed with analysis")
            return {
                'status': 'error',
                'error_message': 'No valid effect sizes calculated',
                'input_table': input_table
            }
        
        print(f"✓ Effect size calculations complete: {len(effect_sizes_df)} cause-outcome combinations")
        
        # Step 3: Perform statistical testing for dual outcomes
        print("\n" + "=" * 40)
        print("STEP 3: STATISTICAL TESTING")
        print("=" * 40)
        
        statistical_results_df = perform_statistical_tests(df, available_wgc_cols, outcome_cols)
        
        if statistical_results_df.empty:
            print("❌ No statistical tests completed - cannot proceed with analysis")
            return {
                'status': 'error',
                'error_message': 'No valid statistical tests completed',
                'input_table': input_table
            }
        
        print(f"✓ Statistical testing complete: {len(statistical_results_df)} cause-outcome combinations tested")
        
        # Step 4: Apply FDR correction separately for each outcome
        print("\n" + "=" * 40)
        print("STEP 4: FDR CORRECTION BY OUTCOME")
        print("=" * 40)
        
        statistical_results_with_fdr = apply_fdr_correction_to_results(statistical_results_df, outcome_cols)
        
        if statistical_results_with_fdr.empty:
            print("❌ FDR correction failed - cannot proceed with analysis")
            return {
                'status': 'error',
                'error_message': 'FDR correction failed',
                'input_table': input_table
            }
        
        print(f"✓ FDR correction complete: separate correction applied for each outcome")
        
        # Merge effect sizes with statistical results for forest plot generation
        print("\n⏳ Merging effect sizes with statistical results for visualization...")
        
        # Merge on 'cause' and 'outcome' columns
        merged_results = pd.merge(
            effect_sizes_df, 
            statistical_results_with_fdr[['cause', 'outcome', 'p_value', 'p_value_fdr', 'test_used']], 
            on=['cause', 'outcome'], 
            how='left'
        )
        
        if merged_results.empty:
            print("❌ Failed to merge effect sizes with statistical results")
            return {
                'status': 'error',
                'error_message': 'Failed to merge effect sizes with statistical results',
                'input_table': input_table
            }
        
        print(f"✓ Successfully merged {len(merged_results)} cause-outcome combinations")
        print(f"✓ Merged results contain both effect sizes and statistical test results")
        
        # Pipeline status update
        print("\n" + "=" * 40)
        print("PIPELINE STATUS")
        print("=" * 40)
        print("✓ Step 1: Data loading and validation - COMPLETE")
        print("✓ Step 2: Effect size calculations - COMPLETE")
        print("✓ Step 3: Statistical testing - COMPLETE")
        print("✓ Step 4: FDR correction - COMPLETE")
        
        # Step 5: Generate dual forest plots
        print("⏳ Step 5: Forest plot generation - STARTING")
        try:
            create_forest_plots(merged_results, output_base_dir, row_order=None)
            print("✓ Step 5: Forest plot generation - COMPLETE")
        except Exception as e:
            print(f"⚠ Step 5: Forest plot generation - FAILED: {str(e)}")
            raise
        
        # Step 6: Implement organized output structure
        print("\n" + "=" * 40)
        print("STEP 6: ORGANIZED OUTPUT STRUCTURE")
        print("=" * 40)
        
        try:
            output_files = export_comprehensive_summary_tables(
                merged_results, 
                statistical_results_with_fdr, 
                effect_sizes_df, 
                summary_tables_dir,
                input_table
            )
            print("✓ Step 6: Organized output structure - COMPLETE")
        except Exception as e:
            print(f"⚠ Step 6: Organized output structure - FAILED: {str(e)}")
            raise
        
        # Return comprehensive summary
        summary = {
            'status': 'complete',
            'completed_steps': ['data_loading', 'effect_sizes', 'statistical_testing', 'fdr_correction', 'visualization', 'output_organization'],
            'pending_steps': [],
            'input_table': input_table,
            'database_path': db_path,
            'output_directories': {
                'base': output_base_dir,
                'forest_plots': forest_plots_dir,
                'summary_tables': summary_tables_dir
            },
            'data_summary': {
                'n_records': len(df),
                'n_weight_gain_causes': len(available_wgc_cols),
                'n_outcomes': len(outcome_cols),
                'weight_gain_causes': available_wgc_cols,
                'outcomes': outcome_cols
            },
            'effect_sizes_summary': {
                'n_cause_outcome_combinations': len(effect_sizes_df),
                'outcomes_analyzed': effect_sizes_df['outcome'].unique().tolist(),
                'causes_analyzed': effect_sizes_df['cause'].unique().tolist()
            },
            'statistical_testing_summary': {
                'n_tests_performed': len(statistical_results_df),
                'outcomes_tested': statistical_results_df['outcome'].unique().tolist(),
                'causes_tested': statistical_results_df['cause'].unique().tolist(),
                'valid_tests': len(statistical_results_df[statistical_results_df['p_value'].notna()]),
                'failed_tests': len(statistical_results_df[statistical_results_df['p_value'].isna()]),
                'test_types_used': statistical_results_df['test_used'].value_counts().to_dict()
            },
            'fdr_correction_summary': {
                'n_tests_with_fdr': len(merged_results),
                'outcomes_corrected': merged_results['outcome'].unique().tolist(),
                'causes_corrected': merged_results['cause'].unique().tolist(),
                'valid_fdr_corrections': len(merged_results[merged_results['p_value_fdr'].notna()]),
                'failed_fdr_corrections': len(merged_results[merged_results['p_value_fdr'].isna()]),
                'fdr_summary_by_outcome': statistical_results_with_fdr.attrs.get('fdr_summary', {})
            },
            'visualization_summary': {
                'n_forest_plots_generated': len(merged_results['outcome'].unique()) * 2,  # 2 plot types per outcome
                'outcomes_visualized': merged_results['outcome'].unique().tolist(),
                'causes_visualized': merged_results['cause'].unique().tolist(),
                'plot_types': ['risk_ratios', 'risk_differences']
            },
            'output_files_summary': {
                'exported_files': output_files,
                'n_summary_tables': len(output_files),
                'summary_table_types': list(output_files.keys()),
                'forest_plots_directory': forest_plots_dir,
                'summary_tables_directory': summary_tables_dir
            }
        }
        
        print(f"\n✓ All implementation tasks complete")
        print(f"✓ Data loaded and validated for dual-outcome analysis")
        print(f"✓ Effect sizes calculated for all cause-outcome combinations")
        print(f"✓ Statistical testing completed with appropriate test selection")
        print(f"✓ FDR correction applied separately for each outcome")
        print(f"✓ Forest plots generated for all outcomes and effect measures")
        print(f"✓ Comprehensive summary tables exported with descriptive filenames")
        print(f"✓ Organized output structure implemented with error handling")
        
        return summary
        
    except Exception as e:
        error_msg = f"Error in descriptive visualizations pipeline: {str(e)}"
        print(f"\n❌ {error_msg}")
        
        return {
            'status': 'error',
            'error_message': error_msg,
            'input_table': input_table
        }


if __name__ == "__main__":
    # Example usage for testing
    print("Testing descriptive visualizations pipeline...")
    
    # Test with default parameters
    result = run_descriptive_visualizations("timetoevent_wgc_compl")
    
    print("\nTest result:")
    print(result)