"""
FDR Correction Utilities Module

This module provides reusable False Discovery Rate (FDR) correction utilities
that can be applied across different analysis contexts including descriptive
comparisons, regression analyses, subgroup analyses, and any statistical
pipeline requiring multiple testing correction.

The module implements the Benjamini-Hochberg method for FDR correction using
statsmodels and provides robust error handling for various edge cases.

Functions:
    apply_fdr_correction: Apply Benjamini-Hochberg FDR correction to p-values
    collect_pvalues_from_dataframe: Extract p-values from DataFrame columns
    integrate_corrected_pvalues: Add corrected p-value columns to DataFrame
    format_pvalue_for_output: Format p-values for publication-ready output

Usage Examples:
    # Basic FDR correction
    corrected_pvals = apply_fdr_correction([0.05, 0.01, 0.001, 0.2])
    
    # DataFrame-based workflow
    pvals = collect_pvalues_from_dataframe(df, ['p_val_col1', 'p_val_col2'])
    corrections = {col: apply_fdr_correction(vals) for col, vals in pvals.items()}
    df_corrected = integrate_corrected_pvalues(df, corrections)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging
from statsmodels.stats.multitest import multipletests

# Configure logging
logger = logging.getLogger(__name__)


def apply_fdr_correction(
    p_values: List[float], 
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> List[float]:
    """
    Apply False Discovery Rate correction to a list of p-values using statsmodels.
    
    This function implements robust FDR correction with comprehensive error handling
    for various edge cases. It can be reused across different analysis contexts
    including descriptive comparisons, regression analyses, and subgroup analyses.
    
    Args:
        p_values (List[float]): List of p-values to correct. Can contain NaN values.
        method (str, optional): FDR correction method. Defaults to 'fdr_bh' 
                               (Benjamini-Hochberg).
        alpha (float, optional): Family-wise error rate. Defaults to 0.05.
    
    Returns:
        List[float]: FDR-corrected p-values. NaN values in input are preserved
                    as NaN in output. Returns original p-values if correction fails.
    
    Raises:
        None: All exceptions are caught and logged. Function returns original
              p-values on error to maintain analysis pipeline stability.
    
    Examples:
        >>> p_vals = [0.05, 0.01, 0.001, 0.2, np.nan]
        >>> corrected = apply_fdr_correction(p_vals)
        >>> # Returns corrected p-values with NaN preserved
        
        >>> # Use in regression analysis context
        >>> model_pvals = [model1.pvalues, model2.pvalues, model3.pvalues]
        >>> corrected_models = [apply_fdr_correction(pvals) for pvals in model_pvals]
    """
    try:
        # Convert to numpy array for easier handling
        p_array = np.array(p_values, dtype=float)
        
        # Handle edge case: empty input
        if len(p_array) == 0:
            logger.warning("FDR correction skipped: empty p-value list provided")
            return []
        
        # Identify valid (non-NaN) p-values
        valid_mask = ~np.isnan(p_array)
        valid_pvals = p_array[valid_mask]
        
        # Handle edge case: no valid p-values
        if len(valid_pvals) == 0:
            logger.warning("FDR correction skipped: no valid p-values found (all NaN)")
            return p_values.copy() if isinstance(p_values, list) else p_array.tolist()
        
        # Handle edge case: single p-value
        if len(valid_pvals) == 1:
            logger.info("FDR correction applied to single p-value (no adjustment needed)")
            return p_values.copy() if isinstance(p_values, list) else p_array.tolist()
        
        # Apply FDR correction using statsmodels
        rejected, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
            valid_pvals, 
            alpha=alpha, 
            method=method
        )
        
        # Create result array with original shape
        result = p_array.copy()
        result[valid_mask] = corrected_pvals
        
        logger.info(f"FDR correction applied successfully: {len(valid_pvals)} p-values corrected using {method}")
        
        return result.tolist()
        
    except Exception as e:
        logger.error(f"FDR correction failed: {str(e)}. Returning original p-values.")
        return p_values.copy() if isinstance(p_values, list) else p_values


def collect_pvalues_from_dataframe(
    df: pd.DataFrame, 
    pvalue_columns: List[str]
) -> Dict[str, List[float]]:
    """
    Extract p-values from specified columns in a DataFrame.
    
    This generic utility function can be used across any analysis that stores
    p-values in DataFrame columns, including descriptive comparisons, regression
    results, subgroup analyses, and sensitivity analyses.
    
    Args:
        df (pd.DataFrame): DataFrame containing p-value columns
        pvalue_columns (List[str]): List of column names containing p-values
    
    Returns:
        Dict[str, List[float]]: Dictionary mapping column names to lists of p-values.
                               Missing columns are skipped with a warning.
    
    Examples:
        >>> # Descriptive comparisons context
        >>> pval_cols = ['Cohort comparison: p-value', 'Age: p-value', 'Gender: p-value']
        >>> pvals = collect_pvalues_from_dataframe(results_df, pval_cols)
        
        >>> # Regression analysis context  
        >>> reg_cols = ['model1_pval', 'model2_pval', 'model3_pval']
        >>> reg_pvals = collect_pvalues_from_dataframe(regression_df, reg_cols)
        
        >>> # Subgroup analysis context
        >>> subgroup_cols = ['subgroup_A_pval', 'subgroup_B_pval']
        >>> sub_pvals = collect_pvalues_from_dataframe(subgroup_df, subgroup_cols)
    """
    pvalue_dict = {}
    
    for col in pvalue_columns:
        if col in df.columns:
            # Extract p-values, converting to float and handling non-numeric values
            try:
                pvals = pd.to_numeric(df[col], errors='coerce').tolist()
                pvalue_dict[col] = pvals
                logger.debug(f"Collected {len(pvals)} p-values from column '{col}'")
            except Exception as e:
                logger.warning(f"Failed to extract p-values from column '{col}': {str(e)}")
        else:
            logger.warning(f"P-value column '{col}' not found in DataFrame. Skipping.")
    
    logger.info(f"Collected p-values from {len(pvalue_dict)} columns")
    return pvalue_dict


def integrate_corrected_pvalues(
    df: pd.DataFrame, 
    corrections: Dict[str, List[float]], 
    suffix: str = " (FDR-corrected)"
) -> pd.DataFrame:
    """
    Add FDR-corrected p-value columns to DataFrame immediately after their corresponding raw columns.
    
    This function inserts corrected p-value columns right after their raw counterparts,
    making it easier to compare raw vs. corrected values in the output tables.
    
    Args:
        df (pd.DataFrame): Original DataFrame to add corrected columns to
        corrections (Dict[str, List[float]]): Dictionary mapping original column 
                                            names to corrected p-values
        suffix (str, optional): Suffix to append to original column names for 
                               corrected columns. Defaults to " (FDR-corrected)".
    
    Returns:
        pd.DataFrame: DataFrame with corrected p-value columns inserted immediately 
                     after their corresponding raw columns
    
    Examples:
        >>> # Before: ['Variable', 'Group1', 'Group2', 'Age: p-value', 'Sex: p-value']
        >>> df_corrected = integrate_corrected_pvalues(df, corrections)
        >>> # After: ['Variable', 'Group1', 'Group2', 'Age: p-value', 'Age: p-value (FDR-corrected)', 'Sex: p-value', 'Sex: p-value (FDR-corrected)']
    """
    df_result = df.copy()
    
    # Keep track of columns we've processed to avoid duplicates
    processed_columns = set()
    
    # Process corrections in the order they appear in the original DataFrame
    original_columns = list(df_result.columns)
    
    for original_col in original_columns:
        if original_col in corrections and original_col not in processed_columns:
            corrected_vals = corrections[original_col]
            corrected_col = original_col + suffix
            
            try:
                # Ensure corrected values list matches DataFrame length
                if len(corrected_vals) != len(df_result):
                    logger.warning(
                        f"Length mismatch for column '{original_col}': "
                        f"DataFrame has {len(df_result)} rows, "
                        f"corrected values has {len(corrected_vals)} values. "
                        f"Skipping integration."
                    )
                    continue
                
                # Find the position of the original column
                original_col_idx = df_result.columns.get_loc(original_col)
                
                # Insert the corrected column right after the original column
                # First, add the corrected column to the end
                df_result[corrected_col] = corrected_vals
                
                # Then reorder columns to put the corrected column right after the original
                cols = list(df_result.columns)
                
                # Remove the corrected column from its current position (end)
                cols.remove(corrected_col)
                
                # Insert it right after the original column
                cols.insert(original_col_idx + 1, corrected_col)
                
                # Reorder the DataFrame
                df_result = df_result[cols]
                
                processed_columns.add(original_col)
                logger.debug(f"Added corrected p-value column '{corrected_col}' after '{original_col}'")
                
            except Exception as e:
                logger.error(f"Failed to integrate corrected p-values for column '{original_col}': {str(e)}")
    
    # Handle any remaining corrections that weren't in the original column order
    for original_col, corrected_vals in corrections.items():
        if original_col not in processed_columns:
            corrected_col = original_col + suffix
            
            try:
                if len(corrected_vals) != len(df_result):
                    logger.warning(
                        f"Length mismatch for column '{original_col}': "
                        f"DataFrame has {len(df_result)} rows, "
                        f"corrected values has {len(corrected_vals)} values. "
                        f"Skipping integration."
                    )
                    continue
                
                # If original column doesn't exist, just add to the end
                df_result[corrected_col] = corrected_vals
                logger.debug(f"Added corrected p-value column '{corrected_col}' at end (original column not found)")
                
            except Exception as e:
                logger.error(f"Failed to integrate corrected p-values for column '{original_col}': {str(e)}")
    
    added_cols = len([col for col in df_result.columns if suffix in col])
    logger.info(f"Integrated {added_cols} FDR-corrected p-value columns adjacent to their raw counterparts")
    
    return df_result


def format_pvalue_for_output(
    p_value: float, 
    threshold: float = 0.001,
    decimal_places: int = 3
) -> str:
    """
    Format p-values for publication-ready output.
    
    This utility function provides consistent p-value formatting across all
    analysis outputs including regression tables, comparison tables, and
    summary statistics. Useful for creating publication-ready results.
    
    Args:
        p_value (float): P-value to format
        threshold (float, optional): Threshold below which to show '<threshold'. 
                                   Defaults to 0.001.
        decimal_places (int, optional): Number of decimal places for regular p-values.
                                      Defaults to 3.
    
    Returns:
        str: Formatted p-value string (e.g., '<0.001', '0.045', 'N/A')
    
    Examples:
        >>> format_pvalue_for_output(0.0001)
        '<0.001'
        >>> format_pvalue_for_output(0.045)
        '0.045'
        >>> format_pvalue_for_output(np.nan)
        'N/A'
        
        >>> # Custom formatting for different contexts
        >>> format_pvalue_for_output(0.0001, threshold=0.01, decimal_places=2)
        '<0.01'
    """
    if pd.isna(p_value):
        return 'N/A'
    
    try:
        p_val = float(p_value)
        
        if p_val < threshold:
            return f'<{threshold}'
        else:
            return f'{p_val:.{decimal_places}f}'
            
    except (ValueError, TypeError):
        return 'N/A'