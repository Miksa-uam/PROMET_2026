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
    suffix: str = '(FDR-corrected)'
) -> pd.DataFrame:
    """
    FINALIZED: Integrates FDR-corrected p-values back into the DataFrame,
    and correctly interleaves the new columns next to the original p-value columns.
    """
    df_copy = df.copy()
    
    # First, add all the new FDR columns to the DataFrame copy
    for original_col, corrected_pvals in corrections.items():
        fdr_col_name = f"{original_col} {suffix}"
        # Ensure the length matches the DataFrame index
        if len(corrected_pvals) == len(df_copy):
            df_copy[fdr_col_name] = [f"{p:.4f}" if pd.notna(p) else "N/A" for p in corrected_pvals]
        else:
            print(f"Warning: Length mismatch for column {original_col}. Skipping integration.")

    # Now, determine the final, interleaved column order
    new_order = []
    for col in df.columns:
        new_order.append(col)
        # If the current column is a p-value column, add its FDR version right after
        if col in corrections:
            fdr_col_name = f"{col} {suffix}"
            if fdr_col_name in df_copy.columns:
                new_order.append(fdr_col_name)

    # Ensure all new columns are included, even if logic missed something
    final_ordered_columns = new_order + [c for c in df_copy.columns if c not in new_order]
    
    return df_copy[final_ordered_columns]

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


def add_fdr_corrected_pvalues_to_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR correction to all p-value columns in a comparison table.
    
    This function identifies p-value columns (those ending with ': p-value'),
    applies FDR correction, and adds corrected columns adjacent to raw ones.
    
    Args:
        df (pd.DataFrame): DataFrame with p-value columns to correct
    
    Returns:
        pd.DataFrame: DataFrame with FDR-corrected p-value columns added
    """
    # Identify p-value columns
    pvalue_columns = [col for col in df.columns if col.endswith(': p-value')]
    
    if not pvalue_columns:
        logger.warning("No p-value columns found for FDR correction")
        return df.copy()
    
    # Collect p-values
    pvalue_dict = collect_pvalues_from_dataframe(df, pvalue_columns)
    
    # Apply FDR correction to each column
    corrections = {}
    for col, pvals in pvalue_dict.items():
        corrected_pvals = apply_fdr_correction(pvals)
        corrections[col] = corrected_pvals
    
    # Integrate corrected p-values
    df_corrected = integrate_corrected_pvalues(df, corrections)
    
    return df_corrected


def create_publication_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a publication-ready table with significance markers.
    
    This function takes a DataFrame with raw and FDR-corrected p-values
    and creates a clean table with significance markers (* and **).
    
    Args:
        df (pd.DataFrame): DataFrame with raw and corrected p-value columns
    
    Returns:
        pd.DataFrame: Publication-ready table with significance markers
    """
    pub_df = df.copy()
    
    # Find all p-value column pairs (raw and corrected)
    raw_pval_cols = [col for col in df.columns if col.endswith(': p-value')]
    corrected_pval_cols = [col for col in df.columns if col.endswith(': p-value (FDR-corrected)')]
    
    # Create significance markers for each column
    for raw_col in raw_pval_cols:
        corrected_col = raw_col + ' (FDR-corrected)'
        
        if corrected_col in corrected_pval_cols:
            # Create significance column
            sig_col = raw_col.replace(': p-value', '')
            
            # Add significance markers to the main value column
            if sig_col in pub_df.columns:
                for idx in pub_df.index:
                    raw_p = pd.to_numeric(pub_df.loc[idx, raw_col], errors='coerce')
                    corrected_p = pd.to_numeric(pub_df.loc[idx, corrected_col], errors='coerce')
                    
                    significance = ''
                    if not pd.isna(corrected_p) and corrected_p < 0.05:
                        significance = '**'  # FDR-corrected significant
                    elif not pd.isna(raw_p) and raw_p < 0.05:
                        significance = '*'   # Raw significant only
                    
                    if significance:
                        current_val = str(pub_df.loc[idx, sig_col])
                        pub_df.loc[idx, sig_col] = current_val + significance
    
    # Remove p-value columns from publication table
    cols_to_remove = raw_pval_cols + corrected_pval_cols
    pub_df = pub_df.drop(columns=cols_to_remove, errors='ignore')
    
    return pub_df