"""
WGC_VISUALIZATIONS.PY
Self-contained module for WGC-based descriptive analysis and visualization.

This module provides:
- Statistical comparisons (WGC group vs population mean)
- Visualization types (violin, stacked bar, forest)
- Integrated statistical testing with FDR correction
- Configurable labels and plot parameters
- Modular, task-specific functions for notebook use
- Forest plots with Risk Ratio and Risk Difference calculations

Adapted from cluster_descriptions.py for WGC-wise comparisons.

Author: Adapted from cluster_descriptions.py
Version: 1.0
"""

import os
import json
import sqlite3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_PALETTE = sns.color_palette("husl", 12)
POPULATION_COLOR = 'blueviolet'
WGC_COLOR = 'blue'  # Consistent color for WGC groups
ACHIEVED_COLOR = '#4361EE'
NOT_ACHIEVED_COLOR = '#EC5B57'

# WGC column names in the database
WGC_COLUMNS = [
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

# =============================================================================
# HELPER FUNCTIONS - Configuration Loading
# =============================================================================

def _resolve_path(path: str) -> str:
    """Resolve relative paths to absolute paths from script location."""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved = os.path.join(script_dir, path)
    if os.path.exists(resolved):
        return resolved
    if os.path.exists(path):
        return path
    return path

def load_name_map(json_path: str) -> Dict[str, str]:
    """Load variable name mappings from JSON file."""
    json_path = _resolve_path(json_path)
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: Name map file not found at '{json_path}'.")
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load name map file '{json_path}'. Error: {e}")
        return {}

def get_nice_name(variable: str, name_map: Dict[str, str]) -> str:
    """Get human-readable name for variable."""
    return name_map.get(variable, variable.replace('_', ' ').title())

def load_wgc_data(
    db_path: str,
    table_name: str = 'timetoevent_wgc_all'
) -> pd.DataFrame:
    """
    Load WGC data from database.
    
    Args:
        db_path: Path to database file
        table_name: Name of table containing WGC data
    
    Returns:
        DataFrame with WGC columns and outcome variables
    """
    print(f"Loading WGC data from {table_name}...")
    
    with sqlite3.connect(db_path) as conn:
        wgc_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    print(f"  ✓ Loaded {len(wgc_df)} records")
    
    # Verify WGC columns exist
    missing_wgc = [col for col in WGC_COLUMNS if col not in wgc_df.columns]
    if missing_wgc:
        print(f"  ⚠️ Warning: Missing WGC columns: {missing_wgc}")
    
    available_wgc = [col for col in WGC_COLUMNS if col in wgc_df.columns]
    print(f"  ✓ Available WGC columns: {len(available_wgc)}")
    
    return wgc_df

# =============================================================================
# HELPER FUNCTIONS - Statistical Testing
# =============================================================================

def mann_whitney_u_test(series1: pd.Series, series2: pd.Series) -> float:
    """Perform Mann-Whitney U test and return p-value."""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p_val = mannwhitneyu(s1, s2, alternative='two-sided')
        return p_val
    except ValueError:
        return 1.0

def chi_squared_test(series1: pd.Series, series2: pd.Series) -> float:
    """
    Perform Chi-squared test with Fisher's exact test fallback.
    """
    from scipy.stats import fisher_exact
    
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    
    contingency_table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return np.nan
    
    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        if (expected < 5).any():
            if contingency_table.shape == (2, 2):
                _, p_val = fisher_exact(contingency_table)
            else:
                print(f"    ⚠️ Low expected frequencies but table > 2x2, using chi-squared")
        
        return p_val
    except ValueError:
        return np.nan

def calculate_wgc_pvalues(
    wgc_df: pd.DataFrame,
    variable: str,
    wgc_columns: List[str],
    is_categorical: bool = False
) -> Dict[str, float]:
    """
    Calculate p-values for each WGC group vs entire population.
    
    Args:
        wgc_df: DataFrame with WGC columns and outcome variable
        variable: Variable to test
        wgc_columns: List of WGC column names
        is_categorical: Whether variable is categorical
    
    Returns:
        Dictionary mapping wgc_name to p-value
    """
    pvalues = {}
    
    test_type = "chi-squared (categorical)" if is_categorical else "Mann-Whitney U (continuous)"
    print(f"    Using {test_type} test for variable '{variable}'")
    
    population_data = wgc_df[variable]
    
    for wgc_col in wgc_columns:
        if wgc_col not in wgc_df.columns:
            continue
        
        # Get patients with this WGC
        wgc_subset = wgc_df[wgc_df[wgc_col] == 1][variable]
        
        if is_categorical:
            p_val = chi_squared_test(wgc_subset, population_data)
        else:
            p_val = mann_whitney_u_test(wgc_subset, population_data)
        
        pvalues[wgc_col] = p_val
    
    return pvalues

def apply_fdr_correction(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Apply FDR correction to p-values."""
    if not pvalues:
        return {}
    
    valid_items = [(k, v) for k, v in pvalues.items() if pd.notna(v)]
    if not valid_items:
        return pvalues
    
    keys, vals = zip(*valid_items)
    
    try:
        _, corrected, _, _ = multipletests(vals, method='fdr_bh')
        return dict(zip(keys, corrected))
    except:
        return pvalues

def _annotate_significance(
    ax: plt.Axes,
    x: float,
    y: float,
    p_raw: Optional[float],
    p_fdr: Optional[float],
    alpha: float
):
    """Add significance markers to plot."""
    text = ''
    if p_fdr is not None and p_fdr < alpha:
        text = '**'
    elif p_raw is not None and p_raw < alpha:
        text = '*'
    if text:
        ax.text(x, y, text, ha='center', va='bottom', fontsize=20, color='black', weight='bold', clip_on=False)

def calculate_risk_metrics(
    wgc_df: pd.DataFrame,
    outcome_variable: str,
    wgc_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate risk ratios and risk differences for binary outcome across WGC groups.
    
    Each WGC group is compared to the ENTIRE population.
    
    Args:
        wgc_df: DataFrame with WGC columns and binary outcome
        outcome_variable: Name of binary outcome variable (0/1)
        wgc_columns: List of WGC column names
    
    Returns:
        DataFrame with risk metrics for each WGC
    """
    results = []
    
    # Population risk (entire dataset)
    pop_events = wgc_df[outcome_variable].sum()
    pop_n = len(wgc_df)
    pop_risk = pop_events / pop_n if pop_n > 0 else 0
    
    for wgc_col in wgc_columns:
        if wgc_col not in wgc_df.columns:
            continue
        
        wgc_subset = wgc_df[wgc_df[wgc_col] == 1]
        
        # WGC group risk
        wgc_events = wgc_subset[outcome_variable].sum()
        wgc_n = len(wgc_subset)
        wgc_risk = wgc_events / wgc_n if wgc_n > 0 else 0
        
        # Risk Ratio
        if pop_risk > 0:
            rr = wgc_risk / pop_risk
            se_log_rr = np.sqrt((1/wgc_events - 1/wgc_n) + (1/pop_events - 1/pop_n)) if wgc_events > 0 and pop_events > 0 else np.nan
            if pd.notna(se_log_rr):
                rr_ci_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
                rr_ci_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)
            else:
                rr_ci_lower, rr_ci_upper = np.nan, np.nan
        else:
            rr, rr_ci_lower, rr_ci_upper = np.nan, np.nan, np.nan
        
        # Risk Difference
        rd = wgc_risk - pop_risk
        se_rd = np.sqrt((wgc_risk * (1 - wgc_risk) / wgc_n) + (pop_risk * (1 - pop_risk) / pop_n))
        rd_ci_lower = rd - 1.96 * se_rd
        rd_ci_upper = rd + 1.96 * se_rd
        
        results.append({
            'wgc_name': wgc_col,
            'risk_wgc': wgc_risk,
            'risk_population': pop_risk,
            'risk_ratio': rr,
            'rr_ci_lower': rr_ci_lower,
            'rr_ci_upper': rr_ci_upper,
            'risk_difference': rd,
            'rd_ci_lower': rd_ci_lower,
            'rd_ci_upper': rd_ci_upper
        })
    
    return pd.DataFrame(results)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def wgc_continuous_distributions(
    wgc_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    wgc_columns: Optional[List[str]] = None,
    name_map_path: str = 'human_readable_variable_names.json',
    calculate_significance: bool = True,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    variable_configs: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    Generate violin plots for continuous variables comparing WGC groups to population.
    
    Each WGC group is compared to the ENTIRE population.
    
    Args:
        wgc_df: DataFrame with WGC columns and outcomes
        variables: List of continuous variables to plot
        output_dir: Directory to save plots
        wgc_columns: List of WGC columns to include (default: all WGC_COLUMNS)
        name_map_path: Path to variable name mappings
        calculate_significance: Whether to calculate and display p-values
        fdr_correction: Whether to apply FDR correction
        alpha: Significance threshold
        title: Global title template
        ylabel: Global y-axis label
        xlabel: Global x-axis label
        variable_configs: Per-variable overrides
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    if wgc_columns is None:
        wgc_columns = [col for col in WGC_COLUMNS if col in wgc_df.columns]
    
    print(f"\nGenerating violin plots for {len(variables)} variables...")
    
    for variable in variables:
        print(f"  Processing: {variable}")
        
        if variable not in wgc_df.columns:
            print(f"    ⚠️ Variable '{variable}' not found in data. Skipping.")
            continue
        
        if wgc_df[variable].dropna().empty:
            print(f"    ⚠️ No valid data for '{variable}'. Skipping.")
            continue
        
        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_wgc_pvalues(wgc_df, variable, wgc_columns, is_categorical=False)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)
        
        var_config = variable_configs.get(variable, {}) if variable_configs else {}
        var_title = var_config.get('title', title)
        var_ylabel = var_config.get('ylabel', ylabel)
        var_xlabel = var_config.get('xlabel', xlabel)
        
        try:
            _plot_single_violin(
                wgc_df, variable, wgc_columns,
                name_map, output_dir,
                sig_raw, sig_fdr, alpha,
                var_title, var_ylabel, var_xlabel
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")

def _plot_single_violin(
    wgc_df: pd.DataFrame,
    variable: str,
    wgc_columns: List[str],
    name_map: Dict,
    output_dir: str,
    sig_raw: Optional[Dict],
    sig_fdr: Optional[Dict],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
):
    """Internal function to plot a single horizontal violin plot with all WGC groups."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    n_wgc = len(wgc_columns)
    
    # Population is the entire dataset
    population_data = wgc_df[variable].dropna()
    pop_median = population_data.median()
    pop_n = len(population_data)
    
    # Prepare data for all WGC groups in one plot
    plot_data_list = []
    stats_data = []
    
    for wgc_col in wgc_columns:
        wgc_label = get_nice_name(wgc_col, name_map)
        wgc_data = wgc_df[wgc_df[wgc_col] == 1][variable].dropna()
        
        # Population data for this WGC position
        for val in population_data:
            plot_data_list.append({
                'value': val,
                'wgc': wgc_label,
                'type': 'Population'
            })
        
        # WGC group data
        for val in wgc_data:
            plot_data_list.append({
                'value': val,
                'wgc': wgc_label,
                'type': 'WGC Group'
            })
        
        # Calculate stats
        wgc_median = wgc_data.median()
        wgc_n = len(wgc_data)
        
        p_raw = sig_raw.get(wgc_col) if sig_raw else None
        p_fdr = sig_fdr.get(wgc_col) if sig_fdr else None
        
        stats_data.append({
            'wgc_col': wgc_col,
            'wgc_label': wgc_label,
            'pop_n': pop_n,
            'wgc_n': wgc_n,
            'pop_median': pop_median,
            'wgc_median': wgc_median,
            'p_raw': p_raw,
            'p_fdr': p_fdr
        })
    
    plot_df = pd.DataFrame(plot_data_list)
    stats_df = pd.DataFrame(stats_data)
    
    # Create figure with proper margins
    fig = plt.figure(figsize=(max(14, n_wgc * 2), 10))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.75])
    
    # Create split violins
    wgc_labels_ordered = [get_nice_name(col, name_map) for col in wgc_columns]
    
    sns.violinplot(
        data=plot_df, x='wgc', y='value', hue='type',
        split=True, inner='quart',
        order=wgc_labels_ordered,
        palette={'Population': POPULATION_COLOR, 'WGC Group': WGC_COLOR},
        ax=ax
    )
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.95),
              title='', frameon=True, fancybox=True,
              fontsize=13, edgecolor='gray',
              facecolor='white', framealpha=0.8)
    
    # Get data range
    y_max = plot_df['value'].max()
    y_min = plot_df['value'].min()
    y_range = y_max - y_min
    
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    ax.set_xlim(-0.6, n_wgc - 0.4)
    ax.set_xticks(range(n_wgc))
    ax.set_xticklabels([])
    
    # Styling
    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f'Distribution of {nice_name}'
    plot_ylabel = ylabel.format(variable=nice_name) if ylabel else nice_name
    
    ax.set_ylabel(plot_ylabel, fontsize=16)
    ax.tick_params(labelsize=10)
    ax.set_xlabel('')
    
    fig_width, fig_height = fig.get_size_inches()
    
    # Add statistical boxes and annotations
    for i, row in stats_df.iterrows():
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig_width
        
        stats_text = f"Median: {row['wgc_median']:.1f}"
        fig.text(x_fig, 0.87, stats_text,
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Significance asterisks
        sig_text = ''
        if row['p_fdr'] is not None and pd.notna(row['p_fdr']) and row['p_fdr'] < alpha:
            sig_text = '**'
        elif row['p_raw'] is not None and pd.notna(row['p_raw']) and row['p_raw'] < alpha:
            sig_text = '*'
        
        if sig_text:
            fig.text(x_fig, 0.08, sig_text,
                    ha='center', va='center', fontsize=20, weight='bold')
        
        # P-values
        if row['p_raw'] is not None and pd.notna(row['p_raw']):
            p_text = f"p={row['p_raw']:.3f}"
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']):
                p_text += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.06, p_text,
                    ha='center', va='top', fontsize=12, style='italic')
        
        # WGC labels
        fig.text(x_fig, 0.02, f"{row['wgc_label']}\n(n={row['wgc_n']})",
                ha='right', va='top', fontsize=16, rotation=45)
    
    # Population stats box
    pop_stats_text = f"Population median: {pop_median:.1f}"
    fig.text(0.96, 0.87, pop_stats_text,
            ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Significance legend
    sig_legend = "Significance:\n* p < 0.05 (raw)\n** p < 0.05 (FDR-corrected)"
    fig.text(0.96, 0.74, sig_legend, ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.suptitle(plot_title, fontsize=20, weight='bold', y=0.96)
    
    # Save
    output_path = os.path.join(output_dir, f'{variable}_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def wgc_categorical_distributions(
    wgc_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    wgc_columns: Optional[List[str]] = None,
    name_map_path: str = 'human_readable_variable_names.json',
    calculate_significance: bool = True,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_labels: Optional[Dict[str, str]] = None,
    variable_configs: Optional[Dict[str, Dict[str, Any]]] = None
):
    """
    Generate stacked bar plots for categorical variables comparing WGC groups to population.
    
    Each WGC group is compared to the ENTIRE population.
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    if wgc_columns is None:
        wgc_columns = [col for col in WGC_COLUMNS if col in wgc_df.columns]
    
    print(f"\nGenerating stacked bar plots for {len(variables)} variables...")
    
    for variable in variables:
        print(f"  Processing: {variable}")
        
        if variable not in wgc_df.columns:
            print(f"    ⚠️ Variable '{variable}' not found in data. Skipping.")
            continue
        
        if wgc_df[variable].dropna().empty:
            print(f"    ⚠️ No valid data for '{variable}'. Skipping.")
            continue
        
        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_wgc_pvalues(wgc_df, variable, wgc_columns, is_categorical=True)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)
        
        var_config = variable_configs.get(variable, {}) if variable_configs else {}
        var_title = var_config.get('title', title)
        var_ylabel = var_config.get('ylabel', ylabel)
        var_xlabel = var_config.get('xlabel', xlabel)
        var_legend_labels = var_config.get('legend_labels', legend_labels)
        
        try:
            _plot_single_stacked_bar(
                wgc_df, variable, wgc_columns,
                name_map, output_dir,
                sig_raw, sig_fdr, alpha,
                var_title, var_ylabel, var_xlabel, var_legend_labels
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")

def _plot_single_stacked_bar(
    wgc_df: pd.DataFrame,
    variable: str,
    wgc_columns: List[str],
    name_map: Dict,
    output_dir: str,
    sig_raw: Optional[Dict],
    sig_fdr: Optional[Dict],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_labels: Optional[Dict] = None
):
    """Internal function to plot a single stacked bar chart."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data - population is entire dataset
    results = []
    pop_prop_1 = wgc_df[variable].mean() * 100
    pop_n_1 = wgc_df[variable].sum()
    pop_n_0 = len(wgc_df) - pop_n_1
    
    results.append({
        'group': 'Whole population',
        'prop_0': 100 - pop_prop_1,
        'prop_1': pop_prop_1,
        'n_0': int(pop_n_0),
        'n_1': int(pop_n_1),
        'n_total': len(wgc_df),
        'wgc_col': None,
        'p_raw': None,
        'p_fdr': None
    })
    
    for wgc_col in wgc_columns:
        wgc_subset = wgc_df[wgc_df[wgc_col] == 1]
        prop_1 = wgc_subset[variable].mean() * 100
        n_1 = wgc_subset[variable].sum()
        n_0 = len(wgc_subset) - n_1
        wgc_label = get_nice_name(wgc_col, name_map)
        
        results.append({
            'group': wgc_label,
            'prop_0': 100 - prop_1,
            'prop_1': prop_1,
            'n_0': int(n_0),
            'n_1': int(n_1),
            'n_total': len(wgc_subset),
            'wgc_col': wgc_col,
            'p_raw': sig_raw.get(wgc_col) if sig_raw else None,
            'p_fdr': sig_fdr.get(wgc_col) if sig_fdr else None
        })
    
    plot_df = pd.DataFrame(results)
    
    # Create figure
    fig = plt.figure(figsize=(max(14, len(plot_df) * 1.8), 10))
    ax = fig.add_axes([0.1, 0.25, 0.85, 0.60])
    
    x_pos = np.arange(len(plot_df))
    
    # Get legend labels
    if legend_labels:
        label_1 = legend_labels.get('achieved', 'Achieved')
        label_0 = legend_labels.get('not_achieved', 'Not Achieved')
    else:
        label_1 = 'Achieved'
        label_0 = 'Not Achieved'
    
    # Plot bars
    ax.bar(x_pos, plot_df['prop_1'], label=label_1, color=ACHIEVED_COLOR, alpha=0.8)
    ax.bar(x_pos, plot_df['prop_0'], bottom=plot_df['prop_1'], label=label_0, color=NOT_ACHIEVED_COLOR, alpha=0.8)
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.95),
              title='', frameon=True, fancybox=True,
              fontsize=13, edgecolor='gray',
              facecolor='white', framealpha=0.8)
    
    # Population mean line
    ax.axhline(y=pop_prop_1, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add significance asterisks
    for i, row in plot_df.iterrows():
        if row['wgc_col'] is not None:
            sig_text = ''
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']) and row['p_fdr'] < alpha:
                sig_text = '**'
            elif row['p_raw'] is not None and pd.notna(row['p_raw']) and row['p_raw'] < alpha:
                sig_text = '*'
            if sig_text:
                ax.text(i, pop_prop_1 + 1, sig_text, ha='center', va='bottom',
                       fontsize=20, weight='bold')
    
    # Set axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    
    # Styling
    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f'{nice_name}'
    plot_ylabel = ylabel if ylabel else 'Percentage (%)'
    plot_xlabel = xlabel if xlabel else 'Weight Gain Causes'
    
    ax.set_ylabel(plot_ylabel, fontsize=16)
    ax.tick_params(labelsize=10)
    
    # Add N (%) labels
    for i, row in plot_df.iterrows():
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig.get_size_inches()[0]
        
        top_label = f"{row['n_0']} ({row['prop_0']:.1f}%)"
        top_color = NOT_ACHIEVED_COLOR
        bottom_label = f"{row['n_1']} ({row['prop_1']:.1f}%)"
        bottom_color = ACHIEVED_COLOR
        
        fig.text(x_fig, 0.87, top_label,
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=top_color, alpha=0.8, edgecolor='gray'))
        
        fig.text(x_fig, 0.23, bottom_label,
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bottom_color, alpha=0.8, edgecolor='gray'))
        
        # P-values
        if row['wgc_col'] is not None and row['p_raw'] is not None and pd.notna(row['p_raw']):
            p_text = f"p={row['p_raw']:.3f}"
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']):
                p_text += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.2, p_text, ha='center', va='top',
                    fontsize=12, style='italic')
        
        # WGC labels
        fig.text(x_fig, 0.15, f"{row['group']}\n(n={row['n_total']})",
                ha='right', va='top', fontsize=16, rotation=45)
    
    # Population mean label
    fig.text(0.96, 0.67, f"- - - - - - Population average\n            ({pop_prop_1:.1f}%)",
            ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Significance legend
    sig_text = "Significance:\n* p < 0.05 (raw)\n** p < 0.05 (FDR-corrected)"
    fig.text(0.96, 0.74, sig_text, ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    fig.suptitle(plot_title, fontsize=20, weight='bold', y=0.96)
    
    # Save
    output_path = os.path.join(output_dir, f'{variable}_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_wgc_forest(
    wgc_df: pd.DataFrame,
    outcome_variable: str,
    output_dir: str,
    wgc_columns: Optional[List[str]] = None,
    name_map_path: str = 'human_readable_variable_names.json',
    effect_type: str = 'both',
    title_rr: Optional[str] = None,
    title_rd: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_rr: Optional[str] = None,
    xlabel_rd: Optional[str] = None
):
    """
    Generate forest plots for risk ratios and/or risk differences comparing WGC groups to population.
    
    Each WGC group is compared to the ENTIRE population.
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    if wgc_columns is None:
        wgc_columns = [col for col in WGC_COLUMNS if col in wgc_df.columns]
    
    print(f"\nGenerating forest plot(s) for '{outcome_variable}'...")
    
    if outcome_variable not in wgc_df.columns:
        print(f"  ⚠️ Variable '{outcome_variable}' not found in data. Skipping.")
        return
    
    if wgc_df[outcome_variable].dropna().empty:
        print(f"  ⚠️ No valid data for '{outcome_variable}'. Skipping.")
        return
    
    # Calculate risk metrics
    risk_df = calculate_risk_metrics(wgc_df, outcome_variable, wgc_columns)
    
    # Generate plots
    if effect_type in ['RR', 'both']:
        _plot_single_forest(
            risk_df, outcome_variable, 'RR', output_dir,
            name_map, title_rr, ylabel, xlabel_rr
        )
        print(f"  ✓ Risk Ratio plot saved")
    
    if effect_type in ['RD', 'both']:
        _plot_single_forest(
            risk_df, outcome_variable, 'RD', output_dir,
            name_map, title_rd, ylabel, xlabel_rd
        )
        print(f"  ✓ Risk Difference plot saved")

def _plot_single_forest(
    risk_df: pd.DataFrame,
    outcome_variable: str,
    plot_type: str,
    output_dir: str,
    name_map: Dict,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
):
    """Internal function to plot a single forest plot (RR or RD)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    

    # Apply sorting
    # Determine which column to sort by
    if plot_type == 'RR':
        sort_col = 'risk_ratio'
    else:
        sort_col = 'risk_difference'
    # ascending=True puts the smallest value at the BOTTOM and largest at the TOP.
    # ascending=False puts the largest value at the BOTTOM and smallest at the TOP.
    plot_data = risk_df.sort_values(by=sort_col, ascending=True)

    # Alternatively, sort by wgc_name
    # plot_data = risk_df.sort_values('wgc_name', ascending=True)
    
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(plot_data) * 0.5)),
                                   gridspec_kw={'width_ratios': [3, 0.01], 'wspace': 0.01})
    
    y_pos = np.arange(len(plot_data))
    
    # Select data based on plot type
    if plot_type == 'RR':
        effect = plot_data['risk_ratio']
        ci_lower = plot_data['rr_ci_lower']
        ci_upper = plot_data['rr_ci_upper']
        ref_value = 1
        default_xlabel = 'Risk ratio'
        default_title = f'Risk ratios: {get_nice_name(outcome_variable, name_map)}'
    else:  # RD
        effect = plot_data['risk_difference']
        ci_lower = plot_data['rd_ci_lower']
        ci_upper = plot_data['rd_ci_upper']
        ref_value = 0
        default_xlabel = 'Risk difference'
        default_title = f'Risk differences: {get_nice_name(outcome_variable, name_map)}'
    
    # Use consistent color for all WGC groups
    colors = [WGC_COLOR] * len(plot_data)
    
    # Plot error bars
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax.errorbar(
            x=effect.iloc[i],
            y=y_pos[i],
            xerr=[[effect.iloc[i] - ci_lower.iloc[i]], [ci_upper.iloc[i] - effect.iloc[i]]],
            fmt='o',
            color=colors[i],
            ecolor=colors[i],
            elinewidth=2,
            capsize=5,
            markersize=8
        )
    
    # Reference line
    ax.axvline(x=ref_value, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Get WGC labels for y-axis
    wgc_labels = [get_nice_name(wgc, name_map) for wgc in plot_data['wgc_name']]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wgc_labels, fontsize=16)
    
    # Configurable labels
    if title:
        plot_title = title
    else:
        plot_title = default_title
    
    if xlabel:
        plot_xlabel = xlabel
    else:
        plot_xlabel = default_xlabel
    
    ax.set_title(plot_title, fontsize=20, weight='bold')
    ax.set_xlabel(plot_xlabel, fontsize=18)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Right y-axis with effect sizes and CIs
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.grid(False)
    
    # Format CI labels
    ci_labels = []
    for _, row in plot_data.iterrows():
        if plot_type == 'RR':
            label = f"RR: {row['risk_ratio']:.2f} [{row['rr_ci_lower']:.2f}-{row['rr_ci_upper']:.2f}]"
        else:  # RD
            label = f"RD: {row['risk_difference']*100:.1f}% [{row['rd_ci_lower']*100:.1f}-{row['rd_ci_upper']*100:.1f}%]"
        ci_labels.append(label)
    
    ax2.set_yticklabels(ci_labels, fontsize=16, ha='left')
    ax2.set_ylabel('Effect Size [95% CI]', fontsize=18, labelpad=15)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', pad=0)
    ax2.set_xticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Save
    suffix = 'rr' if plot_type == 'RR' else 'rd'
    output_path = os.path.join(output_dir, f'{outcome_variable}_forest_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
