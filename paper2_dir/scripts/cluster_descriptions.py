"""
CLUSTER_DESCRIPTIONS.PY
Self-contained module for cluster-based descriptive analysis and visualization.

This module provides:
- Statistical comparisons (cluster vs population mean)
- All visualization types (violin, stacked bar, lollipop, forest, heatmap)
- Integrated statistical testing with FDR correction
- Configurable labels, colors, and plot parameters (Option A2: individual + per-variable)
- Modular, task-specific functions for notebook use
- Forest plots with Risk Ratio and Risk Difference calculations
- Lollipop plots with significance markers

Author: Refactored from descriptive_visualizations.py
Version: 2.0
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
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_PALETTE = sns.color_palette("husl", 12)
POPULATION_COLOR = 'blueviolet'
CLUSTER_COLOR = 'aquamarine'
ACHIEVED_COLOR = '#4361EE'
NOT_ACHIEVED_COLOR = '#EC5B57'


# =============================================================================
# HELPER FUNCTIONS - Configuration Loading
# =============================================================================

def _resolve_path(path: str) -> str:
    """Resolve relative paths to absolute paths from script location."""
    if os.path.isabs(path):
        return path
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve relative to script directory
    resolved = os.path.join(script_dir, path)
    if os.path.exists(resolved):
        return resolved
    # Try relative to current working directory
    if os.path.exists(path):
        return path
    return path  # Return original if nothing works

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

def load_cluster_config(json_path: str) -> Dict[str, Dict[str, str]]:
    """Load cluster labels and colors from JSON file."""
    json_path = _resolve_path(json_path)
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: Cluster config file not found at '{json_path}'. Using defaults.")
        return {'cluster_labels': {}, 'cluster_colors': {}}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load cluster config '{json_path}'. Error: {e}")
        return {'cluster_labels': {}, 'cluster_colors': {}}

def get_nice_name(variable: str, name_map: Dict[str, str]) -> str:
    """Get human-readable name for variable."""
    return name_map.get(variable, variable.replace('_', ' ').title())

def get_cluster_label(cluster_id: int, cluster_config: Dict) -> str:
    """Get human-readable label for cluster."""
    labels = cluster_config.get('cluster_labels', {})
    # Convert cluster_id to string for lookup
    return labels.get(str(cluster_id), f'Cluster {cluster_id}')

def get_cluster_color(cluster_id: int, cluster_config: Dict, default_palette: list) -> str:
    """Get color for cluster."""
    colors = cluster_config.get('cluster_colors', {})
    # Convert cluster_id to string for lookup
    if str(cluster_id) in colors:
        return colors[str(cluster_id)]
    return default_palette[cluster_id % len(default_palette)]

def _extract_percentage_from_table_cell(cell_value: str) -> float:
    """
    Extract percentage from table cell format 'N (X.X%)'.
    
    Args:
        cell_value: String in format like "123 (45.6%)"
    
    Returns:
        Float percentage value, or np.nan if parsing fails
    """
    import re
    try:
        match = re.search(r'\((\d+\.?\d*)%\)', str(cell_value))
        if match:
            return float(match.group(1))
        return np.nan
    except (ValueError, AttributeError):
        return np.nan

def _parse_cluster_column_header(header: str, cluster_config: Dict) -> Optional[int]:
    """
    Extract cluster ID from column header.
    
    Supports both new format "[Cluster Name]: Mean (±SD) / N (%)" 
    and old format "Cluster X: Mean/N".
    
    Args:
        header: Column header string
        cluster_config: Cluster configuration dictionary
    
    Returns:
        Cluster ID as integer, or None if parsing fails
    """
    import re
    
    # Try new format: "[Cluster Name]: Mean (±SD) / N (%)"
    # Extract the label part before the colon
    match = re.match(r'^(.+?):\s*Mean', header)
    if match:
        label = match.group(1).strip()
        # Look up this label in cluster_config to find the ID
        labels = cluster_config.get('cluster_labels', {})
        for cluster_id_str, cluster_label in labels.items():
            if cluster_label == label:
                return int(cluster_id_str)
    
    # Fallback to old format: "Cluster X: Mean/N"
    match = re.match(r'^Cluster\s+(\d+):', header)
    if match:
        return int(match.group(1))
    
    return None

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
    Uses Fisher's exact test when expected frequencies are < 5.
    """
    from scipy.stats import fisher_exact
    
    s1 = series1.dropna()
    s2 = series2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    
    # Create contingency table
    contingency_table = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return np.nan
    
    try:
        # Check expected frequencies
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        # Use Fisher's exact test if any expected frequency < 5
        if (expected < 5).any():
            # Fisher's exact test only works for 2x2 tables
            if contingency_table.shape == (2, 2):
                _, p_val = fisher_exact(contingency_table)
            # For larger tables, use chi-squared with warning
            else:
                print(f"    ⚠️ Low expected frequencies but table > 2x2, using chi-squared")
        
        return p_val
    except ValueError:
        return np.nan

def _infer_variable_type(series: pd.Series, threshold: int = 10) -> str:
    """
    Infer if variable is continuous or categorical based on unique values.
    
    Args:
        series: Data series to analyze
        threshold: Maximum unique values for categorical classification (default: 10)
    
    Returns:
        'continuous' or 'categorical'
    
    Edge cases:
        - Empty series: returns 'categorical' (safe default)
        - All NaN values: returns 'categorical' (safe default)
        - Single unique value: returns 'categorical'
    """
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return 'categorical'  # Safe default for empty/all-NaN series
    
    unique_count = clean_series.nunique()
    return 'categorical' if unique_count <= threshold else 'continuous'

def calculate_cluster_pvalues(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str = 'cluster_id',
    is_categorical: bool = False
    ) -> Dict[int, float]:
    """
    Calculate p-values for each cluster vs entire clustered population.
    
    Each cluster is compared to ALL data (entire clustered population),
    not to an external population.
    
    Returns:
        Dictionary mapping cluster_id to p-value
    """
    pvalues = {}
    clusters = sorted(cluster_df[cluster_col].unique())
    
    # Print test type for transparency
    test_type = "chi-squared (categorical)" if is_categorical else "Mann-Whitney U (continuous)"
    print(f"    Using {test_type} test for variable '{variable}'")
    
    # The "population" is the entire clustered dataset
    population_data = cluster_df[variable]
    
    for cluster_id in clusters:
        cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id][variable]
        
        if is_categorical:
            p_val = chi_squared_test(cluster_subset, population_data)
        else:
            p_val = mann_whitney_u_test(cluster_subset, population_data)
        
        pvalues[cluster_id] = p_val
    
    return pvalues

def apply_fdr_correction(pvalues: Dict[int, float]
    ) -> Dict[int, float]:
    """Apply FDR correction to p-values."""
    if not pvalues:
        return {}
    
    # Extract valid p-values
    valid_items = [(k, v) for k, v in pvalues.items() if pd.notna(v)]
    if not valid_items:
        return pvalues
    
    keys, vals = zip(*valid_items)
    
    # Apply FDR correction
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

def extract_pvalues_for_lollipop(
    results_df: pd.DataFrame,
    variables: List[str],
    cluster_df: pd.DataFrame,
    cluster_col: str = 'cluster_id',
    cluster_config_path: str = 'cluster_config.json',
    name_map_path: str = 'human_readable_variable_names.json'
    ) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Helper function to extract p-values from analyze_cluster_vs_population results.
    
    This simplifies the process of getting p-values for lollipop plots.
    
    Args:
        results_df: DataFrame returned by analyze_cluster_vs_population
        variables: List of RAW variable names (snake_case) to extract p-values for
        cluster_df: Original cluster DataFrame (to get cluster IDs)
        cluster_col: Column name for cluster IDs
        cluster_config_path: Path to cluster configuration (for label lookup)
        name_map_path: Path to variable name mappings (for nice name lookup)
    
    Returns:
        Tuple of (pvalues_raw, pvalues_fdr) dictionaries
        Each dict maps {raw_variable_name: {cluster_id: p_value}}
    
    Example:
        >>> results_df = analyze_cluster_vs_population(...)
        >>> pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
        ...     results_df, ['age', 'bmi'], cluster_df
        ... )
        >>> plot_cluster_lollipop(..., pvalues_raw=pvalues_raw, pvalues_fdr=pvalues_fdr)
    """
    cluster_config = load_cluster_config(cluster_config_path)
    name_map = load_name_map(name_map_path)
    clusters = sorted(cluster_df[cluster_col].unique())
    
    pvalues_raw = {}
    pvalues_fdr = {}
    
    for var in variables:
        pvalues_raw[var] = {}
        pvalues_fdr[var] = {}
        
        # Convert raw variable name to nice name for lookup in results_df
        nice_name = get_nice_name(var, name_map)
        
        # Get row for this variable using nice name
        var_row = results_df[results_df['Variable'] == nice_name]
        
        if var_row.empty:
            print(f"  ⚠️ Warning: Variable '{var}' not found in results_df")
            continue
        
        # Extract p-values for each cluster
        for cluster_id in clusters:
            # Try new format with cluster labels first
            cluster_label = get_cluster_label(cluster_id, cluster_config)
            p_raw_col = f'{cluster_label}: p-value'
            p_fdr_col = f'{cluster_label}: p-value (FDR-corrected)'
            
            # Fallback to old format if new format not found
            if p_raw_col not in var_row.columns:
                p_raw_col = f'Cluster {cluster_id}: p-value'
            if p_fdr_col not in var_row.columns:
                p_fdr_col = f'Cluster {cluster_id}: p-value (FDR-corrected)'
            
            if p_raw_col in var_row.columns:
                pvalues_raw[var][cluster_id] = var_row[p_raw_col].values[0]
            
            if p_fdr_col in var_row.columns:
                pvalues_fdr[var][cluster_id] = var_row[p_fdr_col].values[0]
    
    print(f"✓ Extracted p-values for {len(pvalues_raw)} variables across {len(clusters)} clusters")
    
    return pvalues_raw, pvalues_fdr

def calculate_risk_metrics(
    cluster_df: pd.DataFrame,
    outcome_variable: str,
    cluster_col: str = 'cluster_id'
    ) -> pd.DataFrame:
    """
    Calculate risk ratios and risk differences for binary outcome across clusters.
    
    Each cluster is compared to the ENTIRE clustered population (all clusters combined).
    
    Args:
        cluster_df: DataFrame with cluster assignments and binary outcome
        outcome_variable: Name of binary outcome variable (0/1)
        cluster_col: Column name for cluster IDs
    
    Returns:
        DataFrame with columns: cluster_id, risk_cluster, risk_population, 
                               risk_ratio, rr_ci_lower, rr_ci_upper,
                               risk_difference, rd_ci_lower, rd_ci_upper
    """
    from scipy.stats import norm
    
    clusters = sorted(cluster_df[cluster_col].unique())
    results = []
    
    # Population risk (entire clustered dataset)
    pop_events = cluster_df[outcome_variable].sum()
    pop_n = len(cluster_df)
    pop_risk = pop_events / pop_n if pop_n > 0 else 0
    
    for cluster_id in clusters:
        cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id]
        
        # Cluster risk
        cluster_events = cluster_subset[outcome_variable].sum()
        cluster_n = len(cluster_subset)
        cluster_risk = cluster_events / cluster_n if cluster_n > 0 else 0
        
        # Risk Ratio
        if pop_risk > 0:
            rr = cluster_risk / pop_risk
            # 95% CI for RR using log transformation
            se_log_rr = np.sqrt((1/cluster_events - 1/cluster_n) + (1/pop_events - 1/pop_n)) if cluster_events > 0 and pop_events > 0 else np.nan
            if pd.notna(se_log_rr):
                rr_ci_lower = np.exp(np.log(rr) - 1.96 * se_log_rr)
                rr_ci_upper = np.exp(np.log(rr) + 1.96 * se_log_rr)
            else:
                rr_ci_lower, rr_ci_upper = np.nan, np.nan
        else:
            rr, rr_ci_lower, rr_ci_upper = np.nan, np.nan, np.nan
        
        # Risk Difference
        rd = cluster_risk - pop_risk
        # 95% CI for RD
        se_rd = np.sqrt((cluster_risk * (1 - cluster_risk) / cluster_n) + (pop_risk * (1 - pop_risk) / pop_n))
        rd_ci_lower = rd - 1.96 * se_rd
        rd_ci_upper = rd + 1.96 * se_rd
        
        results.append({
            'cluster_id': cluster_id,
            'risk_cluster': cluster_risk,
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

def cluster_continuous_distributions(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
    calculate_significance: bool = True,
    fdr_correction: bool = True,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    variable_configs: Optional[Dict[str, Dict[str, Any]]] = None):
    """
    Generate violin plots for continuous variables comparing clusters to clustered population.
    
    Each cluster is compared to the ENTIRE clustered population (all clusters combined).
    
    Args:
        cluster_df: DataFrame with cluster assignments and outcomes
        variables: List of continuous variables to plot
        output_dir: Directory to save plots
        cluster_col: Column name for cluster IDs
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration
        calculate_significance: Whether to calculate and display p-values
        fdr_correction: Whether to apply FDR correction
        alpha: Significance threshold
        title: Global title template (can use {variable} placeholder)
        ylabel: Global y-axis label (can use {variable} placeholder)
        xlabel: Global x-axis label
        variable_configs: Per-variable overrides dict with keys matching variable names,
                         values are dicts with 'title', 'ylabel', 'xlabel' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    print(f"\nGenerating violin plots for {len(variables)} variables...")
    
    for variable in variables:
        print(f"  Processing: {variable}")
        
        # Validate columns
        if variable not in cluster_df.columns:
            print(f"    ⚠️ Variable '{variable}' not found in data. Skipping.")
            continue
        
        # Check for empty data
        if cluster_df[variable].dropna().empty:
            print(f"    ⚠️ No valid data for '{variable}'. Skipping.")
            continue
        
        # Calculate significance if requested
        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_cluster_pvalues(cluster_df, variable, cluster_col, is_categorical=False)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)
        
        # Get variable-specific config
        var_config = variable_configs.get(variable, {}) if variable_configs else {}
        var_title = var_config.get('title', title)
        var_ylabel = var_config.get('ylabel', ylabel)
        var_xlabel = var_config.get('xlabel', xlabel)
        
        # Generate plot
        try:
            _plot_single_violin(
                cluster_df, variable, cluster_col,
                name_map, cluster_config, output_dir,
                sig_raw, sig_fdr, alpha,
                var_title, var_ylabel, var_xlabel
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")

def _plot_single_violin(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str,
    name_map: Dict,
    cluster_config: Dict,
    output_dir: str,
    sig_raw: Optional[Dict],
    sig_fdr: Optional[Dict],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None):
    """Internal function to plot a single horizontal violin plot with all clusters."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    clusters = sorted(cluster_df[cluster_col].unique())
    n_clusters = len(clusters)
    
    # Population is the entire clustered dataset
    population_data = cluster_df[variable].dropna()
    pop_median = population_data.median()
    pop_q1 = population_data.quantile(0.25)
    pop_q3 = population_data.quantile(0.75)
    pop_n = len(population_data)
    
    # Prepare data for all clusters in one plot
    plot_data_list = []
    stats_data = []
    
    for cluster_id in clusters:
        cluster_label = get_cluster_label(cluster_id, cluster_config)
        cluster_color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
        cluster_data = cluster_df[cluster_df[cluster_col] == cluster_id][variable].dropna()
        
        # Population data for this cluster position
        for val in population_data:
            plot_data_list.append({
                'value': val,
                'cluster': cluster_label,
                'type': 'Population'
            })
        
        # Cluster data
        for val in cluster_data:
            plot_data_list.append({
                'value': val,
                'cluster': cluster_label,
                'type': 'Cluster'
            })
        
        # Calculate stats
        cluster_median = cluster_data.median()
        cluster_q1 = cluster_data.quantile(0.25)
        cluster_q3 = cluster_data.quantile(0.75)
        cluster_n = len(cluster_data)
        
        p_raw = sig_raw.get(cluster_id) if sig_raw else None
        p_fdr = sig_fdr.get(cluster_id) if sig_fdr else None
        
        stats_data.append({
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_color': cluster_color,
            'pop_n': pop_n,
            'cluster_n': cluster_n,
            'pop_median': pop_median,
            'pop_q1': pop_q1,
            'pop_q3': pop_q3,
            'cluster_median': cluster_median,
            'cluster_q1': cluster_q1,
            'cluster_q3': cluster_q3,
            'p_raw': p_raw,
            'p_fdr': p_fdr
        })
    
    plot_df = pd.DataFrame(plot_data_list)
    stats_df = pd.DataFrame(stats_data)
    
    # Create figure with proper margins for outer layers
    fig = plt.figure(figsize=(max(14, n_clusters * 2), 10))
    # Adjust subplot to leave room: [left, bottom, width, height]
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.75])  # Adjust padding for labels
    
    # Create split violins for each cluster
    cluster_labels_ordered = [get_cluster_label(cid, cluster_config) for cid in clusters]
    
    # Plot using seaborn with custom palette
    palette_dict = {}
    for _, row in stats_df.iterrows():
        palette_dict[row['cluster_label']] = {
            'Population': POPULATION_COLOR,
            'Cluster': row['cluster_color']
        }
    
    # Create the plot
    sns.violinplot(
        data=plot_df, x= 'cluster', y='value', hue='type', # hue = 'type' will automatically create a type legend (Population vs Cluster)
        split=True, inner='quart', 
        order=cluster_labels_ordered,
        palette={'Population': POPULATION_COLOR, 'Cluster': CLUSTER_COLOR},  # Will override per cluster
        ax=ax
    )

    # Set the location of the 'type' box, containing color codes for the split violins
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.95), 
          title='', frameon=True, fancybox=True,
          fontsize=13, 
          edgecolor='black',
          facecolor='white',
          framealpha=0.8
          )
    
    # Get data range for violin plot
    y_max = plot_df['value'].max()
    y_min = plot_df['value'].min()
    y_range = y_max - y_min
    
    # Set axis limits with padding for violins
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    ax.set_xlim(-0.6, n_clusters - 0.4)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([])
    
    # Styling
    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f'Distribution of {nice_name}'
    plot_ylabel = ylabel.format(variable=nice_name) if ylabel else nice_name
    
    ax.set_ylabel(plot_ylabel, fontsize=16)
    ax.tick_params(labelsize=10)

    # Remove automatic x axis label
    ax.set_xlabel('')
    
    # Get figure dimensions for coordinate conversion
    fig_width, fig_height = fig.get_size_inches()
    
    # MIDDLE LAYER: Statistical boxes, asterisks, p-values (in figure coordinates)
    for i, row in stats_df.iterrows():
        # Convert data x-coord to figure coord
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig_width

        stats_text = f"n={row['cluster_n']}\n Median: {row['cluster_median']:.1f}"
        fig.text(x_fig, 0.87, stats_text, 
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Significance asterisks below violins
        sig_text = ''
        if row['p_fdr'] is not None and pd.notna(row['p_fdr']) and row['p_fdr'] < alpha:
            sig_text = '**'
        elif row['p_raw'] is not None and pd.notna(row['p_raw']) and row['p_raw'] < alpha:
            sig_text = '*'
        
        if sig_text:
            fig.text(x_fig, 0.08, sig_text, 
                    ha='center', va='center', fontsize=20, weight='bold')
        
        # P-values below asterisks (NO "cluster" label)
        if row['p_raw'] is not None and pd.notna(row['p_raw']):
            p_text = f"p={row['p_raw']:.3f}"
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']):
                p_text += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.06, p_text, 
                    ha='center', va='top', fontsize=12, style='italic')
        
        # Cluster labels below p-values
        fig.text(x_fig, 0.02, f"{row['cluster_label']}\n(n={row['cluster_n']})", 
                ha='right', va='top', fontsize=16, rotation=45)
    
    # X-axis label at outermost position (below all cluster labels)
    # fig.text(0.5, 0.001, 'cluster', ha='center', fontsize=12, weight='bold')
    
    # Population n and median box
    pop_stats_text = f"Population n={row['pop_n']}\nPopulation median: {row['pop_median']:.1f}"
    fig.text(0.96, 0.87, pop_stats_text, 
            ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Significance legend box
    sig_legend = "Significance:\n* p < 0.05 (raw)\n** p < 0.05 (FDR-corrected)"
    fig.text(0.96, 0.74, sig_legend, ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # FRAME: Title
    fig.suptitle(plot_title, fontsize=20, weight='bold', y=0.96)
    
    # Save
    output_path = os.path.join(output_dir, f'{variable}_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def cluster_categorical_distributions(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_dir: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
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
    Generate stacked bar plots for categorical variables comparing clusters to clustered population.
    
    Each cluster is compared to the ENTIRE clustered population (all clusters combined).
    
    Args:
        cluster_df: DataFrame with cluster assignments and outcomes
        variables: List of binary variables to plot
        output_dir: Directory to save plots
        cluster_col: Column name for cluster IDs
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration
        calculate_significance: Whether to calculate and display p-values
        fdr_correction: Whether to apply FDR correction
        alpha: Significance threshold
        title: Global title template (can use {variable} placeholder)
        ylabel: Global y-axis label
        xlabel: Global x-axis label
        legend_labels: Global legend labels (e.g., {'achieved': 'Female', 'not_achieved': 'Male'})
        variable_configs: Per-variable overrides dict with keys matching variable names,
                         values are dicts with 'title', 'ylabel', 'xlabel', 'legend_labels' keys
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    print(f"\nGenerating stacked bar plots for {len(variables)} variables...")
    
    for variable in variables:
        print(f"  Processing: {variable}")
        
        # Validate columns
        if variable not in cluster_df.columns:
            print(f"    ⚠️ Variable '{variable}' not found in data. Skipping.")
            continue
        
        # Check for empty data
        if cluster_df[variable].dropna().empty:
            print(f"    ⚠️ No valid data for '{variable}'. Skipping.")
            continue
        
        # Calculate significance if requested
        sig_raw, sig_fdr = None, None
        if calculate_significance:
            sig_raw = calculate_cluster_pvalues(cluster_df, variable, cluster_col, is_categorical=True)
            if fdr_correction:
                sig_fdr = apply_fdr_correction(sig_raw)
        
        # Get variable-specific config
        var_config = variable_configs.get(variable, {}) if variable_configs else {}
        var_title = var_config.get('title', title)
        var_ylabel = var_config.get('ylabel', ylabel)
        var_xlabel = var_config.get('xlabel', xlabel)
        var_legend_labels = var_config.get('legend_labels', legend_labels)
        
        # Generate plot
        try:
            _plot_single_stacked_bar(
                cluster_df, variable, cluster_col,
                name_map, cluster_config, output_dir,
                sig_raw, sig_fdr, alpha,
                var_title, var_ylabel, var_xlabel, var_legend_labels
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")

def _plot_single_stacked_bar(
    cluster_df: pd.DataFrame,
    variable: str,
    cluster_col: str,
    name_map: Dict,
    cluster_config: Dict,
    output_dir: str,
    sig_raw: Optional[Dict],
    sig_fdr: Optional[Dict],
    alpha: float,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_labels: Optional[Dict] = None
    ):
    """Internal function to plot a single stacked bar chart with improved design."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    clusters = sorted(cluster_df[cluster_col].unique())
    
    # Prepare data - population is entire clustered dataset
    results = []
    pop_prop_1 = cluster_df[variable].mean() * 100
    pop_n_1 = cluster_df[variable].sum()
    pop_n_0 = len(cluster_df) - pop_n_1
    
    results.append({
        'group': 'Whole population',
        'prop_0': 100 - pop_prop_1,
        'prop_1': pop_prop_1,
        'n_0': int(pop_n_0),
        'n_1': int(pop_n_1),
        'n_total': len(cluster_df),
        'group_id': None,
        'p_raw': None,
        'p_fdr': None
    })
    
    for cluster_id in clusters:
        cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id]
        prop_1 = cluster_subset[variable].mean() * 100
        n_1 = cluster_subset[variable].sum()
        n_0 = len(cluster_subset) - n_1
        cluster_label = get_cluster_label(cluster_id, cluster_config)
        
        results.append({
            'group': cluster_label,
            'prop_0': 100 - prop_1,
            'prop_1': prop_1,
            'n_0': int(n_0),
            'n_1': int(n_1),
            'n_total': len(cluster_subset),
            'group_id': cluster_id,
            'p_raw': sig_raw.get(cluster_id) if sig_raw else None,
            'p_fdr': sig_fdr.get(cluster_id) if sig_fdr else None
        })
    
    plot_df = pd.DataFrame(results)
    
    # Create figure with proper margins for outer layers
    fig = plt.figure(figsize=(max(14, len(plot_df) * 1.8), 10))
    # Adjust subplot to leave room: [left, bottom, width, height]
    ax = fig.add_axes([0.1, 0.25, 0.85, 0.60])  # Leave 25% bottom, 15% top for labels
    
    x_pos = np.arange(len(plot_df))
    
    # Get legend labels
    if legend_labels:
        label_1 = legend_labels.get('achieved', 'Achieved')
        label_0 = legend_labels.get('not_achieved', 'Not Achieved')
    else:
        label_1 = 'Achieved'
        label_0 = 'Not Achieved'
    
    # INNERMOST LAYER: Plot bars (0-100% range ONLY)
    ax.bar(x_pos, plot_df['prop_1'], label=label_1, color=ACHIEVED_COLOR, alpha=0.8)
    ax.bar(x_pos, plot_df['prop_0'], bottom=plot_df['prop_1'], label=label_0, color=NOT_ACHIEVED_COLOR, alpha=0.8)

    # Display and position the bar plot legend (color codes)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.95), 
          title='', frameon=True, fancybox=True,
          fontsize=13, 
          edgecolor='black',
          facecolor='white',
          framealpha=0.8
          )

    # Population mean line
    ax.axhline(y=pop_prop_1, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add significance asterisks just above population line (INSIDE plot)
    for i, row in plot_df.iterrows():
        if row['group_id'] is not None:
            sig_text = ''
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']) and row['p_fdr'] < alpha:
                sig_text = '**'
            elif row['p_raw'] is not None and pd.notna(row['p_raw']) and row['p_raw'] < alpha:
                sig_text = '*'
            if sig_text:
                ax.text(i, pop_prop_1 + 1, sig_text, ha='center', va='bottom', 
                       fontsize=20, weight='bold')
    
    # Set axis limits STRICTLY to 0-100
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-0.6, len(plot_df) - 0.4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    
    # Styling
    nice_name = get_nice_name(variable, name_map)
    plot_title = title.format(variable=nice_name) if title else f'{nice_name}'
    plot_ylabel = ylabel if ylabel else 'Percentage (%)'
    plot_xlabel = xlabel if xlabel else 'Clusters'
    
    ax.set_ylabel(plot_ylabel, fontsize=16)
    ax.tick_params(labelsize=10)
    
    # MIDDLE LAYER: N (%) labels in figure coordinates (truly outside plot)
    for i, row in plot_df.iterrows():
        # Convert data coords to figure coords
        x_fig = ax.transData.transform((i, 0))[0] / fig.dpi / fig.get_size_inches()[0]
        
        # Top box always shows class 1 (achieved), bottom box always shows class 0 (not achieved)
        # This matches the stacking order where class 1 is at bottom, class 0 is stacked on top
        top_label = f"{row['n_0']} ({row['prop_0']:.1f}%)"
        top_color = NOT_ACHIEVED_COLOR  # Match bar color
        bottom_label = f"{row['n_1']} ({row['prop_1']:.1f}%)"
        bottom_color = ACHIEVED_COLOR  # Match bar color

        # Top box (above plot) - corresponds to top of stacked bar (class 0)
        fig.text(x_fig, 0.87, top_label, 
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=top_color, alpha=0.8, edgecolor='gray'))
        
        # Bottom box (below plot) - corresponds to bottom of stacked bar (class 1)
        fig.text(x_fig, 0.23, bottom_label, 
                ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bottom_color, alpha=0.8, edgecolor='gray'))
        
        # P-values (below bottom box)
        if row['group_id'] is not None and row['p_raw'] is not None and pd.notna(row['p_raw']):
            p_text = f"p={row['p_raw']:.3f}"
            if row['p_fdr'] is not None and pd.notna(row['p_fdr']):
                p_text += f"\np(FDR)={row['p_fdr']:.3f}"
            fig.text(x_fig, 0.2, p_text, ha='center', va='top', 
                    fontsize=12, style='italic')
        
        # Cluster labels (below p-values)
        fig.text(x_fig, 0.15, f"{row['group']}\n(n={row['n_total']})", 
                ha='right', va='top', fontsize=16, rotation=45)
    
    # X-axis label
    # fig.text(0.5, 0.02, plot_xlabel, ha='center', fontsize=12, weight='bold')

    # Population mean label with dashed line legend
    fig.text(0.96, 0.67, f"- - - - - - Population average\n            ({pop_prop_1:.1f}%)", 
            ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Significance legend box
    sig_text = "Significance:\n* p < 0.05 (raw)\n** p < 0.05 (FDR-corrected)"
    fig.text(0.96, 0.74, sig_text, ha='left', va='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # FRAME: Title
    fig.suptitle(plot_title, fontsize=20, weight='bold', y=0.96)
    
    # Save
    output_path = os.path.join(output_dir, f'{variable}_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_cluster_vs_population(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_db_path: str,
    output_table_name: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
    variable_types: Optional[Dict[str, str]] = None,
    fdr_correction: bool = True,
    alpha: float = 0.05
    ) -> pd.DataFrame:
    """
    Analyze WGC prevalence across clusters vs entire clustered population mean.
    Generates both detailed and publication-ready tables.
    
    IMPORTANT: Each cluster is compared to the ENTIRE clustered population (all clusters combined),
    not to an external population.
    
    Statistical Test Selection:
        - Continuous variables (>10 unique values): Mann-Whitney U test
        - Categorical variables (≤10 unique values): Chi-squared test (with Fisher's exact fallback)
        - Test selection is automatic when variable_types is None, or explicit when provided
    
    Args:
        cluster_df: DataFrame with cluster assignments and WGC variables
        variables: List of variables to analyze
        output_db_path: Path to output database
        output_table_name: Base name for output tables
        cluster_col: Column name for cluster IDs
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration (default: 'cluster_config.json')
        variable_types: Optional dict mapping variable names to 'continuous' or 'categorical'.
                       If None, types will be inferred automatically using _infer_variable_type().
                       Example: {'age': 'continuous', 'bmi': 'continuous', 'wgc_medication': 'categorical'}
        fdr_correction: Whether to apply FDR correction (Benjamini-Hochberg method)
        alpha: Significance threshold for marking results (default: 0.05)
    
    Returns:
        DataFrame with analysis results including:
            - Population mean/N for each variable
            - Cluster-specific mean/N for each variable
            - Raw p-values for each cluster comparison
            - FDR-corrected p-values (if fdr_correction=True)
            - FDR columns are positioned immediately after their corresponding raw p-value columns
    
    Notes:
        - When variable_types is None, a warning is printed and types are inferred
        - Test selection is logged for each variable during processing
        - Results are saved to two tables: detailed (with p-values) and publication-ready (with asterisks)
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Cluster vs Clustered Population Mean")
    print("="*60)
    print("Note: Each cluster compared to entire clustered population")
    print("="*60)
    
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    clusters = sorted(cluster_df[cluster_col].unique())
    
    # Infer variable types if not provided
    if variable_types is None:
        print("  ⚠️ No variable types provided, inferring from data...")
        variable_types = {}
        for variable in variables:
            if variable in cluster_df.columns:
                variable_types[variable] = _infer_variable_type(cluster_df[variable])
                print(f"    {variable}: {variable_types[variable]}")
    
    # Build results table
    results = []
    
    # N row - population is the entire clustered dataset
    n_row = {'Variable': 'N', 'Whole population: Mean (±SD) / N (%)': len(cluster_df)}
    for cluster_id in clusters:
        cluster_label = get_cluster_label(cluster_id, cluster_config)
        n_row[f'{cluster_label}: Mean (±SD) / N (%)'] = len(cluster_df[cluster_df[cluster_col] == cluster_id])
        n_row[f'{cluster_label}: p-value'] = 'N/A'
    results.append(n_row)
    
    # Process each variable
    for variable in variables:
        if variable not in cluster_df.columns:
            print(f"  ⚠️ Variable '{variable}' not found. Skipping.")
            continue
        
        print(f"  Processing: {variable}")
        
        # Use human-readable variable name for display ONLY
        row = {'Variable': get_nice_name(variable, name_map)}
        
        # Get variable type
        var_type = variable_types.get(variable, 'categorical')
        
        # Format population data based on type
        if var_type == 'continuous':
            pop_mean = cluster_df[variable].mean()
            pop_sd = cluster_df[variable].std()
            row['Whole population: Mean (±SD) / N (%)'] = f"{pop_mean:.2f} (±{pop_sd:.2f})"
        else:  # categorical
            pop_mean = cluster_df[variable].mean() * 100
            pop_n = cluster_df[variable].sum()
            row['Whole population: Mean (±SD) / N (%)'] = f"{int(pop_n)} ({pop_mean:.1f}%)"
        
        # Each cluster
        for cluster_id in clusters:
            cluster_label = get_cluster_label(cluster_id, cluster_config)
            cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id]
            
            # Format cluster data based on type
            if var_type == 'continuous':
                cluster_mean = cluster_subset[variable].mean()
                cluster_sd = cluster_subset[variable].std()
                row[f'{cluster_label}: Mean (±SD) / N (%)'] = f"{cluster_mean:.2f} (±{cluster_sd:.2f})"
                print(f"    Using Mann-Whitney U test (continuous) for cluster {cluster_id}")
                p_val = mann_whitney_u_test(cluster_subset[variable], cluster_df[variable])
            else:  # categorical
                cluster_mean = cluster_subset[variable].mean() * 100
                cluster_n = cluster_subset[variable].sum()
                row[f'{cluster_label}: Mean (±SD) / N (%)'] = f"{int(cluster_n)} ({cluster_mean:.1f}%)"
                print(f"    Using chi-squared test (categorical) for cluster {cluster_id}")
                p_val = chi_squared_test(cluster_subset[variable], cluster_df[variable])
            
            row[f'{cluster_label}: p-value'] = p_val
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction with dynamic column insertion
    if fdr_correction:
        print("  Applying FDR correction...")
        p_cols = [col for col in results_df.columns if 'p-value' in col and 'FDR' not in col]
        
        # Build new column order list that inserts FDR columns immediately after raw p-value columns
        new_columns = []
        for col in results_df.columns:
            new_columns.append(col)
            # If this is a raw p-value column, insert FDR column after it
            if col in p_cols:
                # Extract valid p-values
                pvals = pd.to_numeric(results_df[col], errors='coerce')
                valid_mask = pvals.notna() & (pvals != 'N/A')
                
                if valid_mask.sum() > 0:
                    valid_pvals = pvals[valid_mask]
                    _, corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
                    
                    # Create FDR column
                    fdr_col = col.replace('p-value', 'p-value (FDR-corrected)')
                    results_df[fdr_col] = np.nan
                    results_df.loc[valid_mask, fdr_col] = corrected
                    new_columns.append(fdr_col)
        
        # Reorder DataFrame columns
        results_df = results_df[new_columns]
    
    # Save detailed table
    detailed_table = f"{output_table_name}_detailed"
    with sqlite3.connect(output_db_path) as conn:
        results_df.to_sql(detailed_table, conn, if_exists='replace', index=False)
    print(f"  ✓ Detailed table saved: {detailed_table}")
    
    # Create publication-ready table (with asterisks)
    pub_df = results_df.copy()
    data_cols = [col for col in pub_df.columns if ': Mean (±SD) / N (%)' in col and 'Whole population' not in col]
    p_cols = [col for col in pub_df.columns if 'p-value' in col]
    
    for data_col in data_cols:
        # Extract cluster label (everything before ": Mean")
        cluster_label = data_col.split(': Mean')[0]
        raw_p_col = f'{cluster_label}: p-value'
        fdr_p_col = f'{cluster_label}: p-value (FDR-corrected)'
        
        if raw_p_col in pub_df.columns:
            raw_p = pd.to_numeric(pub_df[raw_p_col], errors='coerce')
            fdr_p = pd.to_numeric(pub_df[fdr_p_col], errors='coerce') if fdr_p_col in pub_df.columns else pd.Series([np.nan] * len(pub_df))
            
            # Add asterisks based on significance
            for idx in pub_df.index:
                if pd.notna(fdr_p.iloc[idx]) and fdr_p.iloc[idx] < alpha:
                    pub_df.at[idx, data_col] = str(pub_df.at[idx, data_col]) + '**'
                elif pd.notna(raw_p.iloc[idx]) and raw_p.iloc[idx] < alpha:
                    pub_df.at[idx, data_col] = str(pub_df.at[idx, data_col]) + '*'
    
    # Drop p-value columns
    pub_df = pub_df.drop(columns=p_cols, errors='ignore')
    
    # Save publication table
    with sqlite3.connect(output_db_path) as conn:
        pub_df.to_sql(output_table_name, conn, if_exists='replace', index=False)
    print(f"  ✓ Publication table saved: {output_table_name}")
    
    print("="*60)
    
    # Display DataFrame with pandas display options configured for wide tables
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\n" + "="*80)
    print("RESULTS TABLE (Detailed)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    return results_df


# =============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cluster_lollipop(
    cluster_df: pd.DataFrame,
    variables: List[str],
    output_filename: str,
    output_dir: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
    pvalues_raw: Optional[Dict[str, Dict[int, float]]] = None,
    pvalues_fdr: Optional[Dict[str, Dict[int, float]]] = None,
    results_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
    ):
    """
    Generate multi-variable lollipop plot showing percent change from clustered population mean.
    
    Each cluster is compared to the ENTIRE clustered population (all clusters combined).
    
    Args:
        cluster_df: DataFrame with cluster assignments and outcomes
        variables: List of continuous variables to compare
        output_filename: Name for output file
        output_dir: Directory to save plot
        cluster_col: Column name for cluster IDs
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration
        pvalues_raw: Optional dict mapping {variable: {cluster_id: p_value}} for raw p-values
                     If None and results_df provided, will be extracted automatically
        pvalues_fdr: Optional dict mapping {variable: {cluster_id: p_value}} for FDR-corrected p-values
                     If None and results_df provided, will be extracted automatically
        results_df: Optional DataFrame from analyze_cluster_vs_population() for automatic p-value extraction
        alpha: Significance threshold (default: 0.05)
        title: Custom plot title
        ylabel: Custom y-axis label (variable names)
        xlabel: Custom x-axis label
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    print("\nGenerating lollipop plot...")
    
    # Auto-extract p-values if results_df provided but p-values not provided
    if results_df is not None and (pvalues_raw is None or pvalues_fdr is None):
        print("  Extracting p-values from results_df...")
        pvalues_raw, pvalues_fdr = extract_pvalues_for_lollipop(
            results_df, variables, cluster_df, cluster_col, cluster_config_path, name_map_path
        )
    
    # Prepare data - population is entire clustered dataset
    lollipop_data = []
    clusters = sorted(cluster_df[cluster_col].unique())
    
    for variable in variables:
        if variable not in cluster_df.columns:
            continue
        
        # Population mean is the mean of entire clustered dataset
        pop_mean = cluster_df[variable].mean()
        if pd.isna(pop_mean) or pop_mean == 0:
            continue
        
        for cluster_id in clusters:
            cluster_subset = cluster_df[cluster_df[cluster_col] == cluster_id]
            cluster_mean = cluster_subset[variable].mean()
            
            if pd.notna(cluster_mean):
                pct_change = ((cluster_mean - pop_mean) / pop_mean) * 100
                
                lollipop_data.append({
                    'variable': variable,
                    'cluster': f'Cluster {int(cluster_id)}',
                    'value': pct_change
                })
    
    if not lollipop_data:
        print("  ⚠️ No data for lollipop plot")
        return
    
    lollipop_df = pd.DataFrame(lollipop_data)
    
    # Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    
    variables_in_data = lollipop_df['variable'].unique()
    clusters_in_data = sorted(lollipop_df['cluster'].unique(), key=lambda x: int(x.split(' ')[-1]))
    
    fig, ax = plt.subplots(figsize=(12, 2 + len(variables_in_data) * len(clusters_in_data) * 0.2))
    
    # Get cluster colors
    cluster_colors = {}
    for cluster in clusters_in_data:
        cluster_id = int(cluster.split(' ')[-1])
        color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
        cluster_label = get_cluster_label(cluster_id, cluster_config)
        cluster_colors[cluster] = {'color': color, 'label': cluster_label}
    
    y_pos = 0
    y_ticks, y_tick_labels = [], []
    
    for var in variables_in_data:
        var_data = lollipop_df[lollipop_df['variable'] == var].set_index('cluster').reindex(clusters_in_data).dropna()
        if var_data.empty:
            continue
        
        y_range = np.arange(y_pos, y_pos + len(var_data))
        
        ax.hlines(y=y_range, xmin=0, xmax=var_data['value'],
                 colors=[cluster_colors[c]['color'] for c in var_data.index], linewidth=2)
        ax.scatter(var_data['value'], y_range,
                  c=[cluster_colors[c]['color'] for c in var_data.index], s=50, zorder=10)
        
        # Add significance markers
        if pvalues_raw or pvalues_fdr:
            for i, cluster in enumerate(var_data.index):
                cluster_id = int(cluster.split(' ')[-1])
                x_val = var_data.loc[cluster, 'value']
                y_val = y_range[i]
                
                # Get p-values for this variable-cluster combination
                p_raw = pvalues_raw.get(var, {}).get(cluster_id) if pvalues_raw else None
                p_fdr = pvalues_fdr.get(var, {}).get(cluster_id) if pvalues_fdr else None
                
                # Determine significance marker
                sig_text = ''
                if p_fdr is not None and pd.notna(p_fdr) and p_fdr < alpha:
                    sig_text = '**'
                elif p_raw is not None and pd.notna(p_raw) and p_raw < alpha:
                    sig_text = '*'
                
                if sig_text:
                    # Position marker slightly to the right of the point
                    x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                    ax.text(x_val + x_offset, y_val, sig_text, ha='left', va='center', 
                           fontsize=16, color='black', weight='bold')
        
        if y_range.size > 0:
            y_ticks.append(np.mean(y_range))
            y_tick_labels.append(get_nice_name(var, name_map))
        
        y_pos += len(var_data)
        if len(variables_in_data) > 1 and var != variables_in_data[-1]:
            y_pos += 0.5
            ax.axhline(y=y_pos - 0.75, color='grey', linestyle='-', linewidth=2.0)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=12)
    ax.invert_yaxis()
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Configurable labels
    if title:
        plot_title = title
    else:
        plot_title = 'Multi-Cluster Comparison: Percent Change vs Population Mean'
    
    if xlabel:
        plot_xlabel = xlabel
    else:
        plot_xlabel = 'Percent Change vs Population Mean (%)'
    
    ax.set_title(plot_title, fontsize=16, weight='bold')
    ax.set_xlabel(plot_xlabel, fontsize=14)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=cluster_colors[cluster]['color'],
                  label=cluster_colors[cluster]['label'], linestyle='None', markersize=8)
        for cluster in clusters_in_data
    ]
    ax.legend(handles=legend_elements, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  ✓ Lollipop plot saved: {output_path}")

def _parse_cluster_column_header(header: str, cluster_config: Dict) -> Optional[int]:

    # Try to match against cluster labels from config (new format)
    cluster_labels = cluster_config.get('cluster_labels', {})
    for cluster_id_str, label in cluster_labels.items():
        if header.startswith(f'{label}:'):
            try:
                return int(cluster_id_str)
            except ValueError:
                continue
    
    # Fallback to old format: "Cluster X: Mean/N" or "Cluster X: p-value"
    match = re.match(r'^Cluster\s+(\d+):', header)
    if match:
        return int(match.group(1))
    
    return None

def plot_cluster_heatmap(
    results_df: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    variables: List[str],
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
    alpha: float = 0.05,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    cbar_label: Optional[str] = None
    ):
    """
    Generate heatmap from cluster vs population analysis results.
    
    Args:
        results_df: DataFrame from analyze_cluster_vs_population
        output_filename: Name for output file
        output_dir: Directory to save plot
        variables: List of variables to include
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration
        alpha: Significance threshold
        title: Custom plot title
        ylabel: Custom y-axis label (variable names)
        xlabel: Custom x-axis label (cluster names)
        cbar_label: Custom colorbar label
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    print("\nGenerating heatmap...")
    
    # Extract prevalence data
    prevalence_data = []
    sig_map_raw = {}
    sig_map_fdr = {}
    
    # Get cluster IDs from column names - support both old and new formats
    cluster_cols = [col for col in results_df.columns 
                    if (': Mean (±SD) / N (%)' in col or ': Mean/N' in col) 
                    and col != 'Whole population: Mean (±SD) / N (%)']
    
    # Parse cluster IDs from headers
    cluster_id_map = {}  # Maps column name to cluster ID
    for col in cluster_cols:
        cluster_id = _parse_cluster_column_header(col, cluster_config)
        if cluster_id is not None:
            cluster_id_map[col] = cluster_id
    
    clusters = sorted(cluster_id_map.values())
    
    for _, row in results_df.iterrows():
        wgc_var = row['Variable']
        
        # Skip the N row
        if wgc_var == 'N':
            continue
        
        # The Variable column contains nice names, so we need to find the matching raw variable
        # by checking if any requested variable's nice name matches
        raw_var = None
        for requested_var in variables:
            if wgc_var == get_nice_name(requested_var, name_map):
                raw_var = requested_var
                break
        
        if raw_var is None:
            continue
        
        sig_map_raw[raw_var] = {}
        sig_map_fdr[raw_var] = {}
        
        # Extract population prevalence data first
        pop_col = 'Whole population: Mean (±SD) / N (%)'
        if pop_col in row:
            try:
                pop_str = str(row[pop_col])
                pop_n = int(pop_str.split('(')[0].strip())
                pop_pct = _extract_percentage_from_table_cell(pop_str)
                
                if not np.isnan(pop_pct):
                    # Insert population data as first entry with cluster_id='Population'
                    prevalence_data.append({
                        'wgc_variable': raw_var,
                        'cluster_id': 'Population',
                        'prevalence_%': pop_pct,
                        'n': pop_n
                    })
            except (ValueError, IndexError) as e:
                print(f"  ⚠️ Could not parse population data for {raw_var}: {e}")
        
        # Process each cluster column
        for mean_n_col, cluster_id in cluster_id_map.items():
            if mean_n_col not in row:
                continue
            
            try:
                mean_n_str = str(row[mean_n_col])
                n = int(mean_n_str.split('(')[0].strip())
                pct = _extract_percentage_from_table_cell(mean_n_str)
                
                if not np.isnan(pct):
                    prevalence_data.append({
                        'wgc_variable': raw_var,
                        'cluster_id': cluster_id,
                        'prevalence_%': pct,
                        'n': n
                    })
                
                # Extract p-values - need to find matching p-value columns
                cluster_label = get_cluster_label(cluster_id, cluster_config)
                p_raw_col = f'{cluster_label}: p-value'
                p_fdr_col = f'{cluster_label}: p-value (FDR-corrected)'
                
                # Also try old format for backward compatibility
                if p_raw_col not in row:
                    p_raw_col = f'Cluster {cluster_id}: p-value'
                if p_fdr_col not in row:
                    p_fdr_col = f'Cluster {cluster_id}: p-value (FDR-corrected)'
                
                if p_raw_col in row:
                    sig_map_raw[raw_var][cluster_id] = row[p_raw_col]
                if p_fdr_col in row:
                    sig_map_fdr[raw_var][cluster_id] = row[p_fdr_col]
            except (ValueError, IndexError) as e:
                print(f"  ⚠️ Could not parse data for {raw_var}, cluster {cluster_id}: {e}")
                continue
    
    if not prevalence_data:
        print("  ⚠️ No prevalence data for heatmap")
        return
    
    prevalence_df = pd.DataFrame(prevalence_data)
    
    # Create heatmap with cluster-specific color scales
    plt.style.use('seaborn-v0_8-whitegrid')
    
    matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='prevalence_%')
    n_matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='n')
    
    # Reorder columns to ensure 'Population' appears first
    if 'Population' in matrix.columns:
        cols = ['Population'] + [c for c in matrix.columns if c != 'Population']
        matrix = matrix[cols]
        n_matrix = n_matrix[cols]
    
    # Reorder rows to match input variables list order
    # Use the variables list to ensure exact ordering
    matrix = matrix.reindex(variables)
    n_matrix = n_matrix.reindex(variables)
    
    # Get labels - handle Population column specially
    cluster_labels = []
    for cid in matrix.columns:
        if cid == 'Population':
            cluster_labels.append('Whole population')
        else:
            cluster_labels.append(get_cluster_label(cid, cluster_config))
    
    # Create labels from the variables list to maintain order
    wgc_labels = [get_nice_name(wgc, name_map) for wgc in variables]
    
    # Create annotations - iterate over variables list instead of matrix.index
    annotations = np.empty_like(matrix, dtype=object)
    for i, wgc in enumerate(variables):
        for j, cluster_id in enumerate(matrix.columns):
            pct = matrix.iloc[i, j]
            n = n_matrix.iloc[i, j]
            
            # Get significance marker (only for cluster columns, not Population)
            sig_marker = ''
            if cluster_id != 'Population':
                if sig_map_fdr and wgc in sig_map_fdr:
                    p_fdr = sig_map_fdr[wgc].get(cluster_id)
                    if p_fdr is not None and pd.notna(p_fdr) and p_fdr < alpha:
                        sig_marker = '**'
                if not sig_marker and sig_map_raw and wgc in sig_map_raw:
                    p_raw = sig_map_raw[wgc].get(cluster_id)
                    if p_raw is not None and pd.notna(p_raw) and p_raw < alpha:
                        sig_marker = '*'
            
            if pd.notna(pct) and pd.notna(n):
                annotations[i, j] = f"{int(n)} ({pct:.1f}%){sig_marker}"
            else:
                annotations[i, j] = ""
    
    # Plot with multiple subplots for cluster-specific color scales
    n_cols = len(matrix.columns)
    n_rows = len(variables)
    
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(max(12, n_cols * 1.5), max(8, n_rows * 0.6)),
        gridspec_kw={'wspace': 0.01}
    )
    
    # Handle single column case
    if n_cols == 1:
        axes = [axes]
    
    # Create colormaps for each column
    for j, (cluster_id, ax) in enumerate(zip(matrix.columns, axes)):
        # Get data for this column
        col_data = matrix.iloc[:, j:j+1].values
        
        # Create cluster-specific colormap
        if cluster_id == 'Population':
            # White-to-red for population column
            cmap = LinearSegmentedColormap.from_list('pop', ['white', 'red'])
        else:
            # White-to-cluster-color for cluster columns
            cluster_color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
            cmap = LinearSegmentedColormap.from_list(f'cluster_{cluster_id}', ['white', cluster_color])
        
        # Plot using imshow
        im = ax.imshow(col_data, cmap=cmap, vmin=0, vmax=100, aspect='auto')
        
        # Add annotations manually
        for i in range(n_rows):
            text = annotations[i, j]
            if text:
                ax.text(0, i, text, ha='center', va='center', fontsize=10)
        
        # Add column label at bottom, rotated and centered
        ax.set_xlabel(cluster_labels[j], fontsize=10, rotation=45, ha='right')
        
        # Configure axes - remove all ticks and gridlines
        ax.set_xticks([])
        ax.set_yticks(range(n_rows))
        ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
        
        # Add row labels to leftmost axis only, with rotation
        if j == 0:
            ax.set_yticklabels(wgc_labels, fontsize=10, rotation=0, ha='right')
        else:
            ax.set_yticklabels([])
        
        # Only show outer borders, no internal gridlines
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['right'].set_linewidth(1.0)
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        
        # Turn off grid completely
        ax.grid(False)
    
    # Configurable labels
    if title:
        plot_title = title
    else:
        plot_title = 'Weight Gain Cause Prevalence by Cluster'
    
    if ylabel:
        plot_ylabel = ylabel
    else:
        plot_ylabel = 'Weight Gain Causes'

    if xlabel:
        plot_xlabel = xlabel
    else:
        plot_xlabel = 'Clusters'
    
    # Add overall title
    fig.suptitle(plot_title, fontsize=16, weight='bold', y=0.96)
    
    # Add y-axis label further to the left to avoid overlap with rotated variable labels
    fig.text(0.0001, 0.5, plot_ylabel, fontsize=14, rotation=90, va='center', ha='center')
    
    # Save
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  ✓ Heatmap saved: {output_path}")

def plot_cluster_forest(
    cluster_df: pd.DataFrame,
    outcome_variable: str,
    output_dir: str,
    cluster_col: str = 'cluster_id',
    name_map_path: str = 'human_readable_variable_names.json',
    cluster_config_path: str = 'cluster_config.json',
    effect_type: str = 'both',
    title_rr: Optional[str] = None,
    title_rd: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_rr: Optional[str] = None,
    xlabel_rd: Optional[str] = None
    ):
    """
    Generate forest plots for risk ratios and/or risk differences comparing clusters to population.
    
    Each cluster is compared to the ENTIRE clustered population (all clusters combined).
    
    Args:
        cluster_df: DataFrame with cluster assignments and binary outcome
        outcome_variable: Name of binary outcome variable (0/1)
        output_dir: Directory to save plots
        cluster_col: Column name for cluster IDs
        name_map_path: Path to variable name mappings
        cluster_config_path: Path to cluster configuration
        effect_type: 'RR' for risk ratio only, 'RD' for risk difference only, 'both' for both (default)
        title_rr: Custom title for risk ratio plot
        title_rd: Custom title for risk difference plot
        ylabel: Custom y-axis label (cluster names)
        xlabel_rr: Custom x-axis label for risk ratio plot
        xlabel_rd: Custom x-axis label for risk difference plot
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    print(f"\nGenerating forest plot(s) for '{outcome_variable}'...")
    
    # Validate outcome variable
    if outcome_variable not in cluster_df.columns:
        print(f"  ⚠️ Variable '{outcome_variable}' not found in data. Skipping.")
        return
    
    if cluster_df[outcome_variable].dropna().empty:
        print(f"  ⚠️ No valid data for '{outcome_variable}'. Skipping.")
        return
    
    # Calculate risk metrics
    risk_df = calculate_risk_metrics(cluster_df, outcome_variable, cluster_col)
    
    # Generate plots based on effect_type
    if effect_type in ['RR', 'both']:
        _plot_single_forest(
            risk_df, outcome_variable, 'RR', output_dir,
            name_map, cluster_config, cluster_col,
            title_rr, ylabel, xlabel_rr
        )
        print(f"  ✓ Risk Ratio plot saved")
    
    if effect_type in ['RD', 'both']:
        _plot_single_forest(
            risk_df, outcome_variable, 'RD', output_dir,
            name_map, cluster_config, cluster_col,
            title_rd, ylabel, xlabel_rd
        )
        print(f"  ✓ Risk Difference plot saved")

def _plot_single_forest(
    risk_df: pd.DataFrame,
    outcome_variable: str,
    plot_type: str,
    output_dir: str,
    name_map: Dict,
    cluster_config: Dict,
    cluster_col: str,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None
    ):
    """Internal function to plot a single forest plot (RR or RD)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort by cluster_id
    plot_data = risk_df.sort_values('cluster_id', ascending=False)
    
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(plot_data) * 0.5)), 
                                   gridspec_kw={'width_ratios': [3, 0.8], 'wspace': 0.05})
    
    y_pos = np.arange(len(plot_data))
    
    # Select data based on plot type
    if plot_type == 'RR':
        effect = plot_data['risk_ratio']
        ci_lower = plot_data['rr_ci_lower']
        ci_upper = plot_data['rr_ci_upper']
        ref_value = 1
        default_xlabel = 'Risk Ratio'
        default_title = f'Risk Ratios: {get_nice_name(outcome_variable, name_map)}'
    else:  # RD
        effect = plot_data['risk_difference']
        ci_lower = plot_data['rd_ci_lower']
        ci_upper = plot_data['rd_ci_upper']
        ref_value = 0
        default_xlabel = 'Risk Difference'
        default_title = f'Risk Differences: {get_nice_name(outcome_variable, name_map)}'
    
    # Get cluster colors
    colors = [get_cluster_color(cid, cluster_config, DEFAULT_PALETTE) for cid in plot_data['cluster_id']]
    
    # Plot error bars with cluster colors
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
    
    # Get cluster labels for y-axis
    cluster_labels = [get_cluster_label(cid, cluster_config) for cid in plot_data['cluster_id']]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cluster_labels, fontsize=12)
    
    # Configurable labels
    if title:
        plot_title = title
    else:
        plot_title = default_title
    
    if xlabel:
        plot_xlabel = xlabel
    else:
        plot_xlabel = default_xlabel
    
    ax.set_title(plot_title, fontsize=16, weight='bold')
    ax.set_xlabel(plot_xlabel, fontsize=14)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Right y-axis with effect sizes and CIs
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    
    # Format CI labels
    ci_labels = []
    for _, row in plot_data.iterrows():
        if plot_type == 'RR':
            label = f"RR: {row['risk_ratio']:.2f} [{row['rr_ci_lower']:.2f}-{row['rr_ci_upper']:.2f}]"
        else:  # RD
            label = f"RD: {row['risk_difference']*100:.1f}% [{row['rd_ci_lower']*100:.1f}-{row['rd_ci_upper']*100:.1f}%]"
        ci_labels.append(label)
    
    ax2.set_yticklabels(ci_labels, fontsize=10, ha='left')  # Left-align to bring closer
    ax2.set_ylabel('Effect Size [95% CI]', fontsize=12, labelpad=15)  # Add padding to ylabel
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', pad=0)  # Remove tick padding to bring labels closer
    ax2.set_xticks([])
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)  # Remove right spine too for cleaner look
    
    # Save
    suffix = 'rr' if plot_type == 'RR' else 'rd'
    output_path = os.path.join(output_dir, f'{outcome_variable}_forest_{suffix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_merge_cluster_data(
    cluster_db_path: str,
    main_db_path: str,
    cluster_table: str,
    cluster_column: str,
    outcome_table: str
    ) -> pd.DataFrame:
    """
    Convenience function to load and merge cluster data.
    
    Note: The "population" for comparisons is the entire clustered dataset,
    not a separate population table.
    
    Returns:
        DataFrame with cluster assignments and outcomes merged
    """
    print("Loading cluster data...")
    
    # Load cluster assignments
    with sqlite3.connect(cluster_db_path) as conn:
        cluster_labels = pd.read_sql_query(
            f"SELECT medical_record_id, {cluster_column} as cluster_id FROM {cluster_table}",
            conn
        )
    
    print(f"  ✓ Loaded {len(cluster_labels)} cluster assignments")
    
    # Load outcome data
    with sqlite3.connect(main_db_path) as conn:
        outcome_df = pd.read_sql_query(f"SELECT * FROM {outcome_table}", conn)
    
    print(f"  ✓ Loaded {len(outcome_df)} outcome records")
    
    # Merge
    cluster_df = outcome_df.merge(cluster_labels, on='medical_record_id', how='inner')
    
    print(f"  ✓ Merged: {len(cluster_df)} records with clusters")
    print(f"  ✓ Clusters: {sorted(cluster_df['cluster_id'].unique())}")
    print(f"  ✓ Population for comparisons: entire clustered dataset (n={len(cluster_df)})")
    
    return cluster_df
