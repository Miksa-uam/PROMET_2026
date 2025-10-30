"""
Descriptive Visualizations Swiss Army Knife

Clean, general-purpose visualization engines for population comparisons.
Each function takes clean data and produces publication-ready plots.

Functions:
- create_lollipop_plot: Percent change comparisons between cohorts
- create_forest_plot: Risk ratios/differences for binary outcomes  
- create_split_violin_plot: Distribution comparisons for continuous variables
- create_stacked_bar_plot: Binary variable distributions across groups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _apply_plot_styling() -> None:
    """Apply consistent publication-ready styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

def _calculate_percent_change(ref_val: float, comp_val: float) -> float:
    """Calculate percent change: ((comp - ref) / ref) * 100"""
    if pd.isna(ref_val) or pd.isna(comp_val) or ref_val == 0:
        return np.nan
    return ((comp_val - ref_val) / ref_val) * 100

def _calculate_risk_ratio(data: pd.DataFrame, outcome_col: str, group_col: str) -> Tuple[float, float, float]:
    """Calculate risk ratio with 95% CI."""
    # Create 2x2 contingency table
    exposed = data[data[group_col] == 1]
    unexposed = data[data[group_col] == 0]
    
    a = len(exposed[exposed[outcome_col] == 1])  # events in exposed
    b = len(exposed[exposed[outcome_col] == 0])  # non-events in exposed  
    c = len(unexposed[unexposed[outcome_col] == 1])  # events in unexposed
    d = len(unexposed[unexposed[outcome_col] == 0])  # non-events in unexposed
    
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add continuity correction
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)
    
    if risk_unexposed == 0:
        return np.inf, np.inf, np.inf
    
    rr = risk_exposed / risk_unexposed
    
    # Calculate 95% CI using log transformation
    se_log_rr = np.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c+d)))
    log_rr = np.log(rr)
    
    ci_lower = np.exp(log_rr - 1.96 * se_log_rr)
    ci_upper = np.exp(log_rr + 1.96 * se_log_rr)
    
    return rr, ci_lower, ci_upper

def _calculate_risk_difference(data: pd.DataFrame, outcome_col: str, group_col: str) -> Tuple[float, float, float]:
    """Calculate risk difference with 95% CI."""
    exposed = data[data[group_col] == 1]
    unexposed = data[data[group_col] == 0]
    
    a = len(exposed[exposed[outcome_col] == 1])
    b = len(exposed[exposed[outcome_col] == 0])
    c = len(unexposed[unexposed[outcome_col] == 1])
    d = len(unexposed[unexposed[outcome_col] == 0])
    
    if a == 0 or b == 0 or c == 0 or d == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    
    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)
    rd = risk_exposed - risk_unexposed
    
    # Calculate 95% CI
    se_rd = np.sqrt((a*b)/((a+b)**3) + (c*d)/((c+d)**3))
    ci_lower = rd - 1.96 * se_rd
    ci_upper = rd + 1.96 * se_rd
    
    return rd, ci_lower, ci_upper

# =============================================================================
# VISUALIZATION FUNCTIONS  
# =============================================================================

def create_lollipop_plot(
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame, 
    variables: List[str],
    output_path: str = None,
    title: str = None,
    reference_label: str = "Reference",
    comparison_label: str = "Comparison",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create lollipop plot showing percent change between two cohorts.
    
    Args:
        reference_data: DataFrame with reference cohort (denominator for % change)
        comparison_data: DataFrame with comparison cohort 
        variables: List of column names to compare
        output_path: Optional path to save plot
        title: Plot title
        reference_label: Label for reference group
        comparison_label: Label for comparison group
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    _apply_plot_styling()
    
    # Calculate percent changes
    results = []
    for var in variables:
        if var not in reference_data.columns or var not in comparison_data.columns:
            continue
            
        ref_val = reference_data[var].mean()
        comp_val = comparison_data[var].mean()
        pct_change = _calculate_percent_change(ref_val, comp_val)
        
        if not pd.isna(pct_change):
            results.append({'variable': var, 'percent_change': pct_change})
    
    if not results:
        raise ValueError("No valid comparisons found for the given variables")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('percent_change', key=abs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = range(len(df_results))
    
    # Plot lollipops
    for i, (_, row) in enumerate(df_results.iterrows()):
        ax.plot([0, row['percent_change']], [i, i], 'o-', 
                color='steelblue', linewidth=2, markersize=8, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_results['variable'])
    ax.set_xlabel(f'Percent change: {comparison_label} vs {reference_label}')
    ax.set_ylabel('Variables')
    
    if title:
        ax.set_title(title)
    
    # Add reference line at 0%
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_forest_plot(
    data: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    effect_type: str = "risk_ratio",
    output_path: str = None,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create forest plot showing risk ratios or risk differences.
    
    Args:
        data: DataFrame with outcome and grouping variables
        outcome_col: Binary outcome column name (0/1)
        group_col: Binary grouping column name (0/1) 
        effect_type: "risk_ratio" or "risk_difference"
        output_path: Optional path to save plot
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    _apply_plot_styling()
    
    # Calculate effect size
    if effect_type == "risk_ratio":
        effect, ci_lower, ci_upper = _calculate_risk_ratio(data, outcome_col, group_col)
        reference_value = 1.0
        x_label = "Risk Ratio"
        use_log_scale = True
    elif effect_type == "risk_difference":
        effect, ci_lower, ci_upper = _calculate_risk_difference(data, outcome_col, group_col)
        reference_value = 0.0
        x_label = "Risk Difference"
        use_log_scale = False
    else:
        raise ValueError("effect_type must be 'risk_ratio' or 'risk_difference'")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot point and error bars
    ax.errorbar([effect], [0], xerr=[[effect - ci_lower], [ci_upper - effect]], 
                fmt='o', markersize=10, capsize=8, capthick=2, 
                color='steelblue', markerfacecolor='steelblue')
    
    # Add reference line
    ax.axvline(x=reference_value, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    
    if use_log_scale and effect > 0 and ci_lower > 0:
        ax.set_xscale('log')
    
    if title:
        ax.set_title(title)
    
    # Add effect size text
    if effect_type == "risk_ratio":
        effect_text = f"RR = {effect:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f})"
    else:
        effect_text = f"RD = {effect:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})"
    
    ax.text(0.02, 0.98, effect_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_split_violin_plot(
    data: pd.DataFrame,
    variable: str,
    group_cols: List[str],
    output_path: str = None,
    title: str = None,
    group_labels: List[str] = None,
    p_values: List[float] = None,
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    Create split violin plot comparing distributions across multiple binary groups.
    
    Args:
        data: DataFrame with continuous variable and grouping columns
        variable: Continuous variable column name
        group_cols: List of binary grouping column names (0/1)
        output_path: Optional path to save plot  
        title: Plot title
        group_labels: Labels for the groups (if None, uses column names)
        p_values: Optional list of p-values for significance testing (same order as group_cols)
        figsize: Figure size (width, height) - auto-calculated if None
    
    Returns:
        matplotlib Figure object
    """
    _apply_plot_styling()
    
    # Auto-calculate figure size based on number of groups
    if figsize is None:
        width = max(8, len(group_cols) * 1.5)
        figsize = (width, 6)
    
    # Prepare data for plotting
    plot_data_list = []
    
    for i, group_col in enumerate(group_cols):
        group_label = group_labels[i] if group_labels and i < len(group_labels) else group_col.replace('_', ' ').title()
        
        # Get data for absent group (0)
        absent_data = data[data[group_col] == 0][variable].dropna()
        for val in absent_data:
            plot_data_list.append({
                'value': val,
                'group': group_label,
                'status': 'Absent',
                'group_idx': i
            })
        
        # Get data for present group (1)
        present_data = data[data[group_col] == 1][variable].dropna()
        for val in present_data:
            plot_data_list.append({
                'value': val,
                'group': group_label,
                'status': 'Present', 
                'group_idx': i
            })
    
    if not plot_data_list:
        raise ValueError("No valid data found for plotting")
    
    plot_df = pd.DataFrame(plot_data_list)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create split violin plot
    sns.violinplot(data=plot_df, x='group_idx', y='value', hue='status',
                   split=True, inner='quart', 
                   palette={'Absent': '#5B9BD5', 'Present': '#E07B39'},
                   ax=ax, linewidth=1.5)
    
    # Customize x-axis labels
    group_names = [group_labels[i] if group_labels and i < len(group_labels) 
                   else group_cols[i].replace('_', ' ').title() 
                   for i in range(len(group_cols))]
    
    ax.set_xticks(range(len(group_cols)))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    
    # Add sample sizes and significance markers if p_values provided
    if p_values is not None:
        for i, (group_col, p_val) in enumerate(zip(group_cols, p_values)):
            absent_data = data[data[group_col] == 0][variable].dropna()
            present_data = data[data[group_col] == 1][variable].dropna()
            
            # Add sample sizes above violins
            y_max = ax.get_ylim()[1]
            sample_text = f'n={len(absent_data)}|{len(present_data)}'
            ax.text(i, y_max * 1.05, sample_text, ha='center', va='bottom', fontsize=8)
            
            # Add significance marker
            if pd.notna(p_val):
                if p_val < 0.01:
                    ax.text(i, y_max * 1.15, '**', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
                elif p_val < 0.05:
                    ax.text(i, y_max * 1.15, '*', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Groups')
    ax.set_ylabel(variable.replace('_', ' ').title())
    
    if title:
        ax.set_title(title, pad=20)
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', loc='upper right')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_stacked_bar_plot(
    data: pd.DataFrame,
    variable: str,
    group_cols: List[str],
    reference_data: pd.DataFrame = None,
    output_path: str = None,
    title: str = None,
    group_labels: List[str] = None,
    p_values: List[float] = None,
    figsize: Tuple[int, int] = None
) -> plt.Figure:
    """
    Create stacked bar plot comparing binary variable distributions across multiple groups.
    
    Args:
        data: DataFrame with binary variable and grouping columns
        variable: Binary variable column name (0/1)
        group_cols: List of grouping column names to compare
        reference_data: Optional reference cohort for comparison line
        output_path: Optional path to save plot
        title: Plot title  
        group_labels: Labels for the groups
        p_values: Optional list of p-values for significance testing (same order as group_cols)
        figsize: Figure size (width, height) - auto-calculated if None
    
    Returns:
        matplotlib Figure object
    """
    _apply_plot_styling()
    
    # Auto-calculate figure size based on number of groups
    if figsize is None:
        width = max(8, len(group_cols) * 1.2 + 2)  # +2 for reference if present
        figsize = (width, 6)
    
    # Calculate proportions for each group
    results = []
    
    # Add reference data if provided
    if reference_data is not None:
        ref_prop = reference_data[variable].mean() * 100
        results.append({
            'group': 'Reference',
            'prop_0': 100 - ref_prop,
            'prop_1': ref_prop,
            'n': len(reference_data)
        })
    
    # Calculate proportions for each group
    for i, group_col in enumerate(group_cols):
        group_data = data[data[group_col] == 1]  # Only where group is present
        if len(group_data) == 0:
            continue
            
        prop_1 = group_data[variable].mean() * 100
        prop_0 = 100 - prop_1
        
        group_name = group_labels[i] if group_labels and i < len(group_labels) else group_col.replace('_', ' ').title()
        
        results.append({
            'group': group_name,
            'prop_0': prop_0,
            'prop_1': prop_1,
            'n': len(group_data)
        })
    
    if not results:
        raise ValueError("No valid groups found for plotting")
    
    df_results = pd.DataFrame(results)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(df_results))
    
    # Create stacked bars
    bars_0 = ax.bar(x_pos, df_results['prop_0'], label='Category 0', 
                    color='#FF9999', alpha=0.8)
    bars_1 = ax.bar(x_pos, df_results['prop_1'], bottom=df_results['prop_0'], 
                    label='Category 1', color='#66B2FF', alpha=0.8)
    
    # Add reference line if reference data provided
    if reference_data is not None:
        ref_prop = reference_data[variable].mean() * 100
        ax.axhline(y=ref_prop, color='black', linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Reference ({ref_prop:.1f}%)')
    
    # Add sample size labels and significance markers
    for i, (bar_0, bar_1, row) in enumerate(zip(bars_0, bars_1, df_results.itertuples())):
        # Add n labels above bars
        ax.text(bar_0.get_x() + bar_0.get_width()/2, 105, f'n={row.n}',
                ha='center', va='bottom', fontweight='bold')
        
        # Add significance markers if p_values provided (skip reference bar)
        if p_values is not None and reference_data is not None and i > 0:
            p_idx = i - 1  # Adjust index for reference bar
            if p_idx < len(p_values) and pd.notna(p_values[p_idx]):
                if p_values[p_idx] < 0.01:
                    ax.text(bar_0.get_x() + bar_0.get_width()/2, 115, '**',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
                elif p_values[p_idx] < 0.05:
                    ax.text(bar_0.get_x() + bar_0.get_width()/2, 115, '*',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif p_values is not None and reference_data is None and i < len(p_values):
            if pd.notna(p_values[i]):
                if p_values[i] < 0.01:
                    ax.text(bar_0.get_x() + bar_0.get_width()/2, 115, '**',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
                elif p_values[i] < 0.05:
                    ax.text(bar_0.get_x() + bar_0.get_width()/2, 115, '*',
                           ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Groups')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 120 if p_values is not None else 110)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_results['group'], rotation=45, ha='right')
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

