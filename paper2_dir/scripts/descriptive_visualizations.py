# =============================================================================
# MODULE: DESCRIPTIVE_VISUALIZATIONS.PY
# VERSION: 4.0 (Corrected & Finalized)
#
# DESCRIPTION:
# A final, corrected version of the generalized visualization toolkit. This
# version specifically fixes issues with split-violin and stacked-bar plots
# to match user requirements and reference images.
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any

# --- CONFIGURATION & SETUP ---
DEFAULT_PALETTE = sns.color_palette("husl", 12)
POPULATION_COLOR = 'lightgrey'
ACHIEVED_COLOR = '#66B2FF'
NOT_ACHIEVED_COLOR = '#FF9999'


# --- HELPER FUNCTIONS (Unchanged) ---
def load_name_map(json_path: str) -> Dict[str, str]:
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
    return name_map.get(variable, variable.replace('_', ' ').title())

def _annotate_significance(ax: plt.Axes, x: float, y: float, p_raw: Optional[float], p_fdr: Optional[float], alpha: float):
    text = ''
    if p_fdr is not None and p_fdr < alpha:
        text = '**'
    elif p_raw is not None and p_raw < alpha:
        text = '*'
    if text:
        ax.text(x, y, text, ha='center', va='bottom', fontsize=20, color='black', weight='bold')

# =============================================================================
# CORE VISUALIZATION FUNCTIONS (CORRECTED)
# =============================================================================

def plot_distribution_comparison(
    df: pd.DataFrame,
    population_df: pd.DataFrame,
    variable: str,
    group_col: str,
    output_filename: str,
    name_map_path: str,
    output_dir: str,
    significance_map_raw: Optional[Dict[Any, float]] = None,
    significance_map_fdr: Optional[Dict[Any, float]] = None,
    alpha: float = 0.05
):
    """
    CORRECTED: Creates a true split-violin plot comparing each group's distribution
    to the total population's distribution.
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    plt.style.use('seaborn-v0_8-whitegrid')
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    fig, ax = plt.subplots(figsize=(max(12, n_groups * 2), 8))

    # --- Data Preparation for Split Violin ---
    plot_data_list = []
    for group in groups:
        # Population data for the left side of the violin
        for val in population_df[variable].dropna():
            plot_data_list.append({'value': val, 'cluster_group': f'Cluster {group}', 'status': 'Population'})
        # Cluster data for the right side of the violin
        for val in df[df[group_col] == group][variable].dropna():
            plot_data_list.append({'value': val, 'cluster_group': f'Cluster {group}', 'status': 'Cluster'})
    
    if not plot_data_list:
        print(f"✗ No data to plot for {variable}.")
        plt.close()
        return

    plot_df = pd.DataFrame(plot_data_list)

    # --- Plotting ---
    sns.violinplot(
        data=plot_df, x='cluster_group', y='value', hue='status',
        split=True, inner='quart',
        palette={'Population': '#5B9BD5', 'Cluster': '#E07B39'},
        ax=ax
    )

    # --- Annotation and Styling ---
    y_max = plot_df['value'].max()
    for i, group in enumerate(groups):
        p_raw = significance_map_raw.get(group) if significance_map_raw else None
        p_fdr = significance_map_fdr.get(group) if significance_map_fdr else None
        _annotate_significance(ax, i, y_max * 1.02, p_raw, p_fdr, alpha)

    nice_variable_name = get_nice_name(variable, name_map)
    ax.set_title(f'Distribution of {nice_variable_name}: Population vs Clusters', fontsize=16, weight='bold')
    ax.set_ylabel(nice_variable_name, fontsize=14)
    ax.set_xlabel('Clusters', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Split-violin plot saved to: {output_path}")


def plot_stacked_bar_comparison(
    df: pd.DataFrame, population_df: pd.DataFrame, variable: str, group_col: str,
    output_filename: str, name_map_path: str, output_dir: str,
    significance_map_raw: Optional[Dict[Any, float]] = None,
    significance_map_fdr: Optional[Dict[Any, float]] = None, alpha: float = 0.05
):
    """FINALIZED: Creates a stacked bar chart with the correct stacking order."""
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    plt.style.use('seaborn-v0_8-whitegrid')
    groups = sorted(df[group_col].unique())
    
    results = []
    pop_prop_1 = population_df[variable].mean() * 100
    results.append({'group': 'Reference', 'prop_0': 100 - pop_prop_1, 'prop_1': pop_prop_1, 'n': len(population_df)})
    for group in groups:
        cluster_data = df[df[group_col] == group]
        prop_1 = cluster_data[variable].mean() * 100
        results.append({'group': f'Cluster {group}', 'prop_0': 100 - prop_1, 'prop_1': prop_1, 'n': len(cluster_data)})
    plot_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 1.2), 7))
    x_pos = np.arange(len(plot_df))
    
    # --- Corrected Stacking Order ---
    ax.bar(x_pos, plot_df['prop_1'], label='Achieved / Class 1', color=ACHIEVED_COLOR, alpha=0.8)
    ax.bar(x_pos, plot_df['prop_0'], bottom=plot_df['prop_1'], label='Not Achieved / Class 0', color=NOT_ACHIEVED_COLOR, alpha=0.8)
    
    ax.axhline(y=pop_prop_1, color='black', linestyle='--', linewidth=2, alpha=0.8, label=f'Population Mean ({pop_prop_1:.1f}%)')
    
    for i, row in plot_df.iterrows():
        ax.text(i, 102, f'n={row["n"]}', ha='center', va='bottom', fontweight='bold')
        if row['group'] != 'Reference':
            cluster_num = int(row['group'].split(' ')[-1])
            p_raw = significance_map_raw.get(cluster_num) if significance_map_raw else None
            p_fdr = significance_map_fdr.get(cluster_num) if significance_map_fdr else None
            _annotate_significance(ax, i, 108, p_raw, p_fdr, alpha)

    nice_variable_name = get_nice_name(variable, name_map)
    ax.set_title(f'Proportion of {nice_variable_name} vs Population', fontsize=16, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14); ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylim(0, 120); ax.set_xticks(x_pos); ax.set_xticklabels(plot_df['group'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Corrected stacked bar plot saved to: {os.path.join(output_dir, output_filename)}")

def plot_multi_lollipop(
    data_df: pd.DataFrame, output_filename: str, name_map_path: str, output_dir: str
):
    """
    FINALIZED: Creates the multi-variable lollipop plot, corrected for y-axis
    labeling and overall robustness.
    """
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Ensure value column is numeric
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
    data_df.dropna(subset=['value'], inplace=True)
    if data_df.empty:
        print(f"✗ No valid data for lollipop plot.")
        return

    variables = data_df['variable'].unique()
    # Sort clusters numerically based on the integer after 'Cluster '
    clusters = sorted(data_df['cluster'].unique(), key=lambda x: int(x.split(' ')[-1]))
    
    fig, ax = plt.subplots(figsize=(12, 2 + len(variables) * len(clusters) * 0.2))
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    cluster_colors = {cluster: color for cluster, color in zip(clusters, colors)}

    y_pos = 0
    y_ticks, y_tick_labels = [], []
    
    for var in variables:
        var_data = data_df[data_df['variable'] == var].set_index('cluster').reindex(clusters).dropna()
        if var_data.empty: continue

        # Define the range of y-positions for the current variable
        y_range = np.arange(y_pos, y_pos + len(var_data))
        
        ax.hlines(y=y_range, xmin=0, xmax=var_data['value'], colors=[cluster_colors.get(c, 'gray') for c in var_data.index], linewidth=2)
        ax.scatter(var_data['value'], y_range, c=[cluster_colors.get(c, 'gray') for c in var_data.index], s=50, zorder=10)
        
        # Store the midpoint for the variable label
        if y_range.size > 0:
            y_ticks.append(np.mean(y_range))
            y_tick_labels.append(get_nice_name(var, name_map))
        
        y_pos += len(var_data)
        # Add a gap between variable groups
        if len(variables) > 1:
            y_pos += 0.5 
            if var != variables[-1]:
                ax.axhline(y=y_pos - 0.75, color='grey', linestyle='-', linewidth=0.5)

    ax.set_yticks(y_ticks); ax.set_yticklabels(y_tick_labels, fontsize=12)
    ax.invert_yaxis() # Display variables from top to bottom
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Multi-Cluster Comparison: Percent Change vs Population Mean', fontsize=16, weight='bold')
    ax.set_xlabel('Percent Change vs Population Mean (%)', fontsize=14)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)

    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=cluster, linestyle='None', markersize=8) for cluster, color in cluster_colors.items()]
    ax.legend(handles=legend_elements, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Multi-Lollipop plot saved to: {os.path.join(output_dir, output_filename)}")

def plot_forest(
    results_df: pd.DataFrame, output_filename: str, name_map_path: str, output_dir: str, 
    effect_col: str = 'effect', ci_lower_col: str = 'ci_lower', ci_upper_col: str = 'ci_upper', 
    label_col: str = 'group', title: str = 'Forest Plot'
):
    """FINALIZED: Creates a robust forest plot for pre-calculated effect sizes."""
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plot_data = results_df.sort_values(by=label_col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.5)))
    y_pos = np.arange(len(plot_data))

    errors = [plot_data[effect_col] - plot_data[ci_lower_col], plot_data[ci_upper_col] - plot_data[effect_col]]
    ax.errorbar(x=plot_data[effect_col], y=y_pos, xerr=errors, fmt='o', color='steelblue', ecolor='steelblue',
                elinewidth=2, capsize=5, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--')
    
    ax.set_yticks(y_pos); ax.set_yticklabels(plot_data[label_col])
    ax.set_title(title, fontsize=16, weight='bold'); ax.set_xlabel('Risk Difference', fontsize=14)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)

    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Forest plot saved to: {os.path.join(output_dir, output_filename)}")
