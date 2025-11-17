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

def load_cluster_config(json_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load cluster labels and colors from JSON file.
    Returns dict with 'cluster_labels' and 'cluster_colors' keys.
    Falls back to defaults if file doesn't exist.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: Cluster config file not found at '{json_path}'. Using defaults.")
        return {
            'cluster_labels': {},
            'cluster_colors': {}
        }
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load cluster config '{json_path}'. Error: {e}")
        return {'cluster_labels': {}, 'cluster_colors': {}}

def get_cluster_label(cluster_id: int, cluster_config: Dict) -> str:
    """Get human-readable label for cluster, with fallback to default"""
    labels = cluster_config.get('cluster_labels', {})
    return labels.get(str(cluster_id), f'Cluster {cluster_id}')

def get_cluster_color(cluster_id: int, cluster_config: Dict, default_palette: list) -> str:
    """Get color for cluster, with fallback to default palette"""
    colors = cluster_config.get('cluster_colors', {})
    if str(cluster_id) in colors:
        return colors[str(cluster_id)]
    # Fallback to default palette
    return default_palette[cluster_id % len(default_palette)]

def validate_cluster_config(cluster_ids: List[int], cluster_config: Dict) -> None:
    """
    Validate that all cluster IDs in data have corresponding labels/colors.
    Print warnings for missing cluster IDs and use defaults.
    """
    if not cluster_config:
        return
    
    labels = cluster_config.get('cluster_labels', {})
    colors = cluster_config.get('cluster_colors', {})
    
    missing_labels = [cid for cid in cluster_ids if str(cid) not in labels]
    missing_colors = [cid for cid in cluster_ids if str(cid) not in colors]
    
    if missing_labels:
        print(f"⚠️ Warning: Cluster IDs {missing_labels} not found in cluster_labels config. Using default labels.")
    
    if missing_colors:
        print(f"⚠️ Warning: Cluster IDs {missing_colors} not found in cluster_colors config. Using default colors.")

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
    alpha: float = 0.05,
    cluster_config_path: Optional[str] = None
):
    """
    CORRECTED: Creates a true split-violin plot comparing each group's distribution
    to the total population's distribution.
    """
    # Validate required columns exist
    required_cols = [group_col, variable]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}")
    
    if variable not in population_df.columns:
        raise ValueError(f"Missing required column '{variable}' in population_df")
    
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    # Load cluster config if provided
    cluster_config = {}
    if cluster_config_path:
        cluster_config = load_cluster_config(cluster_config_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    fig, ax = plt.subplots(figsize=(max(12, n_groups * 2), 8))

    # Check for empty data after dropping missing values
    if df[variable].dropna().empty:
        print(f"⚠️ Warning: No valid data for variable '{variable}' in df after dropping missing values. Skipping plot.")
        return
    
    if population_df[variable].dropna().empty:
        print(f"⚠️ Warning: No valid data for variable '{variable}' in population_df after dropping missing values. Skipping plot.")
        return
    
    # --- Data Preparation for Split Violin ---
    plot_data_list = []
    group_labels = []
    for group in groups:
        # Get cluster label if cluster config is provided
        if cluster_config:
            group_label = get_cluster_label(group, cluster_config)
        else:
            group_label = f'Cluster {group}'
        group_labels.append(group_label)
        
        # Population data for the left side of the violin
        for val in population_df[variable].dropna():
            plot_data_list.append({'value': val, 'cluster_group': group_label, 'status': 'Population'})
        # Cluster data for the right side of the violin
        for val in df[df[group_col] == group][variable].dropna():
            plot_data_list.append({'value': val, 'cluster_group': group_label, 'status': 'Cluster'})
    
    if not plot_data_list:
        print(f"⚠️ Warning: No data to plot for variable '{variable}'. Skipping plot.")
        return

    plot_df = pd.DataFrame(plot_data_list)

    # --- Plotting with cluster colors if available ---
    if cluster_config and cluster_config.get('cluster_colors'):
        # Create color palette for clusters
        cluster_colors = {}
        for i, group in enumerate(groups):
            color = get_cluster_color(group, cluster_config, DEFAULT_PALETTE)
            cluster_colors[group_labels[i]] = color
        
        # Use custom colors for the cluster side
        palette = {'Population': '#5B9BD5', 'Cluster': '#E07B39'}
    else:
        palette = {'Population': '#5B9BD5', 'Cluster': '#E07B39'}
    
    sns.violinplot(
        data=plot_df, x='cluster_group', y='value', hue='status',
        split=True, inner='quart',
        palette=palette,
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
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Split-violin plot saved to: {output_path}")

def plot_stacked_bar_comparison(
    df: pd.DataFrame, population_df: pd.DataFrame, variable: str, group_col: str,
    output_filename: str, name_map_path: str, output_dir: str,
    significance_map_raw: Optional[Dict[Any, float]] = None,
    significance_map_fdr: Optional[Dict[Any, float]] = None, alpha: float = 0.05,
    cluster_config_path: Optional[str] = None
):
    """FINALIZED: Creates a stacked bar chart with the correct stacking order."""
    # Validate required columns exist
    required_cols = [group_col, variable]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}")
    
    if variable not in population_df.columns:
        raise ValueError(f"Missing required column '{variable}' in population_df")
    
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    # Load cluster config if provided
    cluster_config = {}
    if cluster_config_path:
        cluster_config = load_cluster_config(cluster_config_path)
    
    # Check for empty data after dropping missing values
    if df[variable].dropna().empty:
        print(f"⚠️ Warning: No valid data for variable '{variable}' in df after dropping missing values. Skipping plot.")
        return
    
    if population_df[variable].dropna().empty:
        print(f"⚠️ Warning: No valid data for variable '{variable}' in population_df after dropping missing values. Skipping plot.")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    groups = sorted(df[group_col].unique())
    
    # Validate cluster config if provided
    if cluster_config_path and cluster_config:
        validate_cluster_config(list(groups), cluster_config)
    
    results = []
    pop_prop_1 = population_df[variable].mean() * 100
    results.append({'group': 'Reference', 'prop_0': 100 - pop_prop_1, 'prop_1': pop_prop_1, 'n': len(population_df), 'group_id': None})
    for group in groups:
        cluster_data = df[df[group_col] == group]
        prop_1 = cluster_data[variable].mean() * 100
        # Get cluster label if cluster config is provided
        if cluster_config:
            group_label = get_cluster_label(group, cluster_config)
        else:
            group_label = f'Cluster {group}'
        results.append({'group': group_label, 'prop_0': 100 - prop_1, 'prop_1': prop_1, 'n': len(cluster_data), 'group_id': group})
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
            cluster_num = row['group_id']
            p_raw = significance_map_raw.get(cluster_num) if significance_map_raw else None
            p_fdr = significance_map_fdr.get(cluster_num) if significance_map_fdr else None
            _annotate_significance(ax, i, 108, p_raw, p_fdr, alpha)

    nice_variable_name = get_nice_name(variable, name_map)
    ax.set_title(f'Proportion of {nice_variable_name} vs Population', fontsize=16, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14); ax.set_xlabel('Groups', fontsize=14)
    ax.set_ylim(0, 120); ax.set_xticks(x_pos); ax.set_xticklabels(plot_df['group'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    
    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Corrected stacked bar plot saved to: {os.path.join(output_dir, output_filename)}")

def plot_multi_lollipop(
    data_df: pd.DataFrame, output_filename: str, name_map_path: str, output_dir: str,
    cluster_config_path: Optional[str] = None
    ):
    """
    FINALIZED: Creates the multi-variable lollipop plot, corrected for y-axis
    labeling and overall robustness.
    """
    # Validate required columns exist
    required_cols = ['variable', 'cluster', 'value']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data_df: {missing_cols}")
    
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    # Load cluster config if provided
    cluster_config = {}
    if cluster_config_path:
        cluster_config = load_cluster_config(cluster_config_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Ensure value column is numeric
    data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
    data_df.dropna(subset=['value'], inplace=True)
    if data_df.empty:
        print(f"⚠️ Warning: No valid data for lollipop plot after dropping missing values. Skipping plot.")
        return

    variables = data_df['variable'].unique()
    # Sort clusters numerically based on the integer after 'Cluster '
    clusters = sorted(data_df['cluster'].unique(), key=lambda x: int(x.split(' ')[-1]))
    
    # Validate cluster config if provided
    if cluster_config_path and cluster_config:
        cluster_ids = [int(c.split(' ')[-1]) for c in clusters]
        validate_cluster_config(cluster_ids, cluster_config)
    
    fig, ax = plt.subplots(figsize=(12, 2 + len(variables) * len(clusters) * 0.2))
    
    # Use cluster colors from config if available
    cluster_colors = {}
    for i, cluster in enumerate(clusters):
        cluster_id = int(cluster.split(' ')[-1])
        if cluster_config:
            color = get_cluster_color(cluster_id, cluster_config, DEFAULT_PALETTE)
            # Get cluster label for legend
            cluster_label = get_cluster_label(cluster_id, cluster_config)
            cluster_colors[cluster] = {'color': color, 'label': cluster_label}
        else:
            color = plt.cm.Set3(np.linspace(0, 1, len(clusters)))[i]
            cluster_colors[cluster] = {'color': color, 'label': cluster}

    y_pos = 0
    y_ticks, y_tick_labels = [], []
    
    for var in variables:
        var_data = data_df[data_df['variable'] == var].set_index('cluster').reindex(clusters).dropna()
        if var_data.empty: continue

        # Define the range of y-positions for the current variable
        y_range = np.arange(y_pos, y_pos + len(var_data))
        
        ax.hlines(y=y_range, xmin=0, xmax=var_data['value'], 
                 colors=[cluster_colors.get(c, {}).get('color', 'gray') for c in var_data.index], linewidth=2)
        ax.scatter(var_data['value'], y_range, 
                  c=[cluster_colors.get(c, {}).get('color', 'gray') for c in var_data.index], s=50, zorder=10)
        
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

    # Use cluster labels in legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color=cluster_colors[cluster]['color'], 
                                  label=cluster_colors[cluster]['label'], linestyle='None', markersize=8) 
                      for cluster in clusters]
    ax.legend(handles=legend_elements, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Multi-Lollipop plot saved to: {os.path.join(output_dir, output_filename)}")

def plot_forest(
    results_df: pd.DataFrame, output_filename: str, name_map_path: str, output_dir: str, 
    effect_col: str = 'effect', ci_lower_col: str = 'ci_lower', ci_upper_col: str = 'ci_upper', 
    label_col: str = 'group', title: str = 'Forest Plot',
    cluster_config_path: Optional[str] = None,
    effect_type: str = 'RD'
):
    """
    Creates a forest plot for pre-calculated effect sizes with secondary y-axis showing effect sizes and CIs.
    
    Args:
        results_df: DataFrame with effect sizes and confidence intervals
        output_filename: Name for saved plot file
        name_map_path: Path to human_readable_variable_names.json
        output_dir: Directory to save plot
        effect_col: Column name for effect size (default: 'effect')
        ci_lower_col: Column name for lower CI bound (default: 'ci_lower')
        ci_upper_col: Column name for upper CI bound (default: 'ci_upper')
        label_col: Column name for group labels (default: 'group')
        title: Plot title (default: 'Forest Plot')
        cluster_config_path: Optional path to cluster_config.json for cluster label support
        effect_type: Type of effect measure - 'RD' for risk difference or 'RR' for risk ratio (default: 'RD')
    """
    # Validate required columns exist
    required_cols = [effect_col, ci_lower_col, ci_upper_col, label_col]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in results_df: {missing_cols}")
    
    os.makedirs(output_dir, exist_ok=True)
    name_map = load_name_map(name_map_path)
    
    # Load cluster config if provided
    cluster_config = {}
    if cluster_config_path:
        cluster_config = load_cluster_config(cluster_config_path)
    
    # Check for empty data
    if results_df.empty:
        print(f"⚠️ Warning: Empty DataFrame for forest plot. Skipping plot.")
        return
    
    # Validate cluster config if provided
    if cluster_config_path and cluster_config:
        # Extract cluster IDs from labels that contain "Cluster"
        cluster_ids = []
        for label in results_df[label_col]:
            if 'Cluster' in str(label):
                try:
                    cluster_id = int(str(label).split()[-1])
                    cluster_ids.append(cluster_id)
                except (ValueError, IndexError):
                    pass
        if cluster_ids:
            validate_cluster_config(cluster_ids, cluster_config)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plot_data = results_df.sort_values(by=label_col, ascending=False)
    fig, ax = plt.subplots(figsize=(14, max(6, len(plot_data) * 0.5)))
    y_pos = np.arange(len(plot_data))

    errors = [plot_data[effect_col] - plot_data[ci_lower_col], plot_data[ci_upper_col] - plot_data[effect_col]]
    ax.errorbar(x=plot_data[effect_col], y=y_pos, xerr=errors, fmt='o', color='steelblue', ecolor='steelblue',
                elinewidth=2, capsize=5, markersize=8)
    
    # Set reference line based on effect type
    ref_value = 1 if effect_type == 'RR' else 0
    ax.axvline(x=ref_value, color='red', linestyle='--', linewidth=2, label=f'Reference ({effect_type}={ref_value})')
    
    # Apply cluster labels if cluster config is provided
    y_labels = []
    for label in plot_data[label_col]:
        if cluster_config and 'Cluster' in str(label):
            # Extract cluster ID from label like "Cluster 0"
            try:
                cluster_id = int(str(label).split()[-1])
                y_labels.append(get_cluster_label(cluster_id, cluster_config))
            except (ValueError, IndexError):
                y_labels.append(label)
        else:
            y_labels.append(label)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=16, weight='bold')
    
    # Set x-axis label based on effect type
    x_label = 'Risk Ratio' if effect_type == 'RR' else 'Risk Difference'
    ax.set_xlabel(x_label, fontsize=14)
    ax.grid(True, axis='x', which='both', linestyle='--', linewidth=0.5)
    
    # Add secondary y-axis on the right with effect sizes and CIs
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    
    # Format labels based on effect type
    ci_labels = []
    for _, row in plot_data.iterrows():
        if effect_type == 'RR':
            label = f"RR: {row[effect_col]:.2f} [{row[ci_lower_col]:.2f}-{row[ci_upper_col]:.2f}]"
        else:  # RD
            label = f"RD: {row[effect_col]:.1f}% [{row[ci_lower_col]:.1f}-{row[ci_upper_col]:.1f}%]"
        ci_labels.append(label)
    
    ax2.set_yticklabels(ci_labels, fontsize=10)
    ax2.set_ylabel('Effect Size [95% CI]', fontsize=12)

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Forest plot saved to: {output_path}")

def plot_wgc_cluster_heatmap(
    prevalence_df: pd.DataFrame,
    output_filename: str,
    name_map_path: str,
    cluster_config_path: str,
    output_dir: str,
    significance_map_raw: Optional[Dict[str, Dict[int, float]]] = None,
    significance_map_fdr: Optional[Dict[str, Dict[int, float]]] = None,
    alpha: float = 0.05
):
    """
    Create a heatmap showing WGC prevalence across clusters.
    
    Args:
        prevalence_df: DataFrame with columns ['wgc_variable', 'cluster_id', 'prevalence_%', 'n']
        output_filename: Name for saved plot file
        name_map_path: Path to human_readable_variable_names.json
        cluster_config_path: Path to cluster_config.json
        output_dir: Directory to save plot
        significance_map_raw: {wgc_variable: {cluster_id: p_value}}
        significance_map_fdr: {wgc_variable: {cluster_id: fdr_p_value}}
        alpha: Significance threshold (default: 0.05)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configurations
    name_map = load_name_map(name_map_path)
    cluster_config = load_cluster_config(cluster_config_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Validate required columns
    required_cols = ['wgc_variable', 'cluster_id', 'prevalence_%', 'n']
    missing_cols = [col for col in required_cols if col not in prevalence_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in prevalence_df: {missing_cols}")
    
    # Check for empty data
    if prevalence_df.empty:
        print(f"⚠️ Warning: Empty DataFrame for heatmap. Skipping plot.")
        return
    
    # Pivot data to matrix format: rows=WGCs, columns=clusters
    matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='prevalence_%')
    
    # Get cluster labels for columns
    cluster_labels = [get_cluster_label(cid, cluster_config) for cid in matrix.columns]
    
    # Get WGC labels for rows
    wgc_labels = [get_nice_name(wgc, name_map) for wgc in matrix.index]
    
    # Also need n_matrix for sample counts
    n_matrix = prevalence_df.pivot(index='wgc_variable', columns='cluster_id', values='n')
    
    # Create annotations with "n (%)** " format
    annotations = np.empty_like(matrix, dtype=object)
    for i, wgc in enumerate(matrix.index):
        for j, cluster_id in enumerate(matrix.columns):
            pct = matrix.iloc[i, j]
            n = n_matrix.iloc[i, j]
            
            # Get significance marker
            sig_marker = ''
            if significance_map_fdr and wgc in significance_map_fdr:
                p_fdr = significance_map_fdr[wgc].get(cluster_id)
                if p_fdr is not None and p_fdr < alpha:
                    sig_marker = '**'
            if not sig_marker and significance_map_raw and wgc in significance_map_raw:
                p_raw = significance_map_raw[wgc].get(cluster_id)
                if p_raw is not None and p_raw < alpha:
                    sig_marker = '*'
            
            # Format: "n (%)** " where n is count, % is percentage, ** is significance
            if pd.notna(pct) and pd.notna(n):
                annotations[i, j] = f"{int(n)} ({pct:.1f}%){sig_marker}"
            else:
                annotations[i, j] = ""
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(cluster_labels) * 1.5), max(8, len(wgc_labels) * 0.6)))
    
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt='',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Prevalence (%)'},
        xticklabels=cluster_labels,
        yticklabels=wgc_labels,
        ax=ax,
        linewidths=0.5,
        linecolor='white'
    )
    
    ax.set_title('Weight Gain Cause Prevalence by Cluster', fontsize=16, weight='bold')
    ax.set_xlabel('Clusters', fontsize=14)
    ax.set_ylabel('Weight Gain Causes', fontsize=14)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()  # Display in notebook
    plt.close()
    print(f"✓ Heatmap saved to: {output_path}")
