"""
CLUSTER_DESCRIPTIONS.PY
Comprehensive cluster analysis module for descriptive statistics and visualizations.

This module provides an integrated pipeline for cluster-based analysis:
- Statistical comparisons (cluster vs population mean)
- All visualization types (violin, stacked bar, lollipop, forest, heatmap)
- Configurable labels, colors, and plot parameters
- FDR-corrected significance testing
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import existing utilities
from descriptive_visualizations import (
    plot_distribution_comparison,
    plot_stacked_bar_comparison,
    plot_multi_lollipop,
    plot_forest,
    plot_wgc_cluster_heatmap
)
from descriptive_comparisons import (
    cluster_vs_population_mean_analysis,
    get_cause_cols,
    get_variable_types
)
from paper12_config import descriptive_comparisons_config


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class cluster_analysis_config:
    """Configuration for cluster analysis pipeline."""
    
    # Database paths
    cluster_db_path: str
    main_db_path: str
    
    # Table names
    cluster_table: str  # e.g., 'clust_labels_bl_nobc_bw_pam_goldstd'
    cluster_column: str  # e.g., 'pam_k7'
    outcome_table: str  # e.g., 'timetoevent_wgc_compl'
    population_table: str  # e.g., 'timetoevent_all'
    
    # Configuration files
    cluster_config_path: str  # Path to cluster_config.json
    name_map_path: str  # Path to human_readable_variable_names.json
    
    # Output settings
    output_dir: str
    output_db_path: str  # Where to save analysis tables
    output_table_prefix: str  # e.g., 'cluster_k7'
    
    # Analysis settings
    variables_to_analyze: List[str]  # Continuous variables for violin plots
    categorical_variables: List[str]  # Binary variables for stacked bars
    wgc_variables: List[str]  # WGC variables for heatmap
    row_order: List[Tuple[str, str]]  # For table formatting
    
    # Statistical settings
    fdr_correction: bool = True
    alpha: float = 0.05
    
    # Plot customization
    legend_labels: Optional[Dict[str, str]] = None  # e.g., {'achieved': 'Yes', 'not_achieved': 'No'}


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_cluster_data(config: cluster_analysis_config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge cluster assignments with outcome data.
    
    Returns:
        Tuple of (cluster_df, population_df)
    """
    print(f"Loading cluster data from {config.cluster_db_path}...")
    
    # Load cluster assignments
    with sqlite3.connect(config.cluster_db_path) as conn:
        cluster_labels = pd.read_sql_query(
            f"SELECT medical_record_id, {config.cluster_column} as cluster_id FROM {config.cluster_table}",
            conn
        )
    
    print(f"  ✓ Loaded {len(cluster_labels)} cluster assignments")
    print(f"  ✓ Clusters: {sorted(cluster_labels['cluster_id'].unique())}")
    
    # Load outcome data
    with sqlite3.connect(config.main_db_path) as conn:
        outcome_df = pd.read_sql_query(f"SELECT * FROM {config.outcome_table}", conn)
        population_df = pd.read_sql_query(f"SELECT * FROM {config.population_table}", conn)
    
    print(f"  ✓ Loaded {len(outcome_df)} outcome records")
    print(f"  ✓ Loaded {len(population_df)} population records")
    
    # Merge cluster assignments with outcomes
    cluster_df = outcome_df.merge(
        cluster_labels,
        on='medical_record_id',
        how='inner'
    )
    
    print(f"  ✓ Merged data: {len(cluster_df)} records with clusters")
    
    return cluster_df, population_df


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def run_cluster_statistical_analysis(
    cluster_df: pd.DataFrame,
    config: cluster_analysis_config
) -> pd.DataFrame:
    """
    Run cluster vs population mean analysis for WGC variables.
    Generates both detailed and publication-ready tables.
    
    Returns:
        DataFrame with analysis results (for heatmap generation)
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Cluster vs Population Mean")
    print("="*60)
    
    # Create descriptive_comparisons_config for the analysis
    analysis_config = descriptive_comparisons_config(
        analysis_name=f'{config.output_table_prefix}_analysis',
        input_cohort_name=config.outcome_table,
        mother_cohort_name=config.population_table,
        row_order=config.row_order,
        demographic_output_table='',
        demographic_strata=[],
        wgc_output_table='',
        wgc_strata=[],
        cluster_vs_mean_output_table=f'{config.output_table_prefix}_wgc_vs_mean',
        fdr_correction=config.fdr_correction
    )
    
    # Run analysis
    with sqlite3.connect(config.output_db_path) as conn:
        results_df = cluster_vs_population_mean_analysis(
            df=cluster_df,
            config=analysis_config,
            conn=conn,
            cluster_col='cluster_id'
        )
    
    print(f"✓ Analysis complete. Tables saved to {config.output_db_path}")
    
    return results_df


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def generate_violin_plots(
    cluster_df: pd.DataFrame,
    population_df: pd.DataFrame,
    config: cluster_analysis_config,
    significance_maps: Optional[Dict] = None
):
    """Generate violin plots for continuous variables."""
    print("\n" + "="*60)
    print("GENERATING VIOLIN PLOTS")
    print("="*60)
    
    for variable in config.variables_to_analyze:
        print(f"  Generating violin plot for: {variable}")
        
        try:
            plot_distribution_comparison(
                df=cluster_df,
                population_df=population_df,
                variable=variable,
                group_col='cluster_id',
                output_filename=f'{config.output_table_prefix}_{variable}_violin.png',
                name_map_path=config.name_map_path,
                cluster_config_path=config.cluster_config_path,
                output_dir=config.output_dir,
                significance_map_raw=significance_maps.get('raw', {}).get(variable) if significance_maps else None,
                significance_map_fdr=significance_maps.get('fdr', {}).get(variable) if significance_maps else None,
                alpha=config.alpha
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")


def generate_stacked_bar_plots(
    cluster_df: pd.DataFrame,
    population_df: pd.DataFrame,
    config: cluster_analysis_config,
    significance_maps: Optional[Dict] = None
):
    """Generate stacked bar plots for categorical variables."""
    print("\n" + "="*60)
    print("GENERATING STACKED BAR PLOTS")
    print("="*60)
    
    for variable in config.categorical_variables:
        print(f"  Generating stacked bar for: {variable}")
        
        try:
            plot_stacked_bar_comparison(
                df=cluster_df,
                population_df=population_df,
                variable=variable,
                group_col='cluster_id',
                output_filename=f'{config.output_table_prefix}_{variable}_bar.png',
                name_map_path=config.name_map_path,
                cluster_config_path=config.cluster_config_path,
                output_dir=config.output_dir,
                significance_map_raw=significance_maps.get('raw', {}).get(variable) if significance_maps else None,
                significance_map_fdr=significance_maps.get('fdr', {}).get(variable) if significance_maps else None,
                alpha=config.alpha
            )
            print(f"    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")


def generate_lollipop_plot(
    cluster_df: pd.DataFrame,
    population_df: pd.DataFrame,
    config: cluster_analysis_config
):
    """Generate multi-variable lollipop plot showing percent change from population mean."""
    print("\n" + "="*60)
    print("GENERATING LOLLIPOP PLOT")
    print("="*60)
    
    # Prepare lollipop data
    lollipop_data = []
    
    for variable in config.variables_to_analyze:
        pop_mean = population_df[variable].mean()
        
        for cluster_id in sorted(cluster_df['cluster_id'].unique()):
            cluster_subset = cluster_df[cluster_df['cluster_id'] == cluster_id]
            cluster_mean = cluster_subset[variable].mean()
            
            if pd.notna(pop_mean) and pop_mean != 0:
                pct_change = ((cluster_mean - pop_mean) / pop_mean) * 100
                
                lollipop_data.append({
                    'variable': variable,
                    'cluster': f'Cluster {int(cluster_id)}',
                    'value': pct_change
                })
    
    if lollipop_data:
        lollipop_df = pd.DataFrame(lollipop_data)
        
        try:
            plot_multi_lollipop(
                data_df=lollipop_df,
                output_filename=f'{config.output_table_prefix}_multi_lollipop.png',
                name_map_path=config.name_map_path,
                cluster_config_path=config.cluster_config_path,
                output_dir=config.output_dir
            )
            print("  ✓ Lollipop plot saved")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ No data for lollipop plot")


def generate_forest_plot(
    cluster_df: pd.DataFrame,
    population_df: pd.DataFrame,
    config: cluster_analysis_config,
    effect_type: str = 'RD'
):
    """Generate forest plot for effect sizes (risk differences or ratios)."""
    print("\n" + "="*60)
    print(f"GENERATING FOREST PLOT ({effect_type})")
    print("="*60)
    
    # Calculate effect sizes for a key binary outcome
    # Using first categorical variable as example
    if not config.categorical_variables:
        print("  ⚠ No categorical variables specified for forest plot")
        return
    
    outcome_var = config.categorical_variables[0]
    forest_data = []
    
    pop_rate = population_df[outcome_var].mean()
    
    for cluster_id in sorted(cluster_df['cluster_id'].unique()):
        cluster_subset = cluster_df[cluster_df['cluster_id'] == cluster_id]
        cluster_rate = cluster_subset[outcome_var].mean()
        
        if effect_type == 'RD':
            # Risk difference
            effect = (cluster_rate - pop_rate) * 100  # As percentage
            # Simple CI (would need proper calculation in production)
            se = np.sqrt(cluster_rate * (1 - cluster_rate) / len(cluster_subset))
            ci_lower = effect - 1.96 * se * 100
            ci_upper = effect + 1.96 * se * 100
        else:  # RR
            # Risk ratio
            effect = cluster_rate / pop_rate if pop_rate > 0 else 1
            # Simple CI (would need proper calculation in production)
            ci_lower = effect * 0.8
            ci_upper = effect * 1.2
        
        forest_data.append({
            'group': f'Cluster {int(cluster_id)}',
            'effect': effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
    
    if forest_data:
        forest_df = pd.DataFrame(forest_data)
        
        try:
            plot_forest(
                results_df=forest_df,
                output_filename=f'{config.output_table_prefix}_forest_{effect_type.lower()}.png',
                name_map_path=config.name_map_path,
                output_dir=config.output_dir,
                cluster_config_path=config.cluster_config_path,
                effect_type=effect_type,
                title=f'Cluster Effect Sizes: {outcome_var}'
            )
            print(f"  ✓ Forest plot ({effect_type}) saved")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def generate_heatmap(
    results_df: pd.DataFrame,
    config: cluster_analysis_config
):
    """Generate heatmap from cluster vs population analysis results."""
    print("\n" + "="*60)
    print("GENERATING HEATMAP")
    print("="*60)
    
    if results_df is None:
        print("  ⚠ No analysis results available for heatmap")
        return
    
    # Extract prevalence data from results
    prevalence_data = []
    sig_map_raw = {}
    sig_map_fdr = {}
    
    clusters = sorted([int(col.split()[1]) for col in results_df.columns if 'Cluster' in col and 'Mean/N' in col])
    
    for _, row in results_df.iterrows():
        wgc_var = row['Variable']
        
        # Skip delimiter rows and non-WGC variables
        if wgc_var not in config.wgc_variables:
            continue
        
        sig_map_raw[wgc_var] = {}
        sig_map_fdr[wgc_var] = {}
        
        for cluster_id in clusters:
            cluster_label = f'Cluster {cluster_id}'
            
            # Parse "n (%)" format from Mean/N column
            mean_n_col = f'{cluster_label}: Mean/N'
            if mean_n_col not in row:
                continue
            
            mean_n_str = str(row[mean_n_col])
            
            try:
                # Extract n and percentage
                n = int(mean_n_str.split('(')[0].strip())
                pct = float(mean_n_str.split('(')[1].split('%')[0])
                
                prevalence_data.append({
                    'wgc_variable': wgc_var,
                    'cluster_id': cluster_id,
                    'prevalence_%': pct,
                    'n': n
                })
                
                # Extract p-values
                p_raw_col = f'{cluster_label}: p-value'
                p_fdr_col = f'{cluster_label}: p-value (FDR-corrected)'
                
                if p_raw_col in row:
                    sig_map_raw[wgc_var][cluster_id] = row[p_raw_col]
                if p_fdr_col in row:
                    sig_map_fdr[wgc_var][cluster_id] = row[p_fdr_col]
            except (ValueError, IndexError) as e:
                print(f"  ⚠ Could not parse data for {wgc_var}, cluster {cluster_id}: {e}")
                continue
    
    if prevalence_data:
        prevalence_df = pd.DataFrame(prevalence_data)
        
        try:
            plot_wgc_cluster_heatmap(
                prevalence_df=prevalence_df,
                output_filename=f'{config.output_table_prefix}_wgc_heatmap.png',
                name_map_path=config.name_map_path,
                cluster_config_path=config.cluster_config_path,
                output_dir=config.output_dir,
                significance_map_raw=sig_map_raw,
                significance_map_fdr=sig_map_fdr,
                alpha=config.alpha
            )
            print("  ✓ Heatmap saved")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ No prevalence data extracted for heatmap")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_cluster_analysis_pipeline(config: cluster_analysis_config):
    """
    Run complete cluster analysis pipeline:
    1. Load and merge data
    2. Statistical analysis (cluster vs population mean)
    3. Generate all visualizations
    """
    print("="*60)
    print("CLUSTER ANALYSIS PIPELINE")
    print("="*60)
    print(f"Output directory: {config.output_dir}")
    print(f"Output database: {config.output_db_path}")
    print("="*60)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 1: Load data
    cluster_df, population_df = load_cluster_data(config)
    
    # Step 2: Statistical analysis
    results_df = run_cluster_statistical_analysis(cluster_df, config)
    
    # Step 3: Generate visualizations
    generate_violin_plots(cluster_df, population_df, config)
    generate_stacked_bar_plots(cluster_df, population_df, config)
    generate_lollipop_plot(cluster_df, population_df, config)
    generate_forest_plot(cluster_df, population_df, config, effect_type='RD')
    generate_heatmap(results_df, config)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"All outputs saved to: {config.output_dir}")
    print(f"Analysis tables saved to: {config.output_db_path}")
    print("="*60)
