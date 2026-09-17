import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Optional
import argparse
from scipy.stats import ttest_ind

# Import FDR correction utilities and Welch's t-test from descriptive_comparisons
try:
    from fdr_correction_utils import apply_fdr_correction
except ImportError:
    print("Warning: FDR correction utilities not found. Statistical testing will use raw p-values only.")
    apply_fdr_correction = None


@dataclass
class ViolinPlotConfig:
    """Configuration for WGC violin plots"""
    # Database and table settings
    db_path: str = "dbs/pnk_db2_p2_in.sqlite"
    table_name: str = "timetoevent_wgc_compl"
    
    # Variables to plot
    continuous_vars: List[str] = None
    
    # Output settings
    output_dir: str = "../outputs/violin_plots"
    figure_size: tuple = (12, 8)
    dpi: int = 300
    
    def __post_init__(self):
        if self.continuous_vars is None:
            self.continuous_vars = [
                "baseline_weight_kg",
                "total_wl_%",  # Changed from total_wl_kg to total_wl_%
                "total_followup_days"
            ]


def welchs_ttest(series1, series2):
    """Performs Welch's t-test and returns the raw p-value (from descriptive_comparisons.py)"""
    s1 = pd.to_numeric(series1, errors='coerce').dropna()
    s2 = pd.to_numeric(series2, errors='coerce').dropna()
    if len(s1) < 2 or len(s2) < 2:
        return np.nan
    _, p_val = ttest_ind(s1, s2, equal_var=False, nan_policy='omit')
    return p_val


def get_significance_markers(p_values, fdr_corrected_p_values=None):
    """
    Get significance markers based on p-values and FDR-corrected p-values
    Returns: list of markers ('', '*', '**')
    """
    markers = []
    
    for i, p_val in enumerate(p_values):
        if pd.isna(p_val):
            markers.append('')
        else:
            # Check FDR-corrected significance first
            if fdr_corrected_p_values is not None and i < len(fdr_corrected_p_values) and not pd.isna(fdr_corrected_p_values[i]):
                if fdr_corrected_p_values[i] < 0.05:
                    markers.append('**')  # Significant after FDR correction
                elif p_val < 0.05:
                    markers.append('*')   # Significant before FDR correction only
                else:
                    markers.append('')    # Not significant
            else:
                # Only raw p-values available
                if p_val < 0.05:
                    markers.append('*')   # Significant
                else:
                    markers.append('')    # Not significant
    
    return markers


def get_wgc_columns(df):
    """
    Identify weight gain cause columns from the dataframe.
    Based on the pattern from descriptive_comparisons.py
    """
    # Common WGC column patterns - adjust based on your actual column names
    potential_wgc_cols = [
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
        "none_of_above"  # Added the missing 12th category
    ]
    
    # Filter to only columns that actually exist in the dataframe
    wgc_cols = [col for col in potential_wgc_cols if col in df.columns]
    
    # If none found, try to detect binary columns that might be WGC
    if not wgc_cols:
        binary_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and set(df[col].dropna().unique()).issubset({0, 1}):
                binary_cols.append(col)
        
        # Filter out obvious non-WGC columns
        exclude_patterns = ['id', 'date', 'age', 'sex', 'bmi', 'weight', 'dropout', 'achieved']
        wgc_cols = [col for col in binary_cols 
                   if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    return wgc_cols


def create_wgc_violin_plot(df, variable, wgc_columns, config):
    """
    Create violin plots comparing variable distributions by WGC presence/absence
    """
    # Calculate number of subplots needed
    n_causes = len(wgc_columns)
    n_cols = min(3, n_causes)  # Max 3 columns
    n_rows = (n_causes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(config.figure_size[0], config.figure_size[1] * n_rows / 2))
    
    # Handle single subplot case
    if n_causes == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_causes > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#FF6B6B', '#4ECDC4']  # Red for absent, teal for present
    
    for i, cause in enumerate(wgc_columns):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # Prepare data for this cause
        plot_data = []
        
        # Group 0: Cause absent
        absent_data = df[df[cause] == 0][variable].dropna()
        for val in absent_data:
            plot_data.append({'value': val, 'group': 'Absent', 'cause': cause})
        
        # Group 1: Cause present  
        present_data = df[df[cause] == 1][variable].dropna()
        for val in present_data:
            plot_data.append({'value': val, 'group': 'Present', 'cause': cause})
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{cause.replace("_", " ").title()}')
            continue
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        violin_parts = ax.violinplot([absent_data, present_data], 
                                   positions=[0, 1], 
                                   showmeans=False, 
                                   showmedians=True,
                                   showextrema=False)
        
        # Color the violins
        for j, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[j])
            pc.set_alpha(0.7)
        
        # Add quartile lines
        quartiles_absent = np.percentile(absent_data, [25, 75]) if len(absent_data) > 0 else [0, 0]
        quartiles_present = np.percentile(present_data, [25, 75]) if len(present_data) > 0 else [0, 0]
        
        # Draw IQR lines
        if len(absent_data) > 0:
            ax.vlines(0, quartiles_absent[0], quartiles_absent[1], color='black', linestyle='-', lw=2)
        if len(present_data) > 0:
            ax.vlines(1, quartiles_present[0], quartiles_present[1], color='black', linestyle='-', lw=2)
        
        # Customize subplot
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Absent', 'Present'])
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_title(f'{cause.replace("_", " ").title()}\n(n={len(absent_data)} vs {len(present_data)})')
        
        # Add median values as text
        if len(absent_data) > 0:
            median_absent = np.median(absent_data)
            ax.text(0, ax.get_ylim()[1] * 0.95, f'Median: {median_absent:.1f}', 
                   ha='center', va='top', fontsize=8)
        
        if len(present_data) > 0:
            median_present = np.median(present_data)
            ax.text(1, ax.get_ylim()[1] * 0.95, f'Median: {median_present:.1f}', 
                   ha='center', va='top', fontsize=8)
    
    # Hide empty subplots
    for i in range(n_causes, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Distribution of {variable.replace("_", " ").title()} by Weight Gain Causes', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def create_enhanced_split_violin_plot(df, variable, wgc_columns, config):
    """
    Create enhanced split violin plots with statistical testing and improved visuals
    """
    # First, let's investigate negative values issue and clean data
    print(f"Investigating data for {variable}:")
    data_series = pd.to_numeric(df[variable], errors='coerce')
    print(f"  Min value: {data_series.min()}")
    print(f"  Max value: {data_series.max()}")
    print(f"  Negative values count: {(data_series < 0).sum()}")
    
    # For follow-up days, negative values are impossible - filter them out
    if variable == 'total_followup_days' and (data_series < 0).sum() > 0:
        print(f"  WARNING: Found {(data_series < 0).sum()} negative follow-up days - filtering them out")
        df = df[df[variable] >= 0].copy()
        data_series = pd.to_numeric(df[variable], errors='coerce')
        print(f"  After filtering - Min: {data_series.min()}, Max: {data_series.max()}")
    
    # Prepare data for plotting and statistical testing
    plot_data = []
    p_values = []
    test_results = []
    
    print(f"Performing statistical tests for {variable}...")
    
    for cause in wgc_columns:
        # Get data for absent and present groups
        absent_data = df[df[cause] == 0][variable].dropna()
        present_data = df[df[cause] == 1][variable].dropna()
        
        # Perform Welch's t-test
        p_val = welchs_ttest(absent_data, present_data)
        p_values.append(p_val)
        
        # Store test results
        test_results.append({
            'cause': cause,
            'n_absent': len(absent_data),
            'n_present': len(present_data),
            'median_absent': absent_data.median() if len(absent_data) > 0 else np.nan,
            'median_present': present_data.median() if len(present_data) > 0 else np.nan,
            'p_value': p_val
        })
        
        if not pd.isna(p_val):
            print(f"  {cause}: p-value = {p_val:.4f}")
        else:
            print(f"  {cause}: insufficient data")
        
        # Add data for plotting with proper indexing
        for val in absent_data:
            plot_data.append({
                'value': val, 
                'status': 'no',
                'cause_idx': wgc_columns.index(cause),  # Use index for proper alignment
                'cause_name': cause.replace('_', ' ').title()
            })
        
        for val in present_data:
            plot_data.append({
                'value': val, 
                'status': 'yes',
                'cause_idx': wgc_columns.index(cause),  # Use index for proper alignment
                'cause_name': cause.replace('_', ' ').title()
            })
    
    # Apply FDR correction if available
    fdr_corrected_p_values = None
    if apply_fdr_correction is not None:
        try:
            valid_p_values = [p for p in p_values if not pd.isna(p)]
            if len(valid_p_values) > 0:
                corrected_valid = apply_fdr_correction(valid_p_values)
                # Reconstruct full array
                fdr_corrected_p_values = []
                corrected_idx = 0
                for p in p_values:
                    if pd.isna(p):
                        fdr_corrected_p_values.append(np.nan)
                    else:
                        fdr_corrected_p_values.append(corrected_valid[corrected_idx])
                        corrected_idx += 1
                print(f"Applied FDR correction to {len(valid_p_values)} tests")
        except Exception as e:
            print(f"FDR correction failed: {e}")
            fdr_corrected_p_values = None
    
    # Get significance markers
    significance_markers = get_significance_markers(p_values, fdr_corrected_p_values)
    
    if not plot_data:
        print(f"No data available for {variable}")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot with enhanced styling
    fig_width = max(18, len(wgc_columns) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create split violin plot using cause_idx for proper alignment
    sns.violinplot(data=plot_df, x='cause_idx', y='value', hue='status', 
                   split=True, inner='quart',
                   palette={'no': '#5B9BD5', 'yes': '#E07B39'},
                   ax=ax, linewidth=1.5)
    
    # Customize plot appearance
    ax.set_xlabel('Weight Gain Causes', fontsize=14, fontweight='bold')
    ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(f'Distribution of {variable.replace("_", " ").title()} by Weight Gain Causes', 
                 fontsize=16, fontweight='bold', pad=50)
    
    # For publication clarity: set meaningful y-axis bounds for variables with natural limits
    if variable == 'total_followup_days':
        # Follow-up days cannot be negative - set lower bound to 0 for publication clarity
        ax.set_ylim(bottom=0)
    elif variable == 'baseline_weight_kg':
        # Weight cannot be negative - set lower bound to 0 for publication clarity  
        ax.set_ylim(bottom=0)
    
    # Create proper x-axis labels with sample sizes and line breaks
    x_labels = []
    x_positions = list(range(len(wgc_columns)))
    
    for i, result in enumerate(test_results):
        cause_name = result['cause'].replace('_', ' ').title()
        
        # Implement smart line breaks for better readability
        words = cause_name.split()
        if len(words) > 2:
            # Break into 2 lines for better readability
            mid = len(words) // 2
            cause_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        elif len(cause_name) > 12:
            # For long single/double words, break at logical points
            if ' And ' in cause_name:
                cause_name = cause_name.replace(' And ', '\nAnd ')
            elif ' Or ' in cause_name:
                cause_name = cause_name.replace(' Or ', '\nOr ')
        
        # Add sample sizes to the label
        label_with_n = f"{cause_name}\n(n={result['n_absent']}|{result['n_present']})"
        x_labels.append(label_with_n)
    
    # Set x-axis with proper alignment and diagonal rotation
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10, ha='right', rotation=45)
    
    # Add sample sizes and median values as text annotations above violins
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    text_height = y_max + (y_max - y_min) * 0.08
    
    for i, result in enumerate(test_results):
        # Add sample sizes and medians above each violin
        if result['n_absent'] > 0 and result['n_present'] > 0:
            annotation_text = (f"n={result['n_absent']} | n={result['n_present']}\n"
                             f"Med: {result['median_absent']:.1f} | {result['median_present']:.1f}")
            ax.text(i, text_height, annotation_text,
                   ha='center', va='bottom', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                           edgecolor='gray', alpha=0.9))
    
    # Add significance markers and p-values at the bottom
    y_bottom = y_min - (y_max - y_min) * 0.25
    
    for i, (marker, result) in enumerate(zip(significance_markers, test_results)):
        # Add significance marker
        if marker:
            ax.text(i, y_bottom + (y_max - y_min) * 0.08, marker, 
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='black')
        
        # Add p-values
        if not pd.isna(result['p_value']):
            if fdr_corrected_p_values is not None and i < len(fdr_corrected_p_values):
                corrected_p = fdr_corrected_p_values[i]
                if not pd.isna(corrected_p):
                    p_text = f'p_raw = {result["p_value"]:.3f}\np_fdr = {corrected_p:.3f}'
                else:
                    p_text = f'p_raw = {result["p_value"]:.3f}\np_fdr = --'
            else:
                p_text = f'p_raw = {result["p_value"]:.3f}'
            
            ax.text(i, y_bottom, p_text, 
                   ha='center', va='top', fontsize=8, 
                   style='italic', color='gray')
    
    # Move legend outside plot area and improve styling
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, title='', loc='upper left', 
                      bbox_to_anchor=(1.02, 1), frameon=True, 
                      fancybox=True, shadow=True, fontsize=12)
    
    # Add significance legend
    if fdr_corrected_p_values is not None:
        sig_legend_text = ['* p < 0.05 (raw)', '** p < 0.05 (FDR-corrected)']
    else:
        sig_legend_text = ['* p < 0.05']
    
    # Create custom legend for significance
    from matplotlib.lines import Line2D
    sig_legend_elements = []
    for text in sig_legend_text:
        sig_legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor='black', markersize=12, 
                                        label=text, linestyle='None'))
    
    legend2 = ax.legend(handles=sig_legend_elements, loc='upper left', 
                       bbox_to_anchor=(1.02, 0.8), frameon=True, 
                       fancybox=True, shadow=True, title='Significance', fontsize=10)
    
    # Add both legends to the plot
    ax.add_artist(legend)
    
    # Extend y-axis to accommodate p-values and annotations
    ax.set_ylim(y_bottom - (y_max - y_min) * 0.05, text_height + (y_max - y_min) * 0.08)
    
    # Adjust layout to prevent label cutoff and accommodate legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85, right=0.8)
    
    return fig


# Keep old function for backward compatibility
def create_split_violin_plot(df, variable, wgc_columns, config):
    """Wrapper for backward compatibility"""
    return create_enhanced_split_violin_plot(df, variable, wgc_columns, config)


def main():
    parser = argparse.ArgumentParser(description='Generate violin plots for WGC comparisons')
    parser.add_argument('--db-path', default="dbs/pnk_db2_p2_in.sqlite", 
                       help='Path to SQLite database')
    parser.add_argument('--table', default="timetoevent_wgc_compl", 
                       help='Table name in database')
    parser.add_argument('--variables', nargs='+', 
                       default=["baseline_weight_kg", "total_wl_%", "total_followup_days"],
                       help='Variables to plot')
    parser.add_argument('--output-dir', default="../outputs/violin_plots",
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create config
    config = ViolinPlotConfig(
        db_path=args.db_path,
        table_name=args.table,
        continuous_vars=args.variables,
        output_dir=args.output_dir
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {config.db_path}, table: {config.table_name}")
    
    try:
        with sqlite3.connect(config.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {config.table_name}", conn)
        
        print(f"Loaded {len(df)} records")
        
        # Identify WGC columns
        wgc_columns = get_wgc_columns(df)
        print(f"Found {len(wgc_columns)} WGC columns: {wgc_columns}")
        
        if not wgc_columns:
            print("No WGC columns found! Please check your data.")
            return
        
        # Generate split violin plots for each variable
        for variable in config.continuous_vars:
            if variable not in df.columns:
                print(f"Warning: Variable '{variable}' not found in data. Skipping.")
                continue
            
            print(f"Creating split violin plot for {variable}...")
            
            # Create enhanced split violin plot with statistical testing
            fig = create_enhanced_split_violin_plot(df, variable, wgc_columns, config)
            if fig:
                output_path = os.path.join(config.output_dir, f"{variable}_wgc_enhanced_split_violin.png")
                fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved enhanced split violin plot: {output_path}")
        
        print("All plots generated successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())