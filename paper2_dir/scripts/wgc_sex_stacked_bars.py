import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Optional
import argparse
from scipy.stats import chi2_contingency, fisher_exact
import sys

# Import FDR correction utilities
try:
    from fdr_correction_utils import apply_fdr_correction
except ImportError:
    print("Warning: FDR correction utilities not found. Statistical testing will use raw p-values only.")
    apply_fdr_correction = None


@dataclass
class StackedBarConfig:
    """Configuration for WGC categorical distribution stacked bar charts"""
    # Database and table settings
    db_path: str = "dbs/pnk_db2_p2_in.sqlite"
    table_name: str = "timetoevent_wgc_compl"
    
    # Output settings
    output_dir: str = "../outputs/stacked_bars"
    figure_size: tuple = (14, 8)
    dpi: int = 300


@dataclass
class CategoricalVariable:
    """Configuration for a categorical variable to analyze"""
    name: str
    column: str
    labels: List[str]  # [label_for_0, label_for_1]
    colors: List[str]  # [color_for_0, color_for_1]
    title: str
    filename_suffix: str


def perform_statistical_test_generic(df, wgc_column, categorical_column):
    """
    Perform statistical test comparing categorical variable distribution between 
    whole population and patients with specific WGC present.
    Uses Fisher's exact test for small samples, chi-squared for larger samples.
    """
    # Get whole population categorical distribution
    pop_cat0 = len(df[df[categorical_column] == 0])
    pop_cat1 = len(df[df[categorical_column] == 1])
    
    # Get WGC present group categorical distribution
    wgc_present = df[df[wgc_column] == 1]
    wgc_cat0 = len(wgc_present[wgc_present[categorical_column] == 0])
    wgc_cat1 = len(wgc_present[wgc_present[categorical_column] == 1])
    
    # Create contingency table
    # Rows: Population vs WGC group
    # Columns: Category 0 vs Category 1
    contingency_table = np.array([
        [pop_cat0, pop_cat1],      # Whole population
        [wgc_cat0, wgc_cat1]       # WGC present group
    ])
    
    # Check if any cell is 0 (would cause issues)
    if (contingency_table == 0).any():
        return np.nan, "insufficient_data"
    
    # Decide between Fisher's exact test and chi-squared test
    # Use Fisher's exact if any expected frequency < 5
    try:
        chi2, _, dof, expected = chi2_contingency(contingency_table)
        
        if (expected < 5).any():
            # Use Fisher's exact test for small samples
            # For 2x2 table, we can use fisher_exact directly
            odds_ratio, p_value = fisher_exact(contingency_table)
            return p_value, "fisher_exact"
        else:
            # Use chi-squared test for larger samples
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            return p_value, "chi_squared"
            
    except ValueError as e:
        return np.nan, "error"


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
            if fdr_corrected_p_values is not None and not pd.isna(fdr_corrected_p_values[i]):
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


def prepare_categorical_variables(df):
    """
    Prepare categorical variables for analysis, including derived variables
    """
    # Calculate population median age for age grouping
    median_age = pd.to_numeric(df["age"], errors="coerce").median()
    
    # Create derived categorical variables
    df = df.copy()
    
    # Age group: above/below median
    df['age_above_median'] = (pd.to_numeric(df["age"], errors="coerce") >= median_age).astype(int)
    
    # BMI group: above/below 30
    df['bmi_above_30'] = (pd.to_numeric(df["baseline_bmi"], errors="coerce") >= 30).astype(int)
    
    # Define categorical variables to analyze
    categorical_vars = [
        CategoricalVariable(
            name="Sex",
            column="sex_f",
            labels=["Male", "Female"],
            colors=["#FF9999", "#B8860B"],
            title="Sex Distribution by Weight Gain Causes",
            filename_suffix="sex"
        ),
        CategoricalVariable(
            name="Age Group",
            column="age_above_median",
            labels=[f"Below Median ({median_age:.1f}y)", f"Above Median ({median_age:.1f}y)"],
            colors=["#87CEEB", "#4682B4"],  # Light blue, Steel blue
            title=f"Age Distribution by Weight Gain Causes (Median = {median_age:.1f} years)",
            filename_suffix="age"
        ),
        CategoricalVariable(
            name="BMI Group",
            column="bmi_above_30",
            labels=["BMI < 30", "BMI ≥ 30"],
            colors=["#98FB98", "#228B22"],  # Light green, Forest green
            title="BMI Distribution by Weight Gain Causes",
            filename_suffix="bmi"
        ),
        CategoricalVariable(
            name="60-Day Dropout",
            column="60d_dropout",
            labels=["No Dropout", "Dropout"],
            colors=["#DDA0DD", "#8B008B"],  # Plum, Dark magenta
            title="60-Day Dropout Distribution by Weight Gain Causes",
            filename_suffix="dropout"
        ),
        CategoricalVariable(
            name="10% Weight Loss",
            column="10%_wl_achieved",
            labels=["Not Achieved", "Achieved"],
            colors=["#F0E68C", "#DAA520"],  # Khaki, Goldenrod
            title="10% Weight Loss Achievement by Weight Gain Causes",
            filename_suffix="weight_loss"
        )
    ]
    
    return df, categorical_vars


def get_wgc_columns(df):
    """
    Identify weight gain cause columns from the dataframe.
    Based on the pattern from descriptive_comparisons.py
    """
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
        "none_of_above"
    ]
    
    # Filter to only columns that actually exist in the dataframe
    wgc_cols = [col for col in potential_wgc_cols if col in df.columns]
    return wgc_cols


def create_sex_wgc_stacked_bar(df, wgc_columns, config):
    """
    Create stacked bar chart showing sex distribution by WGC presence/absence
    """
    # Prepare data for plotting
    plot_data = []
    
    for cause in wgc_columns:
        # Get data for this cause
        cause_absent = df[df[cause] == 0]
        cause_present = df[df[cause] == 1]
        
        # Count males and females for absent group
        males_absent = len(cause_absent[cause_absent['sex_f'] == 0])
        females_absent = len(cause_absent[cause_absent['sex_f'] == 1])
        
        # Count males and females for present group
        males_present = len(cause_present[cause_present['sex_f'] == 0])
        females_present = len(cause_present[cause_present['sex_f'] == 1])
        
        # Add to plot data
        plot_data.append({
            'cause': cause.replace('_', ' ').title(),
            'group': 'Absent',
            'males': males_absent,
            'females': females_absent,
            'total': males_absent + females_absent
        })
        
        plot_data.append({
            'cause': cause.replace('_', ' ').title(),
            'group': 'Present', 
            'males': males_present,
            'females': females_present,
            'total': males_present + females_present
        })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=config.figure_size)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set up the bar positions
    causes = [cause.replace('_', ' ').title() for cause in wgc_columns]
    x_pos = np.arange(len(causes))
    bar_width = 0.35
    
    # Create arrays for absent and present groups
    absent_data = plot_df[plot_df['group'] == 'Absent']
    present_data = plot_df[plot_df['group'] == 'Present']
    
    # Create stacked bars for absent group (left bars)
    males_absent = absent_data['males'].values
    females_absent = absent_data['females'].values
    
    bars1_male = ax.bar(x_pos - bar_width/2, males_absent, bar_width, 
                       label='Male', color='#FF9999', alpha=0.8)
    bars1_female = ax.bar(x_pos - bar_width/2, females_absent, bar_width, 
                         bottom=males_absent, label='Female', color='#B8860B', alpha=0.8)
    
    # Create stacked bars for present group (right bars)
    males_present = present_data['males'].values
    females_present = present_data['females'].values
    
    bars2_male = ax.bar(x_pos + bar_width/2, males_present, bar_width, 
                       color='#FF9999', alpha=0.8)
    bars2_female = ax.bar(x_pos + bar_width/2, females_present, bar_width, 
                         bottom=males_present, color='#B8860B', alpha=0.8)
    
    # Add sample size labels on bars
    for i, cause in enumerate(causes):
        # Absent group labels
        if males_absent[i] > 0:
            ax.text(x_pos[i] - bar_width/2, males_absent[i]/2, str(males_absent[i]), 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        if females_absent[i] > 0:
            ax.text(x_pos[i] - bar_width/2, males_absent[i] + females_absent[i]/2, 
                   str(females_absent[i]), ha='center', va='center', fontweight='bold', 
                   fontsize=9, color='white')
        
        # Present group labels
        if males_present[i] > 0:
            ax.text(x_pos[i] + bar_width/2, males_present[i]/2, str(males_present[i]), 
                   ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        if females_present[i] > 0:
            ax.text(x_pos[i] + bar_width/2, males_present[i] + females_present[i]/2, 
                   str(females_present[i]), ha='center', va='center', fontweight='bold', 
                   fontsize=9, color='white')
    
    # Customize the plot
    ax.set_xlabel('Weight Gain Causes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Sex Distribution by Weight Gain Causes (Absent vs Present)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(causes, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add group labels below x-axis
    for i in range(len(causes)):
        ax.text(x_pos[i] - bar_width/2, -ax.get_ylim()[1] * 0.05, 'Absent', 
               ha='center', va='top', fontsize=8, fontweight='bold')
        ax.text(x_pos[i] + bar_width/2, -ax.get_ylim()[1] * 0.05, 'Present', 
               ha='center', va='top', fontsize=8, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    return fig


def create_enhanced_proportional_stacked_bar_generic(df, wgc_columns, categorical_var, config):
    """
    Create enhanced proportional stacked bar chart with statistical testing,
    reference line, and significance markers for any categorical variable
    """
    # Prepare data for plotting - start with whole population
    plot_data = []
    p_values = []
    
    # First bar: Whole population (no statistical test needed)
    cat_col = categorical_var.column
    total_cat0 = len(df[df[cat_col] == 0])
    total_cat1 = len(df[df[cat_col] == 1])
    total_population = total_cat0 + total_cat1
    
    cat0_pct_total = (total_cat0 / total_population * 100) if total_population > 0 else 0
    cat1_pct_total = (total_cat1 / total_population * 100) if total_population > 0 else 0
    
    plot_data.append({
        'cause': 'Whole Population',
        'cat0_pct': cat0_pct_total,
        'cat1_pct': cat1_pct_total,
        'cat0_n': total_cat0,
        'cat1_n': total_cat1,
        'total': total_population
    })
    p_values.append(np.nan)  # No test for whole population
    
    # Then add data for each WGC where cause is present + perform statistical tests
    print("Performing statistical tests...")
    test_methods = []
    
    for cause in wgc_columns:
        # Get data only for patients where this cause is present
        cause_present = df[df[cause] == 1]
        
        # Count categories for present group
        cat0_present = len(cause_present[cause_present[cat_col] == 0])
        cat1_present = len(cause_present[cause_present[cat_col] == 1])
        total_present = cat0_present + cat1_present
        
        # Calculate proportions
        cat0_pct = (cat0_present / total_present * 100) if total_present > 0 else 0
        cat1_pct = (cat1_present / total_present * 100) if total_present > 0 else 0
        
        # Perform statistical test (chi-squared or Fisher's exact)
        p_val, test_method = perform_statistical_test_generic(df, cause, cat_col)
        p_values.append(p_val)
        test_methods.append(test_method)
        
        # Add to plot data
        plot_data.append({
            'cause': cause.replace('_', ' ').title(),
            'cat0_pct': cat0_pct,
            'cat1_pct': cat1_pct,
            'cat0_n': cat0_present,
            'cat1_n': cat1_present,
            'total': total_present,
            'p_value': p_val,
            'test_method': test_method
        })
        
        if not pd.isna(p_val):
            print(f"  {cause}: p-value = {p_val:.4f} ({test_method})")
        else:
            print(f"  {cause}: {test_method}")
    
    # Apply FDR correction if available
    fdr_corrected_p_values = None
    if apply_fdr_correction is not None:
        try:
            # Only correct non-NaN p-values (exclude whole population)
            valid_p_values = [p for p in p_values[1:] if not pd.isna(p)]
            if len(valid_p_values) > 0:
                corrected_valid = apply_fdr_correction(valid_p_values)
                # Reconstruct full array with NaN for whole population and invalid tests
                fdr_corrected_p_values = [np.nan]  # Whole population
                corrected_idx = 0
                for p in p_values[1:]:
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
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot with extra space for legends
    fig, ax = plt.subplots(figsize=(config.figure_size[0], config.figure_size[1] + 1))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set up the bar positions
    causes = plot_df['cause'].values
    x_pos = np.arange(len(causes))
    
    # Get data arrays
    cat0_pct = plot_df['cat0_pct'].values
    cat1_pct = plot_df['cat1_pct'].values
    cat0_n = plot_df['cat0_n'].values
    cat1_n = plot_df['cat1_n'].values
    totals = plot_df['total'].values
    
    # Create stacked bars using categorical variable colors
    # Note: Category 1 is on top, Category 0 is on bottom
    bars_cat0 = ax.bar(x_pos, cat0_pct, label=categorical_var.labels[0], 
                      color=categorical_var.colors[0], alpha=0.9)
    bars_cat1 = ax.bar(x_pos, cat1_pct, bottom=cat0_pct, label=categorical_var.labels[1], 
                      color=categorical_var.colors[1], alpha=0.9)
    
    # Add dashed reference line at whole population category 0 percentage
    reference_line_y = cat0_pct_total
    ax.axhline(y=reference_line_y, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # Add significance markers on the reference line (plain asterisks, no circles)
    for i, marker in enumerate(significance_markers):
        if marker and i > 0:  # Skip whole population (index 0)
            ax.text(x_pos[i], reference_line_y, marker, 
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color='black')
    
    # Add sample size and percentage labels in colored boxes
    for i in range(len(causes)):
        # Category 1 labels (top of bar) - white box with category 1 color border
        if cat1_n[i] > 0:
            cat1_text = f'{cat1_n[i]}\n({cat1_pct[i]:.1f}%)'
            ax.text(x_pos[i], 108, cat1_text, 
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=categorical_var.colors[1], linewidth=2))
        
        # Category 0 labels (bottom of bar) - white box with category 0 color border
        if cat0_n[i] > 0:
            cat0_text = f'{cat0_n[i]}\n({cat0_pct[i]:.1f}%)'
            ax.text(x_pos[i], -8, cat0_text, 
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=categorical_var.colors[0], linewidth=2))
        
        # Add raw and corrected p-values below x-axis labels (skip whole population)
        if i > 0 and i < len(plot_df):
            p_val = plot_df.iloc[i]['p_value']
            if not pd.isna(p_val):
                # Get corrected p-value if available and format on separate lines
                if fdr_corrected_p_values is not None and i < len(fdr_corrected_p_values):
                    corrected_p = fdr_corrected_p_values[i]
                    if not pd.isna(corrected_p):
                        p_text = f'p_raw = {p_val:.3f}\np_fdr = {corrected_p:.3f}'
                    else:
                        p_text = f'p_raw = {p_val:.3f}\np_fdr = --'
                else:
                    p_text = f'p_raw = {p_val:.3f}'
                
                ax.text(x_pos[i], -20, p_text, 
                       ha='center', va='top', fontsize=7, 
                       style='italic', color='gray')
    
    # Customize the plot
    ax.set_xlabel('Weight Gain Causes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(categorical_var.title, 
                 fontsize=14, fontweight='bold', pad=60)
    ax.set_ylim(-25, 120)  # Extended range to accommodate labels and p-values
    
    # Set y-axis to only show positive values (0, 20, 40, 60, 80, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    
    # Create x-axis labels with sample sizes
    x_labels = []
    for i, (cause, total) in enumerate(zip(causes, totals)):
        x_labels.append(f'{cause}\n(n={total})')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    
    # Move legend outside plot area (upper right, outside the grid)
    # Reverse the order so Category 1 appears first (since it's on top of bars)
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend([handles[1], handles[0]], [labels[1], labels[0]], 
                       loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, 
                       fancybox=True, shadow=True, title=categorical_var.name)
    
    # Add significance legend below the sex legend
    if fdr_corrected_p_values is not None:
        sig_legend_text = ['* p < 0.05 (raw)', '** p < 0.05 (FDR-corrected)']
    else:
        sig_legend_text = ['* p < 0.05']
    
    # Create custom legend for significance
    from matplotlib.lines import Line2D
    sig_legend_elements = []
    for text in sig_legend_text:
        sig_legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                        markerfacecolor='black', markersize=10, 
                                        label=text, linestyle='None'))
    
    legend2 = ax.legend(handles=sig_legend_elements, loc='upper left', 
                       bbox_to_anchor=(1.02, 0.8), frameon=True, 
                       fancybox=True, shadow=True, title='Significance')
    
    # Add both legends to the plot
    ax.add_artist(legend1)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout to accommodate legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85, right=0.8)
    
    return fig


# Keep the old function for backward compatibility
def create_enhanced_proportional_stacked_bar(df, wgc_columns, config):
    """Wrapper for backward compatibility - creates sex distribution chart"""
    # Create sex categorical variable for backward compatibility
    sex_var = CategoricalVariable(
        name="Sex",
        column="sex_f",
        labels=["Male", "Female"],
        colors=["#FF9999", "#B8860B"],
        title="Sex Distribution by Weight Gain Causes",
        filename_suffix="sex"
    )
    return create_enhanced_proportional_stacked_bar_generic(df, wgc_columns, sex_var, config)

def create_simple_proportional_stacked_bar(df, wgc_columns, config):
    """Wrapper for backward compatibility"""
    return create_enhanced_proportional_stacked_bar(df, wgc_columns, config)

def generate_all_categorical_plots(df, wgc_columns, config):
    """
    Generate stacked bar charts for all categorical variables
    """
    # Prepare data and get categorical variables
    df_prepared, categorical_vars = prepare_categorical_variables(df)
    
    generated_files = []
    
    for cat_var in categorical_vars:
        print(f"\nGenerating plot for {cat_var.name}...")
        
        # Check if the required column exists
        if cat_var.column not in df_prepared.columns:
            print(f"Warning: Column '{cat_var.column}' not found. Skipping {cat_var.name}.")
            continue
        
        # Check if we have valid data
        valid_data = df_prepared[cat_var.column].dropna()
        if len(valid_data) == 0:
            print(f"Warning: No valid data for '{cat_var.column}'. Skipping {cat_var.name}.")
            continue
        
        # Generate the plot
        try:
            fig = create_enhanced_proportional_stacked_bar_generic(
                df_prepared, wgc_columns, cat_var, config
            )
            
            # Save the plot
            output_path = os.path.join(
                config.output_dir, 
                f"wgc_{cat_var.filename_suffix}_enhanced_proportional_bars.png"
            )
            fig.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
            plt.close(fig)
            
            generated_files.append(output_path)
            print(f"  Saved: {output_path}")
            
        except Exception as e:
            print(f"  Error generating plot for {cat_var.name}: {str(e)}")
            continue
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description='Generate stacked bar charts for WGC sex distributions')
    parser.add_argument('--db-path', default="dbs/pnk_db2_p2_in.sqlite", 
                       help='Path to SQLite database')
    parser.add_argument('--table', default="timetoevent_wgc_compl", 
                       help='Table name in database')
    parser.add_argument('--output-dir', default="../outputs/stacked_bars",
                       help='Output directory for plots')

    
    args = parser.parse_args()
    
    # Create config
    config = StackedBarConfig(
        db_path=args.db_path,
        table_name=args.table,
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
        
        # Check for sex column
        if 'sex_f' not in df.columns:
            print("Error: 'sex_f' column not found in data!")
            return 1
        
        # Identify WGC columns
        wgc_columns = get_wgc_columns(df)
        print(f"Found {len(wgc_columns)} WGC columns: {wgc_columns}")
        
        if not wgc_columns:
            print("No WGC columns found! Please check your data.")
            return 1
        
        # Generate all categorical distribution charts
        print("Creating enhanced proportional stacked bar charts for all categorical variables...")
        generated_files = generate_all_categorical_plots(df, wgc_columns, config)
        
        if generated_files:
            print(f"\nSuccessfully generated {len(generated_files)} stacked bar charts:")
            for file_path in generated_files:
                print(f"  - {os.path.basename(file_path)}")
        else:
            print("No charts were generated.")
        
        print("\nAll stacked bar charts generated successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())