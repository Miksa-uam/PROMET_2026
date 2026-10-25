"""
WGC vs Mother Cohort Lollipop Plot
==================================

Creates a publication-ready lollipop plot showing percentage differences between
WGC cohort and mother cohort (all patients) across key clinical variables.

The plot shows:
- Percentage change: ((WGC_value - Mother_value) / Mother_value) * 100
- Statistical significance indicators (* for raw p<0.05, ** for FDR-corrected p<0.05)
- Clean, unified axis for publication quality

Author: Generated for descriptive analysis pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def extract_comparison_data(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Extract comparison data from the descriptive comparisons database table.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database containing comparison results
    table_name : str
        Name of the table containing demographic stratification results
        
    Returns:
    --------
    pd.DataFrame
        Cleaned comparison data with variables and statistics
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    # Filter out delimiter rows, N row, and section headers
    df = df[~df['Variable'].str.startswith('delim_', na=False)]
    df = df[df['Variable'] != 'N']
    df = df[df['Variable'].notna()]
    
    # Filter out section headers (these don't have actual data values)
    section_headers = [
        'Demographics and baseline anthropometry',
        'Treatment outcomes', 
        'Self-reported causes of weight gain'
    ]
    df = df[~df['Variable'].isin(section_headers)]
    
    return df

def parse_mean_sd_values(value_str: str) -> Tuple[float, float]:
    """
    Parse mean ± SD format strings to extract mean and standard deviation.
    
    Parameters:
    -----------
    value_str : str
        String in format "mean ± sd" or "mean ± sd | median [q1–q3]"
        
    Returns:
    --------
    Tuple[float, float]
        (mean, sd) values, or (np.nan, np.nan) if parsing fails
    """
    if pd.isna(value_str) or value_str == "N/A":
        return np.nan, np.nan
    
    try:
        # Handle format with median info: "mean ± sd | median [q1–q3]"
        if " | " in value_str:
            mean_sd_part = value_str.split(" | ")[0]
        else:
            mean_sd_part = value_str
        
        # Parse "mean ± sd"
        if " ± " in mean_sd_part:
            parts = mean_sd_part.split(" ± ")
            mean = float(parts[0])
            sd = float(parts[1])
            return mean, sd
        else:
            # Single value, assume no SD
            return float(mean_sd_part), np.nan
            
    except (ValueError, IndexError):
        return np.nan, np.nan

def parse_n_perc_values(value_str: str) -> Tuple[int, float]:
    """
    Parse N (%) format strings to extract count and percentage.
    
    Parameters:
    -----------
    value_str : str
        String in format "N (percentage%)"
        
    Returns:
    --------
    Tuple[int, float]
        (count, percentage) values, or (np.nan, np.nan) if parsing fails
    """
    if pd.isna(value_str) or value_str == "N/A":
        return np.nan, np.nan
    
    try:
        # Parse "N (percentage%)"
        if " (" in value_str and ")" in value_str:
            parts = value_str.split(" (")
            count = int(parts[0])
            perc_part = parts[1].replace(")", "").replace("%", "")
            percentage = float(perc_part)
            return count, percentage
        else:
            return np.nan, np.nan
            
    except (ValueError, IndexError):
        return np.nan, np.nan

def calculate_percentage_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate percentage changes between WGC cohort and mother cohort.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Comparison data from descriptive analysis
        
    Returns:
    --------
    pd.DataFrame
        Data with calculated percentage changes and significance indicators
    """
    results = []
    
    for _, row in df.iterrows():
        variable = row['Variable']
        mother_val = row['Parent cohort']
        wgc_val = row['Observed cohort']
        # Extract p-values and convert to float
        p_value_raw = row.get('Cohort comparison: p-value', np.nan)
        p_value_fdr = row.get('Cohort comparison: p-value (FDR-corrected)', np.nan)
        
        # Convert p-values to float if they're strings
        try:
            if isinstance(p_value_raw, str):
                p_value_raw = float(p_value_raw)
        except (ValueError, TypeError):
            p_value_raw = np.nan
            
        try:
            if isinstance(p_value_fdr, str):
                p_value_fdr = float(p_value_fdr)
        except (ValueError, TypeError):
            p_value_fdr = np.nan
        
        # Determine variable type and extract values
        if isinstance(mother_val, str) and " ± " in mother_val:
            # Continuous variable (mean ± SD format)
            mother_mean, _ = parse_mean_sd_values(mother_val)
            wgc_mean, _ = parse_mean_sd_values(wgc_val)
            
            if not (np.isnan(mother_mean) or np.isnan(wgc_mean)) and mother_mean != 0:
                pct_change = ((wgc_mean - mother_mean) / mother_mean) * 100
                value_type = 'continuous'
                mother_value = mother_mean
                wgc_value = wgc_mean
            else:
                continue
                
        elif isinstance(mother_val, str) and " (" in mother_val:
            # Categorical variable (N (%) format)
            _, mother_perc = parse_n_perc_values(mother_val)
            _, wgc_perc = parse_n_perc_values(wgc_val)
            
            if not (np.isnan(mother_perc) or np.isnan(wgc_perc)) and mother_perc != 0:
                pct_change = ((wgc_perc - mother_perc) / mother_perc) * 100
                value_type = 'categorical'
                mother_value = mother_perc
                wgc_value = wgc_perc
            else:
                continue
        else:
            continue
        
        # Determine significance
        significance = ""
        if pd.notna(p_value_fdr) and isinstance(p_value_fdr, (int, float)) and p_value_fdr < 0.05:
            significance = "**"  # FDR-corrected significant
        elif pd.notna(p_value_raw) and isinstance(p_value_raw, (int, float)) and p_value_raw < 0.05:
            significance = "*"   # Raw p-value significant
        
        results.append({
            'variable': variable,
            'variable_type': value_type,
            'mother_value': mother_value,
            'wgc_value': wgc_value,
            'percent_change': pct_change,
            'p_value_raw': p_value_raw,
            'p_value_fdr': p_value_fdr,
            'significance': significance
        })
    
    return pd.DataFrame(results)

def create_lollipop_plot(data: pd.DataFrame, output_path: str = None, 
                        title: str = "Key clinical differences between patients with and without reported weight gain causes",
                        figsize: Tuple[int, int] = (10, 8),
                        preserve_order: bool = False, 
                        wgc_n: int = None, mother_n: int = None) -> None:
    """
    Create a publication-ready lollipop plot showing percentage differences.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with percentage changes and significance indicators
    output_path : str, optional
        Path to save the plot (if None, displays only)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size (width, height)
    preserve_order : bool
        If True, preserves the order of variables as provided in data.
        If False, sorts by absolute percentage change for better visualization.
    wgc_n : int, optional
        Sample size for WGC cohort (patients with weight gain causes)
    mother_n : int, optional
        Sample size for mother cohort (all patients)
    """
    # Sort variables based on preference
    if preserve_order:
        data_sorted = data.iloc[::-1]  # Reverse order so first item appears at top
        print("Preserving variable order as specified (reversed for top-to-bottom display)")
    else:
        data_sorted = data.reindex(data['percent_change'].abs().sort_values(ascending=True).index)
        print("Sorting variables by absolute percentage change")
    
    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the lollipop plot
    y_positions = range(len(data_sorted))
    
    # Plot the stems (lines from 0 to each point)
    for i, (_, row) in enumerate(data_sorted.iterrows()):
        ax.plot([0, row['percent_change']], [i, i], 'o-', 
                color='steelblue', linewidth=2, markersize=8, alpha=0.7)
    
    # Add significance indicators
    for i, (_, row) in enumerate(data_sorted.iterrows()):
        if row['significance']:
            # Position the asterisk slightly to the right of the dot
            x_offset = 0.5 if row['percent_change'] >= 0 else -0.5
            ax.text(row['percent_change'] + x_offset, i, row['significance'], 
                   fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(data_sorted['variable'], fontsize=10)
    ax.set_xlabel('Percent change: relative differences among WGC patients compared with all patients', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clinical characteristics, adherence and outcomes', fontsize=12, fontweight='bold')
    
    # Create dynamic title with sample sizes if provided
    if wgc_n is not None and mother_n is not None:
        full_title = f"Key clinical differences between patients with (n = {wgc_n}) and without (n = {mother_n - wgc_n}) reported weight gain causes"
    else:
        full_title = title
    
    ax.set_title(full_title, fontsize=14, fontweight='bold', pad=20)
    
    # Add reference line at 0%
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend for significance
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='steelblue', linewidth=2, 
                   markersize=8, alpha=0.7, label='Percent difference'),
        plt.Line2D([0], [0], marker='$*$', color='black', linewidth=0, 
                   markersize=12, label='* p < 0.05 (raw)'),
        plt.Line2D([0], [0], marker='$**$', color='black', linewidth=0, 
                   markersize=12, label='** p < 0.05 (FDR-corrected)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
              fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Lollipop plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def generate_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of the percentage changes and significance.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with percentage changes and significance indicators
        
    Returns:
    --------
    pd.DataFrame
        Summary table with formatted results
    """
    summary = data.copy()
    summary['percent_change_formatted'] = summary['percent_change'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    summary['p_value_raw_formatted'] = summary['p_value_raw'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
    )
    summary['p_value_fdr_formatted'] = summary['p_value_fdr'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A"
    )
    
    return summary[['variable', 'variable_type', 'percent_change_formatted', 
                   'p_value_raw_formatted', 'p_value_fdr_formatted', 'significance']]

def main(db_path: str, table_name: str, output_plot_path: str = None, 
         output_table_path: str = None, variables_to_include: List[str] = None,
         title: str = "Key clinical differences between patients with and without reported weight gain causes",
         figsize: Tuple[int, int] = (10, 8), preserve_order: bool = False):
    """
    Main function to generate WGC vs Mother cohort lollipop plot.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database with comparison results
    table_name : str
        Name of the demographic stratification table
    output_plot_path : str, optional
        Path to save the lollipop plot
    output_table_path : str, optional
        Path to save the summary table
    variables_to_include : List[str], optional
        List of specific variables to include in the plot (if None, includes all)
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size (width, height)
    preserve_order : bool
        If True, preserves the order of variables_to_include list.
        If False, sorts by absolute percentage change.
    """
    print("Generating WGC vs Mother Cohort Lollipop Plot...")
    
    # Extract and process data
    print("Extracting comparison data...")
    df = extract_comparison_data(db_path, table_name)
    
    # Extract sample sizes from the 'N' row if available
    wgc_n = None
    mother_n = None
    try:
        with sqlite3.connect(db_path) as conn:
            n_df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE Variable = 'N'", conn)
            if not n_df.empty:
                mother_n = int(n_df['Parent cohort'].iloc[0]) if pd.notna(n_df['Parent cohort'].iloc[0]) else None
                wgc_n = int(n_df['Observed cohort'].iloc[0]) if pd.notna(n_df['Observed cohort'].iloc[0]) else None
                print(f"Sample sizes - WGC cohort: {wgc_n}, Mother cohort: {mother_n}")
    except Exception as e:
        print(f"Could not extract sample sizes: {e}")
    
    print("Calculating percentage changes...")
    pct_data = calculate_percentage_changes(df)
    
    # Filter to specific variables if requested
    if variables_to_include is not None:
        if preserve_order:
            # Preserve the order specified in variables_to_include
            pct_data_filtered = []
            for var in variables_to_include:
                matching_rows = pct_data[pct_data['variable'] == var]
                if not matching_rows.empty:
                    pct_data_filtered.append(matching_rows.iloc[0])
            pct_data = pd.DataFrame(pct_data_filtered)
            print(f"Filtered to {len(pct_data)} requested variables (order preserved)")
        else:
            pct_data = pct_data[pct_data['variable'].isin(variables_to_include)]
            print(f"Filtered to {len(pct_data)} requested variables")
    
    if len(pct_data) == 0:
        print("Warning: No valid comparison data found for plotting.")
        return
    
    print(f"Found {len(pct_data)} variables for comparison:")
    for _, row in pct_data.iterrows():
        print(f"  {row['variable']}: {row['percent_change']:+.1f}% {row['significance']}")
    
    # Create the plot
    print("Creating lollipop plot...")
    create_lollipop_plot(pct_data, output_plot_path, title, figsize, preserve_order, wgc_n, mother_n)
    
    # Generate summary table
    print("Generating summary table...")
    summary_table = generate_summary_table(pct_data)
    
    if output_table_path:
        summary_table.to_csv(output_table_path, index=False)
        print(f"Summary table saved to: {output_table_path}")
    
    print("Lollipop plot generation complete!")
    return pct_data, summary_table

def create_lollipop_from_raw_data(input_db_path: str, mother_cohort_table: str, 
                                 wgc_cohort_table: str, output_plot_path: str = None,
                                 output_table_path: str = None, 
                                 variables_to_include: List[str] = None,
                                 title: str = "Key clinical differences between patients with and without reported weight gain causes",
                                 figsize: Tuple[int, int] = (10, 8),
                                 preserve_order: bool = False):
    """
    Create lollipop plot directly from raw cohort data (alternative to using comparison table).
    
    Parameters:
    -----------
    input_db_path : str
        Path to database containing the raw cohort data
    mother_cohort_table : str
        Name of table containing mother cohort (all patients)
    wgc_cohort_table : str
        Name of table containing WGC cohort
    output_plot_path : str, optional
        Path to save the lollipop plot
    output_table_path : str, optional
        Path to save the summary table
    variables_to_include : List[str], optional
        List of specific variables to include in the plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size (width, height)
    """
    print(f"Creating lollipop plot from raw data...")
    print(f"Mother cohort: {mother_cohort_table}")
    print(f"WGC cohort: {wgc_cohort_table}")
    
    # Load raw data
    with sqlite3.connect(input_db_path) as conn:
        df_mother = pd.read_sql_query(f"SELECT * FROM {mother_cohort_table}", conn)
        df_wgc = pd.read_sql_query(f"SELECT * FROM {wgc_cohort_table}", conn)
    
    print(f"Loaded {len(df_mother)} mother cohort and {len(df_wgc)} WGC cohort records")
    
    # Calculate direct comparisons
    pct_data = calculate_direct_percentage_changes(df_mother, df_wgc, variables_to_include)
    
    if len(pct_data) == 0:
        print("Warning: No valid comparison data found for plotting.")
        return
    
    print(f"Found {len(pct_data)} variables for comparison:")
    for _, row in pct_data.iterrows():
        print(f"  {row['variable']}: {row['percent_change']:+.1f}% {row['significance']}")
    
    # Create the plot with sample sizes
    wgc_n = len(df_wgc)
    mother_n = len(df_mother)
    print("Creating lollipop plot...")
    create_lollipop_plot(pct_data, output_plot_path, title, figsize, preserve_order, wgc_n, mother_n)
    
    # Generate summary table
    print("Generating summary table...")
    summary_table = generate_summary_table(pct_data)
    
    if output_table_path:
        summary_table.to_csv(output_table_path, index=False)
        print(f"Summary table saved to: {output_table_path}")
    
    print("Lollipop plot generation complete!")
    return pct_data, summary_table

def calculate_direct_percentage_changes(df_mother: pd.DataFrame, df_wgc: pd.DataFrame, 
                                      variables_to_include: List[str] = None) -> pd.DataFrame:
    """
    Calculate percentage changes directly from raw cohort dataframes.
    
    Parameters:
    -----------
    df_mother : pd.DataFrame
        Mother cohort (all patients) data
    df_wgc : pd.DataFrame
        WGC cohort data
    variables_to_include : List[str], optional
        Specific variables to analyze
        
    Returns:
    --------
    pd.DataFrame
        Data with calculated percentage changes
    """
    results = []
    
    # Get common columns
    common_cols = set(df_mother.columns) & set(df_wgc.columns)
    if variables_to_include:
        common_cols = common_cols & set(variables_to_include)
    
    # Remove ID columns
    exclude_cols = {'patient_id', 'medical_record_id', 'medical_record_start_date'}
    common_cols = common_cols - exclude_cols
    
    for col in common_cols:
        mother_data = df_mother[col].dropna()
        wgc_data = df_wgc[col].dropna()
        
        if len(mother_data) == 0 or len(wgc_data) == 0:
            continue
        
        # Determine if continuous or categorical
        if mother_data.dtype in ['int64', 'float64'] and len(mother_data.unique()) > 10:
            # Continuous variable
            mother_mean = mother_data.mean()
            wgc_mean = wgc_data.mean()
            
            if mother_mean != 0:
                pct_change = ((wgc_mean - mother_mean) / mother_mean) * 100
                
                # Statistical test
                try:
                    from scipy.stats import ttest_ind
                    _, p_value = ttest_ind(wgc_data, mother_data, equal_var=False)
                except:
                    p_value = np.nan
                
                results.append({
                    'variable': col,
                    'variable_type': 'continuous',
                    'mother_value': mother_mean,
                    'wgc_value': wgc_mean,
                    'percent_change': pct_change,
                    'p_value_raw': p_value,
                    'p_value_fdr': np.nan,  # Would need FDR correction across all tests
                    'significance': "*" if pd.notna(p_value) and p_value < 0.05 else ""
                })
        
        elif set(mother_data.unique()).issubset({0, 1}) or mother_data.dtype == 'bool':
            # Binary categorical variable
            mother_prop = mother_data.mean() * 100  # Convert to percentage
            wgc_prop = wgc_data.mean() * 100
            
            if mother_prop != 0:
                pct_change = ((wgc_prop - mother_prop) / mother_prop) * 100
                
                # Chi-square test
                try:
                    from scipy.stats import chi2_contingency
                    contingency = pd.crosstab(
                        pd.concat([pd.Series(['mother']*len(mother_data), index=mother_data.index),
                                  pd.Series(['wgc']*len(wgc_data), index=wgc_data.index)]),
                        pd.concat([mother_data, wgc_data])
                    )
                    _, p_value, _, _ = chi2_contingency(contingency)
                except:
                    p_value = np.nan
                
                results.append({
                    'variable': col,
                    'variable_type': 'categorical',
                    'mother_value': mother_prop,
                    'wgc_value': wgc_prop,
                    'percent_change': pct_change,
                    'p_value_raw': p_value,
                    'p_value_fdr': np.nan,
                    'significance': "*" if pd.notna(p_value) and p_value < 0.05 else ""
                })
    
    return pd.DataFrame(results)

def get_available_variables(db_path: str, table_name: str) -> List[str]:
    """
    Get list of available variables from the comparison table.
    
    Parameters:
    -----------
    db_path : str
        Path to the SQLite database
    table_name : str
        Name of the comparison table
        
    Returns:
    --------
    List[str]
        List of available variable names
    """
    df = extract_comparison_data(db_path, table_name)
    return df['Variable'].tolist()

def get_variable_categories() -> Dict[str, List[str]]:
    """
    Get predefined categories of variables for easy selection.
    
    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with category names as keys and variable lists as values
    """
    return {
        'demographics': [
            'Sex (% of females)',
            'Age (years)', 
            'Height (m)',
            'Baseline weight (kg)',
            'Baseline BMI (kg/m²)',
            'Baseline fat mass (%)',
            'Baseline muscle mass (%)'
        ],
        'engagement': [
            'Follow-up length (days)',
            'Number of visits',
            'Number of total measurements',
            'Average measurement frequency (days)',
            'Instant dropouts (n)',
            '40-day dropouts (n)',
            '60-day dropouts (n)',
            '80-day dropouts (n)'
        ],
        'weight_loss_outcomes': [
            'Total weight loss (kg)',
            'Total weight loss (%)',
            'BMI reduction (kg/m²)',
            'Total fat mass loss (%)',
            'Total muscle mass change (%)',
            'Achieved 5% weight loss (n)',
            'Days to 5% weight loss',
            'Achieved 10% weight loss (n)',
            'Days to 10% weight loss',
            'Achieved 15% weight loss (n)',
            'Days to 15% weight loss'
        ],
        'short_term_outcomes': [
            '40-day weight loss (%)',
            '40-day BMI reduction (kg/m²)',
            '40-day fat mass loss (%)',
            '40-day muscle mass change (%)',
            '60-day weight loss (%)',
            '60-day BMI reduction (kg/m²)',
            '60-day fat mass loss (%)',
            '60-day muscle mass change (%)'
        ],
        'weight_gain_causes': [
            'Women\'s health and pregnancy (yes/no)',
            'Mental health (yes/no)',
            'Family issues (yes/no)',
            'Medication, disease or injury (yes/no)',
            'Physical inactivity (yes/no)',
            'Eating habits (yes/no)',
            'Schedule (yes/no)',
            'Smoking cessation (yes/no)',
            'Treatment discontinuation or relapse (yes/no)',
            'COVID-19 pandemic (yes/no)',
            'Lifestyle circumstances (yes/no)',
            'None of the above (yes/no)'
        ]
    }

if __name__ == "__main__":
    # Use the correct database path and table name from descriptive_comparisons.py
    db_path = "../dbs/pnk_db2_p2_out.sqlite"  # Output database where comparison results are stored
    table_name = "wgc_cmpl_dmgrph_strt"       # WGC complete demographic stratification table
    output_plot = "../outputs/wgc_mother_cohort_lollipop.png"
    output_table = "../outputs/wgc_mother_cohort_summary.csv"
    
    main(db_path, table_name, output_plot, output_table)