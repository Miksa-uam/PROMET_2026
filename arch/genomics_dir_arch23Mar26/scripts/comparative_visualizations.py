"""
COMPARATIVE_VISUALIZATIONS.PY
Version: 1.0

Description:
A visualization module designed to create polished, publication-ready plots comparing two cohorts 
(e.g., General Population vs Genomics Subset). It bridges the visual style of cluster_descriptions.py 
with the simple 2-group logic of descriptive_comparisons.py.

Features:
- Split Violin Plots for continuous variables (Population vs Subset).
- Stacked Bar Charts for categorical variables.
- Auto-detection of variable types based on row_order configuration.
- Integrated statistical annotation (medians, counts, p-values).

Usage:
    from comparative_visualizations import run_comparative_visualizations
    run_comparative_visualizations(df_input, df_mother, config, output_dir)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency
from typing import List, Dict, Tuple, Optional, Any
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Visual Style Constants (Inherited from cluster_descriptions_copy.py)
POPULATION_COLOR = '#A0A0A0'   # Neutral Gray for Mother Cohort
SUBSET_COLOR = '#00CC96'       # Distinct Teal/Green for Genomics Subset
ACHIEVED_COLOR = '#4361EE'     # Blue for "Yes/Achieved" in stacked bars
NOT_ACHIEVED_COLOR = '#EC5B57' # Red for "No/Not Achieved" in stacked bars

# Plot Layout Constants
FIG_SIZE_VIOLIN = (10, 6)
FIG_SIZE_BAR = (8, 6)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_ANNOT = 12

# =============================================================================
# HELPER FUNCTIONS - Statistics & Format
# =============================================================================

def calculate_p_value_continuous(s1, s2):
    """Mann-Whitney U test for continuous variables."""
    s1 = pd.to_numeric(s1, errors='coerce').dropna()
    s2 = pd.to_numeric(s2, errors='coerce').dropna()
    if len(s1) < 1 or len(s2) < 1:
        return np.nan
    try:
        _, p = mannwhitneyu(s1, s2, alternative='two-sided')
        return p
    except ValueError:
        return np.nan

def calculate_p_value_categorical(s1, s2):
    """Chi-squared test for categorical variables."""
    s1 = s1.dropna()
    s2 = s2.dropna()
    if s1.empty or s2.empty:
        return np.nan
    # Create contingency table
    contingency = pd.crosstab(
        index=np.concatenate([np.zeros(len(s1)), np.ones(len(s2))]),
        columns=np.concatenate([s1, s2])
    )
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan
    try:
        _, p, _, _ = chi2_contingency(contingency)
        return p
    except ValueError:
        return np.nan

def format_p_value(p):
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p:.3f}"

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_split_violin(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    variable: str, 
    pretty_name: str, 
    output_path: str
):
    """
    Creates a detailed split violin plot:
    - Left side: Mother Cohort (Population)
    - Right side: Input Cohort (Subset)
    Includes annotation boxes for medians and p-value.
    """
    # Clean Data
    s_sub = pd.to_numeric(df_subset[variable], errors='coerce').dropna()
    s_pop = pd.to_numeric(df_mother[variable], errors='coerce').dropna()
    
    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping {variable}: No valid data.")
        return

    # Prepare Combined DataFrame for Seaborn
    # 'Dummy' y-axis variable needed for simple violin split
    df_plot_sub = pd.DataFrame({'Value': s_sub, 'Group': 'Subset', 'Split': 'Violin'})
    df_plot_pop = pd.DataFrame({'Value': s_pop, 'Group': 'Population', 'Split': 'Violin'})
    df_combined = pd.concat([df_plot_pop, df_plot_sub])

    # Statistics
    med_sub = s_sub.median()
    med_pop = s_pop.median()
    p_val = calculate_p_value_continuous(s_sub, s_pop)
    p_str = format_p_value(p_val)

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_VIOLIN)

    # Draw Violin
    sns.violinplot(
        data=df_combined, 
        x='Split', 
        y='Value', 
        hue='Group', 
        split=True, 
        inner='quart',
        palette={'Population': POPULATION_COLOR, 'Subset': SUBSET_COLOR},
        ax=ax
    )

    # Customization
    ax.set_title(f'Distribution of {pretty_name}', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel(pretty_name, fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('')
    ax.set_xticklabels([]) # Remove the 'Violin' label from x-axis
    
    # Legend
    ax.legend(title='Cohort', loc='upper right', frameon=True, fancybox=True, framealpha=0.9)

    # Annotations (Median Boxes)
    # Get plot dims to position text nicely
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # Population Stats Box (Left)
    text_pop = f"Population (n={len(s_pop)})\nMedian: {med_pop:.2f}"
    ax.text(
        -0.25, y_max - (y_range * 0.05), text_pop, 
        ha='center', va='top', fontsize=FONT_SIZE_ANNOT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=POPULATION_COLOR, linewidth=2, alpha=0.9)
    )

    # Subset Stats Box (Right)
    text_sub = f"Subset (n={len(s_sub)})\nMedian: {med_sub:.2f}"
    ax.text(
        0.25, y_max - (y_range * 0.05), text_sub, 
        ha='center', va='top', fontsize=FONT_SIZE_ANNOT,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=SUBSET_COLOR, linewidth=2, alpha=0.9)
    )

    # P-value (Center Bottom)
    text_sig = f"Statistical Comparison:\n{p_str}"
    ax.text(
        0, y_min + (y_range * 0.05), text_sig, 
        ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT + 2, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Violin: {os.path.basename(output_path)}")


def plot_stacked_bar(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    variable: str, 
    pretty_name: str, 
    output_path: str
):
    """
    Creates a stacked bar chart:
    - Bar 1: Mother Cohort (Population)
    - Bar 2: Input Cohort (Subset)
    Shows proportion of Yes/No (1/0) or True/False.
    """
    # Clean Data (ensure binary 0/1)
    s_sub = df_subset[variable].dropna().astype(int)
    s_pop = df_mother[variable].dropna().astype(int)
    
    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping {variable}: No valid data.")
        return

    # Calculate Proportions
    prop_sub = s_sub.mean() * 100
    prop_pop = s_pop.mean() * 100
    
    n_sub = len(s_sub)
    n_pop = len(s_pop)

    p_val = calculate_p_value_categorical(s_sub, s_pop)
    p_str = format_p_value(p_val)

    # Prepare Data for Plotting
    # Columns: Group, Prop_Achieved, Prop_NotAchieved
    plot_data = [
        {'Group': 'Population', 'Achieved': prop_pop, 'Not': 100 - prop_pop, 'N': n_pop},
        {'Group': 'Subset',     'Achieved': prop_sub, 'Not': 100 - prop_sub, 'N': n_sub}
    ]
    df_plot = pd.DataFrame(plot_data)

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_BAR)

    # Create Stacked Bars
    indices = np.arange(len(df_plot))
    width = 0.5

    p1 = ax.bar(indices, df_plot['Achieved'], width, color=ACHIEVED_COLOR, label='Yes / Achieved', alpha=0.9)
    p2 = ax.bar(indices, df_plot['Not'], width, bottom=df_plot['Achieved'], color=NOT_ACHIEVED_COLOR, label='No / Not Achieved', alpha=0.9)

    # Customization
    ax.set_title(f'{pretty_name} (%)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Percentage (%)', fontsize=FONT_SIZE_LABEL)
    ax.set_xticks(indices)
    ax.set_xticklabels(df_plot['Group'], fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 115) # Extra space for text

    ax.legend(loc='upper right', frameon=True)

    # Annotations on bars
    for i, row in df_plot.iterrows():
        # Label Achieved % inside blue bar
        if row['Achieved'] > 10: # Only show if bar is large enough
            ax.text(i, row['Achieved']/2, f"{row['Achieved']:.1f}%", ha='center', va='center', color='white', weight='bold', fontsize=FONT_SIZE_ANNOT)
        
        # Label N count on top
        ax.text(i, 102, f"n={row['N']}", ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT)

    # P-value annotation joining the bars
    # Draw a line between bars
    x1, x2 = 0, 1
    y, h = 108, 2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
    ax.text((x1+x2)*0.5, y+h, p_str, ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Bar Chart: {os.path.basename(output_path)}")


def plot_data_collection_timeline(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    date_col: str, 
    output_path: str
):
    """
    Overlapped histogram of data collection periods (by Month).
    """
    # Convert to datetime if needed
    s_sub = pd.to_datetime(df_subset[date_col], errors='coerce').dropna()
    s_pop = pd.to_datetime(df_mother[date_col], errors='coerce').dropna()

    if s_sub.empty or s_pop.empty:
        print(f"  ⚠️ Skipping Timeline: No valid dates in {date_col}")
        return

    # Create Month Periods (e.g., 2023-01)
    # converting to period 'M' and then to timestamp for plotting
    sub_months = s_sub.dt.to_period('M').dt.to_timestamp()
    pop_months = s_pop.dt.to_period('M').dt.to_timestamp()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Histograms
    # Using 'stepfilled' or 'bar' with alpha for overlap
    # We define bins based on the full range
    
    all_dates = pd.concat([sub_months, pop_months])
    if all_dates.empty: return

    min_date, max_date = all_dates.min(), all_dates.max()
    
    # Create bins by month
    bins = pd.date_range(start=min_date, end=max_date + pd.offsets.MonthBegin(1), freq='MS')
    
    # Plot Population first (Gray)
    ax.hist(
        pop_months, bins=bins, color=POPULATION_COLOR, alpha=0.5, 
        label=f'Population (n={len(s_pop)})', edgecolor='none'
    )
    
    # Plot Subset (Teal)
    ax.hist(
        sub_months, bins=bins, color=SUBSET_COLOR, alpha=0.7, 
        label=f'Subset (n={len(s_sub)})', edgecolor='none'
    )

    ax.set_title('Data Collection Timeline (Monthly Volumes)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Number of Records', fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Timeline: {os.path.basename(output_path)}")


def plot_kaplan_meier_comparison(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    time_col: str, 
    target_pct: int, 
    output_path: str
):
    """
    Kaplan-Meier Analysis for Time to X% Weight Loss using Lifelines.
    Constructs Time/Event data using 'days_to_X%_wl', 'X%_wl_achieved', and 'total_followup_days'.
    """
    event_col = f"{target_pct}%_wl_achieved"
    censor_col = "total_followup_days" # Fallback if event not achieved

    # Check columns
    required = [time_col, event_col, censor_col]
    
    # Data Preparation Function
    def prep_surv_data(df):
        msg_missing = [c for c in required if c not in df.columns]
        if msg_missing:
            return None, None
            
        T = []
        E = []
        
        # Robust row-wise data construction
        for i, row in df.iterrows():
            achieved = row[event_col]
            time_to_event = row[time_col]
            followup = row[censor_col]
            
            # Logic:
            # If Achieved == 1: Time = time_to_event, Event = 1
            # If Achieved == 0: Time = followup, Event = 0
            if pd.isna(achieved):
                continue
                
            if achieved == 1:
                # If time is missing despite achieved, we have corrupt data -> exclude
                if pd.isna(time_to_event):
                    continue 
                T.append(time_to_event)
                E.append(1)
            else:
                # Censored
                if pd.isna(followup):
                    continue
                T.append(followup)
                E.append(0)
        return np.array(T), np.array(E)

    T_sub, E_sub = prep_surv_data(df_subset)
    T_pop, E_pop = prep_surv_data(df_mother)
    
    if T_sub is None or len(T_sub) < 5 or T_pop is None or len(T_pop) < 5:
        # print(f"  ⚠️ Skipping KM {target_pct}%: Insufficient data.")
        return

    # Plot Setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_VIOLIN)

    # Use Lifelines Fitters
    kmf_pop = KaplanMeierFitter()
    kmf_sub = KaplanMeierFitter()

    # Fit and Plot Population (Background/Control)
    kmf_pop.fit(T_pop, event_observed=E_pop, label=f'Population (n={len(T_pop)})')
    kmf_pop.plot_survival_function(ax=ax, color=POPULATION_COLOR, linewidth=2.5, alpha=0.8, ci_alpha=0.1)

    # Fit and Plot Subset (Comparison)
    kmf_sub.fit(T_sub, event_observed=E_sub, label=f'Subset (n={len(T_sub)})')
    kmf_sub.plot_survival_function(ax=ax, color=SUBSET_COLOR, linewidth=2.5, alpha=0.9, ci_alpha=0.2)

    # Log-Rank Test
    results = logrank_test(T_pop, T_sub, event_observed_A=E_pop, event_observed_B=E_sub)
    p_val = results.p_value
    p_str = format_p_value(p_val)

    # Stats Annotation
    text_sig = f"Log-Rank Test:\n{p_str}"
    ax.text(
        0.05, 0.05, text_sig, 
        transform=ax.transAxes,
        ha='left', va='bottom', fontsize=FONT_SIZE_ANNOT + 2, weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9)
    )

    ax.set_title(f'Time to {target_pct}% Weight Loss (Kaplan-Meier)', fontsize=FONT_SIZE_TITLE, weight='bold', pad=20)
    ax.set_ylabel('Survival Probability (Not Achieved)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlabel('Days from Baseline', fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 1.05)
    
    # Clean up legend (Lifelines adds one by default, but we customize placement)
    ax.legend(loc='lower left', bbox_to_anchor=(0.3, 0.05), frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved KM Curve: {os.path.basename(output_path)}")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_comparative_visualizations(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    config: Any, 
    output_dir: str
):
    """
    Main execution function.
    Iterates through row_order in config, identifies variable types, and generates plots.
    """
    print("\n" + "="*60)
    print("STARTING COMPARATIVE VISUALIZATIONS")
    print("="*60)
    print(f"Output Directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    row_order = config.row_order

    for variable, pretty_name in row_order:
        # Skip UI delimiters/headers
        if variable == 'N' or variable.startswith('delim_'):
            continue
        
        # Verify existence
        if variable not in df_subset.columns or variable not in df_mother.columns:
            # print(f"  ⚠️ Variable {variable} not found in data.")
            continue
            
        # Determine Variable Type
        # Logic adapted from descriptive_comparisons.py heuristics
        is_categorical = False
        
        # Heuristic 1: Explicit known categorical markers
        if (variable == "sex_f" or 
            variable == "instant_dropout" or 
            variable.endswith("_achieved") or 
            variable.endswith("_dropout") or 
            variable.startswith("country_")):
            is_categorical = True
            
        # Heuristic 2: Low cardinality (e.g. < 10 unique values) implies categorical
        # Here, this is NOT optimal, because we have low-cardinality continous vars such as number of medical records - commented out
        # elif df_subset[variable].nunique() <= 10:
        #     is_categorical = True

        # Clean filename
        safe_name = variable.replace('%', 'pct').replace(' ', '_').replace('/', '_')
        
        if is_categorical:
            out_file = os.path.join(output_dir, f"{safe_name}_stack.png")
            plot_stacked_bar(df_subset, df_mother, variable, pretty_name, out_file)
        else:
            out_file = os.path.join(output_dir, f"{safe_name}_violin.png")
            plot_split_violin(df_subset, df_mother, variable, pretty_name, out_file)

    # --- 2. Temporal Analysis (Timeline) ---
    if 'baseline_date' in df_subset.columns and 'baseline_date' in df_mother.columns:
        print("\nPossible Timeline Analysis detected...")
        tl_path = os.path.join(output_dir, "data_collection_timeline.png")
        plot_data_collection_timeline(df_subset, df_mother, 'baseline_date', tl_path)

    # --- 3. Survival Analysis (Kaplan-Meier) ---
    # Targets: 5%, 10%, 15%
    for target in [5, 10, 15]:
        time_col = f"days_to_{target}%_wl"
        # Check if column exists in both
        if time_col in df_subset.columns and time_col in df_mother.columns:
            km_path = os.path.join(output_dir, f"time_to_{target}pct_wl_km.png")
            plot_kaplan_meier_comparison(df_subset, df_mother, time_col, target, km_path)

    print("\n" + "="*60)
    print("VISUALIZATION RUN COMPLETE")
    print("="*60)
