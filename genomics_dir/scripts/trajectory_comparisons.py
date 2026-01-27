"""
TRAJECTORY_COMPARISONS.PY
Version: 1.0

Description:
A module for visualizing individual-level weight loss trajectories (Spaghetti Plots).
It compares 'Mother Cohort' (Population) vs 'Input Cohort' (Genomics/Subset) in a 3-panel layout:
1. Panel 1: Population Trajectories + Smoothed Mean
2. Panel 2: Subset Trajectories + Smoothed Mean
3. Panel 3: Overlay of Smoothed Means

Data Source:
- Cohort IDs are supplied as DataFrames (must contain 'medical_record_id').
- Raw measurements are pulled from the 'measurements_filtered' table in 'pnk_db2_filtered.sqlite'.

Dependencies:
- pandas, numpy, matplotlib, seaborn, statsmodels
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

# Suppress LOWESS warnings (optional)
warnings.filterwarnings('ignore')

# constants
MEASUREMENTS_TABLE = "measurements_filtered"
ID_COL = "medical_record_id"
DATE_COL = "measurement_date"
WEIGHT_COL = "weight_kg"

# Styling
POPULATION_COLOR = '#ff0000'  # red
SUBSET_COLOR = '#002aff'      # blue
SMOOTH_FRAC = 0.3             # Fraction for LOWESS smoothing
ALPHA_SPAGHETTI = 0.05        # Transparency for individual lines
FIG_SIZE = (18, 6)            # Wide layout for 3 panels

def load_measurements(
    cohort_df: pd.DataFrame, 
    measurements_db_path: str
) -> pd.DataFrame:
    """
    Fetches raw measurements for the given cohort IDs from the filtered database.
    Calculates 'days_from_baseline' for each medical record.
    """
    if ID_COL not in cohort_df.columns:
        raise ValueError(f"Cohort DataFrame must contain '{ID_COL}'")

    ids_to_fetch = cohort_df[ID_COL].unique().astype(str).tolist()
    
    if not ids_to_fetch:
        return pd.DataFrame()

    print(f"  Fetching measurements for {len(ids_to_fetch)} records...")
    
    # Chunking to avoid SQLite limits if necessary (though usually fine for <999 vars)
    # Here we use a temp table approach for robustness with large lists of IDs
    
    conn = sqlite3.connect(measurements_db_path)
    
    # optimized loading: push IDs to a temp table then join
    try:
        cohort_df[[ID_COL]].astype(str).to_sql('temp_cohort_ids', conn, if_exists='replace', index=False)
        
        query = f"""
            SELECT m.{ID_COL}, m.{DATE_COL}, m.{WEIGHT_COL}
            FROM {MEASUREMENTS_TABLE} m
            INNER JOIN temp_cohort_ids t ON m.{ID_COL} = t.{ID_COL}
            WHERE m.{WEIGHT_COL} IS NOT NULL
            ORDER BY m.{ID_COL}, m.{DATE_COL}
        """
        df_meas = pd.read_sql_query(query, conn)
        
        # Determine baseline date (min date per record)
        # Convert date column safely
        df_meas[DATE_COL] = pd.to_datetime(df_meas[DATE_COL])
        
        # Groupby transform to get baseline
        df_meas['baseline_date'] = df_meas.groupby(ID_COL)[DATE_COL].transform('min')
        df_meas['days_from_baseline'] = (df_meas[DATE_COL] - df_meas['baseline_date']).dt.days
        
        # Remove negative days (sanity check) and outlier logic if needed? 
        # For now, keep everything >= 0
        df_meas = df_meas[df_meas['days_from_baseline'] >= 0]
        
    finally:
        conn.close()
        
    print(f"  ✓ Loaded {len(df_meas)} measurement points.")
    return df_meas


def calculate_smoothed_trajectory(
    df_meas: pd.DataFrame, 
    frac: float = SMOOTH_FRAC
):
    """
    Calculates a LOWESS smoothed trajectory for the given measurements.
    Returns (x_smooth, y_smooth, y_sem).
    """
    # Create daily statistics for aggregation first (reduces points for LOWESS)
    daily = df_meas.groupby('days_from_baseline')[WEIGHT_COL].agg(['mean', 'std', 'count']).reset_index()
    daily = daily[daily['count'] > 5] # Only smooth where we have minimal data
    
    if daily.empty:
        return None, None, None

    # Calculate Standard Error of Mean for confidence interval
    daily['sem'] = daily['std'] / np.sqrt(daily['count'])
    
    # LOWESS Smoothing
    # lowess returns [x, y] sorted by x
    z = sm.nonparametric.lowess(daily['mean'], daily['days_from_baseline'], frac=frac)
    x_smooth = z[:, 0]
    y_smooth = z[:, 1]
    
    # Smooth the SEM as well for the band
    # We might need to handle NaNs in SEM (count=1 -> std=NaN)
    daily['sem'] = daily['sem'].fillna(0)
    z_sem = sm.nonparametric.lowess(daily['sem'], daily['days_from_baseline'], frac=frac)
    sem_smooth = z_sem[:, 1]
    
    return x_smooth, y_smooth, sem_smooth


def plot_split_mean(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    mean_followup: float,
    color: str,
    label_prefix: str
):
    """
    Helper to plot the smoothed mean as Solid (<= mean_followup) and Dashed (> mean_followup).
    """
    if x is None or len(x) == 0:
        return
        
    mask_solid = x <= mean_followup
    mask_dashed = x > mean_followup
    
    # Solid Portion
    if np.any(mask_solid):
        ax.plot(x[mask_solid], y[mask_solid], color=color, linewidth=3, label=f'{label_prefix} Mean')
        
    # Dashed Portion
    if np.any(mask_dashed):
        # We need to connect the last solid point to the first dashed point to avoid a gap
        # Append the last solid point to the dashed arrays if exists
        x_dash = x[mask_dashed]
        y_dash = y[mask_dashed]
        
        if np.any(mask_solid):
            x_dash = np.insert(x_dash, 0, x[mask_solid][-1])
            y_dash = np.insert(y_dash, 0, y[mask_solid][-1])
            
        ax.plot(x_dash, y_dash, color=color, linewidth=3, linestyle='--')

def plot_panel(
    ax: plt.Axes, 
    df_meas: pd.DataFrame, 
    title: str, 
    color: str, 
    spaghetti_color: str = 'gray',
    show_spaghetti: bool = True
):
    """
    Plots a single panel (Spaghetti + Smoothed Mean).
    Includes split line logic (Solid vs Dashed based on mean follow-up).
    """
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Calculate Mean Follow-up
    # Max day for each patient, then mean of those maxes
    mean_followup = df_meas.groupby(ID_COL)['days_from_baseline'].max().mean()
    
    # 1. Spaghetti (Individual Lines)
    if show_spaghetti:
        # Plot ALL trajectories with rasterization for performance
        # Groupby is slow for 20k+ lines, so we just iterate fast or use LineCollection (advanced)
        # Using simple loop with rasterized=True is a good middle ground
        
        # To speed up plotting, we can put NaNs between trajectories and plot one giant line
        # But that messes up alphas overlapping.
        # We will stick to loop but trust rasterized=True to save the file size/render time.
        
        # Optimization: groupby is slow. 
        # Get unique IDs, then mask.
        unique_ids = df_meas[ID_COL].unique()
        
        for pid in unique_ids:
            # Masking is faster than groupby for simple extraction
            # Assuming dataframe is sorted by ID (which it is from SQL)
            # Actually, standard pandas plotting is the bottleneck. 
            pass 
        
        # Use a vectorized line plotting approach if possible? 
        # For simplicity and code safety, we stick to standard plot command but optimized.
        # Plotting 20k lines individually IS slow in Python loop.
        
        # Let's assume the user is willing to wait a few seconds for "ALL subjects".
        # We use a very low alpha.
        
        # Actually, let's just plot groups.
        for _, grp in df_meas.groupby(ID_COL):
            ax.plot(grp['days_from_baseline'], grp[WEIGHT_COL], 
                   color=spaghetti_color, alpha=0.02, linewidth=0.5, rasterized=True)

    # 2. Smoothed Mean
    x, y, sem = calculate_smoothed_trajectory(df_meas)
    
    if x is not None:
        plot_split_mean(ax, x, y, mean_followup, color, 'Smoothed')
        
        # Confidence Interval (Full Band for simplicity, or split?)
        # Let's plot full band with lower alpha
        ax.fill_between(x, y - 1.96*sem, y + 1.96*sem, color=color, alpha=0.2, label='95% CI')
    
    ax.set_xlabel("Days from Baseline")
    ax.set_ylabel("Weight (kg)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add vertical line for mean follow-up? 
    # Optional, but the switch from solid to dashed usually indicates it enough.
    # ax.axvline(mean_followup, color=color, linestyle=':', alpha=0.5, label=f'Mean Follow-up: {int(mean_followup)}d')

    if show_spaghetti:
        # Custom legend to avoid clutter from thousands of lines
        # Reseting legend to only show what we labeled explicitly
        handles, labels = ax.get_legend_handles_labels()
        # Filter unique labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def run_trajectory_comparison(
    df_subset: pd.DataFrame, 
    df_mother: pd.DataFrame, 
    measurements_db_path: str,
    output_dir: str,
    filename: str = "trajectory_comparison_spaghetti.png",
    cutoff_days: int = 365
):
    """
    Main Orchestrator for Trajectory Comparison.
    """
    print("\n" + "="*60)
    print("STARTING TRAJECTORY COMPARISON (SPAGHETTI PLOTS)")
    print("="*60)
    
    # 1. Load Measurements
    print("Population Cohort:")
    meas_pop = load_measurements(df_mother, measurements_db_path)
    
    print("Subset Cohort:")
    meas_sub = load_measurements(df_subset, measurements_db_path)
    
    # 2. Apply Cutoff (e.g. 1 year)
    if cutoff_days:
        meas_pop = meas_pop[meas_pop['days_from_baseline'] <= cutoff_days]
        meas_sub = meas_sub[meas_sub['days_from_baseline'] <= cutoff_days]
        print(f"  Applied cutoff: {cutoff_days} days")

    if meas_pop.empty or meas_sub.empty:
        print("  ⚠️ Insufficient measurement data to plot.")
        return

    # 3. Setup Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, sharey=True, sharex=True)
    
    # Panel 1: Population
    plot_panel(axes[0], meas_pop, f"Population (n={meas_pop[ID_COL].nunique()})", POPULATION_COLOR, spaghetti_color='gray')
    
    # Panel 2: Subset (Green Spaghetti)
    plot_panel(axes[1], meas_sub, f"Genomics Subset (n={meas_sub[ID_COL].nunique()})", SUBSET_COLOR, spaghetti_color=SUBSET_COLOR)
    
    # Panel 3: Overlay
    axes[2].set_title("Smoothed Trajectory Overlay", fontsize=14, fontweight='bold', pad=15)
    
    # Add spaghetti to overlay? (User requested "individual trajectories, not only smoothed")
    # Low alpha gray for population
    # Low alpha green for subset
    # This might be messy, but requested.
    for _, grp in meas_pop.groupby(ID_COL):
        axes[2].plot(grp['days_from_baseline'], grp[WEIGHT_COL], color='gray', alpha=0.01, linewidth=0.5, rasterized=True)
    for _, grp in meas_sub.groupby(ID_COL):
        axes[2].plot(grp['days_from_baseline'], grp[WEIGHT_COL], color=SUBSET_COLOR, alpha=0.01, linewidth=0.5, rasterized=True)
    
    # Calculate smooths and mean follow-up for Overlay
    x_pop, y_pop, sem_pop = calculate_smoothed_trajectory(meas_pop)
    mean_fu_pop = meas_pop.groupby(ID_COL)['days_from_baseline'].max().mean()
    
    x_sub, y_sub, sem_sub = calculate_smoothed_trajectory(meas_sub)
    mean_fu_sub = meas_sub.groupby(ID_COL)['days_from_baseline'].max().mean()
    
    if x_pop is not None:
        plot_split_mean(axes[2], x_pop, y_pop, mean_fu_pop, POPULATION_COLOR, 'Population')
        axes[2].fill_between(x_pop, y_pop - 1.96*sem_pop, y_pop + 1.96*sem_pop, color=POPULATION_COLOR, alpha=0.1)
        
    if x_sub is not None:
        plot_split_mean(axes[2], x_sub, y_sub, mean_fu_sub, SUBSET_COLOR, 'Subset')
        axes[2].fill_between(x_sub, y_sub - 1.96*sem_sub, y_sub + 1.96*sem_sub, color=SUBSET_COLOR, alpha=0.2)
        
    axes[2].set_xlabel("Days from Baseline")
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # Custom legend for Overlay
    handles, labels = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    
    plt.tight_layout()
    import os
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"  ✓ Saved Trajectory Plot: {filename}")
    print("="*60)
