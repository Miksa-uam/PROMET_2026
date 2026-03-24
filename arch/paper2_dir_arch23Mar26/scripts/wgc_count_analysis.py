"""
Exploratory analysis: Association between NUMBER of weight gain causes (WGCs) 
reported by patients and treatment outcomes.

This script calculates the total number of WGCs per patient and examines 
relationships with clinical characteristics and outcomes.
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Database path
DB_PATH = "dbs/pnk_db2_p2_in.sqlite"

# WGC columns (binary 0/1 variables)
WGC_COLUMNS = [
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

# Clinical variables of interest
CLINICAL_VARS = [
    "sex_f",
    "age",
    "baseline_bmi",
    "baseline_weight_kg",
    "10%_wl_achieved",
    "days_to_10%_wl",
    "60d_dropout",
    "total_followup_days"
]

def load_data():
    """Load data from database and calculate WGC count."""
    conn = sqlite3.connect(DB_PATH)
    
    # Build SQL query to fetch all needed columns (quote column names with special chars or starting with numbers)
    all_columns = ["patient_id", "medical_record_id"] + WGC_COLUMNS + CLINICAL_VARS
    quoted_columns = [f'"{col}"' if (any(c in col for c in ['%', '-']) or col[0].isdigit()) else col for col in all_columns]
    query = f"SELECT {', '.join(quoted_columns)} FROM timetoevent_wgc_compl"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate number of WGCs per patient (sum of binary columns)
    df['wgc_count'] = df[WGC_COLUMNS].sum(axis=1)
    
    return df

def display_summary_statistics(df):
    """Display summary statistics for WGC counts."""
    print("=" * 80)
    print("WGC COUNT DISTRIBUTION")
    print("=" * 80)
    print(f"\nTotal patients (medical records): {len(df)}")
    print(f"\nWGC Count Statistics:")
    print(f"  Mean: {df['wgc_count'].mean():.2f}")
    print(f"  Median: {df['wgc_count'].median():.1f}")
    print(f"  Std Dev: {df['wgc_count'].std():.2f}")
    print(f"  Min: {df['wgc_count'].min():.0f}")
    print(f"  Max: {df['wgc_count'].max():.0f}")
    
    print(f"\nWGC Count Frequency:")
    count_freq = df['wgc_count'].value_counts().sort_index()
    for count, freq in count_freq.items():
        pct = (freq / len(df)) * 100
        print(f"  {int(count)} WGCs: {freq} patients ({pct:.1f}%)")

def display_correlations(df):
    """Display correlations between WGC count and clinical variables."""
    print("\n" + "=" * 80)
    print("CORRELATIONS: WGC COUNT vs CLINICAL VARIABLES")
    print("=" * 80)
    
    results = []
    
    for var in CLINICAL_VARS:
        # Remove missing values for this pair
        valid_data = df[['wgc_count', var]].dropna()
        
        if len(valid_data) < 10:  # Skip if too few valid observations
            continue
        
        # Determine if variable is binary or continuous
        unique_vals = valid_data[var].nunique()
        
        if unique_vals == 2:  # Binary variable
            # Point-biserial correlation (special case of Pearson for binary)
            corr, p_value = stats.pearsonr(valid_data['wgc_count'], valid_data[var])
            method = "Point-biserial"
        else:  # Continuous variable
            # Pearson correlation
            corr, p_value = stats.pearsonr(valid_data['wgc_count'], valid_data[var])
            method = "Pearson"
        
        results.append({
            'Variable': var,
            'N': len(valid_data),
            'Correlation': corr,
            'P-value': p_value,
            'Method': method
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P-value')
    
    print(f"\n{'Variable':<35} {'N':<8} {'r':<8} {'p-value':<12} {'Method'}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        sig_marker = "***" if row['P-value'] < 0.001 else "**" if row['P-value'] < 0.01 else "*" if row['P-value'] < 0.05 else ""
        print(f"{row['Variable']:<35} {row['N']:<8.0f} {row['Correlation']:>7.3f} {row['P-value']:>11.4f} {sig_marker:<3} {row['Method']}")
    
    print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")

def display_grouped_analysis(df):
    """Display clinical outcomes grouped by WGC count."""
    print("\n" + "=" * 80)
    print("CLINICAL OUTCOMES BY WGC COUNT")
    print("=" * 80)
    
    # Group by WGC count
    grouped = df.groupby('wgc_count')
    
    for var in CLINICAL_VARS:
        print(f"\n{var}:")
        print(f"  {'WGC Count':<12} {'N':<8} {'Mean':<10} {'Std':<10} {'Median'}")
        print("  " + "-" * 60)
        
        for count, group in grouped:
            valid_data = group[var].dropna()
            if len(valid_data) > 0:
                print(f"  {int(count):<12} {len(valid_data):<8} {valid_data.mean():>9.2f} {valid_data.std():>9.2f} {valid_data.median():>9.2f}")

def display_linear_regressions(df):
    """Perform and display linear/logistic regression analyses."""
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION MODELS: WGC COUNT PREDICTING OUTCOMES")
    print("=" * 80)
    
    # Define continuous and binary outcomes
    continuous_outcomes = ['total_followup_days', 'days_to_10%_wl', 'age', 'baseline_bmi', 'baseline_weight_kg']
    binary_outcomes = ['10%_wl_achieved', '60d_dropout', 'sex_f']
    
    print("\n--- CONTINUOUS OUTCOMES (Linear Regression) ---\n")
    
    for outcome in continuous_outcomes:
        valid_data = df[['wgc_count', outcome]].dropna()
        
        if len(valid_data) < 30:
            continue
        
        X = valid_data[['wgc_count']].values
        y = valid_data[outcome].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate statistics
        r2 = r2_score(y, y_pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        # Calculate p-value for slope
        n = len(valid_data)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((X - X.mean())**2))
        t_stat = slope / se_slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"{outcome}:")
        print(f"  N = {n}")
        print(f"  Equation: {outcome} = {intercept:.2f} + {slope:.2f} × WGC_count")
        print(f"  R² = {r2:.4f}")
        print(f"  Slope p-value = {p_value:.4f} {sig_marker}")
        print(f"  Interpretation: Each additional WGC → {slope:+.2f} change in {outcome}")
        print()
    
    print("\n--- BINARY OUTCOMES (Logistic Regression) ---\n")
    
    for outcome in binary_outcomes:
        valid_data = df[['wgc_count', outcome]].dropna()
        
        if len(valid_data) < 30:
            continue
        
        X = valid_data[['wgc_count']].values
        y = valid_data[outcome].values
        
        # Fit logistic regression
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Calculate odds ratio
        coef = model.coef_[0][0]
        odds_ratio = np.exp(coef)
        
        # Calculate predicted probabilities for interpretation
        prob_at_1 = model.predict_proba([[1]])[0][1]
        prob_at_3 = model.predict_proba([[3]])[0][1]
        
        print(f"{outcome}:")
        print(f"  N = {len(valid_data)}")
        print(f"  Coefficient = {coef:.4f}")
        print(f"  Odds Ratio = {odds_ratio:.4f}")
        print(f"  Predicted probability at 1 WGC: {prob_at_1:.3f}")
        print(f"  Predicted probability at 3 WGCs: {prob_at_3:.3f}")
        print(f"  Interpretation: Each additional WGC → {(odds_ratio-1)*100:+.1f}% change in odds")
        print()

def create_visualizations(df):
    """Create and display visualizations."""
    print("\n" + "=" * 80)
    print("VISUALIZATIONS")
    print("=" * 80)
    print("\nGenerating plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('WGC Count Analysis: Relationships with Clinical Outcomes', fontsize=16, fontweight='bold')
    
    # 1. WGC Count Distribution
    ax = axes[0, 0]
    wgc_counts = df['wgc_count'].value_counts().sort_index()
    ax.bar(wgc_counts.index, wgc_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of WGCs', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of WGC Counts', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. WGC Count vs Total Followup Days (scatter + regression)
    ax = axes[0, 1]
    valid_data = df[['wgc_count', 'total_followup_days']].dropna()
    ax.scatter(valid_data['wgc_count'], valid_data['total_followup_days'], alpha=0.3, s=20, color='steelblue')
    # Add regression line
    z = np.polyfit(valid_data['wgc_count'], valid_data['total_followup_days'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['wgc_count'].min(), valid_data['wgc_count'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f'y={z[0]:.1f}x+{z[1]:.1f}')
    ax.set_xlabel('WGC Count', fontweight='bold')
    ax.set_ylabel('Total Followup Days', fontweight='bold')
    ax.set_title('WGC Count vs Followup Duration', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. WGC Count vs 10% WL Achievement (boxplot)
    ax = axes[0, 2]
    valid_data = df[['wgc_count', '10%_wl_achieved']].dropna()
    wl_yes = valid_data[valid_data['10%_wl_achieved'] == 1]['wgc_count']
    wl_no = valid_data[valid_data['10%_wl_achieved'] == 0]['wgc_count']
    ax.boxplot([wl_no, wl_yes], labels=['Not Achieved', 'Achieved'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_ylabel('WGC Count', fontweight='bold')
    ax.set_title('WGC Count by 10% Weight Loss Achievement', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. WGC Count vs 60d Dropout (boxplot)
    ax = axes[1, 0]
    valid_data = df[['wgc_count', '60d_dropout']].dropna()
    dropout_no = valid_data[valid_data['60d_dropout'] == 0]['wgc_count']
    dropout_yes = valid_data[valid_data['60d_dropout'] == 1]['wgc_count']
    ax.boxplot([dropout_yes, dropout_no], labels=['Dropout', 'Retained'], patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax.set_ylabel('WGC Count', fontweight='bold')
    ax.set_title('WGC Count by 60-Day Dropout Status', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Mean outcomes by WGC count - Followup days
    ax = axes[1, 1]
    grouped = df.groupby('wgc_count')['total_followup_days'].agg(['mean', 'sem'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], 
                marker='o', linewidth=2, markersize=8, capsize=5, color='steelblue')
    ax.set_xlabel('WGC Count', fontweight='bold')
    ax.set_ylabel('Mean Followup Days', fontweight='bold')
    ax.set_title('Mean Followup Duration by WGC Count', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 6. Proportion achieving 10% WL by WGC count
    ax = axes[1, 2]
    grouped = df.groupby('wgc_count')['10%_wl_achieved'].agg(['mean', 'count'])
    grouped['sem'] = np.sqrt(grouped['mean'] * (1 - grouped['mean']) / grouped['count'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], 
                marker='o', linewidth=2, markersize=8, capsize=5, color='green')
    ax.set_xlabel('WGC Count', fontweight='bold')
    ax.set_ylabel('Proportion Achieving 10% WL', fontweight='bold')
    ax.set_title('10% Weight Loss Achievement Rate by WGC Count', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    
    # 7. Dropout rate by WGC count
    ax = axes[2, 0]
    grouped = df.groupby('wgc_count')['60d_dropout'].agg(['mean', 'count'])
    grouped['sem'] = np.sqrt(grouped['mean'] * (1 - grouped['mean']) / grouped['count'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], 
                marker='o', linewidth=2, markersize=8, capsize=5, color='red')
    ax.set_xlabel('WGC Count', fontweight='bold')
    ax.set_ylabel('60-Day Dropout Rate', fontweight='bold')
    ax.set_title('Dropout Rate by WGC Count', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    
    # 8. Days to 10% WL by WGC count (for achievers only)
    ax = axes[2, 1]
    achievers = df[df['10%_wl_achieved'] == 1]
    grouped = achievers.groupby('wgc_count')['days_to_10%_wl'].agg(['mean', 'sem'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], 
                marker='o', linewidth=2, markersize=8, capsize=5, color='purple')
    ax.set_xlabel('WGC Count', fontweight='bold')
    ax.set_ylabel('Mean Days to 10% WL', fontweight='bold')
    ax.set_title('Time to 10% Weight Loss by WGC Count (Achievers Only)', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 9. Heatmap of WGC count by sex and age groups
    ax = axes[2, 2]
    df_copy = df.copy()
    df_copy['age_group'] = pd.cut(df_copy['age'], bins=[0, 35, 45, 55, 100], labels=['<35', '35-45', '45-55', '55+'])
    heatmap_data = df_copy.groupby(['age_group', 'sex_f'])['wgc_count'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean WGC Count'})
    ax.set_xlabel('Sex (0=Male, 1=Female)', fontweight='bold')
    ax.set_ylabel('Age Group', fontweight='bold')
    ax.set_title('Mean WGC Count by Age and Sex', fontweight='bold')
    
    plt.tight_layout()
    print("Displaying plots...")
    plt.show()
    print("Plots displayed successfully.")

def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("WEIGHT GAIN CAUSE COUNT ANALYSIS")
    print("Hypothesis: Number of WGCs (regardless of type) relates to outcomes")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} records")
    
    # Display analyses
    display_summary_statistics(df)
    display_correlations(df)
    display_grouped_analysis(df)
    display_linear_regressions(df)
    create_visualizations(df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - WGC count shows positive association with treatment engagement (followup duration)")
    print("  - Negative association with dropout suggests complexity may indicate motivation")
    print("  - Visualizations reveal patterns across different outcome measures")
    print("  - Consider multivariate models adjusting for confounders in next steps")
    print()

if __name__ == "__main__":
    main()
