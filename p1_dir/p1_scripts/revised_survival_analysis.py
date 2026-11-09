"""
Revised Survival Analysis Module (Paper 1 Revision)

Scientific Purpose:
Addresses Reviewer Comment 2 regarding high attrition (48.5% dropout at 60 days),
informative censoring, and speed of achieving 10% weight loss:
1. Compares 60-day completers vs. 60-day dropouts on baseline traits, treatment engagement,
   and self-reported eating behaviors (with SMDs and Benjamini-Hochberg FDR q-values).
2. Corrects the Kaplan-Meier analysis:
   - Eliminates the selection distortion of dropping non-achievers.
   - Properly right-censors all 1,052 patients across the observational timeframe.
   - Inverts the curve to Cumulative Incidence (1 - S(t)), starting at 0% and rising upward.
   - Restores square aspect ratio, publication styling, and a clean Numbers-at-Risk table.
   - Supports flexible time horizon capping (e.g. 60/65 days vs. full follow-up).
3. Implements Competing Risks Analysis (Aalen-Johansen Cumulative Incidence Function):
   - Formally accounts for early patient dropout as a competing risk to 10% weight loss.
   - Uses a pure continuous time-to-event competing risk definition (no arbitrary milestone flags).
   - Conducts formal inferential hypothesis testing via both Unadjusted and Multivariable-Adjusted
     Cause-Specific Cox Proportional Hazards Models (adjusting for age, sex, and baseline BMI).
   - Adds aligned Numbers-at-Risk tables below the CIF curves.
   - Displays both Unadjusted and Adjusted Hazard Ratios with p-values in plot titles.
   - Eliminates boundary tie distortions and duplicate file saves.
"""

import os
import sys
import sqlite3
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Suppress benign numerical tie-breaking warnings from lifelines
warnings.filterwarnings("ignore", message=".*Tied event times were detected.*")

# Ensure local script directory is on sys.path
for script_dir in [os.path.join(".", "p1_scripts"), "p1_scripts", "."]:
    if os.path.exists(os.path.join(script_dir, "stats_helpers.py")):
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        break

from stats_helpers import (
    format_mean_sd,
    format_n_perc,
    format_median_iqr,
    format_pval,
    calculate_effect_size,
    format_effect_size,
    adjust_pvalues_fdr,
    run_bootstrap_comparison,
)

from lifelines import KaplanMeierFitter, AalenJohansenFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Publication configuration
PUBLICATION_NAMES = {
    "emotional_eating_yn": "Emotional eating (yes/no)",
    "emotional_eating_value_likert": "Emotional eating score (1-10)",
    "emotional_eating_median_split": "Emotional eating score (1-10)",
    "quantity_control_likert": "Quantity control score (1-10)",
    "quantity_control_median_split": "Quantity control score (1-10)",
}

# ==============================================================================
# 1. 60-DAY DROPOUT VS. COMPLETER COMPARISON
# ==============================================================================

DROPOUT_COMP_VARS = [
    ("N", "N", "header"),
    # Demographics & baseline anthropometry
    ("delim_demo", "Demographics and baseline anthropometry", "header"),
    ("sex_f", "Sex (% of females)", "categorical"),
    ("age", "Age (years)", "continuous"),
    # ("height_m", "Height (m)", "continuous"),
    ("baseline_weight_kg", "Baseline weight (kg)", "continuous"),
    ("baseline_bmi", "Baseline BMI (kg/m²)", "continuous"),
    ("baseline_fat_%", "Baseline fat mass (%)", "continuous"),
    # ("baseline_muscle_%", "Baseline muscle mass (%)", "continuous"),
    # Eating behaviors
    ("delim_eb", "Self-reported eating behaviors", "header"),
    ("hunger_yn", "Hunger (yes/no)", "categorical"),
    ("satiety_yn", "Satiety (yes/no)", "categorical"),
    ("emotional_eating_yn", "Emotional eating (yes/no)", "categorical"),
    ("emotional_eating_value_likert", "Emotional eating score (1-10)", "continuous"),
    ("quantity_control_likert", "Quantity control score (1-10)", "continuous"),
    ("impulse_control_likert", "Impulse control score (1-10)", "continuous"),
    # Engagement & follow-up metrics
    ("delim_eng", "Treatment engagement and follow-up", "header"),
    ("total_followup_days", "Total follow-up length (days)", "continuous"),
    # ("dietitian_visits", "Number of dietitian visits", "continuous"),
    # ("nr_total_measurements", "Total number of measurements", "continuous"),
    # ("avg_days_between_measurements", "Average measurement frequency (days)", "continuous"),
    # Treatment outcomes
    ("delim_out", "Clinical weight loss outcomes", "header"),
    ("total_wl_%", "Total weight loss (%)", "continuous"),
    # ("final_bmi", "Final BMI (kg/m²)", "continuous"),
    ("bmi_reduction", "BMI reduction (kg/m²)", "continuous"),
    ("10%_wl_achieved", "Achieved 10% weight loss (n)", "categorical"),
    ("days_to_10%_wl", "Days to 10% weight loss", "continuous"),
]

def compare_60d_dropouts_vs_completers(df):
    """
    Compares 60-day completers (60d_dropout == 0) versus 60-day dropouts (60d_dropout == 1)
    on baseline traits, engagement metrics, eating behaviors, and outcomes.
    
    Args:
        df (pd.DataFrame): Cohort dataframe from timetoevent_eb_compl (N=1,052).
        
    Returns:
        pd.DataFrame: Formatted comparison table with effect sizes (SMDs) and FDR q-values.
    """
    print("\n--- Running 60-Day Dropout vs. Completer Comparison ---")
    g_comp = df[df["60d_dropout"] == 0].copy()
    g_drop = df[df["60d_dropout"] == 1].copy()
    
    col_comp_name = f"Completers (n={len(g_comp)})"
    col_drop_name = f"Dropouts (n={len(g_drop)})"
    
    rows = []
    raw_p_list = []
    
    # N row
    rows.append({
        "Variable": "N",
        col_comp_name: str(len(g_comp)),
        col_drop_name: str(len(g_drop)),
        "Effect Size (95% CI)": "N/A",
        "Raw p-value": "N/A",
        "FDR p-value": "N/A"
    })
    
    tested_indices = []
    
    for var, label, vtype in DROPOUT_COMP_VARS:
        if vtype == "header":
            if var == "N":
                continue
            rows.append({
                "Variable": label,
                col_comp_name: "",
                col_drop_name: "",
                "Effect Size (95% CI)": "",
                "Raw p-value": "",
                "FDR p-value": ""
            })
            continue
            
        if var not in df.columns:
            continue
            
        s_c = g_comp[var].dropna()
        s_d = g_drop[var].dropna()
        
        if vtype == "continuous":
            val_c = f"{format_mean_sd(s_c)} | {format_median_iqr(s_c)}"
            val_d = f"{format_mean_sd(s_d)} | {format_median_iqr(s_d)}"
        else:
            val_c = format_n_perc(s_c)
            val_d = format_n_perc(s_d)
            
        es = calculate_effect_size(s_c, s_d, vtype)
        es_str = format_effect_size(es)
        
        comp_res = run_bootstrap_comparison(s_c, s_d, vtype, run_bootstrap=False)
        p_raw = comp_res["std_p"]
        
        rows.append({
            "Variable": label,
            col_comp_name: val_c,
            col_drop_name: val_d,
            "Effect Size (95% CI)": es_str,
            "raw_p": p_raw
        })
        tested_indices.append(len(rows) - 1)
        raw_p_list.append(p_raw)
        
    fdr_p_list = adjust_pvalues_fdr(raw_p_list)
    
    for idx, p_raw, p_fdr in zip(tested_indices, raw_p_list, fdr_p_list):
        rows[idx]["Raw p-value"] = format_pval(p_raw)
        rows[idx]["FDR p-value"] = format_pval(p_fdr)
        del rows[idx]["raw_p"]
        
    res_df = pd.DataFrame(rows)
    print("Dropout comparison table successfully created.")
    return res_df


# ==============================================================================
# 2. CORRECTED KAPLAN-MEIER WITH NUMBERS-AT-RISK
# ==============================================================================

def prepare_survival_cohort(df, time_horizon_cap=None):
    """
    Constructs proper survival duration and right-censoring indicators for all 1,052 patients.
    Supports capping follow-up at a specific time horizon (e.g., 60 or 65 days).
    
    Args:
        df (pd.DataFrame): Cohort dataframe from timetoevent_eb_compl (N=1,052).
        time_horizon_cap (int, optional): Maximum follow-up time in days (e.g. 65). Defaults to None.
        
    Returns:
        tuple: (pd.DataFrame, float) containing prepared dataframe and median Likert score.
    """
    df_surv = df.copy()
    
    if time_horizon_cap is not None:
        achieved = (df_surv["10%_wl_achieved"] == 1) & (df_surv["days_to_10%_wl"] <= time_horizon_cap)
        df_surv["event"] = np.where(achieved, 1, 0)
        df_surv["duration"] = np.where(
            achieved,
            df_surv["days_to_10%_wl"],
            df_surv["total_followup_days"]
        )
        # Censored patients active past the cap are placed immediately after the window
        df_surv["duration"] = pd.to_numeric(df_surv["duration"], errors="coerce").fillna(1).clip(lower=1, upper=time_horizon_cap + 0.1)
    else:
        achieved = (df_surv["10%_wl_achieved"] == 1)
        df_surv["event"] = np.where(achieved, 1, 0)
        df_surv["duration"] = np.where(
            achieved,
            df_surv["days_to_10%_wl"],
            df_surv["total_followup_days"]
        )
        df_surv["duration"] = pd.to_numeric(df_surv["duration"], errors="coerce").fillna(1).clip(lower=1, upper=180.1)
    
    num_likert = pd.to_numeric(df_surv["emotional_eating_value_likert"], errors="coerce")
    median_ee = num_likert.median()
    df_surv["emotional_eating_median_split"] = (num_likert >= median_ee).astype(int)
    
    num_qc = pd.to_numeric(df_surv["quantity_control_likert"], errors="coerce")
    median_qc = num_qc.median()
    df_surv["quantity_control_median_split"] = (num_qc >= median_qc).astype(int)
    
    return df_surv, median_ee


def add_clean_colored_at_risk_table(ax, durations_list, colors, xticks):
    """
    Renders an elegant, fixed-width Numbers-at-Risk table below the x-axis.
    Uses colored horizontal line segments matching the strata curves instead of
    variable-length text strings. This guarantees 100% identical figure proportions
    across all plots and eliminates 'KM_estimate' artifacts.
    
    Args:
        ax (matplotlib.axes.Axes): Target plot axes.
        durations_list (list of pd.Series/np.ndarray): Durations for [group_0, group_1].
        colors (list of str): Plot line colors [color_0, color_1].
        xticks (list of int): Time points where active counts are evaluated.
    """
    y_header = -0.15
    y_row0 = -0.21
    y_row1 = -0.27
    
    line_x0 = -0.09
    line_x1 = -0.04
    
    # 1. Header 'At risk' aligned above the line indicators
    ax.text(
        line_x0, y_header, "At risk", transform=ax.transAxes,
        ha="left", va="center", fontsize=9.5, fontweight="bold", clip_on=False
    )
    
    # 2. Strata colored line indicators matching curves and legend
    ax.plot(
        [line_x0, line_x1], [y_row0, y_row0], transform=ax.transAxes,
        color=colors[0], linewidth=2.5, clip_on=False
    )
    ax.plot(
        [line_x0, line_x1], [y_row1, y_row1], transform=ax.transAxes,
        color=colors[1], linewidth=2.5, clip_on=False
    )
    
    # 3. Numeric counts at each tick mark
    trans_x_data = ax.get_xaxis_transform()
    for tick in xticks:
        c0 = (durations_list[0] >= tick).sum()
        c1 = (durations_list[1] >= tick).sum()
        ax.text(
            tick, y_row0, str(int(c0)), transform=trans_x_data,
            ha="center", va="center", fontsize=9.5, clip_on=False
        )
        ax.text(
            tick, y_row1, str(int(c1)), transform=trans_x_data,
            ha="center", va="center", fontsize=9.5, clip_on=False
        )


def plot_corrected_kaplan_meier(df_surv, strata_col, strata_label, output_dir, time_horizon_cap=None, show_plot=True):
    """
    Generates and saves a square, publication-grade Kaplan-Meier plot with proper right-censoring,
    inverted Cumulative Incidence scaling (1 - S(t)), and a clean Numbers-at-Risk table.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    
    max_days = time_horizon_cap if time_horizon_cap is not None else 180
    
    groups = sorted(df_surv[strata_col].dropna().unique())
    if len(groups) < 2:
        return
        
    g0 = df_surv[df_surv[strata_col] == groups[0]]
    g1 = df_surv[df_surv[strata_col] == groups[1]]
    
    lr_res = logrank_test(
        g0["duration"], g1["duration"],
        event_observed_A=g0["event"], event_observed_B=g1["event"]
    )
    p_val = lr_res.p_value
    p_text = f"Log-rank p={p_val:.3f}"
    
    pub_strata_name = PUBLICATION_NAMES.get(strata_col, strata_label)
    if strata_col.endswith("_yn"):
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: No"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: Yes"
        file_suffix = strata_col
    else:
        if "emotional_eating" in strata_col:
            orig_col = "emotional_eating_value_likert"
        elif "quantity_control" in strata_col:
            orig_col = "quantity_control_likert"
        else:
            orig_col = strata_col.replace("_median_split", "")
            
        if orig_col in df_surv.columns:
            med_val = int(pd.to_numeric(df_surv[orig_col], errors="coerce").median())
            label_0 = f"{pub_strata_name.split('(')[0].strip()}: < median ({med_val})"
            label_1 = f"{pub_strata_name.split('(')[0].strip()}: >= median ({med_val})"
            file_suffix = orig_col
        else:
            label_0 = f"{pub_strata_name}: Group 0"
            label_1 = f"{pub_strata_name}: Group 1"
            file_suffix = strata_col
        
    kmf_0 = KaplanMeierFitter()
    kmf_0.fit(g0["duration"], event_observed=g0["event"])
    
    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(g1["duration"], event_observed=g1["event"])

    val0_km = (1.0 - kmf_0.predict(max_days)) * 100
    val1_km = (1.0 - kmf_1.predict(max_days)) * 100

    lbl_0_km = f"{label_0} ({val0_km:.1f}%)"
    lbl_1_km = f"{label_1} ({val1_km:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {0: "blue", 1: "orange"}
    
    kmf_0.plot_cumulative_density(ax=ax, color=colors[0], linewidth=1.8, ci_alpha=0.12, label=lbl_0_km)
    kmf_1.plot_cumulative_density(ax=ax, color=colors[1], linewidth=1.8, ci_alpha=0.12, label=lbl_1_km)

    star = "*" if p_val < 0.05 else ""
    ax.text(
        max_days + 1.0, val0_km / 100, f"{val0_km:.1f}%{star}",
        color=colors[0], va="center", ha="left", fontweight="bold", fontsize=10, clip_on=False
    )
    ax.text(
        max_days + 1.0, val1_km / 100, f"{val1_km:.1f}%{star}",
        color=colors[1], va="center", ha="left", fontweight="bold", fontsize=10, clip_on=False
    )
    
    title_time = f" (0 to {time_horizon_cap} Days)" if time_horizon_cap is not None else ""
    ax.set_title(
        f"Time to 10% weight loss by {pub_strata_name}\n{p_text}", fontsize=13, # fontweight="bold"
    )
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Probability of weight loss target achievement", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title=pub_strata_name, loc="upper left", fontsize=10, frameon=True)
    ax.set_ylim(0, 1.05)
    
    # Regular 15-day intervals for clean table alignment
    if max_days <= 75:
        xticks = [0, 15, 30, 45, 60]
    else:
        xticks = [0, 30, 60, 90, 120, 150, 180]
        
    ax.set_xticks(xticks)
    ax.set_xlim(0, max_days)
    
    add_clean_colored_at_risk_table(ax, [g0["duration"], g1["duration"]], [colors[0], colors[1]], xticks)
    
    fig.subplots_adjust(left=0.12, right=0.88, top=0.90, bottom=0.20)
    
    os.makedirs(output_dir, exist_ok=True)
    cap_str = f"_{time_horizon_cap}d" if time_horizon_cap is not None else ""
    svg_path = os.path.join(output_dir, f"km_10perc_wl{cap_str}_by_{file_suffix}.svg")
    png_path = os.path.join(output_dir, f"km_10perc_wl{cap_str}_by_{file_suffix}.png")
    
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        
    print(f"  - Saved KM SVG: {svg_path}")
    print(f"  - Saved KM PNG: {png_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)
    
    return {
        "strata": pub_strata_name,
        "time_horizon_cap": time_horizon_cap if time_horizon_cap is not None else "Full (180d)",
        "log_rank_p": p_val,
        "group_0_label": label_0,
        "group_0_n": len(g0),
        "group_0_events": int(g0["event"].sum()),
        "group_1_label": label_1,
        "group_1_n": len(g1),
        "group_1_events": int(g1["event"].sum())
    }


# ==============================================================================
# 3. COMPETING RISKS ANALYSIS (CUMULATIVE INCIDENCE FUNCTION)
# ==============================================================================

def prepare_competing_risks_cohort(df, time_horizon_cap=None):
    """
    Constructs competing risks variables using pure continuous follow-up logic.
    Completely removes arbitrary milestone flags (instant_dropout, 60d_dropout, 80d_dropout).
    
    Event definitions:
    - Status 1: 10% weight loss achieved on or before time_horizon_cap (Target Event)
    - Status 2: Ceased follow-up prior to time_horizon_cap without 10% WL (Competing Event)
    - Status 0: Active in follow-up past time_horizon_cap without 10% WL (Right-censored at cap + 0.1)
    
    Args:
        df (pd.DataFrame): Cohort dataframe from timetoevent_eb_compl (N=1,052).
        time_horizon_cap (int, optional): Maximum observation horizon (e.g. 65). Defaults to None (180).
        
    Returns:
        tuple: (pd.DataFrame, float) containing prepared dataframe and median Likert score.
    """
    df_cr = df.copy()
    max_days = time_horizon_cap if time_horizon_cap is not None else 180
    
    status = []
    duration = []
    
    for _, row in df_cr.iterrows():
        achieved_within_cap = (row["10%_wl_achieved"] == 1) and (row["days_to_10%_wl"] <= max_days)
        
        if achieved_within_cap:
            status.append(1)  # Target Event
            duration.append(max(1, row["days_to_10%_wl"]))
        elif row["total_followup_days"] < max_days:
            status.append(2)  # Competing Dropout Event
            duration.append(max(1, row["total_followup_days"]))
        else:
            status.append(0)  # Censored: Active in program past max_days
            # Placed immediately after the observation window so active patients remain in the risk set
            duration.append(max_days + 0.1)
            
    df_cr["cr_status"] = status
    df_cr["cr_duration"] = duration
    
    num_likert = pd.to_numeric(df_cr["emotional_eating_value_likert"], errors="coerce")
    median_ee = num_likert.median()
    df_cr["emotional_eating_median_split"] = (num_likert >= median_ee).astype(int)
    
    num_qc = pd.to_numeric(df_cr["quantity_control_likert"], errors="coerce")
    median_qc = num_qc.median()
    df_cr["quantity_control_median_split"] = (num_qc >= median_qc).astype(int)
    
    return df_cr, median_ee


def test_competing_risks_significance(df_cr, strata_col):
    """
    Performs formal inferential hypothesis testing for the target event in competing risks
    using Cause-Specific Cox Proportional Hazards Models:
    1. Unadjusted (bivariate): cr_duration ~ strata_col
    2. Multivariable-Adjusted: cr_duration ~ strata_col + age + sex_f + baseline_bmi
    
    In cause-specific survival, Event 1 (10% WL) is the event of interest;
    Event 2 (competing dropout) is treated as censored at its dropout time.
    
    Args:
        df_cr (pd.DataFrame): Cohort prepared by prepare_competing_risks_cohort.
        strata_col (str): Column name to test.
        
    Returns:
        dict: Unadjusted and Adjusted Hazard Ratios, 95% CIs, and p-values.
    """
    df_test = df_cr.copy()
    df_test["cs_event"] = (df_test["cr_status"] == 1).astype(int)
    
    # 1. Unadjusted model
    cph_unadj = CoxPHFitter()
    cph_unadj.fit(
        df_test[["cr_duration", "cs_event", strata_col]],
        duration_col="cr_duration",
        event_col="cs_event"
    )
    
    unadj_hr = cph_unadj.hazard_ratios_[strata_col]
    ci_u = cph_unadj.confidence_intervals_.loc[strata_col]
    unadj_hr_lower = np.exp(ci_u.iloc[0])
    unadj_hr_upper = np.exp(ci_u.iloc[1])
    unadj_p = cph_unadj.summary.loc[strata_col, "p"]
    
    # 2. Multivariable-Adjusted model (adjusting for age, sex_f, baseline_bmi)
    covars = [c for c in ["age", "sex_f", "baseline_bmi"] if c in df_test.columns]
    if covars:
        adj_cols = ["cr_duration", "cs_event", strata_col] + covars
        df_adj = df_test[adj_cols].dropna()
        cph_adj = CoxPHFitter()
        cph_adj.fit(
            df_adj,
            duration_col="cr_duration",
            event_col="cs_event"
        )
        adj_hr = cph_adj.hazard_ratios_[strata_col]
        ci_a = cph_adj.confidence_intervals_.loc[strata_col]
        adj_hr_lower = np.exp(ci_a.iloc[0])
        adj_hr_upper = np.exp(ci_a.iloc[1])
        adj_p = cph_adj.summary.loc[strata_col, "p"]
    else:
        adj_hr = unadj_hr
        adj_hr_lower = unadj_hr_lower
        adj_hr_upper = unadj_hr_upper
        adj_p = unadj_p

    return {
        "hazard_ratio": unadj_hr,
        "hr_lower": unadj_hr_lower,
        "hr_upper": unadj_hr_upper,
        "p_value": unadj_p,
        "adj_hazard_ratio": adj_hr,
        "adj_hr_lower": adj_hr_lower,
        "adj_hr_upper": adj_hr_upper,
        "adj_p_value": adj_p
    }


def plot_competing_risks_cif(df_cr, strata_col, strata_label, output_dir, time_horizon_cap=None, plot_dropout=False, show_plot=True):
    """
    Plots the Cumulative Incidence Function (CIF) with elegant square styling,
    adds Numbers-at-Risk below the axes, and displays both Unadjusted and Adjusted
    Cause-Specific Hazard Ratios with significance p-values in the title.
    
    Supports toggling competing dropout curves (plot_dropout=True/False).
    
    Args:
        df_cr (pd.DataFrame): Cohort prepared by prepare_competing_risks_cohort.
        strata_col (str): Column name to stratify by.
        strata_label (str): Human-readable publication label for stratification.
        output_dir (str): Directory where figures will be saved.
        time_horizon_cap (int, optional): X-axis maximum limit (e.g. 65). Defaults to None (180).
        plot_dropout (bool): Whether to plot competing dropout curves alongside 10% WL. Defaults to False.
        show_plot (bool): Whether to display the plot inline. Defaults to True.
        
    Returns:
        dict: Summary statistics including Unadjusted and Adjusted Cause-Specific HRs and p-values.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    
    max_days = time_horizon_cap if time_horizon_cap is not None else 180
    
    groups = sorted(df_cr[strata_col].dropna().unique())
    g0 = df_cr[df_cr[strata_col] == groups[0]]
    g1 = df_cr[df_cr[strata_col] == groups[1]]
    
    pub_strata_name = PUBLICATION_NAMES.get(strata_col, strata_label)
    if strata_col.endswith("_yn"):
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: No"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: Yes"
        file_suffix = strata_col
    else:
        if "emotional_eating" in strata_col:
            orig_col = "emotional_eating_value_likert"
        elif "quantity_control" in strata_col:
            orig_col = "quantity_control_likert"
        else:
            orig_col = strata_col.replace("_median_split", "")
            
        if orig_col in df_cr.columns:
            med_val = int(pd.to_numeric(df_cr[orig_col], errors="coerce").median())
            label_0 = f"{pub_strata_name.split('(')[0].strip()}: < median ({med_val})"
            label_1 = f"{pub_strata_name.split('(')[0].strip()}: >= median ({med_val})"
            file_suffix = orig_col
        else:
            label_0 = f"{pub_strata_name}: Group 0"
            label_1 = f"{pub_strata_name}: Group 1"
            file_suffix = strata_col
        
    # Statistical significance test for competing risks (Unadjusted & Adjusted)
    sig_res = test_competing_risks_significance(df_cr, strata_col)
    unadj_hr = sig_res["hazard_ratio"]
    unadj_p = sig_res["p_value"]
    adj_hr = sig_res["adj_hazard_ratio"]
    adj_p = sig_res["adj_p_value"]
    
    sig_text = f"Cause-specific HR = {unadj_hr:.2f} (p = {unadj_p:.3f}) | Adjusted HR* = {adj_hr:.2f} (p = {adj_p:.3f})"
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {0: "blue", 1: "orange"}
    
    # Event 1: 10% Weight Loss Target Achievement
    ajf_0 = AalenJohansenFitter(calculate_variance=True)
    ajf_0.fit(g0["cr_duration"], g0["cr_status"], event_of_interest=1)
    
    ajf_1 = AalenJohansenFitter(calculate_variance=True)
    ajf_1.fit(g1["cr_duration"], g1["cr_status"], event_of_interest=1)

    val0 = ajf_0.cumulative_density_["CIF_1"].loc[:max_days].iloc[-1] * 100
    val1 = ajf_1.cumulative_density_["CIF_1"].loc[:max_days].iloc[-1] * 100

    lbl_0_target = f"{label_0} ({val0:.1f}%)" if not plot_dropout else f"{label_0} (10% WL)"
    lbl_1_target = f"{label_1} ({val1:.1f}%)" if not plot_dropout else f"{label_1} (10% WL)"

    ajf_0.plot(ax=ax, color=colors[0], linewidth=1.8, label=lbl_0_target)
    ajf_1.plot(ax=ax, color=colors[1], linewidth=1.8, label=lbl_1_target)

    # Annotate final cumulative incidence outside the right plot border
    if not plot_dropout:
        star = "*" if (unadj_p < 0.05 or adj_p < 0.05) else ""
        ax.text(
            max_days + 1.0, val0 / 100, f"{val0:.1f}%{star}",
            color=colors[0], va="center", ha="left", fontweight="bold", fontsize=10, clip_on=False
        )
        ax.text(
            max_days + 1.0, val1 / 100, f"{val1:.1f}%{star}",
            color=colors[1], va="center", ha="left", fontweight="bold", fontsize=10, clip_on=False
        )
    
    # Event 2: Competing Early Dropout (optional)
    if plot_dropout:
        ajf_drop_0 = AalenJohansenFitter(calculate_variance=True)
        ajf_drop_0.fit(g0["cr_duration"], g0["cr_status"], event_of_interest=2, label=f"{label_0} (Dropout)")
        ajf_drop_0.plot(ax=ax, color=colors[0], linestyle="--", linewidth=1.5)
        
        ajf_drop_1 = AalenJohansenFitter(calculate_variance=True)
        ajf_drop_1.fit(g1["cr_duration"], g1["cr_status"], event_of_interest=2, label=f"{label_1} (Dropout)")
        ajf_drop_1.plot(ax=ax, color=colors[1], linestyle="--", linewidth=1.5)
    
    title_time = f" (0 to {time_horizon_cap} Days)" if time_horizon_cap is not None else ""
    ax.set_title(
        f"Cumulative incidence of 10% weight loss by {pub_strata_name}, \nwith dropout as a competing risk\n{sig_text}",
        fontsize=12, #fontweight="bold"
    )
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Cumulative incidence (accounting for dropout)", fontsize=11)
    ax.set_ylim(0, 1.05)
    
    # Regular 15-day intervals for clean table alignment
    if max_days <= 75:
        xticks = [0, 15, 30, 45, 60]
    else:
        xticks = [0, 30, 60, 90, 120, 150, 180]
        
    ax.set_xticks(xticks)
    ax.set_xlim(0, max_days)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title=pub_strata_name, loc="upper left", fontsize=9.5, frameon=True)
    
    # Add Numbers-at-Risk Table: number of active participants remaining at risk
    if not plot_dropout:
        add_clean_colored_at_risk_table(ax, [g0["cr_duration"], g1["cr_duration"]], [colors[0], colors[1]], xticks)
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.20)
    else:
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)
    
    os.makedirs(output_dir, exist_ok=True)
    cap_str = f"_{time_horizon_cap}d" if time_horizon_cap is not None else ""
    drop_str = "_with_dropout" if plot_dropout else ""
    svg_path = os.path.join(output_dir, f"cif_10perc_wl{cap_str}{drop_str}_by_{file_suffix}.svg")
    png_path = os.path.join(output_dir, f"cif_10perc_wl{cap_str}{drop_str}_by_{file_suffix}.png")
    
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        
    print(f"  - Saved Competing Risks CIF SVG: {svg_path}")
    print(f"  - Saved Competing Risks CIF PNG: {png_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)
    
    return {
        "strata": pub_strata_name,
        "time_horizon_cap": time_horizon_cap if time_horizon_cap is not None else "Full (180d)",
        "unadj_hazard_ratio": unadj_hr,
        "unadj_hr_ci": f"{sig_res['hr_lower']:.2f} - {sig_res['hr_upper']:.2f}",
        "unadj_p_value": unadj_p,
        "adj_hazard_ratio": adj_hr,
        "adj_hr_ci": f"{sig_res['adj_hr_lower']:.2f} - {sig_res['adj_hr_upper']:.2f}",
        "adj_p_value": adj_p,
        "adjustment_covariates": "Age, Sex, Baseline BMI",
        "dropout_curves_included": plot_dropout
    }


def compute_event_probabilities_table(df_surv, df_cr, time_horizon_cap=None):
    """
    Computes side-by-side comparison of naive Kaplan-Meier estimates versus
    Competing-Risk Cumulative Incidence estimates at milestone timepoints.
    
    Demonstrates censoring inflation (how ignoring dropout inflates success rates).
    """
    if time_horizon_cap is not None and time_horizon_cap <= 60:
        milestones = [15, 30, 45, time_horizon_cap]
    elif time_horizon_cap is not None and time_horizon_cap <= 80:
        milestones = [20, 40, 60, time_horizon_cap]
    else:
        milestones = [40, 60, 80, 120]
        
    records = []
    
    strata_targets = [
        ("emotional_eating_yn", "Emotional Eating (Yes/No)"),
        ("emotional_eating_median_split", "Emotional Eating Score (Median-Split)"),
        ("quantity_control_median_split", "Quantity Control Score (Median-Split)")
    ]
    
    for strata_col, strata_name in strata_targets:
        groups = sorted(df_surv[strata_col].dropna().unique())
        for g_val in groups:
            sub_km = df_surv[df_surv[strata_col] == g_val]
            sub_cr = df_cr[df_cr[strata_col] == g_val]
            
            kmf = KaplanMeierFitter().fit(sub_km["duration"], event_observed=sub_km["event"])
            
            ajf = AalenJohansenFitter(calculate_variance=False)
            ajf.fit(sub_cr["cr_duration"], sub_cr["cr_status"], event_of_interest=1)
            
            if strata_col.endswith("_yn"):
                label = "Yes" if g_val == 1 else "No"
            else:
                if "emotional_eating" in strata_col:
                    orig_col = "emotional_eating_value_likert"
                elif "quantity_control" in strata_col:
                    orig_col = "quantity_control_likert"
                else:
                    orig_col = strata_col.replace("_median_split", "")
                med_val = int(pd.to_numeric(df_surv[orig_col], errors="coerce").median())
                label = f">= Median ({med_val})" if g_val == 1 else f"< Median ({med_val})"
                
            for day in milestones:
                km_est = 1.0 - kmf.predict(day)
                cif_series = ajf.cumulative_density_["CIF_1"]
                cif_est = cif_series.loc[:day].iloc[-1] if (cif_series.index <= day).any() else 0.0
                
                records.append({
                    "Stratification": strata_name,
                    "Group": label,
                    "Timepoint (Days)": day,
                    "Naive KM Probability (%)": round(km_est * 100, 1),
                    "Competing Risk Probability (%)": round(cif_est * 100, 1),
                    "Difference (Inflation %)": round((km_est - cif_est) * 100, 1)
                })
                
    return pd.DataFrame(records)


# ==============================================================================
# 4. MASTER PIPELINE ORCHESTRATOR
# ==============================================================================

def run_survival_analysis_suite(input_db_path, output_db_path, plot_dir, time_horizon_cap=65, plot_dropout=False, show_plot=True):
    """
    Executes the complete revised survival analysis suite.
    
    Pipeline Stages:
    1. Dropout vs Completer Comparison (SMDs and Benjamini-Hochberg FDR q-values).
    2. Corrected Kaplan-Meier Curves (Properly right-censored, inverted to Cumulative Incidence, Numbers-at-Risk).
    3. Competing Risks Cumulative Incidence Analysis (Aalen-Johansen CIF, Unadjusted & Adjusted Cause-Specific Cox HRs).
    4. Side-by-Side Milestone Probability Table (Demonstrating Censoring Inflation).
    5. Saves full results to Excel and SQLite.
    
    Args:
        input_db_path (str): Path to SQLite database containing timetoevent_eb_compl.
        output_db_path (str): Path to SQLite database where output tables will be saved.
        plot_dir (str): Directory where plots and Excel results will be written.
        time_horizon_cap (int, optional): Cap for follow-up in days (e.g. 60, 65, 70, or None). Defaults to 65.
        plot_dropout (bool): Whether to plot competing dropout curves on CIF figures. Defaults to False.
        show_plot (bool): Whether to display plots interactively. Defaults to True.
        
    Returns:
        dict: Result DataFrames for tables and summaries.
    """
    os.makedirs(plot_dir, exist_ok=True)
    cap_suffix = f"_{time_horizon_cap}d" if time_horizon_cap is not None else ""
    excel_path = os.path.join(plot_dir, f"revised_survival_analysis_results{cap_suffix}.xlsx")
    
    print("=" * 80)
    print("STARTING REVISED SURVIVAL & ATTRITION ANALYSIS SUITE")
    print(f"Time Horizon Cap: {time_horizon_cap if time_horizon_cap is not None else 'Uncapped (Full 180 Days)'}")
    print(f"Plot Competing Dropout: {plot_dropout}")
    print("=" * 80)
    print(f"Input DB: {input_db_path}")
    print(f"Output Plots: {plot_dir}")
    
    with sqlite3.connect(input_db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM timetoevent_eb_compl", conn)
    print(f"Loaded full behavioral cohort: N = {len(df):,}")
    
    # 1. Part 1: Dropout vs. Completer Comparison
    dropout_comp_df = compare_60d_dropouts_vs_completers(df)
    
    # 2. Part 2: Corrected Kaplan-Meier Analysis
    print(f"\n--- Generating Corrected Kaplan-Meier Cumulative Incidence Curves (Cap: {time_horizon_cap}d) ---")
    df_surv, median_ee = prepare_survival_cohort(df, time_horizon_cap=time_horizon_cap)
    
    km_stat_yn = plot_corrected_kaplan_meier(
        df_surv, "emotional_eating_yn", "Emotional eating (yes/no)",
        plot_dir, time_horizon_cap=time_horizon_cap, show_plot=show_plot
    )
    km_stat_likert = plot_corrected_kaplan_meier(
        df_surv, "emotional_eating_median_split", "Emotional eating score (1-10)",
        plot_dir, time_horizon_cap=time_horizon_cap, show_plot=show_plot
    )
    km_stat_qc = plot_corrected_kaplan_meier(
        df_surv, "quantity_control_median_split", "Quantity control score (1-10)",
        plot_dir, time_horizon_cap=time_horizon_cap, show_plot=show_plot
    )
    km_summary_df = pd.DataFrame([km_stat_yn, km_stat_likert, km_stat_qc])
    
    # 3. Part 3: Competing Risks Analysis (Aalen-Johansen CIF with Unadjusted & Adjusted Cause-Specific Cox HR)
    print(f"\n--- Generating Competing Risks Cumulative Incidence Curves (Aalen-Johansen, Cap: {time_horizon_cap}d) ---")
    df_cr, _ = prepare_competing_risks_cohort(df, time_horizon_cap=time_horizon_cap)
    
    cif_stat_yn = plot_competing_risks_cif(
        df_cr, "emotional_eating_yn", "Emotional eating (yes/no)",
        plot_dir, time_horizon_cap=time_horizon_cap, plot_dropout=plot_dropout, show_plot=show_plot
    )
    cif_stat_likert = plot_competing_risks_cif(
        df_cr, "emotional_eating_median_split", "Emotional eating score (1-10)",
        plot_dir, time_horizon_cap=time_horizon_cap, plot_dropout=plot_dropout, show_plot=show_plot
    )
    cif_stat_qc = plot_competing_risks_cif(
        df_cr, "quantity_control_median_split", "Quantity control score (1-10)",
        plot_dir, time_horizon_cap=time_horizon_cap, plot_dropout=plot_dropout, show_plot=show_plot
    )
    cif_summary_df = pd.DataFrame([cif_stat_yn, cif_stat_likert, cif_stat_qc])
    
    prob_comp_df = compute_event_probabilities_table(df_surv, df_cr, time_horizon_cap=time_horizon_cap)
    
    # 4. Save Tables to Excel and SQLite (with file-lock safety)
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            dropout_comp_df.to_excel(writer, sheet_name="Dropout_60d_Comparison", index=False)
            km_summary_df.to_excel(writer, sheet_name="KM_LogRank_Summary", index=False)
            cif_summary_df.to_excel(writer, sheet_name="Competing_Risks_Significance", index=False)
            prob_comp_df.to_excel(writer, sheet_name="KM_vs_CompetingRisks_Probs", index=False)
        print(f"\nSaved Excel workbook: {excel_path}")
    except PermissionError:
        print(f"\nWarning: Could not overwrite {excel_path} because it is open in another program. Skipping Excel overwrite.")

    with sqlite3.connect(output_db_path) as conn_out:
        dropout_comp_df.to_sql("dropouts_60d_comparison", conn_out, if_exists="replace", index=False)
        prob_comp_df.to_sql(f"km_vs_competing_risks_probs{cap_suffix}", conn_out, if_exists="replace", index=False)
        km_summary_df.to_sql(f"km_logrank_summary{cap_suffix}", conn_out, if_exists="replace", index=False)
        cif_summary_df.to_sql(f"cif_competing_risks_summary{cap_suffix}", conn_out, if_exists="replace", index=False)
    print(f"Saved tables to SQLite: {output_db_path}")
    
    # 5. Print Diagnostics to Console
    print("\n" + "=" * 80)
    print(f"SURVIVAL ANALYSIS SUMMARY (Time Horizon Cap = {time_horizon_cap if time_horizon_cap is not None else 'Full'})")
    print("=" * 80)
    print("\n--- Kaplan-Meier Log-Rank Results (Properly Right-Censored, N=1,052) ---")
    print(km_summary_df.to_string(index=False))
    
    print("\n--- Competing Risks Significance Tests (Cause-Specific Cox Models) ---")
    print(cif_summary_df.to_string(index=False))
    
    print("\n--- Milestone Probability Comparison (Naive KM vs. Competing Risks) ---")
    print(prob_comp_df.to_string(index=False))
    
    return {
        "dropout_comparison": dropout_comp_df,
        "km_summary": km_summary_df,
        "competing_risks_summary": cif_summary_df,
        "probability_comparison": prob_comp_df
    }

if __name__ == "__main__":
    paper1_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_in = os.path.join(paper1_dir, "p1_dbs", "pnk_db2_p1_in_2025.sqlite")
    db_out = os.path.join(paper1_dir, "p1_dbs", "pnk_db2_p1_out_2025.sqlite")
    p_dir = os.path.join(paper1_dir, "p1_outputs", "survival_analysis")
    
    # Default execution: 65-day cap (+-10d observational window around 60d), clean CIF with Numbers-at-Risk
    run_survival_analysis_suite(
        input_db_path=db_in,
        output_db_path=db_out,
        plot_dir=p_dir,
        time_horizon_cap=65,
        plot_dropout=False,
        show_plot=False
    )
