"""
Revised Survival Analysis Module (Paper 1 Revision)

Scientific Purpose:
Addresses Reviewer Comment 2 regarding high attrition, informative censoring,
and 60-day weight loss target achievement:
1. Compares 60-day completers vs. 60-day dropouts on baseline traits, treatment engagement,
   and self-reported eating behaviors (with SMDs and Benjamini-Hochberg FDR q-values).
2. Corrects the Kaplan-Meier analysis:
   - Eliminates the selection distortion of dropping non-achievers.
   - Properly right-censors all 1,052 patients across the observational timeframe (0 to 180 days).
   - Inverts the curve to Cumulative Incidence (1 - S(t)), starting at 0% and rising upward.
   - Restores square aspect ratio, publication styling, and a clean Numbers-at-Risk table.
3. Implements Competing Risks Analysis (Aalen-Johansen Cumulative Incidence Function):
   - Formally accounts for early patient dropout as a competing risk to 10% weight loss.
   - Conducts formal inferential hypothesis testing via Cause-Specific Cox Hazard Models.
   - Displays Cause-Specific Hazard Ratio (HR) and p-value directly in plot titles.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    "emotional_eating_value_likert": "Emotional eating score (1-10)"
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
    ("height_m", "Height (m)", "continuous"),
    ("baseline_weight_kg", "Baseline weight (kg)", "continuous"),
    ("baseline_bmi", "Baseline BMI (kg/m²)", "continuous"),
    ("baseline_fat_%", "Baseline fat mass (%)", "continuous"),
    ("baseline_muscle_%", "Baseline muscle mass (%)", "continuous"),
    # Engagement & follow-up metrics
    ("delim_eng", "Treatment engagement and follow-up", "header"),
    ("total_followup_days", "Total follow-up length (days)", "continuous"),
    ("dietitian_visits", "Number of dietitian visits", "continuous"),
    ("nr_total_measurements", "Total number of measurements", "continuous"),
    ("avg_days_between_measurements", "Average measurement frequency (days)", "continuous"),
    # Eating behaviors
    ("delim_eb", "Self-reported eating behaviors", "header"),
    ("hunger_yn", "Hunger awareness (yes/no)", "categorical"),
    ("satiety_yn", "Satiety responsiveness (yes/no)", "categorical"),
    ("emotional_eating_yn", "Emotional eating (yes/no)", "categorical"),
    ("emotional_eating_value_likert", "Emotional eating score (1-10)", "continuous"),
    ("quantity_control_likert", "Quantity control score (1-10)", "continuous"),
    ("impulse_control_likert", "Impulse control score (1-10)", "continuous"),
    # Treatment outcomes
    ("delim_out", "Clinical weight loss outcomes", "header"),
    ("total_wl_%", "Total weight loss (%)", "continuous"),
    ("final_bmi", "Final BMI (kg/m²)", "continuous"),
    ("bmi_reduction", "BMI reduction (kg/m²)", "continuous"),
    ("10%_wl_achieved", "Achieved 10% weight loss (n)", "categorical"),
]

def compare_60d_dropouts_vs_completers(df):
    """
    Compares 60-day completers (60d_dropout == 0) versus 60-day dropouts (60d_dropout == 1)
    on baseline traits, engagement metrics, eating behaviors, and outcomes.
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

def prepare_survival_cohort(df):
    """
    Constructs proper survival duration and right-censoring indicators for all 1,052 patients.
    """
    df_surv = df.copy()
    achieved = (df_surv["10%_wl_achieved"] == 1)
    
    df_surv["event"] = np.where(achieved, 1, 0)
    df_surv["duration"] = np.where(
        achieved,
        df_surv["days_to_10%_wl"],
        df_surv["total_followup_days"]
    )
    df_surv["duration"] = pd.to_numeric(df_surv["duration"], errors="coerce").fillna(1).clip(lower=1)
    
    num_likert = pd.to_numeric(df_surv["emotional_eating_value_likert"], errors="coerce")
    median_ee = num_likert.median()
    df_surv["emotional_eating_median_split"] = (num_likert >= median_ee).astype(int)
    
    return df_surv, median_ee


def plot_corrected_kaplan_meier(df_surv, strata_col, strata_label, output_dir, max_days=180, show_plot=True):
    """
    Generates and saves a square, publication-grade Kaplan-Meier plot with proper right-censoring
    and a clean Numbers-at-Risk table.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    
    groups = sorted(df_surv[strata_col].dropna().unique())
    if len(groups) < 2:
        return
        
    g0 = df_surv[df_surv[strata_col] == groups[0]]
    g1 = df_surv[df_surv[strata_col] == groups[1]]
    
    # Calculate log-rank test p-value across all 1,052 patients
    lr_res = logrank_test(
        g0["duration"], g1["duration"],
        event_observed_A=g0["event"], event_observed_B=g1["event"]
    )
    p_val = lr_res.p_value
    p_text = f"Log-rank p={p_val:.3f}"
    
    # Clean publication labels matching the original script
    pub_strata_name = PUBLICATION_NAMES.get(strata_col, strata_label)
    if strata_col == "emotional_eating_yn":
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: No"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: Yes"
        file_suffix = "emotional_eating_yn"
    else:
        med_val = int(df_surv["emotional_eating_value_likert"].median())
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: < median ({med_val})"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: >= median ({med_val})"
        file_suffix = "emotional_eating_value_likert"
        
    kmf_0 = KaplanMeierFitter()
    kmf_0.fit(g0["duration"], event_observed=g0["event"], label=label_0)
    
    kmf_1 = KaplanMeierFitter()
    kmf_1.fit(g1["duration"], event_observed=g1["event"], label=label_1)
    
    # Create Figure with square proportions matching original
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {0: "blue", 1: "orange"}
    
    kmf_0.plot_cumulative_density(ax=ax, color=colors[0], linewidth=1.8, ci_alpha=0.12)
    kmf_1.plot_cumulative_density(ax=ax, color=colors[1], linewidth=1.8, ci_alpha=0.12)
    
    ax.set_title(f"Time to 10% weight loss by {pub_strata_name}\n{p_text}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Probability of weight loss target achievement", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title=pub_strata_name, loc="upper left", fontsize=10, frameon=True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_days)
    
    # Streamlined Numbers-at-Risk Table: show ONLY "At risk" counts
    add_at_risk_counts(kmf_0, kmf_1, ax=ax, fig=fig, rows_to_show=["At risk"])
    ax.set_xlim(0, max_days)
    
    fig.tight_layout(pad=1.5)
    
    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, f"km_10perc_wl_by_{file_suffix}.svg")
    png_path = os.path.join(output_dir, f"km_10perc_wl_by_{file_suffix}.png")
    
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  - Saved KM SVG: {svg_path}")
    print(f"  - Saved KM PNG: {png_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)
    
    return {
        "strata": pub_strata_name,
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

def prepare_competing_risks_cohort(df):
    """
    Constructs competing risks variables:
    - Status 1: 10% weight loss achieved (Target Event)
    - Status 2: Early dropout prior to 10% weight loss (Competing Event)
    - Status 0: Censored active
    """
    df_cr = df.copy()
    
    status = []
    duration = []
    
    for _, row in df_cr.iterrows():
        if row["10%_wl_achieved"] == 1:
            status.append(1)
            duration.append(max(1, row["days_to_10%_wl"]))
        elif row["instant_dropout"] == 1 or row["60d_dropout"] == 1 or row["80d_dropout"] == 1:
            status.append(2)
            duration.append(max(1, row["total_followup_days"]))
        else:
            status.append(0)
            duration.append(max(1, row["total_followup_days"]))
            
    df_cr["cr_status"] = status
    df_cr["cr_duration"] = duration
    
    num_likert = pd.to_numeric(df_cr["emotional_eating_value_likert"], errors="coerce")
    median_ee = num_likert.median()
    df_cr["emotional_eating_median_split"] = (num_likert >= median_ee).astype(int)
    
    return df_cr, median_ee


def test_competing_risks_significance(df_cr, strata_col):
    """
    Performs formal inferential hypothesis testing for the target event in competing risks
    using a Cause-Specific Cox Proportional Hazards Model.
    
    Returns:
        dict: Hazard Ratio, 95% CI, and p-value.
    """
    df_test = df_cr.copy()
    # In cause-specific survival, Event 1 is the event; Event 2 (competing dropout) is censored at its time
    df_test["cs_event"] = (df_test["cr_status"] == 1).astype(int)
    
    cph = CoxPHFitter()
    cph.fit(
        df_test[["cr_duration", "cs_event", strata_col]],
        duration_col="cr_duration",
        event_col="cs_event"
    )
    
    hr = cph.hazard_ratios_[strata_col]
    ci = cph.confidence_intervals_.loc[strata_col]
    hr_lower = np.exp(ci.iloc[0])
    hr_upper = np.exp(ci.iloc[1])
    p_val = cph.summary.loc[strata_col, "p"]
    
    return {
        "hazard_ratio": hr,
        "hr_lower": hr_lower,
        "hr_upper": hr_upper,
        "p_value": p_val
    }


def plot_competing_risks_cif(df_cr, strata_col, strata_label, output_dir, max_days=180, show_plot=True):
    """
    Plots the Cumulative Incidence Function (CIF) with elegant square styling,
    and displays the Cause-Specific Hazard Ratio and significance p-value in the title.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    
    groups = sorted(df_cr[strata_col].dropna().unique())
    g0 = df_cr[df_cr[strata_col] == groups[0]]
    g1 = df_cr[df_cr[strata_col] == groups[1]]
    
    pub_strata_name = PUBLICATION_NAMES.get(strata_col, strata_label)
    if strata_col == "emotional_eating_yn":
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: No"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: Yes"
        file_suffix = "emotional_eating_yn"
    else:
        med_val = int(df_cr["emotional_eating_value_likert"].median())
        label_0 = f"{pub_strata_name.split('(')[0].strip()}: < median ({med_val})"
        label_1 = f"{pub_strata_name.split('(')[0].strip()}: >= median ({med_val})"
        file_suffix = "emotional_eating_value_likert"
        
    # Statistical significance test for competing risks
    sig_res = test_competing_risks_significance(df_cr, strata_col)
    hr = sig_res["hazard_ratio"]
    p_val = sig_res["p_value"]
    sig_text = f"Cause-specific HR = {hr:.2f} (p = {p_val:.3f})"
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {0: "blue", 1: "orange"}
    
    ajf_0 = AalenJohansenFitter(calculate_variance=True)
    ajf_0.fit(g0["cr_duration"], g0["cr_status"], event_of_interest=1, label=label_0)
    ajf_0.plot(ax=ax, color=colors[0], linewidth=1.8)
    
    ajf_1 = AalenJohansenFitter(calculate_variance=True)
    ajf_1.fit(g1["cr_duration"], g1["cr_status"], event_of_interest=1, label=label_1)
    ajf_1.plot(ax=ax, color=colors[1], linewidth=1.8)
    
    ax.set_title(
        f"Cumulative Incidence of 10% Weight Loss (Competing Risks)\nby {pub_strata_name}\n{sig_text}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("Cumulative Incidence (Accounting for Dropout)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_days)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title=pub_strata_name, loc="upper left", fontsize=10, frameon=True)
    fig.tight_layout(pad=1.5)
    
    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, f"cif_10perc_wl_by_{file_suffix}.svg")
    png_path = os.path.join(output_dir, f"cif_10perc_wl_by_{file_suffix}.png")
    
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"  - Saved Competing Risks CIF SVG: {svg_path}")
    print(f"  - Saved Competing Risks CIF PNG: {png_path}")
    
    if show_plot:
        plt.show()
    plt.close(fig)
    
    return {
        "strata": pub_strata_name,
        "hazard_ratio": hr,
        "p_value": p_val,
        "hr_lower": sig_res["hr_lower"],
        "hr_upper": sig_res["hr_upper"]
    }


def compute_event_probabilities_table(df_surv, df_cr):
    """
    Computes side-by-side comparison of naive Kaplan-Meier estimates versus
    Competing-Risk Cumulative Incidence estimates at milestone timepoints (Day 40, 60, 80, 120).
    """
    milestones = [40, 60, 80, 120]
    records = []
    
    for strata_col, strata_name in [
        ("emotional_eating_yn", "Emotional Eating (Yes/No)"),
        ("emotional_eating_median_split", "Emotional Eating Score (Median-Split)")
    ]:
        groups = sorted(df_surv[strata_col].dropna().unique())
        for g_val in groups:
            sub_km = df_surv[df_surv[strata_col] == g_val]
            sub_cr = df_cr[df_cr[strata_col] == g_val]
            
            kmf = KaplanMeierFitter().fit(sub_km["duration"], event_observed=sub_km["event"])
            
            ajf = AalenJohansenFitter(calculate_variance=False)
            ajf.fit(sub_cr["cr_duration"], sub_cr["cr_status"], event_of_interest=1)
            
            label = "Yes" if g_val == 1 else "No"
            if "Median" in strata_name:
                med_val = int(df_surv["emotional_eating_value_likert"].median())
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

def run_survival_analysis_suite(input_db_path, output_db_path, plot_dir, show_plot=True):
    """
    Executes the complete revised survival analysis suite.
    """
    os.makedirs(plot_dir, exist_ok=True)
    excel_path = os.path.join(plot_dir, "revised_survival_analysis_results.xlsx")
    
    print("=" * 80)
    print("STARTING REVISED SURVIVAL & ATTRITION ANALYSIS SUITE")
    print("=" * 80)
    print(f"Input DB: {input_db_path}")
    print(f"Output Plots: {plot_dir}")
    
    with sqlite3.connect(input_db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM timetoevent_eb_compl", conn)
    print(f"Loaded full behavioral cohort: N = {len(df):,}")
    
    # 1. Part 1: Dropout vs. Completer Comparison
    dropout_comp_df = compare_60d_dropouts_vs_completers(df)
    
    # 2. Part 2: Corrected Kaplan-Meier Analysis (Square, 0 to 180 days, clean Numbers-at-Risk)
    print("\n--- Generating Corrected Kaplan-Meier Cumulative Incidence Curves ---")
    df_surv, median_ee = prepare_survival_cohort(df)
    
    km_stat_yn = plot_corrected_kaplan_meier(
        df_surv, "emotional_eating_yn", "Emotional eating (yes/no)",
        plot_dir, max_days=180, show_plot=show_plot
    )
    km_stat_likert = plot_corrected_kaplan_meier(
        df_surv, "emotional_eating_median_split", "Emotional eating score (1-10)",
        plot_dir, max_days=180, show_plot=show_plot
    )
    km_summary_df = pd.DataFrame([km_stat_yn, km_stat_likert])
    
    # 3. Part 3: Competing Risks Analysis (Square, with Cause-Specific HR and p-values)
    print("\n--- Generating Competing Risks Cumulative Incidence Curves (Aalen-Johansen) ---")
    df_cr, _ = prepare_competing_risks_cohort(df)
    
    cif_stat_yn = plot_competing_risks_cif(
        df_cr, "emotional_eating_yn", "Emotional eating (yes/no)",
        plot_dir, max_days=180, show_plot=show_plot
    )
    cif_stat_likert = plot_competing_risks_cif(
        df_cr, "emotional_eating_median_split", "Emotional eating score (1-10)",
        plot_dir, max_days=180, show_plot=show_plot
    )
    cif_summary_df = pd.DataFrame([cif_stat_yn, cif_stat_likert])
    
    prob_comp_df = compute_event_probabilities_table(df_surv, df_cr)
    
    # 4. Save Tables to Excel and SQLite
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        dropout_comp_df.to_excel(writer, sheet_name="Dropout_60d_Comparison", index=False)
        km_summary_df.to_excel(writer, sheet_name="KM_LogRank_Summary", index=False)
        cif_summary_df.to_excel(writer, sheet_name="Competing_Risks_Significance", index=False)
        prob_comp_df.to_excel(writer, sheet_name="KM_vs_CompetingRisks_Probs", index=False)
    print(f"\nSaved Excel workbook: {excel_path}")
    
    with sqlite3.connect(output_db_path) as conn_out:
        dropout_comp_df.to_sql("dropouts_60d_comparison", conn_out, if_exists="replace", index=False)
        prob_comp_df.to_sql("km_vs_competing_risks_probs", conn_out, if_exists="replace", index=False)
    print(f"Saved tables to SQLite: {output_db_path}")
    
    # 5. Print Diagnostics to Console
    print("\n" + "=" * 80)
    print("SURVIVAL ANALYSIS SUMMARY FOR MANUSCRIPT REVISION")
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
    p_dir = os.path.join(paper1_dir, "p1_outputs", "km_plots")
    run_survival_analysis_suite(db_in, db_out, p_dir, show_plot=False)
