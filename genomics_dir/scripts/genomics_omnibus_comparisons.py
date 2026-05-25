# genomics_omnibus_comparisons.py
#
# Unified engine for multi-group comparisons.
# Provides two independent entry points:
#   run_omnibus_table(cfg)  — Kruskal-Wallis / Chi² descriptive table saved to SQLite
#   run_omnibus_viz(cfg)    — Alluvial, KM, and violin plots saved as HTML
#
# Both entry points load their own data from SQL using only the columns they need.
# No shared state between them; call from separate notebook cells.
#
# Dependencies (must be importable):
#   fdr_correction_utils
#   sqlite3, pandas, numpy, scipy, lifelines, plotly, statsmodels

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OmnibusTableConfig:
    """Configuration for the multi-group descriptive comparison table.

    Parameters
    ----------
    paths              : paths_config object (paper_in_db, paper_out_db)
    cohort_tables      : {sql_table_name: group_code}  — same structure as everywhere else
    row_order          : [(col_name, pretty_label), ...]  — defines rows AND columns loaded
    output_table       : name of the table written to paper_out_db
    display            : {group_code: display_label}  — renames columns in the saved table
    fdr_correction     : apply Benjamini-Hochberg FDR correction to all p-values
    """
    paths:           object
    cohort_tables:   Dict[str, str]
    row_order:       List[Tuple[str, str]]
    output_table:    str
    display:         Dict[str, str]
    fdr_correction:  bool = True


@dataclass
class OmnibusVizConfig:
    """Configuration for the multi-group visualisation suite (alluvial, KM, violin).

    Parameters
    ----------
    paths                   : paths_config object
    cohort_tables           : {sql_table_name: group_code}
    group_colors            : {group_code: hex_color}
    master_group_order      : ordered list of all possible group_codes (active ones auto-detected)
    display                 : {group_code: display_label}
    output_dir              : folder where HTML files are written (created if absent)
    outputs                 : {'alluvial': filename, 'km': filename, 'violin': filename}
                              Omit a key to skip that plot.

    Landmark / outcome variables (drive column names derived automatically)
    -----------------------------------------------------------------------
    landmark_day            : N in Nd_dropout and Nd_wl_% (e.g. 120)
    wl_target               : T in T%_wl_achieved and days_to_T%_wl (e.g. 10)
    include_instant_dropout : if True, instant_dropout column is loaded and shown
                              as a separate slice in the alluvial adherence axis

    KM
    --
    km_time_col             : follow-up time column (default 'total_followup_days')
                              Used for dropout curve. WL time derived from days_to_T%_wl.

    Plot titles / subtitles (all optional — sensible defaults provided)
    -------------------------------------------------------------------
    alluvial_title, alluvial_subtitle
    km_title, km_subtitle
    violin_title, violin_subtitle
    """
    paths:                  object
    cohort_tables:          Dict[str, str]
    group_colors:           Dict[str, str]
    master_group_order:     List[str]
    display:                Dict[str, str]
    output_dir:             str
    outputs:                Dict[str, str]

    landmark_day:           int  = 120
    wl_target:              int  = 10
    include_instant_dropout: bool = True
    km_time_col:            str  = "total_followup_days"

    alluvial_title:    str = "Patient Flow: Personalization Timing, Adherence and Weight Loss"
    km_title:          str = "Kaplan-Meier Survival Curves by Personalization Group"
    violin_title:      str = ""   # auto-filled from landmark_day if blank
    alluvial_subtitle: str = "\nFirst medical records only"
    km_subtitle:       str = "\nFirst medical records only | Shaded area = 95% CI"
    violin_subtitle:   str = "\nViolin width ∝ % reaching landmark | Box = IQR"

    # ── Derived column name properties (read-only, computed from scalar fields) ──
    @property
    def dropout_col(self) -> str:
        return f"{self.landmark_day}d_dropout"

    @property
    def wl_pct_col(self) -> str:
        return f"{self.landmark_day}d_wl_%"

    @property
    def wl_achieved_col(self) -> str:
        return f"{self.wl_target}%_wl_achieved"

    @property
    def days_to_wl_col(self) -> str:
        return f"days_to_{self.wl_target}%_wl"

    @property
    def cols_alluvial(self) -> List[str]:
        cols = [self.dropout_col, self.wl_achieved_col]
        if self.include_instant_dropout:
            cols.insert(0, "instant_dropout")
        return cols

    @property
    def cols_km(self) -> List[str]:
        return [self.km_time_col, self.wl_achieved_col, self.days_to_wl_col]

    @property
    def cols_violin(self) -> List[str]:
        return [self.dropout_col, self.wl_pct_col]

# =========================
# HELPER FUNCTIONS
# =========================

def format_mean_sd(series):
    """Formats a series into 'mean ± SD' string, handling non-numeric data."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    mean = series.mean()
    sd = series.std()
    return f"{mean:.2f} \u00B1 {sd:.2f}"

def format_n_perc(series):
    """Formats a binary categorical series into 'N (%)' for the positive class (1)."""
    series = series.dropna()
    if series.empty:
        return "0 (0.0%)"
    n_positive = series.sum()
    n_total = len(series)
    perc = (n_positive / n_total) * 100 if n_total > 0 else 0
    return f"{int(n_positive)} ({perc:.1f}%)"

def format_median_iqr(series):
    """Formats a series into 'median [IQR]' string."""
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty:
        return "N/A"
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"{median:.2f} [{q1:.2f}–{q3:.2f}]"

def get_cause_cols(row_order: list) -> list:
    """Identifies weight gain cause columns from the ROW_ORDER config."""
    wgc_cols = []
    in_wgc_section = False
    for var, _ in row_order:
        if var == "delim_wgc":
            in_wgc_section = True
            continue
        if var.startswith("delim_") and in_wgc_section:
            break
        if in_wgc_section:
            wgc_cols.append(var)
    return wgc_cols

def get_variable_types(df, cause_cols):
    """Determines if a variable is continuous, categorical, or availability."""
    var_types = {}
    for col in df.columns:
        if (col in ["sex_f", "instant_dropout"] or col.startswith("country_") or col.endswith("_achieved") or col.endswith("_dropout") or col in cause_cols):
            var_types[col] = "categorical"
        elif col in ["patient_id", "medical_record_id", "medical_record_start_date"]:
            continue
        else:
            var_types[col] = "continuous"
    return var_types

def apply_fdr_correction(
    p_values: List[float], 
    method: str = 'fdr_bh',
    alpha: float = 0.05) -> List[float]:
    """
    Apply False Discovery Rate correction to a list of p-values using statsmodels.
    
    This function implements robust FDR correction with comprehensive error handling
    for various edge cases. It can be reused across different analysis contexts
    including descriptive comparisons, regression analyses, and subgroup analyses.
    
    Args:
        p_values (List[float]): List of p-values to correct. Can contain NaN values.
        method (str, optional): FDR correction method. Defaults to 'fdr_bh' 
                               (Benjamini-Hochberg).
        alpha (float, optional): Family-wise error rate. Defaults to 0.05.
    
    Returns:
        List[float]: FDR-corrected p-values. NaN values in input are preserved
                    as NaN in output. Returns original p-values if correction fails.
    
    Examples:
        >>> p_vals = [0.05, 0.01, 0.001, 0.2, np.nan]
        >>> corrected = apply_fdr_correction(p_vals)
        >>> # Returns corrected p-values with NaN preserved
        
        >>> # Use in regression analysis context
        >>> model_pvals = [model1.pvalues, model2.pvalues, model3.pvalues]
        >>> corrected_models = [apply_fdr_correction(pvals) for pvals in model_pvals]
    """
    try:
        # Convert to numpy array for easier handling
        p_array = np.array(p_values, dtype=float)
        
        # Handle edge case: empty input
        if len(p_array) == 0:
            return []
        
        # Identify valid (non-NaN) p-values
        valid_mask = ~np.isnan(p_array)
        valid_pvals = p_array[valid_mask]
        
        # Handle edge case: no valid p-values
        if len(valid_pvals) == 0:
            return p_values.copy() if isinstance(p_values, list) else p_array.tolist()
        
        # Handle edge case: single p-value
        if len(valid_pvals) == 1:
            return p_values.copy() if isinstance(p_values, list) else p_array.tolist()
        
        # Apply FDR correction using statsmodels
        rejected, corrected_pvals, alpha_sidak, alpha_bonf = multipletests(
            valid_pvals, 
            alpha=alpha, 
            method=method
        )
        
        # Create result array with original shape
        result = p_array.copy()
        result[valid_mask] = corrected_pvals
                
        return result.tolist()
        
    except Exception as e:
        return p_values.copy() if isinstance(p_values, list) else p_values

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_groups(paths, cohort_tables: Dict[str, str], cols_needed: List[str]) -> pd.DataFrame:
    """Load one or more cohort tables from SQLite, tag with group code."""
    dfs = []
    with sqlite3.connect(paths.paper_in_db) as conn:
        for table_name, group_code in cohort_tables.items():
            quoted = ", ".join(f"`{c}`" for c in cols_needed)
            q = f"SELECT {quoted} FROM `{table_name}`"
            df = pd.read_sql_query(q, conn)
            df["group"] = group_code
            dfs.append(df)
    if not dfs:
        raise ValueError("No data loaded — cohort_tables is empty or all queries failed.")
    return pd.concat(dfs, ignore_index=True)


def _active_order(df: pd.DataFrame, master_order: List[str]) -> List[str]:
    """Return only the group codes present in df, in master_order sequence."""
    present = set(df["group"].unique())
    return [g for g in master_order if g in present]


def _hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _format_p(p) -> str:
    if pd.isna(p):
        return "N/A"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT 1 — OMNIBUS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def run_omnibus_table(cfg: OmnibusTableConfig) -> pd.DataFrame:
    """Run multi-group Kruskal-Wallis / Chi² table and save to paper_out_db.

    Returns the results DataFrame (wide format, one row per variable).
    """
    # ── 1. Load data ──────────────────────────────────────────────────────────
    cols_to_load = [var for var, _ in cfg.row_order]
    print(f"Connecting to {cfg.paths.paper_in_db}")
    cohort_data: Dict[str, pd.DataFrame] = {}
    with sqlite3.connect(cfg.paths.paper_in_db) as conn:
        for table_name, group_code in cfg.cohort_tables.items():
            quoted = ", ".join(f"`{c}`" for c in cols_to_load)
            df = pd.read_sql_query(f"SELECT {quoted} FROM `{table_name}`", conn)
            cohort_data[table_name] = df
            print(f"  {group_code:30s} {len(df)} rows  ← {table_name}")

    # ── 2. Classify variable types ────────────────────────────────────────────
    cause_cols = get_cause_cols(cfg.row_order)
    dummy_df = pd.DataFrame(columns=cols_to_load)
    var_types = get_variable_types(dummy_df, cause_cols)

    # ── 3. Compute descriptives and run tests ─────────────────────────────────
    print(f"\nRunning tests across {len(cfg.cohort_tables)} cohorts...")
    rows_list = []

    # N row
    n_row = {"Variable": "N"}
    for table_name, group_code in cfg.cohort_tables.items():
        col_key = cfg.display.get(group_code, group_code)
        n_row[col_key] = str(len(cohort_data[table_name]))
    n_row["Statistic"] = np.nan
    n_row["raw_p_value"] = np.nan
    rows_list.append(n_row)

    for var, pretty_name in cfg.row_order:
        vtype = var_types.get(var, "continuous")
        row_dict = {"Variable": pretty_name}
        samples_for_test = []

        for table_name, group_code in cfg.cohort_tables.items():
            col_key = cfg.display.get(group_code, group_code)
            df = cohort_data[table_name]
            if var not in df.columns:
                row_dict[col_key] = "N/A"
                continue

            if vtype == "continuous":
                series = pd.to_numeric(df[var], errors="coerce").dropna()
                mean_sd  = format_mean_sd(series)
                med_iqr  = format_median_iqr(series)
                row_dict[col_key] = f"{mean_sd} [{med_iqr}]" if mean_sd != "N/A" else "N/A"
                samples_for_test.append(series.values)
            elif vtype == "categorical":
                series = df[var].dropna()
                row_dict[col_key] = format_n_perc(series)
                samples_for_test.append(series.values)

        valid_samples = [s for s in samples_for_test if len(s) > 0]
        stat, p_val = np.nan, np.nan
        if len(valid_samples) > 1:
            try:
                if vtype == "continuous":
                    stat, p_val = kruskal(*valid_samples)
                elif vtype == "categorical":
                    group_labels = []
                    for i, sample in enumerate(valid_samples):
                        group_labels.extend([i] * len(sample))
                    all_values = np.concatenate(valid_samples)
                    contingency = pd.crosstab(np.array(group_labels), all_values)
                    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                        stat, p_val, _, _ = chi2_contingency(contingency)
            except Exception:
                pass

        row_dict["Statistic"] = stat
        row_dict["raw_p_value"] = p_val
        rows_list.append(row_dict)

    results_df = pd.DataFrame(rows_list)

    # ── 4. FDR correction ─────────────────────────────────────────────────────
    if cfg.fdr_correction:
        print("Applying Benjamini-Hochberg FDR correction...")
        mask = results_df["raw_p_value"].notna()
        valid_pvals = results_df.loc[mask, "raw_p_value"].tolist()
        if valid_pvals:
            corrected = apply_fdr_correction(valid_pvals)
            results_df.loc[mask, "fdr_corrected_p_value"] = corrected

    # ── 5. Save ───────────────────────────────────────────────────────────────
    with sqlite3.connect(cfg.paths.paper_out_db) as conn_out:
        results_df.to_sql(cfg.output_table, conn_out, if_exists="replace", index=False)
    print(f"\nSaved → {cfg.paths.paper_out_db} :: {cfg.output_table}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT 2 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_omnibus_viz(cfg: OmnibusVizConfig) -> Dict:
    """Run whichever visualisations are listed in cfg.outputs.

    Returns a dict of {plot_key: (fig, output_path)}.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    if "alluvial" in cfg.outputs:
        results["alluvial"] = _make_alluvial(cfg)
    if "km" in cfg.outputs:
        results["km"] = _make_km(cfg)
    if "violin" in cfg.outputs:
        results["violin"] = _make_violin(cfg)
    return results


# ── Alluvial ──────────────────────────────────────────────────────────────────

def _make_alluvial(cfg: OmnibusVizConfig):
    df = _load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_alluvial)
    active = _active_order(df, cfg.master_group_order)

    # Adherence axis: categories derived from landmark_day
    # Order: instant_dropout (optional) → Nd_dropout → reached landmark
    dropout_label  = f"dropout_{cfg.landmark_day}d"
    reached_label  = f"reached_{cfg.landmark_day}d"

    def _classify_adherence(row):
        if cfg.include_instant_dropout and row.get("instant_dropout", 0) == 1:
            return "instant_dropout"
        if row[cfg.dropout_col] == 1:
            return dropout_label
        return reached_label

    df["adherence"] = df.apply(_classify_adherence, axis=1)

    # WL outcome axis
    achieved_label    = f"achieved_{cfg.wl_target}pct_wl"
    not_achieved_label = f"not_achieved_{cfg.wl_target}pct_wl"

    df["wl_outcome"] = df[cfg.wl_achieved_col].apply(
        lambda v: achieved_label if v == 1 else not_achieved_label
    )

    # Build display labels for the auto-generated category keys
    # (so the user does not need to pre-populate DISPLAY for derived labels)
    auto_display = {
        dropout_label:      f"{cfg.landmark_day}-day dropout",
        reached_label:      f"Reached {cfg.landmark_day} days",
        achieved_label:     f"Achieved ≥{cfg.wl_target}% WL",
        not_achieved_label: f"Did NOT achieve ≥{cfg.wl_target}% WL",
        "instant_dropout":  "Instant dropout",
    }
    disp = {**auto_display, **cfg.display}   # notebook DISPLAY overrides auto-labels

    # Only include categories that actually appear in data
    adherence_order_candidates = ["instant_dropout", dropout_label, reached_label]
    wl_order_candidates        = [not_achieved_label, achieved_label]
    adherence_order = [c for c in adherence_order_candidates if c in df["adherence"].values]
    wl_order        = [c for c in wl_order_candidates        if c in df["wl_outcome"].values]

    df["group"]     = pd.Categorical(df["group"],     categories=active,          ordered=True)
    df["adherence"] = pd.Categorical(df["adherence"], categories=adherence_order, ordered=True)
    df["wl_outcome"]= pd.Categorical(df["wl_outcome"],categories=wl_order,        ordered=True)

    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            go.parcats.Dimension(
                values=df["group"].map(disp),
                label="Group",
                categoryorder="array",
                categoryarray=[disp[c] for c in active],
            ),
            go.parcats.Dimension(
                values=df["adherence"].map(disp),
                label="Adherence",
                categoryorder="array",
                categoryarray=[disp[c] for c in adherence_order],
            ),
            go.parcats.Dimension(
                values=df["wl_outcome"].map(disp),
                label="Weight Loss Outcome",
                categoryorder="array",
                categoryarray=[disp[c] for c in wl_order],
            ),
        ],
        line=dict(color=df["group"].map(cfg.group_colors).tolist(), shape="hspline"),
        labelfont=dict(size=17, family="Arial"),
        tickfont= dict(size=15, family="Arial"),
        arrangement="freeform",
        bundlecolors=True,
        hoverinfo="count+probability",
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    )])
    fig.update_layout(
        title=dict(
            text=f"{cfg.alluvial_title}{cfg.alluvial_subtitle}",
            x=0.5, font=dict(size=16, family="Arial"),
        ),
        font=dict(size=14, family="Arial"),
        height=750, width=1300,
        margin=dict(l=120, r=200, t=120, b=40),
        paper_bgcolor="white",
    )
    out = Path(cfg.output_dir) / cfg.outputs["alluvial"]
    fig.write_html(str(out))
    print(f"  Alluvial saved → {out}")
    return fig, out


# ── KM ────────────────────────────────────────────────────────────────────────

def _km_traces(df, time_col, event_col, group_col, group_order, colors, display, invert=False):
    traces = []
    all_T, all_E, all_G = [], [], []
    for group_code in group_order:
        sub = df[df[group_col] == group_code]
        if sub.empty:
            continue
        T = sub[time_col].values
        E = sub[event_col].values
        all_T.append(T); all_E.append(E)
        all_G.append(np.full(len(T), group_code))
        kmf = KaplanMeierFitter().fit(T, event_observed=E)
        t_vals  = kmf.survival_function_.index.values
        s_vals  = kmf.survival_function_["KM_estimate"].values
        ci_low  = kmf.confidence_interval_["KM_estimate_lower_0.95"].values
        ci_high = kmf.confidence_interval_["KM_estimate_upper_0.95"].values
        if invert:
            s_vals = 1 - s_vals
            ci_low, ci_high = 1 - ci_high, 1 - ci_low
        color_hex = colors[group_code]
        traces.append(go.Scatter(
            x=np.concatenate([t_vals, t_vals[::-1]]),
            y=np.concatenate([ci_high, ci_low[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(color_hex, 0.15),
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=t_vals, y=s_vals, mode="lines",
            line=dict(color=color_hex, width=2.5, shape="hv"),
            name=f"{display[group_code]} (n={len(T)})",
        ))
    lr = multivariate_logrank_test(
        np.concatenate(all_T), np.concatenate(all_G), np.concatenate(all_E)
    )
    return traces, lr.p_value


def _make_km(cfg: OmnibusVizConfig):
    df = _load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_km)
    active = _active_order(df, cfg.master_group_order)

    # Dropout curve: time = total_followup_days, event = 1 for everyone
    # (everyone eventually dropped out or reached data cutoff; no censoring flag needed
    #  unless you add one explicitly — pass cfg.km_censor_col if that ever changes)
    df["_dropout_time"]  = df[cfg.km_time_col]
    df["_dropout_event"] = 1   # all observations are complete follow-up endpoints

    # WL achievement curve
    df["_wl_event"] = df[cfg.wl_achieved_col].fillna(0).astype(int)
    df["_wl_time"]  = np.where(
        df["_wl_event"] == 1,
        df[cfg.days_to_wl_col],
        df[cfg.km_time_col],
    )

    df_dropout = df.dropna(subset=["_dropout_time"]).copy()
    df_wl      = df.dropna(subset=["_wl_time"]).copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Time to Dropout", f"Time to ≥{cfg.wl_target}% Weight Loss"],
        horizontal_spacing=0.12,
    )

    tr_do, p_do = _km_traces(
        df_dropout, "_dropout_time", "_dropout_event",
        "group", active, cfg.group_colors, cfg.display, invert=False,
    )
    for tr in tr_do:
        fig.add_trace(tr, row=1, col=1)
    fig.add_annotation(
        text=f"Log-rank: {_format_p(p_do)}",
        xref="x domain", yref="y domain", x=0.97, y=0.97,
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(240,240,240,0.85)", bordercolor="gray", borderwidth=1,
        row=1, col=1,
    )

    tr_wl, p_wl = _km_traces(
        df_wl, "_wl_time", "_wl_event",
        "group", active, cfg.group_colors, cfg.display, invert=True,
    )
    for tr in tr_wl:
        tr.showlegend = False
        fig.add_trace(tr, row=1, col=2)
    fig.add_annotation(
        text=f"Log-rank: {_format_p(p_wl)}",
        xref="x2 domain", yref="y2 domain", x=0.03, y=0.97,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(240,240,240,0.85)", bordercolor="gray", borderwidth=1,
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Days from baseline", row=1, col=1)
    fig.update_xaxes(title_text="Days from baseline", row=1, col=2)
    fig.update_yaxes(title_text="Probability of remaining enrolled", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text=f"Cumulative probability of ≥{cfg.wl_target}% WL", range=[0, 1.05], row=1, col=2)
    fig.update_layout(
        title=dict(text=f"{cfg.km_title}{cfg.km_subtitle}", x=0.5, font=dict(size=16, family="Arial")),
        font=dict(size=13, family="Arial"),
        height=550, width=1300,
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(
            title="Group", x=1.01, y=0.5, xanchor="left",
            font=dict(size=12, family="Arial"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="lightgray", borderwidth=1,
        ),
        margin=dict(l=70, r=160, t=100, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False)
    out = Path(cfg.output_dir) / cfg.outputs["km"]
    fig.write_html(str(out))
    print(f"  KM saved → {out}")
    return fig, out


# ── Violin ────────────────────────────────────────────────────────────────────

# def _make_violin(cfg: OmnibusVizConfig):
#     df = _load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_violin)
#     active = _active_order(df, cfg.master_group_order)

#     violin_title = cfg.violin_title or f"Weight Loss at {cfg.landmark_day} Days by Personalization Group"

#     group_stats = {}
#     for group_code in active:
#         sub     = df[df["group"] == group_code]
#         n_total = len(sub)
#         # Eligibility: patients who reached the landmark (dropout flag == 0) AND have a valid measurement
#         eligible = sub[sub[cfg.dropout_col] == 0][cfg.wl_pct_col].dropna()
#         group_stats[group_code] = dict(
#             values    = eligible.values,
#             n_total   = n_total,
#             n_reached = len(eligible),
#             pct_reached = len(eligible) / n_total if n_total else 0,
#             median = eligible.median()             if len(eligible) else np.nan,
#             q1     = eligible.quantile(0.25)       if len(eligible) else np.nan,
#             q3     = eligible.quantile(0.75)       if len(eligible) else np.nan,
#         )

#     # Global KW test
#     kw_groups = [group_stats[g]["values"] for g in active if len(group_stats[g]["values"]) >= 5]
#     kw_p = kruskal(*kw_groups).pvalue if len(kw_groups) >= 2 else np.nan

#     # Pairwise MWU with FDR correction
#     pairs = list(combinations(active, 2))
#     raw_pvals = []
#     for g1, g2 in pairs:
#         v1, v2 = group_stats[g1]["values"], group_stats[g2]["values"]
#         raw_pvals.append(
#             mannwhitneyu(v1, v2, alternative="two-sided").pvalue
#             if len(v1) >= 5 and len(v2) >= 5 else np.nan
#         )
#     valid = [p for p in raw_pvals if not np.isnan(p)]
#     corrected = multipletests(valid, alpha=0.05, method="fdr_bh")[1] if len(valid) else []
#     corr_iter = iter(corrected)
#     pair_results = {}
#     for (g1, g2), raw_p in zip(pairs, raw_pvals):
#         if np.isnan(raw_p):
#             pair_results[(g1, g2)] = {"sig": False, "corrected": np.nan}
#         else:
#             cp = next(corr_iter)
#             pair_results[(g1, g2)] = {"sig": cp < 0.05, "corrected": cp}

#     fig = go.Figure()
#     max_pct   = max(s["pct_reached"] for s in group_stats.values()) if group_stats else 1
#     MAX_HW    = 0.35

#     for x_pos, group_code in enumerate(active):
#         stats  = group_stats[group_code]
#         vals   = stats["values"]
#         color  = cfg.group_colors[group_code]
#         label  = cfg.display[group_code]
#         half_w = (stats["pct_reached"] / max_pct) * MAX_HW if max_pct else MAX_HW

#         if len(vals) < 5:
#             fig.add_trace(go.Scatter(
#                 x=[x_pos], y=[stats["median"]], mode="markers",
#                 marker=dict(color=color, size=10, symbol="diamond"),
#                 name=label, showlegend=True,
#             ))
#             continue

#         y_min, y_max = vals.min() - 1, vals.max() + 1
#         y_grid  = np.linspace(y_min, y_max, 300)
#         kde     = gaussian_kde(vals, bw_method="scott")
#         density = kde(y_grid)
#         density_scaled = density / density.max() * half_w
#         x_right  = x_pos + density_scaled
#         x_left   = x_pos - density_scaled[::-1]
#         x_outline = np.concatenate([x_right, x_left, [x_right[0]]])
#         y_outline = np.concatenate([y_grid,  y_grid[::-1], [y_grid[0]]])

#         fig.add_trace(go.Scatter(
#             x=x_outline, y=y_outline, fill="toself",
#             fillcolor=_hex_to_rgba(color, 0.55),
#             line=dict(color=color, width=1.5),
#             name=label, legendgroup=group_code, showlegend=True, hoverinfo="skip",
#         ))

#         med = stats["median"]
#         med_d = kde([med])[0] / density.max() * half_w
#         fig.add_trace(go.Scatter(
#             x=[x_pos - med_d, x_pos + med_d], y=[med, med], mode="lines",
#             line=dict(color="white", width=3),
#             legendgroup=group_code, showlegend=False, hoverinfo="skip",
#         ))

#         q1, q3 = stats["q1"], stats["q3"]
#         iqr_d  = kde(np.array([q1, q3])) / density.max() * half_w
#         box_hw = min(iqr_d) * 0.6
#         fig.add_trace(go.Scatter(
#             x=[x_pos-box_hw, x_pos+box_hw, x_pos+box_hw, x_pos-box_hw, x_pos-box_hw],
#             y=[q1, q1, q3, q3, q1],
#             fill="toself", fillcolor=_hex_to_rgba(color, 0.85),
#             line=dict(color="white", width=1),
#             legendgroup=group_code, showlegend=False, hoverinfo="skip",
#         ))

#         fig.add_annotation(
#             x=x_pos, y=med,
#             text=f"{med:.1f}%", showarrow=False,
#             font=dict(size=11, family="Arial", color="white"),
#         )
#         fig.add_annotation(
#             x=x_pos, y=y_min - 1.5,
#             text=f"{stats['pct_reached']*100:.0f}% reached {cfg.landmark_day}d<br>n={stats['n_reached']}/{stats['n_total']}",
#             showarrow=False,
#             font=dict(size=11, family="Arial", color=color),
#             yref="y",
#         )

#     all_vals = np.concatenate([s["values"] for s in group_stats.values() if len(s["values"]) > 0])
#     y_max_data   = np.percentile(all_vals, 99) if len(all_vals) else 1
#     bracket_step = abs(y_max_data) * 0.08 if y_max_data else 1
#     kw_color     = "#CC0000" if (not np.isnan(kw_p) and kw_p < 0.05) else "#888888"

#     fig.add_annotation(
#         x=0.5, y=1.06, xref="paper", yref="paper",
#         text=f"Kruskal-Wallis: {_format_p(kw_p)}", showarrow=False,
#         font=dict(size=12, family="Arial", color=kw_color), align="center",
#     )

#     bracket_y = y_max_data + bracket_step
#     for g1, g2 in [k for k, v in pair_results.items() if v["sig"]]:
#         x1, x2 = active.index(g1), active.index(g2)
#         p_str   = _format_p(pair_results[(g1, g2)]["corrected"])
#         fig.add_shape(type="line", x0=x1, x1=x2, y0=bracket_y, y1=bracket_y,
#                       line=dict(color="#444444", width=1.5))
#         fig.add_shape(type="line", x0=x1, x1=x1,
#                       y0=bracket_y - bracket_step * 0.2, y1=bracket_y,
#                       line=dict(color="#444444", width=1.5))
#         fig.add_shape(type="line", x0=x2, x1=x2,
#                       y0=bracket_y - bracket_step * 0.2, y1=bracket_y,
#                       line=dict(color="#444444", width=1.5))
#         fig.add_annotation(
#             x=(x1+x2)/2, y=bracket_y + bracket_step * 0.15,
#             text=p_str, showarrow=False,
#             font=dict(size=10, family="Arial", color="#444444"),
#         )
#         bracket_y += bracket_step * 1.1

#     fig.update_yaxes(range=[all_vals.min() - 3, bracket_y + bracket_step] if len(all_vals) else [0, 1])
#     fig.update_layout(
#         title=dict(text=f"{violin_title}{cfg.violin_subtitle}", x=0.5, font=dict(size=16, family="Arial")),
#         xaxis=dict(
#             tickvals=list(range(len(active))),
#             ticktext=[cfg.display[g] for g in active],
#             tickfont=dict(size=13, family="Arial"),
#             showgrid=False, zeroline=False,
#         ),
#         yaxis=dict(
#             title=f"Weight loss at {cfg.landmark_day} days (%)",
#             title_font=dict(size=13, family="Arial"),
#             tickfont=dict(size=12),
#             showgrid=True, gridcolor="rgba(200,200,200,0.4)",
#             zeroline=True, zerolinecolor="rgba(150,150,150,0.5)", zerolinewidth=1,
#         ),
#         plot_bgcolor="white", paper_bgcolor="white",
#         font=dict(family="Arial"),
#         height=650, width=1000,
#         showlegend=False,
#         margin=dict(l=70, r=40, t=100, b=100),
#     )
#     out = Path(cfg.output_dir) / cfg.outputs["violin"]
#     fig.write_html(str(out))
#     print(f"  Violin saved → {out}")
#     return fig, out

def _make_violin(cfg: OmnibusVizConfig):
    df = _load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_violin)
    active = _active_order(df, cfg.master_group_order)

    violin_title = cfg.violin_title or f"Weight Loss at {cfg.landmark_day} Days by Personalization Group"

    group_stats = {}
    for group_code in active:
        sub = df[df["group"] == group_code]
        n_total = len(sub)

        # Eligibility: reached landmark and has valid WL measurement
        eligible = sub[sub[cfg.dropout_col] == 0][cfg.wl_pct_col].dropna()

        group_stats[group_code] = dict(
            values=eligible.values,
            n_total=n_total,
            n_reached=len(eligible),
            pct_reached=len(eligible) / n_total if n_total else 0,
            mean=eligible.mean() if len(eligible) else np.nan,
            median=eligible.median() if len(eligible) else np.nan,
            q1=eligible.quantile(0.25) if len(eligible) else np.nan,
            q3=eligible.quantile(0.75) if len(eligible) else np.nan,
        )

    # Omnibus test for weight loss
    kw_groups = [group_stats[g]["values"] for g in active if len(group_stats[g]["values"]) >= 5]
    wl_p = kruskal(*kw_groups).pvalue if len(kw_groups) >= 2 else np.nan

    # Omnibus test for adherence (% reaching landmark)
    # rows = groups, cols = [reached, not_reached]
    adherence_table = []
    for g in active:
        reached = group_stats[g]["n_reached"]
        not_reached = group_stats[g]["n_total"] - reached
        adherence_table.append([reached, not_reached])

    adherence_table = np.array(adherence_table)
    adh_p = np.nan
    if adherence_table.shape[0] > 1 and adherence_table.shape[1] == 2:
        try:
            _, adh_p, _, _ = chi2_contingency(adherence_table)
        except Exception:
            adh_p = np.nan

    # Pairwise MWU with FDR correction (unchanged)
    pairs = list(combinations(active, 2))
    raw_pvals = []
    for g1, g2 in pairs:
        v1, v2 = group_stats[g1]["values"], group_stats[g2]["values"]
        raw_pvals.append(
            mannwhitneyu(v1, v2, alternative="two-sided").pvalue
            if len(v1) >= 5 and len(v2) >= 5 else np.nan
        )

    valid = [p for p in raw_pvals if not np.isnan(p)]
    corrected = multipletests(valid, alpha=0.05, method="fdr_bh")[1] if len(valid) else []
    corr_iter = iter(corrected)
    pair_results = {}
    for (g1, g2), raw_p in zip(pairs, raw_pvals):
        if np.isnan(raw_p):
            pair_results[(g1, g2)] = {"sig": False, "corrected": np.nan}
        else:
            cp = next(corr_iter)
            pair_results[(g1, g2)] = {"sig": cp < 0.05, "corrected": cp}

    fig = go.Figure()
    max_pct = max(s["pct_reached"] for s in group_stats.values()) if group_stats else 1
    MAX_HW = 0.35

    for x_pos, group_code in enumerate(active):
        stats = group_stats[group_code]
        vals = stats["values"]
        color = cfg.group_colors[group_code]
        label = cfg.display[group_code]
        half_w = (stats["pct_reached"] / max_pct) * MAX_HW if max_pct else MAX_HW

        if len(vals) < 5:
            fig.add_trace(go.Scatter(
                x=[x_pos], y=[stats["mean"]], mode="markers",
                marker=dict(color=color, size=10, symbol="diamond"),
                name=label, showlegend=True,
            ))
            continue

        y_min, y_max = vals.min() - 1, vals.max() + 1
        y_grid = np.linspace(y_min, y_max, 300)
        kde = gaussian_kde(vals, bw_method="scott")
        density = kde(y_grid)
        density_scaled = density / density.max() * half_w
        x_right = x_pos + density_scaled
        x_left = x_pos - density_scaled[::-1]
        x_outline = np.concatenate([x_right, x_left, [x_right[0]]])
        y_outline = np.concatenate([y_grid, y_grid[::-1], [y_grid[0]]])

        # Violin
        fig.add_trace(go.Scatter(
            x=x_outline, y=y_outline, fill="toself",
            fillcolor=_hex_to_rgba(color, 0.55),
            line=dict(color=color, width=1.5),
            name=label, legendgroup=group_code, showlegend=True, hoverinfo="skip",
        ))

        # Median line kept in the background, as in original
        med = stats["median"]
        med_d = kde([med])[0] / density.max() * half_w
        fig.add_trace(go.Scatter(
            x=[x_pos - med_d, x_pos + med_d], y=[med, med], mode="lines",
            line=dict(color="white", width=3),
            legendgroup=group_code, showlegend=False, hoverinfo="skip",
        ))

        # IQR box
        q1, q3 = stats["q1"], stats["q3"]
        iqr_d = kde(np.array([q1, q3])) / density.max() * half_w
        box_hw = min(iqr_d) * 0.6
        fig.add_trace(go.Scatter(
            x=[x_pos-box_hw, x_pos+box_hw, x_pos+box_hw, x_pos-box_hw, x_pos-box_hw],
            y=[q1, q1, q3, q3, q1],
            fill="toself", fillcolor=_hex_to_rgba(color, 0.85),
            line=dict(color="white", width=1),
            legendgroup=group_code, showlegend=False, hoverinfo="skip",
        ))

        # Mean label displayed, not median
        mean_val = stats["mean"]
        fig.add_annotation(
            x=x_pos, y=mean_val,
            text=f"{mean_val:.1f}%", showarrow=False,
            font=dict(size=11, family="Arial", color="white"),
        )

        # Keep original bottom adherence labels
        fig.add_annotation(
            x=x_pos, y=y_min - 1.5,
            text=f"{stats['pct_reached']*100:.0f}% reached {cfg.landmark_day}d<br>n={stats['n_reached']}/{stats['n_total']}",
            showarrow=False,
            font=dict(size=11, family="Arial", color=color),
            yref="y",
        )

    all_vals = np.concatenate([s["values"] for s in group_stats.values() if len(s["values"]) > 0])
    y_max_data = np.percentile(all_vals, 99) if len(all_vals) else 1
    bracket_step = abs(y_max_data) * 0.08 if y_max_data else 1

    # Two elegant boxed omnibus p-values, same visual style as KM
    fig.add_annotation(
        text=f"{cfg.landmark_day}-day weight loss: {_format_p(wl_p)}",
        xref="paper", yref="paper", x=0.36, y=0.97,
        xanchor="center", yanchor="top", showarrow=False,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(240,240,240,0.85)",
        bordercolor="gray", borderwidth=1,
    )
    fig.add_annotation(
        text=f"{cfg.landmark_day}-day adherence: {_format_p(adh_p)}",
        xref="paper", yref="paper", x=0.64, y=0.97,
        xanchor="center", yanchor="top", showarrow=False,
        font=dict(size=12, family="Arial"),
        bgcolor="rgba(240,240,240,0.85)",
        bordercolor="gray", borderwidth=1,
    )

    # Pairwise significant brackets unchanged
    bracket_y = y_max_data + bracket_step
    for g1, g2 in [k for k, v in pair_results.items() if v["sig"]]:
        x1, x2 = active.index(g1), active.index(g2)
        p_str = _format_p(pair_results[(g1, g2)]["corrected"])
        fig.add_shape(
            type="line", x0=x1, x1=x2, y0=bracket_y, y1=bracket_y,
            line=dict(color="#444444", width=1.5)
        )
        fig.add_shape(
            type="line", x0=x1, x1=x1,
            y0=bracket_y - bracket_step * 0.2, y1=bracket_y,
            line=dict(color="#444444", width=1.5)
        )
        fig.add_shape(
            type="line", x0=x2, x1=x2,
            y0=bracket_y - bracket_step * 0.2, y1=bracket_y,
            line=dict(color="#444444", width=1.5)
        )
        fig.add_annotation(
            x=(x1 + x2) / 2, y=bracket_y + bracket_step * 0.15,
            text=p_str, showarrow=False,
            font=dict(size=10, family="Arial", color="#444444"),
        )
        bracket_y += bracket_step * 1.1

    fig.update_yaxes(range=[all_vals.min() - 3, bracket_y + bracket_step] if len(all_vals) else [0, 1])

    fig.update_layout(
        title=dict(
            text=f"{violin_title}{cfg.violin_subtitle}",
            x=0.5,
            font=dict(size=16, family="Arial")
        ),
        xaxis=dict(
            tickvals=list(range(len(active))),
            ticktext=[cfg.display[g] for g in active],
            tickfont=dict(size=13, family="Arial"),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=f"Weight loss at {cfg.landmark_day} days (%)",
            title_font=dict(size=13, family="Arial"),
            tickfont=dict(size=12),
            showgrid=True, gridcolor="rgba(200,200,200,0.4)",
            zeroline=True, zerolinecolor="rgba(150,150,150,0.5)", zerolinewidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial"),
        height=650,
        width=1000,
        showlegend=False,
        margin=dict(l=70, r=40, t=100, b=100),
    )

    out = Path(cfg.output_dir) / cfg.outputs["violin"]
    fig.write_html(str(out))
    print(f"  Violin saved → {out}")
    return fig, out