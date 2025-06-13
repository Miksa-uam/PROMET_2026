# ====IMPORTS====

import os
import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

# ====GENERAL PREPROCESSING====

def load_data(db_path, table_name):
    """Load data from SQLite table."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Preprocessing function for outcome and adherence predictions
def preprocess_for_regression(df, scenarios):
    """Prepare data: ensure predictors/covariates are numeric, booleans are 0/1."""
    df = df.copy()
    needed_cols = set()
    for s in scenarios:
        needed_cols.update(s['predictors'])
        needed_cols.update(s['covariates'])
    for col in needed_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Preprocessing functions for EB predictions by WGC
def preprocess_for_eb_by_wgc_linreg(df, predictors):
    df = df.copy()
    for col in predictors:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
def preprocess_for_eb_by_wgc_logreg(df, predictors, outcome):
    df = df.copy()
    for col in predictors + [outcome]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).astype(float)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ====LOGISTIC OUTCOME PREDICTIONS - INFERENCE====

def generate_logreg_scenarios(
    outcome_types,
    time_windows,
    target_percentages,
    main_predictors,
    adjustment_sets,
    input_table=None,
    df_all=None
):
    """
    Generates a list of scenario dictionaries for logistic regression analyses.
    If input_table == "timetoevent_eb_wgc_compl", includes all weight gain cause columns as predictors.
    """
    scenarios = []
    # Add weight gain cause columns if needed
    wgc_predictors = []
    if input_table in ("timetoevent_eb_wgc_compl", "timetoevent_wgc_compl") and df_all is not None:
        cols = list(df_all.columns)
        start = cols.index('weight_gain_cause_en') + 1
        end = cols.index('genomics_sample_id')
        wgc_predictors = cols[start:end]
        main_predictors = list(main_predictors) + wgc_predictors

    for outcome_type in outcome_types:
        for tw in time_windows:
            if outcome_type == 'target_wl' and tw == 'instant':
                continue
            if outcome_type == 'dropout' and tw == 'total':
                continue
            target_percs = target_percentages if outcome_type == 'target_wl' else [None]
            for target_perc in target_percs:
                for predictor in main_predictors:
                    for covariates, adj_label in adjustment_sets:
                        name_parts = []
                        time_label = f"{tw}d" if isinstance(tw, int) else tw
                        if outcome_type == 'target_wl':
                            name_parts.append(time_label)
                            if target_perc is not None:
                                name_parts.append(f"{target_perc}p")
                            name_parts.append('wl')
                        elif outcome_type == 'dropout':
                            name_parts.extend([time_label, "dropout"])
                        name_parts.append(predictor.replace('_likert','').replace('_yn',''))
                        name_parts.append(adj_label)
                        scenario = {
                            'name': "_".join(name_parts),
                            'outcome_type': outcome_type,
                            'time_window': tw,
                            'target_perc': target_perc,
                            'predictors': [predictor],
                            'covariates': covariates
                        }
                        scenarios.append(scenario)
    return scenarios

def define_logreg_outcome(df, scenario):
    """Create binary outcome column for the scenario."""
    df = df.copy()
    name = scenario['name']
    outcome_type = scenario['outcome_type']
    tw = scenario['time_window']
    perc = scenario.get('target_perc')
    if outcome_type == 'target_wl':
        col = f"wl_{tw}d_%" if isinstance(tw, int) else "total_wl_%"
        df[f"{name}_outcome"] = (df[col].abs() >= perc).astype(int)
    elif outcome_type == 'dropout':
        if tw == 'instant':
            df[f"{name}_outcome"] = (df['nr_total_measurements'] == 1).astype(int)
        else:
            col = f"{tw}d_dropout"
            df[f"{name}_outcome"] = df[col].fillna(0).astype(int)
    return df

def run_logistic_regression(df, outcome_col, predictors):
    """Fit logistic regression and return summary DataFrame, skipping collinear predictors."""
    X = sm.add_constant(df[predictors])
    y = df[outcome_col]
    # Check for collinearity or constant columns
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        print(f"Skipping scenario due to collinear or constant predictors: {predictors}")
        if constant_cols:
            print(f"  Predictors with zero variance: {constant_cols}")
        else:
            print("  Collinearity detected (not just constant columns).")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e} for predictors: {predictors}")
        print("  Unique values per predictor:")
        for col in X.columns:
            print(f"    {col}: {X[col].unique()}")
        return None
    except Exception as e:
        print(f"Other error: {e} for predictors: {predictors}")
        return None
    params = result.params
    conf = result.conf_int()
    or_ci = np.exp(conf)
    summary = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'Odds Ratio': np.exp(params.values),
        'OR CI Lower 95%': or_ci[0].values,
        'OR CI Upper 95%': or_ci[1].values,
        'P-Value': result.pvalues.values
    })
    return summary

def run_logreg_pipeline(db_path, input_table, scenarios, output_table, output_db_path):
    """Main pipeline: load, preprocess, run all scenarios, save results."""
    df = load_data(db_path, input_table)
    df = preprocess_for_regression(df, scenarios)
    all_results = []
    for scenario in scenarios:
        df_scen = define_logreg_outcome(df, scenario)
        outcome_col = f"{scenario['name']}_outcome"
        cols = scenario['predictors'] + scenario['covariates']
        data = df_scen[[outcome_col] + cols].dropna()
        if data[outcome_col].nunique() < 2 or data.empty:
            continue
        summary = run_logistic_regression(data, outcome_col, cols)
        if summary is not None:
            summary.insert(0, 'scenario', scenario['name'])
            all_results.append(summary)
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        with sqlite3.connect(output_db_path) as conn:
            results_df.to_sql(output_table, conn, if_exists='replace', index=False)
        print(f"Saved results to {output_db_path}, table '{output_table}'.")

# ====LINEAR OUTCOME PREDICTIONS - INFERENCE====

def generate_linreg_scenarios(
    outcome_types,
    time_windows,
    main_predictors,
    adjustment_sets,
    input_table=None,
    df_all=None
):
    """
    Generates a list of scenario dictionaries for linear regression analyses.
    If input_table == "timetoevent_eb_wgc_compl", includes all weight gain cause columns as predictors.
    """
    scenarios = []
    # Add weight gain cause columns if needed
    wgc_predictors = []
    if input_table in ("timetoevent_eb_wgc_compl", "timetoevent_wgc_compl") and df_all is not None:
        cols = list(df_all.columns)
        start = cols.index('weight_gain_cause_en') + 1
        end = cols.index('genomics_sample_id')
        wgc_predictors = cols[start:end]
        main_predictors = list(main_predictors) + wgc_predictors

    for outcome_type in outcome_types:
        for tw in time_windows:
            for predictor in main_predictors:
                for covariates, adj_label in adjustment_sets:
                    name_parts = []
                    time_label = f"{tw}d" if isinstance(tw, int) else tw
                    name_parts.append(time_label)
                    name_parts.append(outcome_type)
                    name_parts.append(predictor.replace('_likert','').replace('_yn',''))
                    name_parts.append(adj_label)
                    scenario = {
                        'name': "_".join(name_parts),
                        'outcome_type': outcome_type,
                        'time_window': tw,
                        'predictors': [predictor],
                        'covariates': covariates
                    }
                    scenarios.append(scenario)
    return scenarios


def define_linreg_outcome(df, scenario):
    """
    Create continuous outcome column for the scenario.
    Supports: weight_change, fat_change, muscle_change at given time window or total.
    """
    df = df.copy()
    name = scenario['name']
    outcome_type = scenario['outcome_type']
    tw = scenario['time_window']
    # Define outcome column names in your data accordingly
    if outcome_type == 'weight_change':
        col = f"wl_{tw}d_%" if isinstance(tw, int) else "total_wl_%"
    elif outcome_type == 'fat_change':
        col = f"{tw}d_fat_loss_%" if isinstance(tw, int) else "total_fat_loss_%"
    elif outcome_type == 'muscle_change':
        col = f"{tw}d_muscle_change_%" if isinstance(tw, int) else "total_muscle_change_%"
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")
    df[f"{name}_outcome"] = df[col]
    return df

def run_linear_regression(df, outcome_col, predictors):
    """
    Fit linear regression and return summary DataFrame, skipping collinear predictors.
    """
    X = sm.add_constant(df[predictors])
    y = df[outcome_col]
    # Check for collinearity or constant columns
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        print(f"Skipping scenario due to collinear or constant predictors: {predictors}")
        if constant_cols:
            print(f"  Predictors with zero variance: {constant_cols}")
        else:
            print("  Collinearity detected (not just constant columns).")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X)
            result = model.fit()
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e} for predictors: {predictors}")
        print("  Unique values per predictor:")
        for col in X.columns:
            print(f"    {col}: {X[col].unique()}")
        return None
    except Exception as e:
        print(f"Other error: {e} for predictors: {predictors}")
        return None
    params = result.params
    conf = result.conf_int()
    summary = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'CI Lower 95%': conf[0].values,
        'CI Upper 95%': conf[1].values,
        'P-Value': result.pvalues.values
    })
    return summary

def run_linreg_pipeline(db_path, input_table, scenarios, output_table, output_db_path):
    """Main pipeline: load, preprocess, run all scenarios, save results."""
    df = load_data(db_path, input_table)
    df = preprocess_for_regression(df, scenarios)
    all_results = []
    for scenario in scenarios:
        df_scen = define_linreg_outcome(df, scenario)
        outcome_col = f"{scenario['name']}_outcome"
        cols = scenario['predictors'] + scenario['covariates']
        data = df_scen[[outcome_col] + cols].dropna()
        if data.empty:
            continue
        summary = run_linear_regression(data, outcome_col, cols)
        if summary is not None:
            summary.insert(0, 'scenario', scenario['name'])
            all_results.append(summary)
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        with sqlite3.connect(output_db_path) as conn:
            results_df.to_sql(output_table, conn, if_exists='replace', index=False)
        print(f"Saved results to {output_db_path}, table '{output_table}'.")

# ====LINEAR ADHERENCE PREDICTIONS - INFERENCE====

def generate_adherence_scenarios(
    target_percentages,
    main_predictors,
    adjustment_sets,
    input_table=None,
    df_all=None,
    extra_outcomes=None
):
    """
    Generates scenario dictionaries for all adherence outcomes:
    - total_followup_days
    - dietitian_visits
    - nr_total_measurements
    - avg_days_between_measurements
    - days_to_X%_wl (for each X in target_percentages)
    If input_table == "timetoevent_eb_wgc_compl", includes all weight gain cause columns as predictors.
    """
    scenarios = []

    # Dynamically add weight gain cause predictors if appropriate
    wgc_predictors = []
    if input_table in ("timetoevent_eb_wgc_compl", "timetoevent_wgc_compl") and df_all is not None:
        cols = list(df_all.columns)
        start = cols.index('weight_gain_cause_en') + 1
        end = cols.index('genomics_sample_id')
        wgc_predictors = cols[start:end]
        all_predictors = list(main_predictors) + wgc_predictors
    else:
        all_predictors = list(main_predictors)

    # All outcomes except days_to_X%_wl
    static_outcomes = [
        'total_followup_days',
        'dietitian_visits',
        'nr_total_measurements',
        'avg_days_between_measurements'
    ]
    if extra_outcomes:
        static_outcomes += extra_outcomes

    # Scenarios for static outcomes
    for outcome in static_outcomes:
        for predictor in all_predictors:
            for covariates, adj_label in adjustment_sets:
                name = f"{outcome}_{predictor.replace('_likert','').replace('_yn','')}_{adj_label}"
                scenario = {
                    'name': name,
                    'outcome_type': outcome,
                    'predictors': [predictor],
                    'covariates': covariates
                }
                scenarios.append(scenario)

    # Scenarios for days_to_X%_wl
    for perc in target_percentages:
        outcome = f"days_to_{perc}%_wl"
        for predictor in all_predictors:
            for covariates, adj_label in adjustment_sets:
                name = f"{outcome}_{predictor.replace('_likert','').replace('_yn','')}_{adj_label}"
                scenario = {
                    'name': name,
                    'outcome_type': 'days_to_Xpct_wl',
                    'target_perc': perc,
                    'predictors': [predictor],
                    'covariates': covariates
                }
                scenarios.append(scenario)
    return scenarios

def define_adherence_outcome(df, scenario):
    """
    Create continuous adherence outcome column for the scenario.
    Supports all listed outcomes.
    """
    df = df.copy()
    name = scenario['name']
    if scenario['outcome_type'] == 'days_to_Xpct_wl':
        perc = scenario.get('target_perc')
        col = f"days_to_{perc}%_wl"
    else:
        col = scenario['outcome_type']
    df[f"{name}_outcome"] = df[col]
    return df

def run_linear_regression(df, outcome_col, predictors):
    """
    Fit linear regression and return summary DataFrame, skipping collinear predictors.
    """
    X = sm.add_constant(df[predictors])
    y = df[outcome_col]
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        print(f"Skipping scenario due to collinear or constant predictors: {predictors}")
        if constant_cols:
            print(f"  Predictors with zero variance: {constant_cols}")
        else:
            print("  Collinearity detected (not just constant columns).")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X)
            result = model.fit()
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e} for predictors: {predictors}")
        print("  Unique values per predictor:")
        for col in X.columns:
            print(f"    {col}: {X[col].unique()}")
        return None
    except Exception as e:
        print(f"Other error: {e} for predictors: {predictors}")
        return None
    params = result.params
    conf = result.conf_int()
    summary = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'CI Lower 95%': conf[0].values,
        'CI Upper 95%': conf[1].values,
        'P-Value': result.pvalues.values
    })
    return summary

def run_adherence_linreg_pipeline(db_path, input_table, scenarios, output_table, output_db_path):
    """Main pipeline: load, preprocess, run all scenarios, save results."""
    df = load_data(db_path, input_table)
    df = preprocess_for_regression(df, scenarios)
    all_results = []
    for scenario in scenarios:
        df_scen = define_adherence_outcome(df, scenario)
        outcome_col = f"{scenario['name']}_outcome"
        cols = scenario['predictors'] + scenario['covariates']
        data = df_scen[[outcome_col] + cols].dropna()
        if data.empty:
            continue
        summary = run_linear_regression(data, outcome_col, cols)
        if summary is not None:
            summary.insert(0, 'scenario', scenario['name'])
            all_results.append(summary)
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        with sqlite3.connect(output_db_path) as conn:
            results_df.to_sql(output_table, conn, if_exists='replace', index=False)
        print(f"Saved results to {output_db_path}, table '{output_table}'.")

# ====LINEAR EB PREDICTIONS BY WGC - INFERENCE====

def generate_eb_wgc_scenarios(eb_vars, wgc_vars, adjustment_sets):
    scenarios = []
    for eb in eb_vars:
        for wgc in wgc_vars:
            for covariates, adj_label in adjustment_sets:
                scenario = {
                    'eb_var': eb,
                    'wgc_var': wgc,
                    'covariates': covariates,
                    'adj_label': adj_label,
                    'name': f"{eb}__{wgc}__{adj_label}"
                }
                scenarios.append(scenario)
    return scenarios

def run_linear_regression(df, outcome_col, predictors):
    X = sm.add_constant(df[predictors])
    y = df[outcome_col]
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X)
            result = model.fit()
    except Exception:
        return None
    params = result.params
    conf = result.conf_int()
    summary = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'CI Lower 95%': conf[0].values,
        'CI Upper 95%': conf[1].values,
        'P-Value': result.pvalues.values
    })
    return summary

def run_eb_linreg_from_wgc_pipeline(db_path, input_table, eb_vars, adjustment_sets, output_table, output_db_path):
    df = load_data(db_path, input_table)
    # Identify WGC columns (between 'weight_gain_cause_en' and 'genomics_sample_id')
    cols = list(df.columns)
    start = cols.index('weight_gain_cause_en') + 1
    end = cols.index('genomics_sample_id')
    wgc_vars = cols[start:end]
    scenarios = generate_eb_wgc_scenarios(eb_vars, wgc_vars, adjustment_sets)
    all_results = []
    for scenario in scenarios:
        outcome = scenario['eb_var']
        predictors = [scenario['wgc_var']] + scenario['covariates']
        df_proc = preprocess_for_eb_by_wgc_linreg(df, predictors)
        data = df_proc[[outcome] + predictors].dropna()
        if data.empty:
            continue
        summary = run_linear_regression(data, outcome, predictors)
        if summary is not None:
            summary.insert(0, 'scenario', scenario['name'])
            summary.insert(1, 'outcome', outcome)
            summary.insert(2, 'predictor', scenario['wgc_var'])
            summary.insert(3, 'adjustment', scenario['adj_label'])
            all_results.append(summary)
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        with sqlite3.connect(output_db_path) as conn:
            results_df.to_sql(output_table, conn, if_exists='replace', index=False)
        print(f"Saved results to {output_db_path}, table '{output_table}'.")

# ==== LOGISTIC EB PREDICTIONS BY WGC - INFERENCE ====

def generate_eb_wgc_scenarios(eb_vars, wgc_vars, adjustment_sets):
    scenarios = []
    for eb in eb_vars:
        for wgc in wgc_vars:
            for covariates, adj_label in adjustment_sets:
                scenario = {
                    'eb_var': eb,
                    'wgc_var': wgc,
                    'covariates': covariates,
                    'adj_label': adj_label,
                    'name': f"{eb}__{wgc}__{adj_label}"
                }
                scenarios.append(scenario)
    return scenarios

def run_logistic_regression(df, outcome_col, predictors):
    X = sm.add_constant(df[predictors])
    y = df[outcome_col]
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X)
            result = model.fit(disp=0)
    except Exception:
        return None
    params = result.params
    conf = result.conf_int()
    summary = pd.DataFrame({
        'Variable': params.index,
        'Coefficient': params.values,
        'Odds Ratio': np.exp(params.values),
        'CI Lower 95%': np.exp(conf[0].values),
        'CI Upper 95%': np.exp(conf[1].values),
        'P-Value': result.pvalues.values
    })
    return summary

def run_eb_logreg_from_wgc_pipeline(db_path, input_table, eb_binary_vars, eb_likert_vars, adjustment_sets, output_table, output_db_path):
    df = load_data(db_path, input_table)
    cols = list(df.columns)
    start = cols.index('weight_gain_cause_en') + 1
    end = cols.index('genomics_sample_id')
    wgc_vars = cols[start:end]

    medians = {}
    for var in eb_likert_vars:
        if var in df.columns:
            medians[var] = pd.to_numeric(df[var], errors="coerce").median()
            df[f"{var}_over_median"] = pd.to_numeric(df[var], errors="coerce")
            df[f"{var}_over_median"] = np.where(df[f"{var}_over_median"] >= medians[var], 1, 0)
        else:
            medians[var] = None

    # Combine all EB variables for scenarios (binary + over-median likert)
    eb_vars = list(eb_binary_vars) + [f"{v}_over_median" for v in eb_likert_vars if medians[v] is not None]

    scenarios = generate_eb_wgc_scenarios(eb_vars, wgc_vars, adjustment_sets)
    all_results = []
    for scenario in scenarios:
        outcome = scenario['eb_var']
        predictors = [scenario['wgc_var']] + scenario['covariates']
        df_proc = preprocess_for_eb_by_wgc_logreg(df, predictors, outcome)
        data = df_proc[[outcome] + predictors].dropna()
        if data.empty or data[outcome].nunique() < 2:
            continue
        summary = run_logistic_regression(data, outcome, predictors)
        if summary is not None:
            summary.insert(0, 'scenario', scenario['name'])
            all_results.append(summary)
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        with sqlite3.connect(output_db_path) as conn:
            results_df.to_sql(output_table, conn, if_exists='replace', index=False)
        print(f"Saved results to {output_db_path}, table '{output_table}'.")


