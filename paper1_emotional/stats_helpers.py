import numpy as np
import pandas as pd
from scipy import stats

def format_mean_sd(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 1:
        return "N/A"
    return f"{s.mean():.2f} ± {s.std():.2f}"

def format_n_perc(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "0 (0.0%)"
    n = int((s == 1).sum())
    pct = 100 * n / len(s)
    return f"{n} ({pct:.1f}%)"

def format_availability(series):
    n = series.notna().sum()
    pct = 100 * n / len(series)
    return f"{n} ({pct:.1f}%)"

def welchs_ttest(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    _, p = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
    return p

def categorical_pvalue(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    table = [
        [(x == 1).sum(), (x == 0).sum()],
        [(y == 1).sum(), (y == 0).sum()]
    ]
    if np.any(np.array(table) < 5):
        try:
            _, p = stats.fisher_exact(table)
        except Exception:
            return np.nan
    else:
        try:
            _, p, _, _ = stats.chi2_contingency(table)
        except Exception:
            return np.nan
    return p

def format_pval(p):
    if pd.isna(p):
        return "N/A"
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def cohens_d(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    if pooled_sd == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_sd

def format_cohens_d(d):
    if pd.isna(d):
        return "N/A"
    return f"{d:.2f}"