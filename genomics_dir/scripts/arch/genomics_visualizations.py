
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import gaussian_kde, kruskal, mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests


@dataclass
class VizConfig:
    paths: object
    cohort_tables: Dict[str, str]
    group_colors: Dict[str, str]
    master_group_order: List[str]
    display: Dict[str, str]
    output_dir: str
    outputs: Dict[str, str]
    wl_targets: List[int] = field(default_factory=lambda: [10])
    cols_alluvial: List[str] = field(default_factory=lambda: ["instant_dropout", "120d_dropout", "10%_wl_achieved"])
    cols_km: List[str] = field(default_factory=lambda: ["instant_dropout", "120d_dropout", "total_followup_days", "10%_wl_achieved", "days_to_10%_wl"])
    cols_violin: List[str] = field(default_factory=lambda: ["120d_dropout", "120d_wl_%"])
    alluvial_title: str = "Patient Flow: Personalization Timing, Adherence and Weight Loss"
    km_title: str = "Kaplan-Meier Survival Curves by Personalization Group"
    violin_title: str = "Weight Loss at 120 Days by Personalization Group"
    alluvial_subtitle: str = "First medical records only"
    km_subtitle: str = "First medical records only | Shaded area = 95% CI"
    violin_subtitle: str = "Violin width ∝ % reaching 120-day mark | Box = IQR"


def load_groups(paths, cohort_tables, cols_needed):
    dfs = []
    with sqlite3.connect(paths.paper_in_db) as conn:
        for table_name, group_code in cohort_tables.items():
            q = f"SELECT {', '.join([f'`{c}`' for c in cols_needed])} FROM `{table_name}`"
            df = pd.read_sql_query(q, conn)
            df['group'] = group_code
            dfs.append(df)
    if not dfs:
        raise ValueError('No data loaded.')
    return pd.concat(dfs, ignore_index=True)


def active_order(df, master_order):
    return [g for g in master_order if g in df['group'].unique()]


def classify_adherence(row):
    if row['instant_dropout'] == 1:
        return 'instant_dropout'
    if row['120d_dropout'] == 1:
        return 'dropout_lt120d'
    return 'reached_120d'


def classify_wl(row):
    return 'achieved_wl' if row['10%_wl_achieved'] == 1 else 'not_achieved_wl'


def hex_to_rgba(h, a):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{a})'


def format_p(p):
    if pd.isna(p):
        return 'N/A'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def make_alluvial(cfg):
    df = load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_alluvial)
    active = active_order(df, cfg.master_group_order)
    df['adherence'] = df.apply(classify_adherence, axis=1)
    df['wl_outcome'] = df.apply(classify_wl, axis=1)
    # Only keep categories that are actually present in the data:
    adherence_order = [c for c in ['dropout_lt120d', 'reached_120d'] if c in df['adherence'].values] #'instant_dropout',
    wl_order = [c for c in ['not_achieved_wl', 'achieved_wl'] if c in df['wl_outcome'].values]
    df['group'] = pd.Categorical(df['group'], categories=active, ordered=True)
    df['adherence'] = pd.Categorical(df['adherence'], categories=adherence_order, ordered=True)
    df['wl_outcome'] = pd.Categorical(df['wl_outcome'], categories=wl_order, ordered=True)

    fig = go.Figure(data=[go.Parcats(
        dimensions=[
            go.parcats.Dimension(values=df['group'].map(cfg.display), label='Group', categoryorder='array', categoryarray=[cfg.display[c] for c in active]),
            go.parcats.Dimension(values=df['adherence'].map(cfg.display), label='Adherence', categoryorder='array', categoryarray=[cfg.display[c] for c in adherence_order]),
            go.parcats.Dimension(values=df['wl_outcome'].map(cfg.display), label='Weight Loss Outcome', categoryorder='array', categoryarray=[cfg.display[c] for c in wl_order]),
        ],
        line=dict(color=df['group'].map(cfg.group_colors).tolist(), shape='hspline'),
        labelfont=dict(size=17, family='Arial'),
        tickfont=dict(size=15, family='Arial'),
        arrangement='freeform',
        bundlecolors=True,
        hoverinfo='count+probability',
        domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
    )])
    fig.update_layout(
        title=dict(text=f"<b>{cfg.alluvial_title}</b><br><sup>{cfg.alluvial_subtitle}</sup>", x=0.5, font=dict(size=16, family='Arial')), 
        font=dict(size=14, family='Arial'), height=750, width=1300, margin=dict(l=120, r=200, t=120, b=40), paper_bgcolor='white'
    )
    out = Path(cfg.output_dir) / cfg.outputs['alluvial']
    fig.write_html(str(out))
    return fig, out


def km_traces(df, time_col, event_col, group_col, group_order, colors, display, invert=False):
    traces = []
    all_T, all_E, all_G = [], [], []
    for group_code in group_order:
        sub = df[df[group_col] == group_code]
        if sub.empty:
            continue
        T = sub[time_col].values
        E = sub[event_col].values
        all_T.append(T)
        all_E.append(E)
        all_G.append(np.full(len(T), group_code))
        kmf = KaplanMeierFitter().fit(T, event_observed=E)
        t_vals = kmf.survival_function_.index.values
        s_vals = kmf.survival_function_['KM_estimate'].values
        ci_low = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        ci_high = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
        if invert:
            s_vals = 1 - s_vals
            ci_low, ci_high = 1 - ci_high, 1 - ci_low
        color_hex = colors[group_code]
        traces.append(go.Scatter(x=np.concatenate([t_vals, t_vals[::-1]]), y=np.concatenate([ci_high, ci_low[::-1]]), fill='toself', fillcolor=hex_to_rgba(color_hex, 0.15), line=dict(width=0), hoverinfo='skip', showlegend=False))
        traces.append(go.Scatter(x=t_vals, y=s_vals, mode='lines', line=dict(color=color_hex, width=2.5, shape='hv'), name=f"{display[group_code]} (n={len(T)})"))
    lr = multivariate_logrank_test(np.concatenate(all_T), np.concatenate(all_G), np.concatenate(all_E))
    return traces, lr.p_value


def make_km(cfg):
    df = load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_km)
    active = active_order(df, cfg.master_group_order)
    df['dropout_event'] = ((df['instant_dropout'] == 1) | (df['120d_dropout'] == 1)).astype(int)
    df['dropout_time'] = df['total_followup_days']
    df['wl10_event'] = df['10%_wl_achieved'].fillna(0).astype(int)
    df['wl10_time'] = np.where(df['wl10_event'] == 1, df['days_to_10%_wl'], df['total_followup_days'])
    df_dropout = df.dropna(subset=['dropout_time']).copy()
    df_wl10 = df.dropna(subset=['wl10_time']).copy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Time to Dropout', 'Time to ≥10% Weight Loss'], horizontal_spacing=0.12)
    tr_do, p_do = km_traces(df_dropout, 'dropout_time', 'dropout_event', 'group', active, cfg.group_colors, cfg.display, invert=False)
    for tr in tr_do:
        fig.add_trace(tr, row=1, col=1)
    fig.add_annotation(text=f"Log-rank: {format_p(p_do)}", xref='x domain', yref='y domain', x=0.97, y=0.97, xanchor='right', yanchor='top', showarrow=False, font=dict(size=12, family='Arial'), bgcolor='rgba(240,240,240,0.85)', bordercolor='gray', borderwidth=1, row=1, col=1)
    tr_wl, p_wl = km_traces(df_wl10, 'wl10_time', 'wl10_event', 'group', active, cfg.group_colors, cfg.display, invert=True)
    for tr in tr_wl:
        tr.showlegend = False
        fig.add_trace(tr, row=1, col=2)
    fig.add_annotation(text=f"Log-rank: {format_p(p_wl)}", xref='x2 domain', yref='y2 domain', x=0.03, y=0.97, xanchor='left', yanchor='top', showarrow=False, font=dict(size=12, family='Arial'), bgcolor='rgba(240,240,240,0.85)', bordercolor='gray', borderwidth=1, row=1, col=2)
    fig.update_xaxes(title_text='Days from baseline', row=1, col=1)
    fig.update_xaxes(title_text='Days from baseline', row=1, col=2)
    fig.update_yaxes(title_text='Probability of remaining enrolled', range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text='Cumulative probability of ≥10% WL', range=[0, 1.05], row=1, col=2)
    fig.update_layout(title=dict(text=f"<b>{cfg.km_title}</b><br><sup>{cfg.km_subtitle}</sup>", x=0.5, font=dict(size=16, family='Arial')), font=dict(size=13, family='Arial'), height=550, width=1300, paper_bgcolor='white', plot_bgcolor='white', legend=dict(title='Group', x=1.01, y=0.5, xanchor='left', font=dict(size=12, family='Arial'), bgcolor='rgba(255,255,255,0.9)', bordercolor='lightgray', borderwidth=1), margin=dict(l=70, r=160, t=100, b=60))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=False)
    out = Path(cfg.output_dir) / cfg.outputs['km']
    fig.write_html(str(out))
    return fig, out


def make_violin(cfg):
    df = load_groups(cfg.paths, cfg.cohort_tables, cfg.cols_violin)
    active = active_order(df, cfg.master_group_order)
    fig = go.Figure()
    group_stats = {}
    for group_code in active:
        sub = df[df['group'] == group_code]
        n_total = len(sub)
        reached = sub[sub['120d_dropout'] == 0]['120d_wl_%'].dropna()
        group_stats[group_code] = dict(values=reached.values, n_total=n_total, n_reached=len(reached), pct_reached=(len(reached) / n_total if n_total else 0), median=reached.median() if len(reached) else np.nan, q1=reached.quantile(0.25) if len(reached) else np.nan, q3=reached.quantile(0.75) if len(reached) else np.nan)
    kw_groups = [group_stats[g]['values'] for g in active if len(group_stats[g]['values']) >= 5]
    kw_p = kruskal(*kw_groups).pvalue if len(kw_groups) >= 2 else np.nan
    pairs = list(combinations(active, 2))
    raw_pvals = []
    for g1, g2 in pairs:
        v1, v2 = group_stats[g1]['values'], group_stats[g2]['values']
        raw_pvals.append(mannwhitneyu(v1, v2, alternative='two-sided').pvalue if len(v1) >= 5 and len(v2) >= 5 else np.nan)
    valid = [p for p in raw_pvals if not np.isnan(p)]
    corrected = multipletests(valid, alpha=0.05, method='fdr_bh')[1] if len(valid) else []
    corr_iter = iter(corrected)
    pair_results = {}
    for (g1, g2), raw_p in zip(pairs, raw_pvals):
        if np.isnan(raw_p):
            pair_results[(g1, g2)] = {'sig': False, 'corrected': np.nan}
        else:
            cp = next(corr_iter)
            pair_results[(g1, g2)] = {'sig': cp < 0.05, 'corrected': cp}
    max_pct = max(s['pct_reached'] for s in group_stats.values()) if group_stats else 1
    MAX_HALF_WIDTH = 0.35
    for x_pos, group_code in enumerate(active):
        stats = group_stats[group_code]
        vals = stats['values']
        color = cfg.group_colors[group_code]
        label = cfg.display[group_code]
        half_w = (stats['pct_reached'] / max_pct) * MAX_HALF_WIDTH if max_pct else MAX_HALF_WIDTH
        if len(vals) < 5:
            fig.add_trace(go.Scatter(x=[x_pos], y=[stats['median']], mode='markers', marker=dict(color=color, size=10, symbol='diamond'), name=label, showlegend=True))
            continue
        y_min, y_max = vals.min() - 1, vals.max() + 1
        y_grid = np.linspace(y_min, y_max, 300)
        kde = gaussian_kde(vals, bw_method='scott')
        density = kde(y_grid)
        density_scaled = density / density.max() * half_w
        x_right = x_pos + density_scaled
        x_left = x_pos - density_scaled[::-1]
        x_outline = np.concatenate([x_right, x_left, [x_right[0]]])
        y_outline = np.concatenate([y_grid, y_grid[::-1], [y_grid[0]]])
        fig.add_trace(go.Scatter(x=x_outline, y=y_outline, fill='toself', fillcolor=hex_to_rgba(color, 0.55), line=dict(color=color, width=1.5), name=label, legendgroup=group_code, showlegend=True, hoverinfo='skip'))
        med = stats['median']
        med_density = kde([med])[0] / density.max() * half_w
        fig.add_trace(go.Scatter(x=[x_pos - med_density, x_pos + med_density], y=[med, med], mode='lines', line=dict(color='white', width=3), legendgroup=group_code, showlegend=False, hoverinfo='skip'))
        q1, q3 = stats['q1'], stats['q3']
        iqr_density = kde(np.array([q1, q3])) / density.max() * half_w
        box_hw = min(iqr_density) * 0.6
        fig.add_trace(go.Scatter(x=[x_pos - box_hw, x_pos + box_hw, x_pos + box_hw, x_pos - box_hw, x_pos - box_hw], y=[q1, q1, q3, q3, q1], fill='toself', fillcolor=hex_to_rgba(color, 0.85), line=dict(color='white', width=1), legendgroup=group_code, showlegend=False, hoverinfo='skip'))
        fig.add_annotation(x=x_pos, y=stats['median'], text=f"<b>{stats['median']:.1f}%</b>", showarrow=False, font=dict(size=11, family='Arial', color='white'))
        fig.add_annotation(x=x_pos, y=y_min - 1.5, text=f"<b>{stats['pct_reached']*100:.0f}%</b> reached 120d<br><sup>n={stats['n_reached']}/{stats['n_total']}</sup>", showarrow=False, font=dict(size=11, family='Arial', color=color), yref='y')
    all_vals = np.concatenate([s['values'] for s in group_stats.values() if len(s['values']) > 0])
    y_max_data = np.percentile(all_vals, 99) if len(all_vals) else 1
    bracket_step = abs(y_max_data) * 0.08 if y_max_data else 1
    kw_color = '#CC0000' if (kw_p < 0.05 if not np.isnan(kw_p) else False) else '#888888'
    fig.add_annotation(x=0.5, y=1.06, xref='paper', yref='paper', text=f"Kruskal-Wallis: {format_p(kw_p)}", showarrow=False, font=dict(size=12, family='Arial', color=kw_color), align='center')
    bracket_y = y_max_data + bracket_step
    for g1, g2 in [k for k, v in pair_results.items() if v['sig']]:
        x1, x2 = active.index(g1), active.index(g2)
        p_str = format_p(pair_results[(g1, g2)]['corrected'])
        fig.add_shape(type='line', x0=x1, x1=x2, y0=bracket_y, y1=bracket_y, line=dict(color='#444444', width=1.5))
        fig.add_shape(type='line', x0=x1, x1=x1, y0=bracket_y - bracket_step * 0.2, y1=bracket_y, line=dict(color='#444444', width=1.5))
        fig.add_shape(type='line', x0=x2, x1=x2, y0=bracket_y - bracket_step * 0.2, y1=bracket_y, line=dict(color='#444444', width=1.5))
        fig.add_annotation(x=(x1 + x2) / 2, y=bracket_y + bracket_step * 0.15, text=p_str, showarrow=False, font=dict(size=10, family='Arial', color='#444444'))
        bracket_y += bracket_step * 1.1
    fig.update_yaxes(range=[all_vals.min() - 3, bracket_y + bracket_step] if len(all_vals) else [0, 1])
    fig.update_layout(title=dict(text=f"<b>{cfg.violin_title}</b><br><sup>{cfg.violin_subtitle}</sup>", x=0.5, font=dict(size=16, family='Arial')), xaxis=dict(tickvals=list(range(len(active))), ticktext=[cfg.display[g] for g in active], tickfont=dict(size=13, family='Arial'), showgrid=False, zeroline=False), yaxis=dict(title='Weight loss at 120 days (%)', title_font=dict(size=13, family='Arial'), tickfont=dict(size=12), showgrid=True, gridcolor='rgba(200,200,200,0.4)', zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1), plot_bgcolor='white', paper_bgcolor='white', font=dict(family='Arial'), height=650, width=1000, showlegend=False, margin=dict(l=70, r=40, t=100, b=100))
    out = Path(cfg.output_dir) / cfg.outputs['violin']
    fig.write_html(str(out))
    return fig, out


def run_all_visualizations(cfg):
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return {
        'alluvial': make_alluvial(cfg),
        'km': make_km(cfg),
        'violin': make_violin(cfg),
    }
