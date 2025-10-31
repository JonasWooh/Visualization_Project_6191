
# app.py
# Dash app for "YouTube & TikTok Trends 2025" — Interactive EDA Dashboard
import os
import math
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# -------------------------
# Config
# -------------------------
DEFAULT_PATHS = [
    os.getenv("DATA_PATH", "").strip(),
    "data/youtube_shorts_tiktok_trends_2025.csv",
    "data/youtube_shorts_tiktok_trends_2025_ml.csv",
    "../data/youtube_shorts_tiktok_trends_2025.csv",
    "../data/youtube_shorts_tiktok_trends_2025_ml.csv",
]

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

DATA_PATH = _first_existing(DEFAULT_PATHS)
if DATA_PATH is None:
    raise FileNotFoundError(
        "Cannot find data file. Set env DATA_PATH or copy CSV into ./data/"
    )

# -------------------------
# Load & precompute
# -------------------------
df = pd.read_csv(DATA_PATH)

def make_sparse_marks(vmin, vmax, step=10, fmt=str):
    """Return sparse marks dict: {value: label} every `step` units."""
    vmin = int(np.floor(vmin)); vmax = int(np.ceil(vmax))
    return {v: fmt(v) for v in range(vmin, vmax + 1, step)}

num_cols = [
    "views","likes","comments","shares","saves",
    "engagement_rate","share_rate","save_rate","comment_ratio",
    "like_rate","dislike_rate","engagement_per_1k",
    "engagement_like_rate","engagement_comment_rate","engagement_share_rate",
    "duration_sec","avg_watch_time_sec","completion_rate",
    "upload_hour","trend_duration_days","engagement_velocity"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

weekday_map = {
    "Monday":"Monday","Tuesday":"Tuesday","Wednesday":"Wednesday",
    "Thursday":"Thursday","Friday":"Friday","Saturday":"Saturday","Sunday":"Sunday"
}
if "publish_dayofweek" in df.columns:
    df["publish_dayofweek"] = df["publish_dayofweek"].map(weekday_map).fillna(df["publish_dayofweek"])
    cat_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    df["publish_dayofweek"] = pd.Categorical(df["publish_dayofweek"], cat_order, ordered=True)

if "is_weekend" in df.columns:
    df["weekend"] = df["is_weekend"].map({0:"Weekday",1:"Weekend"}).astype("category")

if "country" in df.columns:
    df["country"] = df["country"].astype(str)

platforms = sorted(df["platform"].dropna().unique()) if "platform" in df.columns else []
countries = sorted(df["country"].dropna().unique())[:50] if "country" in df.columns else []
categories = sorted(df["category"].dropna().unique())[:50] if "category" in df.columns else []

app = Dash(__name__)
app.title = "Short-Video Trends 2025 — Interactive EDA"

def kpi_card(title, value, subtitle=None):
    return html.Div([
        html.Div(title, style={"fontSize":"14px","color":"#6b7280"}),
        html.Div(value, style={"fontSize":"24px","fontWeight":"700"}),
        html.Div(subtitle or "", style={"fontSize":"12px","color":"#6b7280"}),
    ], style={
        "padding":"12px 16px","border":"1px solid #e5e7eb","borderRadius":"12px",
        "background":"#fff","boxShadow":"0 1px 2px rgba(0,0,0,0.04)"
    })

app.layout = html.Div([
    html.H2("YouTube & TikTok Trends — Interactive EDA Dashboard", style={"margin":"12px 0 4px"}),
    html.Div("Data source: YouTube Shorts & TikTok Trends 2025 (local CSV)", style={"color":"#6b7280","marginBottom":"12px"}),

    html.Div([
        html.Div([
            html.Label("Platform"),
            dcc.Dropdown(platforms, value=platforms, multi=True, id="f-platform")
        ], style={"flex":"1","minWidth":"180px","marginRight":"12px"}),
        html.Div([
            html.Label("Country (top options)"),
            dcc.Dropdown(countries, multi=True, id="f-country")
        ], style={"flex":"1","minWidth":"220px","marginRight":"12px"}),
        html.Div([
            html.Label("Category (optional)"),
            dcc.Dropdown(categories, multi=True, id="f-category")
        ], style={"flex":"1","minWidth":"220px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"8px","marginBottom":"12px"}),

    html.Div([
        html.Div([
            html.Label("Duration (sec)"),
            dcc.RangeSlider(id="f-duration", min=float(np.nanmin(df["duration_sec"])) if "duration_sec" in df else 0,
                            max=float(np.nanmax(df["duration_sec"])) if "duration_sec" in df else 90,
                            value=[float(np.nanpercentile(df["duration_sec"],5)), float(np.nanpercentile(df["duration_sec"],95))] if "duration_sec" in df else [5,90],
                            step=1, allowCross=False,
                            tooltip={"always_visible":False,"placement":"bottom"})
        ], style={"flex":"2","minWidth":"300px","marginRight":"12px"}),
        html.Div([
            html.Label("Upload hour"),
            dcc.RangeSlider(id="f-hour", min=0, max=23, value=[0,23], step=1,
                            marks={0:"0",6:"6",12:"12",18:"18",23:"23"})
        ], style={"flex":"1","minWidth":"240px"}),
    ], style={"display":"flex","flexWrap":"wrap","gap":"8px","marginBottom":"16px"}),

    html.Div(id="kpi-row", style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(180px,1fr))","gap":"10px","marginBottom":"14px"}),

    dcc.Tabs([
        dcc.Tab(label="Q1 Platform Differences", value="tab-q1"),
        dcc.Tab(label="Q2 When to Post", value="tab-q2"),
        dcc.Tab(label="Q3 Length & Retention", value="tab-q3"),
        dcc.Tab(label="Q4 Topics & Hashtags", value="tab-q4"),
        dcc.Tab(label="Q5 Creators & Trend Cycle", value="tab-q5"),
    ], id="tabs", value="tab-q1", colors={"border":"#e5e7eb","primary":"#334155","background":"#f8fafc"}),

    html.Div(id="tab-body", style={"marginTop":"14px"}),

    html.Div("Tip: use the global filters above to subset the dataset; heavy scatter plots auto-subsample.", 
             style={"color":"#6b7280","fontSize":"12px","marginTop":"10px"})
], style={"maxWidth":"1180px","margin":"0 auto","padding":"14px"})

def _apply_filters(df0, plats, countries_sel, cats, dur_rng, hour_rng):
    dff = df0.copy()
    if plats:
        dff = dff[dff["platform"].isin(plats)]
    if countries_sel and "country" in dff:
        dff = dff[dff["country"].isin(countries_sel)]
    if cats and "category" in dff:
        dff = dff[dff["category"].isin(cats)]
    if dur_rng and "duration_sec" in dff:
        lo, hi = dur_rng
        dff = dff[(dff["duration_sec"] >= lo) & (dff["duration_sec"] <= hi)]
    if hour_rng and "upload_hour" in dff:
        lo, hi = hour_rng
        dff = dff[(dff["upload_hour"] >= lo) & (dff["upload_hour"] <= hi)]
    return dff

def _kpi_value(x):
    if pd.isna(x): 
        return "-"
    if x>=1e6: return f"{x/1e6:.1f}M"
    if x>=1e3: return f"{x/1e3:.1f}k"
    return f"{int(x)}"

@app.callback(
    Output("kpi-row","children"),
    Input("f-platform","value"), Input("f-country","value"), Input("f-category","value"),
    Input("f-duration","value"), Input("f-hour","value"),
)
def update_kpis(plats, countries_sel, cats, dur_rng, hour_rng):
    dff = _apply_filters(df, plats, countries_sel, cats, dur_rng, hour_rng)
    total = len(dff)
    med_er = dff["engagement_rate"].median() if "engagement_rate" in dff else np.nan
    med_dur = dff["duration_sec"].median() if "duration_sec" in dff else np.nan
    med_comp = dff["completion_rate"].median() if "completion_rate" in dff else np.nan
    return [
        kpi_card("Rows", _kpi_value(total), "After filters"),
        kpi_card("Median engagement", f"{med_er:.1%}" if pd.notna(med_er) else "-", ""),
        kpi_card("Median duration", f"{med_dur:.0f}s" if pd.notna(med_dur) else "-", ""),
        kpi_card("Median completion", f"{med_comp:.1%}" if pd.notna(med_comp) else "-", ""),
    ]

@app.callback(
    Output("tab-body","children"),
    Input("tabs","value"),
    Input("f-platform","value"), Input("f-country","value"), Input("f-category","value"),
    Input("f-duration","value"), Input("f-hour","value"),
)
def render_tab(tab, plats, countries_sel, cats, dur_rng, hour_rng):
    dff = _apply_filters(df, plats, countries_sel, cats, dur_rng, hour_rng)

    if tab == "tab-q1":
        figs = []
        if {"platform","engagement_rate"}.issubset(dff.columns):
            fig1 = px.box(dff, x="platform", y="engagement_rate", color="platform",
                          title="Q1-1 Engagement Rate by Platform",
                          points=False)
            fig1.update_yaxes(tickformat=".0%")
            figs.append(dcc.Graph(figure=fig1, config={"displayModeBar":False}))

            long_cols = ["engagement_rate","engagement_like_rate","engagement_comment_rate","engagement_share_rate"]
            present = [c for c in long_cols if c in dff.columns]
            if present:
                mdf = dff.melt(id_vars=["platform"], value_vars=present, var_name="metric", value_name="value")
                fig2 = px.box(mdf, x="metric", y="value", color="platform",
                              title="Q1-2 Engagement Structure by Platform", points=False)
                fig2.update_yaxes(tickformat=".0%")
                figs.append(dcc.Graph(figure=fig2, config={"displayModeBar":False}))

        if {"views","likes","platform"}.issubset(dff.columns):
            sample_n = min(15000, len(dff))
            sdf = dff.sample(sample_n, random_state=42) if len(dff) > sample_n else dff
            fig3 = px.scatter(sdf, x="views", y="likes", color="platform",
                              title="Q1-3 Views vs Likes (log-log)",
                              opacity=0.6)
            fig3.update_xaxes(type="log", tickformat="~s")
            fig3.update_yaxes(type="log", tickformat="~s")
            figs.append(dcc.Graph(figure=fig3, config={"displayModeBar":False}))
        return figs

    if tab == "tab-q2":
        figs = []
        if {"upload_hour","publish_dayofweek","engagement_rate"}.issubset(dff.columns):
            fig21 = px.density_heatmap(dff, x="upload_hour", y="publish_dayofweek",
                                       z="engagement_rate", histfunc="median",
                                       nbinsx=24, nbinsy=7, color_continuous_scale="Viridis",
                                       title="Q2-1 Median Engagement — Hour × Weekday")
            fig21.update_coloraxes(colorbar_title="Median", colorbar_tickformat=".1%")
            figs.append(dcc.Graph(figure=fig21, config={"displayModeBar":False}))

        if {"weekend","engagement_rate"}.issubset(dff.columns):
            fig22 = px.box(dff, x="weekend", y="engagement_rate", points=False,
                           title="Q2-2 Engagement Rate — Weekend vs Weekday")
            fig22.update_yaxes(tickformat=".0%")
            figs.append(dcc.Graph(figure=fig22, config={"displayModeBar":False}))
        return figs

    if tab == "tab-q3":
        figs = []
        if {"duration_sec","completion_rate"}.issubset(dff.columns):
            fig31 = px.density_heatmap(dff, x="duration_sec", y="completion_rate",
                                       nbinsx=60, nbinsy=40, color_continuous_scale="Viridis",
                                       title="Q3-1 Duration vs Completion Rate (Density)")
            fig31.update_yaxes(tickformat=".0%")
            figs.append(dcc.Graph(figure=fig31, config={"displayModeBar":False}))

        if {"duration_sec","avg_watch_time_sec","platform"}.issubset(dff.columns):
            sample_n = min(15000, len(dff))
            sdf = dff.sample(sample_n, random_state=42) if len(dff) > sample_n else dff
            fig32 = px.scatter(sdf, x="duration_sec", y="avg_watch_time_sec", color="platform",
                               opacity=0.5, title="Q3-2 Duration vs Avg Watch Time (by Platform)")
            figs.append(dcc.Graph(figure=fig32, config={"displayModeBar":False}))
        return figs

    if tab == "tab-q4":
        figs = []
        if {"category","engagement_rate"}.issubset(dff.columns):
            g = (dff.groupby("category", as_index=False)
                    .agg(median_eng=("engagement_rate","median"), N=("engagement_rate","size"))
                    .sort_values("median_eng", ascending=False)
                    .head(20))
            fig41 = px.bar(g, x="median_eng", y="category", orientation="h",
                           text=g["N"], title="Q4-1 Top-20 Categories — Median Engagement (N shown on bars)")
            fig41.update_xaxes(tickformat=".1%")
            fig41.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
            figs.append(dcc.Graph(figure=fig41, config={"displayModeBar":False}))

        if {"hashtag","share_rate"}.issubset(dff.columns):
            h = (dff.groupby("hashtag", as_index=False)
                    .agg(median_share=("share_rate","median"), N=("share_rate","size"))
                    .sort_values("median_share", ascending=False)
                    .head(30))
            fig42 = px.bar(h, x="median_share", y="hashtag", orientation="h",
                           text=h["N"], title="Q4-2 Top-30 Hashtags — Median Share Rate (differences are small)")
            fig42.update_xaxes(tickformat=".2%")
            fig42.update_traces(texttemplate="N=%{text}", textposition="outside", cliponaxis=False)
            figs.append(dcc.Graph(figure=fig42, config={"displayModeBar":False}))
        return figs

    if tab == "tab-q5":
        figs = []
        if {"creator_tier","engagement_rate"}.issubset(dff.columns):
            fig51 = px.box(dff, x="creator_tier", y="engagement_rate", points=False,
                           title="Q5-1 Engagement Rate by Creator Tier")
            fig51.update_yaxes(tickformat=".0%")
            figs.append(dcc.Graph(figure=fig51, config={"displayModeBar":False}))

        need = {"trend_duration_days","engagement_velocity"}
        if need.issubset(dff.columns):
            sample_n = min(12000, len(dff))
            sdf = dff.sample(sample_n, random_state=42) if len(dff) > sample_n else dff
            color = "platform" if "platform" in sdf.columns else None
            size = "views" if "views" in sdf.columns else None
            fig52 = px.scatter(sdf, x="trend_duration_days", y="engagement_velocity",
                               color=color, size=size, size_max=18, opacity=0.6,
                               title="Q5-2 Trend Duration vs Engagement Velocity (log y; size=views)")
            fig52.update_yaxes(type="log", tickformat="~s")
            figs.append(dcc.Graph(figure=fig52, config={"displayModeBar":False}))
        return figs

    return html.Div("Tab not found.")

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
