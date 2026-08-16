"""
ADHD Data Explorer
===================
A single Streamlit app with three tabs:
  1. Prevalence & Demographics  -- CDC / NSCH national data (built-in, no download needed)
  2. Adult Health & Activity    -- Kaggle "ADHD Diagnosis Data" (heart rate / activity)
  3. EEG Signals                -- Kaggle "EEG Dataset for ADHD"

Tabs 2 and 3 use a file uploader: download the CSVs from Kaggle (see README.md
for links + instructions), then drop them in here or upload through the UI.

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette (validated categorical order -- see dataviz skill reference)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

CHART_SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

BASE_LAYOUT = dict(
    plot_bgcolor=CHART_SURFACE,
    paper_bgcolor=CHART_SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=PRIMARY_INK),
    margin=dict(l=10, r=10, t=40, b=10),
)


def style_axes(fig, x_grid=False, y_grid=True):
    fig.update_xaxes(showgrid=x_grid, gridcolor=GRIDLINE, zeroline=False, showline=True, linecolor=GRIDLINE)
    fig.update_yaxes(showgrid=y_grid, gridcolor=GRIDLINE, zeroline=False, showline=False, tickfont=dict(color=MUTED_INK))
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_prevalence():
    return pd.read_csv("data/prevalence_national.csv")


@st.cache_data
def load_treatment_range():
    return pd.read_csv("data/treatment_state_range.csv")


# ---------------------------------------------------------------------------
# Tab 1: Prevalence & Demographics
# ---------------------------------------------------------------------------
def render_prevalence_tab():
    st.subheader("ADHD Prevalence & Demographics")
    st.caption("Source: CDC / National Survey of Children's Health (NSCH), 2022-2023, U.S. children ages 3-17.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Children currently diagnosed", "11.7%", help="~7 million U.S. children ages 3-17 (2022-2023 NSCH)")
    c2.metric("Have a co-occurring condition", "78%", help="Among children with a current ADHD diagnosis")
    c3.metric("Moderate or severe cases", "60%", help="Among children with a current ADHD diagnosis")

    st.divider()
    prevalence = load_prevalence()

    def single_series_bar(df, title, x_col="value_pct", y_col="subgroup", suffix="%"):
        d = df.sort_values(x_col, ascending=True)
        fig = go.Figure(
            go.Bar(
                x=d[x_col], y=d[y_col], orientation="h",
                marker=dict(color=BLUE),
                text=[f"{v:g}{suffix}" for v in d[x_col]],
                textposition="outside",
                hovertemplate="%{y}: %{x:g}" + suffix + "<extra></extra>",
            )
        )
        fig.update_layout(title=title, height=280, **BASE_LAYOUT)
        style_axes(fig)
        fig.update_xaxes(title=None, ticksuffix="%")
        fig.update_yaxes(title=None)
        return fig

    col1, col2 = st.columns(2)
    with col1:
        d = prevalence[prevalence.category == "sex"]
        st.plotly_chart(single_series_bar(d, "Diagnosis rate by sex"), use_container_width=True)
    with col2:
        d = prevalence[prevalence.category == "severity"]
        st.plotly_chart(single_series_bar(d, "Severity among diagnosed children"), use_container_width=True)

    d = prevalence[prevalence.category == "race_ethnicity"]
    st.plotly_chart(single_series_bar(d, "Diagnosis rate by race / ethnicity"), use_container_width=True)

    d = prevalence[prevalence.category == "co_occurring"]
    st.plotly_chart(single_series_bar(d, "Co-occurring conditions among diagnosed children"), use_container_width=True)

    st.divider()
    st.markdown("**Treatment received -- how much it varies state to state**")
    tr = load_treatment_range()
    fig = go.Figure()
    for i, row in tr.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.low_pct, row.high_pct], y=[row.treatment_type, row.treatment_type],
                mode="lines+markers",
                line=dict(color=GRIDLINE, width=4),
                marker=dict(size=12, color=[ORANGE, BLUE]),
                hovertemplate="%{x:g}%<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(title="Treatment rate range across states (2022)", height=260, **BASE_LAYOUT)
    style_axes(fig)
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange marker = lowest-treating state's rate, blue marker = highest-treating state's rate, for that treatment type.")

    with st.expander("View underlying data"):
        st.dataframe(prevalence, use_container_width=True)
        st.dataframe(tr, use_container_width=True)

    st.divider()
    st.markdown("**Want a state-by-state map?**")
    st.write(
        "This app ships with national + demographic figures only, since a full 50-state "
        "dataset requires either the free NSCH data query tool "
        "([childhealthdata.org](https://www.childhealthdata.org)) or the "
        "[America's Health Rankings API](https://developers.americashealthrankings.org/) "
        "(free signup). If you get a CSV with `state` and `rate` columns, upload it below "
        "and this tab will render a choropleth map."
    )
    state_file = st.file_uploader("Upload state-level CSV (columns: state, rate)", type="csv", key="state_upload")
    if state_file is not None:
        state_df = pd.read_csv(state_file)
        fig = go.Figure(
            go.Choropleth(
                locations=state_df["state"], locationmode="USA-states", z=state_df["rate"],
                colorscale=[[0, "#cde2fb"], [1, "#0d366b"]], colorbar_title="Rate (%)",
            )
        )
        fig.update_layout(geo_scope="usa", title="ADHD rate by state", **BASE_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Adult Health & Activity (Kaggle upload-driven)
# ---------------------------------------------------------------------------
def render_health_tab():
    st.subheader("Adult Health & Activity")
    st.caption("Dataset: Kaggle - 'ADHD Diagnosis Data' (health, activity, and heart-rate data from adults).")

    st.write(
        "This dataset isn't bundled with the app -- download it from "
        "[Kaggle](https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data) "
        "(free account required, no API key needed for a manual download) and upload the CSV below."
    )
    f = st.file_uploader("Upload the ADHD health/activity CSV", type="csv", key="health_upload")
    if f is None:
        st.info("Once uploaded, you'll be able to pick a group column (e.g. ADHD vs. control) and compare "
                "heart rate / activity distributions and trends over time.")
        return

    df = pd.read_csv(f)
    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    with st.expander("Preview data"):
        st.dataframe(df.head(50), use_container_width=True)

    numeric_cols = df.select_dtypes("number").columns.tolist()
    all_cols = df.columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns detected -- can't chart this file automatically.")
        return

    col1, col2 = st.columns(2)
    metric_col = col1.selectbox("Metric to visualize", numeric_cols)
    group_col = col2.selectbox("Group by (e.g. ADHD vs. control)", ["(none)"] + all_cols)

    if group_col != "(none)" and df[group_col].nunique() <= 8:
        groups = df[group_col].dropna().unique().tolist()
        fig = go.Figure()
        for i, g in enumerate(groups):
            fig.add_trace(go.Box(y=df.loc[df[group_col] == g, metric_col], name=str(g),
                                  marker_color=CATEGORICAL[i % len(CATEGORICAL)]))
        fig.update_layout(title=f"{metric_col} by {group_col}", **BASE_LAYOUT)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure(go.Histogram(x=df[metric_col], marker_color=BLUE))
        fig.update_layout(title=f"Distribution of {metric_col}", **BASE_LAYOUT)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)

    time_cols = [c for c in all_cols if "time" in c.lower() or "date" in c.lower()]
    if time_cols:
        t_col = st.selectbox("Time column for a trend line", time_cols)
        d = df[[t_col, metric_col]].dropna().sort_values(t_col)
        fig = go.Figure(go.Scatter(x=d[t_col], y=d[metric_col], mode="lines", line=dict(color=BLUE, width=2)))
        fig.update_layout(title=f"{metric_col} over time", **BASE_LAYOUT)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: EEG Signals (Kaggle upload-driven)
# ---------------------------------------------------------------------------
def render_eeg_tab():
    st.subheader("EEG Signals")
    st.caption("Dataset: Kaggle - 'EEG Dataset for ADHD' (raw EEG channel data, ADHD vs. control).")

    st.write(
        "Download from [Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd) "
        "and upload a CSV below. This tab expects one column per EEG channel (numeric), "
        "and optionally a label/group column."
    )
    f = st.file_uploader("Upload an EEG CSV", type="csv", key="eeg_upload")
    if f is None:
        st.info("Once uploaded, you'll be able to plot a channel's raw waveform and, if a group/label "
                "column exists, compare signal characteristics between ADHD and control subjects.")
        return

    df = pd.read_csv(f)
    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    with st.expander("Preview data"):
        st.dataframe(df.head(50), use_container_width=True)

    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        st.warning("No numeric channel columns detected.")
        return

    channel = st.selectbox("Channel to plot", numeric_cols)
    n_samples = st.slider("Samples to display", 100, min(5000, len(df)), min(1000, len(df)))
    d = df[channel].iloc[:n_samples]
    fig = go.Figure(go.Scatter(y=d, mode="lines", line=dict(color=BLUE, width=1)))
    fig.update_layout(title=f"{channel} - raw waveform (first {n_samples} samples)", **BASE_LAYOUT)
    style_axes(fig)
    st.plotly_chart(fig, use_container_width=True)

    group_col = st.selectbox("Group / label column (optional)", ["(none)"] + df.columns.tolist())
    if group_col != "(none)" and df[group_col].nunique() <= 8:
        groups = df[group_col].dropna().unique().tolist()
        fig = go.Figure()
        for i, g in enumerate(groups):
            fig.add_trace(go.Box(y=df.loc[df[group_col] == g, channel], name=str(g),
                                  marker_color=CATEGORICAL[i % len(CATEGORICAL)]))
        fig.update_layout(title=f"{channel} distribution by {group_col}", **BASE_LAYOUT)
        style_axes(fig)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ADHD Data Explorer", layout="wide")
st.title("ADHD Data Explorer")

tab1, tab2, tab3 = st.tabs(["Prevalence & Demographics", "Adult Health & Activity", "EEG Signals"])
with tab1:
    render_prevalence_tab()
with tab2:
    render_health_tab()
with tab3:
    render_eeg_tab()

st.divider()
st.caption(
    "This app is for exploratory / educational purposes only and is not a diagnostic tool. "
    "Prevalence figures: CDC / NSCH 2022-2023."
)
