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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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
# Tab 2: Adult Health & Activity (Kaggle "HYPERAKTIV" upload-driven)
# ---------------------------------------------------------------------------
# Curated, human-readable shortlists -- the raw files have 780+ auto-generated
# feature columns (tsfresh) and 360 raw CPT trial columns; we surface the
# interpretable summary columns instead of dumping everything in a dropdown.
ACTIVITY_FEATURE_CANDIDATES = [
    "ACC__mean", "ACC__standard_deviation", "ACC__maximum", "ACC__minimum",
    "ACC__sum_values", "ACC__abs_energy", "ACC__median", "ACC__variance",
]
CPT_FEATURE_CANDIDATES = [
    "Adhd TScore Omissions", "Adhd TScore Commissions", "Adhd TScore HitRT",
    "Adhd TScore VarSE", "Adhd TScore DPrime", "Adhd Confidence Index",
]
CLINICAL_SCALE_CANDIDATES = ["WURS", "ASRS", "MADRS", "HADS_A", "HADS_D"]


def render_health_tab():
    st.subheader("Adult Health & Activity")
    st.caption(
        "Dataset: Kaggle - 'ADHD Diagnosis Data' (the HYPERAKTIV dataset -- clinical info, "
        "wrist activity features, and CPT-II attention-test scores for adults with ADHD and controls)."
    )
    st.write(
        "Download from [Kaggle](https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data) "
        "(free account, no API key needed) and upload the files below. `patient_info.csv` is "
        "required -- it has the ADHD / control label. `features.csv` and the CPT CSV are optional "
        "extras that unlock more metrics to compare. Skip the `activity_data/`, `hrv_data/`, and "
        "`hyperaktiv_with_controls/` folders -- those hold raw per-second sensor data (500MB+) that "
        "this tab doesn't need."
    )

    col1, col2, col3 = st.columns(3)
    pi_file = col1.file_uploader("patient_info.csv (required)", type="csv", key="pi_upload")
    feat_file = col2.file_uploader("features.csv (optional)", type="csv", key="feat_upload")
    cpt_file = col3.file_uploader("CPT CSV (optional)", type="csv", key="cpt_upload")

    if pi_file is None:
        st.info("Upload at least `patient_info.csv` to get started -- it's small (under 1MB) and "
                 "holds the ADHD/control label plus clinical scale scores (WURS, ASRS, MADRS, HADS).")
        return

    pi = pd.read_csv(pi_file, sep=";")
    if "ADHD" not in pi.columns or "ID" not in pi.columns:
        st.error("This doesn't look like patient_info.csv -- expected an 'ID' and 'ADHD' column.")
        return
    pi["Group"] = pi["ADHD"].map({1: "ADHD", 0: "Control"})

    merged = pi.copy()
    metric_options = {}  # display label -> column name
    for c in CLINICAL_SCALE_CANDIDATES:
        if c in merged.columns:
            metric_options[f"Clinical scale: {c}"] = c

    if feat_file is not None:
        feat = pd.read_csv(feat_file, sep=";")
        present = [c for c in ACTIVITY_FEATURE_CANDIDATES if c in feat.columns]
        if present:
            merged = merged.merge(feat[["ID"] + present], on="ID", how="inner")
            for c in present:
                metric_options[f"Wrist activity: {c.replace('ACC__', '')}"] = c
        else:
            st.warning("features.csv was uploaded but none of the expected ACC__ columns were found.")

    if cpt_file is not None:
        cpt = pd.read_csv(cpt_file, sep=";")
        present = [c for c in CPT_FEATURE_CANDIDATES if c in cpt.columns]
        if present:
            merged = merged.merge(cpt[["ID"] + present], on="ID", how="inner")
            for c in present:
                metric_options[f"Attention test (CPT-II): {c}"] = c
        else:
            st.warning("The CPT CSV was uploaded but none of the expected summary-score columns were found.")

    n_adhd = (merged["Group"] == "ADHD").sum()
    n_control = (merged["Group"] == "Control").sum()
    c1, c2 = st.columns(2)
    c1.metric("ADHD patients (matched across uploaded files)", n_adhd)
    c2.metric("Controls (matched across uploaded files)", n_control)

    with st.expander("Preview merged data"):
        st.dataframe(merged.head(50), use_container_width=True)

    st.session_state["health_merged"] = merged
    st.session_state["health_metric_options"] = metric_options

    if not metric_options:
        st.warning("No known metric columns found -- double check you uploaded the right files.")
        return

    label = st.selectbox("Metric to compare", list(metric_options.keys()))
    metric_col = metric_options[label]

    fig = go.Figure()
    for i, g in enumerate(["ADHD", "Control"]):
        vals = merged.loc[merged["Group"] == g, metric_col].dropna()
        fig.add_trace(go.Box(y=vals, name=g, marker_color=CATEGORICAL[i], boxmean=True))
    fig.update_layout(title=f"{label} -- ADHD vs. Control", **BASE_LAYOUT)
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

    # Raw EEG data is too large to hand an LLM directly -- keep only lightweight
    # metadata for the chat tab (row/column counts, channel names).
    st.session_state["eeg_info"] = {"rows": len(df), "columns": df.columns.tolist()}

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
# Tab 4: Chat with the Data
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are a data assistant embedded in an ADHD Data Explorer app.
Answer the user's question using ONLY the data provided below -- do not use outside
knowledge about ADHD, and do not make clinical or diagnostic claims. If the data
provided doesn't contain the answer, say so plainly instead of guessing.
Cite specific numbers from the data when you can. Keep answers concise (a few
sentences, or a short list).

DATA AVAILABLE:
{context}
"""


def build_data_context() -> str:
    parts = []

    prevalence = load_prevalence()
    parts.append("## National ADHD prevalence & demographics (CDC/NSCH 2022-2023)\n" + prevalence.to_markdown(index=False))

    treatment = load_treatment_range()
    parts.append("## Treatment rate range across US states (2022)\n" + treatment.to_markdown(index=False))

    merged = st.session_state.get("health_merged")
    if merged is not None:
        cols = ["ID", "Group"] + [c for c in st.session_state.get("health_metric_options", {}).values()]
        cols = [c for c in dict.fromkeys(cols) if c in merged.columns]
        parts.append(
            "## Adult health/activity data (HYPERAKTIV, uploaded by user)\n"
            f"{len(merged)} matched patients. Group counts: "
            f"{merged['Group'].value_counts().to_dict()}\n\n"
            + merged[cols].to_markdown(index=False)
        )
    else:
        parts.append("## Adult health/activity data\nNot uploaded yet in the Adult Health & Activity tab.")

    eeg_info = st.session_state.get("eeg_info")
    if eeg_info is not None:
        parts.append(
            "## EEG data\n"
            f"An EEG file was uploaded with {eeg_info['rows']} rows and columns: "
            f"{eeg_info['columns']}. (Raw signal values are not included here -- too large.)"
        )
    else:
        parts.append("## EEG data\nNot uploaded yet in the EEG Signals tab.")

    return "\n\n".join(parts)


def render_chat_tab():
    st.subheader("Chat with the Data")
    st.caption("Ask questions in plain English -- the assistant answers using the data currently loaded in this app.")

    if not ANTHROPIC_AVAILABLE:
        st.error("The `anthropic` package isn't installed. Add `anthropic` to requirements.txt and redeploy.")
        return

    api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        st.warning(
            "No Anthropic API key found. Get a free key at "
            "[console.anthropic.com](https://console.anthropic.com), then add it as a secret:\n\n"
            "- **Streamlit Community Cloud:** app dashboard -> Settings -> Secrets, add:\n"
            "  ```\n  ANTHROPIC_API_KEY = \"sk-ant-...\"\n  ```\n"
            "- **Running locally:** create `.streamlit/secrets.toml` with the same line "
            "(and add it to `.gitignore` so it never gets committed)."
        )
        return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    col1, col2 = st.columns([5, 1])
    col2.button("Clear chat", on_click=lambda: st.session_state.update(chat_messages=[]))

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. Which treatment type varies most across states?")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = build_data_context()
    system = CHAT_SYSTEM_PROMPT.format(context=context)

    # Only send recent turns to keep token usage (and cost) bounded.
    recent = st.session_state.chat_messages[-10:]

    with st.chat_message("assistant"):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=600,
                system=system,
                messages=[{"role": m["role"], "content": m["content"]} for m in recent],
            )
            answer = response.content[0].text
        except Exception as e:
            answer = f"Something went wrong calling the AI: {e}"
        st.markdown(answer)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ADHD Data Explorer", layout="wide")
st.title("ADHD Data Explorer")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Prevalence & Demographics", "Adult Health & Activity", "EEG Signals", "Chat with the Data"]
)
with tab1:
    render_prevalence_tab()
with tab2:
    render_health_tab()
with tab3:
    render_eeg_tab()
with tab4:
    render_chat_tab()

st.divider()
st.caption(
    "This app is for exploratory / educational purposes only and is not a diagnostic tool. "
    "Prevalence figures: CDC / NSCH 2022-2023."
)
