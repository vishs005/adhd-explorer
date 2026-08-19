"""
ADHD Data Explorer
===================
A single Streamlit app with four tabs:
  1. Prevalence & Demographics  -- CDC / NSCH national data (built-in, no download needed)
  2. Adult Health & Activity    -- Kaggle "ADHD Diagnosis Data" (HYPERAKTIV), or synthetic sample data
  3. EEG Signals                -- Kaggle "EEG Dataset for ADHD", or synthetic sample data
  4. Chat with the Data         -- Claude-powered chat that answers from whatever's loaded, and can draw charts

Run with:
    streamlit run app.py
"""

import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Theme -- light/dark palettes, both validated colorblind-safe (see dataviz
# skill reference). Charts are re-themed on every rerun based on the sidebar
# toggle, rather than using fixed module-level colors.
# ---------------------------------------------------------------------------
PALETTES = {
    "light": dict(
        BLUE="#2a78d6", ORANGE="#eb6834", AQUA="#1baf7a", YELLOW="#eda100",
        MAGENTA="#e87ba4", GREEN="#008300", VIOLET="#4a3aa7", RED="#e34948",
        SURFACE="#fcfcfb", GRIDLINE="#e1e0d9", MUTED_INK="#898781",
        PRIMARY_INK="#0b0b0b", SECONDARY_INK="#52514e",
    ),
    "dark": dict(
        BLUE="#3987e5", ORANGE="#d95926", AQUA="#199e70", YELLOW="#c98500",
        MAGENTA="#d55181", GREEN="#008300", VIOLET="#9085e9", RED="#e66767",
        SURFACE="#1a1a19", GRIDLINE="#2c2c2a", MUTED_INK="#898781",
        PRIMARY_INK="#ffffff", SECONDARY_INK="#c3c2b7",
    ),
}


def get_theme():
    mode = "dark" if st.session_state.get("dark_mode", False) else "light"
    p = PALETTES[mode]
    categorical = [p["BLUE"], p["ORANGE"], p["AQUA"], p["YELLOW"],
                   p["MAGENTA"], p["GREEN"], p["VIOLET"], p["RED"]]
    base_layout = dict(
        plot_bgcolor=p["SURFACE"], paper_bgcolor=p["SURFACE"],
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=p["PRIMARY_INK"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return {**p, "CATEGORICAL": categorical, "BASE_LAYOUT": base_layout}


def style_axes(fig, theme, x_grid=False, y_grid=True):
    fig.update_xaxes(showgrid=x_grid, gridcolor=theme["GRIDLINE"], zeroline=False,
                      showline=True, linecolor=theme["GRIDLINE"])
    fig.update_yaxes(showgrid=y_grid, gridcolor=theme["GRIDLINE"], zeroline=False,
                      showline=False, tickfont=dict(color=theme["MUTED_INK"]))
    return fig


# ---------------------------------------------------------------------------
# Custom UI polish -- CSS injected once per rerun, theme-aware. Purely
# cosmetic: targets Streamlit's public data-testid hooks so app logic and
# element keys are untouched if a selector ever stops matching.
# ---------------------------------------------------------------------------
def custom_css(theme):
    is_dark = theme["SURFACE"] == PALETTES["dark"]["SURFACE"]
    card_shadow = (
        "0 1px 3px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.4)"
        if is_dark else
        "0 1px 3px rgba(20,20,15,0.08), 0 1px 2px rgba(20,20,15,0.06)"
    )
    glow = f"0 0 0 1px rgba(42,120,214,0.25), 0 10px 30px rgba(42,120,214,0.35)" if not is_dark else \
           f"0 0 0 1px rgba(57,135,229,0.35), 0 10px 30px rgba(57,135,229,0.45)"
    card_bg = "rgba(35,35,34,0.72)" if is_dark else "rgba(255,255,255,0.78)"
    mesh_opacity = "0.55" if is_dark else "0.35"
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }}

    /* Subtle animated gradient mesh drifting behind the content */
    .main .block-container {{
        animation: adhd-fade-in 0.45s ease-out;
        padding-top: 1.4rem;
        position: relative;
    }}
    .main .block-container::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        opacity: {mesh_opacity};
        background:
            radial-gradient(circle at 12% 8%, {theme['BLUE']}33 0%, transparent 38%),
            radial-gradient(circle at 88% 15%, {theme['VIOLET']}33 0%, transparent 40%),
            radial-gradient(circle at 50% 95%, {theme['AQUA']}22 0%, transparent 45%);
        background-size: 180% 180%;
        animation: adhd-mesh-drift 22s ease-in-out infinite;
        pointer-events: none;
    }}
    @keyframes adhd-mesh-drift {{
        0%   {{ background-position: 0% 0%; }}
        50%  {{ background-position: 100% 60%; }}
        100% {{ background-position: 0% 0%; }}
    }}
    @keyframes adhd-fade-in {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes adhd-pop-in {{
        from {{ opacity: 0; transform: scale(0.92) translateY(8px); }}
        to {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    @keyframes adhd-float {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-6px) rotate(-4deg); }}
    }}
    @keyframes adhd-shimmer {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Hero banner -- animated moving gradient */
    .adhd-hero {{
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 30px 34px;
        border-radius: 20px;
        margin-bottom: 6px;
        background: linear-gradient(120deg, {theme['BLUE']} 0%, {theme['VIOLET']} 45%, {theme['AQUA']} 80%, {theme['BLUE']} 100%);
        background-size: 300% 300%;
        animation: adhd-shimmer 9s ease-in-out infinite;
        box-shadow: {glow};
        position: relative;
        overflow: hidden;
    }}
    .adhd-hero-icon {{
        font-size: 48px;
        line-height: 1;
        animation: adhd-float 4s ease-in-out infinite;
        filter: drop-shadow(0 4px 10px rgba(0,0,0,0.25));
    }}
    .adhd-hero-title {{
        color: #ffffff;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin: 0;
        text-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }}
    .adhd-hero-sub {{
        color: rgba(255,255,255,0.92);
        font-size: 15.5px;
        font-weight: 500;
        margin-top: 5px;
    }}
    .adhd-badge-row {{
        display: flex; flex-wrap: wrap; gap: 8px;
        margin: 16px 0 4px 0;
    }}
    .adhd-badge {{
        display: inline-block;
        padding: 6px 15px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 700;
        background: {card_bg};
        backdrop-filter: blur(10px);
        color: {theme['SECONDARY_INK']};
        border: 1px solid {theme['GRIDLINE']};
        box-shadow: {card_shadow};
        opacity: 0;
        animation: adhd-pop-in 0.4s ease-out forwards;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .adhd-badge:hover {{
        transform: translateY(-2px) scale(1.04);
        box-shadow: {glow};
    }}
    .adhd-badge-row .adhd-badge:nth-child(1) {{ animation-delay: 0.05s; }}
    .adhd-badge-row .adhd-badge:nth-child(2) {{ animation-delay: 0.15s; }}
    .adhd-badge-row .adhd-badge:nth-child(3) {{ animation-delay: 0.25s; }}
    .adhd-badge-row .adhd-badge:nth-child(4) {{ animation-delay: 0.35s; }}

    /* Section headings get a gradient accent bar */
    .main h2, .main h3 {{
        font-weight: 800 !important;
        letter-spacing: -0.01em;
        padding-left: 14px;
        border-left: 5px solid transparent;
        border-image: linear-gradient(180deg, {theme['BLUE']}, {theme['VIOLET']}) 1;
    }}

    /* Dividers -> soft gradient line instead of a flat rule */
    .main hr {{
        border: none;
        height: 3px;
        border-radius: 3px;
        background: linear-gradient(90deg, {theme['BLUE']}, {theme['VIOLET']}, transparent);
        opacity: 0.7;
        margin: 1.2rem 0;
    }}

    /* Tabs -> glowing pill style */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {card_bg};
        backdrop-filter: blur(10px);
        padding: 7px;
        border-radius: 16px;
        border: 1px solid {theme['GRIDLINE']};
        box-shadow: {card_shadow};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 11px;
        padding: 9px 20px;
        font-weight: 700;
        color: {theme['SECONDARY_INK']};
        transition: all 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: {theme['GRIDLINE']}55;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {theme['BLUE']}, {theme['VIOLET']}) !important;
        color: #ffffff !important;
        box-shadow: {glow};
    }}

    /* Metric cards -- glassy with hover glow + staggered pop-in */
    [data-testid="stMetric"] {{
        background: {card_bg};
        backdrop-filter: blur(10px);
        border: 1px solid {theme['GRIDLINE']};
        border-top: 3px solid {theme['BLUE']};
        border-radius: 14px;
        padding: 16px 18px 12px 18px;
        box-shadow: {card_shadow};
        opacity: 0;
        animation: adhd-pop-in 0.45s ease-out forwards;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        box-shadow: {glow};
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 800 !important;
        background: linear-gradient(120deg, {theme['BLUE']}, {theme['VIOLET']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 11px;
        border: 1px solid {theme['BLUE']};
        font-weight: 700;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px) scale(1.01);
        box-shadow: {glow};
        border-color: {theme['VIOLET']};
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        border-radius: 14px;
        border: 1px solid {theme['GRIDLINE']};
        box-shadow: {card_shadow};
        overflow: hidden;
        transition: box-shadow 0.15s ease;
    }}
    [data-testid="stExpander"]:hover {{
        box-shadow: {glow};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        border-right: 1px solid {theme['GRIDLINE']};
        background: linear-gradient(180deg, {theme['BLUE']}0d 0%, transparent 25%);
    }}

    /* Chat bubbles */
    [data-testid="stChatMessage"] {{
        border-radius: 16px;
        border: 1px solid {theme['GRIDLINE']};
        box-shadow: {card_shadow};
        transition: box-shadow 0.15s ease;
    }}
    [data-testid="stChatMessage"]:hover {{
        box-shadow: {glow};
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {theme['BLUE']}, {theme['VIOLET']});
        border-radius: 8px;
    }}

    footer, #MainMenu {{ visibility: hidden; }}
    </style>
    """


def render_hero():
    st.markdown(
        """
        <div class="adhd-hero">
            <div class="adhd-hero-icon">🧠</div>
            <div>
                <p class="adhd-hero-title">ADHD Data Explorer</p>
                <p class="adhd-hero-sub">Real CDC prevalence data, a real clinical study, real EEG signals -- and an AI you can chat with.</p>
            </div>
        </div>
        <div class="adhd-badge-row">
            <span class="adhd-badge">📊 CDC / NSCH Data</span>
            <span class="adhd-badge">🩺 Clinical Study (HYPERAKTIV)</span>
            <span class="adhd-badge">🌊 Real EEG Recordings</span>
            <span class="adhd-badge">💬 AI Chat</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def animated_stat_row(theme, stats, height=168):
    """Renders a row of glassy stat cards with count-up numbers via an
    isolated HTML/JS component -- purely decorative, doesn't touch any
    st.session_state or widget the rest of the app depends on."""
    is_dark = theme["SURFACE"] == PALETTES["dark"]["SURFACE"]
    text_color = "#f5f5f4" if is_dark else "#0b0b0b"
    sub_color = theme["MUTED_INK"]
    card_bg = "rgba(255,255,255,0.06)" if is_dark else "rgba(255,255,255,0.9)"
    border_color = theme["GRIDLINE"]

    cards_html = ""
    for i, s in enumerate(stats):
        cards_html += f"""
        <div class="stat-card" style="animation-delay:{i * 0.12}s">
            <div class="stat-icon">{s.get('icon', '📈')}</div>
            <div class="stat-value" data-target="{s['value']}" data-decimals="{s.get('decimals', 0)}" data-suffix="{s.get('suffix', '')}">0{s.get('suffix', '')}</div>
            <div class="stat-label">{s['label']}</div>
            <div class="stat-help">{s.get('help', '')}</div>
        </div>
        """

    html = f"""
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; padding: 4px 0;
            font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
            background: transparent;
        }}
        .stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
        .stat-card {{
            flex: 1; min-width: 160px;
            background: {card_bg};
            border: 1px solid {border_color};
            border-top: 3px solid {theme['BLUE']};
            border-radius: 14px;
            padding: 16px 18px;
            backdrop-filter: blur(8px);
            opacity: 0;
            animation: popIn 0.5s ease-out forwards;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        .stat-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 26px rgba(42,120,214,0.35); }}
        .stat-icon {{ font-size: 22px; margin-bottom: 4px; }}
        .stat-value {{
            font-size: 30px; font-weight: 800;
            background: linear-gradient(120deg, {theme['BLUE']}, {theme['VIOLET']});
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .stat-label {{ font-size: 13.5px; font-weight: 600; color: {text_color}; margin-top: 2px; }}
        .stat-help {{ font-size: 11.5px; color: {sub_color}; margin-top: 3px; }}
        @keyframes popIn {{ from {{ opacity: 0; transform: scale(0.9) translateY(10px); }} to {{ opacity: 1; transform: scale(1) translateY(0); }} }}
    </style>
    </head>
    <body>
        <div class="stat-row">{cards_html}</div>
        <script>
            function countUp(el) {{
                const target = parseFloat(el.dataset.target);
                const decimals = parseInt(el.dataset.decimals, 10);
                const suffix = el.dataset.suffix || '';
                const duration = 1200;
                let start = null;
                function step(ts) {{
                    if (!start) start = ts;
                    const progress = Math.min((ts - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    el.textContent = (target * eased).toFixed(decimals) + suffix;
                    if (progress < 1) requestAnimationFrame(step);
                }}
                requestAnimationFrame(step);
            }}
            document.querySelectorAll('.stat-value').forEach(countUp);
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=height, scrolling=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_prevalence():
    return pd.read_csv("data/prevalence_national.csv")


@st.cache_data
def load_treatment_range():
    return pd.read_csv("data/treatment_state_range.csv")


@st.cache_data
def load_state_treatment():
    return pd.read_csv("data/state_adhd_treatment.csv")


# Sequential single-hue ramp (blue, light -> dark) for choropleth magnitude
# encoding -- per the dataviz skill's palette reference.
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"], [0.17, "#9ec5f4"], [0.33, "#6da7ec"], [0.5, "#3987e5"],
    [0.67, "#256abf"], [0.83, "#184f95"], [1.0, "#0d366b"],
]

PREVALENCE_TITLES = {
    "sex": "Diagnosis rate by sex",
    "race_ethnicity": "Diagnosis rate by race / ethnicity",
    "severity": "Severity among diagnosed children",
    "co_occurring": "Co-occurring conditions among diagnosed children",
}


def state_choropleth_chart(theme, state_df, z_col="rate_pct", label_col="state_name",
                            title="% of children receiving ADD/ADHD treatment, by state",
                            colorbar_title="% of children"):
    fig = go.Figure(
        go.Choropleth(
            locations=state_df["state"], locationmode="USA-states", z=state_df[z_col],
            colorscale=SEQUENTIAL_BLUE, colorbar_title=colorbar_title,
            text=state_df[label_col] if label_col in state_df.columns else None,
            hovertemplate="%{text}: %{z:g}<extra></extra>" if label_col in state_df.columns else "%{location}: %{z:g}<extra></extra>",
            marker_line_color=theme["GRIDLINE"],
        )
    )
    fig.update_layout(
        geo=dict(scope="usa", bgcolor=theme["SURFACE"], lakecolor=theme["SURFACE"]),
        title=title, **theme["BASE_LAYOUT"],
    )
    return fig


def prevalence_bar_chart(theme, df_subset, title, x_col="value_pct", y_col="subgroup", suffix="%"):
    d = df_subset.sort_values(x_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=d[x_col], y=d[y_col], orientation="h",
            marker=dict(color=theme["BLUE"]),
            text=[f"{v:g}{suffix}" for v in d[x_col]],
            textposition="outside",
            hovertemplate="%{y}: %{x:g}" + suffix + "<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=280, **theme["BASE_LAYOUT"])
    style_axes(fig, theme)
    fig.update_xaxes(title=None, ticksuffix="%")
    fig.update_yaxes(title=None)
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Prevalence & Demographics
# ---------------------------------------------------------------------------
def render_prevalence_tab():
    theme = get_theme()
    st.subheader("ADHD Prevalence & Demographics")
    st.caption("Source: CDC / National Survey of Children's Health (NSCH), 2022-2023, U.S. children ages 3-17.")

    animated_stat_row(
        theme,
        [
            {"icon": "🧒", "label": "Children currently diagnosed", "value": 11.7, "decimals": 1, "suffix": "%",
             "help": "~7 million U.S. children ages 3-17 (2022-2023 NSCH)"},
            {"icon": "🔗", "label": "Have a co-occurring condition", "value": 78, "decimals": 0, "suffix": "%",
             "help": "Among children with a current ADHD diagnosis"},
            {"icon": "⚠️", "label": "Moderate or severe cases", "value": 60, "decimals": 0, "suffix": "%",
             "help": "Among children with a current ADHD diagnosis"},
        ],
    )

    st.divider()
    prevalence = load_prevalence()

    col1, col2 = st.columns(2)
    with col1:
        d = prevalence[prevalence.category == "sex"]
        st.plotly_chart(prevalence_bar_chart(theme, d, PREVALENCE_TITLES["sex"]), use_container_width=True, key="prevalence_sex_chart")
    with col2:
        d = prevalence[prevalence.category == "severity"]
        st.plotly_chart(prevalence_bar_chart(theme, d, PREVALENCE_TITLES["severity"]), use_container_width=True, key="prevalence_severity_chart")

    d = prevalence[prevalence.category == "race_ethnicity"]
    st.plotly_chart(prevalence_bar_chart(theme, d, PREVALENCE_TITLES["race_ethnicity"]), use_container_width=True, key="prevalence_race_chart")

    d = prevalence[prevalence.category == "co_occurring"]
    st.plotly_chart(prevalence_bar_chart(theme, d, PREVALENCE_TITLES["co_occurring"]), use_container_width=True, key="prevalence_co_occurring_chart")

    st.divider()
    st.markdown("**Treatment received -- how much it varies state to state**")
    tr = load_treatment_range()
    fig = go.Figure()
    for i, row in tr.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.low_pct, row.high_pct], y=[row.treatment_type, row.treatment_type],
                mode="lines+markers",
                line=dict(color=theme["GRIDLINE"], width=4),
                marker=dict(size=12, color=[theme["ORANGE"], theme["BLUE"]]),
                hovertemplate="%{x:g}%<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(title="Treatment rate range across states (2022)", height=260, **theme["BASE_LAYOUT"])
    style_axes(fig, theme)
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, key="prevalence_treatment_range_chart")
    st.caption("Orange marker = lowest-treating state's rate, blue marker = highest-treating state's rate, for that treatment type.")

    with st.expander("View underlying data"):
        st.dataframe(prevalence, use_container_width=True)
        st.dataframe(tr, use_container_width=True)

    combined_csv = pd.concat([prevalence, tr], axis=0, ignore_index=True, sort=False).to_csv(index=False).encode()
    st.download_button("Download this data as CSV", combined_csv, file_name="adhd_prevalence_data.csv", mime="text/csv")

    st.divider()
    st.markdown("**State-by-state map -- % of children currently receiving ADD/ADHD treatment**")
    st.caption(
        "This is a *different* measure from the treatment-range chart above: it's the share of "
        "**all** children in a state receiving ADD/ADHD treatment (diagnosed or not), not the share "
        "*among already-diagnosed* children. Don't compare the two numbers directly -- they use "
        "different denominators. Source: America's Health Rankings, 2023-2024 (National Survey of "
        "Children's Health)."
    )
    state_df = load_state_treatment()
    fig = state_choropleth_chart(theme, state_df)
    st.plotly_chart(fig, use_container_width=True, key="prevalence_state_map_chart")
    st.caption(
        f"Highest: {state_df.loc[state_df.rate_pct.idxmax(), 'state_name']} "
        f"({state_df.rate_pct.max():g}%) -- Lowest: {state_df.loc[state_df.rate_pct.idxmin(), 'state_name']} "
        f"({state_df.rate_pct.min():g}%) -- National average: 3.4%."
    )
    st.download_button("Download state map data as CSV", state_df.to_csv(index=False).encode(),
                        file_name="adhd_state_treatment.csv", mime="text/csv")

    with st.expander("Use your own state-level data instead"):
        st.write(
            "Upload a CSV with `state` (2-letter code) and `rate` columns to override the map above "
            "with different data -- for example a different year, or a different measure from "
            "[childhealthdata.org](https://www.childhealthdata.org) or the "
            "[America's Health Rankings API](https://developers.americashealthrankings.org/)."
        )
        custom_file = st.file_uploader("Upload state-level CSV (columns: state, rate)", type="csv", key="state_upload")
        if custom_file is not None:
            custom_df = pd.read_csv(custom_file)
            fig = state_choropleth_chart(theme, custom_df, z_col="rate", label_col="state",
                                          title="Custom state-level map", colorbar_title="Rate")
            st.plotly_chart(fig, use_container_width=True, key="prevalence_state_map_custom_chart")


# ---------------------------------------------------------------------------
# Tab 2: Adult Health & Activity (Kaggle "HYPERAKTIV", real upload or sample)
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

# Plain-English meaning of each metric -- these are standard clinical/research
# instruments, not self-explanatory column names.
METRIC_INFO = {
    "WURS": "Wender Utah Rating Scale -- an adult's retrospective self-report of "
            "childhood ADHD symptoms. Higher score = more severe symptoms recalled from childhood.",
    "ASRS": "Adult ADHD Self-Report Scale -- screens for current, present-day ADHD "
            "symptoms in adults. Higher score = more current symptoms.",
    "MADRS": "Montgomery-Asberg Depression Rating Scale -- a 10-item scale measuring "
             "depression severity. Higher score = more severe depression symptoms.",
    "HADS_A": "Hospital Anxiety and Depression Scale, Anxiety subscale -- higher score "
              "= more anxiety symptoms.",
    "HADS_D": "Hospital Anxiety and Depression Scale, Depression subscale -- higher "
              "score = more depression symptoms.",
    "ACC__mean": "Average wrist movement intensity over the recording period (from the wrist accelerometer).",
    "ACC__standard_deviation": "How much wrist movement varied over time -- higher means more erratic/variable activity.",
    "ACC__maximum": "The single most intense movement spike recorded during the monitoring period.",
    "ACC__minimum": "The lowest movement level recorded (closest to stillness).",
    "ACC__sum_values": "Total accumulated movement signal across the whole recording -- a rough measure of overall activity volume.",
    "ACC__abs_energy": "Overall 'energy' of the movement signal (sum of squared values) -- captures both intensity and duration of movement.",
    "ACC__median": "The typical (middle) wrist movement level, less sensitive to brief spikes than the mean.",
    "ACC__variance": "Statistical variance of movement -- same idea as standard deviation, useful for comparing spread.",
    "Adhd TScore Omissions": "From the CPT-II attention test: missed targets (failing to respond when they should have). "
                             "Reflects inattention -- a higher T-score indicates more missed targets.",
    "Adhd TScore Commissions": "From the CPT-II attention test: responded when they shouldn't have (false alarms). "
                                "Reflects impulsivity -- a higher T-score indicates more impulsive responding.",
    "Adhd TScore HitRT": "Average reaction time to correctly-identified targets on the CPT-II attention test.",
    "Adhd TScore VarSE": "How much reaction time varied across the CPT-II test -- higher means less consistent, "
                          "more variable attention over the task.",
    "Adhd TScore DPrime": "A signal-detection score reflecting how well the person distinguished targets from "
                           "non-targets on the CPT-II test.",
    "Adhd Confidence Index": "A composite CPT-II score combining multiple performance measures into one index.",
}


def health_box_plot(theme, merged, metric_col, label):
    fig = go.Figure()
    for i, g in enumerate(["ADHD", "Control"]):
        vals = merged.loc[merged["Group"] == g, metric_col].dropna()
        fig.add_trace(go.Box(y=vals, name=g, marker_color=theme["CATEGORICAL"][i], boxmean=True))
    fig.update_layout(title=f"{label} -- ADHD vs. Control", **theme["BASE_LAYOUT"])
    style_axes(fig, theme)
    return fig


def _health_metric_options_for_columns(present_by_category):
    """present_by_category: dict of category -> list of present column names."""
    metric_options = {}
    for c in present_by_category.get("clinical", []):
        metric_options[f"Clinical scale: {c}"] = c
    for c in present_by_category.get("activity", []):
        metric_options[f"Wrist activity: {c.replace('ACC__', '')}"] = c
    for c in present_by_category.get("cpt", []):
        metric_options[f"Attention test (CPT-II): {c}"] = c
    return metric_options


@st.cache_data
def generate_sample_health_data():
    """Synthetic demo data -- NOT real patients. Shaped like the real HYPERAKTIV
    columns so the rest of the app can't tell the difference, with plausible
    (but made up) group differences so the charts look interesting."""
    rng = np.random.default_rng(42)
    n = 40
    groups = np.array(["ADHD"] * n + ["Control"] * n)

    def draw(adhd_mean, adhd_sd, ctrl_mean, ctrl_sd):
        vals = np.where(groups == "ADHD", rng.normal(adhd_mean, adhd_sd, 2 * n),
                         rng.normal(ctrl_mean, ctrl_sd, 2 * n))
        return np.round(vals, 1)

    df = pd.DataFrame({"ID": range(1, 2 * n + 1), "Group": groups})
    df["ADHD"] = (df["Group"] == "ADHD").astype(int)
    df["WURS"] = draw(55, 15, 25, 12)
    df["ASRS"] = draw(60, 10, 35, 10)
    df["MADRS"] = draw(15, 8, 10, 6)
    df["HADS_A"] = draw(9, 4, 6, 3)
    df["HADS_D"] = draw(7, 4, 4, 3)
    df["ACC__mean"] = draw(120, 20, 100, 15)
    df["ACC__standard_deviation"] = draw(80, 15, 55, 12)
    df["ACC__maximum"] = draw(500, 80, 400, 70)
    df["ACC__minimum"] = np.round(rng.normal(-50, 20, 2 * n), 1)
    df["ACC__sum_values"] = draw(50000, 8000, 42000, 7000)
    df["ACC__abs_energy"] = draw(900000, 150000, 700000, 120000)
    df["ACC__median"] = draw(110, 18, 95, 14)
    df["ACC__variance"] = draw(6400, 1200, 3000, 900)
    df["Adhd TScore Omissions"] = draw(58, 10, 50, 8)
    df["Adhd TScore Commissions"] = draw(62, 10, 50, 8)
    df["Adhd TScore HitRT"] = draw(55, 10, 50, 8)
    df["Adhd TScore VarSE"] = draw(60, 10, 48, 8)
    df["Adhd TScore DPrime"] = draw(48, 10, 55, 8)
    df["Adhd Confidence Index"] = draw(65, 12, 45, 10)

    metric_options = _health_metric_options_for_columns({
        "clinical": CLINICAL_SCALE_CANDIDATES,
        "activity": ACTIVITY_FEATURE_CANDIDATES,
        "cpt": CPT_FEATURE_CANDIDATES,
    })
    return df, metric_options


def render_guess_game(theme, merged, metric_options):
    st.divider()
    st.markdown("**Guess the Group**")
    st.caption(
        "Two anonymized box plots below -- one's the ADHD group, one's Control, for a "
        "randomly picked metric. Test your intuition for how different (or similar!) these "
        "distributions really are."
    )

    st.session_state.setdefault("game_score", {"correct": 0, "total": 0})
    score = st.session_state["game_score"]

    if "game_round" not in st.session_state:
        _new_game_round(metric_options)
    if st.button("New round"):
        _new_game_round(metric_options)
    round_ = st.session_state["game_round"]

    if round_["metric_col"] not in metric_options.values():
        # Data source changed since the last round (e.g. switched from sample to real upload).
        _new_game_round(metric_options)
        round_ = st.session_state["game_round"]

    metric_col = round_["metric_col"]
    a_group, b_group = ("Control", "ADHD") if round_["flip"] else ("ADHD", "Control")

    fig = go.Figure()
    fig.add_trace(go.Box(y=merged.loc[merged["Group"] == a_group, metric_col].dropna(),
                          name="Group A", marker_color=theme["CATEGORICAL"][0], boxmean=True))
    fig.add_trace(go.Box(y=merged.loc[merged["Group"] == b_group, metric_col].dropna(),
                          name="Group B", marker_color=theme["CATEGORICAL"][1], boxmean=True))
    fig.update_layout(title="Mystery metric -- Group A vs. Group B", **theme["BASE_LAYOUT"])
    style_axes(fig, theme)
    st.plotly_chart(fig, use_container_width=True, key="game_chart")

    correct_letter = "A" if a_group == "ADHD" else "B"

    if not round_["answered"]:
        c1, c2, c3 = st.columns(3)
        if c1.button("Group A is ADHD"):
            _score_guess(round_, "A", correct_letter, score)
        if c2.button("Group B is ADHD"):
            _score_guess(round_, "B", correct_letter, score)
        c3.metric("Score", f"{score['correct']}/{score['total']}")
    else:
        st.metric("Score", f"{score['correct']}/{score['total']}")
        if round_["correct"]:
            st.success(f"Correct! Group {correct_letter} was ADHD (metric: {round_['label']}). Click 'New round' to keep going.")
        else:
            st.error(f"Not quite -- Group {correct_letter} was actually ADHD (metric: {round_['label']}). Click 'New round' to try another.")


def _new_game_round(metric_options):
    label = random.choice(list(metric_options.keys()))
    st.session_state["game_round"] = {
        "label": label, "metric_col": metric_options[label],
        "flip": random.choice([True, False]), "answered": False, "correct": False,
    }


def _score_guess(round_, guess_letter, correct_letter, score):
    round_["answered"] = True
    round_["correct"] = guess_letter == correct_letter
    score["total"] += 1
    if round_["correct"]:
        score["correct"] += 1


def render_health_tab():
    theme = get_theme()
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

    st.markdown("**Don't have the files handy?**")
    if st.button("Load synthetic sample data instead"):
        st.session_state["health_sample_mode"] = True
    use_sample = pi_file is None and st.session_state.get("health_sample_mode", False)

    if pi_file is None and not use_sample:
        st.info("Upload at least `patient_info.csv`, or click \"Load synthetic sample data\" above to "
                 "try the app immediately.")
        return

    if use_sample:
        st.info("Showing **synthetic demo data** -- randomly generated, not real patients. "
                 "Upload real files above any time to replace it.")
        merged, metric_options = generate_sample_health_data()
    else:
        pi = pd.read_csv(pi_file, sep=";")
        if "ADHD" not in pi.columns or "ID" not in pi.columns:
            st.error("This doesn't look like patient_info.csv -- expected an 'ID' and 'ADHD' column.")
            return
        pi["Group"] = pi["ADHD"].map({1: "ADHD", 0: "Control"})

        merged = pi.copy()
        present = {"clinical": [c for c in CLINICAL_SCALE_CANDIDATES if c in merged.columns],
                   "activity": [], "cpt": []}

        if feat_file is not None:
            feat = pd.read_csv(feat_file, sep=";")
            found = [c for c in ACTIVITY_FEATURE_CANDIDATES if c in feat.columns]
            if found:
                merged = merged.merge(feat[["ID"] + found], on="ID", how="inner")
                present["activity"] = found
            else:
                st.warning("features.csv was uploaded but none of the expected ACC__ columns were found.")

        if cpt_file is not None:
            cpt = pd.read_csv(cpt_file, sep=";")
            found = [c for c in CPT_FEATURE_CANDIDATES if c in cpt.columns]
            if found:
                merged = merged.merge(cpt[["ID"] + found], on="ID", how="inner")
                present["cpt"] = found
            else:
                st.warning("The CPT CSV was uploaded but none of the expected summary-score columns were found.")

        metric_options = _health_metric_options_for_columns(present)

    n_adhd = (merged["Group"] == "ADHD").sum()
    n_control = (merged["Group"] == "Control").sum()
    c1, c2 = st.columns(2)
    c1.metric("ADHD patients (matched across loaded files)", n_adhd)
    c2.metric("Controls (matched across loaded files)", n_control)

    with st.expander("Preview data"):
        st.dataframe(merged.head(50), use_container_width=True)

    st.session_state["health_merged"] = merged
    st.session_state["health_metric_options"] = metric_options
    st.session_state["health_is_sample"] = use_sample

    if not metric_options:
        st.warning("No known metric columns found -- double check you uploaded the right files.")
        return

    label = st.selectbox("Metric to compare", list(metric_options.keys()))
    metric_col = metric_options[label]

    fig = health_box_plot(theme, merged, metric_col, label)
    st.plotly_chart(fig, use_container_width=True, key="health_metric_chart")

    info = METRIC_INFO.get(metric_col)
    if info:
        st.caption(f"**What this measures:** {info}")

    with st.expander("Glossary -- what all these metrics mean"):
        for cat, cols in [
            ("Clinical scales", CLINICAL_SCALE_CANDIDATES),
            ("Wrist activity", ACTIVITY_FEATURE_CANDIDATES),
            ("CPT-II attention test", CPT_FEATURE_CANDIDATES),
        ]:
            present = [c for c in cols if c in metric_options.values()]
            if not present:
                continue
            st.markdown(f"**{cat}**")
            for c in present:
                st.markdown(f"- **{c}**: {METRIC_INFO.get(c, 'No description available.')}")

    csv_bytes = merged.to_csv(index=False).encode()
    st.download_button("Download this data as CSV", csv_bytes, file_name="adhd_health_data.csv", mime="text/csv")

    render_guess_game(theme, merged, metric_options)


# ---------------------------------------------------------------------------
# Tab 3: EEG Signals (Kaggle upload, or synthetic sample)
# ---------------------------------------------------------------------------
# The international 10-20 electrode placement system -- these column names are
# electrode positions on the scalp, not arbitrary labels.
EEG_CHANNEL_INFO = {
    "Fp1": "Frontopolar, left", "Fp2": "Frontopolar, right",
    "F3": "Frontal, left", "F4": "Frontal, right",
    "F7": "Frontal, far left", "F8": "Frontal, far right",
    "Fz": "Frontal, midline",
    "C3": "Central, left", "C4": "Central, right", "Cz": "Central, midline",
    "T7": "Temporal, left", "T8": "Temporal, right",
    "P3": "Parietal, left", "P4": "Parietal, right",
    "P7": "Parietal, far left", "P8": "Parietal, far right",
    "Pz": "Parietal, midline",
    "O1": "Occipital, left", "O2": "Occipital, right",
}


@st.cache_data
def generate_sample_eeg_data():
    """Synthetic demo EEG data -- sine waves + noise, NOT real brain activity.
    Just enough structure (a few 'recordings' with different noisiness) to make
    the waveform viewer and group comparison feel populated."""
    rng = np.random.default_rng(7)
    channels = list(EEG_CHANNEL_INFO.keys())
    recordings = [("demo_adhd_1", "ADHD"), ("demo_adhd_2", "ADHD"), ("demo_control_1", "Control")]
    n_samples = 800
    t = np.arange(n_samples)

    frames = []
    for rec_id, cls in recordings:
        noise_scale = 220 if cls == "ADHD" else 140  # purely illustrative, not a real finding
        data = {}
        for j, ch in enumerate(channels):
            base = 80 * np.sin(2 * np.pi * t / 40 + j)
            data[ch] = np.round(base + rng.normal(0, noise_scale, n_samples), 1)
        d = pd.DataFrame(data)
        d["Class"] = cls
        d["ID"] = rec_id
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def render_eeg_tab():
    theme = get_theme()
    st.subheader("EEG Signals")
    st.caption("Dataset: Kaggle - 'EEG Dataset for ADHD' (raw EEG channel data, ADHD vs. control).")

    st.write(
        "Download from [Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd) "
        "and upload a CSV below. Each column is an electrode placed on the scalp (the "
        "international '10-20 system'), each row is one instant in time, and the values "
        "are raw voltage readings -- meaningful as a waveform over time, not as standalone numbers."
    )
    f = st.file_uploader("Upload an EEG CSV", type="csv", key="eeg_upload")

    st.markdown("**Don't have the file handy?**")
    if st.button("Load synthetic sample data instead", key="eeg_sample_btn"):
        st.session_state["eeg_sample_mode"] = True
    use_sample = f is None and st.session_state.get("eeg_sample_mode", False)

    if f is None and not use_sample:
        st.info("Upload a file above, or click \"Load synthetic sample data\" to try this tab immediately.")
        return

    if use_sample:
        st.info("Showing **synthetic demo EEG data** -- sine waves plus random noise, not real brain "
                 "activity. Upload a real file above any time to replace it.")
        df = generate_sample_eeg_data()
    else:
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

    id_col = "ID" if "ID" in df.columns else None
    class_col = "Class" if "Class" in df.columns else None

    st.divider()
    st.markdown("**Waveform viewer -- one continuous recording**")

    if id_col:
        # Each ID is one continuous recording; rows for different IDs may be
        # stacked in the file, so filter to a single ID before plotting a
        # waveform -- otherwise the chart jumps between unrelated recordings.
        ids = df[id_col].dropna().unique().tolist()
        rec = st.selectbox(f"Recording ({id_col})", ids)
        rec_df = df[df[id_col] == rec].reset_index(drop=True)
        if class_col:
            rec_class = rec_df[class_col].iloc[0] if len(rec_df) else "unknown"
            st.caption(f"Recording **{rec}** -- {len(rec_df):,} samples, labeled **{rec_class}**")
    else:
        st.caption("No 'ID' column found -- treating the whole file as one continuous recording.")
        rec_df = df
        rec = None

    channel = st.selectbox("Channel to plot", numeric_cols)
    ch_note = EEG_CHANNEL_INFO.get(channel)
    if ch_note:
        st.caption(f"{channel} = {ch_note} electrode (10-20 system)")

    max_samples = max(len(rec_df), 100)
    n_samples = st.slider("Samples to display", 100, min(5000, max_samples), min(1000, max_samples))
    d = rec_df[channel].iloc[:n_samples]
    fig = go.Figure(go.Scatter(y=d, mode="lines", line=dict(color=theme["BLUE"], width=1)))
    fig.update_layout(title=f"{channel} -- raw waveform, recording {rec if id_col else '(whole file)'}", **theme["BASE_LAYOUT"])
    style_axes(fig, theme)
    st.plotly_chart(fig, use_container_width=True, key="eeg_waveform_chart")

    rec_csv = rec_df.to_csv(index=False).encode()
    st.download_button("Download this recording as CSV", rec_csv, file_name="eeg_recording.csv", mime="text/csv")

    st.divider()
    st.markdown("**Group comparison -- ADHD vs. control**")
    st.caption(
        "This pools every raw sample across all recordings, so it's a rough, noisy comparison "
        "(not a per-person statistical test) -- useful for a quick sanity check, not a conclusion."
    )
    default_group = class_col if class_col else "(none)"
    group_options = ["(none)"] + df.columns.tolist()
    group_col = st.selectbox(
        "Group / label column", group_options, index=group_options.index(default_group)
    )
    if group_col != "(none)" and df[group_col].nunique() <= 8:
        groups = df[group_col].dropna().unique().tolist()
        fig = go.Figure()
        for i, g in enumerate(groups):
            fig.add_trace(go.Box(y=df.loc[df[group_col] == g, channel], name=str(g),
                                  marker_color=theme["CATEGORICAL"][i % len(theme["CATEGORICAL"])]))
        fig.update_layout(title=f"{channel} -- all samples, grouped by {group_col}", **theme["BASE_LAYOUT"])
        style_axes(fig, theme)
        st.plotly_chart(fig, use_container_width=True, key="eeg_group_chart")

    with st.expander("What do the electrode names mean?"):
        st.write(
            "Each column is an electrode position under the international 10-20 system: "
            "Fp = frontopolar, F = frontal, C = central, P = parietal, T = temporal, "
            "O = occipital, and 'z' marks the midline. Odd numbers are the left side of "
            "the head, even numbers are the right."
        )
        cols = st.columns(3)
        items = [c for c in numeric_cols if c in EEG_CHANNEL_INFO]
        for i, c in enumerate(items):
            cols[i % 3].markdown(f"- **{c}**: {EEG_CHANNEL_INFO[c]}")


# ---------------------------------------------------------------------------
# Tab 4: Chat with the Data
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are a data assistant embedded in an ADHD Data Explorer app.
Answer the user's question using ONLY the data and glossary provided below -- do not
use outside knowledge about ADHD beyond what's in the glossary, and do not make
clinical or diagnostic claims. If the data provided doesn't contain the answer, say
so plainly instead of guessing. You may use the glossary to explain what a metric or
clinical scale means. When the user asks to see, plot, chart, or compare something
visually, use the show_chart tool instead of only describing it in words. Cite
specific numbers from the data when you can. Keep answers concise (a few sentences,
or a short list).

DATA AVAILABLE:
{context}
"""

CHART_TOOL = {
    "name": "show_chart",
    "description": (
        "Render a chart in the app so the user can see it. Use this whenever the user "
        "asks to see, plot, chart, compare, or visualize a metric, instead of only "
        "describing it in words."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["prevalence", "health", "state_map"],
                "description": "'prevalence' for national CDC/NSCH demographic charts, "
                                "'health' for the loaded adult ADHD-vs-control comparison data, "
                                "'state_map' for the US choropleth of ADD/ADHD treatment rate by state.",
            },
            "metric": {
                "type": "string",
                "description": (
                    "For source='prevalence': one of sex, race_ethnicity, severity, co_occurring. "
                    "For source='health': the exact column name of a metric shown in the health "
                    "data table (e.g. ASRS, ACC__mean, 'Adhd TScore Omissions'). "
                    "For source='state_map': not used, pass an empty string."
                ),
            },
        },
        "required": ["source", "metric"],
    },
}

SUGGESTED_QUESTIONS = [
    "What's the ADHD diagnosis rate by sex?",
    "Which state has the highest treatment rate?",
    "Compare ASRS scores between ADHD and control",
    "What does WURS measure?",
]


def build_data_context() -> str:
    parts = []

    glossary_lines = "\n".join(f"- **{k}**: {v}" for k, v in METRIC_INFO.items())
    parts.append("## Glossary of clinical/technical terms used in this app\n" + glossary_lines)

    prevalence = load_prevalence()
    parts.append("## National ADHD prevalence & demographics (CDC/NSCH 2022-2023)\n" + prevalence.to_markdown(index=False))

    treatment = load_treatment_range()
    parts.append("## Treatment rate range across US states (2022)\n" + treatment.to_markdown(index=False))

    state_df = load_state_treatment()
    parts.append(
        "## % of children receiving ADD/ADHD treatment, by state (America's Health Rankings, 2023-2024)\n"
        "Note: this is share of ALL children (diagnosed or not), a different denominator from the "
        "treatment-range table above which is share AMONG diagnosed children -- don't compare the two directly.\n\n"
        + state_df.to_markdown(index=False)
    )

    merged = st.session_state.get("health_merged")
    if merged is not None:
        cols = ["ID", "Group"] + [c for c in st.session_state.get("health_metric_options", {}).values()]
        cols = [c for c in dict.fromkeys(cols) if c in merged.columns]
        note = " (synthetic sample data, not real patients)" if st.session_state.get("health_is_sample") else " (uploaded by user)"
        parts.append(
            f"## Adult health/activity data{note}\n"
            f"{len(merged)} matched patients. Group counts: "
            f"{merged['Group'].value_counts().to_dict()}\n\n"
            + merged[cols].to_markdown(index=False)
        )
    else:
        parts.append("## Adult health/activity data\nNot loaded yet in the Adult Health & Activity tab.")

    eeg_info = st.session_state.get("eeg_info")
    if eeg_info is not None:
        parts.append(
            "## EEG data\n"
            f"An EEG file was loaded with {eeg_info['rows']} rows and columns: "
            f"{eeg_info['columns']}. (Raw signal values are not included here -- too large.)"
        )
    else:
        parts.append("## EEG data\nNot loaded yet in the EEG Signals tab.")

    return "\n\n".join(parts)


def execute_chart_tool(theme, tool_input):
    """Returns (plotly figure or None, short text result for the model to see)."""
    source = tool_input.get("source")
    metric = tool_input.get("metric")

    if source == "prevalence":
        if metric not in PREVALENCE_TITLES:
            return None, f"Unknown prevalence category '{metric}'. Valid options: {list(PREVALENCE_TITLES.keys())}."
        prevalence = load_prevalence()
        d = prevalence[prevalence.category == metric]
        fig = prevalence_bar_chart(theme, d, PREVALENCE_TITLES[metric])
        return fig, f"Rendered a bar chart: {PREVALENCE_TITLES[metric]}."

    if source == "state_map":
        state_df = load_state_treatment()
        fig = state_choropleth_chart(theme, state_df)
        top = state_df.loc[state_df.rate_pct.idxmax()]
        bottom = state_df.loc[state_df.rate_pct.idxmin()]
        return fig, (f"Rendered the state choropleth map. Highest: {top['state_name']} ({top['rate_pct']:g}%). "
                     f"Lowest: {bottom['state_name']} ({bottom['rate_pct']:g}%). National average: 3.4%.")

    if source == "health":
        merged = st.session_state.get("health_merged")
        metric_options = st.session_state.get("health_metric_options", {})
        if merged is None:
            return None, "No health data is loaded -- tell the user to upload files or load sample data in the Adult Health & Activity tab."
        if metric not in metric_options.values():
            return None, f"'{metric}' isn't an available health metric. Available: {sorted(set(metric_options.values()))}."
        label = next(k for k, v in metric_options.items() if v == metric)
        fig = health_box_plot(theme, merged, metric, label)
        return fig, f"Rendered a box plot comparing '{metric}' between ADHD and Control."

    return None, f"Unknown chart source '{source}'."


def render_chat_tab():
    st.subheader("Chat with the Data")
    st.caption("Ask questions in plain English -- the assistant answers using the data currently loaded "
               "in this app, and can draw charts on request.")

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

    if not st.session_state.chat_messages:
        st.markdown("**Try asking:**")
        chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if chip_cols[i].button(q, key=f"chip_{i}"):
                st.session_state["chat_pending"] = q

    for idx, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True, key=f"hist_chart_{idx}")

    typed = st.chat_input("e.g. Which treatment type varies most across states?")
    prompt = typed or st.session_state.pop("chat_pending", None)
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    theme = get_theme()
    context = build_data_context()
    system = CHAT_SYSTEM_PROMPT.format(context=context)
    # Only send recent turns (as plain text) to keep token usage bounded.
    recent = st.session_state.chat_messages[-10:]
    api_messages = [{"role": m["role"], "content": m["content"]} for m in recent]

    fig = None
    answer = ""
    with st.chat_message("assistant"):
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-haiku-4-5", max_tokens=600, system=system,
                tools=[CHART_TOOL], messages=api_messages,
            )
            answer = "\n".join(b.text for b in response.content if b.type == "text").strip()

            tool_use = next((b for b in response.content if b.type == "tool_use"), None)
            if tool_use is not None:
                fig, result_text = execute_chart_tool(theme, tool_use.input)
                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": result_text}],
                })
                response2 = client.messages.create(
                    model="claude-haiku-4-5", max_tokens=400, system=system,
                    tools=[CHART_TOOL], messages=api_messages,
                )
                follow_up = "\n".join(b.text for b in response2.content if b.type == "text").strip()
                answer = (answer + "\n\n" + follow_up).strip() if answer else follow_up

            if not answer:
                answer = "Here's what I found:" if fig is not None else "I don't have enough loaded data to answer that."
        except Exception as e:
            answer = f"Something went wrong calling the AI: {e}"

        st.markdown(answer)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chat_live_chart_{len(st.session_state.chat_messages)}")

    st.session_state.chat_messages.append({"role": "assistant", "content": answer, "fig": fig})


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ADHD Data Explorer", page_icon="🧠", layout="wide")

with st.sidebar:
    st.markdown("### 🧠 ADHD Data Explorer")
    st.toggle("🌙 Dark mode", key="dark_mode")
    st.divider()
    st.write(
        "Explore ADHD prevalence, adult health/activity data, and EEG signals -- "
        "or chat with an AI about whatever's currently loaded."
    )
    st.caption("Educational/exploratory tool -- not a diagnostic instrument.")

st.markdown(custom_css(get_theme()), unsafe_allow_html=True)
render_hero()

_first_visit = "seen_intro" not in st.session_state
st.session_state["seen_intro"] = True
with st.expander("ℹ️ About this app", expanded=_first_visit):
    st.markdown(
        "This app explores public and research ADHD data across four tabs:\n\n"
        "- **📊 Prevalence & Demographics** -- national CDC/NSCH statistics, works immediately, no download needed\n"
        "- **🩺 Adult Health & Activity** -- real clinical research data (HYPERAKTIV) comparing ADHD-diagnosed "
        "adults to controls, or synthetic sample data if you don't have the files\n"
        "- **🌊 EEG Signals** -- raw brain-electrical-activity waveforms, real or synthetic sample data\n"
        "- **💬 Chat with the Data** -- ask an AI questions about whatever's loaded; it can draw charts too\n\n"
        "This app is for exploratory and educational purposes only -- it is **not a diagnostic tool**, "
        "and nothing here should be used to draw clinical conclusions."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Prevalence & Demographics", "🩺 Adult Health & Activity", "🌊 EEG Signals", "💬 Chat with the Data"]
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
st.caption("Not a diagnostic tool. Prevalence figures: CDC / NSCH 2022-2023.")
