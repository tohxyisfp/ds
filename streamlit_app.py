import random
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="S.Gold Predict",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME / CSS  (obsidian + gold "trading floor" palette)
# =========================================================

PALETTE = {
    "bg": "#FAF7F0",
    "panel": "#FFFFFF",
    "panel_border": "#E4DAC0",
    "gold": "#B8860B",
    "gold_bright": "#9A6B0C",
    "ivory": "#2B2620",
    "muted": "#7A7266",
    "up": "#1F9D5A",
    "down": "#C1442E",
    "grid": "#DED2AE",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #F1EADA 0%, {PALETTE['bg']} 45%) fixed;
}}

/* Headline typography */
h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: {PALETTE['ivory']} !important;
    letter-spacing: 0.2px;
}}

/* ---- Hero header ---- */
.hero-wrap {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    border-bottom: 1px solid {PALETTE['panel_border']};
    padding-bottom: 1rem;
    margin-bottom: 0.4rem;
}}
.hero-title {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(100deg, {PALETTE['gold_bright']}, {PALETTE['gold']} 55%, #a9791f);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0;
}}
.hero-sub {{
    color: {PALETTE['muted']};
    font-size: 0.92rem;
    font-family: 'JetBrains Mono', monospace;
}}

/* ---- Ticker tape ---- */
.ticker-outer {{
    width: 100%;
    overflow: hidden;
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['panel_border']};
    border-radius: 8px;
    padding: 8px 0;
    margin: 0.6rem 0 1.2rem 0;
}}
.ticker-track {{
    display: inline-flex;
    white-space: nowrap;
    animation: ticker-scroll 28s linear infinite;
}}
.ticker-item {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {PALETTE['ivory']};
    padding: 0 2.2rem;
    border-right: 1px solid {PALETTE['panel_border']};
}}
.ticker-item b {{ color: {PALETTE['gold_bright']}; }}
.ticker-up {{ color: {PALETTE['up']}; }}
.ticker-down {{ color: {PALETTE['down']}; }}
@keyframes ticker-scroll {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(-50%); }}
}}

/* ---- Stat / metric cards ---- */
.stat-card {{
    background: linear-gradient(160deg, {PALETTE['panel']}, #F3ECDA);
    border: 1px solid {PALETTE['panel_border']};
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(120,100,50,0.06);
}}
.stat-label {{
    color: {PALETTE['muted']};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-family: 'Inter', sans-serif;
    margin-bottom: 0.15rem;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: {PALETTE['ivory']};
}}
.stat-sub {{
    color: {PALETTE['muted']};
    font-size: 0.72rem;
    margin-top: 0.1rem;
}}

/* ---- Result card ---- */
.result-card {{
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-top: 0.6rem;
    border: 1px solid;
}}
.result-up {{
    background: linear-gradient(135deg, rgba(31,157,90,0.14), rgba(31,157,90,0.03));
    border-color: rgba(31,157,90,0.45);
}}
.result-down {{
    background: linear-gradient(135deg, rgba(193,68,46,0.14), rgba(193,68,46,0.03));
    border-color: rgba(193,68,46,0.45);
}}
.result-flat {{
    background: linear-gradient(135deg, rgba(184,134,11,0.14), rgba(184,134,11,0.03));
    border-color: rgba(184,134,11,0.45);
}}
.result-headline {{
    font-family: 'Fraunces', serif;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
}}
.result-caption {{
    color: {PALETTE['muted']};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}}

/* ---- Actual price callout ---- */
.actual-box {{
    background: {PALETTE['panel']};
    border: 1px solid {PALETTE['panel_border']};
    border-left: 4px solid {PALETTE['gold']};
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    color: {PALETTE['ivory']};
}}
.actual-box b {{ color: {PALETTE['gold_bright']}; font-size: 1.05rem; }}

/* ---- Big comparison table ---- */
.big-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.08rem;
    margin-top: 0.6rem;
}}
.big-table th {{
    text-align: left;
    padding: 0.85rem 1.1rem;
    border-bottom: 2px solid {PALETTE['gold']};
    color: {PALETTE['muted']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'Inter', sans-serif;
}}
.big-table td {{
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid {PALETTE['panel_border']};
    color: {PALETTE['ivory']};
}}
.big-table tr.highlight-row td {{
    background: rgba(184,134,11,0.16);
    font-weight: 700;
    color: {PALETTE['gold_bright']};
}}
.big-table tr.highlight-row {{
    animation: rowGlow 1.1s ease-out;
}}
@keyframes rowGlow {{
    0% {{ background: rgba(184,134,11,0.55); }}
    100% {{ background: transparent; }}
}}

/* ---- Data-updated flash banner ---- */
.update-flash {{
    background: linear-gradient(90deg, rgba(184,134,11,0.22), rgba(184,134,11,0.04));
    border: 1px solid {PALETTE['gold']};
    border-radius: 8px;
    padding: 0.55rem 1rem;
    margin-bottom: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {PALETTE['gold_bright']};
    animation: flashPulse 1s ease-out;
}}
@keyframes flashPulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(184,134,11,0.55); transform: scale(0.985); }}
    60% {{ box-shadow: 0 0 0 16px rgba(184,134,11,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(184,134,11,0); transform: scale(1); }}
}}

/* ---- Full-screen predict effect overlay ---- */
.fx-overlay {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 999999;
    overflow: hidden;
}}
.fx-piece {{
    position: absolute;
    top: -60px;
    animation-name: fxFall;
    animation-timing-function: cubic-bezier(.25,.46,.45,.94);
    animation-fill-mode: forwards;
    filter: drop-shadow(0 0 8px rgba(255,215,0,0.85));
}}
@keyframes fxFall {{
    0%   {{ transform: translate(0, 0) rotate(0deg); opacity: 1; }}
    100% {{ transform: translate(var(--drift), 112vh) rotate(680deg); opacity: 0; }}
}}
.fx-flash {{
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 999998;
    background: radial-gradient(circle at 50% 35%, rgba(255,215,0,0.35), rgba(255,215,0,0) 60%);
    animation: fxFlashFade 1.1s ease-out forwards;
}}
@keyframes fxFlashFade {{
    0% {{ opacity: 0; }}
    25% {{ opacity: 1; }}
    100% {{ opacity: 0; }}
}}

/* ---- Model badge pill ---- */
.badge {{
    display:inline-block;
    background: rgba(184,134,11,0.12);
    border: 1px solid rgba(184,134,11,0.5);
    color: {PALETTE['gold_bright']};
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 6px;
}}

/* ---- Insight callout ---- */
.insight-box {{
    background: rgba(184,134,11,0.07);
    border-left: 3px solid {PALETTE['gold']};
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
    color: {PALETTE['ivory']};
    font-size: 0.9rem;
    line-height: 1.55;
}}
.insight-box b {{ color: {PALETTE['gold_bright']}; }}

/* Buttons */
.stButton>button {{
    background: linear-gradient(120deg, {PALETTE['gold']}, #b8933f);
    color: #14110a;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(184,134,11,0.28);
    color: #14110a;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {PALETTE['panel']};
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-family: 'Inter', sans-serif;
}}

footer {{visibility: hidden;}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    lr_model = joblib.load("models/linear_regression_pipeline.pkl")
    dt_model = joblib.load("models/decision_tree_pipeline.pkl")
    gb_model = joblib.load("models/gradient_boosting_pipeline.pkl")
    rf_model = joblib.load("models/random_forest_pipeline.pkl")
    return lr_model, dt_model, gb_model, rf_model


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Gold_Price.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# =========================================================
# LOAD MODEL COMPARISON
# =========================================================

@st.cache_data
def load_model_comparison():
    return pd.read_csv("data/model_comparison.csv")


# =========================================================
# CREATE FEATURES
# -----------------------------------------------------------------------
# This mirrors EXACTLY the feature-engineering code that produced the
# training data for the saved pipelines (notebook cells 3, 5 & 6),
# reproduced here in the same cumulative order:
#
#   calc_chg          = Price.pct_change() * 100
#   chg_diff           = abs(calc_chg - Chg%)                      <- was unsigned & flipped
#   day_gap            = Date.diff().dt.days (days since prev row) <- was a price gap
#   Daily_Range        = High - Low
#   Daily_Range_Pct     = Daily_Range / Open * 100                  <- was divided by Price
#   Year / Month / DayOfWeek, MA7 / MA30 / Volatility_7 (min_periods=1)
#   Lag1_Price, Lag1_Volume
#   Target_NextClose (reference only, not a model input)
# =========================================================

def create_features(data):
    data = data.copy()

    data["calc_chg"] = data["Price"].pct_change() * 100
    data["chg_diff"] = (data["calc_chg"] - data["Chg%"]).abs()
    data["day_gap"] = data["Date"].diff().dt.days

    data["Daily_Range"] = data["High"] - data["Low"]
    data["Daily_Range_Pct"] = (data["Daily_Range"] / data["Open"]) * 100

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["DayOfWeek"] = data["Date"].dt.day_name()

    data["MA7"] = data["Price"].rolling(window=7, min_periods=1).mean()
    data["MA30"] = data["Price"].rolling(window=30, min_periods=1).mean()
    data["Volatility_7"] = data["Price"].rolling(window=7, min_periods=1).std()

    data["Lag1_Price"] = data["Price"].shift(1)
    data["Lag1_Volume"] = data["Volume"].shift(1)

    # Reference only — the actual next trading day's close, when known.
    # Never fed into a model as an input feature.
    data["Actual_NextClose"] = data["Price"].shift(-1)

    return data


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:
    lr_model, dt_model, gb_model, rf_model = load_models()
    df = load_data()
    df_features = create_features(df)
    model_comparison = load_model_comparison()
except Exception as e:
    st.error("Unable to load the model files or dataset.")
    st.code(str(e))
    st.stop()


# =========================================================
# FEATURES
# =========================================================

feature_cols = [
    "Price", "Open", "High", "Low", "Volume", "Chg%",
    "calc_chg", "chg_diff", "day_gap",
    "Daily_Range", "Daily_Range_Pct",
    "Year", "Month", "DayOfWeek",
    "MA7", "MA30", "Volatility_7",
    "Lag1_Price", "Lag1_Volume",
]

FEATURE_META = {
    "Price":            ("Close Price",           "%.2f", 1.0,  "Market"),
    "Open":             ("Open Price",             "%.2f", 1.0,  "Market"),
    "High":             ("High Price",              "%.2f", 1.0,  "Market"),
    "Low":              ("Low Price",               "%.2f", 1.0,  "Market"),
    "Volume":           ("Volume",                  "%.0f", 1.0,  "Market"),
    "Chg%":             ("Reported Change (%)",     "%.2f", 0.01, "Market"),
    "calc_chg":         ("Computed Change (%)",     "%.4f", 0.01, "Engineered"),
    "chg_diff":         ("Chg% Discrepancy",        "%.4f", 0.01, "Engineered"),
    "day_gap":          ("Days Since Prev. Trade",  "%.0f", 1.0,  "Engineered"),
    "Daily_Range":      ("Daily Range (High-Low)",  "%.2f", 1.0,  "Engineered"),
    "Daily_Range_Pct":  ("Daily Range (% of Open)", "%.4f", 0.01, "Engineered"),
    "MA7":              ("7-Day Moving Average",    "%.2f", 1.0,  "Engineered"),
    "MA30":             ("30-Day Moving Average",   "%.2f", 1.0,  "Engineered"),
    "Volatility_7":     ("7-Day Volatility (Std)",  "%.4f", 0.01, "Engineered"),
    "Lag1_Price":       ("Previous Day Close",      "%.2f", 1.0,  "Engineered"),
    "Lag1_Volume":      ("Previous Day Volume",     "%.0f", 1.0,  "Engineered"),
    "Year":             ("Year",                    "%.0f", 1.0,  "Calendar"),
    "Month":            ("Month",                   "%.0f", 1.0,  "Calendar"),
    "DayOfWeek":        ("Day of Week",              None,   None, "Calendar"),
}


# =========================================================
# MODEL MAP
# =========================================================

MODEL_MAP = {
    "Linear Regression": lr_model,
    "Decision Tree": dt_model,
    "Gradient Boosting": gb_model,
    "Random Forest": rf_model,
}

MODEL_COLORS = {
    "Linear Regression": "#D4AF37",
    "Decision Tree": "#5DA9E9",
    "Gradient Boosting": "#E4572E",
    "Random Forest": "#8E7CC3",
}

best_model_name = model_comparison.sort_values("RMSE").iloc[0]["Model"]


# =========================================================
# HELPERS
# =========================================================

def fmt_num(x, dp=2):
    try:
        return f"{x:,.{dp}f}"
    except Exception:
        return str(x)


def stat_card(label, value, sub=""):
    st.markdown(
        f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_screen_effect(n_pieces=46, sparkly=True):
    """Full-viewport golden burst (coins + sparkles) — injected directly into the
    main page (not an iframe) so it can cover the entire screen with position:fixed."""
    glyphs = ["🪙", "✨", "💰", "⭐"] if sparkly else ["🪙"]
    pieces_html = ""
    for _ in range(n_pieces):
        left = random.uniform(1, 98)
        delay = round(random.uniform(0, 0.6), 2)
        duration = round(random.uniform(1.5, 2.8), 2)
        size = random.randint(16, 34)
        drift = random.randint(-60, 60)
        glyph = random.choice(glyphs)
        pieces_html += (
            f'<div class="fx-piece" style="left:{left}%; '
            f'animation-delay:{delay}s; animation-duration:{duration}s; '
            f'font-size:{size}px; --drift:{drift}px;">{glyph}</div>'
        )

    st.markdown(
        f"""
        <div class="fx-flash"></div>
        <div class="fx-overlay">{pieces_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_update_flash(text):
    """Small self-contained banner that visibly pulses, so it's obvious when
    switching a date has actually refreshed the data below it."""
    st.markdown(
        f"""<div class="update-flash">🔔 {text}</div>""",
        unsafe_allow_html=True,
    )


def build_ticker(data):
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    change = latest["Price"] - prev["Price"]
    pct = (change / prev["Price"]) * 100
    hi_all = data["Price"].max()
    lo_all = data["Price"].min()
    cls = "ticker-up" if change >= 0 else "ticker-down"
    arrow = "▲" if change >= 0 else "▼"

    items = [
        f'Latest Close <b>{fmt_num(latest["Price"], 2)}</b>',
        f'<span class="{cls}">{arrow} {fmt_num(change, 2)} ({pct:+.2f}%)</span>',
        f'Volume <b>{fmt_num(latest["Volume"], 0)}</b>',
        f'All-Time High <b>{fmt_num(hi_all, 2)}</b>',
        f'All-Time Low <b>{fmt_num(lo_all, 2)}</b>',
        f'Records <b>{len(data):,}</b>',
        f'As of <b>{latest["Date"].date()}</b>',
    ]
    track = "".join(f'<span class="ticker-item">{i}</span>' for i in items)
    # duplicate for a seamless scroll loop
    html = f"""
    <div class="ticker-outer">
        <div class="ticker-track">{track}{track}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div>
            <p class="hero-title">🪙 S.Gold Predict</p>
            <p class="hero-sub">Next-Trading-Day Gold Price Forecasting · BMDS2003 Data Science Project</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

build_ticker(df_features)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### 🪙 Market Snapshot")
    latest_row = df_features.iloc[-1]
    st.metric(
        "Latest Close",
        fmt_num(latest_row["Price"], 2),
        delta=f'{fmt_num(latest_row["Price"] - df_features.iloc[-2]["Price"], 2)}',
    )
    st.metric("Latest Volume", fmt_num(latest_row["Volume"], 0))
    st.caption(f"Dataset span: **{df['Date'].min().date()} → {df['Date'].max().date()}**")
    st.caption(f"Total trading records: **{len(df):,}**")
    st.divider()
    st.markdown("### 🏆 Best Model")
    best_row = model_comparison.sort_values("RMSE").iloc[0]
    st.markdown(
        f"**{best_row['Model']}**  \n"
        f"RMSE `{fmt_num(best_row['RMSE'], 2)}` · R² `{best_row['R²']:.4f}`"
    )
    st.caption("Ranked by lowest RMSE on the chronological test set.")
    st.divider()
    st.caption("For academic demonstration purposes only — not financial advice.")


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["📈 Trading Chart", "🔮 Price Prediction", "📊 Model Comparison"]
)


# =========================================================
# TAB 1 — TRADING CHART
# =========================================================

with tab1:
    st.header("Gold Price Historical Trading Chart")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    if "range_start" not in st.session_state:
        st.session_state["range_start"] = max(min_date, max_date - datetime.timedelta(days=365))
    if "range_end" not in st.session_state:
        st.session_state["range_end"] = max_date

    st.write("Quick range:")
    qcols = st.columns(8)
    quick_ranges = {
        "7D": 7, "1M": 30, "3M": 90, "6M": 182,
        "1Y": 365, "2Y": 730, "5Y": 1825, "MAX": None,
    }
    for col, (label, days) in zip(qcols, quick_ranges.items()):
        if col.button(label, use_container_width=True):
            st.session_state["range_end"] = max_date
            st.session_state["range_start"] = min_date if days is None else max(
                min_date, max_date - datetime.timedelta(days=days)
            )
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=st.session_state["range_start"],
            min_value=min_date, max_value=max_date, key="start_date_input",
        )
    with col2:
        end_date = st.date_input(
            "End Date", value=st.session_state["range_end"],
            min_value=min_date, max_value=max_date, key="end_date_input",
        )

    if start_date > end_date:
        st.warning("Start Date must be earlier than End Date.")
    else:
        mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
        range_df = df.loc[mask].reset_index(drop=True)

        if range_df.empty:
            st.info("No trading records in the selected range.")
        else:
            hi_idx = range_df["High"].idxmax()
            lo_idx = range_df["Low"].idxmin()
            net_change = range_df["Price"].iloc[-1] - range_df["Price"].iloc[0]
            net_pct = (net_change / range_df["Price"].iloc[0]) * 100

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                stat_card("Highest Price", fmt_num(range_df.loc[hi_idx, "High"], 2),
                           f'on {range_df.loc[hi_idx, "Date"].date()}')
            with m2:
                stat_card("Lowest Price", fmt_num(range_df.loc[lo_idx, "Low"], 2),
                           f'on {range_df.loc[lo_idx, "Date"].date()}')
            with m3:
                trend_color = PALETTE["up"] if net_change >= 0 else PALETTE["down"]
                stat_card(
                    "Net Change (Range)",
                    f'<span style="color:{trend_color}">{net_change:+,.2f} ({net_pct:+.2f}%)</span>',
                    f"{start_date} → {end_date}",
                )
            with m4:
                stat_card("Avg Daily Volume", fmt_num(range_df["Volume"].mean(), 0),
                           f"{len(range_df):,} trading days")

            st.write("")

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.72, 0.28], vertical_spacing=0.03,
            )
            # Shaded daily high/low band behind the close-price line, so we keep
            # some of the range information a candlestick used to show.
            fig.add_trace(
                go.Scatter(
                    x=range_df["Date"], y=range_df["High"], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=range_df["Date"], y=range_df["Low"], mode="lines",
                    line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(184,134,11,0.14)", name="Daily Range",
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=range_df["Date"], y=range_df["Price"], mode="lines",
                    name="Close Price", line=dict(color=PALETTE["gold_bright"], width=2.6),
                ),
                row=1, col=1,
            )
            vol_colors = np.where(range_df["Price"].diff().fillna(0) >= 0, PALETTE["up"], PALETTE["down"])
            fig.add_trace(
                go.Bar(x=range_df["Date"], y=range_df["Volume"], name="Volume",
                       marker_color=vol_colors, opacity=0.65),
                row=2, col=1,
            )
            fig.update_layout(
                height=560,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["ivory"], family="Inter, sans-serif"),
                xaxis_rangeslider_visible=False,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
            )
            fig.update_xaxes(gridcolor=PALETTE["grid"], showgrid=True)
            fig.update_yaxes(gridcolor=PALETTE["grid"], showgrid=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.subheader("Trading Data")
            st.caption("Pick specific dates to inspect — nothing is pre-selected, choose your own or use a quick pick.")

            date_options = range_df["Date"].dt.date.tolist()

            if "trading_data_multiselect" not in st.session_state:
                st.session_state["trading_data_multiselect"] = []

            st.write("Quick pick:")
            qd_cols = st.columns(5)
            quick_pick_ranges = {
                "Last 7D": 7, "Last 1M": 30, "Last 3M": 90, "Last 1Y": 365, "All shown": None,
            }
            for qcol, (qlabel, qdays) in zip(qd_cols, quick_pick_ranges.items()):
                if qcol.button(qlabel, use_container_width=True, key=f"quickpick_{qlabel}"):
                    if qdays is None:
                        sel = date_options
                    else:
                        cutoff = max(date_options) - datetime.timedelta(days=qdays)
                        sel = [d for d in date_options if d >= cutoff]
                    st.session_state["trading_data_multiselect"] = sorted(sel, reverse=True)
                    st.rerun()

            picked_dates = st.multiselect(
                "Select dates to display",
                options=sorted(date_options, reverse=True),
                key="trading_data_multiselect",
            )

            if picked_dates:
                show_df = range_df[range_df["Date"].dt.date.isin(picked_dates)][
                    ["Date", "Open", "High", "Low", "Price", "Volume", "Chg%"]
                ].sort_values("Date", ascending=False)
                st.dataframe(show_df, use_container_width=True, hide_index=True)
            else:
                st.info("Select at least one date above to see its trading data.")


# =========================================================
# TAB 2 — PRICE PREDICTION
# =========================================================

with tab2:
    st.header("Predict Next Trading Day's Gold Price")
    st.write("Select a date to auto-fill the model's inputs from real market history, then adjust anything you like before predicting.")

    valid_df = df_features.dropna(subset=feature_cols).copy()
    available_dates = sorted(valid_df["Date"].dt.date.tolist(), reverse=True)

    col_model, col_date = st.columns([1, 1])
    with col_model:
        model_choice = st.selectbox(
            "Choose Model",
            list(MODEL_MAP.keys()),
            format_func=lambda m: f"{m}  🏆" if m == best_model_name else m,
        )
        model = MODEL_MAP[model_choice]
        row = model_comparison[model_comparison["Model"] == model_choice].iloc[0]
        st.caption(f"Test-set performance — RMSE `{fmt_num(row['RMSE'],2)}` · MAE `{fmt_num(row['MAE'],2)}` · R² `{row['R²']:.4f}`")

    with col_date:
        min_valid, max_valid = available_dates[-1], available_dates[0]
        valid_dates_set = set(available_dates)
        picked_date = st.date_input(
            "Select Date (full trading history)",
            value=max_valid, min_value=min_valid, max_value=max_valid,
            key="predict_date_input",
        )
        if picked_date in valid_dates_set:
            chosen_date = picked_date
        else:
            earlier = [d for d in available_dates if d <= picked_date]
            chosen_date = earlier[0] if earlier else min_valid
            st.caption(f"⚠️ No trading data on {picked_date} — snapped to nearest trading day **{chosen_date}**.")
        st.caption(f"Full valid range: {min_valid} → {max_valid}")

    selected_row = valid_df[valid_df["Date"].dt.date == chosen_date].iloc[0]

    st.subheader("Input Features")
    render_update_flash(f"Inputs auto-filled from {chosen_date}")
    user_input = {}

    groups = {"Market": [], "Engineered": [], "Calendar": []}
    for feat in feature_cols:
        groups[FEATURE_META[feat][3]].append(feat)

    def render_feature_inputs(feat_list, columns=3):
        cols = st.columns(columns)
        for i, feature in enumerate(feat_list):
            label, fmt, step, _ = FEATURE_META[feature]
            with cols[i % columns]:
                if feature == "DayOfWeek":
                    user_input[feature] = selected_row[feature]
                    st.text_input(label, value=str(selected_row[feature]), disabled=True)
                else:
                    user_input[feature] = st.number_input(
                        label,
                        value=float(selected_row[feature]),
                        step=step,
                        format=fmt,
                        # Key includes the chosen date so the widget remounts (and
                        # actually refreshes its shown value) whenever the date changes.
                        key=f"input_{feature}_{chosen_date}",
                    )

    with st.expander("💰 Market Data", expanded=True):
        render_feature_inputs(groups["Market"])
    with st.expander("🧮 Engineered Indicators", expanded=False):
        render_feature_inputs(groups["Engineered"])
    with st.expander("📅 Calendar", expanded=False):
        render_feature_inputs(groups["Calendar"])

    predict_clicked = st.button("🔮 Predict Next-Day Price", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame([user_input])[feature_cols]

        try:
            predicted_price = float(model.predict(input_df)[0])
            previous_price = float(selected_row["Price"])
            price_change = predicted_price - previous_price
            percentage_change = (price_change / previous_price) * 100

            # Full-screen golden/coin burst — plays for every prediction, right
            # before the result is revealed.
            render_screen_effect(sparkly=(price_change >= 0))

            st.divider()
            st.subheader("Prediction Result")

            if price_change > 0:
                css_class, verdict, icon = "result-up", "Predicted increase", "📈"
            elif price_change < 0:
                css_class, verdict, icon = "result-down", "Predicted decrease", "📉"
            else:
                css_class, verdict, icon = "result-flat", "No significant change predicted", "➖"

            # 1) Predicted result — shown first, bigger font, colour-coded.
            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <p class="result-headline">{icon} {fmt_num(predicted_price, 2)}</p>
                    <p class="result-caption">
                        {verdict} · {price_change:+,.2f} ({percentage_change:+.2f}%) vs. {chosen_date} close of {fmt_num(previous_price,2)}
                        <span class="badge">{model_choice}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if price_change < 0:
                st.warning("⚠️ The model predicts a **decline** for the next trading day. Treat this forecast with extra caution.")

            # 2) Actual price on record — shown afterwards, in its own callout
            #    (kept separate from the predicted headline above).
            actual_next = selected_row.get("Actual_NextClose", np.nan)
            if pd.notna(actual_next):
                err = predicted_price - actual_next
                st.markdown(
                    f"""
                    <div class="actual-box">
                        📚 Actual next-day close on record: <b>{fmt_num(actual_next, 2)}</b>
                        &nbsp;·&nbsp; model error: {err:+,.2f}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3) Then the three summary stat cards.
            c1, c2, c3 = st.columns(3)
            with c1:
                stat_card("Predicted Next-Day Price", fmt_num(predicted_price, 2))
            with c2:
                stat_card("Price Change", f"{price_change:+,.2f}")
            with c3:
                stat_card("Percentage Change", f"{percentage_change:+.2f}%")

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))


# =========================================================
# TAB 3 — MODEL COMPARISON
# =========================================================

with tab3:
    st.header("Model Performance Comparison")
    st.write("How the four trained pipelines perform on the chronological, held-out test set.")

    display_comparison = model_comparison.sort_values("RMSE").reset_index(drop=True)

    def highlight_best(row):
        return ["background-color: rgba(184,134,11,0.14)" if row["Model"] == best_model_name else "" for _ in row]

    st.dataframe(
        display_comparison.style.apply(highlight_best, axis=1).format(
            {"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.4f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["MAE", "RMSE", "R²"]):
        with col:
            fig = go.Figure()
            colors = [
                PALETTE["gold"] if m == best_model_name else "#C9BFA5"
                for m in display_comparison["Model"]
            ]
            fig.add_trace(go.Bar(
                x=display_comparison["Model"], y=display_comparison[metric],
                marker_color=colors, text=display_comparison[metric].round(3), textposition="outside",
            ))
            fig.update_layout(
                title=f"Testing {metric}",
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PALETTE["ivory"]),
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
            )
            fig.update_xaxes(tickangle=-25, gridcolor=PALETTE["grid"])
            fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f"""
        <div class="insight-box">
        <b>Why do the tree-based models look so much worse?</b><br>
        Decision Tree, Random Forest and Gradient Boosting all show a <b>negative R²</b> on the test
        set — worse than simply guessing the average price. This isn't a bug: gold prices in the
        test period (2023–2026) run well above anything the trees saw during training (2014–2023),
        and tree-based models <b>cannot extrapolate</b> beyond the value ranges they were trained on
        — their leaves just predict the highest price bucket they learned. Linear Regression, by
        contrast, can extrapolate along a trend line, which is why it stays highly accurate
        (R² ≈ 0.998) even as gold pushes into new territory.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Compare All Models on a Single Date")
    st.write("Pick a historical date — since the actual next-day close is on record, you can see exactly how each model would have called it.")

    backtest_df = df_features.dropna(subset=feature_cols + ["Actual_NextClose"]).copy()
    backtest_dates = sorted(backtest_df["Date"].dt.date.tolist(), reverse=True)
    bt_min, bt_max = backtest_dates[-1], backtest_dates[0]
    backtest_dates_set = set(backtest_dates)

    picked_compare_date = st.date_input(
        "Select a date to backtest",
        value=bt_max, min_value=bt_min, max_value=bt_max,
        key="compare_date_input",
    )
    if picked_compare_date in backtest_dates_set:
        compare_date = picked_compare_date
    else:
        earlier = [d for d in backtest_dates if d <= picked_compare_date]
        compare_date = earlier[0] if earlier else bt_min
        st.caption(f"⚠️ No backtest data on {picked_compare_date} — snapped to nearest trading day **{compare_date}**.")

    render_update_flash(f"Comparison refreshed for {compare_date}")
    compare_row = backtest_df[backtest_df["Date"].dt.date == compare_date].iloc[0]
    compare_input = pd.DataFrame([compare_row[feature_cols].to_dict()])[feature_cols]

    preds = {}
    for name, m in MODEL_MAP.items():
        try:
            preds[name] = float(m.predict(compare_input)[0])
        except Exception:
            preds[name] = np.nan

    actual_val = float(compare_row["Actual_NextClose"])

    fig2 = go.Figure()
    names = list(preds.keys())
    values = [preds[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    fig2.add_trace(go.Bar(x=names, y=values, marker_color=colors, name="Predicted",
                           text=[fmt_num(v, 2) for v in values], textposition="outside"))
    fig2.add_hline(y=actual_val, line_dash="dash", line_color=PALETTE["ivory"],
                    annotation_text=f"Actual: {fmt_num(actual_val,2)}", annotation_position="top left")
    fig2.update_layout(
        title=f"Predicted Next-Day Close vs. Actual — for {compare_date}",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ivory"]),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    fig2.update_xaxes(gridcolor=PALETTE["grid"])
    fig2.update_yaxes(gridcolor=PALETTE["grid"])
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    abs_pct_errors = [abs((v - actual_val) / actual_val) * 100 for v in values]
    best_row_idx = int(np.argmin(abs_pct_errors))

    rows_html = ""
    for i, name in enumerate(names):
        err = values[i] - actual_val
        highlight_cls = "highlight-row" if i == best_row_idx else ""
        trophy = " 🏆" if i == best_row_idx else ""
        rows_html += f"""
        <tr class="{highlight_cls}">
            <td>{name}{trophy}</td>
            <td>{fmt_num(values[i], 2)}</td>
            <td>{fmt_num(actual_val, 2)}</td>
            <td>{err:+,.2f}</td>
            <td>{abs_pct_errors[i]:.2f}%</td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="big-table">
            <thead>
                <tr><th>Model</th><th>Predicted</th><th>Actual</th><th>Error</th><th>Abs % Error</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "S.Gold Predict | BMDS2003 Data Science Project | "
    "For academic demonstration purposes only — not financial advice."
)

