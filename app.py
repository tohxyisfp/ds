"""
S.Gold Predict — Streamlit Deployment Prototype
BMDS2003 Data Science Assignment — Daily Gold Price Prediction

Run with:  streamlit run app.py
Requires: models/linear_regression.pkl, models/knn.pkl, models/random_forest.pkl,
          models/gradient_boosting.pkl, models/scaler.pkl, models/feature_cols.json
          and Gold_Price.csv in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import datetime

st.set_page_config(page_title="S.Gold Predict", page_icon="🪙", layout="centered")

# ---------------- Load artefacts ----------------
@st.cache_resource
def load_artifacts():
    with open('models/linear_regression.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('models/knn.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open('models/random_forest.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/gradient_boosting.pkl', 'rb') as f:
        gb = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/feature_cols.json') as f:
        feature_cols = json.load(f)
    return lr, knn, rf, gb, scaler, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('Gold_Price.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['Price_lag1'] = df['Price'].shift(1)
    df['Price_lag2'] = df['Price'].shift(2)
    df['Price_lag3'] = df['Price'].shift(3)
    df['Price_lag5'] = df['Price'].shift(5)
    df['Open_lag1']  = df['Open'].shift(1)
    df['High_lag1']  = df['High'].shift(1)
    df['Low_lag1']   = df['Low'].shift(1)
    df['Volume_lag1']= df['Volume'].shift(1)
    df['Chg_lag1']   = df['Chg%'].shift(1)
    df['MA_5']  = df['Price'].shift(1).rolling(5).mean()
    df['MA_10'] = df['Price'].shift(1).rolling(10).mean()
    df['MA_20'] = df['Price'].shift(1).rolling(20).mean()
    df['STD_5'] = df['Price'].shift(1).rolling(5).std()
    df['Volatility_10'] = df['Price'].shift(1).rolling(10).std()
    df['Momentum_5'] = df['Price_lag1'] - df['Price_lag5']
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df.dropna().reset_index(drop=True)

lr, knn, rf, gb, scaler, feature_cols = load_artifacts()
df = load_data()

MODEL_MAP = {
    "Linear Regression (Baseline)": (lr, True),
    "K-Nearest Neighbours": (knn, True),
    "Random Forest": (rf, False),
    "Gradient Boosting": (gb, False),
}

# ---------------- UI ----------------
st.title("🪙 S.Gold Predict")
st.caption("Next-trading-day Gold Price Predictor — BMDS2003 Data Science Project")

tab1, tab2 = st.tabs(["📈 Trading Chart", "🔮 Single Day Prediction"])

# ---------------- TAB 1: Trading chart ----------------
with tab1:
    st.subheader("Gold Price: Historical Trading Chart")
    min_d, max_d = df['Date'].min().date(), df['Date'].max().date()
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start Date", value=max_d - datetime.timedelta(days=365), min_value=min_d, max_value=max_d)
    end_date = c2.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)

    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    chart_df = df.loc[mask, ['Date', 'Price']].set_index('Date')
    st.line_chart(chart_df)
    st.dataframe(df.loc[mask, ['Date','Open','High','Low','Price','Volume','Chg%']].tail(20), use_container_width=True)

# ---------------- TAB 2: Single-day prediction ----------------
with tab2:
    st.subheader("Predict Next Trading Day's Gold Price")
    st.write("Select an existing date to auto-fill the required indicators, or enter your own values.")

    model_choice = st.selectbox("Choose a model", list(MODEL_MAP.keys()), index=0)

    available_dates = df['Date'].dt.date.tolist()
    chosen_date = st.selectbox("Auto-fill from existing date (optional)", options=["-- manual entry --"] + [str(d) for d in available_dates[-60:]])

    if chosen_date != "-- manual entry --":
        row = df[df['Date'].dt.date == pd.to_datetime(chosen_date).date()].iloc[0]
        defaults = {c: float(row[c]) for c in feature_cols}
    else:
        last = df.iloc[-1]
        defaults = {c: float(last[c]) for c in feature_cols}

    st.markdown("**Input Features (auto-filled — adjust if needed):**")
    cols = st.columns(3)
    user_input = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            user_input[feat] = st.number_input(feat, value=defaults[feat], format="%.3f")

    if st.button("Predict", type="primary"):
        model, needs_scaling = MODEL_MAP[model_choice]
        x = pd.DataFrame([user_input])[feature_cols]
        if needs_scaling:
            x_in = scaler.transform(x)
        else:
            x_in = x
        pred_change = model.predict(x_in)[0]
        predicted_price = user_input['Price_lag1'] + pred_change

        st.success(f"💰 Predicted next-day gold price: **{predicted_price:,.2f}**")
        st.info(f"Predicted change from previous close: **{pred_change:+.2f}** "
                f"({pred_change/user_input['Price_lag1']*100:+.2f}%)")

st.markdown("---")
st.caption("Model trained on 2014–2026 daily OHLCV gold price data using lagged, "
           "no-leakage features. For academic demonstration purposes only — not financial advice.")
