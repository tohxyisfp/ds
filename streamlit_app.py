"""
S.Gold Predict — Streamlit Deployment
BMDS2003 Data Science Assignment
Daily Gold Price Prediction

Models:
- Linear Regression
- Decision Tree
- Gradient Boosting
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="S.Gold Predict",
    page_icon="🪙",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🪙 S.Gold Predict")

st.caption(
    "Next-Trading-Day Gold Price Prediction — BMDS2003 Data Science Project"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    lr_model = joblib.load(
        "models/linear_regression_model.pkl"
    )

    dt_model = joblib.load(
        "models/decision_tree_model.pkl"
    )

    gb_model = joblib.load(
        "models/gradient_boosting_model.pkl"
    )

    return lr_model, dt_model, gb_model


# =========================================================
# LOAD PREPROCESSORS
# =========================================================

@st.cache_resource
def load_preprocessors():

    lr_preprocessor = joblib.load(
        "models/linear_regression_preprocessor.pkl"
    )

    dt_preprocessor = joblib.load(
        "models/decision_tree_preprocessor.pkl"
    )

    gb_preprocessor = joblib.load(
        "models/gradient_boosting_preprocessor.pkl"
    )

    return (
        lr_preprocessor,
        dt_preprocessor,
        gb_preprocessor
    )


# =========================================================
# LOAD GOLD PRICE DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Gold_Price.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =========================================================
# LOAD FILES
# =========================================================

try:

    lr_model, dt_model, gb_model = load_models()

    (
        lr_preprocessor,
        dt_preprocessor,
        gb_preprocessor
    ) = load_preprocessors()

    df = load_data()

except Exception as e:

    st.error("Unable to load the model files or dataset.")

    st.code(str(e))

    st.stop()


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(data):

    data = data.copy()

    # -----------------------------
    # Basic features
    # -----------------------------

    data["calc_chg"] = (
        data["Price"].pct_change() * 100
    )

    data["chg_diff"] = (
        data["Chg%"] - data["calc_chg"]
    )

    data["day_gap"] = (
        data["Open"] - data["Price"].shift(1)
    )

    data["Daily_Range"] = (
        data["High"] - data["Low"]
    )

    data["Daily_Range_Pct"] = (
        data["Daily_Range"] /
        data["Price"] * 100
    )

    # -----------------------------
    # Date features
    # -----------------------------

    data["Year"] = data["Date"].dt.year

    data["Month"] = data["Date"].dt.month

    data["DayOfWeek"] = (
        data["Date"].dt.day_name()
    )

    # -----------------------------
    # Moving averages
    # -----------------------------

    data["MA7"] = (
        data["Price"]
        .rolling(7)
        .mean()
    )

    data["MA30"] = (
        data["Price"]
        .rolling(30)
        .mean()
    )

    # -----------------------------
    # Volatility
    # -----------------------------

    data["Volatility_7"] = (
        data["Price"]
        .rolling(7)
        .std()
    )

    # -----------------------------
    # Lag features
    # -----------------------------

    data["Lag1_Price"] = (
        data["Price"].shift(1)
    )

    data["Lag1_Volume"] = (
        data["Volume"].shift(1)
    )

    return data


df_features = create_features(df)


# =========================================================
# DEFINE FEATURES
# =========================================================

feature_cols = [
    "Price",
    "Open",
    "High",
    "Low",
    "Volume",
    "Chg%",
    "calc_chg",
    "chg_diff",
    "day_gap",
    "Daily_Range",
    "Daily_Range_Pct",
    "Year",
    "Month",
    "DayOfWeek",
    "MA7",
    "MA30",
    "Volatility_7",
    "Lag1_Price",
    "Lag1_Volume"
]


# =========================================================
# MODEL MAP
# =========================================================

MODEL_MAP = {

    "Linear Regression": (
        lr_model,
        lr_preprocessor
    ),

    "Decision Tree": (
        dt_model,
        dt_preprocessor
    ),

    "Gradient Boosting": (
        gb_model,
        gb_preprocessor
    )
}


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(
    [
        "📈 Trading Chart",
        "🔮 Price Prediction"
    ]
)


# =========================================================
# TAB 1 — TRADING CHART
# =========================================================

with tab1:

    st.header("Gold Price Historical Trading Chart")

    min_date = (
        df["Date"]
        .min()
        .date()
    )

    max_date = (
        df["Date"]
        .max()
        .date()
    )

    col1, col2 = st.columns(2)

    with col1:

        start_date = st.date_input(
            "Start Date",
            value=max(
                min_date,
                max_date -
                datetime.timedelta(days=365)
            ),
            min_value=min_date,
            max_value=max_date
        )

    with col2:

        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

    if start_date > end_date:

        st.warning(
            "Start Date must be earlier than End Date."
        )

    else:

        mask = (
            (df["Date"].dt.date >= start_date)
            &
            (df["Date"].dt.date <= end_date)
        )

        chart_df = (
            df.loc[
                mask,
                ["Date", "Price"]
            ]
            .set_index("Date")
        )

        st.line_chart(chart_df)

        st.subheader(
            "Recent Trading Data"
        )

        table_df = df.loc[
            mask,
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Price",
                "Volume",
                "Chg%"
            ]
        ].tail(20)

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 2 — PREDICTION
# =========================================================

with tab2:

    st.header(
        "Predict Next Trading Day's Gold Price"
    )

    st.write(
        "Select a date to automatically fill the "
        "model input features."
    )

    # -----------------------------
    # Select model
    # -----------------------------

    model_choice = st.selectbox(
        "Choose a Model",
        list(MODEL_MAP.keys())
    )

    # -----------------------------
    # Select date
    # -----------------------------

    valid_features = df_features.dropna(
        subset=feature_cols
    ).copy()

    available_dates = (
        valid_features["Date"]
        .dt.date
        .tolist()
    )

    chosen_date = st.selectbox(
        "Select Date",
        available_dates[-60:]
    )

    # -----------------------------
    # Get selected row
    # -----------------------------

    row = valid_features[
        valid_features["Date"].dt.date ==
        chosen_date
    ].iloc[0]

    # -----------------------------
    # Input features
    # -----------------------------

    st.subheader(
        "Input Features"
    )

    user_input = {}

    cols = st.columns(3)

    for i, feature in enumerate(feature_cols):

        with cols[i % 3]:

            if feature == "DayOfWeek":

                # Convert weekday to categorical value
                user_input[feature] = row[feature]

                st.text_input(
                    feature,
                    value=str(row[feature]),
                    disabled=True
                )

            else:

                user_input[feature] = st.number_input(
                    feature,
                    value=float(row[feature]),
                    format="%.4f"
                )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "🔮 Predict Next-Day Price",
        type="primary",
        use_container_width=True
    ):

        model, preprocessor = MODEL_MAP[
            model_choice
        ]

        # -----------------------------
        # Create input dataframe
        # -----------------------------

        input_df = pd.DataFrame(
            [user_input]
        )

        input_df = input_df[
            feature_cols
        ]

        # -----------------------------
        # Encode features
        # -----------------------------

        input_encoded = preprocessor.transform(
            input_df
        )

        # -----------------------------
        # Make prediction
        # -----------------------------

        predicted_price = model.predict(
            input_encoded
        )[0]

        # Previous closing price

        previous_price = float(
            row["Price"]
        )

        price_change = (
            predicted_price -
            previous_price
        )

        percentage_change = (
            price_change /
            previous_price
        ) * 100


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Next-Day Price",
                f"{predicted_price:,.2f}"
            )

        with result_col2:

            st.metric(
                "Price Change",
                f"{price_change:+,.2f}"
            )

        with result_col3:

            st.metric(
                "Percentage Change",
                f"{percentage_change:+.2f}%"
            )

        if price_change > 0:

            st.success(
                "📈 The model predicts an increase "
                "in the next trading day's gold price."
            )

        elif price_change < 0:

            st.warning(
                "📉 The model predicts a decrease "
                "in the next trading day's gold price."
            )

        else:

            st.info(
                "The model predicts little or no "
                "change in the gold price."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "S.Gold Predict | BMDS2003 Data Science Project | "
    "For academic demonstration purposes only — "
    "not financial advice."
)
