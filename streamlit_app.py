import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(
    page_title="S.Gold Predict",
    page_icon="🪙",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    lr_model = joblib.load(
        "models/linear_regression_pipeline.pkl"
    )

    dt_model = joblib.load(
        "models/decision_tree_pipeline.pkl"
    )

    gb_model = joblib.load(
        "models/gradient_boosting_pipeline.pkl"
    )

    return lr_model, dt_model, gb_model


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("Gold_Price.csv")

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return df


# =========================================================
# CREATE FEATURES
# =========================================================

def create_features(data):

    data = data.copy()

    data["calc_chg"] = (
        data["Price"].pct_change() * 100
    )

    data["chg_diff"] = (
        data["Chg%"] - data["calc_chg"]
    )

    data["day_gap"] = (
        data["Open"] -
        data["Price"].shift(1)
    )

    data["Daily_Range"] = (
        data["High"] -
        data["Low"]
    )

    data["Daily_Range_Pct"] = (
        data["Daily_Range"] /
        data["Price"] * 100
    )

    data["Year"] = (
        data["Date"].dt.year
    )

    data["Month"] = (
        data["Date"].dt.month
    )

    data["DayOfWeek"] = (
        data["Date"].dt.day_name()
    )

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

    data["Volatility_7"] = (
        data["Price"]
        .rolling(7)
        .std()
    )

    data["Lag1_Price"] = (
        data["Price"].shift(1)
    )

    data["Lag1_Volume"] = (
        data["Volume"].shift(1)
    )

    return data


# =========================================================
# LOAD EVERYTHING
# =========================================================

try:

    lr_model, dt_model, gb_model = load_models()

    df = load_data()

    df_features = create_features(df)

except Exception as e:

    st.error(
        "Unable to load the model files or dataset."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# FEATURES
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

    "Linear Regression": lr_model,

    "Decision Tree": dt_model,

    "Gradient Boosting": gb_model

}


# =========================================================
# TITLE
# =========================================================

st.title("🪙 S.Gold Predict")

st.caption(
    "Next-Trading-Day Gold Price Prediction — "
    "BMDS2003 Data Science Project"
)


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
# TAB 1
# =========================================================

with tab1:

    st.header(
        "Gold Price Historical Trading Chart"
    )

    min_date = (
        df["Date"].min().date()
    )

    max_date = (
        df["Date"].max().date()
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
            "Trading Data"
        )

        display_df = df.loc[
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
            display_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header(
        "Predict Next Trading Day's Gold Price"
    )

    st.write(
        "Select an existing date to automatically "
        "fill the model features."
    )

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------

    model_choice = st.selectbox(
        "Choose Model",
        list(MODEL_MAP.keys())
    )

    model = MODEL_MAP[
        model_choice
    ]

    # -----------------------------------------------------
    # Valid rows
    # -----------------------------------------------------

    valid_df = df_features.dropna(
        subset=feature_cols
    ).copy()

    available_dates = (
        valid_df["Date"]
        .dt.date
        .tolist()
    )

    # -----------------------------------------------------
    # Date selection
    # -----------------------------------------------------

    chosen_date = st.selectbox(
        "Select Date",
        available_dates[-60:]
    )

    selected_row = valid_df[
        valid_df["Date"].dt.date ==
        chosen_date
    ].iloc[0]

    # -----------------------------------------------------
    # Input features
    # -----------------------------------------------------

    st.subheader(
        "Input Features"
    )

    user_input = {}

    cols = st.columns(3)

    for i, feature in enumerate(feature_cols):

        with cols[i % 3]:

            if feature == "DayOfWeek":

                user_input[feature] = (
                    selected_row[feature]
                )

                st.text_input(
                    feature,
                    value=str(
                        selected_row[feature]
                    ),
                    disabled=True
                )

            else:

                user_input[feature] = st.number_input(
                    feature,
                    value=float(
                        selected_row[feature]
                    ),
                    format="%.4f"
                )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict Next-Day Price",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_input]
        )

        input_df = input_df[
            feature_cols
        ]

        try:

            predicted_price = model.predict(
                input_df
            )[0]

            previous_price = float(
                selected_row["Price"]
            )

            price_change = (
                predicted_price -
                previous_price
            )

            percentage_change = (
                price_change /
                previous_price
            ) * 100

            # -------------------------------------------------
            # Results
            # -------------------------------------------------

            st.divider()

            st.subheader(
                "Prediction Result"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Next-Day Price",
                    f"{predicted_price:,.2f}"
                )

            with col2:

                st.metric(
                    "Price Change",
                    f"{price_change:+,.2f}"
                )

            with col3:

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
                    "The model predicts no significant "
                    "change in the gold price."
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "S.Gold Predict | BMDS2003 Data Science Project | "
    "For academic demonstration purposes only — "
    "not financial advice."
)
