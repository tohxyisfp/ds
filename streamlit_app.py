import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Gold Price Prediction Dashboard")
st.write(
    "Compare Linear Regression, Decision Tree, and Gradient Boosting "
    "models for predicting the next gold closing price."
)

st.divider()

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
# LOAD TEST DATA
# =========================================================

@st.cache_data
def load_test_data():

    y_test_lr = joblib.load(
        "data/y_test_flat.pkl"
    )

    y_test_dt = joblib.load(
        "data/dt_y_test_flat.pkl"
    )

    y_test_gb = joblib.load(
        "data/gb_y_test_flat.pkl"
    )

    return y_test_lr, y_test_dt, y_test_gb


# =========================================================
# LOAD PREDICTIONS
# =========================================================

@st.cache_data
def load_predictions():

    lr_pred = joblib.load(
        "data/linear_regression_y_pred.pkl"
    )

    dt_pred = joblib.load(
        "data/decision_tree_y_pred.pkl"
    )

    gb_pred = joblib.load(
        "data/gradient_boosting_y_pred.pkl"
    )

    return lr_pred, dt_pred, gb_pred


# =========================================================
# LOAD ALL FILES
# =========================================================

try:

    lr_model, dt_model, gb_model = load_models()

    (
        lr_preprocessor,
        dt_preprocessor,
        gb_preprocessor
    ) = load_preprocessors()

    (
        y_test_lr,
        y_test_dt,
        y_test_gb
    ) = load_test_data()

    (
        lr_pred,
        dt_pred,
        gb_pred
    ) = load_predictions()

except Exception as e:

    st.error("Unable to load the model files.")

    st.code(str(e))

    st.stop()


# =========================================================
# CALCULATE MODEL PERFORMANCE
# =========================================================

def calculate_metrics(y_actual, y_pred):

    mae = mean_absolute_error(
        y_actual,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            y_pred
        )
    )

    r2 = r2_score(
        y_actual,
        y_pred
    )

    return mae, rmse, r2


lr_mae, lr_rmse, lr_r2 = calculate_metrics(
    y_test_lr,
    lr_pred
)

dt_mae, dt_rmse, dt_r2 = calculate_metrics(
    y_test_dt,
    dt_pred
)

gb_mae, gb_rmse, gb_r2 = calculate_metrics(
    y_test_gb,
    gb_pred
)


# =========================================================
# MODEL COMPARISON
# =========================================================

st.header("Model Performance Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Gradient Boosting"
    ],

    "MAE": [
        lr_mae,
        dt_mae,
        gb_mae
    ],

    "RMSE": [
        lr_rmse,
        dt_rmse,
        gb_rmse
    ],

    "R²": [
        lr_r2,
        dt_r2,
        gb_r2
    ]
})

st.dataframe(
    comparison_df.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# BEST MODEL
# =========================================================

best_model_index = comparison_df["RMSE"].idxmin()

best_model = comparison_df.loc[
    best_model_index,
    "Model"
]

st.success(
    f"🏆 Best performing model based on RMSE: **{best_model}**"
)


# =========================================================
# METRIC CARDS
# =========================================================

st.subheader("Model Metrics")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Linear Regression RMSE",
        f"{lr_rmse:.4f}"
    )

    st.metric(
        "Linear Regression R²",
        f"{lr_r2:.4f}"
    )


with col2:

    st.metric(
        "Decision Tree RMSE",
        f"{dt_rmse:.4f}"
    )

    st.metric(
        "Decision Tree R²",
        f"{dt_r2:.4f}"
    )


with col3:

    st.metric(
        "Gradient Boosting RMSE",
        f"{gb_rmse:.4f}"
    )

    st.metric(
        "Gradient Boosting R²",
        f"{gb_r2:.4f}"
    )


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.divider()

st.header("Actual vs Predicted Gold Price")

model_choice = st.selectbox(
    "Select a model:",
    [
        "Linear Regression",
        "Decision Tree",
        "Gradient Boosting"
    ]
)


if model_choice == "Linear Regression":

    selected_actual = y_test_lr
    selected_pred = lr_pred

elif model_choice == "Decision Tree":

    selected_actual = y_test_dt
    selected_pred = dt_pred

else:

    selected_actual = y_test_gb
    selected_pred = gb_pred


# Create comparison dataframe

prediction_df = pd.DataFrame({
    "Actual Price": selected_actual,
    "Predicted Price": selected_pred
})

# Use last 100 observations for clearer visualization

display_df = prediction_df.tail(100)

st.line_chart(
    display_df
)


# =========================================================
# PREDICTION TABLE
# =========================================================

st.subheader("Prediction Results")

st.dataframe(
    display_df.style.format(
        {
            "Actual Price": "{:.2f}",
            "Predicted Price": "{:.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD PREDICTIONS
# =========================================================

csv = prediction_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Prediction Results",
    data=csv,
    file_name="gold_price_predictions.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Gold Price Prediction Dashboard | "
    "Linear Regression | Decision Tree | Gradient Boosting"
)
