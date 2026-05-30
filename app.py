import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Stock Price Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        os.path.join(
            BASE_DIR,
            "stock_rnn_model.h5"
        )
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "scaler.pkl"
        )
    )

    return model, scaler

model, scaler = load_artifacts()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center;color:#1E88E5;'>
📈 AI Stock Price Forecasting System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;color:gray;'>
SimpleRNN-Based Next Day Stock Price Prediction
</h4>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "📊 Previous 30 Days Closing Prices"
)

prices = []

for i in range(30):

    value = st.sidebar.number_input(
        f"Day {i+1}",
        min_value=0.0,
        value=100.0
    )

    prices.append(value)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(
    "🚀 Predict Next Day Price",
    use_container_width=True
):

    prices_array = np.array(
        prices
    ).reshape(-1, 1)

    scaled_data = scaler.transform(
        prices_array
    )

    X = scaled_data.reshape(
        1,
        30,
        1
    )

    prediction = model.predict(
        X,
        verbose=0
    )

    predicted_price = scaler.inverse_transform(
        prediction
    )[0][0]

    last_price = prices[-1]

    change = (
        predicted_price - last_price
    )

    percent_change = (
        change / last_price
    ) * 100

    # ======================================
    # DASHBOARD
    # ======================================

    st.subheader(
        "📊 Forecast Dashboard"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Current Price",
            f"₹{last_price:.2f}"
        )

    with c2:

        st.metric(
            "Predicted Price",
            f"₹{predicted_price:.2f}"
        )

    with c3:

        st.metric(
            "Expected Change",
            f"{percent_change:.2f}%"
        )

    st.divider()

    # ======================================
    # INVESTMENT SIGNAL
    # ======================================

    if percent_change > 2:

        st.success(
            "🟢 BUY SIGNAL"
        )

    elif percent_change < -2:

        st.error(
            "🔴 SELL SIGNAL"
        )

    else:

        st.warning(
            "🟡 HOLD SIGNAL"
        )

    # ======================================
    # CHART
    # ======================================

    chart_data = prices.copy()

    chart_data.append(
        predicted_price
    )

    days = list(
        range(1, 32)
    )

    fig = plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        days,
        chart_data,
        marker="o"
    )

    plt.xlabel(
        "Days"
    )

    plt.ylabel(
        "Price"
    )

    plt.title(
        "Stock Price Forecast"
    )

    st.pyplot(fig)

    # ======================================
    # ANALYTICS
    # ======================================

    st.subheader(
        "📈 Market Analytics"
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.info(
            f"30-Day Average: ₹{np.mean(prices):.2f}"
        )

    with a2:

        st.info(
            f"Maximum Price: ₹{np.max(prices):.2f}"
        )

    with a3:

        st.info(
            f"Minimum Price: ₹{np.min(prices):.2f}"
        )

    st.divider()

    # ======================================
    # AI INSIGHTS
    # ======================================

    st.subheader(
        "🤖 AI Insights"
    )

    if percent_change > 5:

        st.success(
            "Strong upward trend predicted."
        )

    elif percent_change > 0:

        st.info(
            "Moderate growth expected."
        )

    elif percent_change < -5:

        st.error(
            "Significant downward trend predicted."
        )

    else:

        st.warning(
            "Market expected to remain stable."
        )

# ==========================================
# INFO SECTION
# ==========================================

st.divider()

st.subheader(
    "ℹ Model Information"
)

st.write(
    """
    • Deep Learning Model: SimpleRNN
    
    • Input: Previous 30 Days Closing Prices
    
    • Output: Next Day Predicted Price
    
    • Application: Stock Forecasting & Trend Analysis
    """
)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "AI Stock Price Forecasting System | SimpleRNN Deep Learning Model"
)
