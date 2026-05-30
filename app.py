import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Stock Price Forecasting System",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():

    model_path = os.path.join(
        BASE_DIR,
        "stock_rnn_model.h5"
    )

    scaler_path = os.path.join(
        BASE_DIR,
        "scaler.pkl"
    )

    # IMPORTANT FIX
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    scaler = joblib.load(
        scaler_path
    )

    return model, scaler

try:
    model, scaler = load_artifacts()

except Exception as e:

    st.error(
        f"Model Loading Error: {e}"
    )

    st.stop()

st.title(
    "📈 AI Stock Price Forecasting System"
)

st.markdown(
    "### SimpleRNN Based Next-Day Stock Price Prediction"
)

st.divider()

st.sidebar.header(
    "Previous 30 Days Prices"
)

prices = []

for i in range(30):

    value = st.sidebar.number_input(
        f"Day {i+1}",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    prices.append(value)

if st.button(
    "🚀 Predict Next Day Price",
    use_container_width=True
):

    try:

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

        current_price = prices[-1]

        change = (
            predicted_price -
            current_price
        )

        change_pct = (
            change /
            current_price
        ) * 100

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Price",
                f"₹{current_price:.2f}"
            )

        with col2:

            st.metric(
                "Predicted Price",
                f"₹{predicted_price:.2f}"
            )

        with col3:

            st.metric(
                "Change %",
                f"{change_pct:.2f}%"
            )

        st.divider()

        if change_pct > 2:

            st.success(
                "🟢 BUY SIGNAL"
            )

        elif change_pct < -2:

            st.error(
                "🔴 SELL SIGNAL"
            )

        else:

            st.warning(
                "🟡 HOLD SIGNAL"
            )

        chart_prices = prices.copy()
        chart_prices.append(
            predicted_price
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            range(1, 32),
            chart_prices,
            marker="o"
        )

        ax.set_title(
            "Stock Forecast"
        )

        ax.set_xlabel(
            "Days"
        )

        ax.set_ylabel(
            "Price"
        )

        st.pyplot(fig)

        st.subheader(
            "📊 Market Analytics"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.info(
                f"Average Price: ₹{np.mean(prices):.2f}"
            )

        with c2:

            st.info(
                f"Maximum Price: ₹{np.max(prices):.2f}"
            )

        with c3:

            st.info(
                f"Minimum Price: ₹{np.min(prices):.2f}"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

st.markdown("---")

st.caption(
    "AI Stock Price Forecasting System | SimpleRNN"
)
