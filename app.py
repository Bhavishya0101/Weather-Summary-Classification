import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load Model Files
# ---------------------------
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title("🌦 Weather Summary Prediction")

# ---------------------------
# Feature Names (MUST match training)
# ---------------------------
features = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

# ---------------------------
# User Inputs
# ---------------------------
temp = st.number_input("Temperature (C)", value=20.0)
humidity = st.slider("Humidity", 0.0, 1.0, value=0.5)
wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
wind_bearing = st.number_input("Wind Bearing (degrees)", value=100.0)
visibility = st.number_input("Visibility (km)", value=10.0)
pressure = st.number_input("Pressure (millibars)", value=1000.0)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):

    try:
        # Create DataFrame with correct column names
        input_data = pd.DataFrame([[
            temp,
            humidity,
            wind_speed,
            wind_bearing,
            visibility,
            pressure
        ]], columns=features)

        # 🔥 NO SCALER USED
        pred = model.predict(input_data)

        result = le.inverse_transform(pred)[0]

        st.success(f"🌤 Predicted Weather Summary: {result}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
