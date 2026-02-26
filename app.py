import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load Files
# ---------------------------
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")
df = pd.read_csv("weatherHistory.csv")

st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title("🌦 Weather Summary Prediction")

# ---------------------------
# Feature List
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
# Prediction Button
# ---------------------------
if st.button("Predict"):

    try:
        # Create input DataFrame
        input_data = pd.DataFrame([[
            temp,
            humidity,
            wind_speed,
            wind_bearing,
            visibility,
            pressure
        ]], columns=features)

        # ---------------------------
        # 1️⃣ CHECK EXACT MATCH IN DATASET
        # ---------------------------
        matched_row = df[
            (df["Temperature (C)"] == temp) &
            (df["Humidity"] == humidity) &
            (df["Wind Speed (km/h)"] == wind_speed) &
            (df["Wind Bearing (degrees)"] == wind_bearing) &
            (df["Visibility (km)"] == visibility) &
            (df["Pressure (millibars)"] == pressure)
        ]

        if not matched_row.empty:
            exact_summary = matched_row.iloc[0]["Summary"]
            st.success(f"📊 Exact Dataset Match Found!")
            st.success(f"🌤 Weather Summary: **{exact_summary}**")

        else:
            # ---------------------------
            # 2️⃣ OTHERWISE USE ML MODEL
            # ---------------------------
            pred = model.predict(input_data)
            predicted_label = le.inverse_transform(pred)[0]

            st.warning("⚠ No exact dataset match. Showing ML prediction.")
            st.success(f"🤖 Predicted Weather Summary: **{predicted_label}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
