import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load files
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🌦 Weather Summary Prediction")

# Feature order MUST match training
features = [
    "Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

temp = st.number_input("Temperature (C)", value=20.0)
humidity = st.slider("Humidity", 0.0, 1.0, value=0.5)
wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
wind_bearing = st.number_input("Wind Bearing (degrees)", value=100.0)
visibility = st.number_input("Visibility (km)", value=10.0)
pressure = st.number_input("Pressure (millibars)", value=1000.0)

if st.button("Predict"):
    try:
        # Convert to DataFrame (important for correct feature order)
        input_data = pd.DataFrame([[ 
            temp, 
            humidity, 
            wind_speed, 
            wind_bearing, 
            visibility, 
            pressure
        ]], columns=features)

        # Direct prediction (NO SCALER)
        pred = model.predict(input_data)

        # Convert encoded output back to original Summary text
        summary = le.inverse_transform(pred)[0]

        st.success(f"🌤 Predicted Weather Summary: {summary}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
