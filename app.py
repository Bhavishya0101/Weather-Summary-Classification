import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load Dataset + Model
# ---------------------------
df = pd.read_csv("weatherHistory.csv")
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title("🌦 Weather Summary Prediction")

# ---------------------------
# Feature List (MUST match training order)
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
# Predict Button
# ---------------------------
if st.button("Predict"):

    try:
        # Create DataFrame with correct column names
        input_df = pd.DataFrame([[
            temp,
            humidity,
            wind_speed,
            wind_bearing,
            visibility,
            pressure
        ]], columns=features)

        # Model prediction
        pred_encoded = model.predict(input_df)

        # Convert numeric label back to actual Summary text
        predicted_summary = le.inverse_transform(pred_encoded)[0]

        st.success(f"🌤 Predicted Weather Summary: **{predicted_summary}**")

        # Optional: Show probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            class_labels = le.inverse_transform(np.arange(len(probabilities)))

            prob_df = pd.DataFrame({
                "Weather Summary": class_labels,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            st.subheader("Prediction Confidence")
            st.dataframe(prob_df.head(5))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
