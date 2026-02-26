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
# Feature Names (IMPORTANT)
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
        # Convert input into DataFrame (VERY IMPORTANT)
        input_data = pd.DataFrame([[
            temp,
            humidity,
            wind_speed,
            wind_bearing,
            visibility,
            pressure
        ]], columns=features)

        # Predict
        pred = model.predict(input_data)

        # Convert back to original Summary label
        predicted_label = le.inverse_transform(pred)[0]

        st.success(f"🌤 Predicted Weather Summary: **{predicted_label}**")

        # Optional: Show probabilities (if model supports it)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            class_labels = le.inverse_transform(np.arange(len(probs)))

            prob_df = pd.DataFrame({
                "Weather Summary": class_labels,
                "Probability": probs
            }).sort_values(by="Probability", ascending=False)

            st.subheader("Prediction Confidence")
            st.dataframe(prob_df.head(5))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
