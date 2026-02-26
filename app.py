import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Load Model Files
# ---------------------------
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Weather Prediction", layout="centered")
st.title("🌦 Weather Summary Prediction")

# ---------------------------
# User Inputs
# ---------------------------
temp = st.number_input("Temperature (C)")
humidity = st.slider("Humidity", 0.0, 1.0)
wind_speed = st.number_input("Wind Speed (km/h)")
wind_bearing = st.number_input("Wind Bearing (degrees)")
visibility = st.number_input("Visibility (km)")
pressure = st.number_input("Pressure (millibars)")
# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):

    data = np.array([[ 
        temp, 
        humidity, 
        wind_speed, 
        wind_bearing, 
        visibility, 
        pressure
    ]])

    
    pred = model.predict(data)

    result = le.inverse_transform(pred)[0]

    st.success(f"Predicted Weather: {result}")

