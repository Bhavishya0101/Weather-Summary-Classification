import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🌦 Weather Summary Prediction")

temp = st.number_input("Temperature (C)")
humidity = st.slider("Humidity", 0.0, 1.0)
wind_speed = st.number_input("Wind Speed (km/h)")
wind_bearing = st.number_input("Wind Bearing (degrees)")
visibility = st.number_input("Visibility (km)")
pressure = st.number_input("Pressure (millibars)")

if st.button("Predict"):
    data = np.array([[temp, humidity, wind_speed, wind_bearing, visibility, pressure]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    st.success(le.inverse_transform(pred)[0])