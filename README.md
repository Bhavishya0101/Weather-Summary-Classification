# 🌦 Weather Summary Classification

A Machine Learning based web application that predicts weather conditions such as **Overcast, Partly Cloudy, Mostly Cloudy, Foggy, Clear, Drizzle**, etc., using meteorological features.

This project includes:

- 📊 Interactive Dashboard (Data Analysis + Model Evaluation)
- 🔮 Real-Time Weather Prediction App
- 🤖 Machine Learning Model (Decision Tree Classifier)
- 🚀 Streamlit Deployment

---

## 📌 Project Overview

The goal of this project is to classify weather summaries using features such as:

- Temperature (C)
- Humidity
- Wind Speed (km/h)
- Wind Bearing (degrees)
- Visibility (km)
- Pressure (millibars)

The model predicts the **exact weather summary category** present in the dataset.

---

## 🧠 Machine Learning Model

- Algorithm: Decision Tree Classifier
- Target Variable: `Summary`
- Preprocessing:
  - Label Encoding for Summary
  - Proper feature ordering
- Model Saved Using: `joblib`

---

## 📊 Live Applications

### 🔹 Weather Summary Dashboard

Includes:
- Dataset Overview
- Class Distribution
- Correlation Heatmap
- Model Performance
- Feature Importance

👉 **Live Dashboard:**  
https://weather-summary-classification-8tbdbbr9effm3yytwrbu7k.streamlit.app/

---

### 🔹 Weather Prediction App

Real-time prediction using user input values.

👉 **Live Prediction App:**  
https://weather-summary-classification-5cpmappppbo9fr9fmeur4jn3.streamlit.app/

---
