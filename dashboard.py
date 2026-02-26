import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Weather Dashboard", layout="wide")

st.title("🌦 Weather Summary Classification Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("weatherHistory.csv")

df = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Dataset Overview",
     "Class Distribution",
     "Correlation Heatmap",
     "Model Performance",
     "Feature Importance"]
)

# -------------------------------
# 1️⃣ Dataset Overview
# -------------------------------
if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

# -------------------------------
# 2️⃣ Class Distribution
# -------------------------------
elif menu == "Class Distribution":
    st.subheader("Weather Summary Distribution")

    if "Summary" in df.columns:
        class_counts = df["Summary"].value_counts()
    else:
        class_counts = df["Weather_Summary"].value_counts()

    fig, ax = plt.subplots(figsize=(12,6))
    class_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------
# 3️⃣ Correlation Heatmap
# -------------------------------
elif menu == "Correlation Heatmap":
    st.subheader("Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------
# 4️⃣ Model Performance
# -------------------------------
elif menu == "Model Performance":
    st.subheader("Model Evaluation")

    try:
        # Correct model path
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        le = joblib.load("label_encoder.pkl")

        # Prepare dataset for evaluation
        df_model = df[[
            "Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Pressure (millibars)",
            "Summary"
        ]].dropna()

        # Keep top 10 classes
        top_classes = df_model["Summary"].value_counts().nlargest(10).index
        df_model = df_model[df_model["Summary"].isin(top_classes)]

        X = df_model.drop("Summary", axis=1)
        y = le.transform(df_model["Summary"])

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')

        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision (Weighted)", f"{precision:.4f}")
        st.metric("Recall (Weighted)", f"{recall:.4f}")
        st.metric("F1-Score (Weighted)", f"{f1:.4f}")

        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(y, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading model files: {e}")

# -------------------------------
# 5️⃣ Feature Importance
# -------------------------------
elif menu == "Feature Importance":
    st.subheader("Feature Importance")

    try:
        model = joblib.load("best_model.pkl")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

            features = [
                "Temperature (C)",
                "Humidity",
                "Wind Speed (km/h)",
                "Wind Bearing (degrees)",
                "Visibility (km)",
                "Pressure (millibars)"
            ]

            feat_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)
        else:
            st.info("This model does not support feature importance.")

    except:
        st.warning("Model file not found.")