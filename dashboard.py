import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(page_title="Weather Dashboard", layout="wide")
st.title("🌦 Weather Summary Classification Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("weatherHistory.csv")

df = load_data()

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "Class Distribution",
        "Correlation Heatmap",
        "Model Performance",
        "Feature Importance"
    ]
)

if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

elif menu == "Class Distribution":
    st.subheader("Weather Summary Distribution")

    target_col = "Summary" if "Summary" in df.columns else "Weather_Summary"
    class_counts = df[target_col].value_counts()

    fig, ax = plt.subplots(figsize=(12,6))
    class_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif menu == "Correlation Heatmap":
    st.subheader("Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "Model Performance":
    st.subheader("Model Evaluation")

    try:
        model = joblib.load("best_model.pkl")
        le = joblib.load("label_encoder.pkl")

        features = [
            "Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Pressure (millibars)"
        ]

        df_model = df[features + ["Summary"]].dropna()

        X = df_model[features]
        y = le.transform(df_model["Summary"])

        # 🔥 Works for both Pipeline and normal model
        y_pred = model.predict(X)

        st.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
        st.metric("Precision (Weighted)", f"{precision_score(y, y_pred, average='weighted'):.4f}")
        st.metric("Recall (Weighted)", f"{recall_score(y, y_pred, average='weighted'):.4f}")
        st.metric("F1-Score (Weighted)", f"{f1_score(y, y_pred, average='weighted'):.4f}")

        st.text("Classification Report:")
        st.text(classification_report(y, y_pred))

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Model loading error: {e}")

elif menu == "Feature Importance":
    st.subheader("Feature Importance")

    try:
        model = joblib.load("best_model.pkl")

        # If pipeline, extract estimator
        if hasattr(model, "named_steps"):
            estimator = model.named_steps["model"]
        else:
            estimator = model

        if hasattr(estimator, "feature_importances_"):

            features = [
                "Temperature (C)",
                "Humidity",
                "Wind Speed (km/h)",
                "Wind Bearing (degrees)",
                "Visibility (km)",
                "Pressure (millibars)"
            ]

            importance = estimator.feature_importances_

            feat_df = pd.DataFrame({
                "Feature": features,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)

        else:
            st.info("This model does not support feature importance.")

    except Exception as e:
        st.warning(f"Model file error: {e}")
