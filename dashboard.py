import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# -------------------------------------------------------
# DATASET OVERVIEW
# -------------------------------------------------------
if menu == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

# -------------------------------------------------------
# CLASS DISTRIBUTION
# -------------------------------------------------------
elif menu == "Class Distribution":
    st.subheader("Weather Summary Distribution")

    class_counts = df["Summary"].value_counts()

    fig, ax = plt.subplots(figsize=(12,6))
    class_counts.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -------------------------------------------------------
# CORRELATION HEATMAP
# -------------------------------------------------------
elif menu == "Correlation Heatmap":
    st.subheader("Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# MODEL PERFORMANCE (CORRECT VERSION)
# -------------------------------------------------------
elif menu == "Model Performance":
    st.subheader("Model Evaluation (Test Set Only)")

    try:
        model = joblib.load("best_model.pkl")

        features = [
            "Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Pressure (millibars)"
        ]

        df_model = df[features + ["Summary"]].dropna()

        # 🔥 Apply same filtering used in notebook (Top 10 classes)
        top_classes = df_model["Summary"].value_counts().nlargest(10).index
        df_model = df_model[df_model["Summary"].isin(top_classes)]

        # Encode target fresh (same logic as notebook)
        le = LabelEncoder()
        df_model["Summary"] = le.fit_transform(df_model["Summary"])

        X = df_model[features]
        y = df_model["Summary"]

        # Same split as notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # Predict ONLY on test set
        y_pred = model.predict(X_test)

        # Calculate metrics EXACTLY like notebook
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision (Weighted)", f"{prec:.4f}")
        col3.metric("Recall (Weighted)", f"{rec:.4f}")
        col4.metric("F1-Score (Weighted)", f"{f1:.4f}")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Model loading error: {e}")

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------
elif menu == "Feature Importance":
    st.subheader("Feature Importance")

    try:
        model = joblib.load("best_model.pkl")

        # If model is pipeline, extract final estimator
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
        else:
            estimator = model

        features = [
            "Temperature (C)",
            "Humidity",
            "Wind Speed (km/h)",
            "Wind Bearing (degrees)",
            "Visibility (km)",
            "Pressure (millibars)"
        ]

        # Case 1: Tree-based models
        if hasattr(estimator, "feature_importances_"):

            importance = estimator.feature_importances_

        # Case 2: Linear models
        elif hasattr(estimator, "coef_"):

            importance = np.abs(estimator.coef_).mean(axis=0)

        else:
            st.info("This model does not support feature importance.")
            st.stop()

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Model file error: {e}")