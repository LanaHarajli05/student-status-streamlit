import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Streamlit Page Setup
st.set_page_config(page_title="Predict Final Student Status", layout="centered")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

# File Upload
uploaded_file = st.file_uploader("📤 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    try:
        # Read Excel or CSV
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, sheet_name=None)
            sheet_names = list(df.keys())
            sheet = st.selectbox("📄 Select a Sheet", sheet_names)
            df = df[sheet]
        else:
            df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        if "NAME" not in df.columns or "Final Status" not in df.columns:
            st.error("❌ Dataset must include both 'NAME' and 'Final Status' columns.")
        else:
            df = df.dropna(subset=["NAME", "Final Status"])

            st.success("✅ File uploaded successfully!")
            st.subheader("🗃 Raw Data")
            st.dataframe(df[["NAME", "Final Status"]].head())

            # -------------------------------
            # 📊 Exploratory Data Analysis
            # -------------------------------
            st.subheader("📊 Exploratory Data Analysis")

            st.markdown("**Distribution of Final Status**")
            st.bar_chart(df["Final Status"].value_counts())

            st.markdown("**Correlation Heatmap**")
            df["Final Status Encoded"] = df["Final Status"].astype("category").cat.codes
            df["Student_ID"] = np.arange(len(df))

            corr = df[["Student_ID", "Final Status Encoded"]].corr()

            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            # -------------------------------
            # 🤖 Model: Logistic Regression
            # -------------------------------
            st.subheader("⚙️ Model Configuration")

            st.info("Predicting 'Final Status' using generated Student_ID only (since NAME is not a feature).")

            X = df[["Student_ID"]]
            y = df["Final Status Encoded"]
            name_map = dict(zip(df["Student_ID"], df["NAME"]))
            label_map = dict(enumerate(df["Final Status"].astype("category").cat.categories))

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # -------------------------------
            # ✅ Model Evaluation
            # -------------------------------
            st.subheader("✅ Model Evaluation")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
            st.write(f"**F1-score:** {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, zero_division=0))

            # -------------------------------
            # 📋 Prediction Results
            # -------------------------------
            st.subheader("📋 Student Status Predictions")
            pred_df = X_test.copy()
            pred_df["Actual"] = y_test.map(label_map)
            pred_df["Predicted"] = pd.Series(y_pred, index=pred_df.index).map(label_map)
            pred_df["Student Name"] = pred_df["Student_ID"].map(name_map)

            st.dataframe(pred_df[["Student Name", "Actual", "Predicted"]].reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

else:
    st.info("Upload your dataset to begin.")


