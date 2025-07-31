import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Predict Final Student Status")

st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📂 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        sheet_name = st.selectbox("📄 Select a Sheet", pd.ExcelFile(uploaded_file).sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

    st.success("✅ File uploaded successfully!")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Check if required columns exist
    if "name" not in df.columns or "final status" not in df.columns:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
    else:
        # Rename back to standard names
        df.rename(columns={"name": "NAME", "final status": "Final Status"}, inplace=True)

        # Show EDA
        st.header("📊 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Final Status Distribution")
            st.bar_chart(df["Final Status"].value_counts())

        if "Age" in df.columns:
            with col2:
                st.subheader("Age Distribution")
                st.bar_chart(df["Age"].dropna().astype(int).value_counts().sort_index())
        else:
            st.warning("⚠ 'Age' column not found for age distribution.")

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠ No numeric columns found for correlation heatmap.")

        # Preprocessing for model
        df_model = df.copy()
        df_model.drop(columns=["NAME"], inplace=True)

        for col in df_model.select_dtypes(include=['object']).columns:
            df_model[col] = df_model[col].astype(str)
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])

        # Split features and target
        X = df_model.drop("Final Status", axis=1)
        y = df_model["Final Status"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.header("⚙ Logistic Regression Model")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.markdown(f"- *Accuracy:* {acc:.2f}")
        st.markdown(f"- *Precision:* {prec:.2f}")
        st.markdown(f"- *Recall:* {rec:.2f}")
        st.markdown(f"- *F1 Score (Recommended):* {f1:.2f}")

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df)

        # Predictions for each student
        st.subheader("🔍 Student Prediction Probabilities")

        proba = model.predict_proba(X)
        labels = model.classes_

        results_df = df[["NAME"]].copy()
        for idx, label in enumerate(labels):
            results_df[f"Prob_{label}"] = proba[:, idx]

        st.dataframe(results_df.head(30))  # show first 30 for clarity
