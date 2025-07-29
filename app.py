import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📁 Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("📑 Select Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet)

    df.columns = df.columns.str.strip()

    if "NAME" in df.columns and "Final Status" in df.columns:
        st.success("✅ File uploaded successfully!")

        # Clean Final Status
        df = df[df["Final Status"].notna()]
        df["Final Status"] = df["Final Status"].astype(str).str.strip()

        # Remove invalid values like '207' if exists
        df = df[df["Final Status"] != "207"]

        st.subheader("📄 Raw Data")
        st.dataframe(df[["NAME", "Final Status"]].head())

        # Exploratory Data Analysis
        st.markdown("### 📊 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Final Status Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x="Final Status", palette="Blues", ax=ax1)
            ax1.set_title("Final Status")
            ax1.grid(False)
            st.pyplot(fig1)

        with col2:
            if "Age" in df.columns:
                st.markdown("#### Age Distribution")
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                sns.histplot(df["Age"].dropna(), bins=10, kde=False, color="skyblue", ax=ax2)
                ax2.set_title("Age Distribution")
                ax2.set_xlabel("Age")
                ax2.set_ylabel("Count")
                ax2.grid(False)
                st.pyplot(fig2)

        # Summary statistics
        st.markdown("#### 📌 Summary Statistics")
        st.dataframe(df.select_dtypes(include=np.number).describe().transpose())

        # Correlation Heatmap
        if df.select_dtypes(include=np.number).shape[1] > 1:
            st.markdown("#### 🔥 Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
            sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="Blues", ax=ax_corr)
            st.pyplot(fig_corr)

        # Prepare data for model
        st.markdown("### 🧠 Logistic Regression Model")
        df_model = df.copy()

        drop_cols = ["NAME", "EMAIL", "Final Status"] if "EMAIL" in df_model.columns else ["NAME", "Final Status"]
        X = df_model.drop(columns=drop_cols)
        X = X.select_dtypes(include=["int64", "float64", "object"])

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        y = df_model["Final Status"].astype(str)
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        # Split and fit model
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.markdown(f"*Accuracy:* {accuracy_score(y_test, y_pred):.2f}")
        st.markdown(f"*Precision:* {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.markdown(f"*Recall:* {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1_score(y_test, y_pred, average='weighted'):.2f}")

        # Classification report
        st.markdown("#### 📄 Classification Report")
        report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

        # Display predictions with names
        st.markdown("#### 🎯 Sample Predictions")
        sample_results = df.loc[X_test.index, ["NAME"]].copy()
        sample_results["Actual"] = le_target.inverse_transform(y_test)
        sample_results["Predicted"] = le_target.inverse_transform(y_pred)
        st.dataframe(sample_results.reset_index(drop=True))

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
