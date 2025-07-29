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
        df = df[df["Final Status"] != "207"]  # Remove invalid entry if exists

        st.subheader("📄 Raw Data")
        st.dataframe(df[["NAME", "Final Status"]].head())

        # EDA
        st.markdown("### 📊 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Final Status Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x="Final Status", palette="Blues", ax=ax1)
            ax1.set_title("Final Status")
            st.pyplot(fig1)

        with col2:
            if "Age" in df.columns:
                st.markdown("#### Age Distribution")
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                sns.histplot(df["Age"].dropna(), bins=10, kde=False, color="skyblue", ax=ax2)
                ax2.set_title("Age Distribution")
                ax2.set_xlabel("Age")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)

        # Summary statistics
        st.markdown("#### 📌 Summary Statistics")
        st.dataframe(df.select_dtypes(include=np.number).describe().transpose())

        # Heatmap
        if df.select_dtypes(include=np.number).shape[1] > 1:
            st.markdown("#### 🔥 Correlation Heatmap")
            fig_corr, ax_corr = plt.subplots(figsize=(5, 3))
            sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="Blues", ax=ax_corr)
            st.pyplot(fig_corr)

        # Prepare for modeling
        df_model = df.copy()
        drop_cols = ["NAME", "EMAIL", "Final Status"] if "EMAIL" in df_model.columns else ["NAME", "Final Status"]
        X = df_model.drop(columns=drop_cols, errors="ignore")

        # Label encode object columns with low cardinality
        for col in X.select_dtypes(include="object").columns:
            if X[col].nunique() < 20:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X = X.select_dtypes(include=[np.number])
        y = df_model["Final Status"].astype(str)
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        # Final safety clean before model
        model_data = pd.concat([X, pd.Series(y_encoded, name="Target")], axis=1)
        model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
        X_clean = model_data.drop("Target", axis=1)
        y_clean = model_data["Target"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)

        # Train baseline logistic model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        st.markdown("### 🤖 Logistic Regression Model")
        st.markdown(f"*Accuracy:* {accuracy_score(y_test, y_pred):.2f}")
        st.markdown(f"*Precision:* {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.markdown(f"*Recall:* {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1_score(y_test, y_pred, average='weighted'):.2f}")

        # Report
        st.markdown("#### 📄 Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)).transpose().round(2))

        # Display predictions
        st.markdown("#### 🎯 Predictions")
        predictions_df = df.loc[X_clean.index, ["NAME"]].copy()
        predictions_df["Actual"] = le_target.inverse_transform(y_clean)
        predictions_df["Predicted"] = le_target.inverse_transform(y_pred)
        st.dataframe(predictions_df.reset_index(drop=True))

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
