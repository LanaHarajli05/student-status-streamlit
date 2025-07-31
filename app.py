import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("📑 Select a Sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=1)
    else:
        df = pd.read_csv(uploaded_file)

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if 'NAME' in df.columns and 'Final Status' in df.columns:
        df = df[df['Final Status'].notna()]
        df = df[df['Final Status'] != 207]  # Remove placeholder

        st.success("✅ Data Loaded Successfully")
        st.subheader("📄 Preview Data")
        st.dataframe(df[['NAME', 'Final Status']].head())

        st.subheader("📊 Final Status Distribution")
        final_status_counts = df['Final Status'].value_counts()
        fig, ax = plt.subplots()
        final_status_counts.plot(kind='bar', color=plt.cm.Blues(np.linspace(0.4, 0.9, len(final_status_counts))), ax=ax)
        ax.set_ylabel("Count")
        ax.set_xlabel("Final Status")
        ax.set_title("Distribution of Final Status")
        st.pyplot(fig)

        st.subheader("📊 Age Distribution")
        if 'Age' in df.columns:
            fig2, ax2 = plt.subplots()
            df['Age'].dropna().astype(int).value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax2)
            ax2.set_title("Age Distribution")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

        st.subheader("📋 Summary Statistics")
        st.write(df.describe(include='all'))

        # Encode target
        le_target = LabelEncoder()
        df['Target'] = le_target.fit_transform(df['Final Status'].astype(str))

        # Prepare features
        drop_cols = ['NAME', 'EMAIL', 'Final Status']
        features = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Clean categorical low-cardinality vars
        for col in features.select_dtypes(include='object').columns:
            if features[col].nunique() <= 15:
                features[col] = LabelEncoder().fit_transform(features[col].astype(str))

        # Clean numeric columns
        features = features.select_dtypes(include=[np.number]).dropna(axis=1)

        df_model = features.copy()
        df_model['Target'] = df['Target']
        df_model = df_model.dropna()

        # 🔥 Correlation Heatmap (Smaller)
        st.subheader("📌 Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
        sns.heatmap(df_model.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        # Logistic Regression
        st.subheader("⚙ Logistic Regression Model")
        X = df_model.drop(columns='Target')
        y = df_model['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.markdown(f"*Accuracy:* {acc:.2f}")
        st.markdown(f"*Precision:* {prec:.2f}")
        st.markdown(f"*Recall:* {rec:.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1:.2f}")

        st.text("📑 Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))

        # 🔍 Predictions
        y_full_pred = model.predict(X)
        df_results = pd.DataFrame({
            "Student Name": df['NAME'].values,
            "Actual Final Status": df['Final Status'].values,
            "Predicted Final Status": le_target.inverse_transform(y_full_pred)
        })

        st.subheader("📍 All Student Predictions")
        st.dataframe(df_results)

        st.download_button("📥 Download Predictions", df_results.to_csv(index=False), file_name="predictions.csv")
    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
else:
    st.info("ℹ Upload your dataset to get started.")
