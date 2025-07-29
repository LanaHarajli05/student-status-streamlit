import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Streamlit Page Setup
st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# Upload File
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("📑 Select a Sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if 'NAME' in df.columns and 'Final Status' in df.columns:
        df_filtered = df.copy()

        # Drop rows where Final Status is null
        df_filtered = df_filtered[df_filtered['Final Status'].notna()]

        st.subheader("📄 Raw Data")
        st.dataframe(df_filtered[['NAME', 'Final Status']].head())

        # Clean Target Column
        le_target = LabelEncoder()
        try:
            df_filtered['Target'] = le_target.fit_transform(df_filtered['Final Status'].astype(str))
        except:
            st.error("❌ Error encoding 'Final Status'. Please check for inconsistent values.")
            st.stop()

        # EDA Layout
        st.subheader("📊 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Final Status Distribution")
            st.bar_chart(df_filtered['Final Status'].value_counts())

        with col2:
            st.markdown("#### 📋 Summary Statistics (numeric only)")
            numeric_stats = df_filtered.select_dtypes(include='number').describe().T
            st.dataframe(numeric_stats)

        # Histograms (only for numeric columns)
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df_filtered[col].dropna(), bins=15, edgecolor='black')
            ax.set_title(f"Histogram of {col}")
            ax.grid(False)
            st.pyplot(fig)

        # Encode Categorical Features
        features = df_filtered.drop(columns=['NAME', 'EMAIL', 'Final Status', 'Target'], errors='ignore')
        encoded_df = pd.DataFrame()
        for col in features.columns:
            if features[col].dtype == 'object':
                if df_filtered[col].nunique() < 15:
                    encoded_df[col] = LabelEncoder().fit_transform(df_filtered[col].astype(str))
            else:
                encoded_df[col] = features[col]

        model_df = encoded_df.copy()
        model_df['Target'] = df_filtered['Target']
        model_df = model_df.dropna()

        # Correlation Heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(model_df.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

        # Modeling
        st.subheader("⚙ Logistic Regression Model")
        X = model_df.drop(columns='Target')
        y = model_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid={
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }, cv=3)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.markdown(f"*Accuracy:* {acc:.2f}")
        st.markdown(f"*Precision:* {prec:.2f}")
        st.markdown(f"*Recall:* {rec:.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1:.2f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))

        # Full prediction
        try:
            full_pred = model.predict(X)
            df_results = pd.DataFrame({
                'Student Name': df_filtered.loc[X.index, 'NAME'],
                'Actual Final Status': df_filtered.loc[X.index, 'Final Status'],
                'Predicted Final Status': le_target.inverse_transform(full_pred)
            })

            st.subheader("🧾 All Student Predictions")
            st.dataframe(df_results)
            st.download_button("📥 Download Predictions", df_results.to_csv(index=False), file_name="student_predictions.csv")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
else:
    st.info("ℹ Please upload your dataset to get started.")
