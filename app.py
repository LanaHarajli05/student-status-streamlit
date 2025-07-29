import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Streamlit config
st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# Upload data
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            selected_sheet = st.selectbox("📑 Select a Sheet", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # Clean dataframe
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        if 'NAME' not in df.columns or 'Final Status' not in df.columns:
            st.error("❌ 'NAME' and 'Final Status' columns must be present.")
            st.stop()

        df = df.dropna(subset=['Final Status'])
        df = df[df['Final Status'].apply(lambda x: isinstance(x, str))]

        st.subheader("📄 Raw Data")
        st.dataframe(df[['NAME', 'Final Status']].head())

        # Encode Target
        le_target = LabelEncoder()
        df['Target'] = le_target.fit_transform(df['Final Status'])

        # EDA
        st.subheader("📊 Exploratory Data Analysis")
        st.markdown("#### Final Status Distribution")
        st.bar_chart(df['Final Status'].value_counts())

        # Summary Stats (hide meaningless ones)
        st.markdown("#### Summary Statistics (numeric only)")
        st.write(df.select_dtypes(include=[np.number]).describe())

        # Histograms - Clean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            ax.hist(df[col].dropna(), bins=20, edgecolor='black')
            ax.set_title(f"{col}")
            ax.grid(False)
            st.pyplot(fig)

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

        # Extra EDA
        if 'Gender' in df.columns:
            st.markdown("#### Final Status by Gender")
            gender_plot = pd.crosstab(df['Gender'], df['Final Status'])
            st.bar_chart(gender_plot)

        if 'Age' in df.columns:
            st.markdown("#### Age Distribution by Final Status")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Final Status', y='Age', ax=ax)
            st.pyplot(fig)

        # MODELING
        st.subheader("⚙ Logistic Regression Model")

        # Features
        model_df = df.drop(columns=['NAME', 'EMAIL', 'Final Status'], errors='ignore')
        for col in model_df.select_dtypes(include='object').columns:
            if model_df[col].nunique() <= 10:
                model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))
            else:
                model_df.drop(columns=[col], inplace=True)

        model_df = model_df.dropna()
        if 'Target' not in model_df.columns:
            model_df['Target'] = df.loc[model_df.index, 'Target']

        X = model_df.drop(columns=['Target'])
        y = model_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        model = GridSearchCV(LogisticRegression(max_iter=1000), grid, cv=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.markdown(f"*Accuracy:* {acc:.2f}")
        st.markdown(f"*Precision:* {prec:.2f}")
        st.markdown(f"*Recall:* {rec:.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1:.2f}")

        st.text("📋 Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))

        # Prediction Output
        full_pred = model.predict(X)
        df_results = pd.DataFrame({
            'Student Name': df['NAME'],
            'Actual Final Status': df['Final Status'],
            'Predicted Final Status': le_target.inverse_transform(full_pred)
        })
        st.subheader("📄 All Student Predictions")
        st.dataframe(df_results)

        st.download_button("📥 Download Predictions", df_results.to_csv(index=False), file_name="student_predictions.csv")

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("ℹ Please upload a file to begin.")



