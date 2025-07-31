import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Streamlit Page Configuration
st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# Upload File
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        sheet_names = pd.ExcelFile(uploaded_file).sheet_names
        selected_sheet = st.selectbox("📑 Select a Sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=1)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Clean up unnamed columns safely
    df = df[[col for col in df.columns if not str(col).startswith("Unnamed")]]

    # Check if necessary columns exist
    if 'NAME' in df.columns and 'Final Status' in df.columns:
        df_filtered = df.copy()

        # Remove rows with invalid status like 207
        df_filtered = df_filtered[df_filtered['Final Status'].astype(str) != '207']

        st.subheader("📄 Raw Data")
        st.dataframe(df_filtered[['NAME', 'Final Status']].head())

        # Final Status Distribution
        st.subheader("📊 Exploratory Data Analysis")
        st.markdown("#### Final Status Distribution")
        status_counts = df_filtered['Final Status'].value_counts()
        fig1, ax1 = plt.subplots()
        status_counts.plot(kind='bar', color=sns.color_palette("Blues", len(status_counts)), ax=ax1)
        ax1.set_xlabel("Final Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Summary Stats
        st.markdown("#### Summary Statistics")
        st.write(df_filtered.describe(include='all'))

        # Histograms: Numeric Columns
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            st.markdown(f"#### Histogram: {col}")
            fig2, ax2 = plt.subplots()
            ax2.hist(df_filtered[col].dropna(), bins=15, edgecolor='black', color='skyblue')
            ax2.set_xlabel(col)
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

        # Encode Target Column
        le_target = LabelEncoder()
        df_filtered['Target_Encoded'] = le_target.fit_transform(df_filtered['Final Status'].astype(str))

        # Build feature set
        features = df_filtered.drop(columns=['NAME', 'EMAIL', 'Final Status'], errors='ignore')
        features = features.select_dtypes(include=[np.number]).dropna(axis=1, how='any')

        # Encode low-cardinality categorical features
        cat_cols = df_filtered.select_dtypes(include='object').columns
        for col in cat_cols:
            if col not in ['NAME', 'EMAIL', 'Final Status'] and df_filtered[col].nunique() < 15:
                try:
                    features[col] = LabelEncoder().fit_transform(df_filtered[col].astype(str))
                except Exception:
                    pass

        df_model = features.copy()
        df_model['Target'] = df_filtered['Target_Encoded']
        df_model = df_model.dropna()

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_model.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        # Split and Model
        st.subheader("⚙ Logistic Regression Model")
        X = df_model.drop(columns='Target')
        y = df_model['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GridSearchCV(LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }, cv=3)

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

        st.markdown("### 📋 Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=le_target.classes_))

        # Predict on full dataset
        full_pred = model.predict(X)
        df_results = pd.DataFrame({
            'Student Name': df_filtered['NAME'].values,
            'Actual Final Status': df_filtered['Final Status'].values,
            'Predicted Final Status': le_target.inverse_transform(full_pred)
        })

        st.subheader("🧾 All Student Predictions")
        st.dataframe(df_results)

        st.download_button("📥 Download Predictions", df_results.to_csv(index=False), file_name="student_predictions.csv")

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
else:
    st.info("ℹ Please upload your dataset to begin.")
