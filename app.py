import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

st.set_page_config(page_title="Predict Final Student Status", layout="wide")

# Title
st.markdown("## 🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

# Load and validate dataset
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("📄 Select a Sheet", excel_file.sheet_names)
        df = excel_file.parse(sheet_name=sheet)

    df.columns = df.columns.str.strip()  # remove any leading/trailing spaces in column names

    if 'NAME' in df.columns and 'Final Status' in df.columns:
        st.success("✅ File uploaded successfully!")
        
        # Show raw data
        st.subheader("🧾 Raw Data")
        st.dataframe(df[['NAME', 'Final Status']].head())

        # Filter dataset to drop missing Final Status
        df = df[df['Final Status'].notna()]

        # Exploratory Data Analysis
        st.markdown("### 📊 Exploratory Data Analysis")
        
        # Countplot of Final Status
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            sns.countplot(data=df, x='Final Status', palette="viridis", ax=ax1)
            ax1.set_title("Final Status Distribution")
            ax1.grid(False)
            st.pyplot(fig1)

        # Histogram of Age if available
        if 'Age' in df.columns:
            with col2:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                df['Age'].dropna().astype(int).hist(bins=10, ax=ax2)
                ax2.set_title("Age Distribution")
                ax2.set_xlabel("Age")
                ax2.set_ylabel("Frequency")
                ax2.grid(False)
                st.pyplot(fig2)

        # Summary statistics (only numeric)
        numeric_df = df.select_dtypes(include=[np.number])
        st.markdown("#### 📌 Summary Statistics (numeric only)")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().transpose())

        # Correlation Heatmap
        if numeric_df.shape[1] > 1:
            st.markdown("#### 🔥 Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
            st.pyplot(fig3)

        # Preprocessing for modeling
        st.markdown("### 🧠 Logistic Regression Model")

        df_model = df.copy()
        df_model = df_model.dropna(subset=['Final Status'])
        
        # Drop irrelevant columns
        drop_cols = ['NAME', 'EMAIL', 'Final Status'] if 'EMAIL' in df_model.columns else ['NAME', 'Final Status']
        features = df_model.drop(columns=drop_cols)
        features = features.select_dtypes(include=['object', 'category', 'int64', 'float64'])

        # Encode categorical columns
        for col in features.select_dtypes(include='object').columns:
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))

        # Align feature and target
        X = features
        y = df_model['Final Status']
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        # Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.markdown(f"*Accuracy:* {acc:.2f}")
        st.markdown(f"*Precision:* {prec:.2f}")
        st.markdown(f"*Recall:* {rec:.2f}")
        st.markdown(f"*F1 Score (Recommended):* {f1:.2f}")

        # Classification report
        st.markdown("#### 📄 Classification Report")
        report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)

        # Match prediction with actual NAME
        st.markdown("#### 🎯 Sample Predictions")

        # Match to original names (only in test set)
        test_indices = X_test.index
        predictions_named = df.loc[test_indices, ['NAME']].copy()
        predictions_named['Actual Status'] = le_target.inverse_transform(y_test)
        predictions_named['Predicted Status'] = le_target.inverse_transform(y_pred)

        st.dataframe(predictions_named.reset_index(drop=True))

    else:
        st.error("❌ Dataset must contain both *'NAME'* and *'Final Status'* columns.")
