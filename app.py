import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Predict Final Student Status", layout="wide")

st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    try:
        if uploaded_file.name.endswith('.xlsx'):
            sheet = st.selectbox("Select a Sheet", pd.ExcelFile(uploaded_file).sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Clean columns
        df.columns = df.columns.str.strip().str.lower()

        if not all(col in df.columns for col in ['name', 'final status']):
            st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
        else:
            df = df.dropna()

            # Drop invalid labels if any
            df = df[df['final status'].isin(['Active', 'Graduated', 'Dropped', 'In-Active'])]

            st.success("✅ Data loaded successfully!")

            # ---------- EDA ----------
            st.header("📊 Exploratory Data Analysis")

            st.subheader("🎯 Target Distribution")
            st.bar_chart(df['final status'].value_counts())

            st.subheader("📌 Summary Statistics")
            st.dataframe(df.describe(include='all').transpose())

            st.subheader("🧭 Correlation Heatmap (Numerical Features Only)")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if not numeric_df.empty:
                plt.figure(figsize=(8, 5))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt.gcf())
            else:
                st.info("No numerical features to show correlation.")

            # ---------- Preprocessing ----------
            st.header("⚙️ Train Model & Predict")

            X = df.drop(columns=['name', 'email', 'final status'], errors='ignore')
            y = df['final status']
            
            # One-hot encode categorical vars
            X_encoded = pd.get_dummies(X)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

            # Model selection
            model_name = st.selectbox("Select a Model", ["Logistic Regression", "Random Forest", "XGBoost"])

            if model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ---------- Evaluation ----------
            st.subheader("📈 Evaluation Metrics")
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

            st.subheader("🌀 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt.gcf())

            # ---------- Predict All ----------
            st.header("📋 Predictions for Each Student")

            full_preds = model.predict(X_encoded)
            predicted_labels = le.inverse_transform(full_preds)
            df["predicted status"] = predicted_labels

            st.dataframe(df[["name", "final status", "predicted status"]])

            st.download_button("📥 Download Predictions", data=df.to_csv(index=False), file_name="student_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
