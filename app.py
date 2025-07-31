import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try to load XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

st.set_page_config(page_title="🎓 Student Status Predictor", layout="wide")
st.title("🎓 Predict Final Student Status (Capstone App)")

uploaded_file = st.file_uploader("📂 Upload the Excel file", type=["xlsx"])
if uploaded_file:
    try:
        # Load Excel sheets directly
        all_enrolled = pd.read_excel(uploaded_file, sheet_name='All Enrolled')
        all_enrolled2 = pd.read_excel(uploaded_file, sheet_name='All Enrolled (2)')

        # Clean column names
        all_enrolled.columns = all_enrolled.columns.str.strip().str.lower()
        all_enrolled2.columns = all_enrolled2.columns.str.strip().str.lower()

        # --- EDA Section ---
        st.header("📊 Exploratory Data Analysis (All Enrolled Sheet)")
        eda_df = all_enrolled.copy()

        eda_df = eda_df.loc[:, eda_df.notna().mean() > 0.5]
        eda_df = eda_df.dropna()

        if 'gender' in eda_df.columns:
            st.subheader("🎯 Gender Distribution")
            st.bar_chart(eda_df['gender'].value_counts())

        if 'university' in eda_df.columns:
            st.subheader("🏫 University Distribution")
            st.bar_chart(eda_df['university'].value_counts())

        if 'major' in eda_df.columns:
            st.subheader("🧠 Major Distribution")
            st.bar_chart(eda_df['major'].value_counts())

        if 'age' in eda_df.columns:
            st.subheader("📈 Age Distribution")
            plt.figure()
            sns.histplot(eda_df['age'], bins=15, kde=True)
            st.pyplot(plt.gcf())

        st.subheader("🔥 Correlation Heatmap")
        numeric_eda = eda_df.select_dtypes(include=['int64', 'float64'])
        if not numeric_eda.empty:
            plt.figure(figsize=(6, 4))
            sns.heatmap(numeric_eda.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt.gcf())
        else:
            st.info("No numeric fields available.")

        # --- Modeling Section ---
        st.header("🤖 Predict Student Final Status (All Enrolled 2 Sheet)")
        df = all_enrolled2.copy()
        df = df[['name', 'email', 'gender', 'age', 'university', 'major', 'employment', 'final status']]
        df = df.dropna()

        X = df.drop(columns=['name', 'email', 'final status'], errors='ignore')
        y = df['final status']
        X_encoded = pd.get_dummies(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"] + (["XGBoost"] if xgb_available else []))

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
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

        # Prediction for each student
        st.header("📋 Final Predictions for Each Student")
        all_preds = model.predict(X_encoded)
        predicted_labels = le.inverse_transform(all_preds)
        df["Predicted Status"] = predicted_labels
        st.dataframe(df[["name", "final status", "Predicted Status"]])

    except Exception as e:
        st.error(f"Something went wrong: {e}")

