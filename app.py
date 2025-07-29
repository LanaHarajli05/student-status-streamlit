import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# App Title
st.set_page_config(page_title="Student Status Prediction", layout="centered")
st.title("🎓 Student Status Prediction App")
st.markdown("""
Upload your student dataset and select the features and target column to build a Logistic Regression model that predicts student status.
""")

# File Upload
uploaded_file = st.file_uploader("📁 Upload student data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df.head())

        # EDA
        st.subheader("📊 Exploratory Data Analysis")
        st.write(df.describe(include='all'))

        # Feature and target selection
        st.subheader("⚙️ Model Setup")
        target_col = st.selectbox("🎯 Select the target column (e.g. student status)", df.columns)
        feature_cols = st.multiselect("📌 Select feature columns", [col for col in df.columns if col != target_col])

        if target_col and feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            # Encode target if categorical
            if y.dtype == 'object':
                y = y.astype('category').cat.codes

            # Encode categorical features
            X = pd.get_dummies(X)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train logistic regression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Results
            st.subheader("✅ Model Performance")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error(f"🚫 Something went wrong: {e}")

else:
    st.info("⬆️ Please upload your dataset to begin.")


