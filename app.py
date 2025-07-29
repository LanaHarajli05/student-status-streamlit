import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# App Config
st.set_page_config(page_title="Student Final Status Prediction", layout="centered")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# Upload file
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Clean unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Clean false missing (e.g., spaces)
        df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        df.dropna(inplace=True)

        # Preview
        st.success("✅ File uploaded successfully!")
        st.subheader("📋 Raw Data")
        st.dataframe(df.head())

        # --- EDA ---
        st.subheader("📊 Exploratory Data Analysis")

        # Correlation heatmap (numeric only)
        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            st.markdown("**Correlation Heatmap**")
            corr = numeric_df.corr().round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu")
            st.plotly_chart(fig)

        # Class distribution
        if "Final Status" in df.columns:
            st.markdown("**Final Status Distribution**")
            fig2 = px.histogram(df, x="Final Status", title="Distribution of Final Status")
            st.plotly_chart(fig2)

        # --- Model Setup ---
        st.subheader("⚙️ Model Configuration")
        if "Name" in df.columns and "Final Status" in df.columns:
            feature_cols = [col for col in df.columns if col not in ["Name", "Final Status"]]
            X = df[feature_cols]
            y = df["Final Status"]
            names = df["Name"]

            # Encode categorical features
            X = pd.get_dummies(X)

            # Encode target
            y_encoded = y.astype("category").cat.codes
            label_map = dict(enumerate(y.astype("category").cat.categories))

            # Split
            X_train, X_test, y_train, y_test, name_test = train_test_split(X, y_encoded, names, test_size=0.2, random_state=42)

            # Train model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- Evaluation ---
            st.subheader("📈 Model Evaluation")
            f1 = f1_score(y_test, y_pred, average='macro')
            st.write(f"**F1 Score (macro):** {f1:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred, target_names=label_map.values()))

            # --- Predictions by student ---
            st.subheader("📌 Predicted Status for Test Students")
            predictions_df = pd.DataFrame({
                "Name": name_test.values,
                "Actual Status": y_test.map(label_map),
                "Predicted Status": pd.Series(y_pred).map(label_map)
            })
            st.dataframe(predictions_df)

        else:
            st.error("❌ Dataset must include both 'Name' and 'Final Status' columns.")

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("Upload a dataset to get started.")




