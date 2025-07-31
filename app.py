import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")

st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# Upload Section
uploaded_file = st.file_uploader("📁 Upload Excel or CSV", type=["xlsx", "csv"])
if uploaded_file:
    try:
        # Detect file type
        if uploaded_file.name.endswith('.xlsx'):
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            sheet = st.selectbox("📑 Select Sheet", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
        else:
            df = pd.read_csv(uploaded_file)
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        columns = list(df.columns)

        if 'name' in columns and 'final status' in columns:
            st.success("✅ File loaded successfully!")

            # Rename for uniformity
            df.rename(columns={'name': 'NAME', 'final status': 'Final Status'}, inplace=True)

            # Remove Unnamed
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            # Drop rows with missing Final Status
            df = df.dropna(subset=['Final Status'])

            # ================== EDA ==================
            st.header("📊 Exploratory Data Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Final Status Distribution")
                final_counts = df['Final Status'].value_counts()
                if 207 in final_counts.index or '207' in final_counts.index:
                    final_counts = final_counts.drop(207, errors='ignore')
                    final_counts = final_counts.drop('207', errors='ignore')
                fig1, ax1 = plt.subplots()
                sns.barplot(x=final_counts.index, y=final_counts.values, palette='Blues_d', ax=ax1)
                ax1.set_title("Final Status Count")
                st.pyplot(fig1)

            with col2:
                if 'age' in df.columns:
                    st.markdown("#### Age Distribution")
                    fig2, ax2 = plt.subplots()
                    sns.histplot(df['age'].dropna(), kde=False, bins=15, color="steelblue", edgecolor="black", ax=ax2)
                    ax2.set_title("Age Histogram")
                    st.pyplot(fig2)

            # Heatmap
            st.markdown("#### Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=np.number).columns
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax3)
            st.pyplot(fig3)

            # ================== MODEL ==================
            st.header("⚙ Logistic Regression Model")

            df_model = df.copy()
            le_target = LabelEncoder()
            df_model['Target'] = le_target.fit_transform(df_model['Final Status'])

            # Drop unnecessary columns
            features = df_model.drop(columns=['Final Status', 'Target'], errors='ignore')
            features = features.select_dtypes(include=[np.number]).dropna()

            # Match y to X after dropna
            df_model = df_model.loc[features.index]
            X = features
            y = df_model['Target']

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            st.markdown(f"*Accuracy:* {acc:.2f}")
            st.markdown(f"*Precision:* {prec:.2f}")
            st.markdown(f"*Recall:* {rec:.2f}")
            st.markdown(f"*F1 Score (Recommended):* {f1:.2f}")

            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)).T
            st.dataframe(report_df)

            # ================== PREDICTION ==================
            st.markdown("#### 📋 Predictions")
            pred_all = model.predict(X)
            predictions_df = pd.DataFrame({
                "NAME": df_model["NAME"].values,
                "Actual Final Status": df_model["Final Status"].values,
                "Predicted Final Status": le_target.inverse_transform(pred_all)
            })
            st.dataframe(predictions_df)

            st.download_button("📥 Download Predictions CSV", predictions_df.to_csv(index=False), file_name="student_predictions.csv")

        else:
            st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("ℹ Please upload your dataset to begin.")
