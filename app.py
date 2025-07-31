import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Page config
st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            sheet_names = pd.ExcelFile(uploaded_file).sheet_names
            selected_sheet = st.selectbox("Select a Sheet", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=1)
        else:
            df = pd.read_csv(uploaded_file)

        df.dropna(how="all", axis=1, inplace=True)
        df.columns = [str(col).strip() for col in df.columns]

        if "Graduated" not in df.columns:
            st.error("Dataset must contain a 'Graduated' column (Final Status). Please check your file.")
        else:
            df.rename(columns={"Graduated": "Final Status"}, inplace=True)

            name_column = df.columns[1]  # usually second column like 'AI&DS-1'
            st.success("✅ File uploaded successfully!")

            # Show basic stats
            st.subheader("📊 Exploratory Data Analysis")

            st.markdown("#### Final Status Distribution")
            status_counts = df['Final Status'].value_counts()
            status_counts = status_counts[status_counts.index != '207']  # remove weird value if present
            st.bar_chart(status_counts)

            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'].dropna(), kde=False, bins=15, color='skyblue', edgecolor='black', ax=ax)
            st.pyplot(fig)

            st.markdown("#### Gender Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Gender', data=df, palette='Blues', ax=ax)
            st.pyplot(fig)

            st.markdown("#### Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax)
            st.pyplot(fig)

            # MODEL
            st.subheader("⚙ Logistic Regression Model")

            # Features and target
            model_df = df[[name_column, 'Gender', 'Age', 'Final Status']].dropna()
            X = model_df[['Gender', 'Age']]
            y = model_df['Final Status']

            # Encode categorical
            le_gender = LabelEncoder()
            X['Gender'] = le_gender.fit_transform(X['Gender'])

            le_status = LabelEncoder()
            y_encoded = le_status.fit_transform(y)

            # Train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            st.markdown(f"""
            - *Accuracy*: {acc:.2f}
            - *Precision*: {prec:.2f}
            - *Recall*: {rec:.2f}
            - *F1 Score*: {f1:.2f}
            """)

            # Classification report
            st.markdown("#### 📋 Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_status.classes_, output_dict=True)).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))

            # Predictions
            st.markdown("#### 🎯 Predictions for All Students")
            df_pred = X.copy()
            df_pred['Predicted Status'] = le_status.inverse_transform(model.predict(X))
            df_pred[name_column] = model_df[name_column].values
            st.dataframe(df_pred[[name_column, 'Gender', 'Age', 'Predicted Status']])

    except Exception as e:
        st.error(f"❌ Error: {e}")
