import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Predict Final Student Status", layout="wide")

st.title("🎓 Predict Final Student Status")
st.write("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📁 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
sheet_name = None

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1]
    
    if file_ext == "xlsx":
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select a Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)
        
    if "NAME" in df.columns and "Final Status" in df.columns:
        df = df.dropna(subset=["Final Status"])  # remove rows with missing Final Status
        st.success("✅ Dataset loaded successfully!")
        
        st.markdown("### 🧹 Initial Cleaning")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Drop columns with all nulls
        df = df.dropna(axis=1, how="all")
        
        st.markdown("### 📊 EDA – Summary Stats")
        numeric_cols = df.select_dtypes(include=np.number).columns
        st.dataframe(df[numeric_cols].describe().T)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Final Status Distribution")
            fs_counts = df["Final Status"].value_counts()
            if 207 in fs_counts.index or "207" in fs_counts.index:
                fs_counts = fs_counts.drop(207, errors='ignore')
                fs_counts = fs_counts.drop("207", errors='ignore')
            fig1, ax1 = plt.subplots()
            sns.barplot(x=fs_counts.index, y=fs_counts.values, ax=ax1, palette="Blues_d")
            ax1.set_title("Final Status")
            st.pyplot(fig1)

        with col2:
            st.markdown("#### 📈 Age Distribution")
            if 'Age' in df.columns:
                fig2, ax2 = plt.subplots()
                sns.histplot(df['Age'].dropna(), kde=False, bins=15, ax=ax2, color="skyblue", edgecolor="black")
                ax2.set_title("Age")
                st.pyplot(fig2)

        st.markdown("### 🔥 Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        st.markdown("### 🤖 Logistic Regression Model")

        # Preprocessing
        df_model = df.copy()
        df_model = df_model.dropna(subset=["Final Status"])  # again, just to be safe

        # Encode Final Status
        le_target = LabelEncoder()
        df_model["Target"] = le_target.fit_transform(df_model["Final Status"])

        X = df_model.drop(columns=["NAME", "Final Status", "Target"])
        y = df_model["Target"]

        # Remove columns with non-numeric values
        X = X.select_dtypes(include=[np.number])

        # Drop rows with any NaNs in X
        X_clean = X.dropna()
        y_clean = y[X_clean.index]

        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        st.write(f"*Accuracy:* {acc:.2f}")
        st.write(f"*Precision:* {prec:.2f}")
        st.write(f"*Recall:* {rec:.2f}")
        st.write(f"*F1 Score (Recommended):* {f1:.2f}")

        st.markdown("#### 📄 Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)).T
        st.dataframe(report_df)

        st.markdown("#### 🎯 Predictions")

        matching_idx = X.index.intersection(X_clean.index)
        names_series = df_model.loc[matching_idx, "NAME"].reset_index(drop=True)

        predictions_df = pd.DataFrame({
            "NAME": names_series,
            "Actual": le_target.inverse_transform(y_test),
            "Predicted": le_target.inverse_transform(y_pred)
        })

        st.dataframe(predictions_df)

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
