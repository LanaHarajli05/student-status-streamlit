import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.set_page_config(page_title="Predict Final Student Status", layout="wide")
st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        xl = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select a Sheet", xl.sheet_names)
        df = xl.parse(sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^unnamed')]

    # Ensure required columns exist
    if 'name' not in df.columns or 'final status' not in df.columns:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
    else:
        st.success("✅ File uploaded successfully!")

        st.header("📊 Exploratory Data Analysis")

        # Final Status Distribution
        st.markdown("#### Final Status Distribution")
        filtered_df = df[df['final status'].str.lower().isin(['active', 'inactive', 'drop', 'graduate'])]
        status_counts = filtered_df['final status'].value_counts()
        fig1, ax1 = plt.subplots()
        status_counts.plot(kind='bar', color=sns.color_palette("Blues", len(status_counts)), ax=ax1)
        ax1.set_xlabel("Final Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Age Distribution
        st.markdown("#### Age Distribution")
        possible_age_cols = [col for col in df.columns if 'age' in col]
        if possible_age_cols:
            age_col = possible_age_cols[0]
            try:
                df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
                fig2, ax2 = plt.subplots()
                sns.histplot(df[age_col].dropna(), bins=15, kde=False, color='skyblue', edgecolor='black', ax=ax2)
                ax2.set_xlabel('Age')
                st.pyplot(fig2)
            except Exception as e:
                st.error(f"Could not plot Age Distribution: {e}")
        else:
            st.warning("No valid 'Age' column found.")

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_df.empty:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax3)
            st.pyplot(fig3)
        else:
            st.warning("No numeric columns available for correlation analysis.")

        st.header("⚙ Logistic Regression Model")

        try:
            X = df.drop(columns=['final status', 'name'], errors='ignore')
            y = df['final status']

            # Encode categorical features
            X = X.select_dtypes(include=['object', 'number'])
            for col in X.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Drop rows with missing values
            data = pd.concat([X, y], axis=1).dropna()
            X = data.drop(columns=['final status'])
            y = data['final status']

            y_le = LabelEncoder()
            y_encoded = y_le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"*Accuracy:* {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"*Precision:* {precision_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"*Recall:* {recall_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"*F1 Score (Recommended):* {f1_score(y_test, y_pred, average='weighted'):.2f}")

            st.subheader("📋 Classification Report")
            report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=y_le.classes_, output_dict=True)).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
        except Exception as e:
            st.error(f"❌ Model Error: {e}")
