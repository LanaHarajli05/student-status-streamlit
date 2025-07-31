import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Status Prediction", layout="wide")

st.title("🎓 Predict Final Student Status")
st.write("Upload your dataset to analyze and predict each student’s final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📤 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
sheet_name = None
df = None

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("📄 Select a Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    if df is not None:
        st.success("✅ File uploaded successfully!")

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Confirm required columns
        # Clean and normalize column names
    df.columns = df.columns.str.strip().str.lower()

# Check if required columns exist (in lowercase)
if "name" not in df.columns or "final status" not in df.columns:
    st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
else:
    # Rename to standard casing for consistency
    df.rename(columns={"name": "NAME", "final status": "Final Status"}, inplace=True)
            st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
        else:
            st.header("📊 Exploratory Data Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Final Status Distribution")
                st.bar_chart(df["Final Status"].value_counts())

            with col2:
                st.subheader("Age Distribution")
                if 'Age' in df.columns:
                    st.bar_chart(df['Age'].dropna())
                else:
                    st.warning("⚠ 'Age' column not found in data.")

            st.subheader("Correlation Heatmap")
            num_df = df.select_dtypes(include=['int64', 'float64'])
            if not num_df.empty:
                corr = num_df.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠ No numeric features found for correlation.")

            # Logistic Regression Section
            st.header("⚙ Logistic Regression Model")

            try:
                df_model = df.copy()
                df_model = df_model.drop(columns=["NAME"])  # remove name

                # Encode categorical columns
                for col in df_model.select_dtypes(include="object").columns:
                    df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

                X = df_model.drop("Final Status", axis=1)
                y = df_model["Final Status"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

                st.subheader("📄 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # Optional: Student-Level Predictions
                st.subheader("🎯 Predictions per Student")
                df_results = df.copy()
                df_results['Predicted Status'] = model.predict(X)
                st.dataframe(df_results[['NAME', 'Final Status', 'Predicted Status']])

            except Exception as e:
                st.error(f"❌ Error running model: {e}")
