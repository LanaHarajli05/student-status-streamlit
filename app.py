import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

st.set_page_config(page_title="Predict Final Student Status", layout="wide")

st.title("🎓 Predict Final Student Status")
st.markdown("Upload your dataset to analyze and predict each student's final status (Active, Drop, Graduate, Inactive).")

uploaded_file = st.file_uploader("📤 Upload your Excel (.xlsx) or CSV file", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("📄 Select a Sheet", xls.sheet_names)
        df = pd.read_excel(xls, sheet_name=sheet_name)
    else:
        df = pd.read_csv(uploaded_file)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    if 'NAME' in df.columns and 'Final Status' in df.columns:
        st.success("✅ File uploaded successfully!")

        # Drop rows with missing target
        df = df.dropna(subset=['Final Status'])

        st.subheader("🧪 Exploratory Data Analysis")

        # Final Status distribution
        fig1, ax1 = plt.subplots()
        df['Final Status'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Final Status Distribution")
        ax1.set_xlabel("Final Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Age distribution + by final status
        if 'Age' in df.columns:
            fig2, ax = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(df['Age'].dropna(), bins=10, ax=ax[0], color='lightblue')
            ax[0].set_title("Age Distribution")
            ax[0].set_xlabel("Age")

            sns.boxplot(data=df, x='Final Status', y='Age', ax=ax[1])
            ax[1].set_title("Age vs Final Status")
            st.pyplot(fig2)

        # Correlation heatmap
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.subheader("📊 Correlation Heatmap")
            fig3, ax3 = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues', ax=ax3)
            st.pyplot(fig3)

        # ----------------- MODELING ------------------
        st.subheader("⚙ Logistic Regression Model")

        df_model = df.copy()
        df_model = df_model.dropna()  # Remove all rows with any NaN

        label_encoders = {}
        for col in df_model.columns:
            if df_model[col].dtype == 'object' and col not in ['NAME', 'Final Status']:
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col])
                label_encoders[col] = le

        # Encode target
        le_target = LabelEncoder()
        df_model['Final Status'] = le_target.fit_transform(df_model['Final Status'])

        # Features and Target
        X = df_model.drop(['NAME', 'Final Status'], axis=1)
        y = df_model['Final Status']

        # Final check: must be numeric and no NaNs
        if not X.select_dtypes(include=['number']).equals(X):
            st.error("❌ Non-numeric data detected in features. Please clean dataset.")
        elif X.isnull().sum().sum() > 0:
            st.error("❌ Missing values in features. Please check your data.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            st.markdown(f"*Accuracy:* {acc:.2f}  \n*Precision:* {prec:.2f}  \n*Recall:* {rec:.2f}  \n*F1 Score (Recommended):* {f1:.2f}")

            report = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=False)
            st.text("📋 Classification Report")
            st.text(report)

            # Prediction table
            st.subheader("📈 Prediction Probabilities for Each Student")
            df_predict = df_model[['NAME']].copy()
            proba = model.predict_proba(X)
            for i, class_label in enumerate(le_target.classes_):
                df_predict[class_label] = proba[:, i]

            st.dataframe(df_predict)

    else:
        st.error("❌ Dataset must contain both 'NAME' and 'Final Status' columns.")
