import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Page setup
st.set_page_config(page_title="Student Status Predictor", layout="wide")
st.title("🎓 Predict Final Student Status (Capstone Dashboard)")
st.markdown("This app performs EDA and predicts student outcomes: **Active, Dropped, In-Active, or Graduated**.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

@st.cache_data
def load_data(file):
    xls = pd.ExcelFile(file)
    eda_df = pd.read_excel(xls, sheet_name='All Enrolled')
    model_df = pd.read_excel(xls, sheet_name='All Enrolled (2)')
    return eda_df, model_df

if uploaded_file:
    eda_df, model_df = load_data(uploaded_file)

    # Clean columns
    eda_df.columns = eda_df.columns.str.strip().str.lower()
    model_df.columns = model_df.columns.str.strip().str.lower()

    # Drop unnamed columns
    eda_df = eda_df.loc[:, ~eda_df.columns.str.contains('^unnamed')]
    model_df = model_df.loc[:, ~model_df.columns.str.contains('^unnamed')]

    # Sidebar Filters
    st.sidebar.header("🔍 EDA Filters")
    if 'gender' in eda_df.columns:
        gender_filter = st.sidebar.multiselect("Filter by Gender", options=eda_df['gender'].dropna().unique(), default=eda_df['gender'].dropna().unique())
    else:
        gender_filter = []

    if 'age' in eda_df.columns:
        age_filter = st.sidebar.slider("Select Age Range", int(eda_df['age'].min()), int(eda_df['age'].max()), (int(eda_df['age'].min()), int(eda_df['age'].max())))
    else:
        age_filter = (0, 100)

    # Filtered Data
    eda_filtered = eda_df.copy()
    if 'gender' in eda_df.columns:
        eda_filtered = eda_filtered[eda_filtered['gender'].isin(gender_filter)]
    if 'age' in eda_df.columns:
        eda_filtered = eda_filtered[eda_filtered['age'].between(age_filter[0], age_filter[1])]

    # Tabs
    tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Machine Learning Modeling"])

    with tab1:
        st.header("📊 Exploratory Data Analysis")

        with st.expander("🎯 Final Status Distribution"):
            if 'final status' in eda_filtered.columns and not eda_filtered.empty:
                fig, ax = plt.subplots()
                sns.countplot(data=eda_filtered, x='final status', palette='pastel', ax=ax)
                ax.set_title("Final Status Distribution")
                st.pyplot(fig)
            else:
                st.warning("No valid 'Final Status' data available to plot.")

        with st.expander("📈 Age Distribution"):
            if 'age' in eda_filtered.columns:
                fig2, ax2 = plt.subplots()
                sns.histplot(data=eda_filtered, x='age', kde=True, color='lightblue', bins=15, ax=ax2)
                ax2.set_title("Age Distribution")
                st.pyplot(fig2)

        with st.expander("👩‍🎓 Gender vs Final Status"):
            if 'gender' in eda_filtered.columns and 'final status' in eda_filtered.columns and not eda_filtered.empty:
                fig3, ax3 = plt.subplots()
                sns.countplot(data=eda_filtered, x='gender', hue='final status', palette='pastel', ax=ax3)
                ax3.set_title("Final Status by Gender")
                st.pyplot(fig3)
            else:
                st.warning("No valid Gender vs Final Status data available to plot.")

    with tab2:
        st.header("🤖 Train Machine Learning Models")

        if 'final status' not in model_df.columns:
            st.error("'Final Status' column not found in modeling sheet.")
        else:
            model_df = model_df.dropna(subset=['final status'])
            y = model_df['final status']
            X = model_df.drop(columns=['final status'])

            if 'name' in X.columns:
                X = X.drop(columns=['name'])

            # Encode categorical
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include='object').columns:
                X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

            # Encode target
            try:
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            except Exception as e:
                st.error(f"Label Encoding Error: {e}")
                st.stop()

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            }

            for name, model in models.items():
                st.subheader(f"📌 {name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.2f}")
                col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.2f}")
                col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='macro'):.2f}")

            # Final predictions
            st.subheader("📄 Per-Student Predictions (XGBoost)")
            final_model = models['XGBoost']
            final_preds = final_model.predict(X_encoded)
            model_df['Predicted Status'] = [label_encoder.classes_[i] for i in final_preds]
            st.dataframe(model_df[['name', 'Predicted Status']].reset_index(drop=True))

else:
    st.warning("Please upload the Excel file to proceed.")
