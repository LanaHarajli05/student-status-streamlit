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
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Status Predictor", layout="wide")
st.title("🎓 Predict Final Student Status (Capstone Dashboard)")
st.markdown("This app performs EDA and predicts student outcomes: **Active, Dropped, In-Active, or Graduated**.")

uploaded_file = st.sidebar.file_uploader("📁 Upload Excel File", type=["xlsx"])

@st.cache_data
def load_data(file):
    xls = pd.ExcelFile(file)
    sheet1 = pd.read_excel(xls, sheet_name="All Enrolled")
    sheet2 = pd.read_excel(xls, sheet_name="All Enrolled (2)")
    return sheet1, sheet2

if uploaded_file:
    eda_df, model_df = load_data(uploaded_file)

    # Drop unnamed columns
    eda_df = eda_df.loc[:, ~eda_df.columns.str.contains('^Unnamed')]
    model_df = model_df.loc[:, ~model_df.columns.str.contains('^Unnamed')]

    # Unify column name casing
    eda_df.columns = [col.strip().title() for col in eda_df.columns]
    model_df.columns = [col.strip().title() for col in model_df.columns]

    if 'Final Status' not in eda_df.columns or 'Final Status' not in model_df.columns:
        st.error("❌ Column 'Final Status' not found in dataset. Please check column naming.")
        st.stop()

    st.sidebar.header("🔍 EDA Filters")
    gender_opts = eda_df['Gender'].dropna().unique().tolist()
    gender_filter = st.sidebar.multiselect("Filter by Gender", gender_opts, default=gender_opts)
    age_min, age_max = int(eda_df['Age'].min()), int(eda_df['Age'].max())
    age_filter = st.sidebar.slider("Select Age Range", age_min, age_max, (age_min, age_max))

    eda_filtered = eda_df[(eda_df['Gender'].isin(gender_filter)) & (eda_df['Age'].between(age_filter[0], age_filter[1]))]

    tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Machine Learning Modeling"])

    with tab1:
        st.header("📊 Exploratory Data Analysis")

        # Final Status Distribution
        with st.expander("🎯 Final Status Distribution", expanded=True):
            st.markdown("Shows how student statuses are distributed.")
            if not eda_filtered.empty:
                fig1, ax1 = plt.subplots()
                sns.countplot(data=eda_filtered, x='Final Status', palette='pastel', ax=ax1)
                ax1.set_title("Final Status Distribution")
                st.pyplot(fig1)
            else:
                st.warning("No valid 'Final Status' data available to plot.")

        # Age Distribution
        with st.expander("📈 Age Distribution", expanded=True):
            st.markdown("Histogram of student ages.")
            if not eda_filtered.empty:
                fig2, ax2 = plt.subplots()
                sns.histplot(eda_filtered['Age'], kde=True, bins=15, color='lightblue', ax=ax2)
                ax2.set_title("Age Distribution")
                st.pyplot(fig2)
            else:
                st.warning("No data available after filters for age distribution.")

        # Gender vs Final Status
        with st.expander("👥 Gender vs Final Status", expanded=True):
            st.markdown("Distribution of student status per gender.")
            if not eda_filtered.empty:
                fig3, ax3 = plt.subplots()
                sns.countplot(data=eda_filtered, x='Gender', hue='Final Status', palette='pastel', ax=ax3)
                ax3.set_title("Final Status by Gender")
                st.pyplot(fig3)
            else:
                st.warning("No data available after filters to display gender breakdown.")

    with tab2:
        st.header("🤖 Train Machine Learning Models")

        if 'Final Status' not in model_df.columns:
            st.error("Missing 'Final Status' in modeling sheet.")
            st.stop()

        model_df = model_df.dropna(subset=['Final Status'])
        if model_df.empty:
            st.error("❌ No rows with 'Final Status' found.")
            st.stop()

        y = model_df['Final Status']
        X = model_df.drop(columns=['Final Status', 'Name'], errors='ignore')

        # Encode categorical features
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        y_encoded = LabelEncoder().fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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

        # Final Prediction Table
        st.subheader("📄 Per-Student Predictions (XGBoost)")
        final_model = models['XGBoost']
        preds = final_model.predict(X)
        label_map = {i: label for i, label in enumerate(LabelEncoder().fit(y).classes_)}
        model_df['Predicted Status'] = [label_map[i] for i in preds]

        if 'Name' in model_df.columns:
            st.dataframe(model_df[['Name', 'Predicted Status']])
        else:
            st.dataframe(model_df[['Predicted Status']])

else:
    st.warning("📤 Please upload the Excel file to proceed.")
