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

    # Drop unnamed columns
    eda_df = eda_df.loc[:, ~eda_df.columns.str.contains('^Unnamed')]
    model_df = model_df.loc[:, ~model_df.columns.str.contains('^Unnamed')]

    # Sidebar Filters
    st.sidebar.header("🔍 EDA Filters")
    gender_filter = st.sidebar.multiselect("Filter by Gender", options=eda_df['Gender'].dropna().unique(), default=eda_df['Gender'].dropna().unique())
    age_filter = st.sidebar.slider("Select Age Range", int(eda_df['Age'].min()), int(eda_df['Age'].max()), (int(eda_df['Age'].min()), int(eda_df['Age'].max())))
    eda_filtered = eda_df[(eda_df['Gender'].isin(gender_filter)) & (eda_df['Age'].between(age_filter[0], age_filter[1]))]

    # Tabs
    tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Machine Learning Modeling"])

    with tab1:
        st.header("📊 Exploratory Data Analysis")

        with st.expander("🎯 Final Status Distribution"):
            fig1, ax1 = plt.subplots()
            sns.countplot(data=eda_filtered, x='Final Status', palette='pastel', ax=ax1)
            ax1.set_title("Final Status Distribution")
            st.pyplot(fig1)

        with st.expander("📈 Age Distribution"):
            fig2, ax2 = plt.subplots()
            sns.histplot(data=eda_filtered, x='Age', kde=True, color='lightblue', bins=15, ax=ax2)
            ax2.set_title("Age Distribution")
            st.pyplot(fig2)

        with st.expander("👩‍🎓 Gender vs Final Status"):
            fig3, ax3 = plt.subplots()
            sns.countplot(data=eda_filtered, x='Gender', hue='Final Status', palette='pastel', ax=ax3)
            ax3.set_title("Final Status by Gender")
            st.pyplot(fig3)

    with tab2:
        st.header("🤖 Train Machine Learning Models")

        # Drop rows with missing target
        model_df = model_df.dropna(subset=['Final Status'])
        y = model_df['Final Status']
        X = model_df.drop(columns=['Final Status', 'NAME'])

        # Encode features
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include='object').columns:
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

        # Encode target
        y_encoded = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        # Models
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

        # Final prediction with XGBoost
        st.subheader("📄 Per-Student Predictions (XGBoost)")
        final_model = models["XGBoost"]
        preds = final_model.predict(X_encoded)
        label_map = {i: label for i, label in enumerate(LabelEncoder().fit(y).classes_)}
        model_df['Predicted Status'] = [label_map[i] for i in preds]
        st.dataframe(model_df[['NAME', 'Predicted Status']].reset_index(drop=True))

else:
    st.warning("📁 Please upload the Excel file to start.")
