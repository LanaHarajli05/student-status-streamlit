import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# App setup
st.set_page_config("🎓 Student Status App", layout="wide")
st.title("🎓 Predict Final Student Status (Capstone Dashboard)")

# Load cleaned files
@st.cache_data
def load_data():
    eda = pd.read_csv("eda_final.csv")
    model = pd.read_csv("model_final.csv")
    return eda, model

eda_df, model_df = load_data()

# Tabs for structure
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🤖 Machine Learning Modeling"])

# ============ TAB 1: EDA ============
with tab1:
    st.header("📊 Exploratory Data Analysis")

    # Filters
    st.sidebar.header("🔍 EDA Filters")
    gender_filter = st.sidebar.multiselect("Filter by Gender", options=eda_df['gender'].unique(), default=eda_df['gender'].unique())
    uni_filter = st.sidebar.multiselect("Filter by University", options=eda_df['university'].unique(), default=eda_df['university'].unique())

    eda_filtered = eda_df[(eda_df['gender'].isin(gender_filter)) & (eda_df['university'].isin(uni_filter))]

    with st.expander("🎯 Gender Distribution"):
        st.bar_chart(eda_filtered['gender'].value_counts())
        st.markdown("**Insight:** Gender distribution of enrolled students.")

    with st.expander("🏫 University Distribution"):
        st.bar_chart(eda_filtered['university'].value_counts())
        st.markdown("**Insight:** Shows which universities contribute the most students.")

    with st.expander("🧠 Major Distribution"):
        st.bar_chart(eda_filtered['major'].value_counts())
        st.markdown("**Insight:** Highlights dominant academic backgrounds.")

    with st.expander("📈 Age Distribution"):
        plt.figure(figsize=(6,3))
        sns.histplot(eda_filtered['age'], bins=8, kde=True, color='lightblue', edgecolor='black')
        plt.title("Age Distribution")
        st.pyplot(plt.gcf())
        st.markdown("**Insight:** Age trends among students — younger or older concentration.")

    with st.expander("📊 Final Status Distribution"):
        st.bar_chart(eda_filtered['final status'].value_counts())
        st.markdown("**Insight:** Class balance of final student statuses.")

    with st.expander("📦 Age by Final Status (Boxplot)"):
        plt.figure(figsize=(6,4))
        sns.boxplot(x='final status', y='age', data=eda_filtered, palette='Pastel1')
        st.pyplot(plt.gcf())
        st.markdown("**Insight:** See if age affects likelihood of dropout, graduation, etc.")

# ============ TAB 2: MODELING ============
with tab2:
    st.header("🤖 Machine Learning Modeling")

    # Prepare features
    X = model_df.drop(columns=['name', 'email', 'final status'], errors='ignore')
    y = model_df['final status']
    X_encoded = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    # Model selection
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"] + (["XGBoost"] if xgb_available else []))

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="✅ Accuracy", value=f"{accuracy:.2%}")

    # Classification report
    st.subheader("📋 Evaluation Metrics")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # Confusion matrix
    st.subheader("🌀 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt.gcf())

    # Prediction results
    st.subheader("📋 Final Predictions for Each Student")
    full_preds = model.predict(X_encoded)
    predicted_labels = le.inverse_transform(full_preds)
    model_df["Predicted Status"] = predicted_labels
    st.dataframe(model_df[["name", "final status", "Predicted Status"]])
