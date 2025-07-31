import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

st.set_page_config(page_title="🎓 Student Status Predictor", layout="wide")
st.title("🎓 Predict Final Student Status (Capstone App)")
st.markdown("This app performs EDA and predicts student outcomes: **Active**, **Dropped**, **In-Active**, or **Graduated**.")

# Load EDA + Modeling data
@st.cache_data
def load_data():
    eda_df = pd.read_csv("eda_clean.csv")
    model_df = pd.read_csv("model_clean.csv")
    return eda_df, model_df

eda_df, model_df = load_data()

# ------------------------------------------
# SECTION 1: EDA on All Enrolled
# ------------------------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

st.subheader("🎯 Gender Distribution")
st.bar_chart(eda_df['gender'].value_counts())

st.subheader("🏫 University Distribution")
st.bar_chart(eda_df['university'].value_counts())

st.subheader("🧠 Major Distribution")
st.bar_chart(eda_df['major'].value_counts())

if 'age' in eda_df.columns:
    st.subheader("📈 Age Distribution")
    plt.figure(figsize=(6, 3))
    sns.histplot(eda_df['age'], bins=15, kde=True)
    st.pyplot(plt.gcf())

st.subheader("🔥 Correlation Heatmap (Numerical Only)")
numeric_eda = eda_df.select_dtypes(include=['int64', 'float64'])
if not numeric_eda.empty:
    plt.figure(figsize=(6, 4))
    sns.heatmap(numeric_eda.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
else:
    st.info("No numeric fields found.")

# ------------------------------------------
# SECTION 2: Modeling on All Enrolled (2)
# ------------------------------------------
st.header("🤖 Student Final Status Prediction")

# Clean and encode features
X = model_df.drop(columns=['name', 'email', 'final status'], errors='ignore')
y = model_df['final status']
X_encoded = pd.get_dummies(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Match columns post encoding
X_encoded, y_encoded = X_encoded.align(X_encoded, join='inner', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"] + (["XGBoost"] if xgb_available else []))

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluation Metrics
# ------------------------------------------
st.subheader("📈 Evaluation Metrics")
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

# Confusion Matrix
st.subheader("🌀 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt.gcf())

# ------------------------------------------
# Prediction for All Students
# ------------------------------------------
st.header("📋 Final Predictions for Each Student")

all_preds = model.predict(X_encoded)
predicted_labels = le.inverse_transform(all_preds)
model_df["Predicted Status"] = predicted_labels

st.dataframe(model_df[["name", "final status", "Predicted Status"]])
