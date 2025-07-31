import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("cleaned_student_data.csv")

st.title("🎓 Student Final Status Prediction App")

# ----------------------------------------
# EDA Section
# ----------------------------------------
st.header("📊 Exploratory Data Analysis")

st.subheader("🎯 Target Variable Distribution")
st.bar_chart(df['Final_Status'].value_counts())

st.subheader("📌 Summary Statistics")
st.dataframe(df.describe(include='all').transpose())

st.subheader("🧭 Correlation Heatmap (Numerical Features Only)")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
if not numeric_df.empty:
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
else:
    st.info("No numerical features to show correlation.")

# ----------------------------------------
# Preprocessing
# ----------------------------------------
st.header("⚙️ Model Training and Prediction")

# Drop identifier and target from features
X = df.drop(columns=['NAME', 'EMAIL', 'Final_Status'], errors='ignore')
y = df['Final_Status']

# Encode categorical variables
X_encoded = pd.get_dummies(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Model selection
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
elif model_choice == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Model Evaluation")
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

# ----------------------------------------
# Predictions for Each Student
# ----------------------------------------
st.header("📋 Predictions for All Students")

# Predict for all rows
full_X_encoded = pd.get_dummies(X_encoded)
full_preds = model.predict(full_X_encoded)
predicted_labels = le.inverse_transform(full_preds)

# Add predictions to original dataframe
result_df = df.copy()
result_df["Predicted_Status"] = predicted_labels

st.dataframe(result_df[["NAME", "Final_Status", "Predicted_Status"]])

# Optional: Download predictions
st.download_button("📥 Download Predictions CSV", data=result_df.to_csv(index=False), file_name="student_predictions.csv", mime="text/csv")
