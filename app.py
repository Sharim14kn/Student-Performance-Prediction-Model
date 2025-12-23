import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import requests

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- PATHS ----------------
MODEL_PATH = "student_performance_model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "Student_performance_data.csv"

MODEL_URL = https://drive.google.com/uc?export=download&id=1gVFLj41ESTQgwFQQDIl4aPBrTnb6-pCg


# ---------------- DOWNLOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading ML model... Please wait"):
            response = requests.get(MODEL_URL)
            response.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Performance Index"])  # ⚠️ target column
    return X.columns.tolist()

model = load_model()
scaler = load_scaler()
feature_names = load_features()

# ---------------- UI ----------------
st.title("🎓 Student Performance Prediction")
st.markdown("Machine Learning based Student Performance Prediction System")
st.divider()

with st.form("student_form"):
    st.subheader("📊 Enter Student Details")

    user_input = {}
    for feature in feature_names:
        user_input[feature] = st.number_input(feature, value=0.0)

    submit = st.form_submit_button("🔮 Predict")

if submit:
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.divider()
    st.success(f"🎯 Predicted Performance Score: **{prediction:.2f}**")

st.divider()
st.caption("Built with ❤️ using Streamlit & Machine Learning")
