import streamlit as st
import numpy as np
import joblib
import gdown
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- DOWNLOAD MODEL ----------------
@st.cache_resource
def load_files():
    if not os.path.exists("model.pkl"):
        gdown.download(
            "https://drive.google.com/uc?id=1L__y451voKm1F7OE8cZizHm_Fq3mzTDW",
            "model.pkl",
            quiet=False
        )

    if not os.path.exists("scaler.pkl"):
        gdown.download(
            "https://drive.google.com/uc?id=1x4rRiiEOb_I9yA0UXC1UaanA-s4rYZXb",
            "scaler.pkl",
            quiet=False
        )

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_files()

# ---------------- UI ----------------
st.title("🎓 Student Performance Prediction")
st.markdown("Predict student performance using ML model")

st.divider()

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", [0, 1])
    race = st.selectbox("Race/Ethnicity", [0, 1, 2, 3, 4])
    parental_edu = st.selectbox("Parental Education", [0, 1, 2, 3, 4, 5])
    lunch = st.selectbox("Lunch Type", [0, 1])
    test_prep = st.selectbox("Test Preparation", [0, 1])
    math_score = st.slider("Math Score", 0, 100, 50)
    reading_score = st.slider("Reading Score", 0, 100, 50)

with col2:
    writing_score = st.slider("Writing Score", 0, 100, 50)
    study_hours = st.slider("Study Hours", 0, 10, 4)
    attendance = st.slider("Attendance %", 0, 100, 85)
    internet = st.selectbox("Internet Access", [0, 1])
    family_support = st.selectbox("Family Support", [0, 1])
    extra_classes = st.selectbox("Extra Classes", [0, 1])

# ---------------- PREDICTION ----------------
if st.button("📊 Predict Performance"):
    input_data = np.array([[
        gender,
        race,
        parental_edu,
        lunch,
        test_prep,
        math_score,
        reading_score,
        writing_score,
        study_hours,
        attendance,
        internet,
        family_support,
        extra_classes
    ]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"🎯 Predicted Performance Score: **{prediction:.2f}**")
