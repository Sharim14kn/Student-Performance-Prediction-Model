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

MODEL_URL = "https://drive.google.com/uc?id=1L__y451voKm1F7OE8cZizHm_Fq3mzTDW"
MODEL_PATH = "student_model.pkl"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict student performance using Machine Learning</p>", unsafe_allow_html=True)
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_time = st.number_input("📘 Study Time (hours/day)", 0.0, 24.0, 3.0)
        absences = st.number_input("❌ Absences", 0, 100, 2)
        failures = st.number_input("⚠️ Past Failures", 0, 5, 0)

    with col2:
        age = st.number_input("🎂 Age", 10, 30, 18)
        health = st.slider("💪 Health (1 = Poor, 5 = Excellent)", 1, 5, 3)
        free_time = st.slider("🎮 Free Time (1–5)", 1, 5, 3)

    submit = st.form_submit_button("🔍 Predict Performance")

# ---------------- PREDICTION ----------------
if submit:
    input_data = np.array([[study_time, absences, failures, age, health, free_time]])
    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Final Score: **{prediction[0]:.2f}**")

    if prediction[0] >= 75:
        st.balloons()
        st.markdown("🎉 **Excellent Performance Expected!**")
    elif prediction[0] >= 50:
        st.markdown("🙂 **Average Performance Expected**")
    else:
        st.warning("⚠️ **Needs Improvement**")

st.divider()
st.caption("🚀 Built with Streamlit & Machine Learning")
