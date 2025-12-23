import streamlit as st
import numpy as np
import joblib
import gdown
import os

# ================== CONFIG ==================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# 🔗 Google Drive links
MODEL_URL = "https://drive.google.com/uc?id=1L__y451voKm1F7OE8cZizHm_Fq3mzTDW"
SCALER_URL = "https://drive.google.com/file/d/1x4rRiiEOb_I9yA0UXC1UaanA-s4rYZXb/view?usp=sharing"  # <-- agar scaler hai to yahan link daalo, warna None

MODEL_PATH = "student_model.pkl"
SCALER_PATH = "scaler.pkl"

# ================== LOAD MODEL ==================
@st.cache_resource
def load_artifacts():
    # download model
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = joblib.load(MODEL_PATH)

    scaler = None
    if SCALER_URL:
        if not os.path.exists(SCALER_PATH):
            gdown.download(SCALER_URL, SCALER_PATH, quiet=False)
        scaler = joblib.load(SCALER_PATH)

    return model, scaler


model, scaler = load_artifacts()

# ================== UI ==================
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Machine Learning Based Academic Score Prediction</p>", unsafe_allow_html=True)
st.divider()

# show expected feature count (debug + safety)
st.info(f"✅ Model expects **{model.n_features_in_} features**")

# ================== INPUT FORM ==================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_time = st.number_input("📘 Study Time (hours/day)", 0.0, 24.0, 2.0)
        absences = st.number_input("❌ Absences", 0, 100, 4)
        failures = st.number_input("⚠️ Past Failures", 0, 5, 0)
        age = st.number_input("🎂 Age", 10, 30, 18)

    with col2:
        health = st.slider("💪 Health (1 = Poor, 5 = Excellent)", 1, 5, 3)
        free_time = st.slider("🎮 Free Time (1–5)", 1, 5, 3)
        goout = st.slider("🚶 Go Out (1–5)", 1, 5, 3)
        travel_time = st.slider("🚌 Travel Time (1–4)", 1, 4, 1)

    submit = st.form_submit_button("🔍 Predict Performance")

# ================== PREDICTION ==================
if submit:
    # 🔥 EXACT SAME ORDER AS TRAINING
    input_data = np.array([[
        study_time,
        absences,
        failures,
        age,
        health,
        free_time,
        goout,
        travel_time
    ]])

    # safety check
    if input_data.shape[1] != model.n_features_in_:
        st.error(
            f"❌ Feature mismatch: Model expects {model.n_features_in_} "
            f"but received {input_data.shape[1]}"
        )
        st.stop()

    # apply scaler if exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    # ================== RESULT ==================
    st.success(f"📊 Predicted Final Score: **{prediction:.2f}**")

    if prediction >= 75:
        st.balloons()
        st.markdown("🎉 **Excellent Performance Expected!**")
    elif prediction >= 50:
        st.markdown("🙂 **Average Performance Expected**")
    else:
        st.warning("⚠️ **Needs Improvement**")

st.divider()
st.caption("🚀 Built with Streamlit • Scikit-Learn • Google Drive Models")
