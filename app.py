import streamlit as st
import numpy as np
import joblib
import gdown
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_URL = "https://drive.google.com/uc?id=1L__y451voKm1F7OE8cZizHm_Fq3mzTDW"
MODEL_PATH = "student_model.pkl"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

# ================= UI =================
st.title("🎓 Student Performance Prediction")
st.caption("ML-based academic score prediction")
st.divider()



# ================= INPUT FORM =================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🎂 Age", 10, 25, 18)
        study_time = st.slider("📘 Study Time (1–4)", 1, 4, 2)
        failures = st.slider("⚠️ Past Failures", 0, 4, 0)
        absences = st.number_input("❌ Absences", 0, 100, 4)

    with col2:
        health = st.slider("💪 Health (1–5)", 1, 5, 3)
        free_time = st.slider("🎮 Free Time (1–5)", 1, 5, 3)

    submit = st.form_submit_button("🔍 Predict")

# ================= PREDICTION =================
if submit:
    """
    FINAL FEATURE VECTOR (13 features)
    Order MUST match training order
    """

    input_data = np.array([[
        1,          # school (GP=1 default)
        1,          # sex (Male=1)
        age,        # age
        1,          # address (Urban=1)
        study_time, # studytime
        failures,   # failures
        0,          # schoolsup (No)
        1,          # famsup (Yes)
        0,          # paid classes (No)
        1,          # activities (Yes)
        health,     # health
        absences,   # absences
        free_time   # freetime
    ]])

    # SAFETY CHECK
    if input_data.shape[1] != model.n_features_in_:
        st.error("❌ Feature mismatch even after fix")
        st.stop()

    prediction = model.predict(input_data)[0]

    # ================= RESULT =================
    st.success(f"📊 Predicted Final Score: **{prediction:.2f}**")

    if prediction >= 75:
        st.balloons()
        st.markdown("🎉 **Excellent Performance Expected**")
    elif prediction >= 50:
        st.markdown("🙂 **Average Performance Expected**")
    else:
        st.warning("⚠️ **Needs Improvement**")

st.divider()
st.caption("🚀 Built with Streamlit & Scikit-learn")
