import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction")
st.markdown("Predict student performance using ML model")

# Load model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open("student_performance_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

# -------- INPUT FIELDS (5 FEATURES) --------
hours_studied = st.number_input("📘 Hours Studied", min_value=0.0, max_value=24.0, step=0.5)
previous_scores = st.number_input("📊 Previous Scores", min_value=0.0, max_value=100.0, step=1.0)
extracurricular = st.selectbox("🎯 Extracurricular Activities", ["No", "Yes"])
sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=24.0, step=0.5)
sample_papers = st.number_input("📝 Sample Question Papers Practiced", min_value=0, step=1)

# Convert categorical to numeric
extracurricular = 1 if extracurricular == "Yes" else 0

# Prediction button
if st.button("🔍 Predict Performance"):
    input_data = np.array([[  
        hours_studied,
        previous_scores,
        extracurricular,
        sleep_hours,
        sample_papers
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Performance Index: **{prediction[0]}**")
