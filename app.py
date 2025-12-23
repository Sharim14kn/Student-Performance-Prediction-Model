import streamlit as st
import numpy as np
import pickle
import gdown
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center;'>🎓 Student Performance Prediction</h1>
<p style='text-align:center;color:gray;'>
Predict student performance using Machine Learning
</p>
""", unsafe_allow_html=True)

# ---------------- MODEL DOWNLOAD ----------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gVFLj41ESTQgwFQQDIl4aPBrTnb6-pCg"
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return pickle.load(open(MODEL_PATH, "rb"))

@st.cache_resource
def load_scaler():
    return pickle.load(open("scaler.pkl", "rb"))

model = load_model()
scaler = load_scaler()

# ---------------- INPUT SECTION ----------------
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("📘 Hours Studied", 0, 24, 5)
    attendance = st.number_input("🏫 Attendance (%)", 0, 100, 75)
    sleep_hours = st.number_input("😴 Sleep Hours", 0, 12, 7)

with col2:
    previous_score = st.number_input("📝 Previous Score", 0, 100, 60)
    extracurricular = st.selectbox("🎨 Extracurricular Activities", ["No", "Yes"])
    internet_access = st.selectbox("🌐 Internet Access", ["No", "Yes"])

# Encoding
extra = 1 if extracurricular == "Yes" else 0
internet = 1 if internet_access == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Performance"):
    input_data = np.array([[hours_studied, attendance, sleep_hours,
                             previous_score, extra, internet]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"📊 Predicted Student Performance Score: **{prediction:.2f}**")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
Made with ❤️ using Streamlit & Machine Learning
</p>
""", unsafe_allow_html=True)
