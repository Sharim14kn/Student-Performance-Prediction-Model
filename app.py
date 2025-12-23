import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load("student_performance_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------- UI ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Performance Predictor</h1>
    <p style="text-align:center; color:gray;">
    Predict student performance using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUTS ----------------
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "📘 Hours Studied",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

    previous_scores = st.number_input(
        "📊 Previous Scores",
        min_value=0.0,
        max_value=100.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        step=0.5
    )

with col2:
    extracurricular = st.selectbox(
        "🏆 Extracurricular Activities",
        ["No", "Yes"]
    )

    sample_papers = st.number_input(
        "📝 Sample Question Papers Practiced",
        min_value=0,
        step=1
    )

# Convert categorical → numeric
extracurricular = 1 if extracurricular == "Yes" else 0

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Performance", use_container_width=True):
    try:
        input_data = np.array([[  
            hours_studied,
            previous_scores,
            extracurricular,
            sleep_hours,
            sample_papers
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(
            f"🎯 **Predicted Performance Index:** {round(prediction[0], 2)}"
        )

    except Exception as e:
        st.error("❌ Prediction failed. Please check model files.")
        st.exception(e)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px; color:gray;">
    Built with ❤️ using Streamlit & Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)
