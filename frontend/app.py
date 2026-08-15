"""
Diabetes Risk Prediction - Streamlit Frontend
Run from project root: streamlit run frontend/app.py
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered")

st.title("Diabetes Risk Prediction")
st.write("Enter patient details to estimate diabetes risk and receive lifestyle recommendations.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=100.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)

    with col2:
        insulin = st.number_input("Insulin (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        st.subheader("Results")

        risk_pct = result["risk_percentage"]
        risk_label = result["risk_label"]

        if risk_label == "High":
            st.error(f"Risk Score: {risk_pct}% — {risk_label} risk")
        else:
            st.success(f"Risk Score: {risk_pct}% — {risk_label} risk")

        st.progress(min(int(risk_pct), 100))

        st.subheader("Recommendations")
        for tip in result["recommendations"]:
            st.write(f"- {tip}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Make sure the FastAPI server is running on port 8000.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")