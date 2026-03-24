import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details to predict diabetes</p>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center;'><img src='https://cdn-icons-png.flaticon.com/512/2966/2966489.png' width='120'></div>",
    unsafe_allow_html=True
)

st.divider()

# Create 2 columns
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glu = st.number_input("Glucose", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 0, 150, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    ins = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 30)

st.divider()

# Predict button
if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame(
        [[preg, glu, bp, skin, ins, bmi, dpf, age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.divider()

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Diabetic")
    else:
        st.success("✅ Low Risk: Not Diabetic")

# Footer
st.markdown("---")