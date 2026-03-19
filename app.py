import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction App")

st.write("Enter patient details:")

preg = st.number_input("Pregnancies")
glu = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
ins = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[preg, glu, bp, skin, ins, bmi, dpf, age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")