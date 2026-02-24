import streamlit as st
import joblib
import numpy as np

model = joblib.load("diabetes_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🩺 Diabetes Prediction App")

# Input fields
preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    features = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
