import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('diabetes_dataset.pkl')
scaler = joblib.load('scaler.pkl')

# PAGE CONFIGURATION
st.set_page_config(page_title="Diabetes Prediction", page_icon="🧠", layout="centered")


# CUSTOM CSS STYLE
st.markdown("""
    <style>
        body {
            background-color: #1e1e2f;
            color: #f5f5f5;
        }
        .stApp {
            background-color: #1e1e2f;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            color: #007B8A;
        }
        .stNumberInput>div>div>input {
            background-color: #283593;
            color: white;
        }
        .stButton>button {
            background-color: #2ECC71;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# APP TITLE
st.markdown("""
    <h1 style='text-align: center; color: #007B8A;'>
        🧠 Diabetes Prediction System
    </h1>
""", unsafe_allow_html=True)

# INPUT FORM
preg = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0)
bp = st.number_input('Blood Pressure', min_value=0)
skin = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin Level', min_value=0)
bmi = st.number_input('BMI', min_value=0.0, format="%.1f")
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
age = st.number_input('Age', min_value=1)


# PREDICT BUTTON
if st.button('🎗️ Predict'):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    result = '🟢 Not Diabetic' if prediction[0] == 0 else '🔴 Diabetic'
    st.subheader(f'Result: {result}')

