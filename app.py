import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.write("Choose the 4 values below and click **Predict**.")

# Load the trained model
model = pickle.load(open("svc_heart_model.pkl", "rb"))

# All possible values (same lists as in the notebook, Step 5.1)
sex_values = ["Male", "Female"]
chest_pain_values = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
major_vessels_values = ["Zero", "One", "Two", "Three"]
yes_no_values = ["Yes", "No"]

# Recreate the label encoders exactly like the notebook
sex_encoder = LabelEncoder().fit(sex_values)
chest_pain_encoder = LabelEncoder().fit(chest_pain_values)
major_vessels_encoder = LabelEncoder().fit(major_vessels_values)
exercise_angina_encoder = LabelEncoder().fit(yes_no_values)

# Inputs from the user
sex_input = st.selectbox("Sex", sex_values)
chest_pain_input = st.selectbox("Chest Pain Type", chest_pain_values)
major_vessels_input = st.selectbox("Number of major vessels coloured by fluoroscopy", major_vessels_values)
exercise_angina_input = st.selectbox("Chest pain during exercise (exercise angina)?", yes_no_values, index=1)

if st.button("Predict"):
    # Put the input in a table
    user_input = pd.DataFrame({
        "Sex": [sex_input],
        "ChestPain": [chest_pain_input],
        "MajorVessels": [major_vessels_input],
        "ExerciseAngina": [exercise_angina_input],
    })

    # Change words into numbers
    features = user_input.copy()
    features["Sex"] = sex_encoder.transform(user_input["Sex"])
    features["ChestPain"] = chest_pain_encoder.transform(user_input["ChestPain"])
    features["MajorVessels"] = major_vessels_encoder.transform(user_input["MajorVessels"])
    features["ExerciseAngina"] = exercise_angina_encoder.transform(user_input["ExerciseAngina"])

    # Predict
    result = model.predict(features)[0]

    if result == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE")

st.caption("Student project only. This is not medical advice.")
