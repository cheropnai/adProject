import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('backend/demographic_model.joblib')

# Title
st.title("Demographics Prediction")

# Input
age = st.number_input("Age", min_value=0, max_value=120, value=60)

if st.button("Predict"):
    # Input preparation
    input_data = pd.DataFrame({'AGE': [age]})
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    prediction = model.predict(input_data)[0]
    st.session_state["demographics_result"] = prediction_proba  # Save probability of AD

    # Results
    if prediction == 1:
        st.write("The model predicts the patient is likely to have Alzheimer's Disease.")
    else:
        st.write("The model predicts the patient is not likely to have Alzheimer's Disease.")
    st.write(f"Prediction Confidence: **{prediction_proba:.2f}**")
