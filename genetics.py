import streamlit as st
import numpy as np
import joblib

# Load the trained Random Forest model
model = joblib.load('backend/genetic_model.pkl')

# Title
st.title("Genetic Biomarkers Prediction")

# Inputs for genetic biomarkers
apoe4 = st.number_input("APOE4 (Presence of Gene Variant)", min_value=0, max_value=1, step=1, value=0)
abeta_bl = st.number_input("ABETA_bl Level", min_value=0.0, step=0.01, value=741.5)
tau_bl = st.number_input("TAU_bl Level", min_value=0.0, step=0.01, value=239.7)
ptau_bl = st.number_input("PTAU_bl Level", min_value=0.0, step=0.01, value=22.83)

if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[apoe4, abeta_bl, tau_bl, ptau_bl]])
    
    # Predict using the model
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    st.session_state["genetics_result"] = prediction_proba[1]  # Save probability of Dementia

    # Map prediction to class labels
    class_labels = {0: "MCI/CN (Non-Dementia)", 1: "Dementia"}
    result = class_labels[prediction]

    # Display results
    st.write(f"Predicted Class: **{result}**")
    st.write(f"Probability of Dementia: **{prediction_proba[1]:.2f}**")
