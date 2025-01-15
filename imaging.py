import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('backend/xgboost_model.joblib')

# Title
st.title("Imaging Biomarkers Prediction")

# Inputs
fdg = st.number_input("FDG Level", min_value=0.0, step=0.01, value=1.0)
av45 = st.number_input("AV45 Level", min_value=0.0, step=0.01, value=2.0)
fbb = st.number_input("FBB Level", min_value=0.0, step=0.01, value=0.5)
pib = st.number_input("PIB Level", min_value=0.0, step=0.01, value=1.0)

if st.button("Predict"):
    # Input scaling (adjust with actual means and stds)
    fdg_scaled = (fdg - 1.2) / 0.5
    av45_scaled = (av45 - 2.5) / 0.6
    fbb_scaled = (fbb - 0.8) / 0.3
    pib_scaled = (pib - 1.0) / 0.4

    # Prediction
    input_data = np.array([[fdg_scaled, av45_scaled, fbb_scaled, pib_scaled]])
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]
    # Store the prediction probability in session state
    st.session_state["imaging_result"] = prediction_proba[1]  # Save probability of Dementia


    # Results
    class_labels = {0: "MCI/CN (Non-Dementia)", 1: "Dementia"}
    st.write(f"Predicted Class: **{class_labels[prediction]}**")
    st.write(f"Probability of Dementia: **{prediction_proba[1]:.2f}**")
 