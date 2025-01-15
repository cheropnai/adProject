import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('backend/xgboost_model.joblib')

# Hardcoded scaling parameters (replace these with your actual training set stats)
fdg_mean, fdg_std = 1.2, 0.5  # Example values
av45_mean, av45_std = 2.5, 0.6
fbb_mean, fbb_std = 0.8, 0.3
pib_mean, pib_std = 1.0, 0.4  # Example values for PIB

# Streamlit app title
st.title("Alzheimer's Disease Diagnosis - Imaging Biomarkers")

# Input fields for biomarkers
st.header("Enter Imaging Biomarkers")
fdg = st.number_input("FDG Level", min_value=0.0, step=0.01, value=1.0)
av45 = st.number_input("AV45 Level", min_value=0.0, step=0.01, value=2.0)
fbb = st.number_input("FBB Level", min_value=0.0, step=0.01, value=0.5)
pib = st.number_input("PIB Level", min_value=0.0, step=0.01, value=1.0)

# Predict button
if st.button("Predict"):
    # Scale inputs using hardcoded mean and std
    fdg_scaled = (fdg - fdg_mean) / fdg_std
    av45_scaled = (av45 - av45_mean) / av45_std
    fbb_scaled = (fbb - fbb_mean) / fbb_std
    pib_scaled = (pib - pib_mean) / pib_std

    # Prepare input for prediction
    input_data = np.array([[fdg_scaled, av45_scaled, fbb_scaled, pib_scaled]])

    # Predict using the model
    prediction_proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    # Map the prediction to class labels
    class_labels = {0: "MCI/CN (Non-Dementia)", 1: "Dementia"}
    result = class_labels[prediction]

    # Display the prediction
    st.subheader("Prediction Results")
    st.write(f"Predicted Class: **{result}**")
    st.write(f"Probability of Dementia: **{prediction_proba[1]:.2f}**")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
