import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('backend/demographic_model.joblib')

# Title and description
st.title("Alzheimer's Disease Prediction (Demographics)")
st.write("""
    This application predicts the likelihood of Alzheimer's Disease based on the patient's age.
    The model has been trained on demographic data.
""")

# Input for age
st.header("Patient Details")
age = st.number_input("Enter the patient's age:", min_value=0, max_value=120, value=60)

# Predict button
if st.button("Predict"):
    # Input validation
    if age == 0:
        st.error("Please enter a valid age greater than 0.")
    else:
        # Prepare the input as a DataFrame
        input_data = pd.DataFrame({'AGE': [age]})  # Use a DataFrame with the correct column name
        
        # Make predictions using the loaded model
        prediction_proba = model.predict_proba(input_data)[:, 1][0]  # Probability of AD
        prediction = model.predict(input_data)[0]  # Binary prediction
        
        # Display the results
        if prediction == 1:
            st.warning(f"The model predicts that the patient is likely to have Alzheimer's Disease.")
        else:
            st.success(f"The model predicts that the patient is not likely to have Alzheimer's Disease.")
        
        # Show the probability
        st.info(f"Prediction Confidence (Probability of AD): {prediction_proba:.2f}")

# Footer
st.write("---")
st.write("**Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.")
