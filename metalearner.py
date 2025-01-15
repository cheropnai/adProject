import streamlit as st
import numpy as np
import joblib

# Load the meta-learner model
meta_model = joblib.load('backend/meta_learner_xgb.joblib')

st.title("Meta-Learner for Alzheimer's Disease Prediction")

# Check if predictions from other models are available
if (
    "imaging_result" in st.session_state and
    "demographics_result" in st.session_state and
    "genetics_result" in st.session_state
):
    # Prepare input for the meta-learner
    input_data = np.array([
        st.session_state["imaging_result"],
        st.session_state["demographics_result"],
        st.session_state["genetics_result"]
    ]).reshape(1, -1)

    # Map prediction to class labels
    class_labels = {
        0: "MCI (Mild Cognitive Impairment)",
        1: "Dementia",
        2: "CN (Cognitively Normal)"  # Include this only if the model supports 3 classes
    }

    # Button to trigger prediction
    if st.button("Run Meta-Learner"):
        # Predict using the meta-learner
        meta_prediction_proba = meta_model.predict_proba(input_data)[0]
        meta_prediction = meta_model.predict(input_data)[0]

        # Save results to session state
        st.session_state["meta_learner_result"] = {
            "probability": meta_prediction_proba[1] if len(meta_prediction_proba) > 1 else meta_prediction_proba[0],
            "class": class_labels.get(meta_prediction, "Unknown Prediction"),
        }

        # Safeguard against unexpected keys
        result = class_labels.get(meta_prediction, "Unknown Prediction")

        # Display the results
        st.subheader("Meta-Learner Prediction")
        st.write(f"Predicted Class: **{result}**")
        if meta_prediction_proba.size > 1:
            st.write(f"Probability of Dementia: **{meta_prediction_proba[1]:.2f}**")

        # Display intermediate results
        st.write("### Intermediate Model Results:")
        st.write(f"- Imaging Model Result: **{st.session_state['imaging_result']:.2f}**")
        st.write(f"- Demographics Model Result: **{st.session_state['demographics_result']:.2f}**")
        st.write(f"- Genetics Model Result: **{st.session_state['genetics_result']:.2f}**")
else:
    st.error("Results from all models are required to run the Meta-Learner. Please complete the predictions on the other pages first.")
