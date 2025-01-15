import streamlit as st

st.title("Results Summary")

if (
    st.session_state["imaging_result"] is not None and
    st.session_state["demographics_result"] is not None and
    st.session_state["genetics_result"] is not None and
    st.session_state["meta_learner_result"] is not None
):
    st.subheader("Model Predictions")
    st.write(f"**Imaging Model Probability of Dementia:** {st.session_state['imaging_result']:.2f}")
    st.write(f"**Demographics Model Probability of Dementia:** {st.session_state['demographics_result']:.2f}")
    st.write(f"**Genetics Model Probability of Dementia:** {st.session_state['genetics_result']:.2f}")

    st.subheader("Meta-Learner Final Prediction")
    st.write(f"**Meta-Learner Probability of Dementia:** {st.session_state['meta_learner_result']['probability']:.2f}")
    st.write(f"**Meta-Learner Predicted Class:** {st.session_state['meta_learner_result']['class']}")
else:
    st.error("Some predictions are missing. Please ensure you have completed all model predictions.")
