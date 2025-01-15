# Page navigation
import streamlit as st


# Initialize session state for predictions
if "imaging_result" not in st.session_state:
    st.session_state["imaging_result"] = None
if "demographics_result" not in st.session_state:
    st.session_state["demographics_result"] = None
if "genetics_result" not in st.session_state:
    st.session_state["genetics_result"] = None
if "meta_learner_result" not in st.session_state:
    st.session_state["meta_learner_result"] = None

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Imaging Biomarkers", "Demographics", "Genetics", "Meta-Learner", "Results"]
)

if page == "Home":
    st.title("Alzheimer's Disease Prediction")
    st.write("""
        This application predicts the likelihood of Alzheimer's Disease using different models:
        - **Imaging Biomarkers**
        - **Demographics**
        - **Genetics**
        - **Meta-Learner**
        Navigate through the sections to input data and view predictions.
    """)
    st.warning("""
        **Disclaimer:** This application is for educational purposes only and is not intended for clinical or diagnostic use. 
        The models demonstrated here are experimental and should not replace professional medical advice.
    """)
elif page == "Imaging Biomarkers":
    exec(open("imaging.py").read())
elif page == "Demographics":
    exec(open("demographics.py").read())
elif page == "Genetics":
    exec(open("genetics.py").read())
elif page == "Meta-Learner":
    exec(open("metalearner.py").read())
elif page == "Results":
    exec(open("results.py").read())
