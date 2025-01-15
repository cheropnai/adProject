import streamlit as st
from login import login

# Initialize session state for authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Show login page if not authenticated
if not st.session_state["logged_in"]:
    login()
else:
    # Initialize session state for predictions
    if "imaging_result" not in st.session_state:
        st.session_state["imaging_result"] = None
    if "demographics_result" not in st.session_state:
        st.session_state["demographics_result"] = None
    if "genetics_result" not in st.session_state:
        st.session_state["genetics_result"] = None
    if "meta_learner_result" not in st.session_state:
        st.session_state["meta_learner_result"] = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Imaging Biomarkers", "Demographics", "Genetics", "Meta-Learner", "Results"]
    )

    # Render the selected page
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
