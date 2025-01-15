import streamlit as st

# User credentials
USER_CREDENTIALS = {
    "doctor1": "password123",
    "admin": "adminpass"
}

def login():
    """Login page for user authentication."""
    st.title("Login")

    # Input fields for username and password
    username = st.text_input("Username", key="username")
    password = st.text_input("Password", type="password", key="password")

    # Login button
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.success(f"Welcome, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.experimental_rerun()  # Reload the app to show main navigation
        else:
            st.error("Invalid username or password. Please try again.")

# Function to handle logout
def logout():
    """Logout the user."""
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.experimental_rerun()  # Reload the app to show the login page
