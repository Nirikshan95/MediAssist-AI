import streamlit as st
from supervisor.supervisor_agent import create_supervisor_agent
st.title("MediAssist-AI")
st.write("Welcome to MediAssist-AI, your personal medical assistant powered by AI.")


input=st.chat_message("user")