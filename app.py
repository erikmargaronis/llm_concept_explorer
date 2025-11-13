import streamlit as st

st.set_page_config(
    page_title="LLM Companion App",
    layout="wide",
)
st.title("LLM Companion App")


st.write("Welcome to the LLM Companion App! Use the sidebar to navigate.")

st.sidebar.page_link("pages/tokenization.py", label="Tokenization")
st.sidebar.page_link("pages/sampling.py", label="Sampling")
st.sidebar.page_link("pages/prompt.py", label="Prompt")
